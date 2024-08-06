"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.compute import gen_gaussian_noise
from pyutils.general import logger
from pyutils.quant.lsq import ActQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import _size

from torchonn.layers.base_layer import ONNBaseConv2d
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import (
    PhaseQuantizer,
    checkerboard_to_vector,
    phase_to_voltage,
    upper_triangle_to_vector,
    vector_to_checkerboard,
    vector_to_upper_triangle,
    voltage_to_phase,
)

from .utils import merge_chunks, partition_chunks

__all__ = [
    "MZIBlockConv2d",
]


@MODELS.register_module()
class MZIBlockConv2d(ONNBaseConv2d):
    """
    SVD-based blocking Conv2d layer constructed by cascaded MZIs.
    """

    ## default configs
    default_cfgs = dict(
        miniblock=(
            1,
            1,
            4,
            4,
        ),  # [#tiles, pe per tile, row, col] # i.e., [R, C, k1, k2]
        mode="weight",
        w_bit=32,
        in_bit=32,
        out_bit=32,
        v_max=10.8,
        v_pi=4.36,
        photodetect="coherent",
        decompose_alg="clements",
        device=torch.device("cpu"),
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
        **cfgs,
    ):
        super().__init__()
        self.load_cfgs(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **cfgs,
        )

        ### build trainable parameters
        self.build_parameters(self.mode)
        ### build transform
        self.build_transform()

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.zeros(out_channels, device=self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def load_cfgs(
        self,
        **cfgs,
    ) -> None:
        super().load_cfgs(**cfgs)

        ## verify configs
        assert self.mode in {"weight", "usv", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {self.mode}."
        )
        self.gamma = np.pi / self.v_pi**2
        assert self.photodetect in ["coherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        self.weight = torch.empty(
            self.grid_dim_y, self.grid_dim_x, *self.miniblock, device=self.device
        )
        ## usv mode
        self.U = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-1],
            self.miniblock[-2],
            device=self.device,
        )
        self.S = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            min(self.miniblock[-2:]),
            device=self.device,
        )
        self.V = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            self.miniblock[-1],
            self.miniblock[-1],
            device=self.device,
        )
        ## phase mode
        self.delta_list_U = torch.empty(self.U.shape[:-1], device=self.device)
        self.phase_U = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            self.miniblock[-1] * (self.miniblock[-1] - 1) // 2,
            device=self.device,
        )
        self.phase_S = torch.empty_like(self.S)
        self.delta_list_V = torch.empty(self.V.shape[:-1], device=self.device)
        self.phase_V = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            self.miniblock[-2] * (self.miniblock[-2] - 1) // 2,
            device=self.device,
        )
        # TIA gain
        self.S_scale = torch.randn(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            1,
            device=self.device,
        )

        self.register_parameter_buffer(*self.get_param_buffer_groups(mode=mode))

        self.pack_weights()

    def get_param_buffer_groups(self, mode: str) -> Tensor:
        buffer_groups = {
            "weight": self.weight,
            "U": self.U,
            "S": self.S,
            "V": self.V,
            "phase_U": self.phase_U,
            "phase_S": self.phase_S,
            "phase_V": self.phase_V,
            "S_scale": self.S_scale,
            "delta_list_U": self.delta_list_U,
            "delta_list_V": self.delta_list_V,
        }

        if mode == "weight":
            param_groups = {"weight": Parameter(self.weight)}

        elif mode == "usv":
            param_groups = {
                "U": Parameter(self.U),
                "S": Parameter(self.S),
                "V": Parameter(self.V),
            }
        elif mode == "phase":
            param_groups = {
                "phase_U": Parameter(self.phase_U),
                "phase_S": Parameter(self.phase_S),
                "phase_V": Parameter(self.phase_V),
                "S_scale": Parameter(self.S_scale),
            }
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for name in param_groups:
            del buffer_groups[name]

        return param_groups, buffer_groups

    def pack_weights(self):
        if self.mode == "weight":
            self.weights = {"weight": self.weight}
        elif self.mode == "usv":
            self.weights = {"usv": (self.U, self.S, self.V)}
        elif self.mode == "phase":
            self.weights = {
                "phase": (self.phase_U, self.phase_S, self.phase_V, self.S_scale)
            }
        else:
            raise NotImplementedError

    def reset_parameters(self) -> None:
        W = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            groups=self.groups,
        ).weight.data.to(self.device)
        W = partition_chunks(W.flatten(1), self.weight.shape)
        if self.mode == "weight":
            self.weight.data.copy_(W)
        elif self.mode == "usv":
            if self.device.type == "cpu":
                U, S, V = torch.linalg.svd(W, full_matrices=True)
            else:
                U, S, V = torch.linalg.svd(
                    W, full_matrices=True, driver="gesvd"
                )  # must use QR decomposition
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(torch.ones_like(S, device=self.device))
        elif self.mode == "phase":
            if self.device.type == "cpu":
                U, S, V = torch.linalg.svd(W, full_matrices=True)
            else:
                U, S, V = torch.linalg.svd(
                    W, full_matrices=True, driver="gesvd"
                )  # must use QR decomposition
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V.data.copy_(delta_list)
            self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))
            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())

        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def switch_mode_to(self, mode: str) -> None:
        super().switch_mode_to(mode)
        self.register_parameter_buffer(*self.get_param_buffer_groups(mode=mode))
        self.pack_weights()

    def build_transform(self) -> None:
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch(alg=self.decompose_alg)
        if self.decompose_alg == "clements":
            self.decomposer.v2m = vector_to_checkerboard
            self.decomposer.m2v = checkerboard_to_vector
            mesh_mode = "rectangle"
            crosstalk_filter_size = 5
        elif self.decompose_alg in {"reck", "francis"}:
            self.decomposer.v2m = vector_to_upper_triangle
            self.decomposer.m2v = upper_triangle_to_vector
            mesh_mode = "triangle"
            crosstalk_filter_size = 3

        ### quantization tool
        self.input_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.in_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        self.phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode=mesh_mode,
            device=self.device,
        )
        self.phase_V_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode=mesh_mode,
            device=self.device,
        )
        self.phase_S_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="diagonal",
            device=self.device,
        )

        self.output_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.out_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        ## add input transform
        self.add_transform("input", "input", {"input_transform": self._input_transform})

        ## add weight transform
        if self.mode == "weight":
            ## add a transform called "build_weight" for parameter named "weight"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )
        elif self.mode == "usv":
            ## add a transform called "build_weight" for parameter group ("U", "S", "V") called "usv"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )
        elif self.mode == "phase":
            ## add a transform called "build_weight" for parameter group ("phase_U", "phase_S", "phase_V") called "phase"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )

        ## add output transform
        self.add_transform(
            "output", "output", {"output_transform": self._output_transform}
        )

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        # differentiable feature is gauranteed
        weight = U.matmul(S.unsqueeze(-1) * V)
        self.weight.data.copy_(weight)
        return weight

    def build_weight_from_phase(
        self,
        delta_list_U: Tensor,
        phase_U: Tensor,
        delta_list_V: Tensor,
        phase_V: Tensor,
        phase_S: Tensor,
        update_list: set = {"phase_U", "phase_S", "phase_V"},
    ) -> Tensor:
        ### not differentiable
        ### reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        if "phase_U" in update_list:
            self.U.data.copy_(
                self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U))
            )
        if "phase_V" in update_list:
            self.V.data.copy_(
                self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V))
            )
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.cos().mul_(self.S_scale))
        return self.build_weight_from_usv(self.U, self.S, self.V)

    def build_weight_from_voltage(
        self,
        delta_list_U: Tensor,
        voltage_U: Tensor,
        delta_list_V: Tensor,
        voltage_V: Tensor,
        voltage_S: Tensor,
        gamma_U: Union[float, Tensor],
        gamma_V: Union[float, Tensor],
        gamma_S: Union[float, Tensor],
    ) -> Tensor:
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(
            delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S
        )

    def build_phase_from_usv(
        self, U: Tensor, S: Tensor, V: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        delta_list, phi_mat = self.decomposer.decompose(U.data.clone())
        self.delta_list_U.data.copy_(delta_list)
        self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))

        delta_list, phi_mat = self.decomposer.decompose(V.data.clone())
        self.delta_list_V.data.copy_(delta_list)
        self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))

        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return (
            self.delta_list_U,
            self.phase_U,
            self.delta_list_V,
            self.phase_V,
            self.phase_S,
            self.S_scale,
        )

    def build_usv_from_phase(
        self,
        delta_list_U: Tensor,
        phase_U: Tensor,
        delta_list_V: Tensor,
        phase_V: Tensor,
        phase_S: Tensor,
        S_scale: Tensor,
        update_list: Dict = {"phase_U", "phase_S", "phase_V"},
    ) -> Tuple[Tensor, ...]:
        ### not differentiable
        # reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        if "phase_U" in update_list:
            self.U.data.copy_(
                self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U))
            )
        if "phase_V" in update_list:
            self.V.data.copy_(
                self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V))
            )
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.U, self.S, self.V

    def build_usv_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ### differentiable feature is gauranteed
        # U, S, V = weight.data.svd(some=False)
        # V = V.transpose(-2, -1).contiguous()
        U, S, V = torch.linalg.svd(
            weight.data, full_matrices=True, driver="gesvd"
        )  # must use QR decomposition
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(
        self, weight: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_phase_from_usv(*self.build_usv_from_weight(weight))

    def build_voltage_from_phase(
        self,
        delta_list_U: Tensor,
        phase_U: Tensor,
        delta_list_V: Tensor,
        phase_V: Tensor,
        phase_S: Tensor,
        S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.delta_list_U = delta_list_U
        self.delta_list_V = delta_list_V
        self.voltage_U.data.copy_(phase_to_voltage(phase_U, self.gamma))
        self.voltage_S.data.copy_(phase_to_voltage(phase_S, self.gamma))
        self.voltage_V.data.copy_(phase_to_voltage(phase_V, self.gamma))
        self.S_scale.data.copy_(S_scale)

        return (
            self.delta_list_U,
            self.voltage_U,
            self.delta_list_V,
            self.voltage_V,
            self.voltage_S,
            self.S_scale,
        )

    def build_voltage_from_usv(
        self, U: Tensor, S: Tensor, V: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(U, S, V))

    def build_voltage_from_weight(
        self, weight: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(
            *self.build_phase_from_usv(*self.build_usv_from_weight(weight))
        )

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            self.build_phase_from_weight(self.weight)
        elif src == "usv":
            self.build_phase_from_usv(self.U, self.S, self.V)
            self.build_weight_from_usv(self.U, self.S, self.V)
        elif src == "phase":
            if self.w_bit < 16:
                phase_U = self.phase_U_quantizer(self.phase_U.data)
                phase_S = self.phase_S_quantizer(self.phase_S.data)
                phase_V = self.phase_V_quantizer(self.phase_V.data)
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V
            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase_U = phase_U + gen_gaussian_noise(
                    phase_U,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )
                phase_V = phase_V + gen_gaussian_noise(
                    phase_V,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            self.build_weight_from_phase(
                self.delta_list_U,
                phase_U,
                self.delta_list_V,
                phase_V,
                phase_S,
                self.S_scale,
            )
        elif src == "voltage":
            NotImplementedError
        else:
            raise NotImplementedError

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(
            noise_std, self.phase_U.size(), random_state
        )
        self.phase_S_quantizer.set_gamma_noise(
            noise_std, self.phase_S.size(), random_state
        )
        self.phase_V_quantizer.set_gamma_noise(
            noise_std, self.phase_V.size(), random_state
        )

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        self.phase_S_quantizer.set_bitwidth(w_bit)
        self.phase_V_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        self.output_quantizer.set_bit(out_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def _input_transform(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        return x

    def _weight_transform(
        self, weights: Dict, update_list: set = {"phase_U", "phase_S", "phase_V"}
    ) -> Tensor:
        if self.mode == "weight":
            weight = weights
        elif self.mode == "usv":
            U, S, V = weights
            weight = self.build_weight_from_usv(U, S, V)
        elif self.mode == "phase":
            ### not differentiable
            phase_U, phase_S, phase_V, S_scale = weights
            if (
                self.w_bit < 16
                or self.gamma_noise_std > 1e-5
                or self.crosstalk_factor > 1e-5
            ):
                phase_U = self.phase_U_quantizer(phase_U.data)
                phase_S = self.phase_S_quantizer(phase_S.data)
                phase_V = self.phase_V_quantizer(phase_V.data)

            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase_U = phase_U + gen_gaussian_noise(
                    phase_U,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )
                phase_V = phase_V + gen_gaussian_noise(
                    phase_V,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(
                self.delta_list_U,
                phase_U,
                self.delta_list_V,
                phase_V,
                phase_S,
                update_list=update_list,
            )
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        weight = merge_chunks(weight)[
            : self.out_channels, : self.in_channels_flat
        ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        return weight

    def _output_transform(self, x: Tensor) -> Tensor:
        if self.out_bit < 16:
            x = self.output_quantizer(x)
        return x

    def _forward_impl(self, x: Tensor, weights: Dict[str, Tensor]) -> Tensor:
        weight = weights["weight"]
        x = F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x
