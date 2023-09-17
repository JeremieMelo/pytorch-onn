"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 17:28:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 17:28:45
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pyutils.compute import gen_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device
from torchonn.layers.base_layer import ONNBaseLayer
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

__all__ = [
    "MZILinear",
    "MZIBlockLinear",
]


class MZILinear(ONNBaseLayer):
    """
    SVD-based Linear layer constructed by cascaded MZIs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = True,
        device: Device = torch.device("cpu"),
    ):
        super(MZILinear, self).__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect
        self.decompose_alg = decompose_alg

        ### build trainable parameters
        self.build_parameters(mode)
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch(alg=decompose_alg)
        if decompose_alg == "clements":
            self.decomposer.v2m = vector_to_checkerboard
            self.decomposer.m2v = checkerboard_to_vector
            mesh_mode = "rectangle"
            crosstalk_filter_size = 5
        elif decompose_alg in {"reck", "francis"}:
            self.decomposer.v2m = vector_to_upper_triangle
            self.decomposer.m2v = upper_triangle_to_vector
            mesh_mode = "triangle"
            crosstalk_filter_size = 3

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
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

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        weight = torch.Tensor(self.out_features, self.in_features).to(self.device)
        ## usv mode
        U = torch.Tensor(self.out_features, self.out_features).to(self.device)
        S = torch.Tensor(min(self.out_features, self.in_features)).to(self.device)
        V = torch.Tensor(self.in_features, self.in_features).to(self.device)
        ## phase mode
        delta_list_U = torch.Tensor(self.out_features).to(self.device)
        phase_U = torch.Tensor(self.out_features * (self.out_features - 1) // 2).to(self.device)
        phase_S = torch.Tensor(min(self.out_features, self.in_features)).to(self.device)
        delta_list_V = torch.Tensor(self.in_features).to(self.device)
        phase_V = torch.Tensor(self.in_features * (self.in_features - 1) // 2).to(self.device)
        # TIA gain
        S_scale = torch.ones(1).to(self.device).float()

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "usv":
            self.U = Parameter(U)
            self.S = Parameter(S)
            self.V = Parameter(V)
        elif mode == "phase":
            self.phase_U = Parameter(phase_U)
            self.phase_S = Parameter(phase_S)
            self.phase_V = Parameter(phase_V)
            self.S_scale = Parameter(S_scale)
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "U": U,
            "S": S,
            "V": V,
            "phase_U": phase_U,
            "phase_S": phase_S,
            "phase_V": phase_V,
            "S_scale": S_scale,
            "delta_list_U": delta_list_U,
            "delta_list_V": delta_list_V,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode == "weight":
            init.kaiming_normal_(self.weight.data)
        elif self.mode == "usv":
            W = init.kaiming_normal_(
                torch.empty(self.out_features, self.in_features, dtype=self.U.dtype, device=self.device)
            )
            # U, S, V = torch.svd(W, some=False)
            # V = V.transpose(-2, -1)
            U, S, V = torch.linalg.svd(W, full_matrices=True, driver="gesvd")  # must use QR decomposition
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(torch.ones(S.shape[0], dtype=self.U.dtype, device=self.device))
        elif self.mode == "phase":
            W = init.kaiming_normal_(
                torch.empty(self.out_features, self.in_features, dtype=self.U.dtype, device=self.device)
            )
            # U, S, V = torch.svd(W, some=False)
            # V = V.transpose(-2, -1)
            U, S, V = torch.linalg.svd(W, full_matrices=True, driver="gesvd")  # must use QR decomposition
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

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = True,
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted MZILinear module
        """
        assert isinstance(layer, nn.Linear), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode=mode,
            decompose_alg=decompose_alg,
            photodetect=photodetect,
            device=device,
        ).to(device)
        instance.weight.data.copy_(layer.weight)
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        ### differentiable feature is gauranteed
        if self.out_features == self.in_features:
            weight = torch.mm(U, S.unsqueeze(1) * V)
        elif self.out_features > self.in_features:
            weight = torch.mm(U[:, : self.in_features], S.unsqueeze(1) * V)
        else:
            weight = torch.mm(U * S.unsqueeze(0), V[: self.out_features, :])
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
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
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
        return self.build_weight_from_phase(delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S)

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

        return self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S, self.S_scale

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
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.U, self.S, self.V

    def build_usv_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ### differentiable feature is gauranteed
        # U, S, V = weight.data.svd(some=False) # old version is not precise
        # V = V.transpose(-2, -1).contiguous()
        U, S, V = torch.linalg.svd(weight.data, full_matrices=True, driver="gesvd")  # must use QR decomposition
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(*self.build_usv_from_weight(weight)))

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

            self.build_weight_from_phase(self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, self.S_scale)
        elif src == "voltage":
            NotImplementedError
        else:
            raise NotImplementedError

    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "usv":
            U = self.U
            V = self.V
            S = self.S
            weight = self.build_weight_from_usv(U, S, V)
        elif self.mode == "phase":
            ### not differentiable
            if self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5:
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

            weight = self.build_weight_from_phase(
                self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, update_list=update_list
            )
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return weight

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

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
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()
        else:
            weight = self.weight
        x = F.linear(x, weight, bias=None)
        if self.photodetect:
            x = x.square()

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x


class MZIBlockLinear(ONNBaseLayer):
    """
    SVD-based blocking Linear layer constructed by cascaded MZIs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        miniblock: int = 4,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = True,
        device: Device = torch.device("cpu"),
    ):
        super(MZIBlockLinear, self).__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock))
        self.in_features_pad = self.grid_dim_x * miniblock
        self.out_features_pad = self.grid_dim_y * miniblock
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect
        self.decompose_alg = decompose_alg

        ### build trainable parameters
        self.build_parameters(mode)
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch(alg=decompose_alg)
        if decompose_alg == "clements":
            self.decomposer.v2m = vector_to_checkerboard
            self.decomposer.m2v = checkerboard_to_vector
            mesh_mode = "rectangle"
            crosstalk_filter_size = 5
        elif decompose_alg in {"reck", "francis"}:
            self.decomposer.v2m = vector_to_upper_triangle
            self.decomposer.m2v = upper_triangle_to_vector
            mesh_mode = "triangle"
            crosstalk_filter_size = 3

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
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

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        weight = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock).to(self.device)
        ## usv mode
        U = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock).to(self.device)
        S = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device)
        V = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock).to(self.device)
        ## phase mode
        delta_list_U = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device)
        phase_U = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock * (self.miniblock - 1) // 2).to(
            self.device
        )
        phase_S = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device)
        delta_list_V = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device)
        phase_V = torch.Tensor(self.grid_dim_y, self.grid_dim_x, self.miniblock * (self.miniblock - 1) // 2).to(
            self.device
        )
        # TIA gain
        S_scale = torch.Tensor(self.grid_dim_y, self.grid_dim_x, 1).to(self.device).float()

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "usv":
            self.U = Parameter(U)
            self.S = Parameter(S)
            self.V = Parameter(V)
        elif mode == "phase":
            self.phase_U = Parameter(phase_U)
            self.phase_S = Parameter(phase_S)
            self.phase_V = Parameter(phase_V)
            self.S_scale = Parameter(S_scale)
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "U": U,
            "S": S,
            "V": V,
            "phase_U": phase_U,
            "phase_S": phase_S,
            "phase_V": phase_V,
            "S_scale": S_scale,
            "delta_list_U": delta_list_U,
            "delta_list_V": delta_list_V,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode == "weight":
            init.kaiming_normal_(self.weight.data)
        elif self.mode == "usv":
            W = init.kaiming_normal_(
                torch.empty(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    self.miniblock,
                    self.miniblock,
                    dtype=self.U.dtype,
                    device=self.device,
                )
            )
            # U, S, V = torch.svd(W, some=False)
            # V = V.transpose(-2, -1)
            U, S, V = torch.linalg.svd(W, full_matrices=True, driver="gesvd")  # must use QR decomposition
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(S)
        elif self.mode == "phase":
            W = init.kaiming_normal_(
                torch.empty(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    self.miniblock,
                    self.miniblock,
                    dtype=self.U.dtype,
                    device=self.device,
                )
            )
            # U, S, V = torch.svd(W, some=False)
            # V = V.transpose(-2, -1)
            U, S, V = torch.linalg.svd(W, full_matrices=True, driver="gesvd")  # must use QR decomposition
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

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        miniblock: int = 4,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = True,
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            miniblock (int, optional): miniblock size. Defaults to 4.
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted MZIBlockLinear module
        """
        assert isinstance(layer, nn.Linear), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            miniblock=miniblock,
            mode=mode,
            decompose_alg=decompose_alg,
            photodetect=photodetect,
            device=device,
        ).to(device)
        weight = instance.weight
        tmp = torch.zeros(instance.out_features_pad, instance.in_features_pad, device=instance.device)
        tmp.data[:out_features, :in_features].copy_(layer.weight)
        instance.weight.data.copy_(
            tmp.view(weight.shape[0], weight.shape[2], weight.shape[1], weight.shape[3]).permute(0, 2, 1, 3)
        )
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

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
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
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
        return self.build_weight_from_phase(delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S)

    def build_phase_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        delta_list, phi_mat = self.decomposer.decompose(U.data.clone())
        self.delta_list_U.data.copy_(delta_list)
        self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))

        delta_list, phi_mat = self.decomposer.decompose(V.data.clone())
        self.delta_list_V.data.copy_(delta_list)
        self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))

        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S, self.S_scale

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
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.U, self.S, self.V

    def build_usv_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ### differentiable feature is gauranteed
        # U, S, V = weight.data.svd(some=False)
        # V = V.transpose(-2, -1).contiguous()
        U, S, V = torch.linalg.svd(weight.data, full_matrices=True, driver="gesvd")  # must use QR decomposition
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(*self.build_usv_from_weight(weight)))

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

            self.build_weight_from_phase(self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, self.S_scale)
        elif src == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "usv":
            U = self.U
            V = self.V
            S = self.S
            weight = self.build_weight_from_usv(U, S, V)
        elif self.mode == "phase":
            ### not differentiable
            if self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5:
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
            weight = self.build_weight_from_phase(
                self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, update_list=update_list
            )
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return weight

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

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
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k]
        else:
            weight = self.weight
        weight = merge_chunks(weight)[: self.out_features, : self.in_features]
        x = F.linear(x, weight, bias=None)
        if self.photodetect:
            x = x.square()

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x
