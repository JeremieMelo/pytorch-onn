"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-28 00:13:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-28 00:23:47
"""

from typing import Any, Dict, Optional

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
from torchonn.op.butterfly_op import TrainableButterfly
from torchonn.op.mzi_op import PhaseQuantizer

from .utils import merge_chunks, partition_chunks

__all__ = [
    "FFTONNBlockConv2d",
]


@MODELS.register_module()
class FFTONNBlockConv2d(ONNBaseConv2d):
    """
    Butterfly blocking Conv2d layer.
    J. Gu, et al., "Towards Area-Efficient Optical Neural Networks: An FFT-based Architecture," ASP-DAC 2020.
    https://ieeexplore.ieee.org/document/9045156
    """

    ## default configs
    default_cfgs = dict(
        miniblock=(
            1,
            1,
            4,
            4,
        ),  # [#tiles, pe per tile, row, col] # i.e., [R, C, k1, k2]
        mode="fft",
        w_bit=32,
        in_bit=32,
        out_bit=32,
        v_max=10.8,
        v_pi=4.36,
        photodetect="coherent",
        device=torch.device("cpu"),
    )
    __mode_list__ = ["fft", "hadamard", "zero_bias", "trainable"]

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
        self.build_parameters()
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
        assert self.mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {self.mode}."
        )
        self.gamma = np.pi / self.v_pi**2
        assert self.miniblock[-1] == self.miniblock[-2], logger.error(
            f"Currently only support square miniblock, but got {self.miniblock}."
        )

        assert self.photodetect in ["coherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )

    def build_parameters(self) -> None:
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        self.T = TrainableButterfly(
            length=self.miniblock[-1],
            reverse=False,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            device=self.device,
        )
        self.S = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-1],
            dtype=torch.cfloat,
        ).to(self.device)  # complex frequency-domain weights
        self.Tr = TrainableButterfly(
            length=self.miniblock[-1],
            reverse=True,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            device=self.device,
        )

        self.register_parameter_buffer(*self.get_param_buffer_groups(self.mode))

        self.pack_weights()

    def get_param_buffer_groups(self, mode: str) -> Tensor:
        param_groups = {
            "S": self.S,
        }
        buffer_groups = {"weight": self.weight}
        return param_groups, buffer_groups

    def pack_weights(self):
        ## key is self.mode, which should match the src_name for weight_transform
        self.weights = {self.mode: (self.Tr.phases, self.S, self.T.phases)}

    @property
    def phase_U(self):
        return self.Tr.phases

    @property
    def phase_V(self):
        return self.T.phases

    def reset_parameters(self, mode: Optional[str] = None) -> None:
        mode = mode or self.mode
        W = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            groups=self.groups,
        ).weight.data.to(self.device)
        W = partition_chunks(W.flatten(1), self.weight.shape)

        if self.device.type == "cpu":
            S = torch.linalg.svdvals(W)
        else:
            S = torch.linalg.svdvals(W, driver="gesvd")  # must use QR decomposition
        self.S.data.copy_(S)

        if mode == "zero_bias":
            self.T.reset_parameters(alg="zero")
            self.Tr.reset_parameters(alg="zero")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "hadamard":
            self.T.reset_parameters(alg="hadamard")
            self.Tr.reset_parameters(alg="hadamard")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "fft":
            self.T.reset_parameters(alg="fft")
            self.Tr.reset_parameters(alg="fft")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "trainable":
            self.T.reset_parameters(alg="uniform")
            self.Tr.reset_parameters(alg="uniform")
            self.T.phases.requires_grad_(True)
            self.Tr.phases.requires_grad_(True)
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def switch_mode_to(self, mode: str) -> None:
        super().switch_mode_to(mode)
        self.register_parameter_buffer(*self.get_param_buffer_groups(mode=mode))
        self.pack_weights()

    def build_transform(self) -> None:
        ### quantization tool
        self.input_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.in_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        crosstalk_filter_size = 3
        self.phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="butterfly",
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
            mode="butterfly",
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

    def sync_parameters(
        self, src: str = "weight", steps: int = 3000, verbose: bool = False
    ) -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "phase":
            self.weight.data.copy_(
                partition_chunks(
                    self.transform_weight(self.weights)["weight"],
                    out_shape=self.weight.shape,
                )
            )
        elif src == "weight":
            # weight to phase
            target = self.weight.data
            params = {}
            phase_U = self.Tr.phases
            phase_V = self.T.phases
            if (
                self.w_bit < 16
                or self.gamma_noise_std > 1e-5
                or self.crosstalk_factor > 1e-5
            ) and phase_U.requires_grad:
                phase_U = self.phase_U_quantizer(phase_U)
                phase_V = self.phase_V_quantizer(phase_V)

            if not self.T.phases.requires_grad:
                target = target.matmul(
                    torch.linalg.inv(self.T.build_weight(phase_V.data))
                )
            else:
                params["V"] = self.T.phases
            if not self.Tr.phases.requires_grad:
                target = torch.linalg.inv(self.Tr.build_weight(phase_U.data)).matmul(
                    target
                )
            else:
                params["U"] = self.Tr.phases

            params["S"] = self.S
            if (
                len(params) == 1
            ):  # only has self.S, perform optimal singular value projection
                self.S.data.copy_(torch.linalg.diagonal(target))
            else:  # perform gradient descent to solve
                target = merge_chunks(self.weight.data)[
                    : self.out_channels, : self.in_channels_flat
                ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

                def build_weight_fn():
                    return self.transform_weight(self.weights)["weight"]

                self.map_layer(
                    target=target,
                    param_list=list(params.values()),
                    build_weight_fn=build_weight_fn,
                    mode="regression",
                    num_steps=steps,
                    verbose=verbose,
                )

        else:
            raise NotImplementedError

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(
            noise_std, self.phase_U.size(), random_state
        )
        # self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(
            noise_std, self.phase_V.size(), random_state
        )

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        # self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        # self.phase_S_quantizer.set_bitwidth(w_bit)
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

    def _input_transform(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        return x

    def _weight_transform(
        self, weights: Dict, update_list: set = {"phase_U", "S", "phase_V"}
    ) -> Tensor:
        ### not differentiable
        phase_U, S, phase_V = weights
        if (
            self.w_bit < 16
            or self.gamma_noise_std > 1e-5
            or self.crosstalk_factor > 1e-5
        ) and phase_U.requires_grad:
            phase_U = self.phase_U_quantizer(phase_U)
            phase_V = self.phase_V_quantizer(phase_V)

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

        weight = self.build_weight_from_usv(
            self.Tr.build_weight(phase_U),
            S,
            self.T.build_weight(phase_V),
        )

        weight = merge_chunks(weight)[
            : self.out_channels, : self.in_channels_flat
        ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        if self.photodetect == "coherent":
            weight = weight.real - weight.imag
        else:
            raise NotImplementedError(
                f"Photodetect mode {self.photodetect} not implemented."
            )
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
