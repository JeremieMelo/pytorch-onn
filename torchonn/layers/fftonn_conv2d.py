"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-28 00:13:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-28 00:23:47
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import im2col_2d, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device, _size
from torch.nn.modules.utils import _pair
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.butterfly_op import TrainableButterfly
from torchonn.op.mzi_op import PhaseQuantizer

__all__ = [
    "FFTONNBlockConv2d",
]


class FFTONNBlockConv2d(ONNBaseLayer):
    """
    Butterfly blocking Conv2d layer.
    J. Gu, et al., "Towards Area-Efficient Optical Neural Networks: An FFT-based Architecture," ASP-DAC 2020.
    https://ieeexplore.ieee.org/document/9045156
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "miniblock",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    miniblock: int
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
        miniblock: int = 4,
        mode: str = "fft",
        photodetect: bool = True,
        device: Device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert groups == 1, f"Currently group convolution is not supported, but got group: {groups}"
        self.miniblock = miniblock
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.grid_dim_x = int(np.ceil(self.in_channels_flat / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_channels / miniblock))
        self.in_channels_pad = self.grid_dim_x * miniblock
        self.out_channels_pad = self.grid_dim_y * miniblock
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect

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
        # self.phase_S_quantizer = PhaseQuantizer(
        #     self.w_bit,
        #     self.v_pi,
        #     self.v_max,
        #     gamma_noise_std=0,
        #     crosstalk_factor=0,
        #     crosstalk_filter_size=crosstalk_filter_size,
        #     random_state=0,
        #     mode="diagonal",
        #     device=self.device,
        # )

        ### build trainable parameters
        self.build_parameters()

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(mode=self.mode)

    def build_parameters(self) -> None:
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            self.miniblock,
            self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )
        self.T = TrainableButterfly(
            length=self.miniblock,
            reverse=False,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            phase_quantizer=self.phase_U_quantizer,
            device=self.device,
        )
        self.S = Parameter(
            torch.zeros(self.grid_dim_y, self.grid_dim_x, self.miniblock, dtype=torch.cfloat).to(self.device)
        )  # complex frequency-domain weights
        self.Tr = TrainableButterfly(
            length=self.miniblock,
            reverse=True,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            phase_quantizer=self.phase_V_quantizer,
            device=self.device,
        )

    @property
    def U(self):
        return self.Tr.build_weight()

    @property
    def V(self):
        return self.T.build_weight()

    @property
    def phase_U(self):
        return self.Tr.phases

    @property
    def phase_V(self):
        return self.T.phases

    def reset_parameters(self, mode: Optional[str] = None) -> None:
        mode = mode or self.mode
        W = init.kaiming_normal_(
            torch.empty(
                self.grid_dim_y,
                self.grid_dim_x,
                self.miniblock,
                self.miniblock,
                dtype=torch.cfloat,
                device=self.device,
            )
        )
        _, S, _ = torch.svd(W, compute_uv=False)
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

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        # differentiable feature is gauranteed
        weight = U.matmul(S.unsqueeze(-1) * V)
        self.weight.data.copy_(weight)
        return weight

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        self.weight.data.copy_(self.build_weight_from_usv(self.U.data, self.S.data, self.V.data))

    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        weight = self.build_weight_from_usv(self.U, self.S, self.V)

        return weight

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        # self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

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
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        """
        get the output features size
        """
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        return (int(h_out), int(w_out))

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k]
        else:
            weight = self.weight

        bs = x.size(0)
        offset = int(np.ceil(self.grid_dim_x / 2)) * self.miniblock
        weight = merge_chunks(weight)[
            : self.out_channels, : self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        ]
        _, x, h_out, w_out = im2col_2d(
            W=None,
            X=x,
            stride=self.stride[0],
            padding=self.padding[0],
            w_size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
        )
        x = x.to(torch.cfloat)
        weight_pos = weight[:, :offset]
        weight_neg = weight[:, offset:]
        x_pos = x[:offset, :]
        x_neg = x[offset:, :]
        x_pos = weight_pos.matmul(x_pos)
        x_pos = x_pos.real.square() + x_pos.imag.square()
        x_neg = weight_neg.matmul(x_neg)
        x_neg = x_neg.real.square() + x_neg.imag.square()
        x = x_pos - x_neg
        x = x.view(self.out_channels, h_out, w_out, bs).permute(3, 0, 1, 2).contiguous()

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return x
