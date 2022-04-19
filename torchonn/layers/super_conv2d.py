"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-03 01:54:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 18:19:18
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pyutils.compute import get_complex_energy, im2col_2d
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Module, Parameter, init
from torch.types import Device, _size
from torch.nn.modules.utils import _pair

from .base_layer import ONNBaseLayer

__all__ = ["SuperBlockConv2d"]


class SuperBlockConv2d(ONNBaseLayer):
    """
    description: SVD-based Linear layer. Blocking matrix multiplication.
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        miniblock: int = 8,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        photodetect: bool = False,
        super_layer: Module = None,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(SuperBlockConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert groups == 1, f"Currently group convolution is not supported, but got group: {groups}"
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        if self.in_channels % 2 == 0:  # even channel
            self.grid_dim_x = 2 * int(np.ceil(self.in_channels_flat / 2 / miniblock))
        else:  # odd channel, mostly in the first conv layer
            self.grid_dim_x = int(np.ceil((self.in_channels_flat // 2) / miniblock)) + int(
                np.ceil((self.in_channels_flat - self.in_channels_flat // 2) / miniblock)
            )

        self.grid_dim_y = int(np.ceil(self.out_channels / miniblock))
        self.in_channels_pad = self.grid_dim_x * miniblock
        self.out_channels_pad = self.grid_dim_y * miniblock
        self.miniblock = miniblock

        self.super_layer = super_layer
        self.super_ps_layers = None
        self.set_super_ps_layer()

        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect
        self.device = device

        # build parameters
        self.build_parameters()

        # quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)

        # default set to slow forward
        self.disable_fast_forward()
        # default set no phase noise
        self.set_phase_variation(0)
        # default set no gamma noise
        self.set_gamma_noise(0)
        # default set no crosstalk
        self.set_crosstalk_factor(0)
        # zero pad for input
        self.x_zero_pad = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight = Parameter(
            torch.empty(
                self.grid_dim_y, self.grid_dim_x, self.miniblock, dtype=torch.cfloat, device=self.device
            )
        )
        self.eye = torch.eye(self.miniblock, self.miniblock, dtype=torch.cfloat, device=self.device)
        self.U = self.V = self.eye

    def set_super_layer(self, super_layer) -> None:
        self.super_layer = super_layer

    def set_super_ps_layer(self) -> None:
        if self.super_layer is not None:
            self.super_ps_layers = self.super_layer.build_ps_layers(self.grid_dim_x, self.grid_dim_y)

    def reset_parameters(self) -> None:
        temp = torch.empty(
            self.grid_dim_y * self.miniblock, self.grid_dim_x * self.miniblock, device=self.device
        )
        init.kaiming_normal_(temp)
        temp = temp.view(self.grid_dim_y, self.miniblock, self.grid_dim_x, self.miniblock).permute(0, 2, 1, 3)
        _, s, _ = torch.svd(temp, compute_uv=False)
        self.weight.data.copy_(s)
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def set_super_layer_transfer_matrices(self, U: Tensor, V: Tensor) -> None:
        self.U = U
        self.V = V

    def build_weight(self) -> Tensor:
        # [k,k] -> [k,k]
        # [p, q, k, 1] * [1, 1, k, k] complex = [p, q, k, k] complex
        weight = self.super_layer.get_weight_matrix(self.super_ps_layers, self.weight)
        weight = weight.permute(0, 2, 1, 3).reshape(self.out_channels_pad, self.in_channels_pad)[
            : self.out_channels, : self.in_channels_flat
        ]

        return weight

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        return int(h_out), int(w_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k] or u, s, v
        else:
            weight = self.weight
        _, x, h_out, w_out = im2col_2d(
            W=None,
            X=x,
            stride=self.stride[0],
            padding=self.padding[0],
            w_size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
        )

        inc_pos = int(np.ceil(weight.size(1) / 2))
        x = x.to(torch.complex64)
        x_pos = weight[:, :inc_pos].matmul(x[:inc_pos])  # [outc, h*w*bs]
        x_neg = weight[:, inc_pos:].matmul(x[inc_pos:])  # [outc, h*w*bs]
        if self.photodetect:
            x = get_complex_energy(torch.view_as_real(x_pos)) - get_complex_energy(torch.view_as_real(x_neg))
        else:
            x = x_pos - x_neg
        out = x.view(self.out_channels, h_out, w_out, -1).permute(3, 0, 1, 2)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out
