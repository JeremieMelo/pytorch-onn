"""
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:36:52
LastEditTime: 2022-04-18 17:12:39
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
Description:
"""

import logging
import math  # some math operations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from pyutils.general import print_stat
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.types import Device, _size
from torchonn.op.pcm_op import (
    weight_quantize_fn_log,
    weight_to_quantized_weight,
    weight_to_quantized_weight_cpu,
)

from .base_layer import ONNBaseLayer

__all__ = ["PCMConv2d"]


class PCMConv2d(ONNBaseLayer):
    """
    Conv2d layer based on phase change material (PCM) crossbar array.
    H. Zhu, et al., "ELight: Enabling Efficient Photonic In-Memory Neurocomputing with Life Enhancement", ASP-DAC 2022
    https://arxiv.org/pdf/2112.08512.pdf
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
        "block_size",
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
    __mode_list__ = ["weight", "block"]

    def __init__(
        self,
        ## normal params
        in_channels: int,
        out_channels: int,
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
        block_size: int = 16,
        mode: str = "weight",
        ## quantization
        input_quant_method: str = "uniform_noise",
        ## quant_noise
        quant_ratio: float = 1,
        weight_quant_method: str = "log",
        ## loss_flag
        loss_fn: str = "l1",
        ## pcm params
        pcm_l: float = 0.128,
        assign: bool = True,
        has_zero: bool = True,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(PCMConv2d, self).__init__()

        # param init
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert groups == 1, f"Currently group convolution is not supported, but got group: {groups}"

        self.w_bit = 32
        self.in_bit = 32
        self.pcm_l = pcm_l
        self.quant_ratio = quant_ratio
        self.device = device

        # PTC param
        self.mode = mode
        assert mode in {"weight", "block"}, logging.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {mode}."
        )
        self.block_size = block_size
        self.block_mul_flag = True if (mode == "block") else False

        self.input_quant_method = input_quant_method
        self.weight_quant_method = weight_quant_method
        self.has_zero = has_zero

        self.assign = assign
        self.stuck_fault = False
        self.tolerate_stuck_fault = False

        ## allocate the trainable parameters
        self.weight = None
        self.build_parameters()
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        ## Initialize quantization tools
        self.input_quantizer = input_quantize_fn(
            self.in_bit, alg="normal", device=self.device, quant_ratio=self.quant_ratio
        )

        if self.weight_quant_method == "log":
            self.weight_quantizer = weight_quantize_fn_log(
                self.w_bit,
                power_base=1 - self.pcm_l,
                has_zero=self.has_zero,
                power=True,
                assign=self.assign,
                device=self.device,
            )
        else:
            raise NotImplementedError

        assign_zero_value = 2 ** self.w_bit - 1

        if self.assign and (self.w_bit < 16):
            self.assign_converter = weight_to_quantized_weight(
                self.w_bit, 1 - self.pcm_l, True, self.assign, assign_zero_value, loss_fn
            )
            self.real_assign_converter = weight_to_quantized_weight_cpu(
                self.w_bit, 1 - self.pcm_l, True, self.assign, assign_zero_value
            )
        else:
            self.assign_converter = None
            self.real_assign_converter = None

        # defualt settings
        self.disable_fast_forward()

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def build_parameters(self) -> None:
        if self.mode == "weight":
            self.weight = Parameter(
                torch.Tensor(
                    self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
                ).to(self.device)
            )
        elif self.mode == "block":
            # m*n -> p*q*k*k -> p*k = m; q*k = n
            self.weight = Parameter(
                torch.Tensor(
                    (self.out_channels + self.block_size - 1) // self.block_size,
                    (self.in_channels * self.kernel_size[0] * self.kernel_size[1] + self.block_size - 1)
                    // self.block_size,
                    self.block_size,
                    self.block_size,
                ).to(self.device)
            )
        else:
            self.weight = Parameter(
                torch.Tensor(
                    self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
                ).to(self.device)
            )

    def reset_parameters(self) -> None:
        # init weight for Relu
        init.kaiming_normal_(self.weight.data, mode="fan_out", nonlinearity="relu")

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, 0, 0)

    def build_weight(self) -> Tensor:
        if self.mode in {"weight"}:
            if self.w_bit < 16:
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight
                weight = weight.view(
                    self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
                )
        elif self.mode == "block":
            if self.w_bit < 16:
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight

            # reshape weight
            p, q, k, k = weight.size()
            weight = (
                weight.permute([0, 2, 1, 3])
                .contiguous()
                .view(p * k, q * k)[
                    : self.out_channels, : self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                ]
                .view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
                .contiguous()
            )

        return weight

    def enable_tolerance(self) -> None:
        self.tolerate_stuck_fault = True

    def disable_tolerance(self) -> None:
        self.tolerate_stuck_fault = False

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.weight_quantizer.set_bitwidth(w_bit)
        if self.assign_converter is not None:
            self.assign_converter.set_bitwidth(w_bit)
        if self.real_assign_converter is not None:
            self.real_assign_converter.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def forward(self, x: Tensor) -> Tensor:
        # need first update quant_noise' p at first
        if self.in_bit <= 8:
            if "noise" in self.input_quant_method:
                p = self.quant_ratio if self.training else 1
                self.input_quantizer.set_quant_ratio(p)
                x = self.input_quantizer(x)
            else:
                x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()
        else:
            weight = self.weight

        out = F.conv2d(x, weight, self.bias, self.stride, self.padding)

        return out

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        """
        get the output features size
        """
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        return (int(h_out), int(w_out))

    def get_difference_loss_global_L1(self, loss_flag: bool) -> Tensor:
        """
        Obtain the difference loss between blocks and ref block in a global manner
        """
        loss = 0
        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            weight = self.assign_converter(weight)  # transfer weight to transmission levels
            base = torch.mean(weight, dim=0, keepdim=True).detach()  # get one global reference block
            ref = base.repeat(p * q, 1)

            if loss_flag:
                loss = F.l1_loss(weight, ref, reduction="mean")
            else:
                tmp = F.l1_loss(weight, ref, reduction="none").detach().sum(dim=1).div(k * k)
                self.block_full_differences = tmp.cpu().numpy().tolist()
        else:
            loss = 0

        return loss

    def get_difference_loss_global_L2(self, loss_flag: bool) -> Tensor:
        """L2 loss
        Obtain the difference loss between blocks and ref block in a global manner
        """
        loss = 0
        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            weight = self.assign_converter(weight)
            base = torch.mean(weight, dim=0, keepdim=True).detach()  # get one global reference block
            ref = base.repeat(p * q, 1)

            if loss_flag:
                loss = F.mse_loss(weight, ref, reduction="mean")
            else:
                tmp = F.mse_loss(weight, ref, reduction="none").detach().sum(dim=1).div(k * k)
                self.block_full_differences_real = tmp.cpu().numpy().tolist()
        else:
            loss = 0

        return loss

    def get_difference_loss_row_L1(self, loss_flag: bool) -> Tensor:
        """
        Obtain the difference loss between blocks and ref block in a row-wise manner
        """
        loss = 0

        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            weight = self.assign_converter(weight)

            for i in range(p):
                base = torch.mean(
                    weight[i * q : i * q + q], dim=0, keepdim=True
                ).detach()  # get one reference block for each row
                if i == 0:
                    ref = base.repeat(q, 1)
                else:
                    ref = torch.cat((ref, base.repeat(q, 1)), 0)

            if loss_flag:
                loss = F.l1_loss(weight, ref, reduction="mean")
            else:
                tmp = F.l1_loss(weight, ref, reduction="none").detach().sum(dim=1).div(k * k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row = []
                for i in range(p):
                    self.block_avr_differences_row.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0

        return loss

    def get_difference_loss_row_L2(self, loss_flag: bool) -> Tensor:
        """
        Obtain the difference loss between blocks and ref block in a row-wise manner
        """
        loss = 0

        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            weight = self.assign_converter(weight)
            for i in range(p):
                base = torch.mean(
                    weight[i * q : i * q + q], dim=0, keepdim=True
                ).detach()  # get one reference block for each row
                if i == 0:
                    ref = base.repeat(q, 1)
                else:
                    ref = torch.cat((ref, base.repeat(q, 1)), 0)
            if loss_flag:
                loss = F.mse_loss(weight, ref, reduction="mean")
            else:
                tmp = F.mse_loss(weight, ref, reduction="none").detach().sum(dim=1).div(k * k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row_real = []
                for i in range(p):
                    self.block_avr_differences_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0

        return loss

    def get_difference_loss_nei_L2(self, loss_flag: bool) -> Tensor:
        """
        Obtain the difference loss between neighbouring blocks
        """
        loss = 0

        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            weight = self.assign_converter(weight)

            indices = torch.arange(0, p * q, 1).long()
            indices = indices - 1
            weight_ref = weight.detach().clone()

            for i in range(p):
                base = torch.mean(weight[i * q : i * q + q], dim=0, keepdim=True).detach()
                indices[i * q] = p * q + i
                weight_ref = torch.cat((weight_ref, base), 0)

            weight_ref = weight_ref[indices]

            if loss_flag:
                loss = F.mse_loss(weight, weight_ref, reduction="mean")
            else:
                tmp = F.mse_loss(weight, weight_ref, reduction="none").detach().sum(dim=1).div(k * k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row_real = []
                for i in range(p):
                    self.block_avr_differences_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0

        return loss

    def get_programming_levels_global_real(self, loss_flag: bool) -> Tensor:
        """
        Compute the total number of write operations
            Use the avearge block for each row as the initialization block
        """
        loss = 0

        if self.block_mul_flag == True:
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p * q, -1)
            _, weight = self.real_assign_converter.forward(weight)

            indices = torch.arange(0, p * q, 1).long()
            indices = indices - 1
            weight_ref = weight.detach().clone()
            for i in range(p):
                base = torch.mean(weight[i * q : i * q + q], dim=0, keepdim=True)
                indices[i * q] = p * q + i
                weight_ref = torch.cat((weight_ref, base), 0)

            weight_ref = weight_ref[indices]

            if loss_flag:
                loss = F.l1_loss(weight, weight_ref, reduction="mean")
            else:
                tmp = F.l1_loss(weight, weight_ref, reduction="none").detach().sum(dim=1).div(k * k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.programming_levels_avr_row_real = []
                for i in range(p):
                    self.programming_levels_avr_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0

        return loss
