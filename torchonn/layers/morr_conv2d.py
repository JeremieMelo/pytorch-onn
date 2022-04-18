"""
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-01-27 01:08:44
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:40:18
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.fft
from pyutils.compute import im2col_2d, toeplitz
from pyutils.general import logger
from pyutils.initializer import morr_uniform_
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.types import Device, _size
from torchonn.devices.mrr import MORRConfig_20um_MQ
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused


from .base_layer import ONNBaseLayer

__all__ = ["AllPassMORRCirculantConv2d"]


class AllPassMORRCirculantConv2d(ONNBaseLayer):
    """
    All-pass MORR Conv2d layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors.
    J. Gu, et al., "SqueezeLight: Towards Scalable Optical Neural Networks with Multi-Operand Ring Resonators"
    https://doi.org/10.23919/DATE51398.2021.9474147
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
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
        miniblock: int = 4,
        ### morr parameter
        MORRConfig=MORRConfig_20um_MQ,
        morr_init: bool = True,  # whether to use initialization method customized for MORR
        ### trainable MORR nonlinearity
        trainable_morr_bias: bool = False,
        trainable_morr_scale: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(AllPassMORRCirculantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert groups == 1, f"Currently group convolution is not supported, but got group: {groups}"
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.grid_dim_x = int(np.ceil(self.in_channels_flat / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_channels / miniblock))
        self.in_channels_pad = self.grid_dim_x * miniblock
        self.out_channels_pad = self.grid_dim_y * miniblock
        self.miniblock = miniblock

        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = 32
        self.in_bit = 32
        self.MORRConfig = MORRConfig
        self.morr_init = morr_init
        self.mrr_a = MORRConfig.attenuation_factor
        self.mrr_r = MORRConfig.coupling_factor
        self.trainable_morr_bias = trainable_morr_bias
        self.trainable_morr_scale = trainable_morr_scale
        self.device = device

        ### calculate FWHM (rad)
        self.morr_fwhm = (
            -4
            * np.pi ** 2
            * MORRConfig.radius
            * MORRConfig.effective_index
            * (
                1 / MORRConfig.resonance_wavelength
                - 1 / (MORRConfig.resonance_wavelength - MORRConfig.bandwidth / 2)
            )
        )

        ### allocate parameters
        self.weight = None
        self.x_zero_pad = None
        self.morr_output_scale = None  ## learnable balancing factors implelemt by MRRs
        self.morr_input_bias = None  ## round-trip phase shift bias within MORR
        self.morr_input_scale = None  ## scaling factor for the round-trip phase shift within MORR
        self.morr_gain = (
            100 / (self.in_channels_flat // self.miniblock)
        ) ** 0.5  ## set this TIA gain such that output variance is around 1
        ### build trainable parameters
        self.build_parameters()

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)
        self.weight_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_pos"
        )  ## [0-1] positive only, maintain the original scale
        self.morr_output_scale_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_sym"
        )  ## [-1,1] full-range

        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            a=self.mrr_a, r=self.mrr_r, intensity=True
        )

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.disable_crosstalk()
        ### default set no phase variation
        self.disable_phase_variation()

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(morr_init=morr_init)

        # support fine-grained structured pruning for MORRs
        self.finegrain_drop_mask = None

    def build_parameters(self) -> None:
        ### MORR weights
        self.weight = Parameter(
            torch.ones(
                self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device, dtype=torch.float
            )
        )
        ### learnable balancing factor achieved by MRRs (morr_output_scale)
        ### We use a single scaling factor for each block
        self.morr_output_scale = Parameter(torch.zeros(max(1, self.grid_dim_x // 2) + 1, device=self.device))
        if self.trainable_morr_bias:
            ### initialize with the finest-granularity, i.e., per mini-block
            self.morr_input_bias = Parameter(
                torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float)
            )
        if self.trainable_morr_scale:
            ### initialize with the finest-granularity, i.e., per mini-block
            self.morr_input_scale = Parameter(
                torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float)
            )

    def reset_parameters(self, morr_init: bool = False) -> None:
        if morr_init:
            ### nonlinear curve aware initialization
            morr_uniform_(
                self.weight,
                MORRConfig=self.MORRConfig,
                n_op=self.miniblock,
                biased=self.w_bit >= 16,
                gain=2 if self.in_bit < 16 else 1,
            )
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            ### output distribution aware initialization to output scaling factor
            t1 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([0]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True
            )
            t2 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([self.morr_fwhm * 2.4]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True
            )
            g = ((t2 - t1) / (2.4 * self.morr_fwhm)).item()  ## 0~2.4 FWHM slope as a linear approximation

            self.sigma_out_scale = 4 / (3 * self.grid_dim_x ** 0.5 * g * self.morr_fwhm)
            self.out_scale_quant_gain = None
            init.normal_(self.morr_output_scale, 0, self.sigma_out_scale)

        else:
            nn.init.kaiming_normal_(self.weight)
            nn.init.kaiming_normal_(self.morr_output_scale)
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            self.sigma_out_scale = self.morr_output_scale.data.std().item()
            self.out_scale_quant_gain = None

        if self.morr_input_bias is not None:
            init.zeros_(self.morr_input_bias.data)
        if self.morr_input_scale is not None:
            init.zeros_(self.morr_input_scale.data)

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """

        raise NotImplementedError

    def build_weight(self) -> Tensor:
        if self.w_bit < 16:
            ### differentiable quantizer based on STE to enable QAT (Dorefa-Net, arXiv 2016)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight.abs()  ## have to be all positive
        if self.finegrain_drop_mask is not None:
            weight = weight.mul(self.finegrain_drop_mask.float())

        return weight

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std

    def load_parameters(self, param_dict) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def input_modulator(self, x: Tensor) -> Tensor:
        ### voltage to power, which is proportional to the phase shift
        return x * x

    def set_crosstalk_coupling_matrix(self, coupling_factor: float, drop_perc: float = 0) -> None:
        ### crosstalk coupling matrix is a symmetric matrix, but the intra-MORR crosstalk can be taken as a round-trip phase shift scaling factor, which is proportional to the number of segments after pruned.
        ### See SqueezeLight paper
        ### drop-perc is the pruning percentage.
        assert 0 <= coupling_factor <= 1, logger.error(
            f"Coupling factor must in [0,1], but got {coupling_factor}"
        )

        self.crosstalk_factor = 1 + max(3, (self.miniblock * (1 - drop_perc) - 1)) * coupling_factor

    def enable_crosstalk(self) -> None:
        self.enable_thermal_crosstalk = True

    def disable_crosstalk(self) -> None:
        self.enable_thermal_crosstalk = False

    def set_phase_variation(self, phase_noise_std: float = 0) -> None:
        self.phase_noise_std = phase_noise_std

    def enable_phase_variation(self) -> None:
        self.enable_phase_noise = True

    def disable_phase_variation(self) -> None:
        self.enable_phase_noise = False

    def enable_trainable_morr_scale(self) -> None:
        self.trainable_morr_scale = True

    def disable_trainable_morr_scale(self) -> None:
        self.trainable_morr_scale = False

    def enable_trainable_morr_bias(self) -> None:
        self.trainable_morr_bias = True

    def disable_trainable_morr_bias(self) -> None:
        self.trainable_morr_bias = False

    @property
    def morr_scale(self) -> Tensor:
        return torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.2

    @property
    def morr_bias(self) -> Tensor:
        return self.morr_fwhm * torch.tanh(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))

    def propagate_morr(self, weight: Tensor, x: Tensor) -> Tensor:
        """
        @description: propagate through the analytically calculated transfer matrix of morr.
        @param weight {torch.Tensor} first column vectors in the block-circulant matrix
        @param x {torch.Tensor} input
        @return: y {torch.Tensor} output of MORR array
        """
        ### weights: [p, q, k]
        ### x: [ks*ks*inc, h_out*w_out*bs]

        x = x.t()  # [h_out*w_out*bs, ks*ks*inc]
        x = x.view(x.size(0), self.grid_dim_x, self.miniblock)  # [h_out*w_out*bs, q, k]

        ### injecting crosstalk into weights is more efficient
        if self.enable_thermal_crosstalk and self.crosstalk_factor > 1:
            weight = weight * self.crosstalk_factor

        ### construct block-circulant matrix
        weight = toeplitz(weight).unsqueeze(0)  # [1, p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)  # [h*w*bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)  # [h*w*bs, p, q, k]

        if self.enable_phase_noise and self.phase_noise_std > 1e-5:
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)  # [h*w*bs, p, q, k]

        ### input scaling, learnable MORR nonlinearity
        if self.trainable_morr_scale:
            x = x * self.morr_scale  # [h*w*bs, p, q, k]
        ### input biasing, learnable MORR nonlinearity
        if self.trainable_morr_bias:
            x = x - self.morr_bias

        ### Use theoretical transmission function for trainable MORR nonlinearity
        ### x is the phase detuning, x=0 means on-resonance
        ### x: [h_out*w_out*bs, p, q, k]
        x = self.mrr_roundtrip_phase_to_tr(x)

        ### output scaling or learnable balancing factors
        if self.w_bit < 16:
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if self.out_scale_quant_gain is None:
                self.out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(
                self.out_scale_quant_gain
            )  ### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale

        scale = morr_output_scale[:-1]
        scale_pad = morr_output_scale[-1:]

        ### differential rails
        if self.grid_dim_x % 2 == 0:
            # even blocks
            scale = torch.cat([scale, -scale], dim=0)
        else:
            # odd blocks
            if self.grid_dim_x > 1:
                scale = torch.cat([morr_output_scale, -scale], dim=0)
            else:
                scale = scale_pad
        scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, q]

        x = scale.matmul(x)  # [1,1,1,q]x[h_out*w_out*bs, p, q, k]=[h_out*w_out*bs, p, 1, k]
        x = x.view(x.size(0), -1).t()  # [p*k, h_out*w_out*bs]
        if self.out_channels_pad > self.out_channels:
            x = x[: self.out_channels, :]  # [outc, h_out*w_out*bs]
        return x

    def morr_conv2d(self, X: Tensor, W: Tensor) -> Tensor:
        ### W : [p, q, k]
        n_x = X.size(0)

        _, X_col, h_out, w_out = im2col_2d(
            None,
            X,
            stride=self.stride[0],
            padding=self.padding[0],
            w_size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
        )
        ## zero-padding X_col
        if self.in_channels_pad > self.in_channels_flat:
            if self.x_zero_pad is None or self.x_zero_pad.size(1) != X_col.size(1):
                self.x_zero_pad = torch.zeros(
                    self.in_channels_pad - self.in_channels_flat,
                    X_col.size(1),
                    dtype=torch.float32,
                    device=self.device,
                )

            X_col = torch.cat([X_col, self.x_zero_pad], dim=0)
        # matmul
        out = self.propagate_morr(W, X_col)  # [outc, w_out]
        out = out.view(self.out_channels, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def get_finegrain_drop_mask(self, topk: int) -> Tensor:
        if self.w_bit < 16:
            weight = self.weight_quantizer(self.weight.data)  # [p, q, k]
        else:
            weight = self.weight.data.abs()
        indices = weight.argsort(dim=-1)
        mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)

        drop_indices = indices[:, :, 0:-topk]
        mask.scatter_(2, drop_indices, 0)
        self.finegrain_drop_mask = mask
        return mask

    def apply_finegrain_drop_mask(self, mask: Tensor) -> None:
        if self.w_bit < 16:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), -1000)
        else:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        weight = self.build_weight()
        x = self.input_modulator(x)
        x = self.morr_conv2d(x, weight)

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x
