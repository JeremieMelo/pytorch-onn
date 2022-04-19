"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-04-18 22:15:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-04-18 22:34:04
"""
from typing import Optional

import numpy as np
import torch
from pyutils.compute import get_complex_energy
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Module, Parameter, init
from torch.types import Device

from .base_layer import ONNBaseLayer

__all__ = ["SuperBlockLinear"]


class SuperBlockLinear(ONNBaseLayer):
    """
    description: SVD-based (U Sigma V) Linear layer. Blocking matrix multiplication.
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
        miniblock: int = 8,
        photodetect: bool = False,
        super_layer: Module = None,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(SuperBlockLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock))
        self.in_features_pad = self.grid_dim_x * miniblock
        self.out_features_pad = self.grid_dim_y * miniblock
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
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self, mode: str = "weight") -> None:
        ### this weight is only the Diagonal matrix Sigma
        ### Parameters in U and V are from super_ps_layer
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
        ## Here we use Phi^U, Sigma, and Phi^V to construct W = U(Phi^U) x Sigma x V(Phi^V)
        ## self.weight is only the Diagonal matrix Sigma.
        ## Parameters in U and V are the phase shifters in the SuperBatchedPSLayer (self.super_ps_layer)
        weight = self.super_layer.get_weight_matrix(self.super_ps_layers, self.weight)
        weight = weight.permute(0, 2, 1, 3).reshape(self.out_features_pad, self.in_features_pad)[
            : self.out_features, : self.in_features
        ]
        return weight

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.phase_noise_std = noise_std

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_gamma_noise(self, noise_std: float = 0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit <= 8:
            x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k] or u, s, v
        else:
            weight = self.weight

        inc_pos = int(np.ceil(weight.size(1) / 2))
        weight = weight.t()
        x = x.to(torch.complex64)
        x_pos = x[..., :inc_pos].matmul(weight[:inc_pos, :])  # [bs, outc]
        x_neg = x[..., inc_pos:].matmul(weight[inc_pos:, :])  # [outc, bs]
        if self.photodetect:
            out = get_complex_energy(torch.view_as_real(x_pos)) - get_complex_energy(
                torch.view_as_real(x_neg)
            )
        else:
            out = x_pos - x_neg

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out
