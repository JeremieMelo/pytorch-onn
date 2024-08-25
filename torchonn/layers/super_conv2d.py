"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-03 01:54:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 18:19:18
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.general import logger
from pyutils.quant.lsq import ActQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Module, Parameter
from torch.types import _size

from torchonn.layers.base_layer import ONNBaseConv2d

from .utils import merge_chunks, partition_chunks

__all__ = ["SuperBlockConv2d"]


@MODELS.register_module()
class SuperBlockConv2d(ONNBaseConv2d):
    """
    description: SVD-based Linear layer. Blocking matrix multiplication.
    """

    ## default configs
    default_cfgs = dict(
        miniblock=(
            1,
            1,
            4,
            4,
        ),  # [#tiles, pe per tile, row, col] # i.e., [R, C, k1, k2]
        mode="phase",
        w_bit=32,
        in_bit=32,
        out_bit=32,
        photodetect="coherent",
        device=torch.device("cpu"),
    )
    __mode_list__ = ["phase", "usv", "weight"]

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
        super_layer: Optional[Module] = None,
        **cfgs,
    ):
        super().__init__()
        self.load_cfgs(
            super_layer=super_layer,
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
            self.bias = Parameter(torch.zeros(out_channels, device=self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def load_cfgs(
        self,
        super_layer: Optional[Module] = None,
        **cfgs,
    ) -> None:
        super().load_cfgs(**cfgs)

        ## verify configs
        assert self.mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {self.mode}."
        )

        assert self.miniblock[-1] == self.miniblock[-2], logger.error(
            f"Currently only support square miniblock, but got {self.miniblock}."
        )

        assert self.photodetect in ["coherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )
        self.super_layer = super_layer
        self.super_ps_layers = None

    def build_parameters(self) -> None:
        ### this weight is only the Diagonal matrix Sigma
        ### Parameters in U and V are from super_ps_layer
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )
        self.S = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-1],
            dtype=torch.cfloat,
            device=self.device,
        )
        self.U = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )
        self.V = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )
        self.eye = torch.eye(
            *self.miniblock[-2:], dtype=torch.cfloat, device=self.device
        )
        self.set_super_ps_layer()
        self.register_parameter_buffer(*self.get_param_buffer_groups(mode=self.mode))

        self.pack_weights()

    def get_param_buffer_groups(self, mode: str) -> Tuple[dict, dict]:
        param_groups = {
            "S": self.S,
        }
        buffer_groups = {"weight": self.weight, "U": self.U, "V": self.V}
        return param_groups, buffer_groups

    def pack_weights(self):
        ## key is self.mode, which should match the src_name for weight_transform
        self.weights = {self.mode: (self.super_ps_layers, self.S)}

    def set_super_layer(self, super_layer) -> None:
        self.super_layer = super_layer

    def set_super_ps_layer(self) -> None:
        if self.super_layer is not None:
            self.super_ps_layers = self.super_layer.build_ps_layers(
                self.grid_dim_x, self.grid_dim_y, self.miniblock[:-2], w_bit=self.w_bit
            )

    def reset_parameters(
        self,
    ) -> None:
        ## only reset Sigma and Phases, and bias
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

        for m in self.super_ps_layers:
            m.reset_parameters()

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
        if self.mode == "phase":
            ## add a transform called "build_weight" for parameter named "weight"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )

        ## add output transform
        self.add_transform(
            "output", "output", {"output_transform": self._output_transform}
        )

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.phase_noise_std = noise_std
        for m in self.super_ps_layers:
            m.set_phase_noise(noise_std)

    def set_gamma_noise(
        self, noise_std: float = 0, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for m in self.super_ps_layers:
            m.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        self.output_quantizer.set_bit(out_bit)

    def _input_transform(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        return x

    def _weight_transform(
        self, weights: Dict, update_list: set = {"phase", "S"}
    ) -> Tensor:
        if self.mode == "phase":
            super_ps_layers, S = weights
            U, V = self.super_layer.get_UV(
                super_ps_layers,
                grid_dim_x=S.size(1),
                grid_dim_y=S.size(0),
                miniblock=self.miniblock[:-2],
            )
            print(U.shape, S.shape, V.shape)
            # [p,q,k,k] x [p,q,k,k] -> [p,q,k,k]
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            weight = U.matmul(S.unsqueeze(-1).mul(V))
            self.weight.data.copy_(weight)
        elif self.mode == "usv":
            _, S = weights
            U, V = self.U, self.V
            weight = U.matmul(S.unsqueeze(-1).mul(V))
            self.weight.data.copy_(weight)
        elif self.mode == "weight":
            weight = self.weight
        else:
            raise NotImplementedError

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
