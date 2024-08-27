"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.compute import add_gaussian_noise
from pyutils.general import logger
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import _size

from torchonn.devices.mrr import MRRConfig_5um_HQ
from torchonn.layers.base_layer import ONNBaseConv2d

from .utils import merge_chunks, partition_chunks

__all__ = [
    "TeMPOBlockConv2d",
]


@MODELS.register_module()
class TeMPOBlockConv2d(ONNBaseConv2d):
    """
    TeMPO, Zhang, Yin et al., JAP 2024
    blocking Conv2d layer constructed by time-multiplexed dynamic photonic tensor core.
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
        MRRConfig=MRRConfig_5um_HQ,
        w_bit=32,
        in_bit=32,
        out_bit=32,
        v_max=10.8,
        v_pi=4.36,
        photodetect="incoherent",
        device=torch.device("cpu"),
    )
    __mode_list__ = ["weight"]

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
        self.build_parameters(mode=self.mode)
        ### build transform
        self.build_transform()

        ### default set to slow forward
        self.disable_fast_forward()

        self.set_weight_noise(0)
        self.set_input_noise(0)
        self.set_output_noise(0)
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

        assert self.photodetect in ["incoherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        self.input_scale = None
        self.weight_scale = None

        self.register_parameter_buffer(
            *self.get_param_buffer_groups(mode=mode),
        )

        self.pack_weights()

    def get_param_buffer_groups(self, mode: str) -> Tensor:
        buffer_groups = {}
        if mode == "weight":
            param_groups = {"weight": Parameter(self.weight)}
        else:
            raise NotImplementedError

        return param_groups, buffer_groups

    def pack_weights(self):
        ## key is self.mode, which should match the src_name for weight_transform
        if self.mode == "weight":
            self.weights = {"weight": self.weight}
        else:
            raise NotImplementedError

    def reset_parameters(self, mode: Optional[str] = None) -> None:
        mode = mode or self.mode
        W = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            groups=self.groups,
        ).weight.data.to(self.device)
        W = partition_chunks(W.flatten(1), self.weight.shape)

        if self.mode == "weight":
            self.weight.data.copy_(W)
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

        self.weight_quantizer = WeightQuantizer_LSQ(
            out_features=self.grid_dim_y,
            device=self.device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
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

        self.quantizer_dict = {
            "input": self.input_quantizer,
            "weight": self.weight_quantizer,
            "output": self.output_quantizer,
        }

        ## add input transform
        self.add_transform("input", "input", {"input_transform": self._input_transform})

        ## add weight transform
        if self.mode == "weight":
            ## add a transform called "build_weight" for parameter named "weight"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )

        ## add output transform
        self.add_transform(
            "output", "output", {"output_transform": self._output_transform}
        )

    def sync_parameters(
        self, src: str = "weight", steps: int = 1000, verbose: bool = False
    ) -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            param_list = [p for p in self.parameters() if p.requires_grad]
            target = merge_chunks(self.weight.data.clone())[
                : self.out_channels, : self.in_channels_flat
            ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

            def build_weight_fn():
                return self.transform_weight(self.weights)["weight"]

            self.map_layer(
                target=target,
                param_list=param_list,
                build_weight_fn=build_weight_fn,
                mode="regression",
                num_steps=steps,
                verbose=verbose,
            )

        else:
            raise NotImplementedError

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.weight_quantizer.set_bit(w_bit)

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
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.in_bit < 16:
            x = self.input_quantizer(x)  # [-alpha, alpha] or [0, alpha]
            alpha = self.input_quantizer.alpha.data
        else:
            alpha = x.data.abs().max()
        self.input_scale = alpha.item()

        if self.input_noise_std > 0:
            x = add_gaussian_noise(x, noise_std=self.input_noise_std * self.input_scale)

        return x

    def _weight_transform(
        self, weights: Dict, update_list: set = {"S_scale"}
    ) -> Tensor:
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.mode == "weight":
            weight = weights
            if self.w_bit < 16:
                weight = self.weight_quantizer(weight)  # [-alpha, alpha]
                alpha = self.weight_quantizer.alpha.data
            else:
                alpha = weight.data.abs().max()
            self.weight_scale = alpha.item()

            if self.weight_noise_std > 0:
                ## Warning: noise need to be added to normalized input
                weight = add_gaussian_noise(
                    weight, noise_std=self.weight_noise_std * self.weight_scale
                )

        else:
            raise NotImplementedError

        weight = merge_chunks(weight)[
            : self.out_channels, : self.in_channels_flat
        ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        return weight  # this is normalized weights [-1, 1]

    def _output_transform(self, x: Tensor) -> Tensor:
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.output_noise_std > 0:
            x = add_gaussian_noise(
                x,
                noise_std=self.output_noise_std
                * self.input_scale
                * self.weight_scale
                * self.in_channels_flat**0.5,
            )

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
