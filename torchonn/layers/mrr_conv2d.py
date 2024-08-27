"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise
from pyutils.general import logger
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import _size

from torchonn.devices.mrr import MRRConfig_5um_HQ
from torchonn.layers.base_layer import ONNBaseConv2d
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_func, mrr_tr_to_roundtrip_phase

from .utils import merge_chunks, partition_chunks

__all__ = [
    "AddDropMRRBlockConv2d",
]


@MODELS.register_module()
class AddDropMRRBlockConv2d(ONNBaseConv2d):
    """
    blocking Conv2d layer constructed by AddDropMRR weight banks.
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
    __mode_list__ = ["weight", "phase", "voltage"]

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

        assert self.photodetect in ["incoherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        self.phase = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        # TIA gain
        self.S_scale = torch.ones(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock[:-2],
            1,
            device=self.device,
            dtype=torch.float32,
        )

        self.register_parameter_buffer(
            *self.get_param_buffer_groups(mode=mode),
        )

        self.pack_weights()

        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            self.MRRConfig.attenuation_factor,
            self.MRRConfig.coupling_factor,
            intensity=True,
        )
        self.mrr_tr_to_roundtrip_phase = mrr_tr_to_roundtrip_phase
        self.mrr_weight_to_tr = lambda x: (x + 1) / 2
        self.mrr_tr_to_weight = lambda x: 2 * x - 1

        ## Warning: the transmission of MRR is not from [0, 1], it has insertion loss and limited extinction ratio
        ## min is larger than 0, max is smaller than 1.
        ## If converting a transmission=1 or 0 to phase, it theoretically has no solution, it will be clipped to nearest solution which introduces error.
        ## This can be avoided by considering this into the scaling factor.
        ## If the MRR can only represent weight in [-0.98, 0.92], we will scale the weight to a symmetric range [-0.92, 0.92].
        ## this ultimately limits the dynamic range and induce laser power penalty by a factor of 1/0.92.
        trs = self.mrr_roundtrip_phase_to_tr(
            torch.linspace(-2 * np.pi, 2 * np.pi, 1000)
        )
        mrr_tr_min = trs.min()  # e.g., 0.01
        mrr_tr_max = trs.max()  # e.g., 0.96
        self.weight_scale = min(
            abs(self.mrr_tr_to_weight(mrr_tr_min)),
            abs(self.mrr_tr_to_weight(mrr_tr_max)),
        )  # [-0.98, 0.92] -> 0.92

    def get_param_buffer_groups(self, mode: str) -> Tensor:
        buffer_groups = {
            "phase": self.phase,
            "weight": self.weight,
            "S_scale": self.S_scale,
        }
        if mode == "weight":
            param_groups = {"weight": Parameter(self.weight)}
        elif mode == "phase":
            param_groups = {
                "phase": Parameter(self.phase),
                "S_scale": Parameter(self.S_scale),
            }
        else:
            raise NotImplementedError

        for name in param_groups:
            del buffer_groups[name]

        return param_groups, buffer_groups

    def pack_weights(self):
        ## key is self.mode, which should match the src_name for weight_transform
        if self.mode == "weight":
            self.weights = {"weight": self.weight}
        elif self.mode == "phase":
            self.weights = {self.mode: (self.phase, self.S_scale)}
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
        elif self.mode in {"phase"}:
            scale = (
                W.data.abs().flatten(-2, -1).amax(dim=-1, keepdim=True)
                / self.weight_scale
            )
            self.S_scale.data.copy_(scale)
            self.phase.data.copy_(
                self.mrr_tr_to_roundtrip_phase(
                    self.mrr_weight_to_tr(W.data.div(scale[..., None])),
                    self.MRRConfig.attenuation_factor,
                    self.MRRConfig.coupling_factor,
                )[0]
                % (2 * np.pi)
            )
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

        self.phase_quantizer = WeightQuantizer_LSQ(
            out_features=self.grid_dim_y,
            device=self.device,
            nbits=self.w_bit,
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

        self.quantizer_dict = {
            "input": self.input_quantizer,
            "phase": self.phase_quantizer,
            "S_scale": None,
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
        elif self.mode == "phase":
            ## add a transform called "build_weight" for parameter group ("phase", "S_scale") called "usv"
            self.add_transform(
                self.mode, "weight", {"build_weight": self._weight_transform}
            )

        ## add output transform
        self.add_transform(
            "output", "output", {"output_transform": self._output_transform}
        )

    def build_weight_from_phase(self, phases: Tensor, S_scale: Tensor) -> Tensor:
        weight = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phases)).mul(
            S_scale[..., None]
        )
        self.weight.data.copy_(weight)
        return weight

    def build_weight_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tensor:
        return self.build_weight_from_phase(
            *self.build_phase_from_voltage(voltage, S_scale)
        )

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        self.S_scale.data.copy_(
            weight.data.abs().flatten(-2, -1).amax(dim=-1, keepdim=True)
            / self.weight_scale
        )

        self.phase.data.copy_(
            self.mrr_tr_to_roundtrip_phase(
                self.mrr_weight_to_tr(weight.data.div(self.S_scale.data[..., None])),
                self.MRRConfig.attenuation_factor,
                self.MRRConfig.coupling_factor,
            )[0]
        )
        weight = self.build_weight_from_phase(self.phase, self.S_scale)

        return self.phase, self.S_scale

    def build_voltage_from_phase(
        self,
        phase: Tensor,
        S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def build_phase_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_weight(weight))

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            if self.w_bit < 16:
                weight = self.weight_quantizer(self.weight.data)
            else:
                weight = self.weight
            self.build_phase_from_weight(weight)
        elif src == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase.data % (2 * np.pi))
            else:
                phase = self.phase
            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase = add_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            self.build_weight_from_phase(
                phase,
                self.S_scale,
            )
        elif src == "voltage":
            NotImplementedError
        else:
            raise NotImplementedError

    def build_weight(self) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase % (2 * np.pi))
            else:
                phase = self.phase

            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase = add_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(phase, self.S_scale)
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return weight

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_quantizer.set_bit(w_bit)
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
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def _input_transform(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        return x

    def _weight_transform(
        self, weights: Dict, update_list: set = {"phase", "S_scale"}
    ) -> Tensor:
        if self.mode == "weight":
            weight = weights
            if self.w_bit < 16:
                weight = self.weight_quantizer(weight)
        elif self.mode == "phase":
            phase, S_scale = weights
            if (
                self.w_bit < 16
                or self.gamma_noise_std > 1e-5
                or self.crosstalk_factor > 1e-5
            ):
                phase = self.phase_quantizer(phase % (2 * np.pi))

            if self.phase_noise_std > 1e-5:
                phase = phase + gen_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(
                phase,
                S_scale,
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
