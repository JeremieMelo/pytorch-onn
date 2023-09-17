"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import add_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device

from torchonn.devices.mrr import MRRConfig_5um_HQ
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_func, mrr_tr_to_roundtrip_phase

__all__ = [
    "AddDropMRRLinear",
    "AddDropMRRBlockLinear",
]


class AddDropMRRLinear(ONNBaseLayer):
    """
    Linear layer constructed by cascaded AddDropMRRs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_HQ,
        device: Device = torch.device("cpu"),
    ):
        super(AddDropMRRLinear, self).__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        assert mode in {"weight", "phase"}, logger.error(
            f"Mode not supported. Expected one from (weight, phase) but got {mode}."
        )
        self.MRRConfig = MRRConfig
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32

        ### build trainable parameters
        self.build_parameters(mode)
        ### quantization tool
        self.input_quantizer = input_quantize_fn(
            self.in_bit, alg="dorefa", device=self.device
        )
        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")
        self.phase_quantizer = weight_quantize_fn(self.w_bit, alg="qnn")
        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            self.MRRConfig.attenuation_factor, self.MRRConfig.coupling_factor
        )
        self.mrr_tr_to_roundtrip_phase = mrr_tr_to_roundtrip_phase
        self.mrr_weight_to_tr = lambda x: (x + 1) / 2
        self.mrr_tr_to_weight = lambda x: 2 * x - 1

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
        phase = torch.empty(
            self.out_features,
            self.in_features,
            device=self.device,
        )
        weight = torch.empty(
            self.out_features,
            self.in_features,
            device=self.device,
        )
        # TIA gain
        S_scale = torch.ones(1, device=self.device, dtype=torch.float32)

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "phase":
            self.phase = Parameter(phase)
            self.S_scale = Parameter(S_scale)
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "phase": phase,
            "S_scale": S_scale,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode in {"weight"}:
            init.kaiming_normal_(self.weight.data)
        elif self.mode in {"phase"}:
            init.kaiming_normal_(self.weight.data)
            scale = self.weight.data.abs().max()
            self.S_scale.data.fill_(scale)
            self.phase.data.copy_(
                self.mrr_tr_to_roundtrip_phase(
                    self.mrr_weight_to_tr(self.weight.data.div(scale)),
                    self.MRRConfig.attenuation_factor,
                    self.MRRConfig.coupling_factor,
                )
                % (2 * np.pi)
            )
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_HQ,
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted AddDropMRRLinear module
        """
        assert isinstance(
            layer, nn.Linear
        ), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        ).to(device)
        instance.weight.data.copy_(layer.weight)
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def build_weight_from_phase(self, phases: Tensor) -> Tensor:
        self.weight.data.copy_(
            self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phases)).mul(
                self.S_scale
            )
        )
        return self.weight

    def build_weight_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tensor:
        return self.build_weight_from_phase(
            *self.build_phase_from_voltage(voltage, S_scale)
        )

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        self.S_scale.data.fill_(self.weight.data.abs().max())
        self.phase.data.copy_(
            self.mrr_tr_to_roundtrip_phase(
                self.mrr_weight_to_tr(weight.data.div(self.S_scale.data))
            )
        )
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
            self.build_phase_from_weight(self.weight)
        elif src == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase.data)
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
                phase = self.phase_quantizer(self.phase)
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
        self.phase_quantizer.set_bitwidth(w_bit)
        self.weight_quantizer.set_bitwidth(w_bit)

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
            weight = self.build_weight()  # [out_features, in_features]
        else:
            weight = self.weight

        x = F.linear(
            x,
            weight,
            bias=self.bias,
        )

        return x


class AddDropMRRBlockLinear(ONNBaseLayer):
    """
    blocking Linear layer constructed by cascaded AddDropMRRs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: int
    weight: Tensor
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        miniblock: int = 4,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_HQ,
        device: Device = torch.device("cpu"),
    ):
        super(AddDropMRRBlockLinear, self).__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        assert mode in {"weight", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."
        )
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock))
        self.in_features_pad = self.grid_dim_x * miniblock
        self.out_features_pad = self.grid_dim_y * miniblock

        self.MRRConfig = MRRConfig
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32

        ### build trainable parameters
        self.build_parameters(mode)
        ### quantization tool
        self.input_quantizer = input_quantize_fn(
            self.in_bit, alg="dorefa", device=self.device
        )
        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")
        self.phase_quantizer = weight_quantize_fn(self.w_bit, alg="qnn")
        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            self.MRRConfig.attenuation_factor, self.MRRConfig.coupling_factor
        )
        self.mrr_tr_to_roundtrip_phase = mrr_tr_to_roundtrip_phase
        self.mrr_weight_to_tr = lambda x: (x + 1) / 2
        self.mrr_tr_to_weight = lambda x: 2 * x - 1

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
        phase = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            self.miniblock,
            self.miniblock,
            device=self.device,
        )
        weight = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            self.miniblock,
            self.miniblock,
            device=self.device,
        )
        # TIA gain
        S_scale = torch.ones(
            self.grid_dim_y, self.grid_dim_x, 1, device=self.device, dtype=torch.float32
        )

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "phase":
            self.phase = Parameter(phase)
            self.S_scale = Parameter(S_scale)
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "phase": phase,
            "S_scale": S_scale,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode in {"weight"}:
            init.kaiming_normal_(self.weight.data)
        elif self.mode in {"phase"}:
            init.kaiming_normal_(self.weight.data)
            scale = self.weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
            self.S_scale.data.copy_(scale)
            self.phase.data.copy_(
                self.mrr_tr_to_roundtrip_phase(
                    self.mrr_weight_to_tr(self.weight.data.div(scale[..., None])),
                    self.MRRConfig.attenuation_factor,
                    self.MRRConfig.coupling_factor,
                )
                % (2 * np.pi)
            )
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_HQ,
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted AddDropMRRLinear module
        """
        assert isinstance(
            layer, nn.Linear
        ), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        ).to(device)
        instance.weight.data.copy_(layer.weight)
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def build_weight_from_phase(self, phases: Tensor) -> Tensor:
        self.weight.data.copy_(
            self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phases)).mul(
                self.S_scale[..., None]
            )
        )
        return self.weight

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
            self.weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)
        )
        self.phase.data.copy_(
            self.mrr_tr_to_roundtrip_phase(
                self.mrr_weight_to_tr(weight.data.div(self.S_scale.data[..., None]))
            )
        )
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
            self.build_phase_from_weight(self.weight)
        elif src == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase.data)
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
                phase = self.phase_quantizer(self.phase)
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
        self.phase_quantizer.set_bitwidth(w_bit)
        self.weight_quantizer.set_bitwidth(w_bit)

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
        x = F.linear(
            x,
            weight,
            bias=self.bias,
        )

        return x
