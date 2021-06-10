"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 18:55:05
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 18:55:05
"""
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.types import Device

__all__ = ["ONNBaseLayer"]


class ONNBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    def build_parameters(self) -> None:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.phase_noise_std = noise_std

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {param_name: param_tensor, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
