"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:18:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:18:01
"""
from typing import Any, Dict, Optional, Callable
from torch import nn, Tensor
from torch.types import Device
from pyutils.torch_train import set_torch_deterministic

__all__ = ["ONNBaseModel"]


class ONNBaseModel(nn.Module):
    _conv_linear = tuple()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_phase_variation(self, noise_std: float = 0.0, random_state: Optional[int] = None) -> None:
        self.phase_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_phase_variation(noise_std, random_state=random_state)

    def set_gamma_noise(self, noise_std: float = 0.0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float = 0.0) -> None:
        self.crosstalk_factor = crosstalk_factor
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_input_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, m in self.named_modules():
            if name in param_dict:
                m.load_parameters(param_dict[name])

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.sync_parameters(src=src)

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.switch_mode_to(mode)

    def get_num_parameters(self) -> int:
        return sum(p for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        raise NotImplementedError
