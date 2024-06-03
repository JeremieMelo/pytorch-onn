"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:18:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:18:01
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from mmengine.registry import MODELS
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.types import Device, _size

from torchonn.layers.base_layer import (
    build_activation_layer,
    build_conv_layer,
    build_linear_layer,
    build_norm_layer,
)

__all__ = [
    "ONNBaseModel",
    "LinearBlock",
    "ConvBlock",
]


class ONNBaseModel(nn.Module):
    def __init__(
        self,
        *args,
        conv_cfg=dict(type="TeMPOBlockConv2d"),
        linear_cfg=dict(type="TeMPOBlockLinear"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = (registry.get(conv_cfg["type"]),)
            self._linear = (registry.get(linear_cfg["type"]),)
            self._conv_linear = self._conv + self._linear

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

    def set_phase_variation(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
        self.phase_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_phase_variation(noise_std, random_state=random_state)

    def set_gamma_noise(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
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
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_bitwidth"
            ):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_bitwidth"
            ):
                layer.set_input_bitwidth(in_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_bitwidth"
            ):
                layer.set_output_bitwidth(out_bit)

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

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        conv_cfg=dict(type="MZIBlockConv2d"),
        linear_cfg=dict(type="MZIBlockLinear"),
        verbose: bool = False,
    ):
        new_model = deepcopy(model)
        device = next(new_model.parameters()).device
        conv_type = conv_cfg["type"]
        linear_type = linear_cfg["type"]
        conv_cfg.pop("type")
        linear_cfg.pop("type")
        conv_cfg = conv_cfg.copy()

        def replace_module(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    with MODELS.switch_scope_and_registry(None) as registry:
                        conv_layer = registry.get(conv_type)
                        if conv_layer is not None:
                            setattr(
                                module,
                                name,
                                conv_layer.from_layer(child, **conv_cfg).to(device),
                            )
                            if verbose:
                                logger.info(f"replaced {name} with {conv_type}")
                elif isinstance(child, nn.Linear):
                    with MODELS.switch_scope_and_registry(None) as registry:
                        linear_layer = registry.get(linear_type)
                        if linear_layer is not None:
                            setattr(
                                module,
                                name,
                                linear_layer.from_layer(child, **linear_cfg).to(device),
                            )
                            if verbose:
                                logger.info(f"replaced {name} with {linear_type}")
                else:
                    replace_module(child)

        replace_module(new_model)
        return new_model

    def forward(self, x):
        raise NotImplementedError


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="Linear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_features)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict = dict(type="Conv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
