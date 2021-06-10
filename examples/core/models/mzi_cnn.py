"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-07 03:43:40
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-07 03:43:40
"""

from torchonn.op.mzi_op import project_matrix_to_unitary
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.types import Device, _size
from torchonn.layers import MZIBlockConv2d, MZIBlockLinear
from torchonn.models import ONNBaseModel
from collections import OrderedDict

__all__ = ["MZI_CLASS_CNN"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = False,
        miniblock: int = 8,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.conv = MZIBlockConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            miniblock=miniblock,
            mode=mode,
            decompose_alg=decompose_alg,
            photodetect=photodetect,
            device=device,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        miniblock: int = 8,
        mode: str = "weight",
        decompose_alg: str = "clements",
        photodetect: bool = False,
        activation: bool = True,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.linear = MZIBlockLinear(
            in_features,
            out_features,
            bias=bias,
            miniblock=miniblock,
            mode=mode,
            decompose_alg=decompose_alg,
            photodetect=photodetect,
            device=device,
        )

        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MZI_CLASS_CNN(ONNBaseModel):
    """
    MZI CNN for classification.
    Blocking matrix multiplication, which is much faster and more scalable than implementing the entire weight matrix on an MZI array.
    Each block is implemented by a square MZI array

    """

    _conv_linear = (MZIBlockConv2d, MZIBlockLinear)
    _conv = (MZIBlockConv2d,)
    _linear = (MZIBlockLinear,)

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        kernel_list: List[int] = [32],
        kernel_size_list: List[int] = [3],
        stride_list: List[int] = [1],
        padding_list: List[int] = [1],
        dilation_list: List[int] = [1],
        pool_out_size: int = 5,
        hidden_list: List[int] = [32],
        block_list: List[int] = [8],
        mode: str = "usv",
        decompose_alg: str = "clements",
        photodetect: bool = True,
        bias: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list

        self.pool_out_size = pool_out_size

        self.hidden_list = hidden_list
        self.block_list = block_list
        self.mode = mode
        self.decompose_alg = decompose_alg

        self.photodetect = photodetect
        self.bias = bias

        self.device = device

        self.build_layers()

        self.reset_parameters()

    def build_layers(self):
        self.features = OrderedDict()
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channels = self.in_channels if (idx == 0) else self.kernel_list[idx - 1]

            self.features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                dilation=self.dilation_list[idx],
                groups=1,
                bias=self.bias,
                miniblock=self.block_list[idx],
                mode=self.mode,
                decompose_alg=self.decompose_alg,
                photodetect=self.photodetect,
                device=self.device,
            )
        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, self._conv):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                bias=self.bias,
                miniblock=self.block_list[idx + len(self.kernel_list)],
                mode=self.mode,
                decompose_alg=self.decompose_alg,
                photodetect=self.photodetect,
                activation=True,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = LinearBlock(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.num_classes,
            bias=self.bias,
            miniblock=self.block_list[-1],
            mode=self.mode,
            decompose_alg=self.decompose_alg,
            photodetect=self.photodetect,
            activation=False,
            device=self.device,
        )
        self.classifier = nn.Sequential(self.classifier)

    def unitary_projection(self) -> None:
        assert self.mode == "usv", "Unitary projection can only be applied in usv mode"
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.U.data.copy_(project_matrix_to_unitary(m.U.data))
                m.V.data.copy_(project_matrix_to_unitary(m.V.data))

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
