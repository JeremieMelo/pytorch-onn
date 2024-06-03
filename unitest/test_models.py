"""
Date: 2024-06-02 21:25:07
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-02 21:27:31
FilePath: /pytorch-onn/unitest/test_models.py
"""

import unittest

import numpy as np
import torch
from pyutils.general import TimerCtx, logger
from torchvision.models import resnet18

import torchonn as onn
from torchonn.models.base_model import ONNBaseModel


class CNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((5, 5))
        self.linear = torch.nn.Linear(200, 10, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x



class TestModels(unittest.TestCase):
    def test_model_converter(self):
        device = torch.device("cuda:0")
        # model = resnet18().to(device)
        model = CNN().to(device)
        onn_model = ONNBaseModel.from_model(
            model,
            conv_cfg=dict(type="MZIBlockConv2d", mode="phase", photodetect=False),
            linear_cfg=dict(type="MZIBlockLinear", mode="usv", photodetect=False),
        )
        x = torch.randn(1, 3, 8, 8, device=device)
        y = model(x)
        y_onn = onn_model(x)
        print(y)
        print(y_onn)
        assert torch.allclose(y, y_onn, rtol=1e-4, atol=1e-4), print(
            "converted result max abs error:", y.sub(y_onn).abs().max().item()
        )

        print(onn_model)


if __name__ == "__main__":
    unittest.main()
