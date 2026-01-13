import sys
import unittest

import numpy as np
import torch
from pyutils.general import TimerCtx, logger
from torch import nn
from torchvision.models import resnet18

# import torchonn as onn
# from torchonn.models.base_model import ONNBaseModel
sys.path.append("..")
from torchonn.models.base_model import ONNBaseModel
from torchonn.models.mobilevit import MobileViT


class TestModels(unittest.TestCase):
    def test_model_converter(self):
        device = torch.device("cuda:0")

        model = MobileViT(
            image_size=(224, 224),
            mode="small",  # support ["xx_small", "x_small", "small"] as shown in paper
            num_classes=1000,
            patch_size=(2, 2),
        )

        onn_model = ONNBaseModel.from_model(
            model,
            map_cfgs=dict(
                Conv2d=dict(
                    type="TeMPOBlockConv2d", mode="weight", photodetect="incoherent"
                ),
                Linear=dict(
                    type="TeMPOBlockLinear", mode="weight", photodetect="incoherent"
                ),
                MatMulModule=dict(
                    type="TeMPOBlockMatMul", mode="weight", photodetect="incoherent"
                ),
            ),
            verbose=True,
        )
        x = torch.randn(5, 3, 224, 224)
        y = model(x)  # (5, 1000)
        y_onn = onn_model(x)
        print(y)
        print(y_onn)
        assert torch.allclose(y, y_onn, rtol=1e-4, atol=1e-4), print(
            "converted result max abs error:", y.sub(y_onn).abs().max().item()
        )

        # print(onn_model)


if __name__ == "__main__":
    unittest.main()
