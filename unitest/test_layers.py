"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-10 03:39:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-10 03:39:01
"""
import unittest
from pyutils.general import TimerCtx
import torch
import numpy as np
import torchonn as onn
from pyutils.general import logger
from torchonn.layers import MZIBlockConv2d, MZIBlockLinear, FFTONNBlockLinear, FFTONNBlockConv2d


class TestLayers(unittest.TestCase):
    def test_mziblocklinear(self):
        device = torch.device("cuda:0")
        fc = MZIBlockLinear(8, 8, bias=False, miniblock=4, mode="usv", device=device).to(device)
        fc.reset_parameters()
        x = torch.randn(1, 8, device=device)
        weight = fc.build_weight().data.clone()
        y = fc(x).detach()
        fc.switch_mode_to("phase")
        fc.sync_parameters(src="usv")
        weight2 = fc.build_weight().data.clone()
        y2 = fc(x).detach()
        # print(weight)
        # print(weight2)
        # print(y)
        # print(y2)

        assert np.allclose(weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
            "max abs error:", np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max()
        )
        assert np.allclose(y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
            "max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
        )

    def test_mziblockconv2d(self):
        device = torch.device("cuda:0")
        fc = MZIBlockConv2d(8, 8, 3, bias=False, miniblock=4, mode="usv", device=device).to(device)
        fc.reset_parameters()
        x = torch.randn(1, 8, 4, 4, device=device)
        weight = fc.build_weight().data.clone()
        y = fc(x).detach()
        fc.switch_mode_to("phase")
        fc.sync_parameters(src="usv")
        weight2 = fc.build_weight().data.clone()
        y2 = fc(x).detach()
        # print(weight)
        # print(weight2)
        # print(y)
        # print(y2)

        assert np.allclose(weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
            "max abs error:", np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max()
        )
        assert np.allclose(y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
            "max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
        )

    def test_fftonnblocklinear(self):
        device = torch.device("cuda:0")
        fc = FFTONNBlockLinear(8, 8, bias=False, miniblock=4, mode="fft", device=device).to(device)
        fc.reset_parameters(mode="fft")
        x = torch.randn(1, 8, device=device)
        weight = fc.build_weight().data.clone()
        y = fc(x).detach()
        print(weight)
        print(y)

    def test_fftonnblockconv2d(self):
        device = torch.device("cuda:0")
        fc = FFTONNBlockConv2d(8, 8, 3, bias=False, miniblock=4, mode="fft", device=device).to(device)
        fc.reset_parameters(mode="fft")
        x = torch.randn(1, 8, 4, 4, device=device)
        weight = fc.build_weight().data.clone()
        y = fc(x).detach()
        print(weight)
        print(y)


if __name__ == "__main__":
    unittest.main()
