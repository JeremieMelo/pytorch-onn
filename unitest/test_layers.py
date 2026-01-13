"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-10 03:39:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-10 03:39:01
"""

import unittest
from tabnanny import verbose

import numpy as np
import torch

from torchonn.layers import (  # AllPassMORRCirculantLinear,; AllPassMORRCirculantConv2d,; AddDropMRRConv2d,; MZIConv2d,; MZILinear,
    AddDropMRRBlockConv2d,
    AddDropMRRBlockLinear,
    FFTONNBlockConv2d,
    FFTONNBlockLinear,
    MZIBlockConv2d,
    MZIBlockLinear,
    SuperBlockConv2d,
    SuperBlockLinear,
    TeMPOBlockConv2d,
    TeMPOBlockLinear,
    super_layer_name_dict,
)


class TestLayers(unittest.TestCase):
    # def test_mzilinear(self):
    #     device = torch.device("cuda:0")
    #     layer = MZILinear(8, 8, bias=False, mode="usv", device=device).to(device)
    #     layer.reset_parameters()
    #     x = torch.randn(1, 8, device=device)
    #     weight = layer.build_weight().data.clone()
    #     y = layer(x).detach()
    #     layer.switch_mode_to("phase")
    #     layer.sync_parameters(src="usv")
    #     weight2 = layer.build_weight().data.clone()
    #     y2 = layer(x).detach()
    #     # print(weight)
    #     # print(weight2)
    #     # print(y)
    #     # print(y2)

    #     assert np.allclose(weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
    #         "weight max abs error:", np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max()
    #     )
    #     assert np.allclose(y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
    #         "result max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
    #     )

    #     # test layer conversion
    #     linear = torch.nn.Linear(8, 8, bias=True).to(device)
    #     layer = MZILinear.from_layer(linear, mode="phase", photodetect=False)
    #     y1 = linear(x).detach().cpu().numpy()
    #     y2 = layer(x).detach().cpu().numpy()
    #     # print(y1)
    #     # print(y2)

    #     assert np.allclose(y1, y2, rtol=1e-4, atol=1e-4), print(
    #         "converted result max abs error:", np.abs(y1 - y2).max()
    #     )

    # def test_mziconv2d(self):
    #     device = torch.device("cuda:0")
    #     layer = MZIConv2d(8, 8, 3, bias=False, mode="usv", device=device).to(device)
    #     layer.reset_parameters()
    #     x = torch.randn(1, 8, 4, 4, device=device)
    #     weight = layer.transform_weight(layer.weights)["weight"].data.clone()
    #     y = layer(x).detach()
    #     layer.switch_mode_to("phase")
    #     layer.sync_parameters(src="usv")
    #     weight2 = layer.transform_weight(layer.weights)["weight"].data.clone()
    #     y2 = layer(x).detach()
    #     # print(weight)
    #     # print(weight2)
    #     # print(y)
    #     # print(y2)
    #     # exit(0)

    #     assert np.allclose(weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
    #         "max abs error:", np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max()
    #     )
    #     assert np.allclose(y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4), print(
    #         "max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
    #     )

    #     # test layer conversion
    #     conv2d = torch.nn.Conv2d(8, 8, 3, stride=2, bias=True).to(device)
    #     layer = MZIConv2d.from_layer(conv2d, mode="phase", photodetect=False)
    #     y1 = conv2d(x).detach().cpu().numpy()
    #     y2 = layer(x).detach().cpu().numpy()
    #     # print(y1, y1.shape)
    #     # print(y2, y2.shape)
    #     # exit(0)

    #     assert np.allclose(y1, y2, rtol=1e-4, atol=1e-4), print("max abs error:", np.abs(y1 - y2).max())

    def test_mziblocklinear(self):
        device = torch.device("cuda:0")
        fc = MZIBlockLinear(
            8, 8, bias=False, miniblock=4, mode="usv", device=device
        ).to(device)
        fc.reset_parameters()
        x = torch.randn(1, 8, device=device)
        weight = fc.transform_weight(fc.weights)["weight"].data.clone()
        y = fc(x).detach()
        fc.switch_mode_to("phase")
        fc.sync_parameters(src="usv")
        weight2 = fc.transform_weight(fc.weights)["weight"].data.clone()
        y2 = fc(x).detach()
        # print(weight)
        # print(weight2)
        # print(y)
        # print(y2)

        assert np.allclose(
            weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4
        ), print(
            "weight max abs error:",
            np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max(),
        )
        assert np.allclose(
            y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4
        ), print(
            "result max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
        )

        # test layer conversion
        linear = torch.nn.Linear(8, 8, bias=True).to(device)
        layer = MZIBlockLinear.from_layer(linear, miniblock=4, mode="phase")
        y1 = linear(x).detach()
        y2 = layer(x).detach()
        # print(y1)
        # print(y2)

        assert np.allclose(
            y1, y2, rtol=1e-4, atol=1e-4
        ), f"converted result max abs error: {(y1 - y2).abs().max().item()}"

    def test_mziblockconv2d(self):
        device = torch.device("cuda:0")
        conv2d = MZIBlockConv2d(
            8, 8, 3, bias=False, miniblock=4, mode="usv", device=device
        ).to(device)
        conv2d.reset_parameters()
        x = torch.randn(1, 8, 4, 4, device=device)
        weight = conv2d.transform_weight(conv2d.weights)["weight"].data.clone()
        y = conv2d(x).detach()
        conv2d.switch_mode_to("phase")
        conv2d.sync_parameters(src="usv")
        weight2 = conv2d.transform_weight(conv2d.weights)["weight"].data.clone()
        y2 = conv2d(x).detach()
        # print(weight)
        # print(weight2)
        # print(y)
        # print(y2)

        assert np.allclose(
            weight.cpu().numpy(), weight2.cpu().numpy(), rtol=1e-4, atol=1e-4
        ), print(
            "weight max abs error:",
            np.abs(weight.cpu().numpy() - weight2.cpu().numpy()).max(),
        )
        assert np.allclose(
            y.cpu().numpy(), y2.cpu().numpy(), rtol=1e-4, atol=1e-4
        ), print(
            "result max abs error:", np.abs(y.cpu().numpy() - y2.cpu().numpy()).max()
        )

        # test layer conversion
        conv2d = torch.nn.Conv2d(8, 8, 3, stride=2, bias=True).to(device)
        layer = MZIBlockConv2d.from_layer(conv2d, miniblock=4, mode="phase")
        y1 = conv2d(x).detach()
        y2 = layer(x).detach()
        # print(y1)
        # print(y2)

        assert np.allclose(
            y1, y2, rtol=1e-4, atol=1e-4
        ), f"converted result max abs error: {(y1 - y2).abs().max().item()}"

    def test_fftonnblocklinear(self):
        device = torch.device("cuda:0")
        layer = FFTONNBlockLinear(
            8, 8, bias=False, miniblock=4, mode="fft", device=device
        ).to(device)
        layer.reset_parameters(mode="fft")
        layer.set_input_bitwidth(8)
        layer.set_weight_bitwidth(8)
        x = torch.randn(1, 8, device=device)
        weight = layer.transform_weight(layer.weights)["weight"].data.clone()
        y = layer(x).detach()
        print(weight)
        print(y)

    def test_fftonnblockconv2d(self):
        device = torch.device("cuda:0")
        layer = FFTONNBlockConv2d(
            8, 8, 3, bias=False, miniblock=4, mode="fft", device=device
        ).to(device)
        layer.reset_parameters(mode="fft")
        layer.set_input_bitwidth(8)
        layer.set_weight_bitwidth(8)
        x = torch.randn(1, 8, 4, 4, device=device)
        weight = layer.transform_weight(layer.weights)["weight"].data.clone()
        y = layer(x).detach()
        print(weight)
        print(y)

    def test_tempoblocklinear(self):
        device = torch.device("cuda:0")
        layer = TeMPOBlockLinear(
            8, 8, bias=False, miniblock=4, mode="weight", device=device
        ).to(device)
        layer.reset_parameters()
        layer.set_input_bitwidth(8)
        layer.set_weight_bitwidth(4)
        layer.set_output_bitwidth(8)
        layer.set_input_noise(0.01)
        layer.set_weight_noise(0.01)
        layer.set_output_noise(0.01)
        # layer.sync_parameters(src="weight", verbose=True)
        x = torch.randn(1, 8, device=device)
        weight = layer.transform_weight(layer.weights)["weight"].data.clone()
        y = layer(x).detach()
        print(weight)
        print(y)

        # test layer conversion
        linear = torch.nn.Linear(8, 8, bias=True).to(device)
        layer = TeMPOBlockLinear.from_layer(linear, miniblock=4, mode="weight", w_bit=8)
        y1 = linear(x)
        y2 = layer(x)
        print(y1)
        print(y2)

        assert torch.allclose(
            y1, y2, rtol=3e-2, atol=3e-2
        ), f"converted result max abs error:{(y1 - y2).abs().max().item()}"

    def test_tempoblockconv2d(self):
        device = torch.device("cuda:0")
        layer = TeMPOBlockConv2d(
            8, 8, 3, bias=False, miniblock=4, mode="weight", device=device
        ).to(device)
        layer.reset_parameters(mode="weight")
        layer.set_input_bitwidth(8)
        layer.set_weight_bitwidth(4)
        layer.set_output_bitwidth(8)
        layer.set_input_noise(0.01)
        layer.set_weight_noise(0.01)
        layer.set_output_noise(0.01)
        # layer.sync_parameters(src="weight", verbose=True)
        x = torch.randn(1, 8, 4, 4, device=device)
        weight = layer.transform_weight(layer.weights)["weight"].data.clone()
        y = layer(x).detach()
        print(weight)
        print(y)

        # test layer conversion
        conv2d = torch.nn.Conv2d(8, 8, 3, stride=2, bias=True).to(device)
        layer = TeMPOBlockConv2d.from_layer(conv2d, miniblock=4, mode="weight", w_bit=8)
        y1 = conv2d(x)
        y2 = layer(x)
        print(y1)
        print(y2)

        assert torch.allclose(
            y1, y2, rtol=3e-2, atol=3e-2
        ), f"converted result max abs error:{(y1 - y2).abs().max().item()}"

    # def test_allpassmorrcirculantlinear(self):
    #     device = torch.device("cuda:0")
    #     layer = AllPassMORRCirculantLinear(
    #         8,
    #         8,
    #         bias=True,
    #         miniblock=4,
    #         morr_init=True,
    #         trainable_morr_scale=True,
    #         trainable_morr_bias=True,
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters(morr_init=True)
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, device=device)
    #     weight = layer.build_weight()[0].data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_allpassmorrcirculantconv2d(self):
    #     device = torch.device("cuda:0")
    #     layer = AllPassMORRCirculantConv2d(
    #         8,
    #         8,
    #         3,
    #         bias=True,
    #         miniblock=4,
    #         morr_init=True,
    #         trainable_morr_scale=True,
    #         trainable_morr_bias=True,
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters(morr_init=True)
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, 4, 4, device=device)
    #     weight = layer.build_weight().data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_pcmconv2d(self):
    #     device = torch.device("cuda:0")
    #     layer = PCMConv2d(
    #         8,
    #         8,
    #         3,
    #         bias=True,
    #         block_size=8,
    #         mode="block",
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters()
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, 4, 4, device=device)
    #     weight = layer.build_weight().data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_pcmlinear(self):
    #     device = torch.device("cuda:0")
    #     layer = PCMLinear(
    #         8,
    #         8,
    #         bias=True,
    #         block_size=8,
    #         mode="block",
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters()
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, device=device)
    #     weight = layer.build_weight().data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_superblocklinear(self):
    #     device = torch.device("cuda:0")
    #     arch = dict(
    #         n_waveguides=4,
    #         n_blocks=4,
    #         n_layers_per_block=2,
    #         n_front_share_blocks=4,
    #         share_ps="row_col",
    #         interleave_dc=True,
    #         symmetry_cr=False,
    #         device_cost=dict(
    #             ps_weight=6.8,
    #             dc_weight=1.5,
    #             cr_weight=0.064,
    #             area_upper_bound=120,
    #             area_lower_bound=70,
    #             first_active_block=True,
    #         ),
    #     )

    #     super_layer = super_layer_name_dict["adept"](arch=arch, device=device)

    #     layer = SuperBlockLinear(
    #         8,
    #         8,
    #         bias=True,
    #         miniblock=4,
    #         super_layer=super_layer,
    #         mode="phase",
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters()
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     layer.sync_parameters(src="phase")
    #     layer.sync_parameters(src="weight")
    #     x = torch.randn(1, 8, device=device)
    #     super_layer.build_arch_mask("gumbel_soft")
    #     weight = layer.transform_weight(layer.weights)["weight"].data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_superblockconv2d(self):
    #     device = torch.device("cuda:0")
    #     # arch definition
    #     arch = dict(
    #         n_waveguides=4,
    #         n_blocks=4,
    #         n_layers_per_block=2,
    #         n_front_share_blocks=4,
    #         share_ps="row_col",
    #         interleave_dc=True,
    #         symmetry_cr=False,
    #         device_cost=dict(
    #             ps_weight=6.8,
    #             dc_weight=1.5,
    #             cr_weight=0.064,
    #             area_upper_bound=120,
    #             area_lower_bound=70,
    #             first_active_block=True,
    #         ),
    #     )

    #     # use the arch definition to create a super optical layer
    #     super_layer = super_layer_name_dict["adept"](arch=arch, device=device)

    #     # when creating the super conv2d, pass the super_layer to it
    #     layer = SuperBlockConv2d(
    #         8,
    #         8,
    #         3,
    #         bias=True,
    #         miniblock=4,
    #         super_layer=super_layer,
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters()
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, 4, 4, device=device)

    #     # explicitly build the architecture mask during each training iteration before forward
    #     super_layer.build_arch_mask("gumbel_soft")
    #     weight = layer.transform_weight(layer.weights)["weight"].data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_mrrconv2d(self):
    #     device = torch.device("cuda:0")
    #     layer = AddDropMRRConv2d(
    #         8,
    #         8,
    #         3,
    #         bias=True,
    #         mode="weight",
    #         device=device,
    #     ).to(device)
    #     layer.reset_parameters()
    #     layer.set_input_bitwidth(8)
    #     layer.set_weight_bitwidth(8)
    #     x = torch.randn(1, 8, 4, 4, device=device)
    #     weight = layer.build_weight().data.clone()
    #     y = layer(x).detach()
    #     print(weight)
    #     print(y)

    # def test_mrrblockconv2d(self):
    #     device = torch.device("cuda:0")
    #     conv2d = AddDropMRRBlockConv2d(
    #         8, 8, 3, bias=False, miniblock=4, mode="weight", device=device
    #     ).to(device)
    #     conv2d.reset_parameters()
    #     x = torch.randn(1, 8, 4, 4, device=device)
    #     weight = conv2d.transform_weight(conv2d.weights)["weight"].data.clone()
    #     y = conv2d(x).detach()
    #     conv2d.switch_mode_to("phase")
    #     conv2d.sync_parameters(src="weight")
    #     weight2 = conv2d.transform_weight(conv2d.weights)["weight"].data.clone()
    #     y2 = conv2d(x).detach()
    #     # print(weight)
    #     # print(weight2)
    #     # print(y)
    #     # print(y2)

    #     assert torch.allclose(weight, weight2, rtol=1e-4, atol=1e-4), print(
    #         "weight max abs error:", (weight - weight2).abs().max().item()
    #     )
    #     assert torch.allclose(y, y2, rtol=1e-3, atol=1e-3), print(
    #         "output max abs error:", (y - y2).abs().max().item()
    #     )

    #     # test layer conversion
    #     conv2d = torch.nn.Conv2d(8, 8, 3, stride=2, bias=True).to(device)
    #     layer = AddDropMRRBlockConv2d.from_layer(conv2d, miniblock=4, mode="phase")
    #     y1 = conv2d(x).detach()
    #     y2 = layer(x).detach()
    #     # print(y1)
    #     # print(y2)

    #     assert torch.allclose(y1, y2, rtol=1e-3, atol=1e-3), print(
    #         "converted result max abs error:", (y1 - y2).abs().max().item()
    #     )

    #     layer.set_weight_bitwidth(8)
    #     layer(x).sum().backward()
    #     print(layer.phase.grad.abs().mean())
    #     print(layer.S_scale.grad.abs().mean())

    # def test_mrrblocklinear(self):
    #     device = torch.device("cuda:0")
    #     linear = AddDropMRRBlockLinear(
    #         8, 8, bias=False, miniblock=4, mode="weight", device=device
    #     ).to(device)
    #     linear.reset_parameters()
    #     x = torch.randn(1, 8, device=device)
    #     weight = linear.transform_weight(linear.weights)["weight"].data.clone()
    #     y = linear(x).detach()
    #     linear.switch_mode_to("phase")
    #     linear.sync_parameters(src="weight")
    #     weight2 = linear.transform_weight(linear.weights)["weight"].data.clone()
    #     y2 = linear(x).detach()
    #     print(weight)
    #     print(weight2)
    #     # print(y)
    #     # print(y2)

    #     assert torch.allclose(weight, weight2, rtol=1e-4, atol=1e-4), print(
    #         "weight max abs error:", (weight - weight2).abs().max().item()
    #     )
    #     assert torch.allclose(y, y2, rtol=1e-3, atol=1e-3), print(
    #         "output max abs error:", (y - y2).abs().max().item()
    #     )

    #     # test layer conversion
    #     linear = torch.nn.Linear(8, 8, bias=True).to(device)
    #     layer = AddDropMRRBlockLinear.from_layer(linear, miniblock=4, mode="phase")
    #     y1 = linear(x).detach()
    #     y2 = layer(x).detach()
    #     # print(y1)
    #     # print(y2)

    #     assert torch.allclose(y1, y2, rtol=1e-3, atol=1e-3), print(
    #         "converted result max abs error:", (y1 - y2).abs().max().item()
    #     )

    #     layer.set_weight_bitwidth(8)
    #     layer(x).sum().backward()
    #     print(layer.phase.grad.abs().mean())
    #     print(layer.S_scale.grad.abs().mean())


if __name__ == "__main__":
    unittest.main()
