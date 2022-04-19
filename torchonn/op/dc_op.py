"""
Description: ops for directional couplers
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-04-18 21:22:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-04-18 21:29:47
"""
import torch
from pyutils.general import logger
from pyutils.quantize import uniform_quantize

__all__ = ["dc_quantize_fn"]

class dc_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, quant_ratio=1.0):
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
        """
        super().__init__()
        assert 1 <= w_bit <= 32, logger.error(f"Only support 1 - 32 bit quantization, but got {w_bit}")
        self.w_bit = w_bit

        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.uniform_q = uniform_quantize(k=w_bit, gradient_clip=True)

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.quant_ratio = quant_ratio

    def set_bitwidth(self, bit: int) -> None:
        if bit != self.w_bit:
            self.w_bit = bit
            self.uniform_q = uniform_quantize(k=bit, gradient_clip=True)

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.quant_ratio)
        else:
            quant_noise_mask = None

        if self.w_bit == 32:
            weight_q = torch.tanh(x)
            weight_q = weight_q / torch.max(torch.abs(weight_q))
        elif self.w_bit == 1:
            weight_q = self.uniform_q(x).add(1).mul((1 - 2 ** 0.5 / 2) / 2).add(2 ** 0.5 / 2)  # [0.717, 1] corresponds to 50:50 DC and identity waveguide connection, respectively
            if quant_noise_mask is not None:
                x = x.add((2 + 2 ** 0.5) / 4)  # mean is (0.717+1)/2
                noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                weight_q = x + noise

        else:
            weight = torch.tanh(x)  # [-1, 1]
            r = torch.max(torch.abs(weight.data))
            weight = (weight + r) / (2 * r)  # [0 ~ 1]
            # weight = weight / 2 + 0.5
            weight_q = self.uniform_q(weight)  # [0 ~ 1]
            if quant_noise_mask is not None:
                noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                weight_q = weight + noise

        return weight_q
