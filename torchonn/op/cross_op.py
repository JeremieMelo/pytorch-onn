'''
Description: ops for waveguide crossings
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-04-18 21:43:06
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-04-18 21:43:06
'''

import torch
from torch import Tensor

__all__ = ["clip_grad_value_", "diff_round", "hard_diff_round"]


def clip_grad_value_(parameters, clip_value: float):
    for p in parameters:
        if p.grad is not None:
            if p.grad.is_complex():
                p.grad.data.real.clamp_(min=-clip_value, max=clip_value)
                p.grad.data.imag.clamp_(min=-clip_value, max=clip_value)
            else:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)


class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        mask = (x.max(dim=1, keepdim=True)[0] > 0.95).repeat(1, x.size(-1))
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone()


class HardRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:

        mask = (x.max(dim=1, keepdim=True)[0] > 0.9).repeat(1, x.size(-1))
        ctx.mask = mask
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone().masked_fill_(ctx.mask, 0)


def diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(-2), f"input x has to be a square matrix, but got {x.size()}"
    return RoundFunction.apply(x)


def hard_diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(-2), f"input x has to be a square matrix, but got {x.size()}"
    return HardRoundFunction.apply(x)
