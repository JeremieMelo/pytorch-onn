'''
Date: 2024-06-12 10:48:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-12 10:48:00
FilePath: /pytorch-onn/torchonn/op/nonlinear_op.py
'''
"""
Date: 2024-06-12 00:51:51
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-12 00:53:14
FilePath: /pytorch-onn/torchonn/op/nonlinear_op.py
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

import numpy as np
import torch
from pyutils.general import TimerCtx
from torch import Tensor

sys.path.pop(0)

__all__ = ["mzi_nonlinearity_op"]


def mzi_nonlinearity_op(
    x: Tensor,
    alpha: Tensor | float,
    phase_bias: Tensor | float,
    tia_gain: Tensor | float,
    responsivity: Tensor | float,
    V_pi: Tensor | float,
) -> Tensor:
    """Reprogrammable Electro-Optic Nonlinear Activation Functions for Optical Neural Networks
    Ian A. D. Williamson, Member, IEEE, Tyler W. Hughes, Momchil Minkov, Ben Bartlett, Sunil Pai, Shanhui Fan, Fellow, IEEE, JSTQE 2019
    if x is complex number (mag/phase), then
    f(x) = j(1-alpha)**0.5 * exp(-j(g|x|^2 + phase_bias)/2)*cos((g|x|^2+phase_bias)/2)*x
    If x is intensity, then
    f(x) = (1-alpha) *cos^2((g*x+phase_bias)/2)*x

    Args:
        x (Tensor): Input complex field or intensity
        alpha (Tensor | float): alpha% intensity is coupled out and detected as electrical control
        phase_bias (Tensor | float): phase bias on the MZI
        tia_gain (Tensor | float): TIA gain
        responsivity (Tensor | float): Responsivity of the photodetector
        V_pi (Tensor | float): pi phase shift voltage of the phase shifter

    Returns:
        Tensor: output complex field or intensity
    """
    if torch.is_complex(x):  # complex field
        g = responsivity * np.pi / 2 / V_pi * tia_gain * alpha
        phi = g * x.abs().square() + phase_bias / 2
        return 1j * (1 - alpha) ** 0.5 * torch.exp(-1j * phi) * phi.cos() * x
    else:  # intensity
        g = responsivity * np.pi / 2 / V_pi * tia_gain * (1 - alpha)
        phi = g * x + phase_bias / 2
        return (1 - alpha) * phi.cos().square() * x


if __name__ == "__main__":
    c = 100
    N = 640
    device = torch.device("cuda:0")
    x = torch.randn(N, N, device=device)
    alpha = torch.randn(N, device=device).abs() + 0.1
    phase_bias = torch.randn(N, device=device)
    tia_gain = torch.randn(N, device=device).abs() + 0.1
    responsivity = torch.randn(N, device=device).abs() + 0.1
    V_pi = torch.randn(N, device=device).abs() + 0.1
    for _ in range(10):
        mzi_nonlinearity_op(x, alpha, phase_bias, tia_gain, responsivity, V_pi)
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(c):
            mzi_nonlinearity_op(x, alpha, phase_bias, tia_gain, responsivity, V_pi)
        torch.cuda.synchronize()
    print(t.interval / c)
