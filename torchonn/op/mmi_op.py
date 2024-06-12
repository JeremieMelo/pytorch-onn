'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-02 02:57:22
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-12 01:10:29
'''
"""
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-01 19:10:55
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-01 19:23:02
"""
import logging
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
import numpy as np
import torch
from pyutils.general import TimerCtx
from torch import Tensor
import functools

sys.path.pop(0)

__all__ = ["multiport_mmi", "multiport_mmi_with_ps"]


@functools.lru_cache(maxsize=8)
def _multiport_mmi_phase(n_port: int=2, device: torch.device = torch.device("cuda")):
    x = torch.arange(1, n_port + 1, device=device, dtype=torch.float)
    x, y = torch.meshgrid(x, x)
    sign = ((-1) ** (x + y)).float()
    x = torch.exp(
        1j * ((x - 0.5) - sign * (y - 0.5)).square().mul(np.pi / (-4 * n_port))
    )
    return x.mul(sign)

def multiport_mmi(
    n_port: int = 2,
    transmission: Tensor = None,
    device: torch.device = torch.device("cuda"),
):
    """N by N mmi. Operation Principles for Optical Switches Based on Two Multimode Interference Couplers, JLT 2012

    Args:
        n_port (int): Number of input/output ports. Defaults to 2.
    """
    # assert n_port >= 2, print("[E] n_port must be at least 2.")
    phases = _multiport_mmi_phase(n_port, device)
    if transmission is None:
        transmission = (1 / n_port) ** 0.5
    
    x = phases.mul(transmission) # if transmission is trainable, output will also be trainable.

    return x


# @torch.jit.script
def multiport_mmi_with_ps(
    n_port: int = 2, phases: Tensor=None, ps_loc: str = "before", device: torch.device = torch.device("cuda")
):
    """N by N mmi with N ps at the input ports. Operation Principles for Optical Switches Based on Two Multimode Interference Couplers, JLT 2012

    Args:
        n_port (int): Number of input/output ports. Defaults to 2.
        phases (Tensor): Phase shifters at the input ports. Defaults to None. Shape can be batched phases [..., n_port]
        Return: [..., n_port, n_port] complex transmission matrix. It might be batched if ps is batched
    """
    x = multiport_mmi(n_port, device=device)
    # angle = x.sub(0.5).sub_(y.sub(0.5).mul_(sign)).square_().mul_(np.pi / (-4 * n_port))
    # x = torch.complex(angle.cos(), angle.sin()).mul_(sign.mul_((1 / n_port) ** 0.5))
    # angle = torch.randn(n_port, device=device)
    # ps = torch.exp(1j * angle * np.pi)
    # angle[1::2] += 1
    
    # ps = torch.complex(angle.cos(), angle.sin())
    if phases is not None:
        if ps_loc == "before":
            x = x.mul(phases.unsqueeze(-2))
        elif ps_loc == "after":
            x = phases.unsqueeze(-1).mul(x)
    return x


if __name__ == "__main__":
    c = 100
    N = 640
    for _ in range(10):
        multiport_mmi(64)
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(c):
            multiport_mmi(N)
        torch.cuda.synchronize()
    print(t.interval / c)
    # print(multiport_mmi(4))
    # print(multiport_mmi_with_ps(4, "after"))
    # print(multiport_mmi_with_ps(4, "before"))
