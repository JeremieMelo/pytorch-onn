"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-06-01 02:00:35
"""
from tabnanny import verbose
from torchonn.layers.fftonn_linear import FFTONNBlockLinear
import torch
import torch.fft


def test():
    device = "cuda"
    N = 8
    eye = torch.eye(N, N, dtype=torch.cfloat, device=device)
    P = torch.randn(N, N, dtype=torch.cfloat, device=device).svd()[0]
    B = torch.randn(N, N, dtype=torch.cfloat, device=device).svd()[0]
    sigma = torch.nn.Parameter(torch.randn(N, dtype=torch.cfloat, device=device))
    W = torch.randn(N, N, dtype=torch.cfloat, device=device)

    ## mapping
    S = torch.linalg.inv(B).matmul(W).matmul(torch.linalg.inv(P))
    optimizer = torch.optim.Adam([sigma], lr=1e-3)
    T = 4000
    for i in range(T):
        loss = torch.nn.functional.mse_loss(torch.view_as_real(B.matmul(sigma.unsqueeze(1) * P)), torch.view_as_real(W))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 500 == 0 or i == T - 1:
            print(f"Step: {i}, loss={loss.item()}")
    print(sigma.data)
    print(S)


def test_build_from_layer():
    device = "cuda"
    fc = torch.nn.Linear(8, 8).to(device)
    layer = FFTONNBlockLinear.from_layer(fc, miniblock=4, mode="trainable", verbose=True)

test_build_from_layer()
