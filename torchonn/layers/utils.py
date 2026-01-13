from typing import Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["merge_chunks", "partition_chunks"]


def merge_chunks(x: Tensor) -> Tensor:
    # x = [h1, w1, h2, w2, ...., hk, wk]
    # out: [h1*h2*...*hk, w1*w2*...*wk]

    dim = x.dim()
    x = x.permute(
        list(range(0, dim, 2)) + list(range(1, dim + 1, 2))
    )  # x = [h, bs, w, bs]
    x = x.reshape(np.prod([x.shape[i] for i in range(dim // 2)]), -1)

    return x


def partition_chunks(x: Tensor, out_shape: int | Tuple[int, ...]) -> Tensor:
    ### x: [h1*h2*...*hk, w1*w2*...*wk]
    ### out_shape: (h1, w1, ...)
    ### out: [h1, w1, h2, w2, ...., hk, wk]
    x_shape = (np.prod(out_shape[::2]), np.prod(out_shape[1::2]))
    if x_shape != x.shape:
        x = torch.nn.functional.pad(
            x[None, None], (0, x_shape[1] - x.shape[1], 0, x_shape[0] - x.shape[0])
        )[0, 0]
    in_shape = list(out_shape[::2]) + list(out_shape[1::2])
    x = x.reshape(in_shape)  # [h1, h2, ..., hk, w1, w2, ..., wk]
    x = x.permute(
        torch.arange(len(out_shape)).view(2, -1).t().flatten().tolist()
    )  # [h1, w1, h2, w2, ...., hk, wk]
    return x
