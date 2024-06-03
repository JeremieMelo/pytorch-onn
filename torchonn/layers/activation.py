"""
Date: 2024-06-01 12:37:45
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-01 12:37:45
FilePath: /pytorch-onn/torchonn/layers/activation.py
"""

from mmengine.registry import MODELS
from torch import nn

__all__ = ["ReLUN"]


@MODELS.register_module()
class ReLUN(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLUN}(x) = \min(\max(0,x), N)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLUN(N)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, N, inplace=False):
        super(ReLUN, self).__init__(0.0, N, inplace)

    def extra_repr(self):
        inplace_str = "inplace" if self.inplace else ""
        return inplace_str
