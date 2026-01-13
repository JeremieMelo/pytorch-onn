"""
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditTime: 2022-04-18 16:51:03
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
Description: Helpful function
FilePath: /projects/ELight/ops/utils.py
"""

import numpy as np  # math operations
import torch
from torch import Tensor, nn
from torch.types import Device

__all__ = [
    "weight_quantize_fn_log",
    "weight_to_quantized_weight",
    "weight_to_quantized_weight_cpu",
]


# auto grad weight quantziation func
def weight_quantization(b, power, power_base, assign):
    def efficient_power_quant(x, power_base, b, assign):
        if assign:
            # w = w_pos - w_neg as we use positive and nagative PTCs to represent weight
            ref_value = power_base ** (2**b - 1)  # smallest trasmission factor
            scaleQuantLevel = 1 - ref_value

            x = x.mul(scaleQuantLevel)

            # obtain the quant level
            x_q_levels_l = torch.clamp(
                torch.floor(torch.log(x + ref_value) / np.log(power_base)), 0, 2**b - 1
            )
            x_q_levels_u = torch.clamp(
                torch.ceil(torch.log(x + ref_value) / np.log(power_base)), 0, 2**b - 1
            )

            # convert to uniform domain
            x_q_l = power_base**x_q_levels_l
            x_q_u = power_base**x_q_levels_u

            # generate fake max level mask
            x_q_l_mask = x_q_l < (power_base ** (2**b - 1))
            x_q_u_mask = x_q_u < (power_base ** (2**b - 1))

            # replace the fake max level in low level with 2**b - 1
            x_q_l[x_q_l_mask] = power_base ** (2**b - 1)
            x_q_u[x_q_u_mask] = 0

            # stack low and up bound
            x_q_bound = torch.stack([x_q_l, x_q_u], dim=-1)

            # compute dist and then choose the min one
            # AT! Need add ref_value for fair comparison
            x_q_dist = (x.add(ref_value).unsqueeze(-1) - x_q_bound).abs().min(dim=-1)[1]

            # obtain return value
            x_q = x_q_bound.gather(-1, x_q_dist.unsqueeze(-1)).squeeze(-1)

            # sub ref_value
            x_q = x_q.sub(ref_value).div(scaleQuantLevel)

            return x_q
        else:
            raise NotImplementedError

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):

            input_c = input
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_abs /= alpha  # scale value to 0-1

            if power:
                input_q = efficient_power_quant(input_abs, power_base, b, assign).mul(
                    sign
                )
            else:
                raise NotImplementedError

            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # grad for weights will not be clipped
            return grad_input, None

    return _pq().apply


class weight_quantize_fn_log(nn.Module):
    """Support logrithmic quantization by setting power=True"""

    def __init__(
        self,
        w_bit: int,
        power_base: float = 0.872,
        has_zero: bool = True,
        power: bool = True,
        quant_range: str = "max",
        assign: bool = True,
        device: Device = None,
    ) -> None:
        """Initialization

        Args:
            w_bit (int): weight bitwidth
            power_base (float, optional): The based of power quantization b^i. Defaults to 0.872.
            has_zero (bool, optional): Whether to include zero in the quantized tensor. Defaults to True.
            power (bool, optional): Whether to use power quantization. Defaults to True.
            quant_range (str, optional): Quantization range mode. Defaults to "max".
            assign (bool, optional): whether to assign to real hardware implementation. Defaults to True.
            device (Device, optional): torch Device. Defaults to None.
        """
        super(weight_quantize_fn_log, self).__init__()
        assert (0 < w_bit <= 8) or w_bit == 32
        self.w_bit = w_bit
        self.power = power
        self.power_base = power_base
        self.has_zero = (
            has_zero  # whether implement 0 based on postive and negative PTCs
        )
        self.assign = assign
        self.device = device

        self.weight_q = weight_quantization(
            b=self.w_bit,
            power=self.power,
            power_base=self.power_base,
            assign=self.assign,
        )

        self.quant_range = quant_range

    def set_bitwidth(self, bit: int) -> None:
        if bit != self.w_bit:
            self.w_bit = bit
            self.weight_q = weight_quantization(
                b=self.w_bit,
                power=self.power,
                power_base=self.power_base,
                assign=self.assign,
            )

    def forward(self, weight: Tensor) -> Tensor:
        if self.w_bit == 32:
            weight_q = weight
        else:
            if self.quant_range == "max":
                weight = torch.tanh(weight)
                alpha = torch.max(torch.abs(weight.data))  # scaling factor
            weight_q = self.weight_q(weight, alpha)

        return weight_q


def convert_weight_to_levels(
    bits: int, base: float, power: bool = True, assign: bool = True, loss_fn: str = "l1"
):
    def assign_array_value(x, assign, assign_zero_value, sign):
        """
        args:
            x: data
            assign: whether to assign to real array
            assign_zero_value: define use which level to represent 0; defualt 2**bits - 1
            sign: pass sign in to clarify 0 and - 0
        """
        ### assign levels into positive and negative array
        # copy and one for positive and one for negative: if + 3: postive keep and negative = 0 else -: negative keep and postive 0
        x_pos = torch.abs(x)  # postive array
        x_neg = x_pos  # neagtive array
        if assign:
            x_pos_mask = x >= 0
            x_neg_mask = x < 0

            # convert
            x_pos = x_pos.masked_fill(x_neg_mask, assign_zero_value)
            x_neg = x_neg.masked_fill(x_pos_mask, assign_zero_value)

            # check 0 and -0 using sign(data)
            sign_mask = (
                sign < 0
            )  # use this to indicate which data is minius such that its levels should be also in neg array
            x_zero_mask = x == 0

            # obtain negative zero mask
            negative_zero_mask = torch.logical_and(sign_mask, x_zero_mask)
            # print(negative_zero_mask)

            x_pos[negative_zero_mask] = 2**bits - 1
            x_neg[negative_zero_mask] = 0

        else:
            raise NotImplementedError

        # inverse the size of levels such that it follows the same in the ori weight domain: 1 -> max level
        x_pos = 2**bits - 1 - x_pos
        x_neg = -(2**bits) + 1 + x_neg

        if loss_fn == "l1":
            x_q_levels_p_n = x_pos + x_neg
        elif loss_fn == "l2":
            x_q_levels_p_n = torch.cat((x_pos, x_neg), -1)
        else:
            assert NotImplementedError

        return x_q_levels_p_n, x_pos_mask, x_neg_mask

    def uniform_quant(x, bits):
        # must be scaled to 0-1 before this function
        x = x.mul((2**bits - 1))
        x_q = x.round().div(2**bits - 1)
        x_q_levels = x.round()

        return x_q, x_q_levels

    def efficient_power_quant(x, base, bits, assign):
        """
        efficient power quant impl
        args:
            base
            bits: bits
        """
        if assign:
            ref_value = base ** (2**bits - 1)
            scaleQuantLevel = 1 - ref_value
            x = x.mul(scaleQuantLevel)

            x_q_levels_l = torch.abs(
                torch.clamp(
                    torch.floor(torch.log(x + ref_value) / np.log(base)), 0, 2**bits - 1
                )
            )
            x_q_levels_u = torch.abs(
                torch.clamp(
                    torch.ceil(torch.log(x + ref_value) / np.log(base)), 0, 2**bits - 1
                )
            )

            # convert to uniform domain
            x_q_l = base**x_q_levels_l
            x_q_u = base**x_q_levels_u

            # stack low and up bound
            x_q_bound = torch.stack([x_q_l, x_q_u], dim=-1)
            x_q_levels_bound = torch.stack([x_q_levels_l, x_q_levels_u], dim=-1)

            x_q_dist = (x.add(ref_value).unsqueeze(-1) - x_q_bound).abs().min(dim=-1)[1]

            # obtain return value
            x_q = x_q_bound.gather(-1, x_q_dist.unsqueeze(-1)).squeeze(-1)
            x_q_levels = x_q_levels_bound.gather(-1, x_q_dist.unsqueeze(-1)).squeeze(-1)

            # sub ref_value
            x_q = x_q.sub(ref_value).div(scaleQuantLevel)
        else:
            raise NotImplementedError

        return x_q, x_q_levels

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha, assign_zero_value, grad_update_xor_mask):
            sign = input.sign()
            input_abs = input.abs()
            ctx.alpha = alpha

            if power:
                input_q, input_q_levels = efficient_power_quant(
                    input_abs, base, bits, assign
                )
            else:
                input_q, input_q_levels = uniform_quant(input_abs, bits)

            # ctx.save_for_backward(input, input_q)
            input_q = input_q.mul_(alpha).mul_(sign)

            # mul_ 0 would be a problem thus replace 0 with 1
            sign[sign == 0] = 1
            input_q_levels = input_q_levels.mul_(sign)

            # assign support pos and neg seprately and combined output
            input_q_levels, x_pos_mask, x_neg_mask = assign_array_value(
                input_q_levels, assign, assign_zero_value, sign
            )
            input_q_levels = input_q_levels.div(2**bits - 1)

            ctx.save_for_backward(
                input_abs, x_pos_mask, x_neg_mask, grad_update_xor_mask
            )

            return input_q_levels

        @staticmethod
        def backward(ctx, grad_output):
            input_abs, x_pos_mask, x_neg_mask, grad_update_xor_mask = ctx.saved_tensors
            ref_value = base ** (2**bits - 1)
            scaleQuantLevel = 1 - ref_value

            if loss_fn == "l1":
                grad_input = grad_output.clone()  # grad for weights will not be clipped
                grad_input = (
                    grad_input.mul(scaleQuantLevel)
                    .div(input_abs * scaleQuantLevel + ref_value)
                    .div(-np.log(base))
                    .div(2**bits - 1)
                )

            elif loss_fn == "l2":
                grad_input_tmp = grad_output.clone()
                grad_input1, grad_input2 = torch.chunk(grad_input_tmp, 2, dim=-1)

                # directly use pos and neg mask
                x_neg_mask_real = x_neg_mask
                x_pos_mask_real = x_pos_mask

                grad_input1[x_neg_mask_real] = 0
                grad_input2[x_pos_mask_real] = 0

                grad_input = (
                    (grad_input1.add(grad_input2))
                    .mul(scaleQuantLevel)
                    .div(input_abs * scaleQuantLevel + ref_value)
                    .div(-np.log(base))
                    .div(2**bits - 1)
                )

            # print_stat(grad_input)
            return grad_input, None, None, None

    return _pq().apply


class weight_to_quantized_weight(torch.nn.Module):
    """
    torch.nn.Module to convert weight to quantized levels such that we can compute the levels difference loss
    args:
        bits:
        power: bool, whether to use power
        base: base of power func
        assign: bool, whether to assign to real implementation
        assign_zero_value: adjustable/ trainable zero_value: how to add its grad, check additice quant
        loss_fn: l1 or l2
    """

    def __init__(self, bits, base, power, assign, assign_zero_value, loss_fn):
        super(weight_to_quantized_weight, self).__init__()
        ## init
        self.bits = bits
        self.base = base
        self.power = power
        self.assign = assign
        self.assign_zero_value = assign_zero_value
        self.loss_fn = loss_fn

        ## init converter with bits, base, power, assign
        self.converter = convert_weight_to_levels(
            self.bits, self.base, self.power, self.assign, loss_fn
        )

    def set_assign_zero_value(self, assign_zero_value=None):
        if assign_zero_value is None:
            assign_zero_value = 2**self.bits - 1
        self.assign_zero_value = assign_zero_value

    def set_bitwidth(self, bit: int) -> None:
        if bit != self.bits:
            self.bits = bit
            self.converter = convert_weight_to_levels(
                self.bits, self.base, self.power, self.assign, self.loss_fn
            )
            self.set_assign_zero_value(2**bit - 1)

    def forward(self, x, grad_update_xor_mask=None):
        x = torch.tanh(x)
        alpha = torch.max(torch.abs(x))

        x = x / alpha

        x_q_levels = self.converter(
            x, alpha, self.assign_zero_value, grad_update_xor_mask
        )

        return x_q_levels


##################### CPU #################
def assign_array_value_cpu(x, bits, assign, assign_zero_value, sign, sep_flag):

    x_pos = torch.abs(x)  # postive array
    x_neg = x_pos  # neagtive array
    if assign:
        x_pos_mask = x >= 0
        x_neg_mask = x < 0

        # convert
        x_pos = x_pos.masked_fill(x_neg_mask, assign_zero_value)
        x_neg = x_neg.masked_fill(x_pos_mask, assign_zero_value)

        # check 0 and -0 using sign(data)
        sign_mask = sign < 0
        x_zero_mask = x == 0

        # how to do bool
        negative_zero_mask = torch.logical_and(sign_mask, x_zero_mask)
        # print(negative_zero_mask)

        x_pos[negative_zero_mask] = 2**bits - 1
        x_neg[negative_zero_mask] = 0

        # # cat
        # x_q_levels_p_n =  torch.stack((x_pos, x_neg), -1)
    else:
        raise NotImplementedError
    x_pos = 2**bits - 1 - x_pos  # 2**bits - 1 ~ 0
    x_neg = -(2**bits) + 1 + x_neg  # - (2**bits - 1) ~ 0

    if sep_flag:
        x_q_levels_p_n = torch.cat((x_pos, x_neg), -1)
    else:
        x_q_levels_p_n = x_pos + x_neg

    return x_q_levels_p_n, x_pos_mask, x_neg_mask


def uniform_quant_cpu(x, bits):
    # must be scaled to 0-1 before this function
    x = x.mul((2**bits - 1))
    x_q = x.round().div(2**bits - 1)
    x_q_levels = x.round()

    return x_q, x_q_levels


def efficient_power_quant_cpu(x, base, bits, assign):
    """
    efficient power quant impl
    args:
        base
        bits: bits
    """
    if assign:
        ref_value = base ** (2**bits - 1)

        scaleQuantLevel = 1 - ref_value
        x = x.mul(scaleQuantLevel)

        x_q_levels_l = torch.abs(
            torch.clamp(
                torch.floor(torch.log(x + ref_value) / np.log(base)), 0, 2**bits - 1
            )
        )
        x_q_levels_u = torch.abs(
            torch.clamp(
                torch.ceil(torch.log(x + ref_value) / np.log(base)), 0, 2**bits - 1
            )
        )

        # convert to uniform domain
        x_q_l = base**x_q_levels_l
        x_q_u = base**x_q_levels_u

        # stack low and up bound
        x_q_bound = torch.stack([x_q_l, x_q_u], dim=-1)
        x_q_levels_bound = torch.stack([x_q_levels_l, x_q_levels_u], dim=-1)

        x_q_dist = (x.add(ref_value).unsqueeze(-1) - x_q_bound).abs().min(dim=-1)[1]

        # obtain return value
        x_q = x_q_bound.gather(-1, x_q_dist.unsqueeze(-1)).squeeze(-1)
        x_q_levels = x_q_levels_bound.gather(-1, x_q_dist.unsqueeze(-1)).squeeze(-1)

        x_q = x_q.sub(ref_value).div(scaleQuantLevel)
    else:
        raise NotImplementedError

    return x_q, x_q_levels


class weight_to_quantized_weight_cpu(object):
    def __init__(self, bits, base, power, assign, assign_zero_value, sep_flag=True):
        super(weight_to_quantized_weight_cpu, self).__init__()
        ## init
        self.bits = bits
        self.base = base
        self.power = power
        self.assign = assign
        self.assign_zero_value = assign_zero_value

        self.sep_flag = (
            sep_flag  # whether to sep weight level into pos, neg by torch cat
        )

    def set_assign_zero_value(self, assign_zero_value=None):
        if assign_zero_value is None:
            assign_zero_value = 2**self.bits - 1
        self.assign_zero_value = assign_zero_value

    def set_bitwidth(self, bit: int) -> None:
        if bit != self.bits:
            self.bits = bit
            self.set_assign_zero_value(2**bit - 1)

    def forward(self, input):
        ## emulate quantization flow
        input = torch.tanh(input)
        alpha = torch.max(torch.abs(input))

        sign = input.sign()
        input_abs = input.abs()
        input_abs /= alpha  # scale value to 0-1

        if self.power:
            input_q, input_q_levels = efficient_power_quant_cpu(
                input_abs, self.base, self.bits, self.assign
            )
        else:
            input_q, input_q_levels = uniform_quant_cpu(input_abs, self.bits)

        input_q = input_q.mul_(alpha).mul_(sign)
        # mul_ 0 would be a problem thus replace 0 with 1
        sign[sign == 0] = 1
        input_q_levels = input_q_levels.mul_(sign)
        input_q_levels, x_pos_mask, x_neg_mask = assign_array_value_cpu(
            input_q_levels,
            self.bits,
            self.assign,
            self.assign_zero_value,
            sign,
            self.sep_flag,
        )
        input_q_levels = input_q_levels.div(2**self.bits - 1)

        return input_q, input_q_levels
