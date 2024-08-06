"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-07-18 00:01:34
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:01:36
"""

from pyutils.compute import (
    complex_mult,
    polar_to_complex,
    polynomial,
)
import logging

import numpy as np
import torch

torch._C._jit_set_profiling_executor(False)


__all__ = [
    "mrr_voltage_to_delta_lambda",
    "mrr_tr_to_roundtrip_phase",
    "mrr_roundtrip_phase_to_tr",
    "mrr_roundtrip_phase_to_tr_fused",
    "mrr_roundtrip_phase_to_tr_grad_fused",
    "mrr_roundtrip_phase_to_tr_func",
    "mrr_roundtrip_phase_to_out_phase",
    "mrr_tr_to_out_phase",
    "mrr_roundtrip_phase_to_tr_phase",
    "mrr_roundtrip_phase_to_tr_phase_fused",
    "mrr_modulator",
    "mrr_filter",
    "morr_filter",
    "mrr_fwhm_to_ng",
    "mrr_ng_to_fsr",
    "mrr_finesse",
]


def mrr_voltage_to_delta_lambda(v, alpha, k, gamma, n_g, lambda_0):
    """
    description: micro-ring resonator (MRR) wavelength modulation, \delta\lambda=\delta\n_eff\times\lambda/n_g, \deltan_eff=\gamma k \delta T=\gamma k \alpha v^2\\
    v {torch.Tensor ro np.ndarray} voltage \\
    alpha {scalar} voltage square to temperature change coefficient \\
    k {scalar} parameter \\
    gamma {scalar} power to phase shift coefficient \\
    n_g {scalar} group index, typically from 4 to 4.5\\
    lambda_0 {torch.Tensor or np.ndarray} central wavelength\\
    return delta_lambda {torch.Tensor or np.ndarray} resonance wavelength drift
    """
    delta_neff = gamma * k * alpha * v * v
    delta_lambda = delta_neff * lambda_0 / n_g
    return delta_lambda


def mrr_tr_to_roundtrip_phase(t, a, r):
    """
    description: field transmission to round trip phase shift
    t {torch.Tensor or np.ndarray} field transmission from [0,1] \\
    a {scalar} attenuation coefficient\\
    r {scalar} coupling coefficient\\
    return phi {torch.Tensor or np.ndarray} roune trip phase shift (abs(phase lag))[0, pi], center is 0. phase lag is negative, the sign is moved to the equation
    """
    # the curve has multiple valleies, thus given a t, there is infinite number of rt_phi, we only want [-pi, 0], thus the abs(phase lag) is in [0, pi], acos returns [0, pi], which matches our assumption
    assert 0 <= a <= 1, logging.error(f"Expect a from [0,1] but got {a}")
    assert 0 <= r <= 1, logging.error(f"Expect r from [0,1] but got {r}")
    # given a and r, the curve is fixed, the max and min may not be 1 and 0
    t = t.double()
    cos_phi = (a * a + r * r - t * (1 + r * r * a * a)) / (2 * a * r * (1 - t))
    cos_phi = cos_phi.clamp(0, 1)

    if isinstance(cos_phi, torch.Tensor):
        return cos_phi.acos().float(), cos_phi
    elif isinstance(cos_phi, np.ndarray):
        return np.arccos(cos_phi), cos_phi
    else:
        raise NotImplementedError



def mrr_roundtrip_phase_to_tr(
    rt_phi, a: float = 0.8, r: float = 0.9, poly_coeff=None, intensity: bool = False
):
    """
    description:  round trip phase shift to field transmission
    rt_phi {torch.Tensor or np.ndarray} abs of roundtrip phase shift (abs(phase lag)). range from abs([-pi, 0])=[0, pi]\\
    a {scalar} attenuation coefficient\\
    r {scalar} self-coupling coefficient\\
    poly_coeff {Callable} polynomial coefficients of intensity tranmission-roundtrip phase curve. Default set to None. None for slow computation\\
    intensity {bool scalar} whether output intensity tranmission or field transmission
    return t {torch.Tensor or np.ndarray} mrr through port field/intensity transmission
    """
    if poly_coeff is not None:
        # fast mode, use polynomial to predict the intensity transmission curve
        # if using polynomial, we want fast intensity transmission estimation, instead of field
        # if using coherent light, we will use complex output, we won't use polynomial fit
        t = polynomial(rt_phi.clamp(0, np.pi), poly_coeff).clamp(1e-8, 1)
        if not intensity:
            # avoid NAN
            t = (t + 1e-12).sqrt()
    else:
        # use slow but accurate mode from theoretical equation
        # create e^(-j phi) first
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        # ephi = torch.view_as_complex(polar_to_complex(mag=None, angle=-rt_phi)) ## this sign is from the negativity of phase lag
        # ### Jiaqi: Since PyTorch 1.7 rsub is not supported for autograd of complex, so have to use negate and add
        # a_ephi = -a * ephi
        # t = torch.view_as_real((r + a_ephi)/(1 + r * a_ephi))

        # if(intensity):
        #     t = get_complex_energy(t)
        # else:
        #     t = get_complex_magnitude(t)
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time', row_limit=5))
        ra_cosphi_by_n2 = -2 * r * a * rt_phi.cos()
        t = (a * a + r * r + ra_cosphi_by_n2) / (1 + r * r * a * a + ra_cosphi_by_n2)
        if not intensity:
            # as long as a is not equal to r, t cannot be 0.
            t = t.sqrt()
        return t


@torch.jit.script
def mrr_roundtrip_phase_to_tr_fused(
    rt_phi, a: float = 0.8, r: float = 0.9, intensity: bool = False
):
    """
    description:  round trip phase shift to field transmission
    rt_phi {torch.Tensor or np.ndarray} abs of roundtrip phase shift (abs(phase lag)). range from abs([-pi, 0])=[0, pi]\\
    a {scalar} attenuation coefficient\\
    r {scalar} self-coupling coefficient\\
    intensity {bool scalar} whether output intensity tranmission or field transmission\\
    return t {torch.Tensor or np.ndarray} mrr through port field/intensity transmission
    """

    # use slow but accurate mode from theoretical equation
    # create e^(-j phi) first

    # angle = -rt_phi
    # ephi = torch.view_as_complex(torch.stack([angle.cos(), angle.sin()], dim=-1)) ## this sign is from the negativity of phase lag
    # a_ephi = -a * ephi
    # t = torch.view_as_real((r + a_ephi).div(1 + r * a_ephi))
    # if(intensity):
    #     t = get_complex_energy(t)
    # else:
    #     t = get_complex_magnitude(t)
    ra_cosphi_by_n2 = -2 * r * a * rt_phi.cos()
    t = (a * a + r * r + ra_cosphi_by_n2) / (1 + r * r * a * a + ra_cosphi_by_n2)
    if not intensity:
        # as long as a is not equal to r, t cannot be 0.
        t = t.sqrt()

    return t


@torch.jit.script
def mrr_roundtrip_phase_to_tr_grad_fused(
    rt_phi, a: float = 0.8, r: float = 0.9, intensity: bool = False
):
    """
    description:  round trip phase shift to the gradient of field transmission
    rt_phi {torch.Tensor or np.ndarray} abs of roundtrip phase shift (abs(phase lag)). range from abs([-pi, 0])=[0, pi]\\
    a {scalar} attenuation coefficient\\
    r {scalar} self-coupling coefficient\\
    intensity {bool scalar} whether output intensity tranmission or field transmission\\
    return g {torch.Tensor or np.ndarray} the gradient of mrr through port field/intensity transmission
    """
    if not intensity:
        g = (a * r * (a**2 - 1) * (r**2 - 1) * rt_phi.sin()) / (
            (a**2 + r**2 - 2 * a * r * rt_phi.cos()) ** (1 / 2)
            * (a**2 * r**2 + 1 - 2 * a * r * rt_phi.cos()) ** 1.5
        )
    else:
        g = ((a**2 - 1) * (r**2 - 1) * 2 * a * r * rt_phi.sin()) / (
            a**2 * r**2 + 1 - 2 * a * r * rt_phi.cos()
        ) ** 2
    return g


def mrr_roundtrip_phase_to_tr_func(
    a: float = 0.8, r: float = 0.9, intensity: bool = False
):
    c1 = -2 * a * r
    c2 = a * a + r * r
    c3 = 1 + r * r * a * a - a * a - r * r
    c4 = (a**2 - 1) * (r**2 - 1) * 2 * a * r

    class MRRRoundTripPhaseToTrFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            # ra_cosphi_by_n2 = input.cos().mul_(c1)
            # numerator = ra_cosphi_by_n2.add_(c2)
            # denominator = numerator.add(c3)
            # t = numerator / denominator
            input = input.double()
            t = input.cos().mul_(c1).add_(c2 + c3).reciprocal_().mul_(-c3).add_(1)
            if not intensity:
                # as long as a is not equal to r, t cannot be 0.
                t.sqrt_()
            return t.float()

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors

            denominator = input.cos().mul_(c1).add_(c2 + c3)

            if intensity:
                denominator.square_()
                numerator = input.sin().mul_(c4)
            else:
                numerator = input.sin().mul_(c4 / 2)
                denominator = (
                    denominator.sub(1).pow_(1.5).mul_(denominator.sub(c3).sqrt_())
                )

            grad_input = numerator.div_(denominator).mul_(grad_output)

            return grad_input

    return MRRRoundTripPhaseToTrFunction.apply


def mrr_roundtrip_phase_to_out_phase(rt_phi, a, r):
    """
    description: from round trip phase to output phase response \\
    rt_phi {torch.Tensor or np.ndarray} round trip phase shift\\
    a {scalar} attenuation coefficient\\
    r {scalar} coupling coefficient\\
    return phase {torch.Tensor or np.ndarray} output phase response
    """
    if isinstance(rt_phi, torch.Tensor):
        arctan = torch.atan2
        sin = torch.sin
        cos = torch.cos
    elif isinstance(rt_phi, np.ndarray):
        arctan = np.arctan2
        sin = np.sin
        cos = np.cos
    else:
        raise NotImplementedError
    sin_rt_phi = sin(rt_phi)
    cos_rt_phi = cos(rt_phi)
    # phi = np.pi + rt_phi + arctan(r*sin_rt_phi-2*r*r*a*sin_rt_phi*cos_rt_phi+r*a*a*sin_rt_phi, (a-r*cos_rt_phi)*(1-r*a*cos_rt_phi))
    phi = (
        np.pi
        - rt_phi
        - arctan(r * sin_rt_phi, a - r * cos_rt_phi)
        - arctan(r * a * sin_rt_phi, 1 - r * a * cos_rt_phi)
    )
    return phi


def mrr_tr_to_out_phase(t, a, r, onesided=True):
    """
    description: field transmission to round trip phase shift
    t {torch.Tensor or np.ndarray} field transmission from [0,1] \\
    a {scalar} attenuation coefficient\\
    r {scalar} coupling coefficient\\
    onesided {bool scalar} True if only use half of the curve, output phase range [0, pi]
    return phi {torch.Tensor or np.ndarray} roune trip phase shift
    """
    rt_phi, cos_rt_phi = mrr_tr_to_roundtrip_phase(t, a, r)
    if isinstance(t, torch.Tensor):
        arctan = torch.atan2
        sin = torch.sin
    elif isinstance(t, np.ndarray):
        arctan = np.arctan2
        sin = np.sin
    else:
        raise NotImplementedError
    sin_rt_phi = sin(rt_phi)
    # phi = np.pi + rt_phi + arctan(r*sin_rt_phi-2*r*r*a*sin_rt_phi*cos_rt_phi+r*a*a*sin_rt_phi, (a-r*cos_rt_phi)*(1-r*a*cos_rt_phi))
    phi = (
        np.pi
        - rt_phi
        - arctan(r * sin_rt_phi, a - r * cos_rt_phi)
        - arctan(r * a * sin_rt_phi, 1 - r * a * cos_rt_phi)
    )
    if onesided:
        pass
    return phi


def mrr_roundtrip_phase_to_tr_phase(rt_phi, a, r):
    """
    description: from round trip phase to output transmission with phase response \\
    rt_phi {torch.Tensor or np.ndarray} round trip phase shift\\
    a {scalar} attenuation coefficient\\
    r {scalar} coupling coefficient\\
    return output {torch.Tensor or np.ndarray} transmission with phase response
    """
    # e^(-j phi)
    ephi = torch.view_as_complex(polar_to_complex(mag=None, angle=-rt_phi))
    a_ephi = -a * ephi
    output = torch.view_as_real((r + a_ephi) / (1 + r * a_ephi))
    return output


@torch.jit.script
def mrr_roundtrip_phase_to_tr_phase_fused(rt_phi, a: float, r: float):
    """
    description: from round trip phase to output transmission with phase response \\
    rt_phi {torch.Tensor or np.ndarray} round trip phase shift\\
    a {scalar} attenuation coefficient\\
    r {scalar} coupling coefficient\\
    return output {torch.Tensor or np.ndarray} transmission with phase response
    """
    # e^(-j phi)
    rt_phi = -rt_phi
    rt_phi = torch.complex(rt_phi.cos(), rt_phi.sin())
    rt_phi = -a * rt_phi
    output = torch.view_as_real((r + rt_phi) / (1 + r * rt_phi))
    return output


def mrr_modulator(t, a=0.9, r=0.8):
    """
    @description: all-pass MRR as a modulator. Map from the field intensity of through port transmission to coherent light with phase reponse\\
    @t {torch.Tensor or np.ndarray} field intensity modulation factor\\
    @a {float} attenuation factor from [0,1]. Default: 0.9\\
    @r {float} transmission/self-coupling factor from [0,1]. Default: 0.8\\
    @return: complexed light signal
    """
    phase = mrr_tr_to_out_phase(t, a, r)
    cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
    output_real = t * cos_phase
    output_imag = t * sin_phase
    output = torch.stack([output_real, output_imag], dim=-1)
    return output


def mrr_filter(x, t, a=0.9, r=0.8):
    """
    @description: all-pass MRR as a filter. Map from the input complex light signal to output signal with through port transmission\\
    @x {torch.Tensor or np.ndarray} complexed input light signal\\
    @t {torch.Tensor or np.ndarray} field intensity modulation factor\\
    @a {float} attenuation factor from [0,1]. Default: 0.9\\
    @r {float} transmission/self-coupling factor from [0,1]. Default: 0.8\\
    @return: complexed light signal
    """
    phase = mrr_tr_to_out_phase(t, a, r)
    cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
    phase_shift = torch.complex(cos_phase, sin_phase)
    out = t * complex_mult(x, phase_shift)
    return out


def morr_filter(
    rt_phi, tr_poly_coeff=None, a=0.9, r=0.8, x=None, coherent=False, intensity=False
):
    """
    description: from round trip phase shift to output signal \\
    rt_phi {torch.Tensor or np.ndarray, Optional} round trip phase shift. Default set to None \\
    tr_poly_coeff {Callable} polynomial coefficients of tranmission-roundtrip phase curve. Default set to None. None for slow computation\\
    a {float} attenuation factor from [0,1]. Default: 0.9\\
    r {float} transmission/self-coupling factor from [0,1]. Default: 0.8\\
    x {torch.Tensor or np.ndarray, Optional} input complex light signal {None, real tensor or complex tensor}. Default set to None\\
    coherent {bool scalar, Optional} coherent output or not. Default set to False\\
    intensity {bool scalar, Optional} whether use intensity or field transmission. Default set to False\\
    return output {torch.Tensor or np.ndarray} real tensor if incoherent, complex tensor if coherent
    """
    if not coherent:
        if x is None:
            # unit laser input with incoherent light, 1e^j0
            t = mrr_roundtrip_phase_to_tr(
                rt_phi, a=a, r=r, poly_coeff=tr_poly_coeff, intensity=intensity
            )
            return t
        else:
            # incoherent light with non-unit input, input must be real number
            t = mrr_roundtrip_phase_to_tr(
                rt_phi, a=a, r=r, poly_coeff=tr_poly_coeff, intensity=intensity
            )
            return x * t
    else:
        if x is None:
            # coherent light with unit laser, 1e^j0, treat morr as a mrr modulator
            phase = polar_to_complex(
                mag=None, angle=mrr_roundtrip_phase_to_out_phase(rt_phi, a, r)
            )
            return phase
        else:
            # coherent light with complex input
            return complex_mult(mrr_roundtrip_phase_to_tr_phase(rt_phi, a, r), x)


def mrr_fwhm_to_ng(a, r, radius, lambda0, fwhm):
    """
    description: from full-width half maximum (FWHM) and resonance wavelength to group index n_g (Bogaerts et al., Silicon microring resonators, Laser and Photonics Review 2011, Eq.(7))\\
    a {float} Attention coefficient\\
    r {float} Self-coupling coefficient\\
    radius {float} Radius of the MRR (unit: nm)\\
    lambda0 {float} Resonance wavelength (unit: nm)\\
    fwhm {float} bandwidth or full width half maximum (unit: nm)\\
    return n_g {float} Group index of the MRR
    """
    n_g = (
        (1 - r * a) * lambda0**2 / (2 * np.pi * np.pi * radius * (r * a) ** 0.5 * fwhm)
    )
    return n_g


def mrr_ng_to_fsr(lambda0, n_g, radius):
    """
    description: Calculate the free-spectral range (FSR) based on the central resonance wavelength, group index and MRR radius.
    (Bogaerts et al., Silicon microring resonators, Laser and Photonics Review 2011, Eq.(9))\\
    lambda0 {float} Resonance wavelength (unit: nm)\\
    n_g {float} Group index\\
    radius {float} Radius of the MRR (unit: nm)\\
    return fsr {float} Free-spectral range
    """
    fsr = lambda0**2 / (n_g * 2 * np.pi * radius)
    return fsr


def mrr_finesse(a, r):
    """
    description: Calculate the finesse of the MRR, i.e., finesse=FSR/FWHM=pi*sqrt(ra)/(1-ra) (Bogaerts et al., Silicon microring resonators, Laser and Photonics Review 2011, Eq.(21))\\
    a {float} Attention coefficient\\
    r {float} Self-coupling coefficient\\
    return finesse {float} Finesse of the MRR
    """
    ra = r * a
    finesse = np.pi * ra**0.5 / (1 - ra)
    return finesse
