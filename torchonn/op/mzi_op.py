"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 19:12:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 19:12:42
"""
from functools import lru_cache
from multiprocessing.dummy import Pool
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from pyutils.compute import add_gaussian_noise_cpu, gen_gaussian_noise
from pyutils.general import logger
from pyutils.quantize import uniform_quantize, uniform_quantize_cpu
from pyutils.torch_train import apply_weight_decay, set_torch_deterministic
from torch.tensor import Tensor
from torch.types import Device, _size
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch

__all__ = [
    "phase_quantize_fn_cpu",
    "phase_quantize_fn",
    "voltage_quantize_fn_cpu",
    "voltage_quantize_fn",
    "clip_to_valid_quantized_voltage_cpu",
    "clip_to_valid_quantized_voltage",
    "clip_to_valid_quantized_voltage_",
    "wrap_to_valid_phase",
    "voltage_to_phase_cpu",
    "voltage_to_phase",
    "phase_to_voltage_cpu",
    "phase_to_voltage",
    "upper_triangle_to_vector_cpu",
    "vector_to_upper_triangle_cpu",
    "upper_triangle_to_vector",
    "vector_to_upper_triangle",
    "checkerboard_to_vector",
    "vector_to_checkerboard",
    "complex_to_real_projection",
    "project_matrix_to_unitary",
    "real_matrix_parametrization_cpu",
    "real_matrix_reconstruction_cpu",
    "usv",
    "DiagonalQuantizer",
    "UnitaryQuantizer",
    "PhaseQuantizer",
    "ThermalCrosstalkSimulator",
]


class phase_quantize_fn_cpu(object):
    def __init__(self, p_bit):
        super(phase_quantize_fn_cpu, self).__init__()
        assert p_bit <= 8 or p_bit == 32
        self.p_bit = p_bit
        self.uniform_q = uniform_quantize_cpu(bits=p_bit)
        self.pi = np.pi

    def __call__(self, x):
        if self.p_bit == 32:
            phase_q = x
        elif self.p_bit == 1:
            E = np.mean(np.abs(x))
            phase_q = self.uniform_q(x / E) * E
        else:
            phase = x / 2 / self.pi + 0.5
            phase_q = self.uniform_q(phase) * 2 * self.pi - self.pi
        return phase_q


class voltage_quantize_fn_cpu(object):
    def __init__(self, v_bit, v_pi, v_max):
        super(voltage_quantize_fn_cpu, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi ** 2)
        self.uniform_q = uniform_quantize_cpu(bits=v_bit)
        self.pi = np.pi

    def __call__(
        self, x, voltage_mask_old=None, voltage_mask_new=None, voltage_backup=None, strict_mask=True
    ):
        if self.v_bit == 32:
            voltage_q = x
        elif self.v_bit == 1:
            E = np.mean(np.abs(x))
            voltage_q = self.uniform_q(x / E) * E
        else:
            # min_V = 0
            ### max voltage is determined by the voltage supply, not the phase shifter's characteristics!!! ###
            # max_V = np.sqrt(2*self.pi/self.gamma)
            max_V = self.v_max
            # voltage = (x - min_V) / (max_V - min_V)
            voltage = x / max_V
            # phase_q = 2 * self.uniform_q(phase) - 1
            voltage_q = self.uniform_q(voltage) * max_V

            if voltage_mask_old is not None and voltage_mask_new is not None and voltage_backup is not None:
                if strict_mask == True:
                    # strict mask will always fix masked voltages, even though they are not covered in the new mask
                    # "1" in mask indicates to apply quantization
                    voltage_mask_newly_marked = voltage_mask_new ^ voltage_mask_old
                    voltage_q_tmp = x.copy()
                    # maintain voltages that have already been masked
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    # print("any newly marked voltages:", voltage_mask_newly_marked.any())
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    # only update newly quantized voltages, previously quantized voltages are maintained
                    # if (voltage_backup[voltage_mask_newly_marked].sum() > 1e-4):
                    #     print(voltage_backup[voltage_mask_newly_marked])

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_q = voltage_q_tmp
                else:
                    # non-strict mask will make unmasked voltages trainable again
                    voltage_q_tmp = x.copy()
                    voltage_mask_old = voltage_mask_old & voltage_mask_new
                    voltage_mask_newly_marked = (~voltage_mask_old) & voltage_mask_new
                    # maintain voltages that have already been masked and being masked in the new mask
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    voltage_q = voltage_q_tmp

        return voltage_q


class phase_quantize_fn(torch.nn.Module):
    """
    description: phase shifter voltage control quantization with gamma noise injection and thermal crosstalk
    """

    def __init__(
        self,
        v_bit,
        v_pi,
        v_max,
        gamma_noise_std=0,
        crosstalk_factor=0,
        random_state=None,
        device=torch.device("cuda"),
    ):
        super(phase_quantize_fn, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi ** 2)
        self.gamma_noise_std = gamma_noise_std
        self.crosstalk_factor = crosstalk_factor
        self.voltage_quantizer = voltage_quantize_fn(v_bit, v_pi, v_max)
        self.pi = np.pi
        self.random_state = random_state
        self.device = device

        self.crosstal_simulator = ThermalCrosstalkSimulator(
            plotting=False, filter_size=3, crosstalk_factor=crosstalk_factor, device=self.device
        )

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.crosstal_simulator.set_crosstalk_factor(crosstalk_factor)

    def forward(self, x, mixedtraining_mask=None, mode="triangle"):
        if self.gamma_noise_std > 1e-5:
            gamma = gen_gaussian_noise(
                x,
                noise_mean=self.gamma,
                noise_std=self.gamma_noise_std,
                trunc_range=(self.gamma - 3 * self.gamma_noise_std, self.gamma + 3 * self.gamma_noise_std),
                random_state=self.random_state,
            )
        else:
            gamma = self.gamma
        if self.v_bit >= 16:
            ## no quantization
            ## add gamma noise with quick approach
            phase = (gamma / self.gamma * (x % (2 * np.pi))) % (2 * np.pi)
        else:
            ## quantization
            ## add gamma noise in transform
            phase = voltage_to_phase(
                clip_to_valid_quantized_voltage(
                    self.voltage_quantizer(phase_to_voltage(x, self.gamma)),
                    self.gamma,
                    self.v_bit,
                    self.v_max,
                    wrap_around=True,
                ),
                gamma,
            )
        ## add crosstalk with mixed training mask
        if self.crosstalk_factor > 1e-5:
            phase = self.crosstal_simulator.simple_simulate(phase, mixedtraining_mask, mode)
        return phase


class voltage_quantize_fn(torch.nn.Module):
    def __init__(self, v_bit, v_pi, v_max):
        super(voltage_quantize_fn, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi ** 2)
        self.uniform_q = uniform_quantize(k=v_bit)
        self.pi = np.pi

    def forward(self, x):
        if self.v_bit == 32:
            voltage_q = x
        elif self.v_bit == 1:
            E = x.data.abs().mean()
            voltage_q = self.uniform_q(x / E) * E
        else:
            max_V = self.v_max
            voltage = x / max_V
            voltage_q = self.uniform_q(voltage) * max_V
        return voltage_q


def clip_to_valid_quantized_voltage_cpu(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2 ** v_bit - 1)
    if wrap_around:
        mask = voltages >= v_2pi
        voltages[mask] = 0
    else:
        voltages[voltages > v_2pi] -= v_interval
    return voltages


def clip_to_valid_quantized_voltage(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2 ** v_bit - 1)
    if isinstance(voltages, np.ndarray):
        if wrap_around:
            mask = voltages >= v_2pi
            voltages = voltages.copy()
            voltages[mask] = 0
        else:
            voltages = voltages.copy()
            voltages[voltages > v_2pi] -= v_interval
    elif isinstance(voltages, torch.Tensor):
        if wrap_around:
            mask = voltages.data < v_2pi
            voltages = voltages.mul(mask.float())
            # voltages = voltages.data.clone()
            # voltages[mask] = 0
        else:
            mask = voltages > v_2pi
            voltages = voltages.masked_scatter(mask, voltages[mask] - v_interval)
            # voltages = voltages.data.clone()
            # voltages[voltages > v_2pi] -= v_interval
    else:
        assert 0, logger.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(voltages)}"
        )

    return voltages


def clip_to_valid_quantized_voltage_(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2 ** v_bit - 1)
    if isinstance(voltages, np.ndarray):
        if wrap_around:
            mask = voltages >= v_2pi
            voltages[mask] = 0
        else:
            voltages[voltages > v_2pi] -= v_interval
    elif isinstance(voltages, torch.Tensor):
        if wrap_around:
            mask = voltages >= v_2pi
            voltages.data[mask] = 0
        else:
            voltages.data[voltages > v_2pi] -= v_interval
    else:
        assert 0, logger.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(voltages)}"
        )

    return voltages


def wrap_to_valid_phase(phases, mode="positive"):
    assert mode in {"symmetric", "positive"}
    if mode == "positive":
        phases = phases % (2 * np.pi)
        return phases
    elif mode == "symmetric":
        phases = phases % (2 * np.pi)
        phases.data[phases > np.pi] -= 2 * np.pi
        return phases


def voltage_to_phase_cpu(voltages, gamma):
    # phases = -np.clip(gamma * voltages * voltages, a_min=0, a_max=2 * np.pi)
    # change phase range from [0, 2*pi] to [-pi,pi]
    pi_2 = 2 * np.pi
    phases = (gamma * voltages * voltages) % pi_2
    phases[phases > np.pi] -= pi_2
    return phases


voltage_to_phase = voltage_to_phase_cpu


def phase_to_voltage_cpu(phases, gamma):
    pi = np.pi
    if isinstance(phases, np.ndarray):
        phases_tmp = phases.copy()
        phases_tmp[phases_tmp > 0] -= 2 * pi  # change phase lead to phase lag
        voltage_max = np.sqrt((2 * pi) / gamma)
        voltages = np.clip(np.sqrt(np.abs(phases_tmp / gamma)), a_min=0, a_max=voltage_max)
    else:
        voltages = (phases % (2 * np.pi)).div(gamma).sqrt()
    return voltages


phase_to_voltage = phase_to_voltage_cpu


@lru_cache(maxsize=32)
def upper_triangle_masks_cpu(N):
    rows, cols = np.triu_indices(N, 1)
    masks = (rows, cols - rows - 1)
    return masks


@lru_cache(maxsize=32)
def upper_triangle_masks(N, device=torch.device("cuda")):
    masks = torch.triu_indices(N, N, 1, device=device)
    masks[1, :] -= masks[0, :] + 1
    return masks


def upper_triangle_to_vector(mat, complex=False):
    if isinstance(mat, np.ndarray):
        N = mat.shape[-2] if complex else mat.shape[-1]
        masks = upper_triangle_masks_cpu(N)
        if complex:
            vector = mat[..., masks[0], masks[1], :]
        else:
            vector = mat[..., masks[0], masks[1]]
    elif isinstance(mat, torch.Tensor):
        N = mat.shape[-2] if complex else mat.shape[-1]
        masks = upper_triangle_masks(N, device=mat.device)
        if complex:
            vector = mat[..., masks[0], masks[1], :]
        else:
            vector = mat[..., masks[0], masks[1]]
    else:
        raise NotImplementedError

    return vector


upper_triangle_to_vector_cpu = upper_triangle_to_vector


def vector_to_upper_triangle(vec, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    if isinstance(vec, np.ndarray):
        M = vec.shape[-2] if complex else vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        masks = upper_triangle_masks_cpu(N)
        if complex:
            mat = np.zeros(shape=list(vec.shape[:-2]) + [N, N, vec.shape[-1]], dtype=vec.dtype)
            mat[..., masks[0], masks[1], :] = vec
        else:
            mat = np.zeros(shape=list(vec.shape[:-1]) + [N, N], dtype=vec.dtype)
            mat[..., masks[0], masks[1]] = vec
    elif isinstance(vec, torch.Tensor):
        M = vec.shape[-2] if complex else vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        masks = upper_triangle_masks(N, device=vec.device)
        if complex:
            mat = torch.zeros(
                size=list(vec.size())[:-2] + [N, N, vec.size(-1)], dtype=vec.dtype, device=vec.device
            )
            mat[..., masks[0], masks[1], :] = vec
        else:
            mat = torch.zeros(size=list(vec.size())[:-1] + [N, N], dtype=vec.dtype, device=vec.device)
            mat[..., masks[0], masks[1]] = vec
    else:
        raise NotImplementedError

    return mat


vector_to_upper_triangle_cpu = vector_to_upper_triangle


def checkerboard_to_vector(mat, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    ### even column phases + odd colum phases, compact layoutapplication

    if isinstance(mat, np.ndarray):
        if complex:
            mat = np.transpose(mat, axes=np.roll(np.arange(mat.ndim), 1))
        N = mat.shape[-1]
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = np.swapaxes(mat[..., :upper_oddN:2, ::2], -1, -2).reshape([*mat.shape[:-2], -1])
        vector_odd_col = np.swapaxes(mat[..., 1:upper_evenN:2, 1::2], -1, -2).reshape([*mat.shape[:-2], -1])
        vector = np.concatenate([vector_even_col, vector_odd_col], -1)
        if complex:
            vector = np.transpose(vector, axes=np.roll(np.arange(vector.ndim), -1))
    elif isinstance(mat, torch.Tensor):
        if complex:
            mat = torch.permute(mat, list(np.roll(np.arange(mat.ndim), 1)))
        N = mat.shape[-1]
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = torch.transpose(mat[..., :upper_oddN:2, ::2], -1, -2).reshape(
            list(mat.size())[:-2] + [-1]
        )
        vector_odd_col = torch.transpose(mat[..., 1:upper_evenN:2, 1::2], -1, -2).reshape(
            list(mat.size())[:-2] + [-1]
        )
        vector = torch.cat([vector_even_col, vector_odd_col], -1)
        if complex:
            vector = torch.permute(vector, list(np.roll(np.arange(vector.ndim), -1)))
    else:
        raise NotImplementedError
    return vector


def vector_to_checkerboard(vec, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    ### from compact phase vector (even col + odd col) to clements checkerboard
    if isinstance(vec, np.ndarray):
        if complex:
            vec = np.transpose(vec, axes=np.roll(np.arange(vec.ndim), 1))
        M = vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = vec[..., : (N // 2) * ((N + 1) // 2)]
        vector_odd_col = vec[..., (N // 2) * ((N + 1) // 2) :]
        mat = np.zeros([*vec.shape[:-1], N, N], dtype=vec.dtype)
        mat[..., ::2, :upper_oddN:2] = vector_even_col.reshape([*vec.shape[:-1], (N + 1) // 2, -1])
        mat[..., 1::2, 1:upper_evenN:2] = vector_odd_col.reshape([*vec.shape[:-1], N // 2, -1])
        mat = np.swapaxes(mat, -1, -2)
        if complex:
            mat = np.transpose(mat, axes=np.roll(np.arange(mat.ndim), -1))
    elif isinstance(vec, torch.Tensor):
        if complex:
            vec = torch.permute(vec, list(np.roll(np.arange(vec.ndim), 1)))
        M = vec.size(-1)
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = vec[..., : (N // 2) * ((N + 1) // 2)]
        vector_odd_col = vec[..., (N // 2) * ((N + 1) // 2) :]
        mat = torch.zeros([*vec.shape[:-1], N, N], device=vec.device, dtype=vec.dtype)
        mat[..., ::2, :upper_oddN:2] = vector_even_col.reshape([*vec.shape[:-1], (N + 1) // 2, -1])
        mat[..., 1::2, 1:upper_evenN:2] = vector_odd_col.reshape([*vec.shape[:-1], N // 2, -1])

        mat = torch.transpose(mat, -1, -2)
        if complex:
            mat = torch.permute(mat, list(np.roll(np.arange(mat.ndim), -1)))
    else:
        raise NotImplementedError
    return mat


class ComplexToRealProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mag = torch.abs(x)
        angle = torch.angle(x)
        pos_mask = (angle <= np.pi / 2) & (angle >= -np.pi / 2)
        del angle
        neg_mask = ~pos_mask
        ctx.save_for_backward(x, neg_mask)

        x = torch.empty_like(x.real)
        x[pos_mask] = mag[pos_mask]
        x[neg_mask] = -mag[neg_mask]
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ### the gradient flow through angle is ignored
        x, neg_mask = ctx.saved_tensors()
        grad_mag = grad_output.clone()
        grad_mag[neg_mask] *= -1

        mag = torch.abs(x)
        grad_real = grad_mag * x.real / mag
        grad_imag = grad_mag * x.imag / mag
        return torch.complex(grad_real, grad_imag)


def complex_to_real_projection(x):
    if isinstance(x, np.ndarray):
        mag = np.abs(x)
        mask = x.real < 0
        mag[mask] *= -1
        x = mag
    elif isinstance(x, torch.Tensor):
        mag = x.real.square().add(x.imag.square()).add(1e-12).sqrt()
        mask = x.real < 0
        x = mag.masked_scatter(mask, -mag[mask])
    else:
        raise NotImplementedError
    return x


def project_matrix_to_unitary(W):
    if isinstance(W, np.ndarray):
        U, _, V = np.linalg.svd(W, full_matrices=True)
        U_refine = np.matmul(U, V)
    elif isinstance(W, torch.Tensor):
        U, _, V = torch.svd(W, some=False)
        U_refine = torch.matmul(U, V.transpose(-2, -1))
    else:
        raise NotImplementedError
    return U_refine


def real_matrix_parametrization_cpu(W: np.ndarray, alg: str = "clements"):
    decomposer = RealUnitaryDecomposerBatch(alg=alg)
    M, N = W.shape[0], W.shape[1]
    U, Sigma, V = np.linalg.svd(W, full_matrices=True)

    Sigma = np.diag(Sigma)
    if M > N:
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif M < N:
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decomposer.decompose(U)
    delta_list_V, phi_mat_V = decomposer.decompose(V)

    if alg == "clements":
        m2v = checkerboard_to_vector
    else:
        m2v = upper_triangle_to_vector
    phi_list_U = m2v(phi_mat_U)
    phi_list_V = m2v(phi_mat_V)

    return Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V


def real_matrix_reconstruction_cpu(
    Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V, alg: str = "clements"
):
    decomposer = RealUnitaryDecomposerBatch(alg=alg)
    if alg == "clements":
        v2m = vector_to_checkerboard
    else:
        v2m = vector_to_upper_triangle

    phi_mat_U = v2m(phi_list_U)
    phi_mat_V = v2m(phi_list_V)

    U_recon = decomposer.reconstruct(delta_list_U, phi_mat_U)
    V_recon = decomposer.reconstruct(delta_list_V, phi_mat_V)

    W_recon = np.dot(U_recon, np.dot(Sigma, V_recon))

    return W_recon


def usv(U, S, V):
    """
    description: Inverse SVD which builds matrix W from decomposed uintary matrices. Batched operation is supported\\
    U {torch.Tensor or np.ndarray} Square unitary matrix [..., M, M]\\
    S {torch.Tensor or np.ndarray} Diagonal vector [..., min(M, N)]\\
    V {torch.Tensor or np.ndarray} Square transposed unitary matrix [..., N, N]\\
    return W {torch.Tensor or np.ndarray} constructed MxN matrix [..., M, N]
    """
    if isinstance(U, torch.Tensor):
        if U.size(-1) == V.size(-1):
            W = torch.matmul(U, S.unsqueeze(-1) * V)
        elif U.size(-1) > V.size(-1):
            W = torch.matmul(U[..., : V.size(-1)], S.unsqueeze(-1) * V)
        else:
            W = torch.matmul(U * S.unsqueeze(-2), V[..., : U.size(-1), :])
    elif isinstance(U, np.ndarray):
        if U.shape[-1] == V.shape[-1]:
            W = np.matmul(U, S[..., np.newaxis] * V)
        elif U.shape[-1] > V.shape[-1]:
            W = np.matmul(U[..., : V.shape[-1]], S[..., np.newaxis] * V)
        else:
            W = np.matmul(U * S[..., np.newaxis, :], V[..., : U.shape[-1], :])
    else:
        raise NotImplementedError
    return W


class PhaseQuantizer(torch.nn.Module):
    __mode_list__ = {"rectangle", "triangle", "diagonal", "butterfly"}

    def __init__(
        self,
        bit: int,
        v_pi: float = 4.36,
        v_max: float = 10.8,
        gamma_noise_std: float = 0.0,
        crosstalk_factor: float = 0.0,
        crosstalk_filter_size: int = 5,
        random_state: Optional[int] = None,
        mode: str = "rectangle",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """2021/04/01: Uniform phase-space quantization. Support gamma noise and thermal crosstalk simulation
        Args:
            bit (int): bitwidth
            v_pi (float): Voltage corresponding to pi phase shift
            v_max (float): maximum voltage
            gamma_noise_std (float, optional): std dev of Gaussian phase noise on the gamma coefficient. Defaults to 0.
            crosstalk_factor (float): Crosstalk coefficient. Defaults to 0.
            crosstalk_filter_size (int): Conv kernel size used in crosstalk simulation. Defaults to 5.
            random_state (None or int, optional): random_state for noise injection. Defaults to None.
            mode (str): Mesh structure from (rectangle, triangle, diagonal)
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """
        super().__init__()
        self.bit = bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / v_pi ** 2
        self.gamma_noise_std = gamma_noise_std
        self.crosstalk_factor = crosstalk_factor
        self.crosstalk_filter_size = crosstalk_filter_size
        self.random_state = random_state
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(f"Only support mode in {self.__mode_list__}, but got mode: {mode}.")
        self.device = device

        self.crosstal_simulator = ThermalCrosstalkSimulator(
            plotting=False,
            filter_size=crosstalk_filter_size,
            crosstalk_factor=crosstalk_factor,
            device=self.device,
        )
        self.register_buffer("noisy_gamma", None)  # can be saved in checkpoint

    def set_gamma_noise(self, noise_std: float, size: _size, random_state: Optional[int] = None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state
        if random_state is not None:
            set_torch_deterministic(random_state)
        self.noisy_gamma = (
            torch.nn.init.trunc_normal_(torch.zeros(size, device=self.device))
            .mul_(noise_std)
            .add_(self.gamma)
        )

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.crosstal_simulator.set_crosstalk_factor(crosstalk_factor)

    def set_bitwidth(self, bit: int) -> None:
        self.bit = bit

    def forward(self, x):
        x = x % (2 * np.pi)
        if self.bit < 16:
            if self.mode in {"rectangle", "triangle", "butterfly"}:  # [0, 2pi] quantize
                ratio = 2 * np.pi / (2 ** self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            elif self.mode in {"diagonal"}:  # [0, pi] quantize
                x = torch.where(x > np.pi, 2 * np.pi - x, x)
                ratio = np.pi / (2 ** self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            else:
                raise NotImplementedError(self.mode)

        if self.noisy_gamma is not None:
            x.mul_(self.noisy_gamma.div(self.gamma))

        if self.crosstalk_factor > 1e-5:
            x = self.crosstal_simulator.simple_simulate(x, mixedtraining_mask=None, mode=self.mode)

        return x


def diagonal_quantize_function(x, bit, phase_noise_std=0, random_state=None, gradient_clip=False):
    class DiagonalQuantizeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ### support batched diagonals. input is not a matrix, but a vector which is the diagonal entries.
            S_scale = x.abs().max(dim=-1, keepdim=True)[0]
            x = (x / S_scale).acos()  # phase after acos is from [0, pi]
            ratio = np.pi / (2 ** bit - 1)
            x.div_(ratio).round_().mul_(ratio)
            if phase_noise_std > 1e-5:
                noise = gen_gaussian_noise(
                    x,
                    noise_mean=0,
                    noise_std=phase_noise_std,
                    trunc_range=[-2 * phase_noise_std, 2 * phase_noise_std],
                    random_state=random_state,
                )
                x.add_(noise)
            x.cos_().mul_(S_scale)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            if gradient_clip:
                grad_input = grad_output.clamp(-1, 1)
            else:
                grad_input = grad_output.clone()
            return grad_input

    return DiagonalQuantizeFunction.apply(x)


class DiagonalQuantizer(torch.nn.Module):
    def __init__(self, bit, phase_noise_std=0.0, random_state=None, device=torch.device("cuda")):
        """2021/02/18: New phase quantizer for Sigma matrix in MZI-ONN. Gaussian phase noise is supported. All singular values are normalized by a TIA gain (S_scale), the normalized singular values will be achieved by cos(phi), phi will have [0, pi] uniform quantization.
        We do not consider real MZI implementation, thus voltage quantization and gamma noises are not supported.
        Args:
            bit (int): bitwidth for phase quantization.
            phase_noise_std (float, optional): Std dev for Gaussian phase noises. Defaults to 0.
            random_state (int, optional): random_state to control random noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """

        super().__init__()
        self.bit = bit
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.device = device

    def set_phase_noise_std(self, phase_noise_std=0, random_state=None):
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state

    def forward(self, x):
        ### support batched diagonals. input is not a matrix, but a vector which is the diagonal entries.
        ### this function is differentiable
        x = diagonal_quantize_function(
            x, self.bit, self.phase_noise_std, self.random_state, gradient_clip=True
        )

        return x


class UnitaryQuantizer(torch.nn.Module):
    def __init__(
        self,
        bit,
        phase_noise_std=0.0,
        random_state=None,
        alg="reck",
        mode="phase",
        device=torch.device("cuda"),
    ):
        """2021/02/18: New phase quantizer for Uintary matrix in MZI-ONN. Gaussian phase noise is supported. The quantization considers the MZI implementation in [David AB Miller, Optica'20], but voltage quantization and gamma noises are not supported.
        Args:
            bit (int): bitwidth for phase quantization.
            phase_noise_std (float, optional): Std dev for Gaussian phase noises. Defaults to 0.
            random_state (int, optional): random_state to control random noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """

        super().__init__()
        self.bit = bit
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.alg = alg
        self.decomposer = RealUnitaryDecomposerBatch(alg=alg)
        self.quantizer = PhaseQuantizer(bit, device=device)
        if alg == "clements":
            self.decomposer.m2v = checkerboard_to_vector
            self.decomposer.v2m = vector_to_checkerboard
        else:
            self.decomposer.m2v = upper_triangle_to_vector
            self.decomposer.v2m = vector_to_upper_triangle
        self.device = device
        self.phase_min = (0.5 ** (bit - 2) - 0.5) * np.pi
        self.phase_max = (1.5 - 0.5 ** (bit - 1)) * np.pi
        self.phase_range = self.phase_max - self.phase_min

    def set_phase_noise_std(self, phase_noise_std=0, random_state=None):
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.quantizer.set_phase_noise_std(phase_noise_std, random_state)

    def forward(self, x):
        ### this function is not differentiable
        delta_list, x = self.decomposer.decompose(x.data.clone())
        x = self.decomposer.m2v(x)
        x = self.quantizer(x)
        x = self.decomposer.reconstruct(delta_list, self.decomposer.v2m(x))

        return x


class ThermalCrosstalkSimulator(object):
    __mode_list__ = {"rectangle", "triangle", "diagonal"}

    def __init__(
        self,
        # interval bet/ heat source (um)
        heat_source_interval: float = 8.0,
        # SetPad=0,
        grid_precision: float = 10.0,  # um
        power_density_multipier: float = 1e-3,
        # W/(um K) thermal conductivity
        thermal_conductivity: float = 1.4e-6,
        max_iter: int = 2000,  # max # of iterations
        # material options
        boundary_cond: bool = False,
        # plotting options
        plotting: bool = True,
        display_iter: int = 10,
        hold_time: float = 0.00001,
        filter_size: int = 3,
        crosstalk_factor: float = 0.01,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__()

        self.heat_source_interval = heat_source_interval
        self.grid_precision = grid_precision
        self.power_density_multiplier = power_density_multipier
        self.thermal_conductivity = thermal_conductivity
        self.max_iter = max_iter
        self.boundary_cond = boundary_cond
        self.plotting = plotting
        self.display_iter = display_iter
        self.hold_time = hold_time
        self.filter_size = filter_size
        self.crosstalk_factor = crosstalk_factor
        self.device = device
        self.power_density = None

        # self.init_phase_distribution(self.phases)
        self.init_filter(filter_size, crosstalk_factor)
        self.mixedtraining_mask = None

    def init_filter(self, filter_size: int, crosstalk_factor: float) -> None:
        c = crosstalk_factor
        if filter_size == 3:
            self.filter = torch.tensor([[0, c, 0], [c, 1, c], [0, c, 0]], device=self.device)
        elif filter_size == 5:
            self.filter = torch.tensor(
                [[0, c, 0], [c, 0, c], [0, 1, 0], [c, 0, c], [0, c, 0]], device=self.device
            )
        else:
            raise ValueError(f"Does not support filter sizes other than 3 or 5, but got {filter_size}")
        self.filter.unsqueeze_(0).unsqueeze_(0)

        self.filter_zero_center = self.filter.clone()
        self.filter_zero_center[0, 0, self.filter.size(-2) // 2, self.filter.size(-1) // 2] = 0

    def init_phase_distribution(self, phases: Tensor, dim: int) -> None:
        self.power_density = np.zeros([self.heat_source_interval * dim, self.heat_source_interval * dim])
        cnt = 0
        # for i in range(1, dim):
        #     for j in range(1, dim - i + 1):
        #         self.power_density[self.heat_source_interval*i, self.heat_source_interval*j] = phases[cnt]
        #         cnt = cnt + 1
        pointer = 0
        for i in range(1, dim):
            number_of_sources = dim - i
            interval = self.heat_source_interval
            self.power_density[interval * i, interval : number_of_sources * interval + 1 : interval] = phases[
                pointer : pointer + number_of_sources
            ]
            pointer += number_of_sources

    def simulate(self, phases: Tensor, dim: int) -> None:
        self.init_phase_distribution(phases, dim)
        # *SetSpace      # number of steps in x
        nx = self.power_density.shape[0]
        ny = self.power_density.shape[1]  # *SetSpace   # number of steps in y
        dx = self.grid_precision  # nx/(nx-1) # width of step
        dy = self.grid_precision  # ny/(ny-1) # width of step

        # Initial Conditions
        p = torch.zeros((1, 1, nx, ny)).float().to(self.device)
        power_density = (
            (
                torch.from_numpy(self.power_density.copy()).unsqueeze(0).unsqueeze(0)
                * dx
                * dx
                * dy
                * dy
                * self.thermal_conductivity
                / (2 * (dx * dx + dy * dy))
            )
            .float()
            .to(self.device)
        )
        kernel = torch.from_numpy(
            np.array([[0, dy * dy, 0], [dx * dx, 0, dx * dx], [0, dy * dy, 0]], dtype=np.float32)
        ) / (2 * (dx * dx + dy * dy))
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.zeros(nx, ny, dtype=torch.float32, device=self.device)
        for row in range(1, nx - 2):
            mask[row, 1 : ny - row - 1] = 1

        conv_err = []
        if self.plotting is True:
            plt.ion()  # continuous SetPlotting
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            x = np.linspace(dx / 2, nx - dx / 2, nx)
            y = np.linspace(dy / 2, ny - dy / 2, ny)  # orig no setspace
            X, Y = np.meshgrid(x, y)

        for it in range(self.max_iter + 1):
            # print(f"[I] iteration: {it}")
            out = torch.nn.functional.conv2d(p, kernel, padding=(1, 1))
            out.add_(power_density).mul_(mask)

            conv_err.append((it, (out - p).abs().max().data.item()))
            p = out

            if self.plotting is True and it % (self.display_iter) == 0:
                surf = ax.plot_surface(
                    X, Y, p.squeeze(0).squeeze(0).numpy(), cmap=cm.rainbow, linewidth=0, antialiased=False
                )
                # ax.set_zlim(0,80)
                # ax.set_xlim(0,0.1)
                # ax.set_ylim(0,0.1)
                plt.title("it#%d" % it, y=1)
                ax.set_xlabel("Distance (x%d um)" % (self.grid_precision))
                ax.set_ylabel("Distance (x%d um)" % (self.grid_precision))
                ax.set_zlabel("Temperature (C)")
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)

                plt.show()
                plt.pause(self.hold_time)

        return p.cpu().numpy().astype(np.float64)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.init_filter(self.filter_size, crosstalk_factor)

    def simple_simulate_triangle(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if mixedtraining_mask is None:
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(phases, filter, padding=(padding1, padding2))
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(
                phases.mul(mixedtraining_mask.float()).view(-1, 1, phases.size(-1))
            )
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2)
            )
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate_diagonal(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        return phases

    def simple_simulate_butterfly(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        phases = phases % (2 * np.pi)
        ## [n_level, k/2, 2]
        size = phases.size()

        if mixedtraining_mask is None:
            # [1, 1, n_level, k]
            phases = phases.view([1, 1] + list(size)[:-2] + [phases.size(-1) * phases.size(-2)])
            filter = self.filter
            padding = self.filter_size // 2
            phases = torch.nn.functional.conv2d(phases, filter, padding=(padding, padding))
            phases = phases.view(size)

        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # poassive devices will be influenced by active devices, but will not incluence others

            phases_active = phases * mixedtraining_mask.float()
            filter = self.filter_zero_center
            padding = self.filter_size // 2
            # influence map
            phases_active = torch.nn.functional.conv2d(
                phases_active.view([1, 1] + list(size)[:-2] + [phases.size(-1) * phases.size(-2)]),
                filter,
                padding=(padding, padding),
            )
            # add influence map and original phases together
            phases = phases_active.view_as(phases) + phases

        return phases

    def simple_simulate_rectangle(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if mixedtraining_mask is None:
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(phases, filter, padding=(padding1, padding2))
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(
                phases.mul(mixedtraining_mask.float()).view(-1, 1, phases.size(-1))
            )
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2)
            )
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate(
        self, phases: Tensor, mixedtraining_mask: Optional[Tensor] = None, mode: str = "rectangle"
    ) -> Tensor:
        assert mode in self.__mode_list__, logger.error(f"Only support mode in {self.__mode_list__}. But got mode: {mode}")
        if mode == "triangle":
            return self.simple_simulate_triangle(phases, mixedtraining_mask)
        elif mode == "rectangle":
            return self.simple_simulate_rectangle(phases, mixedtraining_mask)
        elif mode == "diagonal":
            return self.simple_simulate_diagonal(phases, mixedtraining_mask)
        elif mode == "butterfly":
            return self.simple_simulate_butterfly(phases, mixedtraining_mask)
        else:
            return phases


if __name__ == "__main__":
    pass
