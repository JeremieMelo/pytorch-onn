# coding=UTF-8
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 18:22:16
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 18:22:29
"""


from typing import List

import numpy as np
import torch
from pyutils.compute import batch_diag, batch_eye_cpu
from pyutils.general import logger, profile
from scipy.stats import ortho_group
from torch.types import Device

try:
    import matrix_parametrization_cuda
except ImportError as e:
    logger.warning(
        "Cannot import matrix_parametrization_cuda. Decomposers can only work on CPU mode"
    )

__all__ = [
    "RealUnitaryDecomposerBatch",
    "ComplexUnitaryDecomposerBatch",
]


class RealUnitaryDecomposerBatch(object):
    timer = False

    def __init__(
        self,
        min_err: float = 1e-7,
        timer: bool = False,
        determine: bool = False,
        alg: str = "reck",
        dtype=np.float64,
    ) -> None:
        self.min_err = min_err
        self.timer = timer
        self.determine = determine
        assert alg.lower() in {"reck", "clements", "francis"}, logger.error(
            f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}"
        )
        self.set_alg(alg)
        self.dtype = dtype

    def set_alg(self, alg):
        assert alg.lower() in {"reck", "clements", "francis"}, logger.error(
            f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}"
        )
        self.alg = alg

    def build_plane_unitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(N, int), "[E] Matrix size must be positive integer"
        assert (
            isinstance(p, int) and isinstance(q, int) and 0 <= p < q < N
        ), "[E] Integer value p and q must satisfy p < q"
        assert isinstance(phi, float) or isinstance(
            phi, int
        ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def cal_phi_batch_determine(
        self, u1: np.ndarray, u2: np.ndarray, is_first_col=False
    ) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        if is_first_col:
            phi = np.where(
                cond1 & cond2,
                0,
                np.where(
                    cond1_n & cond2,
                    np.where(u1 > min_err, 0, -pi),
                    np.where(
                        cond1 & cond2_n,
                        np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                        np.arctan2(-u2, u1),
                    ),
                ),
            )
        else:
            phi = np.where(
                cond1 & cond2,
                0,
                np.where(
                    cond1_n & cond2,
                    np.where(u1 > min_err, 0, -pi),
                    np.where(
                        cond1 & cond2_n,
                        np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                        np.arctan(-u2 / u1),
                    ),
                ),
            )
        return phi

    def cal_phi_batch_nondetermine(
        self, u1: np.ndarray, u2: np.ndarray, is_first_col=False
    ) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        phi = np.where(
            cond1 & cond2,
            0,
            np.where(
                cond1_n & cond2,
                np.where(u1 > min_err, 0, -pi),
                np.where(
                    cond1 & cond2_n,
                    np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                    np.arctan2(-u2, u1),
                ),
            ),
        )

        return phi

    def cal_phi_determine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            if is_first_col:
                phi = np.arctan2(-u2, u1)  # 4 quadrant4
            else:
                phi = np.arctan(-u2 / u1)

        return phi

    def cal_phi_nondetermine(self, u1, u2):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant4

        return phi

    def decompose_kernel_batch(self, U: np.ndarray, dim, phi_list=None):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[-1]
        if phi_list is None:
            phi_list = np.zeros(list(U.shape[:-2]) + [dim], dtype=np.float64)

        calPhi_batch = (
            self.cal_phi_batch_determine
            if self.determine
            else self.cal_phi_batch_nondetermine
        )
        for i in range(N - 1):

            u1, u2 = U[..., 0, 0], U[..., 0, N - 1 - i]
            phi = calPhi_batch(u1, u2, is_first_col=(i == 0))

            phi_list[..., i] = phi

            p, q = 0, N - i - 1
            c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
            col_p, col_q = U[..., :, p], U[..., :, q]
            U[..., :, p], U[..., :, q] = col_p * c - col_q * s, col_p * s + col_q * c

        return U, phi_list

    def decompose_kernel_determine(self, U, phi_list):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[0]

        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            pi = np.pi
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            min_err = self.min_err

            if u1_abs < min_err and u2_abs < min_err:
                phi = 0
            elif u1_abs >= min_err and u2_abs < min_err:
                phi = 0 if u1 > min_err else -pi
            elif u1_abs < min_err and u2_abs >= min_err:
                phi = -0.5 * pi if u2 > min_err else 0.5 * pi
            else:
                # solve the equation: u'_1n=0
                if i == 0:
                    phi = np.arctan2(-u2, u1)  # 4 quadrant4
                else:
                    phi = np.arctan(-u2 / u1)

            phi_list[i] = phi
            p, q = 0, N - i - 1
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin

        return U, phi_list

    def decompose_kernel_nondetermine(self, U, phi_list):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[0]

        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
            if cond1 & cond2:
                phi = np.arctan2(-u2, u1)
            elif ~cond1 & cond2:
                phi = -half_pi if u2 > min_err else half_pi
            elif cond1 & ~cond2:
                phi = 0 if u1 > min_err else -pi
            else:
                phi = 0

            phi_list[i] = phi
            p, q = 0, N - i - 1
            c = np.cos(phi)
            s = (1 - c * c) ** 0.5 if phi > 0 else -((1 - c * c) ** 0.5)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin

        return U, phi_list

    @profile(timer=timer)
    def decompose_francis_cpu(self, U):
        #### This decomposition has follows the natural reflection of MZIs. Thus the circuit will give a reversed output.
        ### Francis style, 1962
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros([N, N], dtype=self.dtype)
        delta_list = np.zeros(N, dtype=self.dtype)
        decompose_kernel = (
            self.decompose_kernel_determine
            if self.determine
            else self.decompose_kernel_nondetermine
        )

        for i in range(N - 1):
            U, _ = decompose_kernel(U, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_francis_batch(self, U: np.ndarray):
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)

        for i in range(N - 1):
            U, _ = self.decompose_kernel_batch(U, dim=N, phi_list=phi_mat[..., i, :])
            delta_list[..., i] = U[..., 0, 0]
            U = U[..., 1:, 1:]
        else:
            delta_list[..., -1] = U[..., -1, -1]

        return delta_list, phi_mat

    def decompose_francis(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_francis_cpu(U)
            else:
                return self.decompose_francis_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_francis(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.ndim == 2:
                    return torch.from_numpy(self.decompose_francis_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(
                        self.decompose_francis_batch(U.cpu().numpy())
                    )

    @profile(timer=timer)
    def decompose_reck_cpu(self, U):
        """Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        """
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros([N, N], dtype=self.dtype)  ## left upper triangular array.
        """
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        """

        delta_list = np.zeros(N, dtype=self.dtype)  ## D
        """
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        """

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j  ## row
                q = N - 1 - i + j  ## col
                ### rotate two columns such that u2 is nullified to 0
                pi = np.pi
                half_pi = np.pi / 2
                min_err = self.min_err
                ### col q-1 nullifies col q
                u1, u2 = U[p, q - 1], U[p, q]
                u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                if cond1 & cond2:
                    phi = np.arctan2(-u2, u1)
                elif ~cond1 & cond2:
                    phi = -half_pi if u2 > min_err else half_pi
                elif cond1 & ~cond2:
                    phi = 0 if u1 > min_err else -pi
                else:
                    phi = 0
                # phi_mat[p,q] = phi
                # theta_checkerboard[pairwise_index, -j - 1] = phi
                phi_mat[N - i - 2, j] = phi
                c, s = np.cos(phi), np.sin(phi)
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[p:, q - 1], U[p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[p:, q - 1], U[p:, q] = (
                    col_q_m1_cos - col_q_sin,
                    col_q_cos + col_q_m1_sin,
                )

        delta_list = np.diag(
            U
        )  ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_reck_batch(self, U):
        """Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        """
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(U.shape, dtype=self.dtype)  ## left upper triangular array.
        """
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        """

        delta_list = np.zeros(U.shape[:-1], dtype=self.dtype)  ## D
        """
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        """

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j  ## row
                q = N - 1 - i + j  ## col
                ### rotate two columns such that u2 is nullified to 0
                ### col q-1 nullifies col q

                u1, u2 = U[..., p, q - 1], U[..., p, q]
                phi = self.cal_phi_batch_nondetermine(u1, u2)

                phi_mat[..., N - i - 2, j] = phi
                c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[..., p:, q - 1], U[..., p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[..., p:, q - 1], U[..., p:, q] = (
                    col_q_m1_cos - col_q_sin,
                    col_q_cos + col_q_m1_sin,
                )

        delta_list = batch_diag(U)

        return delta_list, phi_mat

    def decompose_reck(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_reck_cpu(U)
            else:
                return self.decompose_reck_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_reck(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.ndim == 2:
                    return torch.from_numpy(self.decompose_reck_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(self.decompose_reck_batch(U.cpu().numpy()))

    @profile(timer=timer)
    def decompose_clements_cpu(self, U):
        """clements Optica 2018 unitary decomposition
        Tmn: [e^iphi x cos(theta)   -sin(theta)]
             [e^iphi x sin(theta)    cos(theta)]
        phi  DC   2 theta  DC ---
        ---  DC   -------  DC ---
        T45 T34 T23 T12 T45 T34 U T12* T34* T23* T12 = D
        U=D T34 T45 T12 T23 T34 T45 T12 T23 T34 T12"""
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(
            [N, N], dtype=self.dtype
        )  ## theta checkerboard that maps to the real MZI mesh layout, which is efficient for parallel reconstruction col-by-col.

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if i % 2 == 0:
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j  ## row
                    q = i - j  ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p, q + 1], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if cond1 & cond2:
                        phi = np.arctan2(-u2, u1)
                    elif ~cond1 & cond2:
                        phi = -half_pi if u2 > min_err else half_pi
                    elif cond1 & ~cond2:
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = (
                        -phi
                    )  ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    phi_mat[pairwise_index, j] = phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[: p + 1, q + 1], U[: p + 1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[: p + 1, q + 1], U[: p + 1, q] = (
                        col_q_p1_cos + col_q_sin,
                        col_q_cos - col_q_p1_sin,
                    )
            else:
                ## odd loop for row rotation
                for j in range(i + 1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p - 1, q], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if cond1 & cond2:
                        phi = np.arctan2(-u2, u1)
                    elif ~cond1 & cond2:
                        phi = -half_pi if u2 > min_err else half_pi
                    elif cond1 & ~cond2:
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    # phi_mat[p,q] = phi

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    phi_mat[pairwise_index, N - 1 - j] = (
                        -phi
                    )  ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[p - 1, j:], U[p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[p - 1, j:], U[p, j:] = (
                        row_p_1_cos - row_p_sin,
                        row_p_cos + row_p_1_sin,
                    )

        delta_list = np.diag(
            U
        )  ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction
        delta_list.setflags(write=True)

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_clements_batch(self, U):
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)
        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if i % 2 == 0:
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j  ## row
                    q = i - j  ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    # half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[..., p : p + 1, q + 1], U[..., p : p + 1, q]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(
                        cond1 & cond2,
                        0,
                        np.where(
                            cond1_n & cond2,
                            np.where(u1 > min_err, 0, -pi),
                            np.where(
                                cond1 & cond2_n,
                                np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                                np.arctan2(-u2, u1),
                            ),
                        ),
                    )
                    phi = (
                        -phi
                    )  ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    phi_mat[..., pairwise_index, j] = phi[..., 0]
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[..., : p + 1, q + 1], U[..., : p + 1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[..., : p + 1, q + 1], U[..., : p + 1, q] = (
                        col_q_p1_cos + col_q_sin,
                        col_q_cos - col_q_p1_sin,
                    )
            else:
                ## odd loop for row rotation
                for j in range(i + 1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    min_err = self.min_err
                    u1, u2 = U[..., p - 1, q : q + 1], U[..., p, q : q + 1]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(
                        cond1 & cond2,
                        0,
                        np.where(
                            cond1_n & cond2,
                            np.where(u1 > min_err, 0, -pi),
                            np.where(
                                cond1 & cond2_n,
                                np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                                np.arctan2(-u2, u1),
                            ),
                        ),
                    )

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    phi_mat[..., pairwise_index, N - 1 - j] = -phi[
                        ..., 0
                    ]  ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[..., p - 1, j:], U[..., p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[..., p - 1, j:], U[..., p, j:] = (
                        row_p_1_cos - row_p_sin,
                        row_p_cos + row_p_1_sin,
                    )

        delta_list = batch_diag(U)
        return delta_list, phi_mat

    def decompose_clements(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_clements_cpu(U)
            else:
                return self.decompose_clements_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_clements(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.ndim == 2:
                    return torch.from_numpy(
                        self.decompose_clements_cpu(U.cpu().numpy())
                    )
                else:
                    return torch.from_numpy(
                        self.decompose_clements_batch(U.cpu().numpy())
                    )

    def decompose(self, U):
        if self.alg == "reck":
            decompose_cpu = self.decompose_reck_cpu
            decompose_batch = self.decompose_reck_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_reck
        elif self.alg == "francis":
            decompose_cpu = self.decompose_francis_cpu
            decompose_batch = self.decompose_francis_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_francis
        elif self.alg == "clements":
            decompose_cpu = self.decompose_clements_cpu
            decompose_batch = self.decompose_clements_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_clements
        else:
            raise NotImplementedError

        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return decompose_cpu(U)
            else:
                return decompose_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                decompose_cuda(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.dim() == 2:
                    delta_list, phi_mat = decompose_cpu(U.cpu().numpy())
                else:
                    delta_list, phi_mat = decompose_batch(U.cpu().numpy())
                return torch.from_numpy(delta_list), torch.from_numpy(phi_mat)

    @profile(timer=timer)
    def reconstruct_francis_cpu(self, delta_list, phi_mat):
        ### Francis style, 1962
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        ### cannot gaurantee the phase range, so this will be slower

        for i in range(N):
            for j in range(N - i - 1):

                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                p = i
                q = N - j - 1
                row_p, row_q = Ur[p, :], Ur[q, :]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, :], Ur[q, :] = row_p_cos - row_q_sin, row_p_sin + row_q_cos

        Ur = delta_list[:, np.newaxis] * Ur

        return Ur

    @profile(timer=timer)
    def reconstruct_francis_batch(
        self, delta_list: np.ndarray, phi_mat: np.ndarray
    ) -> np.ndarray:
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)

        # reconstruct from right to left as in the book chapter
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                c, s = phi_mat_cos[..., i, j : j + 1], phi_mat_sin[..., i, j : j + 1]

                p = i
                q = N - j - 1
                Ur[..., p, :], Ur[..., q, :] = (
                    Ur[..., p, :] * c - Ur[..., q, :] * s,
                    Ur[..., p, :] * s + Ur[..., q, :] * c,
                )

        Ur = delta_list[..., np.newaxis] * Ur

        return Ur

    def reconstruct_francis(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_francis_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_francis_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_francis(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_francis(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_francis_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    @profile(timer=timer)
    def reconstruct_reck_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)
        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ## totally 2n-3 stage
        for i in range(N - 1):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = phi_mat_cos[lower, j], phi_mat_sin[lower, j]
                p = N - 2 - i + j
                q = p + 1

                row_p, row_q = Ur[p, lower:], Ur[q, lower:]
                res = (c + 1j * s) * (row_p + 1j * row_q)
                Ur[p, lower:], Ur[q, lower:] = res.real, res.imag
        Ur = delta_list[:, np.newaxis] * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_reck_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        for i in range(N - 1):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = (
                    phi_mat_cos[..., lower, j : j + 1],
                    phi_mat_sin[..., lower, j : j + 1],
                )
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                ### this rotation is equivalent to complex number multiplication as an acceleration.
                res = (c + 1j * s) * (row_p + 1j * row_q)
                Ur[..., p, lower:], Ur[..., q, lower:] = res.real, res.imag
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_reck_batch_par(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ### 2n-3 stages
        for i in range(2 * N - 3):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = (
                    phi_mat_cos[..., lower, j : j + 1],
                    phi_mat_sin[..., lower, j : j + 1],
                )
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = (
                    row_p_cos - row_q_sin,
                    row_p_sin + row_q_cos,
                )
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    def reconstruct_reck(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_reck_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_reck_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_reck(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_reck(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_reck_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    @profile(timer=timer)
    def reconstruct_clements_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        for i in range(N):  ## N layers
            max_len = 2 * (i + 1)
            ### in odd N, address delta_list[-1] before the first column
            if i == 0 and N % 2 == 1 and delta_list[-1] < 0:
                Ur[-1, :] *= -1
            for j in range((i % 2), N - 1, 2):
                c, s = phi_mat_cos[j, i], phi_mat_sin[j, i]
                ## not the entire row needs to be rotated, only a small working set is used
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                row_p, row_q = Ur[j, lower:upper], Ur[j + 1, lower:upper]
                res = (c + 1j * s) * (row_p + 1j * row_q)
                Ur[j, lower:upper], Ur[j + 1, lower:upper] = res.real, res.imag
            if i == 0 and N % 2 == 0 and delta_list[-1] < 0:
                Ur[-1, :] *= -1
            if (
                i == N - 2 and N % 2 == 1 and delta_list[0] < 0
            ):  ## consider diagonal[0]= {-1,1} before the last layer when N odd
                Ur[0, :] *= -1
        if (
            N % 2 == 0 and delta_list[0] < 0
        ):  ## consider diagonal[0]= {-1,1} after the last layer when N even
            Ur[0, :] *= -1
        return Ur

    @profile(timer=timer)
    def reconstruct_clements_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):  ## N layers
            max_len = 2 * (i + 1)
            ### in odd N, address delta_list[-1] before the first column
            if i == 0 and N % 2 == 1:
                Ur[..., -1, :] *= delta_list[..., -1:]
            for j in range((i % 2), N - 1, 2):
                ## not the entire row needs to be rotated, only a small working set is used
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                c, s = phi_mat_cos[..., j, i : i + 1], phi_mat_sin[..., j, i : i + 1]
                row_p, row_q = Ur[..., j, lower:upper], Ur[..., j + 1, lower:upper]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., j, lower:upper], Ur[..., j + 1, lower:upper] = (
                    row_p_cos - row_q_sin,
                    row_p_sin + row_q_cos,
                )

            if i == 0 and N % 2 == 0:
                Ur[..., -1, :] *= delta_list[..., -1:]
            if (
                i == N - 2 and N % 2 == 1
            ):  ## consider diagonal[0]= {-1,1} before the last layer when N odd
                Ur[..., 0, :] *= delta_list[..., 0:1]
        if N % 2 == 0:  ## consider diagonal[0]= {-1,1} after the last layer when N even
            Ur[..., 0, :] *= delta_list[..., 0:1]

        return Ur

    def reconstruct_clements(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_clements_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_clements_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_clements(
                    delta_list, phi_mat
                )

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_clements(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_clements_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    def reconstruct(self, delta_list, phi_mat):
        if self.alg == "francis":
            reconstruct_cpu = self.reconstruct_francis
            reconstruct_batch = self.reconstruct_francis_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_francis
        elif self.alg == "reck":
            reconstruct_cpu = self.reconstruct_reck_cpu
            reconstruct_batch = self.reconstruct_reck_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_reck
        elif self.alg == "clements":
            reconstruct_cpu = self.reconstruct_clements_cpu
            reconstruct_batch = self.reconstruct_clements_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_clements
        else:
            raise NotImplementedError

        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return reconstruct_cpu(delta_list, phi_mat)
            else:
                return reconstruct_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = reconstruct_cuda(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.ndim == 2:
                    return torch.from_numpy(
                        reconstruct_cpu(delta_list.cpu().numpy(), phi_mat.cpu().numpy())
                    )
                else:
                    return torch.from_numpy(
                        reconstruct_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    def check_identity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def check_unitary(self, U):
        M = np.dot(U, U.T)
        return self.check_identity(M)

    def check_equal(self, M1, M2):
        return (M1.shape == M2.shape) and np.allclose(M1, M2)

    def gen_random_ortho(self, N):
        U = ortho_group.rvs(N)
        logger.info(
            f"Generate random {N}*{N} unitary matrix, check unitary: {self.check_unitary(U)}"
        )
        return U

    def to_degree(self, M):
        return np.degrees(M)


class ComplexUnitaryDecomposerBatch(object):
    timer = False

    def __init__(
        self,
        min_err: float = 1e-7,
        timer: bool = False,
        determine: bool = False,
        alg: str = "reck",
        dtype=np.float64,
    ) -> None:
        self.min_err = min_err
        self.timer = timer
        self.determine = determine
        self.alg = alg
        assert alg.lower() in {"reck", "clements", "francis"}, logger.error(
            f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}"
        )
        self.dtype = dtype

    def set_alg(self, alg):
        assert alg.lower() in {"reck", "clements", "francis"}, logger.error(
            f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}"
        )
        self.alg = alg

    def build_plane_unitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(N, int), "[E] Matrix size must be positive integer"
        assert (
            isinstance(p, int) and isinstance(q, int) and 0 <= p < q < N
        ), "[E] Integer value p and q must satisfy p < q"
        assert isinstance(phi, float) or isinstance(
            phi, int
        ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def cal_phi_batch_determine(
        self, u1: np.ndarray, u2: np.ndarray, is_first_col=False
    ) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        if is_first_col:
            phi = np.where(
                cond1 & cond2,
                0,
                np.where(
                    cond1_n & cond2,
                    np.where(u1 > min_err, 0, -pi),
                    np.where(
                        cond1 & cond2_n,
                        np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                        np.arctan2(-u2, u1),
                    ),
                ),
            )
        else:
            phi = np.where(
                cond1 & cond2,
                0,
                np.where(
                    cond1_n & cond2,
                    np.where(u1 > min_err, 0, -pi),
                    np.where(
                        cond1 & cond2_n,
                        np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                        np.arctan(-u2 / u1),
                    ),
                ),
            )
        return phi

    def cal_phi_batch_nondetermine(
        self, u1: np.ndarray, u2: np.ndarray, is_first_col=False
    ) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        phi = np.where(
            cond1 & cond2,
            0,
            np.where(
                cond1_n & cond2,
                np.where(u1 > min_err, 0, -pi),
                np.where(
                    cond1 & cond2_n,
                    np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                    np.arctan2(-u2, u1),
                ),
            ),
        )

        return phi

    def cal_phi_determine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            if is_first_col:
                phi = np.arctan2(-u2, u1)  # 4 quadrant4
            else:
                phi = np.arctan(-u2 / u1)

        return phi

    def cal_phi_nondetermine(self, u1, u2):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant4

        return phi

    def decompose_kernel_batch(self, U: np.ndarray, dim, phi_list=None):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[-1]
        if phi_list is None:
            phi_list = np.zeros(list(U.shape[:-2]) + [dim], dtype=np.float64)

        calPhi_batch = (
            self.cal_phi_batch_determine
            if self.determine
            else self.cal_phi_batch_nondetermine
        )
        for i in range(N - 1):
            u1, u2 = U[..., 0, 0], U[..., 0, N - 1 - i]
            phi = calPhi_batch(u1, u2, is_first_col=(i == 0))

            phi_list[..., i] = phi

            p, q = 0, N - i - 1
            c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
            col_p, col_q = U[..., :, p], U[..., :, q]
            U[..., :, p], U[..., :, q] = col_p * c - col_q * s, col_p * s + col_q * c

        return U, phi_list

    def decompose_kernel_determine(self, U, phi_list):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[0]

        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            pi = np.pi
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            min_err = self.min_err

            if u1_abs < min_err and u2_abs < min_err:
                phi = 0
            elif u1_abs >= min_err and u2_abs < min_err:
                phi = 0 if u1 > min_err else -pi
            elif u1_abs < min_err and u2_abs >= min_err:
                phi = -0.5 * pi if u2 > min_err else 0.5 * pi
            else:
                # solve the equation: u'_1n=0
                if i == 0:
                    phi = np.arctan2(-u2, u1)  # 4 quadrant4
                else:
                    phi = np.arctan(-u2 / u1)

            phi_list[i] = phi
            p, q = 0, N - i - 1
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin

        return U, phi_list

    def decompose_kernel_nondetermine(self, U, phi_list):
        """return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)"""
        N = U.shape[0]
        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err
        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
            if cond1 & cond2:
                phi = np.arctan2(-u2, u1)
            elif ~cond1 & cond2:
                phi = -half_pi if u2 > min_err else half_pi
            elif cond1 & ~cond2:
                phi = 0 if u1 > min_err else -pi
            else:
                phi = 0

            phi_list[i] = phi
            p, q = 0, N - i - 1
            c = np.cos(phi)
            s = (1 - c * c) ** 0.5 if phi > 0 else -((1 - c * c) ** 0.5)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin

        return U, phi_list

    @profile(timer=timer)
    def decompose_francis_cpu(self, U):
        #### This decomposition has follows the natural reflection of MZIs. Thus the circuit will give a reversed output.
        ### Francis style, 1962
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros([N, N], dtype=self.dtype)
        delta_list = np.zeros(N, dtype=self.dtype)
        decompose_kernel = (
            self.decompose_kernel_determine
            if self.determine
            else self.decompose_kernel_nondetermine
        )

        for i in range(N - 1):
            U, _ = decompose_kernel(U, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_francis_batch(self, U: np.ndarray):
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)

        for i in range(N - 1):
            U, _ = self.decompose_kernel_batch(U, dim=N, phi_list=phi_mat[..., i, :])
            delta_list[..., i] = U[..., 0, 0]
            U = U[..., 1:, 1:]
        else:
            delta_list[..., -1] = U[..., -1, -1]

        return delta_list, phi_mat

    def decompose_francis(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_francis_cpu(U)
            else:
                return self.decompose_francis_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_francis(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.dim() == 2:
                    return torch.from_numpy(self.decompose_francis_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(
                        self.decompose_francis_batch(U.cpu().numpy())
                    )

    @profile(timer=timer)
    def decompose_reck_cpu(self, U):
        """Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        """
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(
            [N, N, 4], dtype=self.dtype
        )  ## phase shifter, theta_t, theta_l, omega_p, omega_w

        """
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        """

        """
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        """

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j  ## row
                q = N - 1 - i + j  ## col
                ### rotate two columns such that u2 is nullified to 0
                pi = np.pi
                half_pi = np.pi / 2
                min_err = self.min_err
                ### col q-1 nullifies col q
                u1, u2 = U[p, q - 1], U[p, q]
                u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                if cond1 & cond2:
                    phi = np.arctan2(-u2, u1)
                elif ~cond1 & cond2:
                    phi = -half_pi if u2 > min_err else half_pi
                elif cond1 & ~cond2:
                    phi = 0 if u1 > min_err else -pi
                else:
                    phi = 0
                if phi < -pi / 2:
                    phi += 2 * np.pi  ## [-pi/2, 3pi/2]

                phi_mat[N - i - 2, j, 0] = (
                    np.pi / 2
                )  ## this absorbs the global phase theta_tot
                phi_mat[N - i - 2, j, 1] = 3 * np.pi / 2
                phi_mat[N - i - 2, j, 2] = 1.5 * np.pi - phi
                phi_mat[N - i - 2, j, 3] = 0.5 * np.pi + phi
                c, s = np.cos(phi), np.sin(phi)
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[p:, q - 1], U[p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[p:, q - 1], U[p:, q] = (
                    col_q_m1_cos - col_q_sin,
                    col_q_cos + col_q_m1_sin,
                )

        delta_list = np.angle(
            np.diag(U)
        )  ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_reck_batch(self, U):
        """Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        U is real matrix
        """
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(
            list(U.shape) + [4], dtype=self.dtype
        )  ## left upper triangular array.
        """
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        """

        """
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        """

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j  ## row
                q = N - 1 - i + j  ## col
                ### rotate two columns such that u2 is nullified to 0
                ### col q-1 nullifies col q

                u1, u2 = U[..., p, q - 1], U[..., p, q]
                phi = self.cal_phi_batch_nondetermine(u1, u2)
                phi[phi < -np.pi / 2] += 2 * np.pi  ## [-pi/2, 3pi/2]

                phi_mat[..., N - i - 2, j, 0] = (
                    np.pi / 2
                )  ## this absorbs the global phase theta_tot
                phi_mat[..., N - i - 2, j, 1] = 3 * np.pi / 2
                phi_mat[..., N - i - 2, j, 2] = 1.5 * np.pi - phi
                phi_mat[..., N - i - 2, j, 3] = 0.5 * np.pi + phi
                c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[..., p:, q - 1], U[..., p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[..., p:, q - 1], U[..., p:, q] = (
                    col_q_m1_cos - col_q_sin,
                    col_q_cos + col_q_m1_sin,
                )

        delta_list = np.angle(batch_diag(U))

        return delta_list, phi_mat

    def decompose_reck(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_reck_cpu(U)
            else:
                return self.decompose_reck_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_reck(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.dim() == 2:
                    return torch.from_numpy(self.decompose_reck_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(self.decompose_reck_batch(U.cpu().numpy()))

    @profile(timer=timer)
    def decompose_clements_cpu(self, U):
        """clements Optica 2018 unitary decomposition
        Tmn: [e^iphi x cos(theta)   -sin(theta)]
             [e^iphi x sin(theta)    cos(theta)]
        phi  DC   2 theta  DC ---
        ---  DC   -------  DC ---
        T45 T34 T23 T12 T45 T34 U T12* T34* T23* T12 = D
        U=D T34 T45 T12 T23 T34 T45 T12 T23 T34 T12"""
        N = U.shape[0]
        assert (
            N > 0 and U.shape[0] == U.shape[1]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(
            [N, N, 4], dtype=self.dtype
        )  ## theta checkerboard that maps to the real MZI mesh layout, which is efficient for parallel reconstruction col-by-col.

        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err

        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if i % 2 == 0:
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j  ## row
                    q = i - j  ## col
                    ### rotate two columns such that u2 is nullified to 0
                    u1, u2 = U[p, q + 1], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if cond1 & cond2:
                        phi = np.arctan2(-u2, u1)
                    elif ~cond1 & cond2:
                        phi = -half_pi if u2 > min_err else half_pi
                    elif cond1 & ~cond2:
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = (
                        -phi
                    )  ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    if phi < -pi / 2:
                        phi += 2 * pi
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    # phi_mat[pairwise_index, j] = phi
                    phi_mat[pairwise_index, j, 0] = (
                        np.pi / 2
                    )  ## this absorbs the global phase theta_tot
                    phi_mat[pairwise_index, j, 1] = 3 * np.pi / 2
                    phi_mat[pairwise_index, j, 2] = 1.5 * np.pi - phi
                    phi_mat[pairwise_index, j, 3] = 0.5 * np.pi + phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[: p + 1, q + 1], U[: p + 1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[: p + 1, q + 1], U[: p + 1, q] = (
                        col_q_p1_cos + col_q_sin,
                        col_q_cos - col_q_p1_sin,
                    )
            else:
                ## odd loop for row rotation
                for j in range(i + 1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p - 1, q], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if cond1 & cond2:
                        phi = np.arctan2(-u2, u1)
                    elif ~cond1 & cond2:
                        phi = -half_pi if u2 > min_err else half_pi
                    elif cond1 & ~cond2:
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = -phi
                    if phi < -pi / 2:
                        phi += 2 * pi

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    # phi_mat[pairwise_index, N - 1 - j] = phi ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    phi_mat[pairwise_index, N - 1 - j, 0] = (
                        np.pi / 2
                    )  ## this absorbs the global phase theta_tot
                    phi_mat[pairwise_index, N - 1 - j, 1] = 3 * np.pi / 2
                    phi_mat[pairwise_index, N - 1 - j, 2] = 1.5 * np.pi - phi
                    phi_mat[pairwise_index, N - 1 - j, 3] = 0.5 * np.pi + phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[p - 1, j:], U[p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[p - 1, j:], U[p, j:] = (
                        row_p_1_cos + row_p_sin,
                        row_p_cos - row_p_1_sin,
                    )
        delta_list = np.angle(np.diag(U))

        ### efficiently absorb delta_list into theta_t and theta_l and move delta_list to the last phase shifter column
        ### since U is real matrix, only delta_list[0] and delta_list[-1] can be -1.
        if N % 2 == 1:
            phi_mat[0, -1, 0] += delta_list[0]
            delta_list[0] = 0
            phi_mat[N - 2, 1, 1] += delta_list[-1]
            delta_list[-1] = 0
        else:
            phi_mat[N - 2, 2, 1] += delta_list[-1]
            delta_list[-1] = 0

        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_clements_batch(self, U):
        N = U.shape[-1]
        assert (
            N > 0 and U.shape[-1] == U.shape[-2]
        ), "[E] Input matrix must be square and N > 0"

        phi_mat = np.zeros(list(U.shape) + [4], dtype=np.float64)
        for i in range(N - 1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if i % 2 == 0:
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j  ## row
                    q = i - j  ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    min_err = self.min_err
                    u1, u2 = U[..., p : p + 1, q + 1], U[..., p : p + 1, q]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(
                        cond1 & cond2,
                        0,
                        np.where(
                            cond1_n & cond2,
                            np.where(u1 > min_err, 0, -pi),
                            np.where(
                                cond1 & cond2_n,
                                np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                                np.arctan2(-u2, u1),
                            ),
                        ),
                    )
                    phi = (
                        -phi
                    )  ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    phi[phi < -pi / 2] += 2 * pi
                    pairwise_index = i - j
                    # phi_mat[pairwise_index, j] = phi
                    phi_mat[..., pairwise_index, j, 0] = (
                        np.pi / 2
                    )  ## this absorbs the global phase theta_tot
                    phi_mat[..., pairwise_index, j, 1] = 3 * np.pi / 2
                    phi_mat[..., pairwise_index, j, 2] = 1.5 * np.pi - phi[..., 0]
                    phi_mat[..., pairwise_index, j, 3] = 0.5 * np.pi + phi[..., 0]

                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[..., : p + 1, q + 1], U[..., : p + 1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[..., : p + 1, q + 1], U[..., : p + 1, q] = (
                        col_q_p1_cos + col_q_sin,
                        col_q_cos - col_q_p1_sin,
                    )
            else:
                ## odd loop for row rotation
                for j in range(i + 1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    min_err = self.min_err
                    u1, u2 = U[..., p - 1, q : q + 1], U[..., p, q : q + 1]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(
                        cond1 & cond2,
                        0,
                        np.where(
                            cond1_n & cond2,
                            np.where(u1 > min_err, 0, -pi),
                            np.where(
                                cond1 & cond2_n,
                                np.where(u2 > min_err, -0.5 * pi, 0.5 * pi),
                                np.arctan2(-u2, u1),
                            ),
                        ),
                    )

                    phi = (
                        -phi
                    )  ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    phi[phi < -pi / 2] += 2 * pi
                    pairwise_index = N + j - i - 2

                    phi_mat[..., pairwise_index, N - 1 - j, 0] = (
                        np.pi / 2
                    )  ## this absorbs the global phase theta_tot
                    phi_mat[..., pairwise_index, N - 1 - j, 1] = 3 * np.pi / 2
                    phi_mat[..., pairwise_index, N - 1 - j, 2] = (
                        1.5 * np.pi - phi[..., 0]
                    )
                    phi_mat[..., pairwise_index, N - 1 - j, 3] = (
                        0.5 * np.pi + phi[..., 0]
                    )

                    # phi_mat[..., pairwise_index, N - 1 - j] = -phi[..., 0] ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[..., p - 1, j:], U[..., p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[..., p - 1, j:], U[..., p, j:] = (
                        row_p_1_cos + row_p_sin,
                        row_p_cos - row_p_1_sin,
                    )

        delta_list = np.angle(batch_diag(U))
        ### efficiently absorb delta_list into theta_t and theta_l and move delta_list to the last phase shifter column
        ### since U is real matrix, only delta_list[0] and delta_list[-1] can be -1.
        if N % 2 == 1:
            phi_mat[..., 0, -1, 0] += delta_list[..., 0]
            delta_list[..., 0] = 0
            phi_mat[..., N - 2, 1, 1] += delta_list[..., -1]
            delta_list[..., -1] = 0
        else:
            phi_mat[..., N - 2, 2, 1] += delta_list[..., -1]
            delta_list[..., -1] = 0
        return delta_list, phi_mat

    def decompose_clements(self, U):
        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return self.decompose_clements_cpu(U)
            else:
                return self.decompose_clements_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_clements(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.dim() == 2:
                    return torch.from_numpy(
                        self.decompose_clements_cpu(U.cpu().numpy())
                    )
                else:
                    return torch.from_numpy(
                        self.decompose_clements_batch(U.cpu().numpy())
                    )

    def decompose(self, U):
        if self.alg == "reck":
            decompose_cpu = self.decompose_reck_cpu
            decompose_batch = self.decompose_reck_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_reck
        elif self.alg == "francis":
            decompose_cpu = self.decompose_francis_cpu
            decompose_batch = self.decompose_francis_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_francis
        elif self.alg == "clements":
            decompose_cpu = self.decompose_clements_cpu
            decompose_batch = self.decompose_clements_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_clements
        else:
            raise NotImplementedError

        if isinstance(U, np.ndarray):
            if len(U.shape) == 2:
                return decompose_cpu(U)
            else:
                return decompose_batch(U)
        else:
            if U.is_cuda:
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(
                    list(U.size())[:-1], dtype=U.dtype, device=U.device
                ).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                decompose_cuda(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if U.dim() == 2:
                    return torch.from_numpy(decompose_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(decompose_batch(U.cpu().numpy()))

    @profile(timer=timer)
    def reconstruct_francis_cpu(self, delta_list, phi_mat):
        ### Francis style, 1962
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                p = i
                q = N - j - 1
                row_p, row_q = Ur[p, :], Ur[q, :]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, :], Ur[q, :] = row_p_cos - row_q_sin, row_p_sin + row_q_cos

        Ur = delta_list[:, np.newaxis] * Ur

        return Ur

    @profile(timer=timer)
    def reconstruct_francis_batch(
        self, delta_list: np.ndarray, phi_mat: np.ndarray
    ) -> np.ndarray:
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)

        # reconstruct from right to left as in the book chapter
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                c, s = phi_mat_cos[..., i, j : j + 1], phi_mat_sin[..., i, j : j + 1]

                p = i
                q = N - j - 1
                Ur[..., p, :], Ur[..., q, :] = (
                    Ur[..., p, :] * c - Ur[..., q, :] * s,
                    Ur[..., p, :] * s + Ur[..., q, :] * c,
                )
        Ur = delta_list[..., np.newaxis] * Ur

        return Ur

    def reconstruct_francis(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_francis_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_francis_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_francis(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_francis(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_francis_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    @profile(timer=timer)
    def reconstruct_reck_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N, dtype=np.complex64)
        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """

        for i in range(N - 1):
            lower = N - 2 - i
            for j in range(i + 1):
                phi = phi_mat[lower, j, :]
                p = N - 2 - i + j
                q = p + 1

                row_p, row_q = Ur[p, lower:], Ur[q, lower:]

                row_p *= np.exp(1j * (phi[0] + (phi[2] + phi[3]) / 2 + np.pi / 2))
                row_q *= np.exp(1j * (phi[1] + (phi[2] + phi[3]) / 2 + np.pi / 2))
                half_delta_theta = (phi[2] - phi[3]) / 2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, lower:], Ur[q, lower:] = (
                    row_p_sin + row_q_cos,
                    row_p_cos - row_q_sin,
                )

        Ur = np.exp(1j * delta_list[:, np.newaxis]) * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_reck_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=np.complex64)

        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """
        for i in range(N - 1):
            lower = N - 2 - i
            for j in range(i + 1):
                phi = phi_mat[..., lower, j : j + 1, :]
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p *= np.exp(
                    1j * (phi[..., 0] + (phi[..., 2] + phi[..., 3]) / 2 + np.pi / 2)
                )
                row_q *= np.exp(
                    1j * (phi[..., 1] + (phi[..., 2] + phi[..., 3]) / 2 + np.pi / 2)
                )
                half_delta_theta = (phi[..., 2] - phi[..., 3]) / 2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = (
                    row_p_sin + row_q_cos,
                    row_p_cos - row_q_sin,
                )
                ### this rotation is equivalent to complex number multiplication as an acceleration.

        Ur = np.exp(1j * delta_list[..., np.newaxis]) * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_reck_batch_par(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        """
        cos, -sin
        sin, cos
        """

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ### 2n-3 stages
        for i in range(2 * N - 3):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = (
                    phi_mat_cos[..., lower, j : j + 1],
                    phi_mat_sin[..., lower, j : j + 1],
                )
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = (
                    row_p_cos - row_q_sin,
                    row_p_sin + row_q_cos,
                )
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    def reconstruct_reck(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_reck_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_reck_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_reck(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_reck(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_clements_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    @profile(timer=timer)
    def reconstruct_clements_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N, dtype=np.complex64)

        for i in range(N):  ## N layers
            max_len = 2 * (i + 1)
            for j in range((i % 2), N - 1, 2):
                # c, s = phi_mat_cos[j, i], phi_mat_sin[j, i]
                phi = phi_mat[j, i, :]
                ## not the entire row needs to be rotated, only a small working set is used
                # row_p, row_q = Ur[j, :], Ur[j+1, :]
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                row_p, row_q = Ur[j, lower:upper], Ur[j + 1, lower:upper]
                row_p *= np.exp(1j * (phi[0] + (phi[2] + phi[3]) / 2 + np.pi / 2))
                row_q *= np.exp(1j * (phi[1] + (phi[2] + phi[3]) / 2 + np.pi / 2))
                half_delta_theta = (phi[2] - phi[3]) / 2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[j, lower:upper], Ur[j + 1, lower:upper] = (
                    row_p_sin + row_q_cos,
                    row_p_cos - row_q_sin,
                )

        if N % 2 == 0:
            ### have to address delta_list[0]
            Ur[0, :] *= np.exp(1j * delta_list[0])
        return Ur

    @profile(timer=timer)
    def reconstruct_clements_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=np.complex64)
        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        for i in range(N):  ## N layers
            max_len = 2 * (i + 1)
            for j in range((i % 2), N - 1, 2):
                ## not the entire row needs to be rotated, only a small working set is used
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                phi = phi_mat[..., j, i : i + 1, :]
                row_p, row_q = Ur[..., j, lower:upper], Ur[..., j + 1, lower:upper]
                row_p *= np.exp(
                    1j * (phi[..., 0] + (phi[..., 2] + phi[..., 3]) / 2 + np.pi / 2)
                )
                row_q *= np.exp(
                    1j * (phi[..., 1] + (phi[..., 2] + phi[..., 3]) / 2 + np.pi / 2)
                )
                half_delta_theta = (phi[..., 2] - phi[..., 3]) / 2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., j, lower:upper], Ur[..., j + 1, lower:upper] = (
                    row_p_sin + row_q_cos,
                    row_p_cos - row_q_sin,
                )

        if N % 2 == 0:
            ### have to address delta_list[0]
            Ur[..., 0, :] *= np.exp(1j * delta_list[..., 0:1])

        return Ur

    def reconstruct_clements(self, delta_list, phi_mat):
        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return self.reconstruct_clements_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_clements_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                U = matrix_parametrization_cuda.reconstruct_clements(
                    delta_list, phi_mat
                )

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        self.reconstruct_clements(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )
                else:
                    return torch.from_numpy(
                        self.reconstruct_clements_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    def reconstruct(self, delta_list, phi_mat):
        if self.alg == "francis":
            reconstruct_cpu = self.reconstruct_francis
            reconstruct_batch = self.reconstruct_francis_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_francis
        elif self.alg == "reck":
            reconstruct_cpu = self.reconstruct_reck_cpu
            reconstruct_batch = self.reconstruct_reck_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_reck
        elif self.alg == "clements":
            reconstruct_cpu = self.reconstruct_clements_cpu
            reconstruct_batch = self.reconstruct_clements_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_clements
        else:
            raise NotImplementedError

        if isinstance(phi_mat, np.ndarray):
            if len(delta_list.shape) == 1:
                return reconstruct_cpu(delta_list, phi_mat)
            else:
                return reconstruct_batch(delta_list, phi_mat)
        else:
            if phi_mat.is_cuda:
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                # U = torch.zeros_like(phi_mat).contiguous()
                U = reconstruct_cuda(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if phi_mat.dim() == 2:
                    return torch.from_numpy(
                        reconstruct_cpu(delta_list.cpu().numpy(), phi_mat.cpu().numpy())
                    )
                else:
                    return torch.from_numpy(
                        reconstruct_batch(
                            delta_list.cpu().numpy(), phi_mat.cpu().numpy()
                        )
                    )

    def check_identity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def check_unitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.check_identity(M)

    def check_equal(self, M1, M2):
        return (M1.shape == M2.shape) and np.allclose(M1, M2)

    def gen_random_ortho(self, N):
        U = ortho_group.rvs(N)
        print(
            f"[I] Generate random {N}*{N} unitary matrix, check unitary: {self.check_unitary(U)}"
        )
        return U

    def to_degree(self, M):
        return np.degrees(M)
