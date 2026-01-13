"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-09 23:08:54
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-09 23:08:54
"""

import unittest

import numpy as np
import torch
from pyutils.general import TimerCtx, logger

import torchonn as onn


class TestOp(unittest.TestCase):
    def test_matrix_parametrization(self):
        B = 1024
        N = 16
        T = 10
        decomposer = onn.op.RealUnitaryDecomposerBatch(alg="clements")
        U = torch.nn.init.orthogonal_(torch.empty(N, N)).double()
        U = torch.zeros(B, N, N, dtype=U.dtype, device=U.device) + U

        for alg in ["clements", "reck", "francis"]:
            decomposer.set_alg(alg)
            ###### decompose ########
            delta_list_gold, phi_mat_gold = decomposer.decompose(U.clone().cpu())

            with TimerCtx() as t:
                for _ in range(T):
                    delta_list, phi_mat = decomposer.decompose(U.clone().cpu())
            print(
                f"\nAlg {alg}: \n\tCPU  decompose {tuple(U.size())} unitary: {t.interval/T:.6f} s"
            )
            assert np.allclose(delta_list.cpu().numpy(), delta_list_gold.numpy())
            assert np.allclose(phi_mat.cpu().numpy(), phi_mat_gold.numpy())

            U = U.cuda()
            for _ in range(5):  # warm GPU
                delta_list, phi_mat = decomposer.decompose(U.clone())
            torch.cuda.synchronize()
            with TimerCtx() as t:
                for _ in range(T):
                    delta_list, phi_mat = decomposer.decompose(U.clone())
                torch.cuda.synchronize()
            print(f"\tCUDA decompose {tuple(U.size())} unitary: {t.interval/T:.6f} s")
            assert np.allclose(
                delta_list.cpu().numpy(), delta_list_gold.numpy(), rtol=1e-4, atol=1e-5
            )
            assert np.allclose(
                phi_mat.cpu().numpy(), phi_mat_gold.numpy(), rtol=1e-4, atol=1e-5
            ), print(
                "max abs error:",
                np.abs(phi_mat.cpu().numpy() - phi_mat_gold.numpy()).max(),
            )

            ###### reconstruct ########
            delta_list, phi_mat = delta_list.cpu(), phi_mat.cpu()
            torch.cuda.synchronize()
            with TimerCtx() as t:
                for _ in range(T):
                    Ur = decomposer.reconstruct(delta_list, phi_mat)
            print(
                f"\nAlg {alg}: \n\tCPU  reconstruct {tuple(Ur.size())} unitary: {t.interval/T:.6f} s"
            )
            assert np.allclose(
                Ur.cpu().numpy(), U.cpu().numpy(), rtol=1e-4, atol=1e-5
            ), print("max abs error:", np.abs(Ur.cpu().numpy() - U.cpu().numpy()).max())

            delta_list, phi_mat = delta_list.cuda(), phi_mat.cuda()
            for _ in range(5):  # warm GPU
                Ur = decomposer.reconstruct(delta_list, phi_mat)
            torch.cuda.synchronize()
            with TimerCtx() as t:
                for _ in range(T):
                    Ur = decomposer.reconstruct(delta_list, phi_mat)
                torch.cuda.synchronize()
            print(
                f"\tCUDA reconstruct {tuple(Ur.size())} unitary: {t.interval/T:.6f} s"
            )
            assert np.allclose(
                Ur.cpu().numpy(), U.cpu().numpy(), rtol=1e-4, atol=1e-5
            ), print("max abs error:", np.abs(Ur.cpu().numpy() - U.cpu().numpy()).max())


if __name__ == "__main__":
    unittest.main()
