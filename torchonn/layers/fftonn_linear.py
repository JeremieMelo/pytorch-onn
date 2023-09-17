"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-27 19:02:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-27 22:17:35
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pyutils.compute import gen_gaussian_noise, get_complex_energy, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.butterfly_op import TrainableButterfly
from torchonn.op.mzi_op import PhaseQuantizer

__all__ = [
    "FFTONNBlockLinear",
]


class FFTONNBlockLinear(ONNBaseLayer):
    """
    Butterfly blocking Linear layer.
    J. Gu, et al., "Towards Area-Efficient Optical Neural Networks: An FFT-based Architecture," ASP-DAC 2020.
    https://ieeexplore.ieee.org/document/9045156
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: int
    weight: Tensor
    __mode_list__ = ["fft", "hadamard", "zero_bias", "trainable"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        miniblock: int = 4,
        mode: str = "fft",
        photodetect: bool = True,
        device: Device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock))
        self.in_features_pad = self.grid_dim_x * miniblock
        self.out_features_pad = self.grid_dim_y * miniblock
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect

        crosstalk_filter_size = 3

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
        self.phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="butterfly",
            device=self.device,
        )
        self.phase_V_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="butterfly",
            device=self.device,
        )
        # self.phase_S_quantizer = PhaseQuantizer(
        #     self.w_bit,
        #     self.v_pi,
        #     self.v_max,
        #     gamma_noise_std=0,
        #     crosstalk_factor=0,
        #     crosstalk_filter_size=crosstalk_filter_size,
        #     random_state=0,
        #     mode="diagonal",
        #     device=self.device,
        # )

        ### build trainable parameters
        self.build_parameters()

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(mode=self.mode)

    def build_parameters(self) -> None:
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            self.miniblock,
            self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )
        self.T = TrainableButterfly(
            length=self.miniblock,
            reverse=False,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            phase_quantizer=self.phase_U_quantizer,
            device=self.device,
        )
        self.S = Parameter(
            torch.zeros(self.grid_dim_y, self.grid_dim_x, self.miniblock, dtype=torch.cfloat).to(self.device)
        )  # complex frequency-domain weights
        self.Tr = TrainableButterfly(
            length=self.miniblock,
            reverse=True,
            bit_reversal=False,
            enable_last_level_phase_shifter=True,
            phase_quantizer=self.phase_V_quantizer,
            device=self.device,
        )

    @property
    def U(self):
        return self.Tr.build_weight()

    @property
    def V(self):
        return self.T.build_weight()

    @property
    def phase_U(self):
        return self.Tr.phases

    @property
    def phase_V(self):
        return self.T.phases

    def reset_parameters(self, mode: Optional[str] = None) -> None:
        mode = mode or self.mode
        W = init.kaiming_normal_(
            torch.empty(
                self.grid_dim_y,
                self.grid_dim_x,
                self.miniblock,
                self.miniblock,
                dtype=torch.cfloat,
                device=self.device,
            )
        )
        _, S, _ = torch.svd(W, compute_uv=False)
        self.S.data.copy_(S)

        if mode == "zero_bias":
            self.T.reset_parameters(alg="zero")
            self.Tr.reset_parameters(alg="zero")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "hadamard":
            self.T.reset_parameters(alg="hadamard")
            self.Tr.reset_parameters(alg="hadamard")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "fft":
            self.T.reset_parameters(alg="fft")
            self.Tr.reset_parameters(alg="fft")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "trainable":
            self.T.reset_parameters(alg="uniform")
            self.Tr.reset_parameters(alg="uniform")
            self.T.phases.requires_grad_(True)
            self.Tr.phases.requires_grad_(True)
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        miniblock: int = 4,
        mode: str = "fft",
        photodetect: bool = True,
        verbose: bool = False,
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            miniblock (int, optional): miniblock size. Defaults to 4.
            mode (str, optional): parametrization mode. Defaults to "fft".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted FFTONNBlockLinear module
        """
        assert isinstance(layer, nn.Linear), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            miniblock=miniblock,
            mode=mode,
            photodetect=photodetect,
            device=device,
        ).to(device)
        weight = instance.weight
        tmp = torch.zeros(instance.out_features_pad, instance.in_features_pad, dtype=torch.cfloat, device=instance.device)
        tmp.data.real[:out_features, :in_features].copy_(layer.weight)
        instance.weight.data.copy_(
            tmp.view(weight.shape[0], weight.shape[2], weight.shape[1], weight.shape[3]).permute(0, 2, 1, 3)
        )
        instance.sync_parameters(src="weight", verbose=verbose)
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        # differentiable feature is gauranteed
        weight = U.matmul(S.unsqueeze(-1) * V)
        self.weight.data.copy_(weight)
        return weight

    def sync_parameters(self, src: str = "weight", steps: int = 3000, verbose: bool = False) -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "phase":
            # phase to weight
            self.weight.data.copy_(self.build_weight_from_usv(self.U.data, self.S.data, self.V.data))
        elif src == "weight":
            # weight to phase
            target = self.weight.data
            params = {}
            if not self.T.phases.requires_grad:
                target = target.matmul(torch.linalg.inv(self.V.data))
            else:
                params["V"] = self.T.phases
            if not self.Tr.phases.requires_grad:
                target = torch.linalg.inv(self.U.data).matmul(target)
            else:
                params["U"] = self.Tr.phases
        
            params["S"] = self.S
            if len(params) == 1:  # only has self.S, perform optimal singular value projection
                self.S.data.copy_(torch.linalg.diagonal(target))
            else:  # perform gradient descent to solve
                optimizer = torch.optim.Adam(list(params.values()), lr=1e-3)
                for i in range(steps):
                    w = self.S
                    if "V" in params:
                        w = w.unsqueeze(-1) * self.V
                        
                    if "U" in params:
                        U = self.U
                        if w.ndim == self.S.ndim + 1:
                            w = U.matmul(w)
                        else:
                            w = U * w.unsqueeze(-2)
                            
                    loss = F.mse_loss(torch.view_as_real(target), torch.view_as_real(w))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if verbose and (i % 500 == 0 or i == steps - 1):
                        logger.info(f"Sync weight to phase: step = {i:5d}, mse = {loss.item():.2e}")

        else:
            raise NotImplementedError
        if verbose:
            with torch.no_grad():
                target = self.weight.data.clone()
                w = self.build_weight()
                error = F.mse_loss(torch.view_as_real(w), torch.view_as_real(target))
                logger.info(f"Mapping Linear to FFTONNBlockLinear: MSE = {error:.2e}")

    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        weight = self.build_weight_from_usv(self.U, self.S, self.V)

        return weight

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        # self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        # self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        # self.phase_S_quantizer.set_bitwidth(w_bit)
        self.phase_V_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        x = x.to(torch.cfloat)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k]
        else:
            weight = self.weight
        offset = int(np.ceil(self.grid_dim_x / 2)) * self.miniblock
        weight = merge_chunks(weight)[: self.out_features, : self.in_features].t()
        weight_pos = weight[:offset, :]
        weight_neg = weight[offset:, :]
        x_pos = x[..., :offset]
        x_neg = x[..., offset:]
        x_pos = x_pos.matmul(weight_pos)
        x_pos = x_pos.real.square() + x_pos.imag.square()
        x_neg = x_neg.matmul(weight_neg)
        x_neg = x_neg.real.square() + x_neg.imag.square()
        x = x_pos - x_neg

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x
