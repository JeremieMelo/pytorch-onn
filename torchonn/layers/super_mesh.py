"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-04-18 21:04:05
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-04-19 01:41:51
"""

import itertools
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from pyutils.compute import gen_gaussian_noise
from pyutils.general import logger
from pyutils.quant.lsq import WeightQuantizer_LSQ
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.types import Device

from torchonn.op.cross_op import hard_diff_round
from torchonn.op.dc_op import dc_quantize_fn

__all__ = [
    "SuperOpticalModule",
    "SuperBatchedPSLayer",
    "SuperDCFrontShareLayer",
    "SuperCRLayer",
    "super_layer_name_dict",
    "ArchSampler",
    "get_named_sample_arch",
    "SuperMeshADEPT",
    "SuperMeshMZI",
    "SuperMeshButterfly",
]


class ArchSampler(object):
    def __init__(self, model: nn.Module, strategy=None, n_layers_per_block=None):
        self.model = model
        self.total_steps = None
        self.arch_space = model.arch_space
        # subnetwork sampling strategy
        # None, limit_diff, progressive
        self.strategy = strategy
        self.sample_arch_old = None
        self.arch_space_stage = None
        self.n_ops_smallest = 0
        self.n_ops_largest = 0

        self.is_block_based = False
        self.n_layers_per_block = n_layers_per_block
        if n_layers_per_block is not None:
            self.is_block_based = True

        self.get_space_stats()
        self.step = 0

        self.n_ops_per_chunk = None
        if strategy["name"] == "progressive":
            self.get_n_ops_per_chunk()

        self.sample_n_ops = None
        self.current_stage = 0
        self.current_chunk = 0

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def get_n_ops_per_chunk(self):
        """separate the space to several subspace"""
        n_chunks = self.strategy["n_chunks"]
        if self.strategy["chunk_mode"] == "same_interval":
            logger.warning(
                "same_interval chunking may cause extra long "
                "time to sample a sub network because of the "
                "Central Limit Theorem of n_ops in a subnet"
            )
            self.n_ops_per_chunk = list(
                np.linspace(
                    self.n_ops_smallest, self.n_ops_largest, n_chunks + 1
                ).astype(int)
            )
        elif self.strategy["chunk_mode"] == "same_n_samples":
            logger.info("estimating the chunks...")
            n_ops_all = []
            n_chunk_est_samples = self.strategy["n_chunk_est_samples"]
            for k in range(n_chunk_est_samples):
                sample_arch = self.get_random_sample_arch()
                n_ops_all.append(self.get_sample_stats(sample_arch))
            n_ops_all.sort()
            idx_all = np.linspace(0, n_chunk_est_samples - 1, n_chunks + 1).astype(int)
            self.n_ops_per_chunk = [n_ops_all[idx] for idx in idx_all]
            self.n_ops_per_chunk[0] = self.n_ops_smallest
            self.n_ops_per_chunk[-1] = self.n_ops_largest
        else:
            raise NotImplementedError(
                f"chunk mode {self.strategy['chunk_mode']} not supported."
            )

    def get_sample_stats(self, sample_arch):
        n_ops = 0
        if self.is_block_based:
            layers_arch = sample_arch[:-1]
            block_arch = sample_arch[-1]
        else:
            layers_arch = self.arch_space
            block_arch = None

        for k, layer_arch in enumerate(layers_arch):
            if not isinstance(layer_arch, Iterable):
                # share front layer
                if not self.is_block_based or k < block_arch * self.n_layers_per_block:
                    n_ops += layer_arch
            else:
                # arbitrary layer
                if not self.is_block_based or k < block_arch * self.n_layers_per_block:
                    n_ops += len(layer_arch)

        return n_ops

    def get_space_stats(self):
        """get the max number and smallest number of ops in the space"""
        if self.is_block_based:
            layers_space = self.arch_space[:-1]
            block_space = self.arch_space[-1]
        else:
            layers_space = self.arch_space
            block_space = None

        for k, layer_space in enumerate(layers_space):
            if not isinstance(layer_space[0], Iterable):
                # share front layer
                self.n_ops_largest += max(layer_space)
                if (
                    not self.is_block_based
                    or k < min(block_space) * self.n_layers_per_block
                ):
                    self.n_ops_smallest += min(layer_space)
            else:
                # arbitrary layer
                self.n_ops_largest += max(list(map(len, layer_space)))
                if (
                    not self.is_block_based
                    or k < min(block_space) * self.n_layers_per_block
                ):
                    self.n_ops_smallest += min(list(map(len, layer_space)))

    def get_random_sample_arch(self):
        sample_arch = []
        for layer_arch_space in self.arch_space:
            layer_arch = np.random.choice(layer_arch_space)
            sample_arch.append(layer_arch)
        return sample_arch

    def get_uniform_sample_arch(self):
        if self.strategy["name"] == "plain" or (
            self.strategy["name"] == "limit_diff" and self.sample_arch_old is None
        ):
            sample_arch = self.get_random_sample_arch()
        elif self.strategy["name"] == "limit_diff":
            """
            limited differences between architectures of two consecutive
            samples
            """
            sample_arch = self.sample_arch_old.copy()
            n_diffs = self.strategy["n_diffs"]
            assert n_diffs <= len(self.arch_space)
            diff_parts_idx = np.random.choice(
                [
                    i
                    for i in np.arange(len(self.arch_space))
                    if len(self.arch_space[i]) > 1
                ],
                n_diffs,
                replace=False,
            )

            for idx in diff_parts_idx:
                sample_arch[idx] = np.random.choice(self.arch_space[idx])
        elif self.strategy["name"] == "progressive":
            """
            different stages have different model capacity.
            In different stage the total number of gates is specified
            """
            n_stages = self.strategy["n_stages"]
            n_chunks = self.strategy["n_chunks"]
            while True:
                sample_arch = self.get_random_sample_arch()
                n_ops = self.get_sample_stats(sample_arch)
                current_stage = int(self.step // (self.total_steps / n_stages))
                current_chunk = current_stage % n_chunks
                self.current_chunk = current_chunk
                self.current_stage = current_stage
                if self.strategy["subspace_mode"] == "expand":
                    # the subspace size is expanding
                    if self.strategy["direction"] == "top_down":
                        if (
                            n_ops
                            >= list(reversed(self.n_ops_per_chunk))[current_chunk + 1]
                        ):
                            break
                    elif self.strategy["direction"] == "bottom_up":
                        if n_ops <= self.n_ops_per_chunk[current_chunk + 1]:
                            break
                    else:
                        raise NotImplementedError(
                            f"Direction mode {self.strategy['direction']} "
                            f"not supported."
                        )
                elif self.strategy["subspace_mode"] == "same":
                    # the subspace size is the same
                    if self.strategy["direction"] == "top_down":
                        left = list(reversed(self.n_ops_per_chunk))[current_chunk + 1]
                        right = list(reversed(self.n_ops_per_chunk))[current_chunk]
                    elif self.strategy["direction"] == "bottom_up":
                        left = self.n_ops_per_chunk[current_chunk]
                        right = self.n_ops_per_chunk[current_chunk + 1]
                    else:
                        raise NotImplementedError(
                            f"Direction mode {self.strategy['direction']} "
                            f"not supported."
                        )

                    if left <= n_ops <= right:
                        break
                else:
                    raise NotImplementedError(
                        f"Subspace mode {self.strategy['subspace_mode']} "
                        f"not supported."
                    )
        elif self.strategy["name"] == "limit_diff_expanding":
            """
            shrink the overall number of blocks and in-block size
            """
            if self.sample_arch_old is None:
                self.sample_arch_old = get_named_sample_arch(self.arch_space, "largest")
            sample_arch = self.sample_arch_old.copy()
            n_stages = self.strategy["n_stages"]
            n_chunks = self.strategy["n_chunks"]
            n_diffs = self.strategy["n_diffs"]
            assert n_diffs <= len(self.arch_space)

            current_stage = int(self.step // (self.total_steps / n_stages))
            current_chunk = current_stage % n_chunks
            self.current_stage = current_stage
            self.current_chunk = current_chunk
            diff_parts_idx = np.random.choice(
                [
                    i
                    for i in np.arange(len(self.arch_space))
                    if len(self.arch_space[i]) > 1
                ],
                n_diffs,
                replace=False,
            )

            for idx in diff_parts_idx:
                layer_arch_space = self.arch_space[idx]
                n_choices = len(layer_arch_space)
                new_space = layer_arch_space[
                    int(
                        round((n_choices - 1) * (1 - (current_chunk + 1) / (n_chunks)))
                    ) :
                ]
                if len(new_space) == 1:
                    sample_arch[idx] = new_space[0]
                else:
                    sample_arch[idx] = np.random.choice(new_space)
        elif self.strategy["name"] == "ldiff_blkexpand":
            """
            shrink the overall number of blocks only
            """
            if self.sample_arch_old is None:
                self.sample_arch_old = get_named_sample_arch(self.arch_space, "largest")
            sample_arch = self.sample_arch_old.copy()
            n_stages = self.strategy["n_stages"]
            n_chunks = self.strategy["n_chunks"]
            n_diffs = self.strategy["n_diffs"]
            assert n_diffs <= len(self.arch_space)

            current_stage = int(self.step // (self.total_steps / n_stages))
            if current_stage >= n_chunks:
                """
                major difference here, after expanding the space,
                will not go back
                """
                current_chunk = n_chunks - 1
            else:
                current_chunk = current_stage
            self.current_stage = current_stage
            self.current_chunk = current_chunk
            diff_parts_idx = np.random.choice(
                [
                    i
                    for i in np.arange(len(self.arch_space))
                    if len(self.arch_space[i]) > 1
                ],
                n_diffs,
                replace=False,
            )
            new_arch_space = self.arch_space.copy()
            n_blk_choices = len(new_arch_space[-1])
            new_arch_space[-1] = new_arch_space[-1][
                int(
                    round((n_blk_choices - 1) * (1 - (current_chunk + 1) / (n_chunks)))
                ) :
            ]
            for idx in diff_parts_idx:
                sample_arch[idx] = np.random.choice(new_arch_space[idx])
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not " f"supported.")

        self.sample_n_ops = self.get_sample_stats(sample_arch)
        self.sample_arch_old = sample_arch
        self.step += 1

        return sample_arch


def get_named_sample_arch(arch_space, name):
    """
    examples:
    - blk1_ratio0.5 means 1 block and each layers' arch is the 0.5 of all
    choices
    - blk8_ratio0.3 means 8 block and each layers' arch is 0.3 of all choices
    - ratio0.4 means 0.4 for each arch choice, including blk if exists
    """

    sample_arch = []
    if name == "smallest":
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[0]
            sample_arch.append(layer_arch)
    elif name == "largest":
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[-1]
            sample_arch.append(layer_arch)
    elif name == "middle":
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[len(layer_arch_space) // 2]
            sample_arch.append(layer_arch)
    elif name.startswith("blk"):
        # decode the block and ratio
        n_block = eval(name.split("_")[0].replace("blk", ""))
        ratio = eval(name.split("_")[1].replace("ratio", ""))
        assert ratio <= 1
        for layer_arch_space in arch_space[:-1]:
            layer_arch = layer_arch_space[
                int(round((len(layer_arch_space) - 1) * ratio))
            ]
            sample_arch.append(layer_arch)
        sample_arch.append(n_block)
    elif name.startswith("ratio"):
        # decode the numerator and denominator
        ratio = eval(name.replace("ratio", ""))
        assert ratio <= 1
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[
                int(round((len(layer_arch_space) - 1) * ratio))
            ]
            sample_arch.append(layer_arch)
    else:
        raise NotImplementedError(f"Arch name {name} not supported.")

    return sample_arch


def get_combinations(inset: List, n=None) -> List[List]:
    all_combs = []
    if n is None:
        # all possible combinations, with different #elements in a set
        for k in range(1, len(inset) + 1):
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))
    elif isinstance(n, int):
        all_combs.extend(list(map(list, itertools.combinations(inset, n))))
    elif isinstance(n, Iterable):
        for k in n:
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))

    return all_combs


class SuperOpticalModule(nn.Module):
    def __init__(self, n_waveguides):
        super().__init__()
        self.n_waveguides = n_waveguides
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        # a structure that can repersent the architecture
        # e.g., for front share layer, a scalar can represent the arch
        # e.g., for permutation layer, a permuted index array can represent the arch
        self.sample_arch = sample_arch

    @property
    def arch_space(self):
        return None

    def count_sample_params(self):
        raise NotImplementedError


class SuperBatchedPSLayer(SuperOpticalModule):
    """Super phase shifter layers. Support batched forward"""

    _share_uv_list = {"global", "row", "col", "none"}

    def __init__(
        self,
        grid_dim_x: int,
        grid_dim_y: int,
        miniblock: Tuple[int],  # miniblock[:-2] from ONNBaselayers
        n_waveguides: int,  # miniblock[-2] or miniblock[-1] from ONNBaselayers
        share_uv: str = "global",
        trainable: bool = True,
        w_bit: int = 32,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__(n_waveguides=n_waveguides)
        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y
        self.miniblock = miniblock
        self.share_uv = share_uv.lower()
        assert (
            self.share_uv in self._share_uv_list
        ), f"share_uv only supports {self._share_uv_list}, but got {share_uv}"
        self.trainable = trainable
        self.w_bit = w_bit
        self.device = device
        self.phase_quantizer = WeightQuantizer_LSQ(
            out_features=self.grid_dim_y,
            device=self.device,
            nbits=self.w_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        self.build_parameters()
        self.reset_parameters()
        self.set_phase_noise(0)

    def set_bitwidth(self, bitwidth: int) -> None:
        self.w_bit = bitwidth
        self.phase_quantizer.set_bit(bitwidth)

    def build_parameters(self):
        """weight is the phase shift of the phase shifter"""
        if self.share_uv == "global":
            self.weight = nn.Parameter(
                torch.empty(
                    1, 1, *self.miniblock, self.n_waveguides, device=self.device
                ),
                requires_grad=self.trainable,
            )
        elif self.share_uv == "row":
            ## use the same PS within a row => share U
            self.weight = nn.Parameter(
                torch.empty(
                    self.grid_dim_y,
                    1,
                    *self.miniblock,
                    self.n_waveguides,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
        elif self.share_uv == "col":
            ## use the same PS within a column => share V
            self.weight = nn.Parameter(
                torch.empty(
                    1,
                    self.grid_dim_x,
                    *self.miniblock,
                    self.n_waveguides,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
        elif self.share_uv == "none":
            ## independent PS for each block
            self.weight = nn.Parameter(
                torch.empty(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    *self.miniblock,
                    self.n_waveguides,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
        else:
            raise ValueError(f"Not supported share_uv: {self.share_uv}")

    def reset_parameters(self, alg="uniform"):
        assert alg in {"normal", "uniform", "identity"}
        if alg == "normal":
            nn.init.normal_(self.weight)
        elif alg == "uniform":
            nn.init.uniform_(self.weight, -np.pi / 2, np.pi)
        elif alg == "identity":
            self.weight.data.zero_()

    def build_weight(self):
        if self.w_bit < 16:
            weight = self.phase_quantizer(self.weight % (2 * np.pi))
        else:
            weight = self.weight

        if self.phase_noise_std > 0:
            weight = weight + gen_gaussian_noise(
                torch.zeros_like(weight), noise_mean=0, noise_std=self.phase_noise_std
            )

        return weight

    def forward(self, x: Tensor) -> Tensor:
        # x[..., q, n_waveguides] complex
        if not x.is_complex():
            x = x.to(torch.cfloat)

        weight = self.build_weight()
        weight = torch.exp(1j * weight)
        # weight: [p, q, ..., k1]
        # x: [bs, p, q, ..., k2]
        x = x.mul(weight)

        return x

    @property
    def arch_space(self):
        ## do not sample PS, we always put a whole column of PS
        choices = [self.n_waveguides]
        return get_combinations(choices)

    def count_sample_params(self):
        return len(self.sample_arch) if self.weight.requires_grad else 0

    def extra_repr(self) -> str:
        s = f"grid_dim_x={self.grid_dim_x}, grid_dim_y={self.grid_dim_y}, share_uv={self.share_uv}, n_waveguides={self.n_waveguides}, sample_arch={self.sample_arch}, trainable={self.trainable}"
        return s

    def set_phase_noise(self, noise_std: float = 0.0):
        self.phase_noise_std = noise_std


class SuperDCFrontShareLayer(SuperOpticalModule):
    """Super directionalcoupler layer with front sharing."""

    def __init__(
        self,
        n_waveguides: int,
        offset: int = 0,
        trainable: bool = False,
        binary: bool = False,
        device: Device = torch.device("cuda:0"),
    ):
        """Initialization

        Args:
            n_waveguides (int): number of waveguides
            offset (int, optional): The waveguide offset of the first DC. Defaults to 0.
            trainable (bool, optional): Whether to have trainable DC transmission factor. Defaults to False.
            binary (bool, optional): Whether to binarize the DC transmission. Defaults to False.
            device (_type_, optional): torch Device. Defaults to torch.device("cuda:0").
        """
        super().__init__(n_waveguides=n_waveguides)
        self.offset = offset
        self.max_arch = (self.n_waveguides - self.offset) // 2
        self.trainable = trainable
        self.binary = binary
        self.device = device
        self.build_parameters()
        self.reset_parameters()
        if self.binary:
            self.weight_quantizer = dc_quantize_fn(w_bit=1)
        else:
            self.weight_quantizer = None

        self.set_dc_noise(0)
        self.fast_mode = False
        self.fast_weight = None

    def build_parameters(self):
        """weight is the transmission factor t in the DC transfer matrix"""
        self.weight = nn.Parameter(
            torch.empty(self.max_arch, device=self.device), requires_grad=self.trainable
        )

    def reset_parameters(self):
        if self.binary:
            nn.init.uniform_(self.weight, -0.01, 0.01)
        else:
            nn.init.constant_(self.weight, 2**0.5 / 2)

    def build_weight(self):
        if self.sample_arch < self.max_arch:
            weight = self.weight[: self.sample_arch]
        else:
            weight = self.weight
        if self.binary:
            weight = self.weight_quantizer(weight)  # binarize to sqrt(2)/2 and 1

        if self.dc_noise_std > 0:
            mask = weight.data > 0.9  # only inject noise when t=sqrt(2)/2
            noise = gen_gaussian_noise(
                torch.zeros_like(weight), noise_mean=0, noise_std=self.dc_noise_std
            )
            noise.masked_fill_(mask, 0)
            weight = weight + noise

        t = weight
        k = (1 - weight.square() + 1e-6).sqrt()  # when t=1, k=0, the grad is nan !!
        w11 = w22 = t.to(torch.complex64)
        w12 = w21 = k.mul(1j)
        weight = torch.stack([w11, w12, w21, w22], dim=-1).view(-1, 2, 2)

        return weight

    def forward(self, x: Tensor) -> Tensor:
        # x[..., n_waveguides] complex

        if not x.is_complex():
            x = x.to(torch.cfloat)

        if self.fast_mode and self.fast_weight is not None:
            weight = self.fast_weight
        else:
            weight = self.build_weight()
        sample_arch = min(self.sample_arch, self.max_arch)
        n_sample_waveguides = int(sample_arch * 2)

        if n_sample_waveguides < x.size(-1):
            out = x[..., self.offset : self.offset + n_sample_waveguides]
            # [1, n//2, 2, 2] x [bs, n//2, 2, 1] = [bs, n//2, 2, 1] -> [bs, n]
            out = (
                weight.unsqueeze(0)
                .matmul(out.view(-1, sample_arch, 2, 1))
                .view(list(x.shape[:-1]) + [n_sample_waveguides])
            )
            out = torch.cat(
                [
                    x[..., : self.offset],
                    out,
                    x[..., self.offset + n_sample_waveguides :],
                ],
                dim=-1,
            )
        else:
            out = (
                weight.unsqueeze(0)
                .matmul(x.reshape(-1, sample_arch, 2, 1))
                .view(list(x.shape[:-1]) + [n_sample_waveguides])
            )

        return out

    @property
    def arch_space(self):
        if self.trainable:
            return [self.max_arch]
        else:
            return list(range(1, self.max_arch + 1))

    def count_sample_params(self):
        return len(self.sample_arch) if self.weight.requires_grad else 0

    def extra_repr(self) -> str:
        s = f"n_waveguides={self.n_waveguides}, sample_arch={self.sample_arch}, offset={self.offset}, trainable={self.trainable}"
        return s

    def set_dc_noise(self, noise_std: float = 0.0):
        self.dc_noise_std = noise_std

    def fix_arch_solution(self):
        """Fix the trainable DC array to a transfer matrix for fast forward"""
        with torch.no_grad():
            if self.binary:
                weight = self.weight_quantizer(self.weight.data)
            else:
                weight = self.weight.data
            t = weight
            k = (1 - weight.square()).sqrt()  # when t=1, k=0, the grad is nan !!
            w11 = w22 = t.to(torch.complex64)
            w12 = w21 = k.mul(1j)
            weight = torch.stack([w11, w12, w21, w22], dim=-1).view(-1, 2, 2)

            self.fast_weight = weight
            self.fast_mode = True
            self.weight.requires_grad_(False)


class SuperCRLayer(SuperOpticalModule):
    """Super waveguide crossing layer"""

    def __init__(
        self,
        n_waveguides: int,
        trainable: bool = True,
        symmetry: bool = False,
        device: Device = torch.device("cuda:0"),
    ):
        """Initialization

        Args:
            n_waveguides (int): number of waveguides
            trainable (bool, optional): Whether to use trainable crossing layer. Defaults to True.
            symmetry (bool, optional): Whether to force symmetric crossing layout. Defaults to False.
            device (_type_, optional): torch Device. Defaults to torch.device("cuda:0").
        """
        super().__init__(n_waveguides=n_waveguides)
        self.trainable = trainable
        self.symmetry = symmetry
        self.device = device
        self.build_parameters()
        self.reset_parameters()
        self.identity_forward = False
        self.set_cr_noise(0)
        self.fast_mode = False
        self.indices = None

    def build_parameters(self):
        self.weight = nn.Parameter(
            torch.empty(self.n_waveguides, self.n_waveguides, device=self.device),
            requires_grad=self.trainable,
        )
        self.eye = torch.eye(self.n_waveguides, device=self.device)
        # ALM multiplier
        self.alm_multiplier = nn.Parameter(
            torch.empty(2, self.n_waveguides, device=self.device), requires_grad=False
        )

    def reset_parameters(self, alg: str = "noisy_identity") -> None:
        assert alg in {
            "ortho",
            "uniform",
            "normal",
            "identity",
            "near_identity",
            "noisy_identity",
            "perm",
            "near_perm",
        }
        if alg == "ortho":
            nn.init.orthogonal_(self.weight)
        elif alg == "uniform":
            nn.init.constant_(self.weight, 1 / self.n_waveguides)
            set_torch_deterministic(0)
            self.weight.data += torch.randn_like(self.weight).mul(0.01)
        elif alg == "normal":
            set_torch_deterministic(0)
            torch.nn.init.xavier_normal_(self.weight)
            self.weight.data += self.weight.data.std() * 3
        elif alg == "identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
        elif alg == "near_identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
            margin = 0.5
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
        elif alg == "noisy_identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
            margin = 0.3
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
            self.weight.data.add_(torch.randn_like(self.weight.data) * 0.05)

        elif alg == "perm":
            self.weight.data.copy_(
                torch.eye(self.weight.size(0), device=self.device)[
                    torch.randperm(self.weight.size(0))
                ]
            )
        elif alg == "near_perm":
            set_torch_deterministic(0)
            self.weight.data.copy_(
                torch.eye(self.weight.size(0), device=self.device)[
                    torch.randperm(self.weight.size(0))
                ]
            )
            margin = 0.9
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
        nn.init.constant_(self.alm_multiplier, 0)

    def enable_identity_forward(self):
        self.identity_forward = True

    def set_butterfly_forward(self, forward: bool = True, level: int = 0) -> None:
        """Set the crossing layer to a certain stage in butterfly transform

        Args:
            forward (bool, optional): forward butterfly or reversed butterfly. Defaults to True.
            level (int, optional): which level or stage in the butterfly transform. Defaults to 0.
        """
        initial_indices = torch.arange(
            0, self.n_waveguides, dtype=torch.long, device=self.device
        )
        block_size = 2 ** (level + 2)
        if forward:
            indices = (
                initial_indices.view(
                    -1, self.n_waveguides // block_size, 2, block_size // 2
                )
                .transpose(dim0=-2, dim1=-1)
                .contiguous()
                .view(-1)
            )
        else:
            indices = initial_indices.view(
                -1, self.n_waveguides // block_size, block_size
            )
            indices = (
                torch.cat([indices[..., ::2], indices[..., 1::2]], dim=-1)
                .contiguous()
                .view(-1)
            )
        eye = torch.eye(self.n_waveguides, device=self.device)[indices, :]
        self.weight.data.copy_(eye)

    def build_weight(self):
        if self.identity_forward:
            return self.weight
        """Enforce DN hard constraints with reparametrization"""

        if self.symmetry:
            weight1 = self.weight[: self.weight.size(0) // 2]
            weight2 = torch.flipud(torch.fliplr(weight1))
            weight = torch.cat([weight1, weight2], dim=0)
        else:
            weight = self.weight
        weight = weight.abs()  # W >= 0
        weight = weight / weight.data.sum(dim=0, keepdim=True)  # Wx1=1
        weight = weight / weight.data.sum(dim=1, keepdim=True)  # W^Tx1=1

        with torch.no_grad():
            perm_loss = (
                weight.data.norm(p=1, dim=0).sub(weight.data.norm(p=2, dim=0)).mean()
                + (1 - weight.data.norm(p=2, dim=1)).mean()
            )
        if perm_loss < 0.05:
            weight = hard_diff_round(
                weight
            )  # W -> P # once it is very close to permutation, it will be trapped and legalized without any gradients.
        return weight

    def get_ortho_loss(self):
        weight = self.build_weight()
        loss = torch.nn.functional.mse_loss(weight.matmul(weight.t()), self.eye)
        return loss

    def get_perm_loss(self):
        """
        Permutation constraint relaxation
        https://www.math.uci.edu/~jxin/AutoShuffleNet_KDD2020F.pdf"""
        weight = self.build_weight()
        loss = (
            weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0)).mean()
            + (1 - weight.norm(p=2, dim=1)).mean()
        )
        return loss

    def get_alm_perm_loss(self, rho: float = 0.1) -> Tensor:
        """Augmented Lagrangian loss of permutation constraint

        Args:
            rho (float, optional): Rho in ALM formulation. Defaults to 0.1.

        Returns:
            Tensor: Loss
        """
        if self.identity_forward:
            return 0
        ## quadratic tern is also controlled multiplier
        weight = self.build_weight()
        d_weight_r = weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0))
        # d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
        d_weight_c = 1 - weight.norm(p=2, dim=1)
        loss = self.alm_multiplier[0].dot(
            d_weight_r + rho / 2 * d_weight_r.square()
        ) + self.alm_multiplier[1].dot(d_weight_c + rho / 2 * d_weight_c.square())
        return loss

    def update_alm_multiplier(
        self, rho: float = 0.1, max_lambda: Optional[float] = None
    ):
        """Update the ALM multiplier lambda

        Args:
            rho (float, optional): Rho in ALM formulation. Defaults to 0.1.
            max_lambda (Optional[float], optional): Maximum multiplier value. Defaults to None.
        """

        if self.identity_forward:
            return
        with torch.no_grad():
            weight = self.build_weight().detach()
            d_weight_r = weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0))
            d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
            self.alm_multiplier[0].add_(
                rho * (d_weight_r + rho / 2 * d_weight_r.square())
            )
            self.alm_multiplier[1].add_(
                rho * (d_weight_c + rho / 2 * d_weight_c.square())
            )
            if max_lambda is not None:
                self.alm_multiplier.data.clamp_max_(max_lambda)

    def get_crossing_loss(self, alg: str = "mse") -> Tensor:
        """Penalize permutation matrices if having too many crossings.

        Args:
            alg (str, optional): Which alg to use [kl, mse]. Defaults to "mse".

        Returns:
            Tensor: Loss
        """
        weight = self.build_weight()
        n = self.n_waveguides
        if alg == "kl":
            return torch.kl_div(weight, self.eye).mean()
        elif alg == "mse":
            return torch.nn.functional.mse_loss(weight, self.eye)

    def _get_num_crossings(self, in_indices):
        res = 0
        for idx, i in enumerate(in_indices):
            for j in range(idx + 1, len(in_indices)):
                if i > in_indices[j]:
                    res += 1
        return res

    def get_num_crossings(self) -> int:
        """Calculate the number of crossings in the current permutation matrix

        Returns:
            int: #Crossings
        """
        if self.identity_forward:
            return 0
        with torch.no_grad():
            weight = self.build_weight().detach()
            in_indices = weight.max(dim=0)[1].cpu().numpy().tolist()
            return self._get_num_crossings(in_indices)

    def forward(self, x: Tensor) -> Tensor:
        # x[..., n_waveguides] real/complex
        if self.identity_forward:
            return x
        if self.fast_mode and self.indices is not None:
            ### indexing as fast permutation layer forward
            return torch.complex(x.real[..., self.indices], x.imag[..., self.indices])

        weight = self.build_weight().to(x.dtype)
        x = x.matmul(weight.t())

        return x

    @property
    def arch_space(self):
        return [1]  # only one selection, this is a differentiable layer

    def count_sample_params(self):
        return 0

    def extra_repr(self) -> str:
        s = f"n_waveguides={self.n_waveguides}, sample_arch={self.sample_arch}, trainable={self.trainable}"
        return s

    def set_cr_noise(self, noise_std: float = 0.0):
        self.cr_noise_std = noise_std

    def check_perm(self, indices):
        return tuple(range(len(indices))) == tuple(
            sorted(indices.cpu().numpy().tolist())
        )

    def unitary_projection(self, w, n_step=10, t=0.005, noise_std=0.01):
        w = w.div(t).softmax(dim=-1).round()
        legal_solution = []
        for i in range(n_step):
            u, s, v = w.svd()
            w = u.matmul(v.permute(-1, -2))
            w.add_(torch.randn_like(w) * noise_std)
            w = w.div(t).softmax(dim=-1)
            indices = w.argmax(dim=-1)
            if self.check_perm(indices):
                n_cr = self._get_num_crossings(indices.cpu().numpy().tolist())
                legal_solution.append((n_cr, w.clone().round()))
        legal_solution = sorted(legal_solution, key=lambda x: x[0])
        w = legal_solution[0][1]
        return w

    def fix_arch_solution(self):
        """Fix the permutation matrices as indices for fast forward"""
        with torch.no_grad():
            weight = self.build_weight().detach().data
            self.indices = torch.argmax(weight, dim=1)
            assert self.check_perm(
                self.indices
            ), f"{self.indices.cpu().numpy().tolist()}"
            self.fast_mode = True
            self.weight.requires_grad_(False)


class SuperMeshBase(SuperOpticalModule):
    def __init__(self, arch: dict = None, device=torch.device("cuda:0")):
        super().__init__(n_waveguides=arch["n_waveguides"])
        self.arch = arch
        self.device = device

        self.n_blocks = arch.get("n_blocks", None)
        assert (
            self.n_blocks % 2 == 0
        ), f"n_blocks in arch should always be an even number, but got {self.n_blocks}"
        self.n_layers_per_block = arch.get("n_layers_per_block", None)
        self.n_front_share_blocks = arch.get("n_front_share_blocks", None)

        self.sample_n_blocks = None

        self.super_layers_all = self.build_super_layers()
        self.fast_mode = False
        self.fast_arch_mask = None

    def build_super_layers(self):
        raise NotImplementedError

    def set_sample_arch(self, sample_arch):
        for k, layer_arch in enumerate(sample_arch[:-1]):
            self.super_layers_all[k].set_sample_arch(layer_arch)
        self.sample_n_blocks = sample_arch[-1]

    def reset_parameters(self) -> None:
        for m in self.super_layers_all:
            m.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for k in range(len(self.super_layers_all)):
            if k < self.sample_n_blocks * self.n_layers_per_block:
                x = self.super_layers_all[k](x)
        return x

    def count_sample_params(self):
        n_params = 0
        for layer_idx, layer in enumerate(self.super_layers_all):
            if layer_idx < self.sample_n_blocks * self.n_layers_per_block:
                n_params += layer.count_sample_params()
        return n_params

    @property
    def arch_space(self) -> List:
        space = [layer.arch_space for layer in self.super_layers_all]
        # for the number of sampled blocks
        space.append(
            list(range(self.n_front_share_blocks, self.n_blocks + 1, 2))
        )  # n_sample_block must be even number
        return space


class SuperMeshADEPT(SuperMeshBase):
    """
    Super Photonic Mesh for automatic differentiable PTC topology search
    J. Gu, et al., "ADEPT: Automatic Differentiable DEsign of Photonic Tensor Cores", DAC 2022
    https://arxiv.org/pdf/2112.08703.pdf
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_gumbel_temperature()
        self.build_sampling_coefficients()
        sample_arch = get_named_sample_arch(self.arch_space, name="largest")
        self.set_sample_arch(sample_arch)

    def build_sampling_coefficients(self):
        self.sampling_coeff = torch.nn.Parameter(torch.zeros(self.n_blocks, 2) + 0.5)
        if self.n_front_share_blocks > 0:
            self.sampling_coeff.data[
                self.n_blocks // 2
                - self.n_front_share_blocks // 2 : self.n_blocks // 2,
                0,
            ] = -100
            self.sampling_coeff.data[
                self.n_blocks // 2
                - self.n_front_share_blocks // 2 : self.n_blocks // 2,
                1,
            ] = 100  # force to choose the block
            self.sampling_coeff.data[-self.n_front_share_blocks // 2 :, 0] = (
                -100
            )  # force to choose the block
            self.sampling_coeff.data[-self.n_front_share_blocks // 2 :, 1] = (
                100  # force to choose the block
            )

    def set_gumbel_temperature(self, T: float = 5.0):
        self.gumbel_temperature = T

    def build_arch_mask(self, mode="gumbel_soft", batch_size: int = 32):
        logits = self.sampling_coeff
        if mode == "gumbel_hard":
            # sample one-hot vectors
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        if mode == "gumbel_soft":
            # sample soft coefficients
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
        if mode == "gumbel_soft_batch":
            # different coefficients for each example
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                torch.log_softmax(logits, dim=-1).unsqueeze(0).repeat(batch_size, 1, 1),
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
        elif mode == "softmax":
            # simple softmax without gumbel noise
            self.arch_mask = torch.softmax(
                logits / self.gumbel_temperature,
                dim=-1,
            )
        elif mode == "largest":
            # always sample the one with largest possibility
            logits = torch.cat([logits[..., 0:1] - 100, logits[..., 1:] + 100], dim=-1)
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                torch.log_softmax(logits, dim=-1),
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        elif mode == "smallest":
            if self.n_front_share_blocks > 0:
                logits = logits.view(2, -1, 2)

                logits_small = torch.cat(
                    [
                        logits[:, : -self.n_front_share_blocks // 2, 0:1] + 100,
                        logits[:, : -self.n_front_share_blocks // 2, 1:] - 100,
                    ],
                    dim=-1,
                )
                logits = torch.cat(
                    [logits_small, logits[:, -self.n_front_share_blocks // 2 :, :]],
                    dim=1,
                ).view(-1, 2)
            else:
                logits = torch.cat(
                    [logits[..., 0:1] + 200, logits[..., 1:] - 200], dim=-1
                )
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        elif mode == "random":
            logits = torch.ones_like(logits)
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )

    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()

        for i in range(self.arch["n_blocks"]):
            super_layers_all.append(
                SuperDCFrontShareLayer(
                    n_waveguides=self.n_waveguides,
                    offset=(
                        i % 2 if self.interleave_dc else 0
                    ),  # interleaved design space
                    trainable=True,
                    binary=True,
                    device=self.device,
                )
            )
            if i == self.n_blocks - 1:  # pseudo-permutation, which is an identity
                layer = SuperCRLayer(
                    n_waveguides=self.n_waveguides,
                    trainable=False,
                    symmetry=self.symmetry_cr,
                    device=self.device,
                )
                layer.reset_parameters(alg="identity")
                layer.enable_identity_forward()
                super_layers_all.append(layer)
            else:
                super_layers_all.append(
                    SuperCRLayer(
                        n_waveguides=self.n_waveguides,
                        trainable=True,
                        symmetry=self.symmetry_cr,
                        device=self.device,
                    )
                )

        return super_layers_all

    def build_ps_layers(
        self, grid_dim_x: int, grid_dim_y: int, miniblock: Tuple[int], w_bit: int = 32
    ) -> nn.ModuleList:
        ## each CONV or Linear need to explicit build ps layers as the main parameters using this function
        ## Assume conv and layer uses blocking matrix multiplication.
        ## Each block can have a batch (i.e., grid_dim_y x grid_dim_x, *miniblock) of PS columns
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        super_ps_layers = nn.ModuleList()
        for i in range(self.arch["n_blocks"]):
            if self.share_ps in {"global", "none"}:
                share_uv = self.share_ps
            elif self.share_ps in {"row_col"}:
                share_uv = "col" if i < self.arch["n_blocks"] // 2 else "row"
            super_ps_layers.append(
                SuperBatchedPSLayer(
                    grid_dim_x=grid_dim_x,
                    grid_dim_y=grid_dim_y,
                    miniblock=miniblock,
                    w_bit=w_bit,
                    share_uv=share_uv,
                    n_waveguides=self.n_waveguides,
                    trainable=True,
                    device=self.device,
                )
            )
        return super_ps_layers

    def set_identity(self) -> None:
        self.set_identity_cr()

    def set_identity_cr(self) -> None:
        for m in self.super_layers_all:
            if isinstance(m, (SuperCRLayer,)):
                m.reset_parameters(alg="identity")

    def forward(
        self, x: Tensor, super_ps_layers: nn.ModuleList, first_chunk: bool = True
    ) -> Tensor:
        """super_ps_layers: nn.ModuleList passed from each caller"""
        if first_chunk:
            # first half chunk is always used for V
            start_block, end_block = 0, self.sample_n_blocks // 2
        else:
            # second half chunk is always used for U
            start_block, end_block = (
                self.n_blocks // 2,
                self.n_blocks // 2 + self.sample_n_blocks // 2,
            )

        # Fast mode: re-training with fixed arch
        if self.fast_mode:
            for i in range(start_block, end_block):
                index = self.fast_arch_mask[i]
                if index == 1:
                    if super_ps_layers is not None:
                        x = super_ps_layers[i](
                            x
                        )  # pass through independent phase shifters before each block
                    for j in range(self.n_layers_per_block):
                        layer_idx = i * self.n_layers_per_block + j
                        x = self.super_layers_all[layer_idx](x)

            return x

        # supermesh search stage
        if self.training:
            for i in range(start_block, end_block):
                res = x
                if super_ps_layers is not None:
                    x = super_ps_layers[i](
                        x
                    )  # pass through independent phase shifters before each block

                for j in range(self.n_layers_per_block):
                    layer_idx = i * self.n_layers_per_block + j
                    x = self.super_layers_all[layer_idx](x)

                # residual path to skip this block
                if self.arch_mask.dim() == 2:  # scalar gumbel
                    x = self.arch_mask[i, 0] * res + self.arch_mask[i, 1] * x
                else:
                    # x [bs, ....], mask [bs, ]
                    arch_mask = self.arch_mask[:, i, :].view(
                        -1, *([1] * (x.dim() - 1)), 2
                    )
                    x = arch_mask[..., 0] * res + arch_mask[..., 1] * x
        else:  # inference, validation, test
            arch_mask = torch.nn.functional.gumbel_softmax(
                self.sampling_coeff.data,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
            for i in range(start_block, end_block):
                res = x
                if super_ps_layers is not None:
                    x = super_ps_layers[i](
                        x
                    )  # pass through independent phase shifters before each block

                for j in range(self.n_layers_per_block):
                    layer_idx = i * self.n_layers_per_block + j
                    x = self.super_layers_all[layer_idx](x)

                # residual path to skip this block
                x = arch_mask[i, 0] * res + arch_mask[i, 1] * x

        return x

    @lru_cache(maxsize=16)
    def _build_probe_matrix(
        self, grid_dim_x: int, grid_dim_y: int, miniblock: Tuple[int]
    ):
        ## miniblock here is the miniblock[:-2] from ONNBaselayer
        eye = torch.eye(self.n_waveguides, dtype=torch.cfloat, device=self.device)[
            (None,) * (2 + len(miniblock))
        ]
        if self.share_ps == "global":
            # [k1, ..., k2]
            eye_U = eye_V = (
                eye.expand(1, 1, *miniblock, -1, -1).movedim(-2, 0).contiguous()
            )
        elif self.share_ps == "row_col":
            # [k1, q, ..., k2]
            eye_V = (
                eye.expand(1, grid_dim_x, *miniblock, -1, -1)
                .movedim(-2, 0)
                .contiguous()
            )
            # [k1, p, ..., k2]
            eye_U = (
                eye.expand(grid_dim_y, 1, *miniblock, -1, -1)
                .movedim(-2, 0)
                .contiguous()
            )
        elif self.share_ps == "none":
            # [k1, p, q, ..., k2]
            eye_V = eye_U = (
                eye.expand(grid_dim_y, grid_dim_x, *miniblock, -1, -1)
                .movedim(-2, 0)
                .contiguous()
            )
        else:
            raise NotImplementedError
        return eye_U, eye_V

    def get_UV(
        self,
        super_ps_layers: nn.ModuleList,
        grid_dim_x: int,
        grid_dim_y: int,
        miniblock: Tuple[int],
    ) -> Tuple[Tensor, Tensor]:
        # return U and V
        eye_U, eye_V = self._build_probe_matrix(
            grid_dim_x, grid_dim_y, miniblock=miniblock
        )

        # print(self.eye_V.size())
        V = self.forward(eye_V, super_ps_layers, first_chunk=True)  # [k1,p,q,..., k2]
        V = V.movedim(0, -2)  # [p,q,..., k1, k2]

        U = self.forward(eye_U, super_ps_layers, first_chunk=False)  # [k1,p,q,..., k2]
        U = U.movedim(0, -2)  # [p,q,..., k1, k2]

        ## re-normalization to control the variance the relaxed U and V
        ## after permutaiton relaxation, U, V might not be unitary
        ## this normalization stabilize the statistics
        ## this has no effects on unitary matrices when training converges
        if not self.fast_mode:
            U = U / U.data.norm(p=2, dim=-1, keepdim=True)  # unit row L2 norm
            V = V / V.data.norm(p=2, dim=-2, keepdim=True)  # unit col L2 norm
        return U, V

    def get_weight_matrix(
        self, super_ps_layers: nn.ModuleList, sigma: Tensor
    ) -> Tensor:
        # sigma [p, q, ..., k2], unique parameters for each caller
        # super_ps_layers, unique parameters for each caller

        U, V = self.get_UV(
            super_ps_layers,
            grid_dim_x=sigma.size(1),
            grid_dim_y=sigma.size(0),
            miniblock=sigma.shape[2:-1],
        )

        # U [1,1,k,k] or [p,1,k,k] or [p,q,k,k]
        # V [1,1,k,k] or [1,q,k,k] or [p,q,k,k]
        sv = sigma.unsqueeze(-1).mul(V)  # [p,q,k,1]*[p,q,k,k]->[p,q,k,k]
        weight = U.matmul(sv)  # [p,q,k,k] x [p,q,k,k] -> [p,q,k,k]
        return weight

    def fix_layer_solution(self):
        ## fix DC and CR solution
        for m in self.super_layers_all:
            m.fix_arch_solution()

    def fix_block_solution(self):
        self.fast_arch_mask = self.sampling_coeff.argmax(dim=-1)
        self.sampling_coeff.requires_grad_(False)
        self.fast_mode = True


class SuperMeshMZI(SuperMeshADEPT):
    """Construct Clements style MZI array"""

    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()
        ## force the block number
        self.arch["n_blocks"] = self.n_blocks = 4 * self.n_waveguides
        for i in range(self.arch["n_blocks"]):
            layer = SuperDCFrontShareLayer(
                n_waveguides=self.n_waveguides,
                n_front_share_waveguides=self.n_front_share_waveguides,
                offset=(i // 2) % 2,  # 001100110011...
                trainable=False,
                binary=True,
                device=self.device,
            )
            layer.weight.data.fill_(-1)  # after binarization, all DCs are maintained
            super_layers_all.append(layer)
            layer = SuperCRLayer(
                n_waveguides=self.n_waveguides,
                trainable=False,
                symmetry=self.symmetry_cr,
                device=self.device,
            )
            layer.reset_parameters(alg="identity")
            layer.enable_identity_forward()
            super_layers_all.append(layer)

        return super_layers_all


class SuperMeshButterfly(SuperMeshADEPT):
    """Construct butterfly-style mesh, e.g., FFT-ONN"""

    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()
        ## force the block number
        self.arch["n_blocks"] = self.n_blocks = int(2 * np.log2(self.n_waveguides))
        for i in range(self.arch["n_blocks"]):
            layer = SuperDCFrontShareLayer(
                n_waveguides=self.n_waveguides,
                n_front_share_waveguides=self.n_front_share_waveguides,
                offset=0,  # 0000...
                trainable=False,
                binary=True,
                device=self.device,
            )
            layer.weight.data.fill_(-1)
            super_layers_all.append(layer)
            layer = SuperCRLayer(
                n_waveguides=self.n_waveguides,
                trainable=False,
                symmetry=self.symmetry_cr,
                device=self.device,
            )
            if i == self.n_blocks // 2 - 1 or i == self.n_blocks - 1:
                layer.reset_parameters(alg="identity")
                layer.enable_identity_forward()
            else:
                forward = i < self.n_blocks // 2
                if forward:
                    level = i % (self.n_blocks // 2)
                else:
                    level = (self.n_blocks // 2 - 2) - i % (self.n_blocks // 2)
                layer.set_butterfly_forward(forward=forward, level=level)
            super_layers_all.append(layer)

        return super_layers_all


super_layer_name_dict = {
    "adept": SuperMeshADEPT,
    "mzi_clements": SuperMeshMZI,
    "butterfly": SuperMeshButterfly,
}
