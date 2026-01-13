"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.compute import add_gaussian_noise
from pyutils.general import logger
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init

from torchonn.devices.mrr import MRRConfig_5um_HQ
from torchonn.layers.base_layer import ONNBaseMatMul

from .utils import merge_chunks, partition_chunks

__all__ = [
    "TeMPOBlockMatMul",
]


@MODELS.register_module()
class TeMPOBlockMatMul(ONNBaseMatMul):
    """
    TeMPO, Zhang, Yin et al., JAP 2024
    blocking Conv2d layer constructed by time-multiplexed dynamic photonic tensor core.
    """

    ## default configs
    default_cfgs = dict(
        miniblock=(
            1,
            1,
            4,
            4,
        ),  # [#tiles, pe per tile, row, col] # i.e., [R, C, k1, k2]
        mode="weight",
        MRRConfig=MRRConfig_5um_HQ,
        mb_bit=32,
        ma_bit=32,
        out_bit=32,
        v_max=10.8,
        v_pi=4.36,
        photodetect="incoherent",
        device=torch.device("cpu"),
    )
    __mode_list__ = ["weight"]

    def __init__(
        self,
        **cfgs,
    ):
        super().__init__()
        self.load_cfgs(
            **cfgs,
        )

        ### build trainable parameters
        # self.build_parameters(mode=self.mode)
        ### build transform
        self.build_transform()

        ### default set to slow forward
        self.disable_fast_forward()

        self.set_matrix_A_noise(0)
        self.set_matrix_B_noise(0)
        self.set_output_noise(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        # if bias:
        #     self.bias = Parameter(torch.zeros(out_features, device=self.device))
        # else:
        #     self.register_parameter("bias", None)

        # self.reset_parameters()

    def load_cfgs(
        self,
        **cfgs,
    ) -> None:
        super().load_cfgs(**cfgs)

        ## verify configs
        assert self.mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {self.mode}."
        )
        self.gamma = np.pi / self.v_pi**2

        assert self.photodetect in ["incoherent"], logger.error(
            f"Photodetect mode {self.photodetect} not implemented. only 'coherent' is supported."
        )

    # def build_parameters(self, mode: str = "weight") -> None:
    #     ## weight mode
    #     self.weight = torch.zeros(
    #         self.grid_dim_y,
    #         self.grid_dim_x,
    #         *self.miniblock,
    #         device=self.device,
    #     )
    #     self.input_scale = None
    #     self.weight_scale = None

    #     self.register_parameter_buffer(
    #         *self.get_param_buffer_groups(mode=mode),
    #     )

    # self.pack_weights()

    # def get_param_buffer_groups(self, mode: str) -> Tensor:
    #     buffer_groups = {}
    #     if mode == "weight":
    #         param_groups = {"weight": Parameter(self.weight)}
    #     else:
    #         raise NotImplementedError

    #     return param_groups, buffer_groups

    # def pack_weights(self):
    #     ## key is self.mode, which should match the src_name for weight_transform
    #     if self.mode == "weight":
    #         self.weights = {"weight": self.weight}
    #     else:
    #         raise NotImplementedError

    # def reset_parameters(self, mode: Optional[str] = None) -> None:
    #     mode = mode or self.mode
    #     W = nn.Linear(
    #         self.in_features,
    #         self.out_features,
    #     ).weight.data.to(self.device)
    #     W = partition_chunks(W, self.weight.shape)

    #     if self.mode == "weight":
    #         self.weight.data.copy_(W)
    #     else:
    #         raise NotImplementedError

    #     if self.bias is not None:
    #         init.uniform_(self.bias, 0, 0)

    def switch_mode_to(self, mode: str) -> None:
        super().switch_mode_to(mode)

    def build_transform(self) -> None:
        ### quantization tool
        self.matrix_A_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.ma_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        self.matrix_B_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.mb_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )

        self.output_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.out_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        self.quantizer_dict = {
            "matrix_A": self.matrix_A_quantizer,
            "matrix_B": self.matrix_B_quantizer,
            "output": self.output_quantizer,
        }

        ## add input transform
        self.add_transform(
            "matrix_A", "matrix_A", {"matrix_A_transform": self._matrix_A_transform}
        )

        ## add weight transform
        if self.mode == "weight":
            ## add a transform called "build_weight" for parameter named "weight"
            self.add_transform(
                "matrix_B", "matrix_B", {"matrix_B_transform": self._matrix_B_transform}
            )

        ## add output transform
        self.add_transform(
            "output", "output", {"output_transform": self._output_transform}
        )

    def sync_parameters(
        self, src: str = "weight", steps: int = 1000, verbose: bool = False
    ) -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        pass
        # if src == "weight":
        #     param_list = [p for p in self.parameters() if p.requires_grad]
        #     target = merge_chunks(self.weight.data.clone())[
        #         : self.out_features, : self.in_features
        #     ]
        #     def build_weight_fn():
        #         return self.transform_weight(self.weights)["weight"]

        #     self.map_layer(
        #         target=target,
        #         param_list=param_list,
        #         build_weight_fn=build_weight_fn,
        #         mode="regression",
        #         num_steps=steps,
        #         verbose=verbose,
        #     )

        # else:
        #     raise NotImplementedError

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.mb_bit = w_bit
        self.matrix_B_quantizer.set_bit(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.ma_bit = in_bit
        self.matrix_A_quantizer.set_bit(in_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        self.output_quantizer.set_bit(out_bit)

    # def load_parameters(self, param_dict: Dict[str, Any]) -> None:
    #     """
    #     description: update parameters based on this parameter dictionary\\
    #     param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
    #     """
    #     super().load_parameters(param_dict=param_dict)

    def _matrix_A_transform(self, x: Tensor) -> Tensor:
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.ma_bit < 16:
            x = self.matrix_A_quantizer(x)  # [-alpha, alpha] or [0, alpha]
            alpha = self.matrix_A_quantizer.alpha.data
        else:
            alpha = x.data.abs().max()
        self.matrix_A_scale = alpha.item()

        if self.matrix_A_noise_std > 0:
            x = add_gaussian_noise(
                x, noise_std=self.matrix_A_noise_std * self.matrix_A_scale
            )

        return x

    def _matrix_B_transform(self, x: Tensor) -> Tensor:
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.mode == "weight":
            if self.mb_bit < 16:
                x = self.matrix_B_quantizer(x)  # [-alpha, alpha]
                alpha = self.matrix_B_quantizer.alpha.data
            else:
                alpha = x.data.abs().max()
            self.matrix_B_scale = alpha.item()

            if self.matrix_B_noise_std > 0:
                ## Warning: noise need to be added to normalized input
                x = add_gaussian_noise(
                    x, noise_std=self.matrix_B_noise_std * self.matrix_B_scale
                )

        else:
            raise NotImplementedError

        return x

    # def _matrix_B_transform(
    #     self, weights: Dict, update_list: set = {"S_scale"}
    # ) -> Tensor:
    #     """
    #     For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
    #     Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
    #     Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
    #     Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
    #     """
    #     if self.mode == "weight":
    #         weight = weights
    #         if self.w_bit < 16:
    #             weight = self.matrix_B_quantizer(weight)  # [-alpha, alpha]
    #             alpha = self.matrix_B_quantizer.alpha.data
    #         else:
    #             alpha = weight.data.abs().max()
    #         self.weight_scale = alpha.item()

    #         if self.weight_noise_std > 0:
    #             ## Warning: noise need to be added to normalized input
    #             weight = add_gaussian_noise(
    #                 weight, noise_std=self.weight_noise_std * self.weight_scale
    #             )

    #     else:
    #         raise NotImplementedError

    #     weight = merge_chunks(weight)[: self.out_features, : self.in_features]

    #     return weight  # this is normalized weights [-1, 1]

    def _output_transform(self, x: Tensor) -> Tensor:
        """
        For adding noise to the chip input, weight, and output, the noise is estimated on normalized computing
        Q(sum_N [(x_q/a + dx) * (w_q/b + dw) + dy]) * (ab) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw) + a*b*dy]) =
        Q(sum_N [(x_q + a*dx) * (w_q + b*dw)] + sqrt(N)*a*b*dy)
        """
        if self.output_noise_std > 0:
            x = add_gaussian_noise(
                x,
                noise_std=self.output_noise_std
                * self.matrix_A_scale
                * self.matrix_B_scale
                * self.x_dim_pad**0.5,
            )

        if self.out_bit < 16:
            x = self.output_quantizer(x)

        return x

    def _forward_impl(self, matrix_A: Tensor, matrix_B: Tensor) -> Tensor:

        output = torch.matmul(
            matrix_A,
            matrix_B,
        )
        return output
