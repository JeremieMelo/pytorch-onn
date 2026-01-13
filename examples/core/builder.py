"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 18:23:17
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 18:23:17
"""

from typing import Tuple

import torch
import torch.nn as nn
from core.models import *
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.general import logger
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from pyutils.lr_scheduler import CosineAnnealingWarmupRestarts
from pyutils.optimizer import SAM, RAdam
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device

__all__ = [
    "make_dataloader",
    "make_model",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader() -> Tuple[DataLoader, DataLoader]:
    train_dataset, validation_dataset = get_dataset(
        configs.dataset.name,
        configs.dataset.img_height,
        configs.dataset.img_width,
        dataset_dir=configs.dataset.root,
        transform=configs.dataset.transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    return train_loader, validation_loader


def make_model(device: Device, random_state: int = None) -> nn.Module:
    model = eval(configs.model.name)(
        img_height=configs.dataset.img_height,
        img_width=configs.dataset.img_width,
        in_channels=configs.dataset.in_channels,
        num_classes=configs.dataset.num_classes,
        kernel_list=configs.model.kernel_list,
        kernel_size_list=configs.model.kernel_size_list,
        stride_list=configs.model.stride_list,
        padding_list=configs.model.padding_list,
        dilation_list=configs.model.dilation_list,
        pool_out_size=configs.model.pool_out_size,
        hidden_list=configs.model.hidden_list,
        block_list=configs.model.block_list,
        mode=configs.model.mode,
        decompose_alg=configs.model.decompose_alg,
        photodetect=False,
        bias=False,
        device=device,
    ).to(device)

    model.reset_parameters(random_state)
    model.set_weight_bitwidth(getattr(configs, "quantize.weight_bit", 32))
    model.set_input_bitwidth(getattr(configs, "quantize.input_bit", 32))
    model.set_gamma_noise(getattr(configs, "noise.gamma_noise_std", 0))
    model.set_crosstalk_factor(getattr(configs, "noise.crosstalk_factor", 0))
    model.set_phase_variation(getattr(configs, "noise.phase_noise_std", 0))

    return model


def make_optimizer(model: nn.Module, name: str = None) -> Optimizer:
    name = name or configs.optimizer.name.lower()
    if name == "sgd":
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            momentum=getattr(configs.optimizer, "momentum", 0.9),
            weight_decay=getattr(configs.optimizer, "weight_decay", 1e-5),
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=getattr(configs.optimizer, "weight_decay", 1e-5),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=getattr(configs.optimizer, "weight_decay", 0.01),
        )
    elif name == "radam":
        optimizer = RAdam(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=getattr(configs.optimizer, "weight_decay", 1e-5),
        )
    elif name == "sam":
        base_optimizer_name = configs.optimizer.base_optimizer.name.lower()
        assert "sam" not in base_optimizer_name, "Base optimizer cannot be SAM"
        base_optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "radam": RAdam,
        }[base_optimizer_name]
        optimizer = SAM(
            (p for p in model.parameters() if p.requires_grad),
            base_optimizer=base_optimizer,
            rho=getattr(configs.optimizer, "rho", 0.5),
            adaptive=getattr(configs.optimizer, "adaptive", True),
            **{
                k: v
                for k, v in configs.optimizer.base_optimizer.dict().items()
                if k != "name"
            }
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    name = configs.scheduler.name.lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs.run.n_epochs,
            eta_min=getattr(configs, "scheduler.lr_min", 0),
        )
    elif name == "warmup_cosine":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=getattr(configs, "scheduler.lr_min", 0),
            warmup_steps=getattr(configs, "scheduler.n_warm_epochs", 5),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion() -> nn.Module:
    name = configs.criterion.name.lower()
    if name == "nll":  # negative logarithmic loss
        criterion = nn.NLLLoss()
    elif name == "mse":  # mean square error
        criterion = nn.MSELoss()
    elif name == "l1":  # L1 loss
        criterion = nn.L1Loss()
    elif name == "ce":  # cross-entropy
        criterion = nn.CrossEntropyLoss()
    elif (
        name == "kll_mixed"
    ):  # knowledge distillation loss with soft KL divegence and hard CE loss
        criterion = KLLossMixed(
            T=getattr(configs.criterion, "temperature", 1),
            alpha=getattr(configs.criterion, "alpha", 0.9),
        )
    elif name == "adaptive_loss_soft":  # alphaNet, adaptive alpha divergence
        criterion = AdaptiveLossSoft(
            alpha_min=getattr(configs.criterion, "alpha_min", -1),
            alpha_max=getattr(configs.criterion, "alpha_max", 1),
            iw_clip=getattr(configs.criterion, "iw_clip", 1000),
        )
    else:
        raise NotImplementedError(name)
    return criterion
