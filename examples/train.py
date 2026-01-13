"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 23:21:35
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 23:21:35
"""

#!/usr/bin/env python3
# coding=UTF-8
import argparse
import datetime
import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import builder
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.metric import accuracy
from pyutils.optimizer import SAM
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    disable_bn,
    enable_bn,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    acc_meter = AverageMeter("Acc", ":.4f")
    loss_meter = AverageMeter("Loss", ":.4f")

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if isinstance(optimizer, SAM):
            ### first step
            optimizer.first_step(zero_grad=True)
            ### second step with frozen bn
            disable_bn(model)
            criterion(model(data), target).backward()
            optimizer.second_step(zero_grad=False)
            enable_bn(model)
        else:
            optimizer.step()
        step += 1

        acc = accuracy(output, target)
        acc_meter.update(acc, data.size(0))
        loss_meter.update(loss.item(), data.size(0))

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info(
                "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] {} {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss_meter,
                    acc_meter,
                )
            )
        if model.mode == "usv":
            ## perform projected gradient descent to maintain U and V in the unitary subspace
            model.unitary_projection()

    scheduler.step()
    lg.info(f"Train Accuracy: {acc_meter.avg*100:.2f} %")


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    acc_meter = AverageMeter("Acc", ":.4f")
    loss_meter = AverageMeter("Loss", ":.4f")
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            loss = criterion(output, target).data.item()
            acc = accuracy(output, target)
            loss_meter.update(loss, data.size(0))
            acc_meter.update(acc, data.size(0))

    loss_vector.append(loss_meter.avg)
    accuracy_vector.append(acc_meter.avg * 100)

    lg.info(
        "\nValidation Epoch: {} Average loss: {:.4f}, Accuracy: {:.2f} %\n".format(
            epoch, loss_meter.avg, acc_meter.avg * 100
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )

    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}"
    if (
        hasattr(configs.checkpoint, "model_comment")
        and len(configs.checkpoint.model_comment) > 0
    ):
        checkpoint += f"_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} starts. PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if int(configs.checkpoint.resume):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=True,
            )

            lg.info("Validate resumed model...")
            validate(model, validation_loader, 0, criterion, lossv, accv, device)

        for epoch in range(0, int(configs.run.n_epochs)):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                device,
            )
            validate(
                model,
                validation_loader,
                epoch,
                criterion,
                lossv,
                accv,
                device,
            )
            saver.save_model(
                model,
                accv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True,
            )
        lg.info(f"Convert model from {configs.model.mode} mode to phase mode...")
        model.sync_parameters(src=configs.model.mode)
        model.switch_mode_to("phase")
        lg.info(f"Validate converted model in phase mode...")
        validate(
            model,
            validation_loader,
            epoch,
            criterion,
            lossv,
            accv,
            device,
        )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
