#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from .tasks import Task


def try_compile_model(model: nn.Module) -> nn.Module:
    try:
        logging.info(f"Compiling {model.__class__.__name__}...")
        model = cast(nn.Module, torch.compile(model))  # type: ignore
    except Exception as e:
        logging.exception(f"Failed to compile {model.__class__.__name__}.", exc_info=e)
    return model


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument("--lr_scheduler_monitor", default="val/loss")
        parser.add_argument("--lr_scheduler_interval", default="epoch")
        parser.link_arguments("lr_scheduler_monitor", "model.init_args.lr_scheduler_monitor")
        parser.link_arguments("lr_scheduler_interval", "model.init_args.lr_scheduler_interval")
        parser.add_argument("--compile", default=False, action="store_true")

    def fit(self, model, **kwargs):
        if self.config.fit.compile:
            try_compile_model(model)
        self.trainer.fit(cast(pl.LightningModule, model), **kwargs)


def main():
    cli = CLI(
        Task,
        pl.LightningDataModule,
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
