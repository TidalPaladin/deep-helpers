#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import cast

import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from .helpers import try_compile_model
from .tasks import Task


def _run_common_setup(cli: LightningCLI, model: pl.LightningModule, mode: str):
    config_for_mode = getattr(cli.config, mode)
    if config_for_mode.compile:
        try_compile_model(model)
    if (precision := config_for_mode.float32_matmul_precision) is not None:
        rank_zero_info(f"Setting float32_matmul_precision to '{precision}'")
        torch.set_float32_matmul_precision(precision)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument("--lr_scheduler_monitor", default="val/loss")
        parser.add_argument("--lr_scheduler_interval", default="epoch")
        parser.link_arguments("lr_scheduler_monitor", "model.init_args.lr_scheduler_monitor")
        parser.link_arguments("lr_scheduler_interval", "model.init_args.lr_scheduler_interval")
        parser.add_argument("--compile", default=False, action="store_true")
        parser.add_argument("--float32_matmul_precision", default=None, choices=[None, "medium", "high", "highest"])

    def fit(self, model, **kwargs):
        _run_common_setup(self, model, "fit")
        self.trainer.fit(cast(pl.LightningModule, model), **kwargs)

    def test(self, model, **kwargs):
        _run_common_setup(self, model, "test")
        self.trainer.test(cast(pl.LightningModule, model), **kwargs)


def main():
    cli = CLI(
        Task,
        pl.LightningDataModule,
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()
