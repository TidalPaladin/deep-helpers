#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from .tasks import Task


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument("--lr_scheduler_monitor", default="val/loss")
        parser.add_argument("--lr_scheduler_interval", default="epoch")
        parser.link_arguments("lr_scheduler_monitor", "model.init_args.lr_scheduler_monitor")
        parser.link_arguments("lr_scheduler_interval", "model.init_args.lr_scheduler_interval")
        parser.add_argument("--compile", default=False, action="store_true")

    def fit(self, model, **kwargs):
        try:
            logging.info("Compiling model...")
            model = torch.compile(model)
        except Exception as e:
            logging.exception("Failed to compile model, proceeding with uncompiled model.", exc_info=e)
        assert isinstance(model, pl.LightningModule)
        self.trainer.fit(model, **kwargs)


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
