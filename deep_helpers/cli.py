#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import cast

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from .helpers import try_compile_model
from .tasks import Task


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

    def test(self, model, **kwargs):
        if self.config.test.compile:
            try_compile_model(model)
        self.trainer.test(cast(pl.LightningModule, model), **kwargs)


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
