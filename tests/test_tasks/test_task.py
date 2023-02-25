#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import pytest
import pytorch_lightning as pl
import torch
import yaml
from deep_helpers.cli import main as cli_main

from tests.conftest import CustomTask


class TestTask:
    def test_configure_optimizers(self):
        optimizer_init = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.001}}
        lr_scheduler_init = {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 10, "gamma": 0.1},
        }
        lr_scheduler_interval = "epoch"
        lr_scheduler_monitor = "val_loss"

        task = CustomTask(optimizer_init, lr_scheduler_init, lr_scheduler_interval, lr_scheduler_monitor)
        result = task.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.Adam)
        assert isinstance(result["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
        assert result["lr_scheduler"]["monitor"] == lr_scheduler_monitor
        assert result["lr_scheduler"]["interval"] == lr_scheduler_interval

    @pytest.mark.parametrize("named_datasets", [False, True])
    @pytest.mark.parametrize("stage", ["fit", "test"])
    @pytest.mark.parametrize(
        "accelerator",
        [
            "cpu",
            pytest.param("gpu", marks=pytest.mark.cuda),
        ],
    )
    def test_run(self, task, default_root_dir, accelerator, datamodule, stage, named_datasets):
        if accelerator == "gpu":
            precision = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "16"
        else:
            precision = 32

        task.named_datasets = named_datasets
        trainer = pl.Trainer(
            fast_dev_run=True,
            default_root_dir=default_root_dir,
            precision=precision,
            accelerator=accelerator,
            devices=1,
        )

        dm = datamodule(batch_size=4)
        func = getattr(trainer, stage)
        func(task, datamodule=dm)

    @pytest.mark.parametrize("stage", ["fit", "test"])
    @pytest.mark.parametrize(
        "accelerator",
        [
            "cpu",
            pytest.param("gpu", marks=pytest.mark.cuda),
        ],
    )
    def test_cli(self, mocker, tmp_path, task, datamodule, default_root_dir, accelerator, stage):
        if accelerator == "gpu":
            precision = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "16"
        else:
            precision = 32

        base_config = {
            "trainer": {
                "fast_dev_run": True,
                "default_root_dir": str(default_root_dir),
                "precision": str(precision),
                "accelerator": accelerator,
                "devices": 1,
            },
            "model": {
                "class_path": f"tests.conftest.{task.__class__.__name__}",
                "init_args": {},
            },
            "data": {
                "class_path": f"tests.conftest.{datamodule().__class__.__name__}",
                "init_args": {
                    "batch_size": 4,
                },
            },
        }

        config = {
            "fit": base_config,
            "test": base_config,
        }

        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        sys.argv = [
            sys.argv[0],
            "--config",
            str(tmp_path / "config.yaml"),
            stage,
        ]

        try:
            cli_main()
        except SystemExit as e:
            raise e.__context__ if e.__context__ is not None else e
