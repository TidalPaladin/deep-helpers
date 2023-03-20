#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import yaml

from deep_helpers.cli import main as cli_main
from deep_helpers.structs import Mode
from deep_helpers.testing import checkpoint_factory
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
    def test_run(self, mocker, task, default_root_dir, accelerator, datamodule, stage, named_datasets):
        if accelerator == "gpu":
            precision = "bf16-mixed" if torch.cuda.get_device_capability(0)[0] >= 8 else "16"
        else:
            precision = 32

        log_spy = mocker.spy(task, "log")
        log_dict_spy = mocker.spy(task, "log_dict")

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
        log_spy.assert_called()
        log_dict_spy.assert_called()

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
            precision = "bf16-mixed" if torch.cuda.get_device_capability(0)[0] >= 8 else "16"
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

    @pytest.mark.parametrize("from_env", [False, True])
    def test_create(self, mocker, task, from_env):
        task.__class__.CHECKPOINT_ENV_VAR = "TEST_CHECKPOINT_PATH"
        checkpoint_path = checkpoint_factory(task)
        if from_env:
            result = task.__class__.create()
        else:
            result = task.__class__.create(checkpoint_path)
        assert isinstance(result, task.__class__)

    @pytest.mark.parametrize("strict", [False, True])
    def test_load_checkpoint(self, mocker, task, strict):
        spy = mocker.spy(task.__class__, "load_state_dict")
        checkpoint_path = checkpoint_factory(task)
        task = task.__class__(checkpoint=checkpoint_path, strict_checkpoint=strict)
        task.setup()
        spy.assert_called()
        call = spy.mock_calls[0]
        assert isinstance(call.args[-1], dict)
        assert call.kwargs["strict"] == strict

    def test_log_val_metrics_on_epoch(self, mocker, task, default_root_dir, datamodule):
        metric = tm.Accuracy(task="multiclass", num_classes=10)
        update = mocker.spy(metric, "update")
        compute = mocker.spy(metric, "compute")
        reset = mocker.spy(metric, "reset")

        def func(state):
            if state.mode == Mode.VAL:
                return tm.MetricCollection({"acc": metric})
            else:
                return tm.MetricCollection({"acc": deepcopy(metric)})

        mocker.patch.object(task, "create_metrics", new=func)
        log_dict_spy = mocker.spy(task, "log_dict")

        trainer = pl.Trainer(
            fast_dev_run=True,
            default_root_dir=default_root_dir,
            precision=32,
            accelerator="cpu",
            devices=1,
        )
        dm = datamodule(batch_size=4)
        trainer.fit(task, datamodule=dm)
        update.assert_called_once()
        reset.assert_called()
        compute.assert_called()

        log_metric_count = sum(1 for call in log_dict_spy.mock_calls for k in call.args[0].keys() if k == "val/acc")
        assert log_metric_count == 1
