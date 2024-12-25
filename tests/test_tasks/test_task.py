#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import yaml
from jsonargparse import ActionConfigFile, ArgumentParser
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR

from deep_helpers.cli import main as cli_main
from deep_helpers.structs import Mode
from deep_helpers.testing import checkpoint_factory
from tests.conftest import DEFAULT_OPTIMIZER_INIT, CustomTask


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
        assert isinstance(result["optimizer"], Adam)
        assert isinstance(result["lr_scheduler"]["scheduler"], StepLR)
        assert result["lr_scheduler"]["monitor"] == lr_scheduler_monitor
        assert result["lr_scheduler"]["interval"] == lr_scheduler_interval

    @pytest.mark.parametrize(
        "parameter_groups,groups",
        [
            (dict(), 1),
            ([{"params": ("bias",), "weight_decay": 0.0}], 2),
            ([{"params": ("Conv2d",), "lr": 1.0}], 2),
            ([{"params": ("conv1",), "lr": 1.0}], 2),
            (
                [
                    {"params": ("Conv2d",), "lr": 1.0},
                    {"params": ("bias",), "lr": 0.1},
                ],
                3,
            ),
        ],
    )
    def test_parameter_groups(self, parameter_groups, groups):
        optimizer_init = {"class_path": "torch.optim.AdamW", "init_args": {"lr": 0.001, "weight_decay": 0.1}}
        lr_scheduler_init = {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 10, "gamma": 0.1},
        }
        lr_scheduler_interval = "epoch"
        lr_scheduler_monitor = "val_loss"

        task = CustomTask(
            optimizer_init,
            lr_scheduler_init,
            lr_scheduler_interval,
            lr_scheduler_monitor,
            parameter_groups=parameter_groups,
        )
        result = task.configure_optimizers()
        assert isinstance(result["optimizer"], AdamW)
        assert len(result["optimizer"].param_groups) == groups
        exp_params = sum(1 for _ in task.parameters())
        actual_params = sum(1 for pg in result["optimizer"].param_groups for p in pg["params"])
        assert exp_params == actual_params

        # The last parameter group should be the default one
        custom_groups = result["optimizer"].param_groups[:-1]
        for i, config in enumerate(parameter_groups):
            for k, v in config.items():
                assert k == "params" or custom_groups[i][k] == v

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
                "init_args": {"optimizer_init": DEFAULT_OPTIMIZER_INIT},
            },
            "data": {
                "class_path": f"tests.conftest.{datamodule().__class__.__name__}",
                "init_args": {
                    "batch_size": 4,
                },
            },
            "compile": True,
            "float32_matmul_precision": "highest",
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

        compile = mocker.patch("deep_helpers.cli.try_compile_model")
        precision = mocker.patch("torch.set_float32_matmul_precision")

        try:
            cli_main()
        except SystemExit as e:
            raise e.__context__ if e.__context__ is not None else e

        compile.assert_called_once()
        precision.assert_called_once_with("highest")

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
    def test_load_torch_checkpoint(self, mocker, task, strict):
        m = mocker.patch("deep_helpers.tasks.task.load_checkpoint")
        checkpoint_path = checkpoint_factory(task)
        task = task.__class__(checkpoint=checkpoint_path, strict_checkpoint=strict)
        task.setup()
        m.assert_called()
        call = m.mock_calls[0]
        assert isinstance(call.args[-1], dict)
        assert call.kwargs["strict"] == strict

    @pytest.mark.parametrize("strict", [False, True])
    def test_load_safetensors_checkpoint(self, mocker, task, strict):
        m = mocker.patch("deep_helpers.tasks.task.load_checkpoint")
        checkpoint_path = checkpoint_factory(task, filename="model.safetensors")
        task = task.__class__(checkpoint=checkpoint_path, strict_checkpoint=strict)
        task.setup()
        m.assert_called()
        call = m.mock_calls[0]
        assert isinstance(call.args[-1], dict)
        assert call.kwargs["strict"] == strict

    @pytest.mark.parametrize("strict", [False, True])
    def test_checkpoint_callback(self, mocker, task, strict):
        checkpoint_path = checkpoint_factory(task, filename="model.safetensors")
        task = task.__class__(checkpoint=checkpoint_path, strict_checkpoint=strict)
        m = mocker.spy(task, "on_task_checkpoint_loaded")
        task.setup()
        m.assert_called()

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

    @pytest.mark.parametrize("subclass", [False, True])
    def test_create_parser(self, subclass):
        parser = ArgumentParser()
        parser = CustomTask.add_args_to_parser(parser, subclass=subclass)
        assert isinstance(parser, ArgumentParser)
        with pytest.raises(SystemExit, match="0"):
            parser.parse_args(["--help"])

    def test_parser_safetensors(self, task):
        # Create the checkpoint
        checkpoint_path = checkpoint_factory(task, filename="model.safetensors")

        # Create the config
        config = {
            "class_path": f"tests.conftest.{task.__class__.__name__}",
            "init_args": {"strict_checkpoint": True},
        }
        with open(checkpoint_path.parent / "config.yaml", "w") as f:
            yaml.dump(config, f)

        parser = ArgumentParser()
        parser.add_argument("--config", action=ActionConfigFile)
        parser = CustomTask.add_args_to_parser(parser)
        cfg = parser.parse_args([str(checkpoint_path)])
        cfg = parser.instantiate_classes(cfg)
        cfg = task.on_after_parse(cfg)

        assert str(cfg.checkpoint) == str(checkpoint_path)
        assert isinstance(cfg.model, nn.Module)

    def test_torchscript(self, task):
        model = task.to_torchscript()
        x = torch.rand(1, 3, 28, 28)
        output = model(x)
        assert isinstance(output, Tensor)

    @pytest.mark.parametrize("log_train_metrics_interval", [1, 4, 8])
    @pytest.mark.parametrize("accumulate_grad_batches", [1, 2, 3])
    def test_log_train_metrics_accumulate_grad_batches(
        self, mocker, task, default_root_dir, datamodule, log_train_metrics_interval, accumulate_grad_batches
    ):
        task.log_train_metrics_interval = log_train_metrics_interval
        # class Foo(MulticlassAccuracy):

        #    def compute(self):
        #        #import pdb; pdb.set_trace()
        #        return super().compute()

        metric = tm.Accuracy(task="multiclass", num_classes=10)
        update = mocker.spy(metric, "update")
        compute = mocker.spy(metric, "compute")
        reset = mocker.spy(metric, "reset")

        def func(state):
            if state.mode == Mode.TRAIN:
                return tm.MetricCollection({"acc": metric})
            else:
                return tm.MetricCollection({"acc": tm.Accuracy(task="multiclass", num_classes=10)})

        mocker.patch.object(task, "create_metrics", new=func)
        mocker.spy(task, "log_dict")

        batch_size = 4
        base_steps = 32
        total_steps = base_steps * accumulate_grad_batches

        trainer = pl.Trainer(
            fast_dev_run=total_steps,
            default_root_dir=default_root_dir,
            precision=32,
            accelerator="cpu",
            devices=1,
            accumulate_grad_batches=accumulate_grad_batches,
            num_sanity_val_steps=0,
        )
        dm = datamodule(length=batch_size * total_steps, batch_size=batch_size)
        trainer.fit(task, datamodule=dm)

        total_steps = base_steps * accumulate_grad_batches
        assert update.call_count == total_steps, "Metric should be updated at each batch"
        assert (
            compute.call_count == base_steps // log_train_metrics_interval
        ), "Metric should be computed at each log_train_metrics_interval"
        assert (
            reset.call_count == compute.call_count + 1
        ), "Metric should be reset at each compute, plus once at epoch start"
