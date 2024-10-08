#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from deep_helpers.callbacks import LoggingCallback, MultiTaskCallbackWrapper
from deep_helpers.cli import main as cli_main
from deep_helpers.tasks import TASKS, MultiTask, Task


class DummyCallback(LoggingCallback):
    def reset(self, *args, **kwargs):
        pass

    def register(self, *args, **kwargs):
        pass

    def prepare_target(self, *args, **kwargs):
        return {}


@pytest.fixture
def callback(mocker):
    callback = DummyCallback("dummy")
    mocker.patch.object(callback, "_on_batch_end")
    mocker.patch.object(callback, "_on_epoch_end")
    return MultiTaskCallbackWrapper(callback, {0})


@pytest.fixture
def multitask(logger):
    optimizer_init = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.001}}
    task = MultiTask(tasks=["custom-task", "custom-task2"], optimizer_init=optimizer_init)
    trainer = pl.Trainer(logger=logger)
    task.trainer = trainer
    task.setup("fit")
    # NOTE: yield here to weakref to logger being gc'ed
    yield task


class TestMultiTask:
    def test_len(self, multitask):
        assert len(multitask) == 2

    def test_task_class_input(self):
        task = TASKS.get("custom-task").instantiate_with_metadata().fn
        multitask = MultiTask(tasks=["custom-task", ("custom-task2", task)])
        assert isinstance(multitask, MultiTask)
        assert len(multitask) == 2

    @pytest.mark.parametrize(
        "key",
        [
            0,
            1,
            "custom-task",
            "custom-task2",
            pytest.param("foo", marks=pytest.mark.xfail(raises=KeyError, strict=True)),
            pytest.param(3, marks=pytest.mark.xfail(raises=IndexError, strict=True)),
        ],
    )
    def test_getitem(self, key, multitask):
        assert isinstance(multitask[key], Task)

    def test_iter(self, multitask):
        iterator = iter(multitask)
        name, val = next(iterator)
        assert name == "custom-task"
        assert isinstance(val, Task)
        name, val = next(iterator)
        assert name == "custom-task2"
        assert isinstance(val, Task)

    @pytest.mark.parametrize(
        "batch_idx,exp",
        [
            (0, "custom-task"),
            (1, "custom-task2"),
            (2, "custom-task"),
            (3, "custom-task2"),
        ],
    )
    def test_get_current_task_name(self, multitask, batch_idx, exp):
        assert multitask.cycle
        assert multitask.get_current_task_name(batch_idx) == exp

    def test_configure_optimizers(self):
        optimizer_init = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.001}}
        lr_scheduler_init = {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 10, "gamma": 0.1},
        }
        lr_interval = "epoch"
        lr_monitor = "val_loss"

        task = MultiTask(
            tasks=["custom-task"],
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            lr_interval=lr_interval,
            lr_monitor=lr_monitor,
        )
        result = task.configure_optimizers()
        assert isinstance(result["optimizer"], Adam)
        assert isinstance(result["lr_scheduler"]["scheduler"], StepLR)
        assert result["lr_scheduler"]["monitor"] == lr_monitor
        assert result["lr_scheduler"]["interval"] == lr_interval

    @pytest.mark.parametrize(
        "hook",
        [
            "on_fit_start",
            "on_train_epoch_end",
        ],
    )
    def test_forward_hook(self, mocker, multitask, hook):
        mocks = []
        for _, task in multitask:
            mock = mocker.patch.object(task, hook)
            mocks.append(mock)
        func = getattr(multitask, hook)
        func()
        for mock in mocks:
            mock.assert_called_once()

    def test_call_hook_override(self, multitask):
        # We just want to make sure that the hook is called without error.
        # This hook is manually subclassed in MultiTask
        multitask.setup("stage")

    def test_checkpoint_getter_setter(self, multitask):
        # Test checkpoint setter
        checkpoint_path = "test_checkpoint_path"
        multitask.checkpoint = checkpoint_path
        assert multitask.checkpoint == Path(checkpoint_path)
        for _, task in multitask:
            assert task.checkpoint == Path(checkpoint_path)

    def test_strict_checkpoint_getter_setter(self, multitask):
        multitask.strict_checkpoint = True
        assert multitask.strict_checkpoint
        for _, task in multitask:
            assert task.strict_checkpoint

    def test_setup_loads_checkpoint(self, mocker, multitask, tmp_path):
        checkpoint = tmp_path / "checkpoint_path"
        cp = {"state_dict": multitask.state_dict()}
        torch.save(cp, checkpoint)
        multitask.checkpoint = checkpoint
        spy = mocker.spy(multitask, "load_state_dict")
        spy2 = mocker.spy(torch, "load")
        multitask.setup("stage")
        spy.assert_called_once()
        spy2.assert_called_once_with(checkpoint, map_location="cpu", weights_only=False)

    def test_find_attribute(self, multitask):
        linear_layer = torch.nn.Linear(10, 10)
        multitask._tasks["custom-task"].linear_layer = linear_layer
        found_attr = multitask.find_attribute("linear_layer")
        assert found_attr == linear_layer

    @pytest.mark.parametrize("attr, attr_type", [("linear_layer", nn.Module), ("weight", nn.Parameter)])
    def test_share_attribute(self, multitask, attr, attr_type):
        if attr_type == nn.Module:
            attr_instance = torch.nn.Linear(10, 10)
            copy = torch.nn.Linear(10, 10)
        else:  # attr_type == nn.Parameter
            attr_instance = nn.Parameter(torch.randn(10, 10))
            copy = nn.Parameter(torch.randn(10, 10))
        multitask._tasks["custom-task"].__setattr__(attr, attr_instance)
        multitask._tasks["custom-task2"].__setattr__(attr, copy)
        multitask.share_attribute(attr)
        for _, task in multitask:
            assert hasattr(task, attr)
            assert task.__getattr__(attr) is attr_instance

    @pytest.mark.parametrize("cycle", [False, True])
    @pytest.mark.parametrize("named_datasets", [False, True])
    @pytest.mark.parametrize("stage", ["fit", "test", "predict"])
    @pytest.mark.parametrize(
        "accelerator",
        [
            "cpu",
            pytest.param("gpu", marks=pytest.mark.cuda),
        ],
    )
    def test_run(self, multitask, default_root_dir, accelerator, datamodule, stage, named_datasets, cycle):
        if accelerator == "gpu":
            precision = "bf16-mixed" if torch.cuda.get_device_capability(0)[0] >= 8 else "16"
        else:
            precision = 32

        multitask.named_datasets = named_datasets
        multitask.cycle = cycle
        trainer = pl.Trainer(
            fast_dev_run=True,
            default_root_dir=default_root_dir,
            precision=precision,
            accelerator=accelerator,
            devices=1,
        )

        dm = datamodule(batch_size=4)
        func = getattr(trainer, stage)
        func(multitask, datamodule=dm)

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
                "class_path": f"deep_helpers.tasks.MultiTask",
                "init_args": {
                    "tasks": ["custom-task"],
                    "optimizer_init": {
                        "class_path": "torch.optim.Adam",
                        "init_args": {"lr": 0.001},
                    },
                },
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

    @pytest.mark.xfail(reason="MultiTask does not support torchscript")
    def test_torchscript(self, multitask):
        model = multitask.to_torchscript()
        x = torch.rand(1, 3, 28, 28)
        model(x)
