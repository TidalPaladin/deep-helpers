#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy, deepcopy
from functools import partial

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.data import DatasetNames, SupportsDatasetNames
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import TASKS, Task
from deep_helpers.testing import handle_cuda_mark
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from torchmetrics.classification import MulticlassAccuracy

from deep_helpers.data import DatasetNames, SupportsDatasetNames
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from deep_helpers.testing import handle_cuda_mark


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


@pytest.fixture(
    params=[
        pytest.param(False, id="no_cuda"),
        pytest.param(True, marks=[pytest.mark.cuda, pytest.mark.ci_skip], id="cuda"),
    ]
)
def cuda(request):
    return request.param


@pytest.fixture
def default_root_dir(tmp_path):
    return tmp_path / "default_root_dir"


@pytest.fixture
def logger(mocker):
    logger = mocker.MagicMock(name="logger", spec_set=WandbLogger)
    return logger


@pytest.fixture
def lightning_module(task):
    return task


@pytest.fixture
def pl_module(lightning_module):
    return lightning_module


@pytest.fixture
def training_example():
    example = {
        "img": torch.rand(3, 32, 32, requires_grad=True),
        "label": torch.randint(0, 10, (1,)),
    }
    return example


@pytest.fixture(params=[1, 5])
def batch(request, example):
    batch_size = request.param
    return default_collate([deepcopy(example) for _ in range(batch_size)])


DEFAULT_OPTIMIZER_INIT = {
    "class_path": "torch.optim.Adam",
    "init_args": {
        "lr": 1e-3,
    },
}


@TASKS(name="custom-task")
class CustomTask(Task):
    def __init__(self, *args, **kwargs):
        if not args and not kwargs.get("optimizer_init", {}):
            kwargs.setdefault("optimizer_init", {}).update(DEFAULT_OPTIMIZER_INIT)
        super().__init__(*args, **kwargs)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def create_metrics(self, state: State) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=10),
            }
        )

    def step(self, batch, batch_idx, state, metrics):
        x = batch["img"]
        y = batch["label"].long().flatten()
        output = {"log": {}}

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        output["log"]["loss_ce"] = loss

        if metrics is not None:
            metrics["acc"](y_hat, y)

        return output


class DummyDataset(Dataset):
    def __init__(self, length, example=None):
        super().__init__()
        self.example = (
            example
            if example is not None
            else {
                "img": torch.rand(3, 32, 32, requires_grad=True),
                "label": torch.randint(0, 10, (1,)),
            }
        )
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return copy(self.example)


class DummyDM(pl.LightningDataModule, SupportsDatasetNames):
    def __init__(self, batch_size: int = 1, example=None):
        super().__init__()
        self.ds = DummyDataset(length=10, example=example)
        self.batch_size = batch_size

    @property
    def _dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, collate_fn=default_collate)

    def train_dataloader(self):
        return self._dataloader

    def val_dataloader(self):
        return self._dataloader

    def test_dataloader(self):
        return self._dataloader

    @property
    def dataset_names(self) -> DatasetNames:
        names = DatasetNames()
        names[(Mode.TRAIN, 0)] = "train_1"
        names[(Mode.TRAIN, 1)] = "train_2"
        names[(Mode.VAL, 0)] = "val_1"
        names[(Mode.VAL, 1)] = "val_2"
        names[(Mode.TEST, 0)] = "test"
        return names


@pytest.fixture
def task(logger):
    task = CustomTask()
    trainer = pl.Trainer(logger=logger)
    task.trainer = trainer
    # NOTE: yield here to weakref to logger being gc'ed
    yield task


@pytest.fixture
def datamodule(training_example):
    return partial(DummyDM, example=training_example)
