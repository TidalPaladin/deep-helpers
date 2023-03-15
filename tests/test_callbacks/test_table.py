#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import pandas as pd
import pytest
from pytorch_lightning.loggers.wandb import WandbLogger

from deep_helpers.callbacks import LoggerIntegration, TableCallback
from tests.test_callbacks.base_callback import BaseCallbackTest


class DummyIntegration(LoggerIntegration):
    logger_type = WandbLogger

    def __call__(
        self,
        target,
        pl_module,
        tag,
        step,
        *args,
        **kwargs,
    ):
        t = {tag: target}
        pl_module.logger.experiment.log(t, commit=False)


data = {
    "sum": [0],
    "p": [0],
}
proto = pd.DataFrame(data)


@dataclass
class CustomTableCallback(TableCallback):
    integrations = [DummyIntegration()]
    proto = proto

    def prepare_target(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        data = {
            "sum": [batch["img"].sum().item()],  # type: ignore
            "p": [outputs["logits"].sigmoid().sum().item()],  # type: ignore
        }
        return pd.DataFrame(data)


# TODO finish building these tests
class TestTableCallback(BaseCallbackTest):
    @pytest.fixture
    def callback(self, modes):
        cb = CustomTableCallback("table", modes=modes)
        return cb
