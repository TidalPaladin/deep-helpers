#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

import pytest
import torchmetrics as tm
from deep_helpers.callbacks import LoggerIntegration, MetricLoggingCallback
from deep_helpers.structs import MetricStateCollection
from pytorch_lightning.loggers.wandb import WandbLogger

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


def metric_factory():
    acc = tm.Accuracy(task="binary")
    return MetricStateCollection(tm.MetricCollection({"acc": acc}))


@dataclass
class CustomMetricCallback(MetricLoggingCallback):
    integrations = [DummyIntegration()]

    state_metrics: MetricStateCollection = field(default_factory=metric_factory(), repr=False)
    log_on_step: bool = False

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
        return {
            "preds": outputs["logits"].sigmoid().amax(dim=-1),  # type: ignore
            "target": batch["label"].flatten().clip(min=0, max=1),  # type: ignore
        }


class TestMetricLoggingCallback(BaseCallbackTest):
    @pytest.fixture
    def callback(self, modes):
        cb = CustomMetricCallback("name", modes=modes, state_metrics=metric_factory())
        return cb
