#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import pytorch_lightning as pl
import torch

from ..structs import MetricStateCollection, Mode, State
from ..tasks import I, O, Task
from .base import LoggingCallback


@dataclass
class MetricLoggingCallback(LoggingCallback[I, O, Dict[str, Any]], ABC):
    state_metrics: MetricStateCollection = field(default_factory=MetricStateCollection, repr=False)
    log_on_step: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.state_metrics:
            raise ValueError("Must provide at least one metric to log")

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        self.state_metrics.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def register(self, state: State, pl_module: Task, *args, **kwargs) -> None:
        if state not in self.state_metrics.states:
            self.state_metrics.register(state, device=torch.device(pl_module.device))

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = pl_module.state
        target = self.prepare_target(trainer, pl_module, outputs, batch, batch_idx)
        self.state_metrics.update(state, **target)

        if self.log_on_step:
            collection = self.state_metrics.get_state(state)
            tag = state.with_postfix(self.name)
            self.wrapped_log(
                collection.compute(),
                pl_module,
                tag,
                trainer.global_step,
            )
            collection.reset()

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        mode: Mode,
    ):
        trainer.global_step
        for state, metric in self.state_metrics.as_dict().items():
            tag = state.with_postfix(self.name)
            if state.mode == mode:
                self.wrapped_log(
                    metric.compute(),
                    pl_module,
                    tag,
                    trainer.global_step,
                )
                metric.reset()
