#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Final, Set, TypeVar, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from ..structs import Mode
from ..tasks import MultiTask


ALL_MODES: Final = [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT]
T = TypeVar("T")
L = TypeVar("L", bound=Logger)
TaskIdentifier = Union[str, int]


def check_is_multitask(task: pl.LightningModule) -> None:
    if not isinstance(task, MultiTask):
        raise TypeError(f"Expected MultiTask, got {type(task)}.")


@dataclass
class MultiTaskCallbackWrapper(Callback):
    r"""Wrapper that calls the wrapped callback's batch start method only if the current
    task is in the list of tasks to run on. Use this to attach callbacks to specific tasks.
    This wrapper only wraps batch start/end hooks, as these are the only hooks that
    require special handling due to task cycling.

    Args:
        wrapped: The callback to wrap
        tasks: Identifiers (task indices, registered names, etc.) of tasks to run on.
    """

    wrapped: Callback
    tasks: Set[TaskIdentifier] = field(default_factory=set)

    def __post_init__(self):
        # Iterate through Callback methods that are not defined in this class and
        # attach them to this class. The attached methods will call the wrapped
        # callback's methods.
        for method_name in dir(Callback):
            # Skip methods that aren't defined in Callback or aren't callable. We only want to
            # forward callable hooks that exist in Callback.
            if method_name.startswith("_") or not callable(getattr(Callback, method_name)):
                continue

            method = getattr(self.wrapped, method_name)

            # There are special cases for batch start and end. We must ensure that
            # stepwise callbacks are only called when the current step's task matches the task
            # for this wrapper.
            if callable(method) and method_name == "on_train_batch_start":
                setattr(self, method_name, self.batch_start_wrapper(method))
            elif callable(method) and method_name == "on_train_batch_end":
                print(method_name)
                setattr(self, method_name, self.batch_end_wrapper(method))
            else:
                setattr(self, method_name, method)

    def identify_task(self, pl_module: MultiTask, batch_idx: int) -> str:
        return pl_module.get_current_task_name(batch_idx)

    def should_run_on_task(self, pl_module: MultiTask, batch_idx: int) -> bool:
        return self.tasks is None or self.identify_task(pl_module, batch_idx) in self.tasks

    def batch_start_wrapper(self, func) -> Any:
        def on_batch_start(
            trainer: pl.Trainer,
            pl_module: MultiTask,
            batch: Any,
            batch_idx: int,
            *args,
            **kwargs,
        ) -> None:
            check_is_multitask(pl_module)
            if self.should_run_on_task(pl_module, batch_idx):
                task = pl_module.get_current_task(batch_idx)
                func(task, trainer, pl_module, batch, batch_idx, *args, **kwargs)

        return on_batch_start

    def batch_end_wrapper(self, func) -> Any:
        def on_batch_end(
            trainer: pl.Trainer,
            pl_module: MultiTask,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            *args,
            **kwargs,
        ) -> None:
            check_is_multitask(pl_module)
            if self.should_run_on_task(pl_module, batch_idx):
                task = pl_module.get_current_task(batch_idx)
                func(task, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

        return on_batch_end
