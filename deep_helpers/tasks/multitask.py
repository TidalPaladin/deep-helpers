#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta
from functools import wraps
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple, TypeVar, cast

import torch.nn as nn
import torchmetrics as tm
from pytorch_lightning.core.hooks import CheckpointHooks, ModelHooks

from ..structs import State
from .task import TASKS, Task


T = TypeVar("T")

HOOKS: Final = [
    ModelHooks,
    CheckpointHooks,
]


def update(dest: Dict, src: Dict) -> None:
    for k, v in src.items():
        if k not in dest or not isinstance(v, dict):
            dest[k] = v
        else:
            dest[k].update(v)


class ForwardHooks(ABCMeta):
    r"""Metaclass for wrapping hooks in MultiTask. This is necessary to ensure that
    the hooks are called for each task in the MultiTask. This is done by wrapping
    the hook in a recursive_task_wrapper, which will call the hook for each task
    in the MultiTask. Hook mixins are defined in `HOOKS`, and hooks to wrap are determined
    by the presence of a method with the same name in the hook class.
    """

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        for hook in HOOKS:
            for method_name in dir(hook):
                method = getattr(hook, method_name)
                # Skip private methods and methods with overriden implementations
                if callable(method) and method_name not in attrs and not method_name.startswith("__"):
                    setattr(new_class, method_name, cls.recursive_task_wrapper(funcname=method_name))

        return new_class

    @staticmethod
    def recursive_task_wrapper(funcname) -> Any:
        def recurse_on_tasks(self, *args, **kwargs):
            assert isinstance(self, MultiTask)
            result: Dict[str, Any] = {}
            for name, task in self:
                assert not isinstance(task, MultiTask)
                func = getattr(task, funcname)
                result[name] = func(*args, **kwargs)
            return result

        return recurse_on_tasks


class MultiTask(Task, metaclass=ForwardHooks):
    r"""A multi-task wrapper around multiple contained tasks.

    Args:
        tasks: A list of task names to instantiate. Tasks are registered in the ``TASKS`` registry.
        checkpoint: A checkpoint to load. If None, no checkpoint is loaded.
        strict_checkpoint: Whether to enforce strict checkpoint loading.
        cycle: Determines how tasks are handled during training. If True, tasks are cycled through
            such that each step is a single task. If False, each step will execute all tasks.

    Keyword Args:
        Forwarded to the contained tasks.
    """

    def __init__(
        self,
        tasks: List[str],
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        cycle: bool = True,
        **kwargs,
    ):
        super().__init__(checkpoint=checkpoint, strict_checkpoint=strict_checkpoint, **kwargs)
        self._tasks = nn.ModuleDict({t: cast(Task, TASKS.get(t).instantiate_with_metadata(**kwargs).fn) for t in tasks})
        self.cycle = cycle

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, name: str) -> Task:
        return cast(Task, self._tasks[name])

    def __iter__(self) -> Iterator[Tuple[str, Task]]:
        for name, task in self._tasks.items():
            yield name, cast(Task, task)

    def setup(self, stage: str):
        for _, task in self:
            # Update the trainer reference in each task
            task.trainer = self.trainer

            # Patch run_logging_loop to bea no-op. We'll call it at MultiTask level.
            # This is necessary because subtasks won't be able to call log() directly.
            func = task.run_logging_loop

            @wraps(func)
            def wrapped(
                state: State,
                source: Task,
                *args,
                **kwargs,
            ) -> None:
                return func(state, self, *args, **kwargs)

            task.run_logging_loop = wrapped

            # Run setup on each task
            task.setup(stage)

    def teardown(self, *args, **kwargs):
        super().teardown(*args, **kwargs)
        # Unpatch run_logging_loop
        for _, task in self:
            task.run_logging_loop = task.__class__.run_logging_loop

    def share_attribute(self, attr_name: str) -> None:
        proto = self.find_attribute(attr_name)
        setattr(self, attr_name, proto)
        self.update_attribute(attr_name, proto)

    def update_attribute(self, attr_name: str, val: nn.Module) -> None:
        for task in self:
            if hasattr(task, attr_name):
                setattr(task, attr_name, val)

    def find_attribute(self, attr_name: str) -> nn.Module:
        for task in self:
            if hasattr(task, attr_name) and isinstance((val := getattr(task, attr_name)), nn.Module):
                return val
        raise AttributeError(attr_name)

    def find_all_attributes(self, attr_name: str) -> List[nn.Module]:
        result: List[nn.Module] = []
        for _, task in self:
            if hasattr(task, attr_name) and isinstance((val := getattr(task, attr_name)), nn.Module):
                result.append(val)
        return result

    def _get_current_task_idx(self, batch_idx: int) -> int:
        return batch_idx % len(self)

    def get_current_task(self, batch_idx: int) -> Task:
        task_idx = self._get_current_task_idx(batch_idx)
        task = list(self._tasks.values())[task_idx]
        assert isinstance(task, Task)
        return task

    def get_current_task_name(self, batch_idx: int) -> str:
        task_idx = self._get_current_task_idx(batch_idx)
        return list(self._tasks.keys())[task_idx]

    def step(self, batch: Any, batch_idx: int, metrics: Optional[tm.MetricCollection] = None) -> Dict[str, Any]:
        raise NotImplementedError(  # pragma: no cover
            "MultiTask does not support step. Use training_step, validation_step, or test_step."
        )

    def create_metrics(self, *args, **kwargs):
        raise NotImplementedError(  # pragma: no cover
            "MultiTask does not support `create_metrics`. Call `create_metrics` on each task individually."
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return self._training_step_cycle(batch, batch_idx) if self.cycle else self._training_step_all(batch, batch_idx)

    def _training_step_cycle(self, batch: Any, batch_idx: int) -> Any:
        task = self.get_current_task(batch_idx)
        output = task.training_step(batch, batch_idx)
        return output

    def _training_step_all(self, batch: Any, batch_idx: int) -> Any:
        output = {}
        for _, task in self:
            task_output = task.training_step(batch, batch_idx)
            update(output, task_output)
        output["loss"] = self.compute_total_loss(output)
        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        output = {}
        for _, task in self:
            task_output = task.validation_step(batch, batch_idx)
            update(output, task_output)
        output["loss"] = self.compute_total_loss(output)
        return output

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        output = {}
        for _, task in self:
            task_output = task.test_step(batch, batch_idx)
            update(output, task_output)
        output["loss"] = self.compute_total_loss(output)
        return output

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        output = {}
        for name, task in self:
            task_output = task.predict_step(batch, batch_idx)
            output[name] = task_output
        return output
