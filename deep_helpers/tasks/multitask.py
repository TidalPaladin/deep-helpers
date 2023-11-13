#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple, TypeVar, Union, cast

import torch.nn as nn
import torchmetrics as tm
from pytorch_lightning.core.hooks import CheckpointHooks, ModelHooks

from ..structs import Mode
from .task import TASKS, StateMixin, Task


T = TypeVar("T")

HOOKS: Final = [
    ModelHooks,
    CheckpointHooks,
]
NOT_FORWARDABLE: Final = ("configure_sharded_model",)  # Deprecated after PL 2.1, will raise exception if forwarded


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
                if (
                    callable(method)
                    and method_name not in attrs
                    and not method_name.startswith("__")
                    and method_name not in NOT_FORWARDABLE
                ):
                    setattr(new_class, method_name, cls.recursive_task_wrapper(funcname=method_name))

        return new_class

    @staticmethod
    def recursive_task_wrapper(funcname) -> Any:
        def recurse_on_tasks(self, *args, **kwargs):
            assert isinstance(self, MultiTask)

            # Ensure we call StateMixin hooks on the MultiTask.
            # We assume these hooks aren't returning anything.
            if hasattr(StateMixin, funcname):
                func = getattr(StateMixin, funcname)
                func(self, *args, **kwargs)

            # Then recurse on each task
            result: Dict[str, Any] = {}
            for name, task in self:
                assert not isinstance(task, MultiTask)
                func = getattr(task, funcname)
                result[name] = func(*args, **kwargs)
            return result

        return recurse_on_tasks


def _get_task(key: Union[str, Tuple[str, Task]], **kwargs) -> Tuple[str, Task]:
    if isinstance(key, str):
        return key, cast(Task, TASKS.get(key).instantiate_with_metadata(**kwargs).fn)
    else:
        name, task = key
        if not isinstance(task, Task):
            raise TypeError(f"Expected Task, got {type(task)}")
        return name, task


class MultiTask(Task, metaclass=ForwardHooks):
    r"""A multi-task wrapper around multiple contained tasks.

    Args:
        tasks: A list of tasks or task names to instantiate. Tasks are registered in the ``TASKS`` registry.
        checkpoint: A checkpoint to load. If None, no checkpoint is loaded.
        strict_checkpoint: Whether to enforce strict checkpoint loading.
        cycle: Determines how tasks are handled during training. If True, tasks are cycled through
            such that each step is a single task. If False, each step will execute all tasks.

    Keyword Args:
        Forwarded to the contained tasks.
    """

    def __init__(
        self,
        tasks: List[Union[str, Tuple[str, Task]]],
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        cycle: bool = True,
        **kwargs,
    ):
        self._tasks = {}
        super().__init__(**kwargs)
        self._tasks = nn.ModuleDict({k: v for k, v in (_get_task(task, **kwargs) for task in tasks)})
        self.cycle = cycle
        self.checkpoint = checkpoint
        self.strict_checkpoint = strict_checkpoint

    @property
    def checkpoint(self) -> Optional[Path]:
        return Path(self._checkpoint) if self._checkpoint is not None else None

    @checkpoint.setter
    def checkpoint(self, value: Optional[Union[str, Path]]) -> None:
        self._checkpoint = Path(value) if isinstance(value, str) else value
        for task in self._tasks.values():
            task.checkpoint = self._checkpoint

    @property
    def strict_checkpoint(self) -> bool:
        return self._strict_checkpoint

    @strict_checkpoint.setter
    def strict_checkpoint(self, value: bool) -> None:
        self._strict_checkpoint = value
        for task in self._tasks.values():
            task.strict_checkpoint = self._strict_checkpoint

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, val: Union[int, str]) -> Task:
        if isinstance(val, str):
            task = self._tasks[val]
        else:
            task = list(self._tasks.values())[val]
        return cast(Task, task)

    def __iter__(self) -> Iterator[Tuple[str, Task]]:
        for name, task in self._tasks.items():
            yield name, cast(Task, task)

    def setup(self, stage: str):
        for _, task in self:
            # Update the trainer reference in each task
            task.trainer = self.trainer

            # Run setup on each task.
            # Enure we don't load the checkpoint for each task.
            task.checkpoint = None
            task.setup(stage)
            task.checkpoint = self.checkpoint

        if self.checkpoint is not None:
            self._safe_load_checkpoint()

    def share_attribute(self, attr_name: str) -> None:
        r"""Share an attribute across all of the contained tasks."""
        proto = self.find_attribute(attr_name)
        setattr(self, attr_name, proto)
        self.update_attribute(attr_name, proto)

    def update_attribute(self, attr_name: str, val: Union[nn.Module, nn.Parameter]) -> None:
        r"""Update an attribute in all of the contained tasks."""
        for _, task in self:
            if hasattr(task, attr_name):
                setattr(task, attr_name, val)

    def find_attribute(self, attr_name: str) -> Union[nn.Module, nn.Parameter]:
        r"""Find an attribute in any of the contained tasks. Returns the first matching attribute."""
        for _, task in self:
            if hasattr(task, attr_name) and isinstance((val := getattr(task, attr_name)), (nn.Module, nn.Parameter)):
                return val
        raise AttributeError(attr_name)

    def find_all_attributes(self, attr_name: str) -> List[Union[nn.Module, nn.Parameter]]:
        r"""Find an attribute in any of the contained tasks. Returns all matching attributes."""
        result: List[Union[nn.Module, nn.Parameter]] = []
        for _, task in self:
            if hasattr(task, attr_name) and isinstance((val := getattr(task, attr_name)), (nn.Module, nn.Parameter)):
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
        assert self.state.mode == Mode.TRAIN
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
        assert self.state.mode == Mode.VAL
        output = {}
        for _, task in self:
            task_output = task.validation_step(batch, batch_idx)
            update(output, task_output)
        output["loss"] = self.compute_total_loss(output)
        return output

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        assert self.state.mode == Mode.TEST
        output = {}
        for _, task in self:
            task_output = task.test_step(batch, batch_idx)
            update(output, task_output)
        output["loss"] = self.compute_total_loss(output)
        return output

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        assert self.state.mode == Mode.PREDICT
        output = {}
        for name, task in self:
            task_output = task.predict_step(batch, batch_idx)
            output[name] = task_output
        return output

    def on_validation_epoch_end(self, *args, **kwargs):
        for _, task in self:
            task.on_validation_epoch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        for _, task in self:
            task.on_test_epoch_end(*args, **kwargs)
