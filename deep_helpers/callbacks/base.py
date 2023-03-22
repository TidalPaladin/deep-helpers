#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Generic, Iterable, List, Optional, Type, TypeVar

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from ..structs import Mode, State
from ..tasks import I, O, Task


ALL_MODES: Final = [Mode.TRAIN, Mode.VAL, Mode.TEST]
T = TypeVar("T")
L = TypeVar("L", bound=Logger)


class LoggerIntegration(ABC, Generic[L, T]):
    r"""Abstraction for integrating with a logger. Implementations of this class
    should be able to take a logging target prepared by the :class:`LoggingCallback`
    and log it to the appropriate logger.
    """
    logger_type: ClassVar[Type[Logger]]

    def integrates_with(self, logger: Logger) -> bool:
        r"""Check if this integration can be used with the given logger."""
        return isinstance(logger, self.logger_type)

    @abstractmethod
    def __call__(
        self,
        target: T,
        pl_module: Task,
        tag: str,
        step: int,
    ) -> None:
        raise NotImplementedError


@dataclass
class LoggingCallback(Callback, ABC, Generic[I, O, T]):
    name: str
    modes: List[Mode] = field(default_factory=lambda: ALL_MODES)
    tasks: Optional[List[Type[Task]]] = None

    integrations: ClassVar[List[LoggerIntegration]]

    def __post_init__(self):
        # TODO: Is there a way to do abstract class variables?
        if not hasattr(self, "integrations"):
            raise AttributeError("Integrations must be defined for LoggingCallback")
        if not self.modes:
            raise ValueError("Must specify at least one mode to log for")
        elif not isinstance(self.modes, Iterable):
            raise TypeError("Modes must be an iterable")
        elif not all(mode in ALL_MODES for mode in self.modes):
            raise ValueError("Invalid mode specified")

    @abstractmethod
    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        r"""Reset the state of this logging callback"""
        raise NotImplementedError

    @abstractmethod
    def register(
        self,
        state: State,
        pl_module: Task,
        example: I,
        prediction: O,
    ) -> None:
        r"""Performs any setup/registration needed for a given state. This method will only be
        called if ``state.mode in self.modes``. It may be called multiple times for a given state.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_target(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> T:
        raise NotImplementedError

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
        r"""Handles callback logic when batch ends."""
        target = self.prepare_target(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)
        self.wrapped_log(target, pl_module, trainer.global_step)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        mode: Mode,
    ):
        r"""Handles callback logic when epoch ends."""
        return

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if not state.sanity_checking and state.mode not in self.modes:
            return
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if not state.sanity_checking and state.mode not in self.modes:
            return
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if not state.sanity_checking and state.mode not in self.modes:
            return
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    def on_train_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.TRAIN])

    def on_validation_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.VAL])

    def on_test_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.TEST])

    def on_train_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TRAIN)

    def on_validation_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.VAL)

    def on_validation_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.VAL)

    def on_test_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TEST)

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: Task):
        pl_module.state = pl_module.state.set_sanity_checking(True)

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: Task):
        pl_module.state = pl_module.state.set_sanity_checking(False)
        self.reset()

    @rank_zero_only
    def wrapped_log(
        self,
        target: Any,
        pl_module: Task,
        tag: str,
        step: int,
    ):
        r"""Wrapper that calls self.log only on rank zero and when not sanity checking"""
        assert hasattr(pl_module, "state") and isinstance(pl_module.state, State)
        assert isinstance(tag, str) and tag
        assert isinstance(step, int) and step >= 0
        if not pl_module.state.sanity_checking:
            for integration in self.integrations:
                if integration.integrates_with(pl_module.logger):
                    integration(target, pl_module, self.name, step)
