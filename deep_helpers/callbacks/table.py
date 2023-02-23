#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd
import pytorch_lightning as pl

from ..structs import DataFrameStateCollection, DistributedDataFrame, Mode, State
from ..tasks import I, O, Task
from .base import LoggingCallback


@dataclass
class TableCallback(LoggingCallback[I, O, pd.DataFrame]):
    proto: Optional[pd.DataFrame] = field(default=None, repr=False)
    _tables: DataFrameStateCollection = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._tables = DataFrameStateCollection(self.proto)

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        self._tables.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def register(self, state: State, pl_module: Task, example: I, prediction: O) -> None:
        if state in self._tables.states:
            return
        if self._tables._proto is None:
            proto = self.prepare_target(pl_module.trainer, pl_module, prediction, example, 0)
        else:
            proto = None
        self._tables.register(state, proto)

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
        if state.mode not in self.modes:
            return

        new_table = DistributedDataFrame(self.prepare_target(trainer, pl_module, outputs, batch, batch_idx))

        if state in self._tables.states:
            old_table = self._tables.get_state(state)
            table = DistributedDataFrame(pd.concat([old_table, new_table]))
        else:
            self._tables.register(state)
            table = new_table

        assert isinstance(table, DistributedDataFrame)
        self._tables.set_state(state, table)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        mode: Mode,
    ):
        for state, table in self._tables.as_dict().items():
            if state.mode != mode:
                continue
            tag = state.with_postfix(self.name)
            table = table.gather_all()
            self.wrapped_log(table, pl_module, tag, trainer.global_step)
            self._tables.remove_state(state)
