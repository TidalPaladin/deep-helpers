#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Set, TypeVar, cast

import pandas as pd
import torch.distributed as dist

from .collection import StateCollection
from .enums import State


T = TypeVar("T")


def all_gather_object(obj: T, group: Optional[Any] = None) -> List[T]:
    if not dist.is_initialized():
        return [obj]
    if group is None:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group)
    gathered_result = [None for _ in range(world_size)]

    # sync and broadcast all
    dist.barrier(group=group)
    dist.all_gather_object(gathered_result, obj, group)

    return cast(List[T], gathered_result)


class DistributedDataFrame(pd.DataFrame):
    def gather_all(self, group: Optional[Any] = None) -> pd.DataFrame:
        r"""Gather this distributed dataframe across processes"""
        gathered = all_gather_object(self, group)
        return pd.concat(gathered)


class DataFrameStateCollection(StateCollection[DistributedDataFrame]):
    _lookup: Dict[State, DistributedDataFrame]

    def __init__(self, proto: Optional[pd.DataFrame] = None):
        super().__init__()
        self._lookup = {}
        self._proto = proto if proto is not None else None

    def update(self, state: State, val: pd.DataFrame) -> None:
        r"""Concatenates the dataframe ``val`` with the current dataframe for state ``state``."""
        old_df = self.get_state(state)
        df = DistributedDataFrame(pd.concat([old_df, val]))
        self.set_state(state, df)

    def set_state(self, state: State, val: DistributedDataFrame) -> None:
        r"""Associates a collection ``val`` with state ``State``."""
        self._lookup[state] = val

    def get_state(self, state: State) -> DistributedDataFrame:
        r"""Returns the collection associated with state ``State``."""
        if state not in self.states:
            raise KeyError(str(state))
        return self._lookup[state]

    def remove_state(self, state: State) -> None:
        r"""Removes state ``state`` if present."""
        if state not in self.states:
            return
        del self._lookup[state]

    @property
    def states(self) -> Set[State]:
        r"""Returns the set of registered ``State`` keys."""
        return set(self._lookup.keys())

    def register(self, state: State, proto: Optional[pd.DataFrame] = None):
        if state in self.states:
            return
        proto = proto if proto is not None else self._proto
        if proto is None:
            raise ValueError("`proto` must be provided if it was not given in constructor")
        ddf_proto = DistributedDataFrame(proto)
        self.set_state(state, ddf_proto)
