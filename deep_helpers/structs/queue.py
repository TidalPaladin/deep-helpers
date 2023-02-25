#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Dict, Generic, Sequence, Set, TypeVar, Union

import torch.nn as nn

from .collection import StateCollection
from .enums import Mode, State


T = TypeVar("T", bound=StateCollection)
P = TypeVar("P")
M = TypeVar("M", bound="nn.Module")
U = TypeVar("U")


@dataclass(order=True)
class PrioritizedItem(Generic[P]):
    priority: Union[int, float]
    value: P = field(compare=False)


class QueueStateCollection(StateCollection[PriorityQueue[PrioritizedItem[P]]]):
    r"""Collection that associates each State with a PriorityQueue."""
    QueueType = PriorityQueue[PrioritizedItem]
    _lookup: Dict[State, QueueType]

    def __init__(self):
        super().__init__()
        self._lookup = {}

    def register(self, state: State, maxsize: int = 0):
        if state in self.states:
            return
        queue = PriorityQueue(maxsize=maxsize)
        self.set_state(state, queue)

    def set_state(self, state: State, val: QueueType) -> None:
        self._lookup[state] = val

    def get_state(self, state: State) -> QueueType:
        r"""Returns the collection associated with state ``State``."""
        if state not in self.states:
            raise KeyError(state)
        return self._lookup[state]

    def remove_state(self, state: State) -> None:
        r"""Removes state ``state`` if present."""
        if state in self.states:
            del self._lookup[state]

    @property
    def states(self) -> Set[State]:
        return set(self._lookup.keys())

    def enqueue(self, state: State, priority: Union[int, float], value: P, *args, **kwargs) -> None:
        item = PrioritizedItem(priority, value)
        queue = self.get_state(state)
        queue.put(item, *args, **kwargs)

    def dequeue(self, state: State, *args, **kwargs) -> PrioritizedItem[P]:
        queue = self.get_state(state)
        return queue.get(*args, **kwargs)

    def empty(self, state: State) -> bool:
        queue = self.get_state(state)
        return queue.empty()

    def qsize(self, state: State) -> int:
        queue = self.get_state(state)
        return queue.qsize()

    def __len__(self) -> int:
        r"""Gets the total number of currently queued items across all states"""
        return sum(self.qsize(state) for state in self.states)

    def reset(
        self: T,
        specific_states: Sequence[State] = [],
        specific_modes: Sequence[Mode] = [],
    ) -> T:
        for k, v in self.as_dict().items():
            if specific_states and k not in specific_states:
                continue
            elif specific_modes and k.mode not in specific_modes:
                continue
            while not v.empty():
                v.get_nowait()
            assert v.empty()
        return self

    def __add__(self: T, other: T) -> T:
        # NOTE: this will modify self inplace - PriorityQueue can't be deepcopied
        output = self
        for state, val in other.as_dict().items():
            output.set_state(state, val)
        return output
