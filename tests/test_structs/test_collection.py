#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Dict, Set, Type

import pandas as pd
import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from deep_helpers.structs import DataFrameStateCollection, MetricStateCollection, QueueStateCollection, StateCollection
from deep_helpers.structs.enums import Mode, State


class SimpleMetricCollection(MetricStateCollection):
    def __init__(self):
        collection = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=10),
                "f1": MulticlassF1Score(num_classes=10),
            }
        )
        super().__init__(collection)


@dataclass
class IntCollection(StateCollection):
    _lookup: Dict[State, int] = field(default_factory=dict)

    def __hash__(self) -> int:
        return object.__hash__(self)

    @property
    def states(self) -> Set[State]:
        return set(self._lookup.keys())

    def register(self, state: State):
        self._lookup[state] = 0

    def set_state(self, state: State, val: int) -> None:
        self._lookup[state] = val

    def get_state(self, state: State) -> int:
        return self._lookup[state]

    def remove_state(self, state: State) -> None:
        del self._lookup[state]


class BaseCollectionTest:
    CLS: Type[StateCollection]
    VAL: Any

    simple_states = [
        pytest.param(State(Mode.TRAIN, None)),
        pytest.param(State(Mode.VAL, "cifar10")),
        pytest.param(State(Mode.TEST, "imagenet")),
    ]

    @pytest.mark.parametrize("state", simple_states)
    def test_register(self, state):
        col = self.CLS()
        col.register(state)
        assert state in col.states

    @pytest.mark.parametrize("state", simple_states)
    def test_register_already_registered(self, state):
        col = self.CLS()
        col.register(state)
        val = col.get_state(state)
        col.register(state)
        assert col.get_state(state) is val

    def test_hash(self):
        col = self.CLS()
        col2 = self.CLS()
        assert hash(col) != hash(col2)

    @pytest.mark.parametrize("state", simple_states)
    def test_set_state(self, state):
        col = self.CLS()
        col.register(state)
        col.set_state(state, self.VAL)
        assert state in col.states

    @pytest.mark.parametrize("state", simple_states)
    def test_get_state(self, state):
        col = self.CLS()
        col.register(state)
        get = col.get_state(state)
        assert isinstance(get, type(self.VAL))

    @pytest.mark.parametrize("state", simple_states)
    def test_remove_state(self, state):
        col = self.CLS()
        col.register(state)
        assert state in col.states
        col.remove_state(state)
        assert state not in col.states

    @pytest.mark.parametrize("state", simple_states)
    def test_as_dict(self, state):
        col = self.CLS()
        col.register(state)
        d = col.as_dict()
        assert isinstance(d, dict)
        assert len(d) == 1
        assert state in d.keys()

    @pytest.mark.parametrize("state", simple_states)
    def test_add(self, state):
        col = self.CLS()
        col.register(state)

        other_col = self.CLS()
        other_state = State(Mode.PREDICT, "other_state")
        other_col.register(other_state)

        added = col + col + other_col
        assert isinstance(other_col, self.CLS)
        assert added.states == {state, other_state}

    @pytest.mark.parametrize("state", simple_states)
    def test_clear(self, state):
        col = self.CLS()
        col.register(state)
        assert state in col.states
        col.clear()
        assert state not in col.states


class TestStateCollection(BaseCollectionTest):
    CLS = IntCollection
    VAL = 0


class TestMetricStateCollection(BaseCollectionTest):
    CLS: Type[MetricStateCollection] = SimpleMetricCollection
    VAL = MetricCollection({})

    def test_summary(self):
        col = MetricStateCollection(MetricCollection({"foo": MulticlassAccuracy(num_classes=10)}))
        state1 = State(Mode.TRAIN, None)
        state2 = State(Mode.VAL, "cifar10")
        col.register(state1)
        col.register(state2)
        s = col.summarize()
        assert isinstance(s, str)

    def test_to_cpu(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        col.register(state)
        device = torch.device("cpu")
        col2 = col.to(device)
        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert metric.device == device

    @pytest.mark.cuda
    def test_to_gpu(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)
        device = torch.device("cuda:0")
        col2 = col.to(device)
        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert metric.device == device

    @pytest.mark.cuda
    def test_register_gpu(self):
        col = self.CLS()
        device = torch.device("cuda:0")
        state = State(Mode.TEST, "cifar10")
        col.register(state, device=device)
        for s, collection in col.as_dict().items():
            for name, metric in collection.items():
                assert metric.device == device

    def test_reset(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)

        p = torch.rand(10, 10)
        t = torch.rand(10, 10).round().long()

        col.get_state(state).update(p, t)
        col.get_state(state2).update(p, t)
        col2 = col.reset()

        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert (metric.tp == 0).all()  # type: ignore
                assert (metric.fp == 0).all()  # type: ignore
                assert (metric.tn == 0).all()  # type: ignore
                assert (metric.fn == 0).all()  # type: ignore


class TestQueueStateCollection(BaseCollectionTest):
    CLS: Type[QueueStateCollection] = QueueStateCollection
    VAL = PriorityQueue()

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_enqueue(self, state):
        col = self.CLS()
        col.register(state)
        col.enqueue(state, 0, "dog")
        col.enqueue(state, 1, "cat")
        queue = col.get_state(state)
        assert queue.qsize() == col.qsize(state) == 2
        assert not col.empty(state)

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_dequeue(self, state):
        col = self.CLS()
        col.register(state)
        col.enqueue(state, 1, "cat")
        col.enqueue(state, 0, "dog")
        queue = col.get_state(state)
        assert queue.qsize() == 2

        item1 = col.dequeue(state)
        item2 = col.dequeue(state)
        assert item1.priority <= item2.priority
        assert item1.value == "dog"
        assert item2.value == "cat"

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_len(self, state):
        col = self.CLS()
        col.register(state)
        state2 = State(Mode.PREDICT, "state2")
        col.register(state2)

        col.enqueue(state, 0, "dog")
        col.enqueue(state, 1, "cat")
        assert len(col) == 2
        col.enqueue(state2, 0, "dog")
        col.enqueue(state2, 1, "cat")
        assert len(col) == 4

    def test_reset(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)

        p = torch.rand(10, 10)
        col.enqueue(state, 0, p)
        col.enqueue(state2, 0, p)
        col2 = col.reset()

        for s, queue in col2.as_dict().items():
            assert queue.empty()


class TableCollection(DataFrameStateCollection):
    def __init__(self):
        proto = pd.DataFrame(columns=["col1", "col2"])
        super().__init__(proto)


class TestDataFrameStateCollection(BaseCollectionTest):
    CLS = TableCollection
    VAL = pd.DataFrame(columns=["col1", "col2"])
