#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Iterable, Optional, Set, Union, cast


class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    PREDICT = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def create(cls, val: Union[str, "Mode"]) -> "Mode":
        if isinstance(val, Mode):
            return val
        elif isinstance(val, str):
            val = val.strip().lower()
            for mode in cast(Iterable, cls):
                if str(mode) == val:
                    return mode
            raise ValueError(f"Invalid mode: {val}")
        else:
            raise TypeError(f"Invalid type: {type(val)}")


@dataclass(frozen=True)
class State:
    mode: Mode = Mode.PREDICT
    dataset: Optional[str] = None
    sanity_checking: bool = False

    _seen_datasets: Set[str] = field(default_factory=set, repr=False)

    def __eq__(self, other: "State") -> bool:
        r"""Two states are equal if they have the same ``mode`` and ``dataset``"""
        return self.mode == other.mode and self.dataset == other.dataset

    def __hash__(self) -> int:
        r"""State hash is based on ``mode`` and ``dataset``"""
        return hash(self.mode) + hash(self.dataset)

    def update(self, mode: Mode, dataset: Optional[str]) -> "State":
        return self.set_mode(mode).set_dataset(dataset)

    def set_mode(self, mode: Mode) -> "State":
        return replace(self, mode=mode)

    def set_dataset(self, name: Optional[str]) -> "State":
        if name is None:
            return replace(self, dataset=None)
        seen_datasets = self._seen_datasets.union({name})
        return replace(self, dataset=name, _seen_datasets=seen_datasets)

    def set_sanity_checking(self, value: bool) -> "State":
        return replace(self, sanity_checking=value)

    @property
    def prefix(self) -> str:
        # TODO: are we sure we want to hide train dataset name?
        if self.mode == Mode.TRAIN or self.dataset is None:
            return f"{str(self.mode)}/"
        else:
            return f"{str(self.mode)}/{self.dataset}/"

    def with_postfix(self, postfix: str) -> str:
        return f"{self.prefix}{postfix}"
