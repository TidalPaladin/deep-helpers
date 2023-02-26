#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from torch import Tensor

from ..structs import Mode


D = TypeVar("D", bound=Dict[str, Any])
DatasetID = Tuple[Mode, str]


@dataclass
class DatasetNames:
    lookup: Dict[Mode, Dict[int, str]] = field(default_factory=dict)

    def __setitem__(self, key: Tuple[Mode, int], value: str) -> None:
        mode, dataloader_idx = key
        dataloader_names = self.lookup.get(mode, {})
        dataloader_names[dataloader_idx] = value
        self.lookup[mode] = dataloader_names

    def __contains__(self, target: Union[Mode, Tuple[Mode, Optional[int]]]) -> bool:
        if isinstance(target, Mode):
            return target in self.lookup
        elif isinstance(target, tuple):
            mode, dataloader_idx = target
            dataloader_idx = dataloader_idx if dataloader_idx is not None else 0
            return mode in self.lookup and dataloader_idx in self.lookup[mode]

    @overload
    def __getitem__(self, target: Mode) -> Dict[int, str]:
        pass

    @overload
    def __getitem__(self, target: Tuple[Mode, Optional[int]]) -> str:
        pass

    def __getitem__(self, target: Union[Mode, Tuple[Mode, Optional[int]]]) -> Union[str, Dict[int, str]]:
        if target not in self:
            raise KeyError(str(target))

        if isinstance(target, Mode):
            return copy(self.lookup[target])
        elif isinstance(target, tuple):
            mode, dataloader_idx = target
            dataloader_idx = dataloader_idx if dataloader_idx is not None else 0
            return self.lookup[mode][dataloader_idx]

    @property
    def all_names(self) -> Iterator[str]:
        for dataloader_names in self.lookup.values():
            for name in dataloader_names.values():
                yield name

    def names_for_mode(self, mode: Mode) -> Iterator[str]:
        if mode not in self.lookup:
            return
        for name in self.lookup[mode].values():
            yield name


@runtime_checkable
class SupportsDatasetNames(Protocol):
    dataset_names: DatasetNames


def uncollate(batch: D) -> Iterator[D]:
    r"""Uncollates a batch dictionary into an iterator of example dictionaries.
    This is the inverse of :func:`collate_fn`. Non-sequence elements are repeated
    for each example in the batch. If examples in the batch have different
    sequence lengths, the iterator will be truncated to the shortest sequence.
    Args:
        batch: The batch dictionary to uncollate.
    Returns:
        An iterator of example dictionaries.
    """
    # separate out sequence-like elements and compute a batch size
    sequences = {k: v for k, v in batch.items() if isinstance(v, (Sequence, Tensor))}
    batch_size = min((len(v) for v in sequences.values()), default=0)

    # repeat non-sequence elements
    non_sequences = {k: [v] * batch_size for k, v in batch.items() if not isinstance(v, (Sequence, Tensor))}

    for idx in range(batch_size):
        result = {k: v[idx] for container in (sequences, non_sequences) for k, v in container.items()}
        yield cast(D, result)
