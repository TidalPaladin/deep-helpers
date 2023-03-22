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


def is_tensor_or_sequence(data: Any) -> bool:
    return isinstance(data, (Tensor, Sequence))


def is_multidim_tensor(data: Any) -> bool:
    return bool(isinstance(data, Tensor) and data.ndim and data.numel())


def uncollate(batch: D) -> Iterator[D]:
    r"""Uncollates a batch dictionary into an iterator of example dictionaries.
    This is the inverse of :func:`collate_fn`. Non-sequence elements are repeated
    for each example in the batch. If examples in the batch have different
    sequence lengths, the iterator will be truncated to the shortest sequence. If length-1
    elements are present, they are expanded to match the batch size.

    Args:
        batch: The batch dictionary to uncollate.
    Returns:
        An iterator of example dictionaries.
    """
    # separate out sequence-like elements
    sequences = {k: v for k, v in batch.items() if is_tensor_or_sequence(v) and not isinstance(v, str)}

    # get the lengths of every sequence or non-0d tensor
    lengths = {k: len(v) for k, v in sequences.items() if (is_multidim_tensor(v) or isinstance(v, Sequence)) and len(v)}

    # Compute a batch size based on the minimum length of all sequences.
    # We will try to skip length 1 elements under the assumption they will be expanded
    has_nontrivial_len = any(v > 1 for v in lengths.values())
    if has_nontrivial_len:
        batch_size = min((v for v in lengths.values() if v > 1), default=0)
    else:
        batch_size = min(lengths.values(), default=0)

    # expand any length-1 sequences to the batch size
    for k, v in sequences.items():
        if isinstance(v, Tensor) and v.numel():
            sequences[k] = (
                v.view(1).expand(batch_size) if not is_multidim_tensor(v) else v.expand(batch_size, *v.shape[1:])
            )
        elif len(v) == 1:
            sequences[k] = list(v) * batch_size

    # repeat non-sequence elements
    non_sequences = {
        k: [v] * batch_size for k, v in batch.items() if not isinstance(v, (Sequence, Tensor)) or isinstance(v, str)
    }

    for idx in range(batch_size):
        result = {k: v[idx] for container in (sequences, non_sequences) for k, v in container.items()}
        yield cast(D, result)
