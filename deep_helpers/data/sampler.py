from functools import cached_property
from itertools import chain, islice
from typing import Iterator, List, Sequence, Sized, TypeVar, Union, cast

from torch.utils.data import BatchSampler, Sampler


SamplerOrSequence = Union[Sampler[int], Sequence[int]]
BatchSamplerOrSequence = Union[BatchSampler, Sampler[List[int]], Sequence[Sequence[int]]]


def apply_offset(samplers: Sequence[Sized], idx: int, sampler_idx: int) -> int:
    r"""Applies an offset to an index from one sampler in a sequence, adjusting it
    to the total length of all samplers. The offset is the sum of the lengths of
    all samplers with a lower index than ``sampler_idx``.

    Args:
        samplers: Samplers from which the offset is calculated.
        idx: Index from the source sampler.
        sampler_idx: Index of the source sampler.

    Returns:
        The index with the offset applied.
    """
    if not all(isinstance(sampler, Sized) for sampler in samplers):
        raise TypeError("All inputs must be Sized")  # pragma: no cover
    if not 0 <= sampler_idx < len(samplers):
        raise IndexError(f"Sampler index {sampler_idx} out of range [0, {len(samplers)})")  # pragma: no cover

    offset = 0
    for i in range(sampler_idx):
        offset += len(cast(Sized, samplers[i]))
    return idx + offset


class ConcatSampler(Sampler):
    r"""Sampler that concatenates multiple samplers together. All input samplers must
    be ``Sized``. The length of the concatenated sampler is the sum of the lengths of
    input samplers. Each sampler is iterated in order, and the indices are offset
    by the sum of the lengths of all previous samplers.

    Args:
        samplers: The samplers to concatenate.
    """

    def __init__(self, samplers: Sequence[SamplerOrSequence]) -> None:
        self.samplers = samplers

    def __len__(self) -> int:
        return sum(len(cast(Sized, s)) for s in self.samplers)

    def __iter__(self) -> Iterator[int]:
        for sampler_idx, sampler in enumerate(self.samplers):

            def map_fn(idx):
                return apply_offset(cast(Sequence[Sized], self.samplers), idx, sampler_idx)

            yield from map(map_fn, iter(sampler))


T = TypeVar("T")


def interleave(*args: Iterator[T]) -> Iterator[T]:
    iterables: List[Iterator[T]] = list(args)
    while iterables:
        for iterable in iterables:
            try:
                yield next(iterable)
            except StopIteration:
                iterables.remove(iterable)


# TODO: This is built-in to Python 3.12
def batch(iterable: Iterator[T], n: int) -> Iterator[List[T]]:
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            return
        yield chunk


class ConcatBatchSampler(BatchSampler):
    r"""Sampler that concatenates multiple batch samplers together. All input samplers must
    be ``Sized``. The length of the concatenated sampler is the sum of the lengths of
    input batch samplers. Each batch sampler is iterated in order, and the indices are offset
    by the sum of the lengths of all previous samplers.

    Args:
        samplers: The samplers for which the batch samplers are defined.
        batch_samplers: The batch samplers to concatenate. The number of batch samplers must
            equal the number of samplers. It is expected that the i'th batch sampler yields
            indices from the i'th sampler.
        method: The method to use for iterating over the batch samplers. Must be one of
            "sequential", "cycle", or "zip. "sequential" iterates over the batch samplers
            sequentially. "cycle" draws a batch from each sampler in turn. "zip" splices
            individual examples from each batch sampler together.
    """

    def __init__(
        self,
        samplers: Sequence[SamplerOrSequence],
        batch_samplers: Sequence[BatchSamplerOrSequence],
        method: str = "sequential",
    ) -> None:
        if not all(isinstance(sampler, Sized) for sampler in samplers):
            raise TypeError("All samplers must be Sized")  # pragma: no cover
        if not all(isinstance(sampler, Sized) for sampler in batch_samplers):
            raise TypeError("All batch samplers must be Sized")  # pragma: no cover
        if len(samplers) != len(batch_samplers):
            raise ValueError(
                f"Number of samplers and batch samplers must be equal: {len(samplers)} != {len(batch_samplers)}"
            )  # pragma: no cover

        self.samplers = samplers
        self.batch_samplers = batch_samplers
        self.method = method

    def __len__(self) -> int:
        return sum(len(cast(Sized, s)) for s in self.batch_samplers)

    @cached_property
    def batch_size(self) -> int:
        return len(next(iter(self.batch_samplers[0])))

    def _iterate_offset_adjusted_batches(self, batch_sampler_idx) -> Iterator[List[int]]:
        batch_sampler = self.batch_samplers[batch_sampler_idx]
        for batch in batch_sampler:
            yield [apply_offset(cast(Sequence[Sized], self.samplers), idx, batch_sampler_idx) for idx in batch]

    def __iter__(self) -> Iterator[List[int]]:
        # Set up iterators that already have the offsets applied.
        batch_sampler_iterators = [
            self._iterate_offset_adjusted_batches(batch_sampler_idx)
            for batch_sampler_idx in range(len(self.batch_samplers))
        ]

        batch_iterator: Iterator[List[int]]
        if self.method == "sequential":
            # In this case we just chain the iterators together.
            batch_iterator = chain(*batch_sampler_iterators)
        elif self.method == "cycle":
            # In this case we interleave the iterators together at the batch level
            batch_iterator = interleave(*batch_sampler_iterators)
        elif self.method == "zip":
            # In this case we interleave the iterators together at the example level
            batch_iterator = (
                list(zipped_batch)
                # First we interleave the batch samplers, getting one batch from each sampler
                for interleaved_batch in batch(interleave(*batch_sampler_iterators), len(self.batch_samplers))
                # Then we interleave the examples from each batch sampler
                for zipped_batch in batch(interleave(*(iter(i) for i in interleaved_batch)), self.batch_size)
            )
        else:
            raise ValueError(f"Invalid method: {self.method}")  # pragma: no cover

        yield from batch_iterator
