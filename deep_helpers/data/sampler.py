from typing import Iterator, List, Sequence, Sized, Union, cast

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
    """

    def __init__(
        self,
        samplers: Sequence[SamplerOrSequence],
        batch_samplers: Sequence[BatchSamplerOrSequence],
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

    def __len__(self) -> int:
        return sum(len(cast(Sized, s)) for s in self.batch_samplers)

    def __iter__(self) -> Iterator[List[int]]:
        for batch_sampler_idx, batch_sampler in enumerate(self.batch_samplers):
            for indices in batch_sampler:
                if not isinstance(indices, Sequence):
                    raise TypeError("Expected batch sampler to yield sequences of indices")  # pragma: no cover

                # NOTE: We get indices as a mini-batch from the batch sampler, but we need to
                # offset them by the sum of the lengths of all previous (non-batch) samplers.
                # This is why we need both the batch samplers and the samplers.
                yield [apply_offset(cast(Sequence[Sized], self.samplers), idx, batch_sampler_idx) for idx in indices]
