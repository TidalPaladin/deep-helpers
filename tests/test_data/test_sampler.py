from typing import List

import pytest

from deep_helpers.data.sampler import BatchSampler, ConcatBatchSampler, ConcatSampler, Sampler


class MockSampler(Sampler):
    def __init__(self, data: List[int]):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class TestConcatSampler:
    @pytest.mark.parametrize(
        "samplers, exp",
        [
            ([MockSampler([1, 2, 3]), MockSampler([1, 2, 3])], 6),
        ],
    )
    def test_len(self, samplers: List[Sampler[int]], exp: int):
        concat_sampler = ConcatSampler(samplers)
        assert len(concat_sampler) == exp

    @pytest.mark.parametrize(
        "samplers, expected_output",
        [
            ([MockSampler([1, 2, 3]), MockSampler([1, 2, 3])], [1, 2, 3, 4, 5, 6]),
            ([MockSampler([3, 2, 1]), MockSampler([3, 2, 1])], [3, 2, 1, 6, 5, 4]),
        ],
    )
    def test_iter(self, samplers: List[Sampler[int]], expected_output: List[int]):
        concat_sampler = ConcatSampler(samplers)
        assert list(concat_sampler) == expected_output


class MockBatchSampler(BatchSampler):
    def __init__(self, data: List[List[int]]):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class TestConcatBatchSampler:
    @pytest.mark.parametrize(
        "samplers, batch_samplers, exp",
        [
            (
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ],
                [
                    MockBatchSampler([[1, 2], [3, 4]]),
                    MockBatchSampler([[1, 2], [3, 4]]),
                ],
                4,
            ),
        ],
    )
    def test_len(self, samplers: List[List[int]], batch_samplers: List[MockBatchSampler], exp: int):
        concat_sampler = ConcatBatchSampler(samplers, batch_samplers)
        assert len(concat_sampler) == exp

    @pytest.mark.parametrize(
        "samplers, batch_samplers, expected_output",
        [
            (
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ],
                [
                    MockBatchSampler([[1, 2], [3, 4]]),
                    MockBatchSampler([[1, 2], [3, 4]]),
                ],
                [[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
            (
                [
                    [1, 2, 3, 4],
                    [4, 3, 2, 1],
                ],
                [
                    MockBatchSampler([[1, 2], [3, 4]]),
                    MockBatchSampler([[4, 3], [2, 1]]),
                ],
                [[1, 2], [3, 4], [8, 7], [6, 5]],
            ),
        ],
    )
    def test_iter(
        self, samplers: List[List[int]], batch_samplers: List[MockBatchSampler], expected_output: List[List[int]]
    ):
        concat_sampler = ConcatBatchSampler(samplers, batch_samplers)
        assert list(concat_sampler) == expected_output
