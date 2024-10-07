from math import isclose
from typing import Final

import pytest
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from deep_helpers.optim.rsqrt import ReciprocalSquareRootLR, get_lr


TEXT_ARGS: Final = "step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, expected"
TEST_CASES: Final = [
    # Start at lr=0.0
    (0, 1.0, 10, 10, 100, 10, 0.0),
    # Linear increase to base_lr over warmup_steps
    (5, 1.0, 10, 10, 100, 10, 0.5),
    (10, 1.0, 10, 10, 100, 10, 1.0),
    # Reciprocal square root schedule
    (11, 1.0, 10, 10, 100, 10, 0.9535),
    (15, 1.0, 10, 10, 100, 10, 0.8165),
    (50, 1.0, 10, 10, 100, 10, 0.4472),
    (75, 1.0, 10, 10, 100, 10, 0.3651),
    (89, 1.0, 10, 10, 100, 10, 0.3352),
    # Linear decrease to lr=0.0 over cooldown_steps
    (90, 1.0, 10, 10, 100, 10, 0.3333),
    (95, 1.0, 10, 10, 100, 10, 0.1667),
    (100, 1.0, 10, 10, 100, 10, 0.0),
]


@pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
def test_get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, expected):
    actual = get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale)
    assert isclose(actual, expected, abs_tol=1e-4)


class TestReciprocalSquareRootLR:

    @pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
    def test_get_lr(self, step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, expected):
        optim = Adam([nn.Parameter(torch.zeros(1))], lr=base_lr)
        schedule = ReciprocalSquareRootLR(warmup_steps, cooldown_steps, total_steps, timescale, optim)
        schedule._step_count = step
        actual = schedule.get_lr()[0]
        assert isclose(actual, expected, abs_tol=1e-4)
