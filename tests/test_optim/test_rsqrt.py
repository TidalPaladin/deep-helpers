from math import isclose
from typing import Final

import pytest
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from deep_helpers.optim.rsqrt import ReciprocalSquareRootLR, get_lr, get_momentum


TEXT_ARGS: Final = "step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, expected"
TEST_CASES: Final = [
    # Start at lr=0.0
    (0, 1.0, 10, 10, 100, 10, 0.0, 0.0),
    # Linear increase to base_lr over warmup_steps
    (5, 1.0, 10, 10, 100, 10, 0.0, 0.5),
    (10, 1.0, 10, 10, 100, 10, 0.0, 1.0),
    # Reciprocal square root schedule
    (11, 1.0, 10, 10, 100, 10, 0.0, 0.9535),
    (15, 1.0, 10, 10, 100, 10, 0.0, 0.8165),
    (50, 1.0, 10, 10, 100, 10, 0.0, 0.4472),
    (75, 1.0, 10, 10, 100, 10, 0.0, 0.3651),
    (89, 1.0, 10, 10, 100, 10, 0.0, 0.3352),
    # Linear decrease to lr=0.0 over cooldown_steps
    (90, 1.0, 10, 10, 100, 10, 0.0, 0.3333),
    (95, 1.0, 10, 10, 100, 10, 0.0, 0.1667),
    (100, 1.0, 10, 10, 100, 10, 0.0, 0.0),
    # Warmup over nonzero initial_lr
    (0, 1.1, 10, 10, 100, 10, 0.1, 0.1),
    (5, 1.1, 10, 10, 100, 10, 0.1, 0.6),
    (10, 1.1, 10, 10, 100, 10, 0.1, 1.1),
]

TEXT_ARGS_MOMENTUM: Final = (
    "step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum, expected"
)
TEST_CASES_MOMENTUM: Final = [
    # Start at momentum=0.95
    (0, 0.85, 10, 10, 100, 10, 0.95, 0.95),
    # Linear decrease to base_momentum over warmup_steps
    (5, 0.85, 10, 10, 100, 10, 0.95, 0.9),
    (10, 0.85, 10, 10, 100, 10, 0.95, 0.85),
    # Reciprocal square root schedule
    (11, 0.85, 10, 10, 100, 10, 0.95, 0.8547),
    (15, 0.85, 10, 10, 100, 10, 0.95, 0.8684),
    (50, 0.85, 10, 10, 100, 10, 0.95, 0.9053),
    (75, 0.85, 10, 10, 100, 10, 0.95, 0.9135),
    (89, 0.85, 10, 10, 100, 10, 0.95, 0.9165),
    (int(1e20) - 11, 0.85, 10, 10, int(1e20), 10, 0.95, 0.95),
    # Linear decrease to lr=0.0 over cooldown_steps
    (90, 0.85, 10, 10, 100, 10, 0.95, 0.9167),
    (95, 0.85, 10, 10, 100, 10, 0.95, 0.9333),
    (100, 0.85, 10, 10, 100, 10, 0.95, 0.95),
]


@pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
def test_get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, expected):
    actual = get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr)
    assert isclose(actual, expected, abs_tol=1e-4)


@pytest.mark.parametrize(TEXT_ARGS_MOMENTUM, TEST_CASES_MOMENTUM)
def test_get_momentum(
    step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum, expected
):
    actual = get_momentum(step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum)
    assert isclose(actual, expected, abs_tol=1e-4)


class TestReciprocalSquareRootLR:

    @pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
    def test_get_lr(self, step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, expected):
        optim = Adam([nn.Parameter(torch.zeros(1))], lr=base_lr)
        schedule = ReciprocalSquareRootLR(optim, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr)
        schedule._step_count = step
        actual = schedule.get_lr()[0]
        assert isclose(actual, expected, abs_tol=1e-4)

    @pytest.mark.parametrize(
        "optimizer",
        [
            Adam([nn.Parameter(torch.zeros(1))], lr=0.001, betas=(0.85, 0.999)),
            SGD([nn.Parameter(torch.zeros(1))], lr=0.001, momentum=0.85),
        ],
    )
    @pytest.mark.parametrize(TEXT_ARGS_MOMENTUM, TEST_CASES_MOMENTUM)
    def test_set_momentum(
        self,
        optimizer,
        step,
        base_momentum,
        warmup_steps,
        cooldown_steps,
        total_steps,
        timescale,
        initial_momentum,
        expected,
    ):
        schedule = ReciprocalSquareRootLR(
            optimizer, warmup_steps, cooldown_steps, total_steps, timescale, 0, initial_momentum
        )
        schedule._step_count = step
        schedule.get_lr()
        if "betas" in optimizer.param_groups[0]:
            assert isclose(optimizer.param_groups[0]["betas"][0], expected, abs_tol=1e-4)
            assert optimizer.param_groups[0]["betas"][1] == 0.999
        else:
            assert isclose(optimizer.param_groups[0]["momentum"], expected, abs_tol=1e-4)

    def test_checkpoint(self):
        optim = Adam([nn.Parameter(torch.zeros(1))], lr=0.1)
        total_steps = 100
        cooldown_steps = 10
        schedule = ReciprocalSquareRootLR(optim, 10, cooldown_steps, total_steps, 10, initial_momentum=0.95)

        # Run the schedule to the start of cooldown
        step = total_steps - cooldown_steps
        for _ in range(step):
            optim.step()
            schedule.step()

        # Load the schedules state dict into a new schedule with more total_steps
        state_dict = schedule.state_dict()
        optim2 = Adam([nn.Parameter(torch.zeros(1))], lr=0.1)
        schedule2 = ReciprocalSquareRootLR(optim2, 10, cooldown_steps, total_steps * 2, 10, initial_momentum=0.95)
        schedule2.load_state_dict(state_dict)

        # The schedules should be at the same LR and momentum, but with one having more steps
        assert schedule2._step_count == schedule._step_count
        assert schedule2.get_lr() == schedule.get_lr()
        assert schedule2.total_steps == total_steps * 2
        assert schedule2.optimizer.param_groups[0]["betas"] == schedule.optimizer.param_groups[0]["betas"]
