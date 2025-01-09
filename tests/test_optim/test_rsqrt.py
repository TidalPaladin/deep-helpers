from math import isclose
from typing import Final

import pytest
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from deep_helpers.optim.rsqrt import ReciprocalSquareRootLR, get_lr, get_momentum


TEXT_ARGS: Final = (
    "step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, peak_steps, expected"
)
TEST_CASES: Final = [
    # Start at lr=0.0
    (0, 1.0, 10, 10, 100, 10, 0.0, 0, 0.0),
    # Linear increase to base_lr over warmup_steps
    (5, 1.0, 10, 10, 100, 10, 0.0, 0, 0.5),
    (10, 1.0, 10, 10, 100, 10, 0.0, 0, 1.0),
    # Stay at max lr for nonzero peak_steps
    (10, 1.0, 10, 10, 100, 10, 0.0, 20, 1.0),
    (15, 1.0, 10, 10, 100, 10, 0.0, 20, 1.0),
    (30, 1.0, 10, 10, 100, 10, 0.0, 20, 1.0),
    (31, 1.0, 10, 10, 100, 10, 0.0, 20, 0.9535),
    # Reciprocal square root schedule
    (11, 1.0, 10, 10, 100, 10, 0.0, 0, 0.9535),
    (15, 1.0, 10, 10, 100, 10, 0.0, 0, 0.8165),
    (50, 1.0, 10, 10, 100, 10, 0.0, 0, 0.4472),
    (75, 1.0, 10, 10, 100, 10, 0.0, 0, 0.3651),
    (89, 1.0, 10, 10, 100, 10, 0.0, 0, 0.3352),
    # Linear decrease to lr=0.0 over cooldown_steps
    (90, 1.0, 10, 10, 100, 10, 0.0, 0, 0.3333),
    (95, 1.0, 10, 10, 100, 10, 0.0, 0, 0.1667),
    (100, 1.0, 10, 10, 100, 10, 0.0, 0, 0.0),
    # Warmup over nonzero initial_lr
    (0, 1.1, 10, 10, 100, 10, 0.1, 0, 0.1),
    (5, 1.1, 10, 10, 100, 10, 0.1, 0, 0.6),
    (10, 1.1, 10, 10, 100, 10, 0.1, 0, 1.1),
    # Very small timescale, asymptote to 0.0
    (19000, 1.1, 10, 10, 20000, 1, 0.1, 0, 0.0080),
    # Error handling
    pytest.param(101, 1.1, 10, 10, 100, 10, 0.1, 0, 1.1, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
]

TEXT_ARGS_MOMENTUM: Final = (
    "step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum, peak_steps, expected"
)
TEST_CASES_MOMENTUM: Final = [
    # Start at momentum=0.95
    (0, 0.85, 10, 10, 100, 10, 0.95, 0, 0.95),
    # Linear decrease to base_momentum over warmup_steps
    (5, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9),
    (10, 0.85, 10, 10, 100, 10, 0.95, 0, 0.85),
    # Stay at max momentum for nonzero peak_steps
    (10, 0.85, 10, 10, 100, 10, 0.95, 20, 0.85),
    (15, 0.85, 10, 10, 100, 10, 0.95, 20, 0.85),
    (30, 0.85, 10, 10, 100, 10, 0.95, 20, 0.85),
    (31, 0.85, 10, 10, 100, 10, 0.95, 20, 0.8547),
    # Reciprocal square root schedule
    (11, 0.85, 10, 10, 100, 10, 0.95, 0, 0.8547),
    (15, 0.85, 10, 10, 100, 10, 0.95, 0, 0.8684),
    (50, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9053),
    (75, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9135),
    (89, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9165),
    (int(1e20) - 11, 0.85, 10, 10, int(1e20), 10, 0.95, 0, 0.95),
    # Linear decrease to lr=0.0 over cooldown_steps
    (90, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9167),
    (95, 0.85, 10, 10, 100, 10, 0.95, 0, 0.9333),
    (100, 0.85, 10, 10, 100, 10, 0.95, 0, 0.95),
    # Very small timescale, asymptote to initial_momentum
    (19000, 0.85, 10, 10, 20000, 1, 0.95, 0, 0.9493),
    (19000, 0.95, 10, 10, 20000, 1, 1.0, 0, 0.9996),
    # Error handling
    pytest.param(101, 0.85, 10, 10, 100, 10, 0.95, 0, 0.95, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
]


@pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
def test_get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, peak_steps, expected):
    actual = get_lr(step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, peak_steps)
    assert isclose(actual, expected, abs_tol=1e-4)


@pytest.mark.parametrize(TEXT_ARGS_MOMENTUM, TEST_CASES_MOMENTUM)
def test_get_momentum(
    step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum, peak_steps, expected
):
    actual = get_momentum(
        step, base_momentum, warmup_steps, cooldown_steps, total_steps, timescale, initial_momentum, peak_steps
    )
    assert isclose(actual, expected, abs_tol=1e-4)


class TestReciprocalSquareRootLR:

    @pytest.mark.parametrize(TEXT_ARGS, TEST_CASES)
    def test_get_lr(
        self, step, base_lr, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, peak_steps, expected
    ):
        optim = Adam([nn.Parameter(torch.zeros(1))], lr=base_lr)
        schedule = ReciprocalSquareRootLR(
            optim, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr, peak_steps=peak_steps
        )
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
        peak_steps,
        expected,
    ):
        # Update optimizer to account for change in base momentum
        if base_momentum != 0.85:
            if isinstance(optimizer, Adam):
                optimizer.defaults["betas"] = (base_momentum, 0.999)
                for param_group in optimizer.param_groups:
                    param_group["betas"] = (base_momentum, 0.999)
                    param_group["base_momentum"] = base_momentum
            else:
                optimizer.defaults["momentum"] = base_momentum
                for param_group in optimizer.param_groups:
                    param_group["momentum"] = base_momentum
                    param_group["base_momentum"] = base_momentum

        schedule = ReciprocalSquareRootLR(
            optimizer, warmup_steps, cooldown_steps, total_steps, timescale, 0, initial_momentum, peak_steps=peak_steps
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

    @pytest.mark.parametrize(
        "step, exp",
        [
            (0, [1e-4, 1e-4]),
            (100, [0.1, 0.01]),
            (1000, [0.0, 0.0]),
        ],
    )
    def test_get_lr_multiple_lrs(self, step, exp):
        base_lr = 0.01
        custom_lr = 0.1
        group1 = {"params": [nn.Parameter(torch.zeros(1))], "lr": custom_lr}
        group2 = {"params": [nn.Parameter(torch.zeros(1))], "lr": base_lr}
        optim = Adam([group1, group2], lr=base_lr)

        warmup_steps = 100
        cooldown_steps = 10
        total_steps = 1000
        timescale = 10
        initial_lr = 1e-4
        schedule = ReciprocalSquareRootLR(optim, warmup_steps, cooldown_steps, total_steps, timescale, initial_lr)
        schedule._step_count = step
        actual = schedule.get_lr()
        assert actual == exp
