from typing import Any, Dict, List

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def get_lr(
    step: int,
    base_lr: float,
    warmup_steps: int,
    cooldown_steps: int,
    total_steps: int,
    timescale: int,
    initial_lr: float,
    peak_steps: int,
) -> float:
    """
    Calculate the learning rate at a given step using a warmup, reciprocal square root, and cooldown schedule.

    Args:
        step: Current step in the training process.
        base_lr: Base learning rate.
        warmup_steps: Number of steps for the warmup phase.
        cooldown_steps: Number of steps for the cooldown phase.
        total_steps: Total number of steps in the training process.
        timescale: Timescale parameter for the reciprocal square root schedule.
        initial_lr: Initial learning rate.
        peak_steps: Number of steps for the peak phase.

    Returns:
        Learning rate at the given step.
    """
    if step > total_steps:
        raise ValueError(f"Step {step} is greater than total steps {total_steps}")

    # Warmup is linear from initial_lr to base_lr over warmup_steps
    if step <= warmup_steps:
        assert initial_lr <= base_lr
        return initial_lr + (base_lr - initial_lr) * step / warmup_steps

    # Find point along the reciprocal square root schedule
    rsqrt_step = min(step, total_steps - cooldown_steps - peak_steps)
    lr = base_lr * (timescale / (rsqrt_step - (warmup_steps + peak_steps - timescale))) ** 0.5

    # Cooldown is linear from current lr to 0 over cooldown_steps
    if step >= total_steps - cooldown_steps:
        return lr * (total_steps - step) / cooldown_steps

    # Otherwise we are not yet in cooldown, so we use the reciprocal square root schedule
    return lr


def get_momentum(
    step: int,
    base_momentum: float,
    warmup_steps: int,
    cooldown_steps: int,
    total_steps: int,
    timescale: int,
    initial_momentum: float,
    peak_steps: int,
) -> float:
    """
    Calculate the momentum at a given step using a warmup, reciprocal square root, and cooldown schedule.

    It is assumed that momentum is cycled inversely to the learning rate, i.e. ``base_momentum <= initial_momentum``.
    Momentum warms up from ``initial_momentum`` to ``base_momentum`` over ``warmup_steps``.
    Then it follows a reciprocal square root schedule until the cooldown phase,
    where it linearly increases from the current momentum back to ``initial_momentum`` over ``cooldown_steps``.

    Args:
        step: Current step in the training process.
        base_momentum: Base momentum.
        warmup_steps: Number of steps for the warmup phase.
        cooldown_steps: Number of steps for the cooldown phase.
        total_steps: Total number of steps in the training process.
        timescale: Timescale parameter for the reciprocal square root schedule.
        initial_momentum: Initial momentum.
        peak_steps: Number of steps for the peak phase.

    Returns:
        Learning rate at the given step.
    """
    if step > total_steps:
        raise ValueError(f"Step {step} is greater than total steps {total_steps}")

    # Warmup is linear from initial_momentum to base_momentum over warmup_steps
    if step <= warmup_steps:
        assert initial_momentum >= base_momentum
        return initial_momentum - (initial_momentum - base_momentum) * step / warmup_steps

    # Find point along the reciprocal square root schedule
    rsqrt_step = min(step, total_steps - cooldown_steps - peak_steps)
    rsqrt = (timescale / (rsqrt_step - (warmup_steps + peak_steps - timescale))) ** 0.5
    momentum = initial_momentum - (initial_momentum - base_momentum) * rsqrt

    # Cooldown is linear from current momentum to initial_momentum over cooldown_steps
    if step >= total_steps - cooldown_steps:
        scale = 1 + (step - total_steps) / cooldown_steps
        return momentum + (initial_momentum - momentum) * scale

    # Otherwise we are not yet in cooldown, so we use the reciprocal square root schedule
    return momentum


class ReciprocalSquareRootLR(LRScheduler):
    """
    Implements a learning rate scheduler with a warmup, reciprocal square root, and cooldown schedule.

    This scheduler adjusts the learning rate according to the following phases:
    1. Warmup: Linearly increases the learning rate from ``initial_lr`` to the base learning rate over a specified number of steps.
    2. Reciprocal Square Root: Adjusts the learning rate according to a reciprocal square root schedule.
    3. Cooldown: Linearly decreases the learning rate from the current learning rate to 0 over a specified number of steps.

    If ``initial_momentum`` is specified, the momentum is cycled inversely to the learning rate.
    The momentum linearly warms up from ``initial_momentum`` to the momentum value set in the optimizer,
    then it follows a reciprocal square root schedule asymptotically towards ``base_momentum``. The cooldown
    phase linearly decreases the momentum from the current momentum to ``initial_momentum`` over the same number of steps.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for the warmup phase.
        cooldown_steps: Number of steps for the cooldown phase.
        total_steps: Total number of steps in the training process.
        timescale: Timescale parameter for the reciprocal square root schedule.
        initial_lr: Initial learning rate.
        initial_momentum: Initial momentum, or None to not change momentum.
        last_epoch: The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        cooldown_steps: int,
        total_steps: int,
        timescale: int,
        initial_lr: float = 0.0,
        initial_momentum: float | None = None,
        peak_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.timescale = timescale
        self.initial_lr = initial_lr
        self.initial_momentum = initial_momentum
        self.peak_steps = peak_steps
        super().__init__(optimizer, last_epoch)
        self._step_count = 0

    def get_lr(self) -> List[float]:
        step = self._step_count

        # Perform an update of momentum
        if self.initial_momentum is not None:
            for param_group in self.optimizer.param_groups:
                if "momentum" in param_group:
                    momentum = param_group["momentum"]
                    base_momentum = param_group.setdefault("base_momentum", momentum)
                    new_momentum = get_momentum(
                        step,
                        base_momentum,
                        self.warmup_steps,
                        self.cooldown_steps,
                        self.total_steps,
                        self.timescale,
                        self.initial_momentum,
                        self.peak_steps,
                    )
                    param_group["momentum"] = new_momentum
                elif "betas" in param_group:
                    momentum, _ = param_group["betas"]
                    base_momentum = param_group.setdefault("base_momentum", momentum)
                    new_momentum = get_momentum(
                        step,
                        base_momentum,
                        self.warmup_steps,
                        self.cooldown_steps,
                        self.total_steps,
                        self.timescale,
                        self.initial_momentum,
                        self.peak_steps,
                    )
                    param_group["betas"] = (new_momentum, param_group["betas"][1])

        return [
            get_lr(
                step,
                base_lr,
                self.warmup_steps,
                self.cooldown_steps,
                self.total_steps,
                self.timescale,
                self.initial_lr,
                self.peak_steps,
            )
            for base_lr in self.base_lrs
        ]

    def load_state_dict(self, state_dict: Dict[str, Any]):
        total_steps = self.total_steps
        self.__dict__.update(state_dict)
        self.total_steps = total_steps
