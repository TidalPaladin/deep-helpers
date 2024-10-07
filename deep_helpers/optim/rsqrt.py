from typing import Any, Dict, List

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def get_lr(
    step: int, base_lr: float, warmup_steps: int, cooldown_steps: int, total_steps: int, timescale: int
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

    Returns:
        Learning rate at the given step.
    """

    # Warmup is linear from 0 to base_lr over warmup_steps
    if step <= warmup_steps:
        return base_lr * step / warmup_steps

    # Find point along the reciprocal square root schedule
    rsqrt_step = min(step, total_steps - cooldown_steps)
    lr = base_lr * (timescale / (rsqrt_step - (warmup_steps - timescale))) ** 0.5

    # Cooldown is linear from current lr to 0 over cooldown_steps
    if step >= total_steps - cooldown_steps:
        return lr * (total_steps - step) / cooldown_steps

    # Otherwise we are not yet in cooldown, so we use the reciprocal square root schedule
    return lr


class ReciprocalSquareRootLR(LRScheduler):
    """
    Implements a learning rate scheduler with a warmup, reciprocal square root, and cooldown schedule.

    This scheduler adjusts the learning rate according to the following phases:
    1. Warmup: Linearly increases the learning rate from 0 to the base learning rate over a specified number of steps.
    2. Reciprocal Square Root: Adjusts the learning rate according to a reciprocal square root schedule.
    3. Cooldown: Linearly decreases the learning rate from the current learning rate to 0 over a specified number of steps.

    Args:
        warmup_steps: Number of steps for the warmup phase.
        cooldown_steps: Number of steps for the cooldown phase.
        total_steps: Total number of steps in the training process.
        timescale: Timescale parameter for the reciprocal square root schedule.
        optimizer: Wrapped optimizer.
        last_epoch: The index of the last epoch. Default: -1.

    Shapes:
        - base_lrs: :math:`(N,)` where :math:`N` is the number of parameter groups.
        - get_lr: :math:`(N,)` where :math:`N` is the number of parameter groups.
    """

    def __init__(
        self,
        warmup_steps: int,
        cooldown_steps: int,
        total_steps: int,
        timescale: int,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.timescale = timescale
        super().__init__(optimizer, last_epoch)
        self._step_count = 0

    def get_lr(self) -> List[float]:
        step = self._step_count
        return [
            get_lr(step, base_lr, self.warmup_steps, self.cooldown_steps, self.total_steps, self.timescale)
            for base_lr in self.base_lrs
        ]

    def load_state_dict(self, state_dict: Dict[str, Any]):
        total_steps = self.total_steps
        self.__dict__.update(state_dict)
        self.total_steps = total_steps
