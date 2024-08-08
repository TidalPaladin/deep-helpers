#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.memory import recursive_detach

from ..data import uncollate
from ..structs import Mode, PrioritizedItem, QueueStateCollection, State
from ..tasks import I, O, Task
from .base import LoggingCallback, T


@dataclass
class QueuedLoggingCallback(LoggingCallback[I, O, T], ABC):
    r"""Callback that implements a limited size priority queue for items seen during an epoch.
    Only the top-k highest priority items from the epoch are retained. All items in the queue are
    logged at epoch completion, or at an interval if desired.

    Args:
        name:
            Name / tag under which to log. State information will be prepended to ``name``.

        modes:
            Modes for which this callback should execute.

        queue_size:
            Size of the priority queue

        target_cls:
            Type of :class:`LoggingTarget` that this callback should create from :class:`Example`,
            :class:`Prediction` pairs.

        flush_interval:
            By default, items will be enqueued over the course of an epoch and logged when the epoch
            concludes. Specifying ``flush_interval`` will flush the priority queue every ``flush_interval`` steps.
            If a ``flush_interval`` is specified, items in the queue at the end of an epoch will be discarded.

        negate_priority:
            If ``True``, use the negation of the priority return by :func:`get_priority`. Use this to log only
            the bottom-k priority items.
    """

    queue_size: int = 8
    flush_interval: int = 0
    negate_priority: bool = False
    queues: QueueStateCollection = field(default_factory=QueueStateCollection)

    _last_flush_step: int = field(init=False, default=0, repr=False)

    @abstractclassmethod
    def get_priority(cls, example: Dict[str, Any], pred: Dict[str, Any]) -> Optional[Union[int, float]]:
        r"""Compute a priority for an example/prediction pair. When logging with a finite
        sized priority queue, only the ``len(queue)`` highest priority images will be logged.
        Typically priority would be assigned based on some metric (loss, entropy, error, etc.).
        If ``None`` is returned, assume the item should not be queued.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.queues)

    def register(self, state: State, *args, **kwargs) -> None:
        r"""Register a queue for a given state."""
        if state not in self.queues.states:
            self.queues.register(state, maxsize=self.queue_size)

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        r"""Reset the state of this logging callback"""
        self.queues.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = pl_module.state

        # try to put this batch into the queue
        self.enqueue(batch, outputs, self.queues.get_state(state))

        # if a flush interval was specified, check if we need to flush
        # TODO should we use batch_idx for checking against flush_interval?
        step = trainer.global_step
        if self.flush_interval and (step % self.flush_interval == 0) and step != self._last_flush_step:
            self.flush_queues(pl_module, state.mode, step)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        mode: Mode,
    ):
        # discard queue at epoch end when using a flush_interval
        if self.flush_interval and not pl_module.state.sanity_checking:
            self.reset(specific_modes=[mode])

        # otherwise flush and log the queue
        else:
            step = trainer.global_step
            self.flush_queues(pl_module, mode, step)

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: Task):
        self.flush_queues(pl_module, pl_module.state.mode, trainer.global_step)
        super().on_sanity_check_end(trainer, pl_module)

    def flush_queues(self, pl_module: Task, mode: Mode, step: int):
        # ensure we only flush queues for the currently ending state
        queues_to_flush: Dict[State, PriorityQueue[PrioritizedItem]] = {
            state: queue for state, queue in self.queues.as_dict().items() if state.mode == mode
        }

        # dequeue and log all targets
        for state, queue in queues_to_flush.items():
            tag = state.with_postfix(self.name)
            targets = [
                self.prepare_target(pl_module.trainer, pl_module, pred, example, 0)
                for example, pred in self.dequeue_all(queue)
            ]
            self.wrapped_log(targets, pl_module, tag, step)
        self._last_flush_step = step

    @torch.no_grad()
    def enqueue(
        self,
        example: I,
        pred: O,
        queue: PriorityQueue[PrioritizedItem],
        batched: bool = True,
        detach: bool = True,
    ) -> bool:
        r"""Enqueue an example/prediction pair to a given queue"""
        # recurse on batched input
        if batched:
            success = False
            for e, p in zip(uncollate(example), uncollate(pred)):  # type: ignore
                e: I
                p: O
                success = success or self.enqueue(e, p, queue, batched=False, detach=detach)
            return success

        # compute priority for enqueue target
        priority = self.get_priority(example, pred)
        if priority is None:
            return False
        priority = priority if not self.negate_priority else -1 * priority

        # Compare priority to the lowest priority item in the queue to determine if we should enqueue
        item = PrioritizedItem(priority, (example, pred))
        if queue.full():
            other = queue.get()
            item = max(item, other)
            insertion = item is not other
        else:
            insertion = True

        # update item if necessary
        if insertion:
            # move to CPU to conserve memory.
            e, p = item.value
            if detach:
                e = recursive_detach(e, to_cpu=True)
                p = recursive_detach(p, to_cpu=True)
            item.value = (e, p)

        # restore queue
        queue.put(item)
        return insertion

    @torch.no_grad()
    def dequeue_all(self, queue: PriorityQueue[PrioritizedItem]) -> Iterator[Tuple[I, O]]:
        r"""Dequeue and iterate through all items in a queue."""
        while not queue.empty():
            item = queue.get()
            assert isinstance(item, PrioritizedItem)
            example, pred = item.value
            yield example, pred
