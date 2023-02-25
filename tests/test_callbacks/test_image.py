#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict

import pytest
import torch
from deep_helpers.callbacks import LoggerIntegration, QueuedLoggingCallback
from deep_helpers.structs import Mode, State
from pytorch_lightning.loggers.wandb import WandbLogger

from tests.test_callbacks.base_callback import BaseCallbackTest


class DummyIntegration(LoggerIntegration):
    logger_type = WandbLogger

    def __call__(
        self,
        target,
        pl_module,
        tag,
        step,
        *args,
        **kwargs,
    ):
        t = {tag: target}
        pl_module.logger.experiment.log(t, commit=False)


class ImageLoggingCallback(QueuedLoggingCallback):
    integrations = [DummyIntegration()]

    @classmethod
    def get_priority(cls, example, output):
        if "logits" in output and output["logits"] is not None:
            return 1 - output["logits"].sigmoid()
        return None

    def prepare_target(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        return batch["img"]


class TestImageLoggingCallback(BaseCallbackTest):
    def test_init(self):
        cb = ImageLoggingCallback("img", queue_size=16)

    @pytest.fixture
    def callback(self, modes):
        cb = ImageLoggingCallback("name", modes)
        return cb

    @pytest.mark.parametrize("queue_size", [4, 8])
    def test_enqueue(self, queue_size):
        eps = 1e-8
        example = dict(
            img=torch.rand(3, 32, 32),
            label=torch.tensor(1.0).view(1),
        )

        # build a dict of priority, prediction pairs
        total_size = 2 * queue_size
        priority = torch.rand(total_size)
        preds: Dict[float, dict] = {}
        for p in priority:
            logit = (1.0 - p).logit(eps=eps).view(1)
            pred = dict(logits=logit)
            preds[p.item()] = pred

        # build another dict of top-k priority pairs
        keep_keys = priority.topk(k=queue_size).values
        keep_preds = {k.item(): preds[k.item()] for k in keep_keys}

        cb = ImageLoggingCallback("img", queue_size=queue_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)
        for k, p in preds.items():
            cb.enqueue(example, p, queue=queue)

        assert queue.qsize() <= queue_size
        queued_priorities = []
        while not queue.empty():
            item = queue.get()
            e, p = item.value
            assert isinstance(e, dict)
            assert isinstance(p, dict)
            queued_priorities.append(item.priority)

        out = torch.tensor(queued_priorities).sort(descending=True).values
        assert torch.allclose(out, keep_keys)

    def test_enqueue_null_priority(self):
        example = dict(
            img=torch.rand(3, 32, 32),
            label=None,
        )
        logit = None
        pred = dict(logits=logit)

        cb = ImageLoggingCallback("img", queue_size=8)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)

        assert cb.get_priority(example, pred) is None
        cb.enqueue(example, pred, queue=queue)
        assert queue.empty()

    def test_dequeue(self):
        eps = 1e-8
        example = dict(
            img=torch.rand(3, 32, 32),
            label=torch.tensor(1.0).view(1),
        )

        total_size = 8
        cb = ImageLoggingCallback("img", queue_size=total_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)

        # build a dict of priority, prediction pairs
        priority = torch.rand(total_size)
        preds: Dict[float, dict] = {}
        for p in priority:
            logit = (1.0 - p).logit(eps=eps).view(1)
            pred = dict(logits=logit)
            preds[p] = pred
            cb.enqueue(example, pred, queue=queue)

        assert queue.qsize() == total_size
        dequeued = list(cb.dequeue_all(queue))

        assert queue.empty()
        assert len(dequeued) == total_size
        for e, p in dequeued:
            assert isinstance(e, dict)
            assert isinstance(p, dict)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "mode,should_log",
        [
            pytest.param(Mode.TRAIN, True),
            pytest.param(Mode.VAL, False),
            pytest.param(Mode.TEST, False),
        ],
    )
    def test_training_log(self, lightning_module, logger, mode, should_log):
        state = State(mode)
        lightning_module.state = state
        cb = ImageLoggingCallback("img", modes=[mode], queue_size=8)

        B = 4
        example = dict(img=torch.rand(B, 3, 32, 32), label=torch.randint(0, 1, (B, 1)))
        pred = dict(logits=torch.rand(B, 1))

        cb.on_train_batch_end(
            lightning_module.trainer,
            lightning_module,
            pred,
            example,
            0,
        )

        if should_log:
            logger.experiment.log.assert_called()
            assert logger.experiment.log.call_count == B
        else:
            logger.experiment.log.assert_not_called()

    @pytest.mark.parametrize("mode", [Mode.VAL, Mode.TEST])
    @pytest.mark.parametrize("queue_size", [4, 8])
    def test_queued_log(self, lightning_module, logger, mode, queue_size):
        state = State(mode)
        lightning_module.state = state
        cb = ImageLoggingCallback("img", queue_size=queue_size)

        B = 4
        for _ in range(3):
            example = dict(img=torch.rand(B, 3, 32, 32), label=torch.randint(0, 1, (B, 1)))
            pred = dict(logits=torch.rand(B, 1))
            cb.on_train_batch_end(
                lightning_module.trainer,
                lightning_module,
                pred,
                example,
                0,
            )

        logger.experiment.log.assert_not_called()
        assert len(cb) <= queue_size

        cb._on_epoch_end(lightning_module.trainer, lightning_module, mode)

        logger.experiment.log.assert_called()
        assert 0 < logger.experiment.log.call_count <= queue_size

    def test_cpu_detach_on_enqueue(self, cuda):
        queue_size = 32
        device = "cuda:0" if cuda else "cpu"

        example = dict(
            img=torch.rand(3, 32, 32, device=device),
            label=torch.tensor(1.0, device=device).view(1),
        )
        pred = dict(logits=torch.rand(1, requires_grad=True, device=device))

        cb = ImageLoggingCallback("img", queue_size=queue_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)
        cb.enqueue(example, pred, queue=queue)

        while not queue.empty():
            item = queue.get()
            e, p = item.value
            assert all(t.device == torch.device("cpu") for t in e.values())
            assert all(t.device == torch.device("cpu") for t in p.values())
            assert all(not t.requires_grad for t in e.values())
            assert all(not t.requires_grad for t in p.values())
