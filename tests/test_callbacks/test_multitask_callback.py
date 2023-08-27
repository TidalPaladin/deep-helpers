#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, cast

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from deep_helpers.callbacks import MultiTaskCallbackWrapper
from deep_helpers.tasks import MultiTask


class TestMultiTaskCallbackWrapper:
    @pytest.fixture
    def multitask(self, logger):
        optimizer_init = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.001}}
        task = MultiTask(tasks=["custom-task", "custom-task2"], optimizer_init=optimizer_init)
        trainer = pl.Trainer(logger=logger)
        task.trainer = trainer
        task.setup("fit")
        assert len(task) == 2
        # NOTE: yield here to weakref to logger being gc'ed
        yield task

    @pytest.mark.parametrize(
        "key,batch_idx,exp",
        [
            ("custom-task", 0, True),
            ("custom-task", 1, False),
            ("custom-task2", 0, False),
            ("custom-task2", 1, True),
        ],
    )
    def test_should_run_on_task(self, mocker, multitask, batch_idx, key, exp):
        callback = mocker.Mock(spec_set=Callback)
        wrapper = MultiTaskCallbackWrapper(callback, {key})
        assert wrapper.should_run_on_task(multitask, batch_idx) == exp

    @pytest.mark.parametrize(
        "key,batch_idx,exp",
        [
            ("custom-task", 0, True),
            ("custom-task", 1, False),
            ("custom-task2", 0, False),
            ("custom-task2", 1, True),
        ],
    )
    def test_on_train_batch_start(self, mocker, multitask, batch_idx, key, exp):
        callback = mocker.MagicMock(spec_set=Callback, name="callback")
        wrapper = MultiTaskCallbackWrapper(callback, {key})
        batch = mocker.MagicMock(name="batch")
        cast(Any, wrapper).on_train_batch_start(
            multitask.trainer,
            multitask,
            batch,
            batch_idx,
        )
        wrapped_hook = callback.on_train_batch_start
        if exp:
            assert wrapper.should_run_on_task(multitask, batch_idx)
            wrapped_hook.assert_called_once_with(
                multitask.get_current_task(batch_idx),
                multitask.trainer,
                multitask,
                batch,
                batch_idx,
            )
        else:
            wrapped_hook.assert_not_called()

    @pytest.mark.parametrize(
        "key,batch_idx,exp",
        [
            ("custom-task", 0, True),
            ("custom-task", 1, False),
            ("custom-task2", 0, False),
            ("custom-task2", 1, True),
        ],
    )
    def test_on_train_batch_end(self, mocker, multitask, batch_idx, key, exp):
        callback = mocker.MagicMock(spec_set=Callback, name="callback")
        wrapper = MultiTaskCallbackWrapper(callback, {key})
        batch = mocker.MagicMock(name="batch")
        output = mocker.MagicMock(name="output")
        cast(Any, wrapper).on_train_batch_end(
            multitask.trainer,
            multitask,
            output,
            batch,
            batch_idx,
        )
        wrapped_hook = callback.on_train_batch_end
        if exp:
            assert wrapper.should_run_on_task(multitask, batch_idx)
            wrapped_hook.assert_called_once_with(
                multitask.get_current_task(batch_idx),
                multitask.trainer,
                multitask,
                output,
                batch,
                batch_idx,
            )
        else:
            wrapped_hook.assert_not_called()
