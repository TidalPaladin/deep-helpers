#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
from deep_helpers.testing import checkpoint_factory


class TestCheckpointFactory:
    @pytest.mark.parametrize(
        "root,filename,set_env,exp",
        [
            (None, None, True, "Task.ckpt"),
            (None, None, False, "Task.ckpt"),
            ("myroot", None, True, "myroot/CustomTask.ckpt"),
            ("myroot", "checkpoint.pth", True, "myroot/checkpoint.pth"),
        ],
    )
    def test_create_checkpoint(self, tmp_path, task, root, filename, set_env, exp):
        if root is not None:
            root = tmp_path / root
        result = checkpoint_factory(task, root, filename, set_env)
        assert result.is_file()
        assert str(result).endswith(exp)

    def test_set_env_var(self, task):
        task.CHECKPOINT_ENV_VAR = "MY_ENV_VAR"
        result = checkpoint_factory(task, set_env=True)
        assert task.CHECKPOINT_ENV_VAR in os.environ
        assert os.environ[task.CHECKPOINT_ENV_VAR] == str(result)
