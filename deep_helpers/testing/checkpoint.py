#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytorch_lightning as pl
import torch
from deep_helpers.tasks import Task
from pytorch_lightning.callbacks import ModelCheckpoint


def checkpoint_factory(
    task: Task,
    root: Optional[Path] = None,
    filename: Optional[str] = None,
    set_env: bool = True,
) -> Path:
    r"""Create a dummy checkpoint for a task. Useful for testing.

    Args:
        task: Task.
        root: Root directory. Defaults to a temporary directory.
        filename: Checkpoint filename. Defaults to None.
        set_env: Whether to set the environment variable for the checkpoint directory.

    Returns:
        Path to the checkpoint.
    """
    if not isinstance(task, Task):
        raise TypeError(f"task must be a Task, not {type(task)}")

    torch.random.manual_seed(42)
    filename = filename or f"{task.__class__.__name__}.ckpt"
    root = root or Path(TemporaryDirectory().name)

    path = Path(root, filename)
    if path.is_file():
        return path

    cb = ModelCheckpoint(path.parent, path.name)
    trainer = pl.Trainer(
        default_root_dir=root, callbacks=[cb], max_steps=1, limit_val_batches=1, logger=[], num_sanity_val_steps=0
    )
    # We need to call fit to initialize the model and save a checkpoint, but it is missing data which is an error.
    # Silently ignore the error, but ensure the checkpoint was created before returning.
    try:
        trainer.fit(task)
    except Exception as e:
        if trainer.model is None:
            raise AttributeError("Model was not attached to the trainer.") from e
    assert trainer.model is not None
    trainer.save_checkpoint(path)
    assert path.is_file(), "fixture failed to create checkpoint file for testing"

    # Set the environment variable for the checkpoint directory.
    if set_env:
        varname = task.CHECKPOINT_ENV_VAR
        if varname is not None:
            os.environ[varname] = str(path)

    return path
