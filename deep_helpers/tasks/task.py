#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, Generic, Iterator, Optional, Set, Type, TypedDict, TypeVar, Union, cast
from safetensors import safe_open

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from registry import Registry
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchmetrics import MetricCollection

# from ..callbacks.wandb import WandBCheckpointCallback
from ..data import SupportsDatasetNames
from ..helpers import load_checkpoint
from ..structs import MetricStateCollection, Mode, State


TASKS = Registry("tasks")


class Output(TypedDict):
    log: Dict[str, Any]
    loss: Tensor


# Input and output types
I = TypeVar("I", bound=Union[Dict[str, Any], TypedDict])
O = TypeVar("O", bound=Union[Dict[str, Any], Output])


class CustomOptimizerMixin(ABC):
    r"""Provides support for custom optimizers at the CLI"""
    optimizer_init: Dict[str, Any]
    lr_scheduler_init: Dict[str, Any]
    lr_scheduler_interval: str
    lr_scheduler_monitor: str
    weight_decay_exemptions: Set[str] = set()

    def _params_weight_decay(self) -> Iterator[nn.Parameter]:
        return (
            param
            for module in cast(nn.Module, self).modules()
            for param_name, param in module.named_parameters()
            if self._needs_weight_decay(module, param_name)
        )

    def _params_no_weight_decay(self) -> Iterator[nn.Parameter]:
        return (
            param
            for module in cast(nn.Module, self).modules()
            for param_name, param in module.named_parameters()
            if not self._needs_weight_decay(module, param_name)
        )

    def _needs_weight_decay(self, module: nn.Module, param_name: str) -> bool:
        return not (
            module.__class__.__name__ in self.weight_decay_exemptions
            or any(part in self.weight_decay_exemptions for part in param_name.split("."))
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        if not self.optimizer_init:
            raise ValueError("No optimizer specified")
        result: Dict[str, Any] = {}
        weight_decay = self.optimizer_init.get("init_args", {}).get("weight_decay", 0.0)
        params = set(cast(nn.Module, self).parameters())
        params_no_weight_decay = set(self._params_no_weight_decay())
        params = params - params_no_weight_decay
        optimizer = instantiate_class(
            [
                {"params": list(params), "weight_decay": weight_decay},
                {"params": list(params_no_weight_decay), "weight_decay": 0.0},
            ],
            self.optimizer_init,
        )
        optimizer.param_groups = [g for g in optimizer.param_groups if g["params"]]
        result["optimizer"] = optimizer

        if self.lr_scheduler_init:
            lr_scheduler: Dict[str, Any] = {}
            scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
            lr_scheduler["scheduler"] = scheduler
            lr_scheduler["monitor"] = self.lr_scheduler_monitor
            lr_scheduler["interval"] = self.lr_scheduler_interval
            result["lr_scheduler"] = lr_scheduler

        return result


class StateMixin:
    state: State
    trainer: Optional[pl.Trainer]
    named_datasets: bool

    def on_train_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.TRAIN, None)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.VAL, None)

    def on_test_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.TEST, None)

    def on_train_batch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.TRAIN, None)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx: int = 0):
        dataset_name = self.get_dataset_name(Mode.VAL, dataloader_idx) if self.named_datasets else None
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.VAL)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx: int = 0):
        dataset_name = self.get_dataset_name(Mode.TEST, dataloader_idx) if self.named_datasets else None
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.TEST)

    def get_dataset_name(self, mode: Mode, dataloader_idx: int) -> Optional[str]:
        names = list(self.dataset_names(mode))
        return names[dataloader_idx] if names else None

    def dataset_names(self, mode: Mode) -> Iterator[str]:
        if self.trainer is None:
            raise AttributeError("Trainer not initialized")
        dm: Optional[pl.LightningDataModule] = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return
        elif isinstance(dm, SupportsDatasetNames):
            for name in cast(SupportsDatasetNames, dm).dataset_names.names_for_mode(mode):
                yield name
        elif hasattr(dm, "name") and dm.name:
            yield dm.name
        else:
            yield dm.__class__.__name__


T = TypeVar("T", bound="Task")


class Task(CustomOptimizerMixin, StateMixin, pl.LightningModule, Generic[I, O], ABC):
    CHECKPOINT_ENV_VAR: ClassVar[Optional[str]] = None

    state: State
    logger: LightningLoggerBase
    trainer: pl.Trainer

    def __init__(
        self,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        weight_decay_exemptions: Set[str] = set(),
    ):
        super(Task, self).__init__()
        self.state = State()
        self.metrics = MetricStateCollection()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.lr_scheduler_interval = lr_interval
        self.lr_scheduler_monitor = lr_monitor
        self.named_datasets = named_datasets
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.strict_checkpoint = strict_checkpoint
        self.log_train_metrics_interval = log_train_metrics_interval
        self.log_train_metrics_on_epoch = log_train_metrics_on_epoch
        self.weight_decay_exemptions = set(weight_decay_exemptions)

    @abstractmethod
    def create_metrics(self, state: State) -> MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> O:
        raise NotImplementedError  # pragma: no cover

    @property
    def attached_task(self) -> "Task":
        r"""Returns the task attached to the current :class:`Trainer`. This is useful when multiple tasks are nested."""
        return self.trainer.lightning_module

    def run_logging_loop(
        self,
        state: State,
        output: O,
        metrics: Optional[tm.MetricCollection] = None,
        add_dataloader_idx: bool = False,
    ) -> None:
        # Manually log loss, PyTorch Lightning 1.9 doesn't do this for us anymore
        self.attached_task.log(
            "loss", cast(Dict[str, Any], output)["loss"], on_step=True, on_epoch=False, prog_bar=True
        )

        # Log things placed into `output` under the key `log` unless they are metrics.
        # Metrics require special handling and will be logged separately.
        scalars_to_log = {
            f"{state.with_postfix(k)}": v for k, v in output.get("log", {}).items() if not isinstance(v, tm.Metric)
        }
        self.attached_task.log_dict(
            scalars_to_log,
            on_step=self.attached_task.trainer.training,
            on_epoch=not self.attached_task.trainer.training,
            prog_bar=False,
            sync_dist=not self.attached_task.trainer.training,
            add_dataloader_idx=add_dataloader_idx,
        )

        # Log metrics
        if self.training and metrics and self.global_step % self.attached_task.log_train_metrics_interval == 0:
            self._log_train_metrics(state, metrics, add_dataloader_idx=add_dataloader_idx)

    def _log_inference_metrics(
        self,
        state: State,
        metrics: tm.MetricCollection,
        **kwargs,
    ) -> None:
        assert state.mode != Mode.TRAIN
        metrics_to_log = (
            cast(
                Dict[str, Tensor],
                {f"{state.with_postfix(k)}": cast(tm.Metric, v).compute() for k, v in metrics.items()},
            )
            if metrics
            else None
        )

        if metrics_to_log:
            self.attached_task.log_dict(
                metrics_to_log,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                **kwargs,
            )

    def _log_train_metrics(
        self,
        state: State,
        metrics: tm.MetricCollection,
        **kwargs,
    ) -> None:
        assert state.mode == Mode.TRAIN
        metrics_to_log = (
            cast(
                Dict[str, Tensor],
                {f"{state.with_postfix(k)}": cast(tm.Metric, v).compute() for k, v in metrics.items()},
            )
            if metrics
            else None
        )

        if metrics_to_log:
            on_step = not self.attached_task.log_train_metrics_on_epoch
            on_epoch = self.attached_task.log_train_metrics_on_epoch
            self.attached_task.log_dict(
                metrics_to_log, on_step=on_step, on_epoch=on_epoch, prog_bar=False, sync_dist=False, **kwargs
            )
            # If logging on step, manually reset the metrics after computing them.
            if on_step:
                for m in metrics.values():
                    if isinstance(m, tm.Metric):
                        m.reset()

    def setup(self, *args, **kwargs):
        if self.checkpoint is not None:
            self._safe_load_checkpoint()

    def _safe_load_checkpoint(self) -> None:
        assert self.checkpoint is not None
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")  # pragma: no cover
        rank_zero_info(f"Loading checkpoint (strict={self.strict_checkpoint}): {checkpoint_path}")

        if checkpoint_path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device=self.device.index) as f:
                # Handle case where "state_dict" is a key in the file
                if isinstance(f, dict) and "state_dict" in f.keys():
                    f = f["state_dict"]
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        load_checkpoint(self, state_dict, strict=self.strict_checkpoint)

    def compute_total_loss(self, output: O) -> Tensor:
        loss = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        return loss

    def training_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TRAIN
        metrics = self.metrics.get(self.state)
        output = self.step(batch, batch_idx, self.state, metrics)
        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, output, metrics)
        return output

    def validation_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.VAL
        metrics = self.metrics.get(self.state)

        # Fork RNG to avoid validation data affecting training data and to ensure that
        # the step is deterministic.
        with torch.random.fork_rng(devices=[self.device] if self.device != torch.device("cpu") else None):
            torch.random.manual_seed(42)
            output = self.step(batch, batch_idx, self.state, metrics)

        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, output, metrics)
        return output

    def test_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TEST
        metrics = self.metrics.get(self.state)

        # Fork RNG to ensure that the step is deterministic.
        with torch.random.fork_rng(devices=[self.device] if self.device != torch.device("cpu") else None):
            torch.random.manual_seed(42)
            output = self.step(batch, batch_idx, self.state, metrics)

        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, output, metrics)
        return output

    def on_fit_start(self):
        r"""Initialize validation/training metrics"""
        # Create metrics for training and validation states
        # TODO: should we unregister these states in on_fit_end?
        for mode in (Mode.TRAIN, Mode.VAL):
            dataset_names = self.dataset_names(mode) if self.named_datasets else [None]
            for name in dataset_names:
                state = State(mode, name if mode == Mode.VAL else None)
                metrics = self.create_metrics(state).to(self.device)
                self.metrics.set_state(state, metrics)

    def on_test_start(self):
        r"""Initialize testing metrics"""
        # Create metrics for test state
        dataset_names = self.dataset_names(Mode.TEST) if self.named_datasets else [None]
        for name in dataset_names:
            state = State(Mode.TEST, name)
            metrics = self.create_metrics(state).to(self.device)
            self.metrics.set_state(state, metrics)

    def on_validation_epoch_end(self, *args, **kwargs):
        state = State(Mode.VAL)
        metrics = self.metrics.get(state)
        if isinstance(metrics, tm.MetricCollection):
            self._log_inference_metrics(state, metrics)

    def on_test_epoch_end(self, *args, **kwargs):
        state = State(Mode.TEST)
        metrics = self.metrics.get(state)
        if isinstance(metrics, tm.MetricCollection):
            self._log_inference_metrics(state, metrics)

    def on_train_epoch_start(self, *args, **kwargs):
        state = State(Mode.TRAIN)
        metrics = self.metrics.get(state)
        if isinstance(metrics, tm.MetricCollection):
            metrics.reset()

    def on_validation_epoch_start(self, *args, **kwargs):
        state = State(Mode.VAL)
        metrics = self.metrics.get(state)
        if isinstance(metrics, tm.MetricCollection):
            metrics.reset()

    def on_test_epoch_start(self, *args, **kwargs):
        state = State(Mode.TEST)
        metrics = self.metrics.get(state)
        if isinstance(metrics, tm.MetricCollection):
            metrics.reset()

    @classmethod
    def _get_checkpoint_path(cls, path: Optional[Path]) -> Path:
        # use a specified path if one was given
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
            return path

        # otherwise, try to find the checkpoint using environment variables
        elif cls.CHECKPOINT_ENV_VAR:
            if cls.CHECKPOINT_ENV_VAR in os.environ:
                path = Path(os.environ[cls.CHECKPOINT_ENV_VAR])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint file {path} specified by {cls.CHECKPOINT_ENV_VAR} does not exist."
                    )
                return path
            else:
                raise ValueError(
                    "Checkpoint path not specified. "
                    f"Either set the environment variable {cls.CHECKPOINT_ENV_VAR} or pass the --checkpoint argument."
                )

        # model doesn't have a default env var and no path was provided
        else:
            raise ValueError(
                f"{cls.__name__} does not support a checkpoint env variable. Please specify a checkpoint path."
            )

    @classmethod
    def load_from_checkpoint(cls: Type[T], checkpoint: Optional[Path], strict: bool = True, **kwargs) -> T:
        checkpoint = cls._get_checkpoint_path(checkpoint)
        metadata = torch.load(checkpoint, map_location="cpu")
        hparams = metadata["hyper_parameters"]
        hparams.pop("checkpoint")
        hparams.update(kwargs)
        model = cls(**hparams)
        model.load_state_dict(metadata["state_dict"], strict=strict)
        model.eval()
        return cast(T, model)

    @classmethod
    def create(cls, checkpoint_path: Optional[Path] = None, **kwargs):
        result = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        result.eval()
        logging.info(f"Loaded {cls.__name__} checkpoint from {checkpoint_path}")
        return cast(T, result)
