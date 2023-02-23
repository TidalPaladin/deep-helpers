#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Generic, Iterator, Optional, TypedDict, TypeVar, Union, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchmetrics import MetricCollection

# from ..callbacks.wandb import WandBCheckpointCallback
from ..data import NamedDataModuleMixin
from ..structs import MetricStateCollection, Mode, State


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

    parameters: Callable[..., Iterator[nn.Parameter]]

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.optimizer_init
        result: Dict[str, Any] = {}
        optimizer = instantiate_class(self.parameters(), self.optimizer_init)
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

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx: int):
        dataset_name = self.get_dataset_name(Mode.VAL, dataloader_idx) if self.named_datasets else None
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.VAL)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx: int):
        dataset_name = self.get_dataset_name(Mode.TEST, dataloader_idx) if self.named_datasets else None
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.TEST)

    def get_dataset_name(self, mode: Mode, dataloader_idx: Optional[int] = None) -> Optional[str]:
        names = list(self.dataset_names(mode))
        if dataloader_idx is None:
            return names[0] if names else None
        else:
            return names[dataloader_idx]

    def dataset_names(self, mode: Mode) -> Iterator[str]:
        if self.trainer is None:
            raise AttributeError("Trainer not initialized")
        dm: Optional[pl.LightningDataModule] = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return
        elif isinstance(dm, NamedDataModuleMixin):
            for name in cast(NamedDataModuleMixin, dm).names_for_mode(mode):
                yield name
        elif hasattr(dm, "name") and dm.name:
            yield dm.name
        else:
            yield dm.__class__.__name__


class WandBMixin:
    global_step: int
    logger: LightningLoggerBase
    trainer: pl.Trainer

    def on_train_batch_end(self, *args, **kwargs):
        self.commit_logs(step=self.global_step)

    @rank_zero_only
    def commit_logs(self, step: Optional[int] = None) -> None:
        if isinstance(self.logger, WandbLogger):
            assert self.global_step >= self.logger.experiment.step

            # final log call with commit=True to flush results
            self.logger.experiment.log.log({}, commit=True, step=self.global_step)
        # ensure all pyplot plots are closed
        plt.close()

    @rank_zero_only
    def wrapped_log(self, items: Dict[str, Any], **kwargs):
        target = {"trainer/global_step": self.trainer.global_step}
        target.update(items)
        self.logger.experiment.log(target, commit=False, **kwargs)

    def setup(self, *args, **kwargs):
        if isinstance(self.logger, WandbLogger):
            self.patch_logger(self.logger)

    def patch_logger(self, logger: WandbLogger) -> WandbLogger:
        r""":class:`WandbLogger` doesn't expect :func:`log` to be called more than a few times per second.
        Additionally, each call to :func:`log` will increment the logger step counter, which results
        in the logged step value being out of sync with ``self.global_step``. This method patches
        a :class:`WandbLogger` to log using ``self.global-step`` and never commit logs. Logs must be commited
        manually (already implemented in :func:`on_train_batch_end`).
        """
        # TODO provide a way to unpatch the logger (probably needed for test/inference)
        log = logger.experiment.log

        def wrapped_log(*args, **kwargs):
            assert self.global_step >= self.logger.experiment.step
            f = partial(log, commit=False)
            kwargs.pop("commit", None)
            return f(*args, **kwargs)

        # attach the original log method as an attribute of the wrapper
        # this allows commiting logs with logger.experiment.log.log(..., commit=True)
        wrapped_log.log = log

        # apply the patch
        logger.experiment.log = wrapped_log
        return logger


class Task(CustomOptimizerMixin, StateMixin, pl.LightningModule, Generic[I, O], ABC):
    CHECKPOINT_ENV_VAR: ClassVar[Optional[str]] = None

    state: State
    logger: LightningLoggerBase
    trainer: pl.Trainer

    def __init__(
        self,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
    ):
        super().__init__()
        self.state = State()
        self.metrics = MetricStateCollection()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.named_datasets = named_datasets

    @abstractmethod
    def create_metrics(self, state: State) -> MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> O:
        raise NotImplementedError

    @classmethod
    def run_logging_loop(
        cls,
        state: State,
        source: "Task",
        output: O,
        metrics: Optional[tm.MetricCollection] = None,
        add_dataloader_idx: bool = False,
    ) -> None:
        # Log things placed into `output` under the key `log` unless they are metrics.
        # Metrics require special handling and will be logged separately.
        scalars_to_log = {
            f"{str(source.state)}/{k}": v for k, v in output.get("log", {}).items() if not isinstance(v, tm.Metric)
        }
        source.log_dict(
            scalars_to_log,
            on_step=source.trainer.training,
            on_epoch=not source.trainer.training,
            prog_bar=False,
            sync_dist=not source.trainer.training,
            add_dataloader_idx=add_dataloader_idx,
        )

        # TODO: it seems necessary to manually call m.compute() for stepwise train time metrics.
        # Is there a better way to deal with this? Also, can metrics be accumulated over N training steps?
        if source.training and metrics:
            source.log_dict(
                cast(dict, metrics),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                add_dataloader_idx=add_dataloader_idx,
            )
            for m in metrics.values():
                if isinstance(m, tm.Metric):
                    m.reset()
        elif metrics:
            source.log_dict(
                cast(dict, metrics),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=add_dataloader_idx,
            )

    def compute_total_loss(self, output: O) -> Tensor:
        loss = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        return loss

    def training_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TRAIN
        metrics = self.metrics.get(self.state)
        output = self.step(batch, batch_idx, self.state, metrics)
        _ = self.compute_total_loss(output)
        output["loss"] = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, self, output, metrics)
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
        output["loss"] = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, self, output, metrics)
        return output

    def test_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TEST
        metrics = self.metrics.get(self.state)

        # Fork RNG to ensure that the step is deterministic.
        with torch.random.fork_rng(devices=[self.device] if self.device != torch.device("cpu") else None):
            torch.random.manual_seed(42)
            output = self.step(batch, batch_idx, self.state, metrics)

        _ = self.compute_total_loss(output)
        output["loss"] = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        output["log"]["loss"] = output["loss"]
        self.run_logging_loop(self.state, self, output, metrics)
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
