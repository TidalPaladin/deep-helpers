#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
from abc import ABC, abstractmethod
from inspect import signature
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import yaml
from jsonargparse import ArgumentParser, Namespace
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
from ..safetensors import iterate_safetensors_state_dict
from ..structs import MetricStateCollection, Mode, State


TASKS = Registry("tasks")


class Output(TypedDict):
    log: Dict[str, Any]
    loss: Tensor


# Input and output types
I = TypeVar("I", bound=Union[Dict[str, Any], TypedDict])
O = TypeVar("O", bound=Union[Dict[str, Any], Output])


def match_parameters(
    module: nn.Module,
    keys: Tuple[str, ...],
    prefix: str = "",
) -> Iterator[nn.Parameter]:
    # Check direct parameters of the current module
    yield from (
        param
        for param_name, param in module.named_parameters(recurse=False, prefix=prefix)
        if (
            # Module class name matches a key exactly
            module.__class__.__name__ in keys
            # Or any part of the parameter name matches a key
            or any(part in keys for part in param_name.split("."))
        )
    )

    # Recurse into submodules
    for name, submodule in module.named_children():
        yield from match_parameters(submodule, keys, prefix=f"{prefix}.{name}")


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


class Task(StateMixin, pl.LightningModule, Generic[I, O], ABC):
    """
    Defines a task for training, validation, and testing with PyTorch Lightning.

    Args:
        optimizer_init: Configuration for the optimizer initialization.
        lr_scheduler_init: Configuration for the learning rate scheduler initialization.
        lr_interval: Interval for updating the learning rate scheduler ('step' or 'epoch').
        lr_monitor: Metric name to monitor for learning rate scheduler updates.
        named_datasets: Flag indicating whether datasets are named.
        checkpoint: Path to a checkpoint file for loading pre-trained weights.
        strict_checkpoint: Flag indicating whether to strictly enforce matching for checkpoint loading.
        log_train_metrics_interval: Interval for logging training metrics.
        log_train_metrics_on_epoch: Flag indicating whether to log training metrics on epoch end.
        parameter_groups: Custom parameter groups for the optimizer. Each list element is a dict with a 'params' key
            whose value is a sequence of parameter names or module class names to match. All other keys are passed
            to the optimizer constructor as keyword arguments.
    """

    # TODO: For now we will retain support for CHECKPOINT_ENV_VAR. However jsonargparse provides a CLI mechanism for
    # env var reading. We should consider using that instead.
    CHECKPOINT_ENV_VAR: ClassVar[str] = "CHECKPOINT"
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
        checkpoint: str | os.PathLike | None = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        parameter_groups: List[Dict[str, Any]] = [],
    ):
        super(Task, self).__init__()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.lr_scheduler_interval = lr_interval
        self.lr_scheduler_monitor = lr_monitor
        self.named_datasets = named_datasets
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.strict_checkpoint = strict_checkpoint
        self.log_train_metrics_interval = log_train_metrics_interval
        self.log_train_metrics_on_epoch = log_train_metrics_on_epoch
        self.parameter_groups = parameter_groups

    @torch.jit.unused
    def _torchscript_unsafe_init(self, *args, **kwargs) -> None:
        r"""Runs init methods that cannot be scripted by TorchScript"""
        self.state = State()
        self.metrics = MetricStateCollection()

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
    @torch.jit.unused
    def attached_task(self):
        r"""Returns the task attached to the current :class:`Trainer`. This is useful when multiple tasks are nested."""
        return self.trainer.lightning_module

    def run_logging_loop(
        self,
        state: State,
        output: O,
        batch_idx: int,
        metrics: Optional[tm.MetricCollection] = None,
        add_dataloader_idx: bool = False,
    ) -> None:
        is_logging_global_step = (self.global_step + 1) % self.attached_task.log_train_metrics_interval == 0
        # NOTE: When using accumulate_grad_batches, global step only increments every `accumulate_grad_batches` steps.
        # If this is not accounted for, we will compute log and reset a metric `accumulate_grad_batches` times per global step.
        # This will clear the metric state that may have been updated over `log_train_metrics_interval` batches.
        is_last_batch_in_accumulation = (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0
        if self.training and metrics and is_logging_global_step and is_last_batch_in_accumulation:
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
                {
                    f"{state.with_postfix(k)}": cast(tm.Metric, v).compute()
                    for k, v in metrics.items()
                    if cast(tm.Metric, v).update_called
                },
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
                {
                    f"{state.with_postfix(k)}": cast(tm.Metric, v).compute()
                    for k, v in metrics.items()
                    if cast(tm.Metric, v).update_called
                },
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
        self._torchscript_unsafe_init()
        if self.checkpoint is not None:
            self._safe_load_checkpoint()

    def _safe_load_checkpoint(self) -> Dict[str, Any]:
        assert self.checkpoint is not None
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")  # pragma: no cover
        rank_zero_info(f"Loading checkpoint (strict={self.strict_checkpoint}): {checkpoint_path}")

        if checkpoint_path.suffix == ".safetensors":
            state_dict = {k: v for k, v in iterate_safetensors_state_dict(checkpoint_path, device=self.device)}
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
        load_checkpoint(self, state_dict, strict=self.strict_checkpoint)
        self.on_task_checkpoint_loaded(checkpoint_path, state_dict)
        return state_dict

    def on_task_checkpoint_loaded(self, path, state_dict: Dict[str, Any]) -> None:
        r"""Called when a checkpoint is loaded as part of :func:`setup`"""

    def compute_total_loss(self, output: O) -> Tensor:
        loss = cast(Tensor, sum(v for k, v in output["log"].items() if k.startswith("loss_")))
        return loss

    def training_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TRAIN
        metrics = self.metrics.get(self.state)
        output = self.step(batch, batch_idx, self.state, metrics)
        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        self.run_logging_loop(self.state, output, batch_idx, metrics)
        return output

    def validation_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.VAL
        metrics = self.metrics.get(self.state)

        # Fork RNG to avoid validation data affecting training data and to ensure that
        # the step is deterministic.
        with torch.random.fork_rng(devices=[self.device] if self.device != torch.device("cpu") else []):
            torch.random.manual_seed(42)
            output = self.step(batch, batch_idx, self.state, metrics)

        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        self.run_logging_loop(self.state, output, batch_idx, metrics)
        return output

    def test_step(self, batch: I, batch_idx: int, *args, **kwargs) -> O:
        assert self.state.mode == Mode.TEST
        metrics = self.metrics.get(self.state)

        # Fork RNG to ensure that the step is deterministic.
        with torch.random.fork_rng(devices=[self.device] if self.device != torch.device("cpu") else []):
            torch.random.manual_seed(42)
            output = self.step(batch, batch_idx, self.state, metrics)

        _ = self.compute_total_loss(output)
        output["loss"] = self.compute_total_loss(output)
        self.run_logging_loop(self.state, output, batch_idx, metrics)
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
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = metadata["hyper_parameters"]
        hparams.pop("checkpoint")

        # Lightning seems to inject additional keys prefixed by "_"
        hparams = {k: v for k, v in hparams.items() if not k.startswith("_")}

        hparams.update(kwargs)
        model = cls(**hparams)
        model.load_state_dict(metadata["state_dict"], strict=strict)
        model.eval()
        return cast(T, model)

    @classmethod
    def create(cls: Type[T], checkpoint_path: Optional[Path] = None, **kwargs) -> T:
        result = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        result.eval()
        logging.info(f"Loaded {cls.__name__} checkpoint from {checkpoint_path}")
        return cast(T, result)

    @classmethod
    def add_args_to_parser(
        cls,
        parser: ArgumentParser,
        key: str = "model",
        skip: Set[str] = set(),
        subclass: bool = True,
        **kwargs,
    ) -> ArgumentParser:
        """
        Adds arguments to the parser for the model.

        This method modifies the provided ArgumentParser object by adding arguments related to the model. It automatically
        skips arguments that are not relevant for inference, such as training-specific parameters. It also handles the
        addition of subclass arguments if the subclass flag is set to True.

        Args:
            parser: The ArgumentParser object to which the arguments will be added.
            key: A string representing the key under which the model's arguments will be nested.
            skip: A set of strings representing the names of arguments to be skipped.
            subclass: A boolean indicating whether to support subclasses. When True, there must be some specification provided
                as to which subclass to use.

        Keyword Args:
            Forwarded to the `add_class_arguments` or `add_subclass_arguments` method.

        Returns:
            The modified ArgumentParser object with the model's arguments added.
        """
        # Since this will be for inference we assume that training related arguments are not needed.
        skip_args = {
            param.name
            for param in signature(Task).parameters.values()
            if param.name not in {"self", "checkpoint", "strict_checkpoint"}
        }.union(skip)

        parser.add_argument(
            "checkpoint",
            type=os.PathLike,
            help="Path to the checkpoint. Can be a lightning or safetensors file.",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            help="Device to use for inference.",
        )

        if subclass:
            parser.add_subclass_arguments(cls, key, skip=cast(Any, skip_args), **kwargs)
            parser.link_arguments("checkpoint", "model.init_args.checkpoint")
        else:
            parser.add_class_arguments(cls, key, skip=cast(Any, skip_args), **kwargs)
            parser.link_arguments("checkpoint", "model.checkpoint")

        return parser

    @classmethod
    def on_after_parse(cls, cfg: Namespace, key: str = "model") -> Namespace:
        """
        Processes the configuration after parsing to ensure the model is correctly instantiated.

        The main function of this method is to eliminate the need for the user to manually specify the model configuration
        in addition to a checkpoint path. It handles model instantiation based on the provided checkpoint as follows:
            - If a model config was provided and instantiated, it is used as is.
            - If a safetensors checkpoint was provided, it attempts to find "config.yaml" or "config.json" in the same
                directory as the checkpoint and uses it to instantiate the model. It is expected that the config file will
                specify the class path and initialization arguments for the model as needed
            - If a lightning checkpoint was provided, it instantiates the calling class directly from hyperparameters
                in the checkpoint. Note that this approach cannot infer the desired class type from the checkpoint.

        Args:
            cfg: The Namespace object containing the parsed arguments.
            key: A string representing the key under which the model's configuration is stored.

        Returns:
            The Namespace object with the model instantiated and added to it.
        """
        # If a model config was not provided try to derive one
        if not isinstance(cfg.get(key, None), cls):
            src = Path(cfg.checkpoint)

            # For SafeTensors checkpoints we will look for an associated config file
            if src.suffix == ".safetensors":
                for loader, suffix in (yaml.safe_load, ".yaml"), (json.load, ".json"):
                    target = (src.parent / "config").with_suffix(suffix)
                    if target.is_file():
                        with open(target, "r") as f:
                            model_cfg = loader(f)
                            # NOTE: Must explicitly instantiate with tuple()
                            cfg.model = instantiate_class(tuple(), init=model_cfg)
                            cfg.model.checkpoint = src
                            break
                else:
                    raise FileNotFoundError(f"Config file not found for {src}")

            # For lightning checkpoints we will try to load the model from the checkpoint
            # NOTE: This only works if desired class to be loaded is `cls`
            # TODO: This will double-load checkpoints. Once in the initial call, and again in setup().
            # I don't know if we want to continue to support lightning checkpoints from the CLI.
            else:
                cfg.model = cls.load_from_checkpoint(src)
                cfg.model.checkpoint = src

        return cfg

    def configure_optimizers(self) -> Dict[str, Any]:
        if not self.optimizer_init:
            raise ValueError("No optimizer specified")
        result: Dict[str, Any] = {}

        # Determine parameter groups
        parameter_groups: List[Dict[str, Any]] = []
        assigned_params: Set[nn.Parameter] = set()
        for config in self.parameter_groups:
            keys = config["params"]
            params = set(p for p in match_parameters(self, keys) if p.requires_grad)
            params = params.difference(assigned_params)
            if params:
                assert params.isdisjoint(assigned_params)
                kwargs = {k: v for k, v in config.items() if k != "params"}
                parameter_groups.append({"params": list(params), **kwargs})
            assigned_params.update(params)
        # Default param group
        parameter_groups.append(
            {"params": [p for p in self.parameters() if p not in assigned_params and p.requires_grad]}
        )

        # Instantiate optimizer
        optimizer = instantiate_class(parameter_groups, self.optimizer_init)
        result["optimizer"] = optimizer

        # Instantiate learning rate scheduler
        if self.lr_scheduler_init:
            lr_scheduler: Dict[str, Any] = {}
            scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
            lr_scheduler["scheduler"] = scheduler
            lr_scheduler["monitor"] = self.lr_scheduler_monitor
            lr_scheduler["interval"] = self.lr_scheduler_interval
            result["lr_scheduler"] = lr_scheduler

        return result
