#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, Iterable, Literal, Set, Sized, Tuple, TypeVar, Union, cast, overload

import torch
import torch.nn as nn
from lightning_utilities.core.rank_zero import rank_zero_info


def load_checkpoint(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True) -> nn.Module:
    """Load weights from a state dict. When loading with ``strict=False``, missing or unexpected
    keys will be ignored and keys with shapes that do not match will be ignored. The rank zero
    process will log information about what layers from `model` were loaded from `state_dict`.

    Args:
        model: The model to load weights into.
        state_dict: The state dict to load weights from.
        strict: Whether to load weights strictly or not.

    Returns:
        The model with loaded weights.
    """
    if strict:
        model.load_state_dict(state_dict)
    else:
        model_state_dict = model.state_dict()
        checkpoint_state_dict = {
            k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        model.load_state_dict(checkpoint_state_dict, strict=False)

        # Summarize count of loaded parameters
        num_loaded_layers = len(checkpoint_state_dict)
        unloaded_layers = {k for k in model_state_dict if k not in checkpoint_state_dict}
        total_layers = len(model_state_dict)
        loaded_percent = num_loaded_layers / max(total_layers, 1) * 100
        rank_zero_info(
            f"Loaded {num_loaded_layers} out of {total_layers} ({loaded_percent:.1f}%) layers from checkpoint."
        )

        # Summarize unloaded layers, reporting the highest level of the hierarchy that is unloaded
        def summarize_unloaded(x: nn.Module, unloaded_layers: Set[str], prefix: str = "") -> Set[str]:
            module_params = set(x.state_dict().keys())
            unloaded_layers = {k.replace(prefix, "") for k in unloaded_layers if k.startswith(prefix)}
            intersection = module_params.intersection(unloaded_layers)

            # If all children of x are unloaded, then x is fully unloaded
            if len(intersection) == len(module_params):
                return {prefix[:-1]}
            # If no children of x are unloaded, then x is fully loaded
            elif not intersection:
                return set()

            summary = {
                f"{prefix}{n}"
                for name, child in x.named_children()
                for n in summarize_unloaded(child, unloaded_layers, prefix=f"{name}.")
            }
            summary = summary.union(f"{prefix}{n}" for n in unloaded_layers) if not summary else summary
            return summary

        unloaded_layers = ", ".join(sorted(summarize_unloaded(model, unloaded_layers)))
        if unloaded_layers:
            rank_zero_info(f"Unloaded layers: {unloaded_layers}")

    return model


def try_compile_model(model: nn.Module) -> nn.Module:
    """
    Attempts to compile the given model. If the compilation fails, logs the exception and returns the uncompiled model.

    Args:
        model: The model to compile.

    Returns:
        The compiled model if successful, otherwise the original uncompiled model.
    """
    try:
        logging.info(f"Compiling {model.__class__.__name__}...")
        model = cast(nn.Module, torch.compile(model))  # type: ignore
    except Exception as e:
        logging.exception(f"Failed to compile {model.__class__.__name__}.", exc_info=e)
    return model


T = TypeVar("T")


@overload
def to_tuple(x: Union[T, Iterable[T]], length: Literal[1]) -> Tuple[T]:
    pass


@overload
def to_tuple(x: Union[T, Iterable[T]], length: Literal[2]) -> Tuple[T, T]:
    pass


@overload
def to_tuple(x: Union[T, Iterable[T]], length: Literal[2]) -> Tuple[T, T, T]:
    pass


@overload
def to_tuple(x: Union[T, Iterable[T]], length: int) -> Tuple[T, ...]:
    pass


def to_tuple(x: Union[T, Iterable[T]], length: int) -> Tuple[T, ...]:
    """
    Converts a value or iterable of values to a tuple.

    Args:
        x: The value or iterable of values to convert to a tuple.
        length: The expected length of the tuple.

    Raises:
        * ValueError: If `x` is a non-str iterable and its length does not match `length`.

    Returns:
        The value or iterable of values as a tuple.
    """
    if isinstance(x, Sized) and len(x) == length:
        return tuple(cast(Iterable[T], x))
    elif isinstance(x, Iterable) and not isinstance(x, str):
        result = tuple(x)
        if not len(result) == length:
            raise ValueError(f"Expected an iterable of length {length}, but got {len(result)}.")
        return result
    else:
        return cast(Tuple[T, ...], (x,) * length)
