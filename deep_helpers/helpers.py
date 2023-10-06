#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, cast

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
        loaded_layers = len(checkpoint_state_dict)
        total_layers = len(model_state_dict)
        loaded_percent = loaded_layers / max(total_layers, 1) * 100
        rank_zero_info(f"Loaded {loaded_layers} out of {total_layers} ({loaded_percent:.1f}%) layers from checkpoint.")
        model.load_state_dict(checkpoint_state_dict, strict=False)
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
