#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict

import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore


def load_checkpoint(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True) -> nn.Module:
    """Load weights from a state dict. When loading with ``strict=False``, missing or unexpected
    keys will be ignored and keys with shapes that do not match will be ignored.

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
    return model
