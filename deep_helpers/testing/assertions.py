#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Iterable, Sized, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing import assert_close  # type: ignore


def assert_has_gradient(module: Union[nn.Module, Tensor], recurse: bool = True):
    r"""Asserts that the parameters in a module have ``requires_grad=True`` and
    that the gradient exists.
    Args:
        module:
            The module to inspect
        recurse:
            Whether or not to recursively run the same assertion on the gradients
            of child modules.
    """
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and module.grad_fn is None:
        raise AssertionError(f"tensor grad == {module.grad_fn}")
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and param.grad is None:
                raise AssertionError(f"param {name} grad == {param.grad}")


def assert_zero_grad(module: nn.Module, recurse: bool = True):
    r"""Asserts that the parameters in a module have zero gradients.
    Useful for checking if `Optimizer.zero_grads()` was called.
    Args:
        module:
            The module to inspect
        recurse:
            Whether or not to recursively run the same assertion on the gradients
            of child modules.
    """
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and not all(module.grad == 0):
        raise AssertionError(f"module.grad == {module.grad}")
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and not (param.grad is None or (~param.grad.bool()).all()):
                raise AssertionError(f"param {name} grad == {param.grad}")


def assert_in_training_mode(module: nn.Module):
    r"""Asserts that the module is in training mode, i.e. ``module.train()``
    was called
    Args:
        module:
            The module to inspect
    """
    __tracebackhide__ = True
    if not module.training:
        raise AssertionError(f"module.training == {module.training}")


def assert_in_eval_mode(module: nn.Module):
    r"""Asserts that the module is in inference mode, i.e. ``module.eval()``
    was called.
    Args:
        module:
            The module to inspect
    """
    __tracebackhide__ = True
    if module.training:
        raise AssertionError(f"module.training == {module.training}")


def assert_is_int_tensor(x: Tensor):
    r"""Asserts that the values of a floating point tensor are integers.
    This test is equivalent to ``torch.allclose(x, x.round())``.
    Args:
        x:
            The first tensor.
        y:
            The second tensor.
    """
    __tracebackhide__ = True
    if not torch.allclose(x, x.round()):
        try:
            assert str(x) == str(x.round())
        except AssertionError as e:
            raise AssertionError(str(e)) from e


def assert_equal(arg1: Any, arg2: Any, **kwargs):
    r"""Asserts equality of two inputs, using ``torch.assert_close`` if both inputs
    are tensors.

    Args:
        arg1:
            The first input.
        arg2:
            The second input.

    Keyword Args:
        kwargs:
            Additional keyword arguments to pass to ``torch.assert_close``.
    """
    __tracebackhide__ = True
    assert isinstance(arg1, type(arg2))

    if isinstance(arg1, Tensor):
        assert_close(arg1, arg2, **kwargs)

    elif isinstance(arg1, dict) and isinstance(arg2, dict):
        for (k1, v1), (k2, v2) in zip(arg1.items(), arg2.items()):
            assert_equal(k1, k2, **kwargs)
            assert_equal(v1, v2, **kwargs)

    elif isinstance(arg1, str) and isinstance(arg2, str):
        assert arg1 == arg2, f"{arg1} != {arg2}"

    elif isinstance(arg1, Iterable) and isinstance(arg2, Iterable):
        if isinstance(arg1, Sized) and isinstance(arg2, Sized):
            assert len(arg1) == len(arg2), f"Length mismatch: {len(arg1)} != {len(arg2)}"
        for a1, a2 in zip(arg1, arg2):
            assert_equal(a1, a2, **kwargs)
    else:
        assert arg1 == arg2, f"{arg1} != {arg2}"


__all__ = [
    "assert_equal",
    "assert_has_gradient",
    "assert_zero_grad",
    "assert_is_int_tensor",
    "assert_in_training_mode",
    "assert_in_eval_mode",
]
