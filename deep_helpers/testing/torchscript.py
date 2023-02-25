#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyright: ignore[reportPrivateImportUsage]

import os
from typing import Any, Iterable

from torch import Tensor
from torch.jit import ScriptModule, load, save, script, trace  # type: ignore
from torch.testing import assert_close  # type: ignore


try:
    import pytest
except ImportError:
    pytest: Any = None


class TorchScriptTestMixin:
    r"""Mixin to test a :class:`torch.nn.Module`'s ability to be scripted using
    :func:`torch.jit.script`, saved to disk, and loaded.
    The following fixtures should be implemented in the subclass:
        * :func:`model` - returns the model to be tested
    """

    @pytest.fixture
    def model(self):
        raise pytest.UsageError(f"Must implement model fixture for {self.__class__.__name__}")

    def test_script(self, model):
        r"""Calls :func:`script` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.
        """
        scripted = script(model)
        assert isinstance(scripted, ScriptModule)

    def test_save_scripted(self, model, tmp_path):
        r"""Calls :func:`script` on the given model and tests that the resultant
        :class:`torch.jit.ScriptModule` can be saved to disk using :func:`save`.
        """
        path = os.path.join(tmp_path, "model.pth")
        scripted = script(model)
        assert isinstance(scripted, ScriptModule)
        save(scripted, path)
        assert os.path.isfile(path)

    def test_load_scripted(self, model, tmp_path):
        r"""Tests that a :class:`torch.jit.ScriptModule` saved to disk using :func:`script` can be
        loaded, and that the loaded object is a :class:`torch.jit.ScriptModule`.
        """
        path = os.path.join(tmp_path, "model.pth")
        scripted = script(model)
        save(scripted, path)
        loaded = load(path)
        assert isinstance(loaded, ScriptModule)


class TorchScriptTraceTestMixin:
    r"""Mixin to test a :class:`torch.nn.Module`'s ability to be traced using
    :func:`trace`, saved to disk, and loaded.
    The following fixtures should be implemented in the subclass:
        * :func:`model` - returns the model to be tested
        * :func:`data` - returns an input to ``model.forward()``.
    """

    @pytest.fixture
    def model(self):
        raise pytest.UsageError(f"Must implement model fixture for {self.__class__.__name__}")

    @pytest.fixture
    def data(self):
        raise pytest.UsageError("Must implement data fixture for {self.__class__.__name__}")

    def test_trace(self, model, data):
        r"""Calls :func:`trace` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.
        """
        traced = trace(model, data)
        assert isinstance(traced, ScriptModule)

    @pytest.mark.cuda
    def test_traced_forward_call(self, model, data):
        r"""Calls :func:`trace` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.
        Because of the size of some models, this test is only run when a GPU is available.
        """
        traced = trace(model, data)
        output = model(data)
        traced_output = traced(data)
        if isinstance(output, Tensor):
            assert_close(output, traced_output)
        elif isinstance(output, Iterable):
            for out, traced_out in zip(output, traced_output):
                assert_close(out, traced_out)
        else:
            pytest.skip()

    def test_save_traced(self, model, tmp_path, data):
        r"""Calls :func:`trace` on the given model and tests that the resultant
        :class:`torch.jit.ScriptModule` can be saved to disk using :func:`save`.
        """
        path = os.path.join(tmp_path, "model.pth")
        traced = trace(model, data)
        assert isinstance(traced, ScriptModule)
        save(traced, path)
        assert os.path.isfile(path)

    def test_load_traced(self, model, tmp_path, data):
        r"""Tests that a :class:`torch.jit.ScriptModule` saved to disk using :func:`trace`
        can be loaded, and that the loaded object is a :class:`torch.jit.ScriptModule`.
        """
        path = os.path.join(tmp_path, "model.pth")
        traced = trace(model, data)
        save(traced, path)
        loaded = load(path)
        assert isinstance(loaded, ScriptModule)
