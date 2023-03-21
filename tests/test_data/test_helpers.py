#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import torch
from torch import Tensor

from deep_helpers.data import DatasetNames, uncollate
from deep_helpers.structs import Mode


class TestUncollate:
    def test_uncollate_tensors(self):
        batch = {"t1": torch.rand(2, 4), "t2": torch.rand(2, 8)}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                assert isinstance(v, Tensor)
                assert (v == batch[k][i]).all()

    def test_uncollate_mixed(self):
        batch = {"t1": torch.rand(2, 4), "paths": [Path("foo.dcm"), Path("bar.dcm")]}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                if isinstance(v, Tensor):
                    assert (v == batch[k][i]).all()
                else:
                    assert v == batch[k][i]

    def test_repeat(self):
        batch = {"t1": torch.rand(2, 4), "paths": 32}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                if isinstance(v, Tensor):
                    assert (v == batch[k][i]).all()
                else:
                    assert v == batch[k]

    def test_uncollate_empty(self):
        batch = {"t1": torch.rand(0, 2, 4), "t2": torch.rand(0, 2, 8)}
        result = list(uncollate(batch))
        assert not result

    def test_uncollate_scalar(self):
        batch = {"t1": torch.tensor(0.0), "t2": torch.tensor([0.0])}
        result = list(uncollate(batch))
        assert len(result) == 1
        assert result[0]["t1"] == 0.0
        assert result[0]["t2"] == 0.0


class TestDatasetNames:
    @pytest.fixture
    def names(self, mode, idx, name):
        names = DatasetNames()
        names[(mode, idx)] = name
        return names

    @pytest.mark.parametrize(
        "mode, idx, name, key, exp",
        [
            (Mode.TRAIN, 0, "foo", Mode.TRAIN, {0: "foo"}),
            (Mode.TRAIN, 0, "foo", (Mode.TRAIN, None), "foo"),
            (Mode.TRAIN, 0, "foo", (Mode.TRAIN, 0), "foo"),
            pytest.param(Mode.TRAIN, 0, "foo", Mode.VAL, None, marks=pytest.mark.xfail(raises=KeyError, strict=True)),
        ],
    )
    def test_getitem(self, names, key, exp):
        assert names[key] == exp

    @pytest.mark.parametrize(
        "mode, idx, name, exp",
        [
            (Mode.TRAIN, 0, "foo", ["foo"]),
            (Mode.TRAIN, 0, "bar", ["bar"]),
        ],
    )
    def test_all_names(self, names, exp):
        assert list(names.all_names) == exp

    @pytest.mark.parametrize(
        "mode, idx, name, key, exp",
        [
            (Mode.TRAIN, 0, "foo", Mode.TRAIN, ["foo"]),
            (Mode.TRAIN, 0, "foo", Mode.VAL, []),
        ],
    )
    def test_names_for_mode(self, names, key, exp):
        assert list(names.names_for_mode(key)) == exp
