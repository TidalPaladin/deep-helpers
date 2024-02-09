#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import torch
from torch import Tensor

from deep_helpers.data import DatasetNames, uncollate
from deep_helpers.structs import Mode


class TestUncollate:
    def test_tensors(self):
        batch = {"t1": torch.rand(2, 4), "t2": torch.rand(2, 8)}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                assert isinstance(v, Tensor)
                assert (v == batch[k][i]).all()

    def test_mixed(self):
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

    def test_empty(self):
        batch = {"t1": torch.rand(0, 2, 4), "t2": torch.rand(0, 2, 8)}
        result = list(uncollate(batch))
        assert not result

    def test_scalar(self):
        batch = {"t1": torch.tensor(0.0), "t2": torch.tensor([0.0])}
        result = list(uncollate(batch))
        assert len(result) == 1
        assert result[0]["t1"] == 0.0
        assert result[0]["t2"] == 0.0

    def test_expand(self):
        batch = {"t1": torch.rand(1, 2), "t2": torch.rand(4, 2), "t3": ["foo"]}
        result = list(uncollate(batch))
        assert len(result) == 4

    def test_expand_scalar(self):
        batch = {"t1": torch.tensor(0.0), "t2": torch.rand(4, 2)}
        result = list(uncollate(batch))
        assert len(result) == 4
        assert result[0]["t1"] == 0.0
        assert result[1]["t1"] == 0.0

    def test_nested_dicts(self):
        batch = {
            "t1": torch.rand(4, 3),
            "d": {
                "t3": torch.rand(4, 2),
                "t4": torch.tensor(0.0),
                "d2": {
                    "t5": torch.rand(4, 1),
                },
            },
        }
        result = list(uncollate(batch))
        assert len(result) == 4
        assert isinstance(result[0]["t1"], Tensor)
        assert isinstance(result[0]["d"], dict)
        assert result[0]["d"]["t3"].shape == (2,)
        assert result[0]["d"]["t4"] == 0.0
        assert result[0]["d"]["d2"]["t5"].shape == (1,)

    def test_nested_dicts_batch_size(self):
        batch = {
            "t1": torch.tensor(0.0),
            "d": {
                "t3": torch.tensor(0.0),
                "t4": torch.tensor(0.0),
                "d2": {
                    "t5": torch.rand(4, 1),
                },
            },
            "t6": torch.rand(4, 1),
        }
        result = list(uncollate(batch))
        assert len(result) == 4
        assert isinstance(result[0]["t1"], Tensor)
        assert isinstance(result[0]["d"], dict)
        assert result[0]["d"]["t3"] == 0.0
        assert result[0]["d"]["t4"] == 0.0
        assert result[0]["d"]["d2"]["t5"].shape == (1,)

    def test_length_mismatch(self):
        batch = {
            "t1": torch.rand(4, 1),
            "t2": torch.rand(2, 1),
        }
        result = list(uncollate(batch))
        assert len(result) == 2
        assert isinstance(result[0]["t1"], Tensor)
        assert isinstance(result[0]["t2"], Tensor)
        assert (result[0]["t1"] == batch["t1"][0]).all()
        assert (result[1]["t1"] == batch["t1"][1]).all()
        assert (result[0]["t2"] == batch["t2"][0]).all()
        assert (result[1]["t2"] == batch["t2"][1]).all()

    @pytest.mark.parametrize(
        "struct,batch_size,exp",
        [
            ({}, None, 0),
            ({}, 4, 4),
        ],
    )
    def test_empty_dict(self, struct, batch_size, exp):
        result = list(uncollate(struct, batch_size))
        assert len(result) == exp

    def test_key_is_empty_dict(self):
        batch = {"t1": torch.rand(1, 2), "t2": torch.rand(4, 2), "t3": {}}
        result = list(uncollate(batch))
        assert len(result) == 4

    def test_nested_empty_keys(self):
        batch = {"t1": torch.rand(1, 2), "t2": torch.rand(4, 2), "t3": {"foo": []}}
        result = list(uncollate(batch))
        assert len(result) == 4
        assert all(r["t3"] == {"foo": []} for r in result)


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
