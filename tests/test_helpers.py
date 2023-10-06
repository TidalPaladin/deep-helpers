#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn

import deep_helpers
from deep_helpers import load_checkpoint
from deep_helpers.helpers import try_compile_model


@pytest.mark.parametrize(
    "checkpoint, model, strict, matching_count",
    [
        pytest.param(nn.Conv2d(3, 10, 3), nn.Conv2d(3, 10, 3), True, 2),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Conv2d(3, 10, 3), True, 0, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Conv2d(3, 10, 3), False, 0),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Linear(3, 5), True, 1, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Linear(3, 5), False, 1),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Identity(), True, 0, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Identity(), False, 0),
        pytest.param(nn.Identity(), nn.Conv2d(3, 5, 3), True, 0, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Identity(), nn.Conv2d(3, 5, 3), False, 0),
    ],
)
def test_load_checkpoint(mocker, checkpoint, model, strict, matching_count):
    # spy the log call
    log_spy = mocker.spy(deep_helpers.helpers, "rank_zero_info")

    spy = mocker.spy(model, "load_state_dict")
    state_dict = checkpoint.state_dict()
    model = load_checkpoint(model, state_dict, strict=strict)

    if not strict:
        total_layers = len(model.state_dict())
        expected_percent = 100 * (matching_count / max(1, total_layers))
        msg = f"Loaded {matching_count} out of {total_layers} ({expected_percent:.1f}%) layers from checkpoint."
        for args in log_spy.call_args_list:
            if msg in args[0][0]:
                break
        else:
            pytest.fail(f"Expected log message: {msg}")
    else:
        spy.assert_called_once()


def test_load_checkpoint_report_unloaded_layers(mocker):
    checkpoint = nn.ModuleDict(
        {
            "conv": nn.Conv2d(3, 10, 3),
            "convdict1": nn.ModuleDict(
                {
                    "conv2": nn.Conv2d(3, 10, 3),
                    "conv3": nn.Conv2d(3, 10, 3),
                }
            ),
            "convdict2": nn.ModuleDict(
                {
                    "conv2": nn.Conv2d(3, 10, 3),
                    "conv3": nn.Conv2d(3, 10, 3),
                    "convlist": nn.ModuleList(
                        [
                            nn.Linear(3, 5),
                        ]
                    ),
                }
            ),
            "linear": nn.Linear(3, 5),
        }
    )
    model = nn.ModuleDict(
        {
            "conv": nn.Conv2d(3, 20, 3),
            "convdict1": nn.ModuleDict(
                {
                    "conv2": nn.Conv2d(3, 20, 3),
                    "conv3": nn.Conv2d(3, 20, 3),
                }
            ),
            "convdict2": nn.ModuleDict(
                {
                    "conv2": nn.Conv2d(3, 20, 3),
                    "conv3": nn.Conv2d(3, 10, 3),
                    "convlist": nn.ModuleList(
                        [
                            nn.Linear(10, 5),
                        ]
                    ),
                }
            ),
            "linear": nn.Linear(3, 5),
        }
    )
    state_dict = checkpoint.state_dict()

    log_spy = mocker.spy(deep_helpers.helpers, "rank_zero_info")
    model = load_checkpoint(model, state_dict, strict=False)
    log_spy.assert_called_with("Unloaded layers: conv, convdict1, convdict2.conv2, convdict2.convlist.0.weight")


class TestTryCompileModel:
    def test_mock_compile(self, mocker):
        m = mocker.patch("torch.compile", return_value=mocker.MagicMock(name="compiled_model"))
        result = try_compile_model(nn.Linear(10, 10))
        m.assert_called_once()
        assert result is m.return_value

    @pytest.mark.parametrize(
        "model",
        [
            nn.Conv2d(3, 10, 3),
            nn.Linear(3, 5),
        ],
    )
    def test_compile(self, mocker, model, caplog):
        spy = mocker.spy(torch, "compile")
        model = try_compile_model(model)
        spy.assert_called_once()
        assert "Exception" not in caplog.text

    def test_exception(self, mocker, caplog):
        model = nn.Linear(10, 10)
        mocker.patch("torch.compile", side_effect=Exception)
        result = try_compile_model(model)
        assert result is model
        assert "Exception" in caplog.text
