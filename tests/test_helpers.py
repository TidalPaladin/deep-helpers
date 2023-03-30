#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch.nn as nn

from deep_helpers import load_checkpoint


@pytest.mark.parametrize(
    "checkpoint, model, strict",
    [
        pytest.param(nn.Conv2d(3, 10, 3), nn.Conv2d(3, 10, 3), True),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Conv2d(3, 10, 3), True, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Conv2d(3, 10, 3), False),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Linear(3, 5), True, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Linear(3, 5), False),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Identity(), True, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Conv2d(3, 5, 3), nn.Identity(), False),
        pytest.param(nn.Identity(), nn.Conv2d(3, 5, 3), True, marks=pytest.mark.xfail(strict=True)),
        pytest.param(nn.Identity(), nn.Conv2d(3, 5, 3), False),
    ],
)
def test_load_checkpoint(mocker, checkpoint, model, strict):
    spy = mocker.spy(model, "load_state_dict")
    state_dict = checkpoint.state_dict()
    model = load_checkpoint(model, state_dict, strict=strict)
    spy.assert_called_once()
