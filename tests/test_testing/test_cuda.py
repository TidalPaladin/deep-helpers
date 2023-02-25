#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from deep_helpers.testing import cuda_available


@pytest.mark.parametrize(
    "available, capability, arch_list, expected",
    [
        (False, (6, 0), ["sm_60"], False),
        (True, (6, 0), ["sm_60"], True),
        (True, (5, 0), ["sm_60"], False),
        (True, "sm_60", ["sm_60"], True),
        (True, "sm_50", ["sm_60"], False),
    ],
)
def test_cuda_available_mocked(mocker, available, capability, arch_list, expected):
    mocker.patch("torch.cuda.is_available", return_value=available)
    mocker.patch("torch.cuda.get_device_capability", return_value=capability)
    mocker.patch("torch.cuda.get_arch_list", return_value=arch_list)
    assert cuda_available() == expected


def test_cuda_available():
    assert isinstance(cuda_available(), bool)
