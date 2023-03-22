#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from deep_helpers.structs.enums import Mode, State


class TestMode:
    @pytest.mark.parametrize(
        "mode,exp",
        [
            (Mode.TRAIN, "train"),
            (Mode.VAL, "val"),
            (Mode.TEST, "test"),
            (Mode.PREDICT, "predict"),
        ],
    )
    def test_str(self, mode, exp):
        assert str(mode) == exp

    @pytest.mark.parametrize(
        "exp,inp",
        [
            (Mode.TRAIN, "train"),
            (Mode.VAL, "val"),
            (Mode.TEST, "test"),
            (Mode.PREDICT, "predict"),
        ],
    )
    def test_create(self, inp, exp):
        assert Mode.create(inp) == exp


class TestState:
    @pytest.mark.parametrize(
        "mode1, mode2, exp",
        [
            (Mode.TRAIN, Mode.TRAIN, True),
            (Mode.TRAIN, Mode.VAL, False),
            (Mode.TEST, Mode.TRAIN, False),
        ],
    )
    def test_eq(self, mode1, mode2, exp):
        state1 = State(mode1)
        state2 = State(mode2)
        assert (state1 == state2) == exp
