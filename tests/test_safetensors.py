import runpy
import sys

import pytest
import torch
from safetensors import safe_open

from deep_helpers.safetensors import convert_to_safetensors


@pytest.fixture
def state_dict():
    duplicated = torch.rand(3, 3)
    return {
        "key": torch.rand(3, 3),
        "dup1": duplicated,
        "dup2": duplicated,
    }


@pytest.fixture
def checkpoint(tmp_path, state_dict):
    path = tmp_path / "source.pt"
    torch.save({"state_dict": state_dict}, path)
    return path


def test_convert_to_safetensors(tmp_path, checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(checkpoint, dest)

    # Load the converted tensor and check if it matches the original tensor
    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert torch.allclose(f.get_tensor("dup1"), state_dict["dup1"])
        assert "dup2" not in f.keys()


def test_safetensors_cli(tmp_path, checkpoint):
    dest = tmp_path / "dest.safetensors"
    sys.argv = [
        sys.argv[0],
        str(checkpoint),
        str(dest),
    ]
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    assert dest.is_file()
