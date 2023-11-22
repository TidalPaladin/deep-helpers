import runpy
import sys

import pytest
import torch
from safetensors import safe_open

from deep_helpers.safetensors import convert_to_safetensors, summarize
from deep_helpers.testing import checkpoint_factory


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


def test_summarize(tmp_path, checkpoint):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(checkpoint, dest)

    # Call summarize and check if it returns a string
    summary = summarize(dest)
    assert isinstance(summary, str)


def test_load_from_task(mocker, tmp_path, task):
    torch_checkpoint = checkpoint_factory(task, root=tmp_path)
    safetensors_checkpoint = tmp_path / "checkpoint.safetensors"
    convert_to_safetensors(torch_checkpoint, safetensors_checkpoint)
    task.checkpoint = safetensors_checkpoint
    spy = mocker.spy(task, "_safe_load_checkpoint")
    task.setup()
    spy.assert_called_once()


def test_safetensors_cli_convert(tmp_path, checkpoint):
    dest = tmp_path / "dest.safetensors"
    sys.argv = [
        sys.argv[0],
        "convert",
        str(checkpoint),
        str(dest),
    ]
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    assert dest.is_file()


def test_safetensors_cli_cat(capsys, tmp_path, checkpoint):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(checkpoint, dest)
    sys.argv = [
        sys.argv[0],
        "cat",
        str(dest),
    ]
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert "Total weights" in captured.out
