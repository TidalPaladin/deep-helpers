import runpy
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from deep_helpers.safetensors import convert_to_safetensors, summarize, transform_safetensors
from deep_helpers.testing import checkpoint_factory


@pytest.fixture(scope="module")
def state_dict():
    duplicated = torch.rand(3, 3)
    return {
        "key": torch.rand(3, 3),
        "dup1": duplicated,
        "dup2": duplicated,
    }


@pytest.fixture(scope="module")
def torch_checkpoint(tmpdir_factory, state_dict):
    path = Path(tmpdir_factory.mktemp("safetensors").join("source.pt"))
    torch.save({"state_dict": state_dict}, str(path))
    return path


@pytest.fixture(scope="module")
def safetensors_checkpoint(tmpdir_factory, state_dict):
    state_dict = {k: v for k, v in state_dict.items() if "dup" not in k}
    path = Path(tmpdir_factory.mktemp("safetensors").join("source.safetensors"))
    save_file(state_dict, path)
    return path


def test_convert_to_safetensors(tmp_path, torch_checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest)

    # Load the converted tensor and check if it matches the original tensor
    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert torch.allclose(f.get_tensor("dup1"), state_dict["dup1"])
        assert "dup2" not in f.keys()


def test_convert_to_safetensors_include(tmp_path, torch_checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest, include=["key"])

    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert "dup1" not in f.keys()
        assert "dup2" not in f.keys()


def test_convert_to_safetensors_exclude(tmp_path, torch_checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest, exclude=["dup1"])

    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert "dup1" not in f.keys()
        assert torch.allclose(f.get_tensor("dup2"), state_dict["dup2"])


def test_convert_to_safetensors_exclude_wildcard(tmp_path, torch_checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest, exclude=["*du*"])

    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert "dup1" not in f.keys()
        assert "dup2" not in f.keys()


def test_convert_to_safetensors_replace(tmp_path, torch_checkpoint, state_dict):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest, replacements=[("dup", "rep")])

    with safe_open(dest, framework="pt") as f:  # type: ignore
        assert torch.allclose(f.get_tensor("key"), state_dict["key"])
        assert "rep1" in f.keys()
        assert torch.allclose(f.get_tensor("rep1"), state_dict["dup1"])
        assert "dup1" not in f.keys()
        assert "dup2" not in f.keys()


def test_summarize(tmp_path, torch_checkpoint):
    dest = tmp_path / "dest.safetensors"
    convert_to_safetensors(torch_checkpoint, dest)

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


@pytest.mark.parametrize("dry_run", [False, True])
def test_transform_safetensors(tmp_path, safetensors_checkpoint, dry_run):
    dest = tmp_path / "dest.safetensors"
    result = transform_safetensors(safetensors_checkpoint, dest, replacements=[("key", "rep")], dry_run=dry_run)
    result = dict(result)
    assert isinstance(result["rep"], Tensor)
    assert dest.is_file() != dry_run


@pytest.mark.parametrize("verbose", [False, True])
def test_safetensors_cli_convert(tmp_path, torch_checkpoint, verbose, capsys):
    dest = tmp_path / "dest.safetensors"
    sys.argv = [
        sys.argv[0],
        "convert",
        str(torch_checkpoint),
        str(dest),
    ]
    if verbose:
        sys.argv.append("--verbose")
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert dest.is_file()
    assert ("Total weights" in captured.out) == verbose


def test_safetensors_cli_cat(capsys, tmp_path, safetensors_checkpoint):
    sys.argv = [
        sys.argv[0],
        "cat",
        str(safetensors_checkpoint),
    ]
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert "Total weights" in captured.out


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_safetensors_cli_transform(capsys, tmp_path, safetensors_checkpoint, verbose, dry_run):
    dest = tmp_path / "dest.safetensors"
    sys.argv = [
        sys.argv[0],
        "transform",
        str(safetensors_checkpoint),
        str(dest),
    ]
    if verbose:
        sys.argv.append("--verbose")
    if dry_run:
        sys.argv.append("--dry-run")
    runpy.run_module("deep_helpers.safetensors", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert dest.is_file() != dry_run
    assert ("Total weights" in captured.out) == (verbose or dry_run)


def test_map_location(mocker, tmp_path, torch_checkpoint):
    dest = tmp_path / "dest.safetensors"
    spy = mocker.spy(torch, "load")
    convert_to_safetensors(torch_checkpoint, dest)
    spy.assert_called_once_with(torch_checkpoint, map_location="cpu", weights_only=False)
