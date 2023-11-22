import argparse
from copy import copy
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_model
from torch import Tensor


def convert_to_safetensors(source: Path, dest: Path) -> None:
    """Converts a PyTorch checkpoint to SafeTensors format.

    This function reads only the state dict from the checkpoint and saves it in SafeTensors format.
    The tensors are deduplicated and made contiguous during the conversion.

    Args:
        source: Path to the source checkpoint.
        dest: Path to the destination file.
    """
    if not source.is_file():
        raise FileNotFoundError(source)  # pragma: no cover
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)  # pragma: no cover

    # Load the state dict
    cp = torch.load(source)
    state_dict = cp["state_dict"]

    # We want to save the state dict using save_model because it will deduplicate the tensors
    # and make them contiguous. However, save_model expects a model, so we need to create a dummy model.
    #
    # TODO: The first line of `save_model` is `state_dict = model.state_dict()`. If there is ever a better
    # way to do this, we should use it.
    class DummyModel(nn.Module):
        def __init__(self, state_dict: Dict[str, Tensor]):
            super().__init__()
            self._state_dict = state_dict

        def state_dict(self) -> Dict[str, Tensor]:
            return copy(self._state_dict)

    model = DummyModel(state_dict)
    save_model(model, str(dest))


def summarize(path: Path) -> str:
    r"""Summarize a safetensors checkpoint.

    Args:
        path: Path to the checkpoint.

    Returns:
        A string with the summary.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    def iterate_weights(cp: Any) -> Iterator[str]:
        for k in cp.keys():
            t = f.get_tensor(k)
            yield f"{k}: {tuple(t.shape)} {t.dtype}"

    try:
        s = f"Summary of {path}:\n"
        with safe_open(path, framework="pt") as f:  # type: ignore
            for line in iterate_weights(f):
                s += f"{line}\n"
            total_weights = sum(f.get_tensor(k).numel() for k in f.keys())
            s += f"Total weights: {total_weights}"
    except Exception as ex:
        raise RuntimeError(f"Failed to summarize {path}.") from ex

    return s


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safetensors", description="SafeTensors utility.")
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a PyTorch checkpoint to SafeTensors format. Reads only the state dict from the checkpoint.",
    )
    convert_parser.add_argument("source", type=Path, help="Path to the source checkpoint.")
    convert_parser.add_argument("dest", type=Path, help="Path to the destination file.")

    cat_parser = subparsers.add_parser(
        "cat",
        help="Print the summary of a SafeTensors checkpoint.",
    )
    cat_parser.add_argument("path", type=Path, help="Path to the checkpoint.")

    return parser


def main(args: argparse.Namespace) -> None:
    if args.command == "convert":
        convert_to_safetensors(args.source, args.dest)
    elif args.command == "cat":
        print(summarize(args.path))


def entrypoint() -> None:
    args = create_parser().parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
