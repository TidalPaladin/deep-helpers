import argparse
from copy import copy
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
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
        raise FileNotFoundError(source)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="safetensors-convert",
        description=(
            "Convert a PyTorch checkpoint to SafeTensors format. Reads only the state dict from the checkpoint. "
        ),
    )
    parser.add_argument("source", type=Path, help="Path to the source checkpoint.")
    parser.add_argument("dest", type=Path, help="Path to the destination file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    convert_to_safetensors(args.source, args.dest)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
