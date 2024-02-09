import argparse
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_model
from torch import Tensor


def replace(s: str, replace: List[Tuple[str, str]]) -> str:
    for pattern, replacement in replace:
        s = re.sub(pattern, replacement, s)
    return s


def convert_to_safetensors(
    source: Path,
    dest: Path,
    include: List[str] = ["*"],
    exclude: List[str] = [],
    replacements: List[Tuple[str, str]] = [],
) -> None:
    """Converts a PyTorch checkpoint to SafeTensors format.

    This function reads only the state dict from the checkpoint and saves it in SafeTensors format.
    The tensors are deduplicated and made contiguous during the conversion.

    Args:
        source: Path to the source checkpoint.
        dest: Path to the destination file.
        include: Wildcards for weight names to include.
        exclude: Wildcards for weight names to exclude.
        replacements: Pattern to replace in the weight names.
    """
    if not source.is_file():
        raise FileNotFoundError(source)  # pragma: no cover
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)  # pragma: no cover

    # Load the state dict
    cp = torch.load(source, map_location="cpu")
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
            return {
                replace(k, replacements): v
                for k, v in self._state_dict.items()
                if any(fnmatch(k, i) for i in include) and not any(fnmatch(k, e) for e in exclude)
            }

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
    convert_parser.add_argument(
        "-i", "--include", nargs="+", default=["*"], help="Wildcards for weight names to include."
    )
    convert_parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Wildcards for weight names to exclude.")
    convert_parser.add_argument(
        "-r",
        "--replace",
        nargs=2,
        action="append",
        default=[],
        metavar=("regex", "replacement"),
        help="Pattern(s) to replace in the weight names.",
    )

    cat_parser = subparsers.add_parser(
        "cat",
        help="Print the summary of a SafeTensors checkpoint.",
    )
    cat_parser.add_argument("path", type=Path, help="Path to the checkpoint.")

    return parser


def main(args: argparse.Namespace) -> None:
    if args.command == "convert":
        convert_to_safetensors(args.source, args.dest, args.include, args.exclude, args.replace)
    elif args.command == "cat":
        print(summarize(args.path))


def entrypoint() -> None:
    args = create_parser().parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
