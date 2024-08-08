import argparse
import re
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, cast

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file, save_model
from torch import Tensor


def replace(s: str, replace: List[Tuple[str, str]]) -> str:
    for pattern, replacement in replace:
        s = re.sub(pattern, replacement, s)
    return s


def apply_transforms(
    entries: Iterator[Tuple[str, Tensor]],
    include: List[str] = ["*"],
    exclude: List[str] = [],
    replacements: List[Tuple[str, str]] = [],
) -> Iterator[Tuple[str, Tensor]]:
    """Applies transformations to entries in a checkpoint dictionary.

    This method intentionally accepts an iterator to reduce memory usage.

    Args:
        entries: An iterator over the entries in the checkpoint dictionary.
        include: Wildcards for weight names to include.
        exclude: Wildcards for weight names to exclude.
        replacements: Patterns of form (pattern, replacement) to replace in the weight names.

    Yields:
        Transformed entries.
    """
    yield from (
        (replace(k, replacements), v)
        for k, v in entries
        if any(fnmatch(k, i) for i in include) and not any(fnmatch(k, e) for e in exclude)
    )


def iterate_safetensors_state_dict(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Iterator[Tuple[str, Tensor]]:
    """Iterates over the state dictionary in a SafeTensors checkpoint.

    Opens a SafeTensors checkpoint file, optionally handling the case where the state dictionary
    is stored under a 'state_dict' key within the file. It iterates over the keys in the file, retrieving each
    tensor and yielding it along with its name. Takes advantage of the fact that SafeTensors loads lazily.

    Args:
        checkpoint_path: The path to the SafeTensors checkpoint file.
        device: The device to load the tensors onto.

    Yields:
        Tuples of the form (name, tensor).
    """
    with safe_open(checkpoint_path, framework="pt", device=device.index) as f:  # type: ignore
        # Handle case where "state_dict" is a key in the file
        if isinstance(f, dict) and "state_dict" in f.keys():
            f = f["state_dict"]
        yield from ((k, cast(Any, f).get_tensor(k)) for k in f.keys())


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

    # Load the state dict and apply the transformations. We do our best to load the state dict
    # lazily. Otherwise we will end up using 2x the memory.
    cp = torch.load(source, map_location="cpu", weights_only=False)
    transform = partial(apply_transforms, include=include, exclude=exclude, replacements=replacements)
    state_dict = {k: v for k, v in transform(cp["state_dict"].items())}

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
            return self._state_dict

    model = DummyModel(state_dict)
    save_model(model, str(dest))


def summarize(target: Path | Iterator[Tuple[str, Tensor]]) -> str:
    r"""Summarize a safetensors checkpoint.

    Args:
        target: Path to the checkpoint or an iterator over the entries in the checkpoint dictionary.

    Returns:
        A string with the summary.
    """
    if isinstance(target, Path):
        if not target.is_file():
            raise FileNotFoundError(target)
        state_dict = iterate_safetensors_state_dict(target)
    else:
        state_dict = target

    # Convert the state dict to a string and track the total number of weights
    total_weights = 0

    def entry_to_str(entry: Tuple[str, Tensor]) -> str:
        k, t = entry
        nonlocal total_weights
        total_weights += t.numel()
        return f"{k}: {tuple(t.shape)} {t.dtype}"

    try:
        s = f"Summary of {str(target) if isinstance(target, Path) else 'checkpoint'}:\n"
        for line in map(entry_to_str, state_dict):
            s += f"{line}\n"
        s += f"Total weights: {total_weights}"
    except Exception as ex:
        raise RuntimeError(f"Failed to summarize {str(target) if isinstance(target, Path) else 'checkpoint'}.") from ex

    return s


def transform_safetensors(
    source: Path,
    dest: Path,
    include: List[str] = ["*"],
    exclude: List[str] = [],
    replacements: List[Tuple[str, str]] = [],
    dry_run: bool = False,
) -> Iterator[Tuple[str, Tensor]]:
    """Transforms a SafeTensors checkpoint.

    Args:
        source: Path to the source checkpoint.
        dest: Path to the destination file.
        include: Wildcards for weight names to include.
        exclude: Wildcards for weight names to exclude.
        replacements: Pattern to replace in the weight names.
        dry_run: If True, do not write the transformed checkpoint to disk.
    """
    if not source.is_file():
        raise FileNotFoundError(source)  # pragma: no cover
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)  # pragma: no cover

    transform = partial(apply_transforms, include=include, exclude=exclude, replacements=replacements)
    if dry_run:
        yield from transform(iterate_safetensors_state_dict(source))
    else:
        state_dict = dict(transform(iterate_safetensors_state_dict(source)))
        save_file(state_dict, str(dest))
        yield from state_dict.items()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safetensors", description="SafeTensors utility.")
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a PyTorch checkpoint to SafeTensors format. Reads only the state dict from the checkpoint.",
    )
    transform_parser = subparsers.add_parser(
        "transform",
        help="Transform a SafeTensors checkpoint.",
    )
    transform_parser.add_argument(
        "-d", "--dry-run", default=False, action="store_true", help="Do not write the transformed checkpoint to disk."
    )
    for subparser in (convert_parser, transform_parser):
        subparser.add_argument("source", type=Path, help="Path to the source checkpoint.")
        subparser.add_argument("dest", type=Path, help="Path to the destination file.")
        subparser.add_argument(
            "-i", "--include", nargs="+", default=["*"], help="Wildcards for weight names to include."
        )
        subparser.add_argument("-e", "--exclude", nargs="+", default=[], help="Wildcards for weight names to exclude.")
        subparser.add_argument(
            "-r",
            "--replace",
            nargs=2,
            action="append",
            default=[],
            metavar=("regex", "replacement"),
            help="Pattern(s) to replace in the weight names.",
        )
        subparser.add_argument(
            "-v", "--verbose", default=False, action="store_true", help="Summarize the checkpoint when done."
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
        if args.verbose:
            print(summarize(args.dest))
    elif args.command == "cat":
        print(summarize(args.path))
    elif args.command == "transform":
        result = transform_safetensors(args.source, args.dest, args.include, args.exclude, args.replace, args.dry_run)
        if args.verbose or args.dry_run:
            print(summarize(result))
        else:
            # Force the iterator to run
            list(result)
    else:
        raise ValueError(f"Invalid command: {args.command}")  # pragma: no cover


def entrypoint() -> None:
    args = create_parser().parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
