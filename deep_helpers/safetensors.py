import pytorch_lightning as pl
from safetensors.torch import save_model
from pathlib import Path
import argparse

def convert_to_safetensors(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)

    model = pl.LightningModule.load_from_checkpoint(source)
    import pdb; pdb.set_trace()
    save_model(model, str(dest))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    convert_to_safetensors(args.source, args.dest)

def entrypoint() -> None:
    args = parse_args()
    main(args)

if __name__ == "__main__":
    entrypoint()