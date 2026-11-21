from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import load_densenet121


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the trained six-class DenseNet121 checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="models/densenet121_best.pth",
        help="Path to densenet121_best.pth",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.checkpoint).expanduser()
    try:
        model = load_densenet121(path, device=device)
        print(f"Checkpoint: {path.resolve()}")
        print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Device: {device}")
        print(f"Architecture: {model.__class__.__name__}")
        print(f"Classifier: {model.classifier.in_features} -> {model.classifier.out_features}")
        print("Checkpoint validation: PASSED")
        return 0
    except Exception as exc:
        print(f"Checkpoint validation: FAILED\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
