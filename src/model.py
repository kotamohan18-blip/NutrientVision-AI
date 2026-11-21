from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import densenet121

CLASS_LABELS = ["B12", "Iron", "Vitamin_A", "Vitamin_C", "Vitamin_D", "Zinc"]
NUM_CLASSES = len(CLASS_LABELS)


def build_densenet121() -> nn.Module:
    """
    Recreate the model head used in the project notebook:
    torchvision DenseNet121 with a six-output linear classifier.
    """
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    if in_features != 1024:
        raise RuntimeError(
            f"Unexpected DenseNet121 classifier width: {in_features}; expected 1024."
        )
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    return model


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    # The supplied notebook saves a plain model.state_dict(). The wrapper cases
    # support common state-dictionary containers without loading arbitrary objects.
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a valid state dictionary.")

    if not all(isinstance(k, str) for k in checkpoint):
        raise ValueError("Checkpoint keys must be strings.")

    # Handle DataParallel checkpoints if a user saved one.
    if all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}

    if "classifier.weight" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing 'classifier.weight'. "
            "Use the trained six-class DenseNet121 checkpoint from the project notebook."
        )

    weight = checkpoint["classifier.weight"]
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (NUM_CLASSES, 1024):
        raise ValueError(
            "Checkpoint classifier shape is not (6, 1024). "
            "This app requires the six-class DenseNet121 checkpoint."
        )
    return checkpoint


def load_densenet121(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    device = device or torch.device("cpu")
    model = build_densenet121()

    # weights_only=True avoids unrestricted pickle object loading. The notebook
    # saves a plain state_dict, which is compatible with this safe loading mode.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This app requires a PyTorch version that supports weights_only=True. "
            "Upgrade PyTorch to the version listed in requirements."
        ) from exc

    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
