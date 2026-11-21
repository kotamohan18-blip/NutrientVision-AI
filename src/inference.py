from __future__ import annotations

from io import BytesIO
from typing import Iterable

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

CLASS_LABELS = ["B12", "Iron", "Vitamin_A", "Vitamin_C", "Vitamin_D", "Zinc"]
MAX_UPLOAD_MB = 12
MAX_IMAGE_PIXELS = 30_000_000

# Matches the deterministic evaluation transform in the project notebook.
INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_uploaded_image(raw_bytes: bytes) -> Image.Image:
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")

    try:
        with Image.open(BytesIO(raw_bytes)) as probe:
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large to process safely.")
            probe.verify()

        with Image.open(BytesIO(raw_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Unsupported or corrupted image. Upload a valid JPG or PNG.") from exc


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")
    return INFERENCE_TRANSFORM(image.convert("RGB"))


@torch.inference_mode()
def predict_images(
    model: torch.nn.Module,
    images: Iterable[Image.Image],
    device: torch.device,
    batch_size: int = 16,
) -> list[dict]:
    image_list = list(images)
    if not image_list:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    model.eval()
    results: list[dict] = []

    for start in range(0, len(image_list), batch_size):
        batch_images = image_list[start : start + batch_size]
        batch_tensor = torch.stack([preprocess_image(image) for image in batch_images])
        batch_tensor = batch_tensor.to(device)

        logits = model(batch_tensor)
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_LABELS):
            raise RuntimeError(
                f"Model returned shape {tuple(logits.shape)}; expected (batch, 6)."
            )

        probabilities = torch.softmax(logits, dim=1).cpu()
        for probability in probabilities:
            top_idx = int(torch.argmax(probability).item())
            scores = {
                label: float(probability[idx].item())
                for idx, label in enumerate(CLASS_LABELS)
            }
            results.append(
                {
                    "predicted_label": CLASS_LABELS[top_idx],
                    "model_score": float(probability[top_idx].item()),
                    "scores": scores,
                }
            )

    return results
