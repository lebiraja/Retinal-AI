"""
Image preprocessing pipeline.

The transform chain here MUST be identical to the validation/test pipeline used
during training (see dataset.py at the project root).  Any deviation will
silently degrade model accuracy at inference time.

Pipeline:
  base64 string → raw bytes → PIL RGB → NumPy array
  → albumentations (Resize + Normalize) → torch Tensor [1, 3, H, W]
"""

from __future__ import annotations

import base64
import io
import logging

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from services.model.src.config import IMAGE_SIZE, NORM_MEAN, NORM_STD

logger = logging.getLogger(__name__)

# Build once at module import — all requests share the same object (thread-safe
# since albumentations transforms are stateless).
_TRANSFORM: A.Compose = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]
)


def preprocess_b64(image_b64: str) -> torch.Tensor:
    """
    Decode a base64-encoded image string and produce a model-ready tensor.

    Args:
        image_b64: Base64-encoded image bytes (any PIL-supported format).

    Returns:
        Float32 tensor of shape [1, 3, IMAGE_SIZE, IMAGE_SIZE].

    Raises:
        ValueError: If the base64 data is malformed or the image cannot be decoded.
    """
    try:
        raw_bytes: bytes = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc

    return preprocess_bytes(raw_bytes)


def preprocess_bytes(raw_bytes: bytes) -> torch.Tensor:
    """
    Decode raw image bytes and produce a model-ready tensor.

    Args:
        raw_bytes: Raw bytes of any PIL-supported image format.

    Returns:
        Float32 tensor of shape [1, 3, IMAGE_SIZE, IMAGE_SIZE].

    Raises:
        ValueError: If the bytes cannot be opened as an image.
    """
    try:
        pil_image: Image.Image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    np_image: np.ndarray = np.array(pil_image, dtype=np.uint8)
    tensor: torch.Tensor = _TRANSFORM(image=np_image)["image"]  # [3, H, W]
    return tensor.unsqueeze(0)                                   # [1, 3, H, W]
