"""
InferenceService — image preprocessing and forward pass.

Pipeline:
    raw bytes → PIL RGB → albumentations → torch tensor → model → sigmoid → probs

The albumentations transform pipeline is built ONCE at module import
(matches the training pipeline in Team-B-Backend/dataset.py exactly).
"""

import io
import logging
import time
from typing import Dict, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from app.config import (
    DEFAULT_THRESHOLD,
    DISEASE_LABELS,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
)
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

# Build once — reused for every request
_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])


def _preprocess(file_bytes: bytes) -> torch.Tensor:
    """
    Decode raw image bytes and produce a model-ready tensor.

    Args:
        file_bytes: Raw bytes of a JPEG/PNG/BMP image.

    Returns:
        Tensor of shape [1, 3, IMAGE_SIZE, IMAGE_SIZE], normalized to ImageNet stats.
    """
    image    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    np_image = np.array(image)
    tensor   = _transform(image=np_image)["image"]  # [3, H, W]
    return tensor.unsqueeze(0)                       # [1, 3, H, W]


def run_inference(
    file_bytes: bytes,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[Dict[str, float], float]:
    """
    Run a full forward pass and return detected diseases + inference time.

    Args:
        file_bytes: Raw image bytes.
        threshold:  Sigmoid probability threshold (0–1). Diseases with
                    prob >= threshold are included in the result dict.

    Returns:
        Tuple of:
          - predictions : { label: probability } for diseases above threshold
          - elapsed_ms  : wall-clock inference time in milliseconds
    """
    model  = ModelService.get_model()
    device = ModelService.get_device()

    tensor = _preprocess(file_bytes).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits: torch.Tensor = model(tensor)                          # [1, 45]
        probs:  np.ndarray   = torch.sigmoid(logits)[0].cpu().numpy() # [45]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predictions: Dict[str, float] = {
        label: round(float(prob), 4)
        for label, prob in zip(DISEASE_LABELS, probs)
        if float(prob) >= threshold
    }

    logger.info(
        "Inference | %.1f ms | threshold=%.2f | %d disease(s) detected",
        elapsed_ms,
        threshold,
        len(predictions),
    )

    return predictions, elapsed_ms
