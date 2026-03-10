"""
ValidationService — lightweight heuristic pre-filter for uploaded images.

Rejects clearly non-fundus images before they reach the GPU inference step,
saving latency and preventing meaningless predictions.

Fundus images have three reliable visual signatures:
  1. Roughly square aspect ratio (ophthalmoscope aperture is circular).
  2. Dark corners — the characteristic circular vignette of a fundus camera.
  3. Warm (red-orange) dominant colour — retinal tissue is highly vascular.

All checks run in CPU memory and complete in < 2 ms on a modern server.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger(__name__)

# ── Tuneable thresholds ────────────────────────────────────────────────────────
_MAX_ASPECT_RATIO = 1.6    # max(w, h) / min(w, h) must be ≤ this
_MIN_DIM          = 64     # reject images smaller than 64 × 64
_CORNER_FRACTION  = 0.10   # sample this fraction of each edge for corners
_MAX_CORNER_MEAN  = 100    # mean corner brightness (0-255) must be ≤ this


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str = ""


def validate_upload_type(content_type: str, allowed: frozenset[str]) -> None:
    """Raise HTTP 400 if content_type is not in the allowed set."""
    if content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Accepted types: {sorted(allowed)}"
            ),
        )


def validate_upload_size(
    file_bytes: bytes,
    filename: str,
    max_mb: int,
) -> None:
    """Raise HTTP 400 (empty) or HTTP 413 (too large)."""
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}': uploaded file is empty.",
        )
    max_bytes = max_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"'{filename}' exceeds the {max_mb} MB upload limit "
                f"({len(file_bytes) / 1_048_576:.1f} MB)."
            ),
        )


def validate_fundus_image(file_bytes: bytes, filename: str) -> None:
    """
    Heuristic fundus-image gate.

    Raises HTTPException 422 with a human-readable explanation if the image
    does not look like a genuine retinal fundus photograph.

    Args:
        file_bytes: Raw bytes of the uploaded image.
        filename:   Original filename (used only in error messages).
    """
    result = _check_fundus(file_bytes)
    if not result.passed:
        raise HTTPException(
            status_code=422,
            detail=(
                "This does not appear to be a retinal fundus photograph. "
                f"Reason: {result.reason}. "
                "Please upload an image captured with a fundus camera or "
                "ophthalmoscope. Documents, selfies, and general photographs "
                "are not supported."
            ),
        )


# ── Internal checks ────────────────────────────────────────────────────────────

def _check_fundus(file_bytes: bytes) -> ValidationResult:
    """Run all heuristic checks and return the first failure (or success)."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        return ValidationResult(passed=False, reason=f"cannot decode image ({exc})")

    w, h = img.size

    # 1. Minimum size
    if w < _MIN_DIM or h < _MIN_DIM:
        return ValidationResult(
            passed=False,
            reason=f"image is {w}×{h} px — too small to be a fundus photograph",
        )

    # 2. Aspect ratio
    ratio = max(w, h) / min(w, h)
    if ratio > _MAX_ASPECT_RATIO:
        return ValidationResult(
            passed=False,
            reason=(
                f"aspect ratio {ratio:.2f}:1 exceeds threshold {_MAX_ASPECT_RATIO}:1 "
                "— fundus images are roughly square"
            ),
        )

    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    H, W = arr.shape[:2]
    cs   = max(int(min(H, W) * _CORNER_FRACTION), 5)

    # 3. Corner darkness — fundus cameras produce a dark vignette ring
    corner_patches = [
        arr[:cs,   :cs  ],   # top-left
        arr[:cs,   W-cs:],   # top-right
        arr[H-cs:, :cs  ],   # bottom-left
        arr[H-cs:, W-cs:],   # bottom-right
    ]
    mean_corner = float(np.mean([p.mean() for p in corner_patches]))
    if mean_corner > _MAX_CORNER_MEAN:
        return ValidationResult(
            passed=False,
            reason=(
                f"corners are too bright (mean intensity {mean_corner:.1f}/255, "
                f"threshold {_MAX_CORNER_MEAN}) — "
                "genuine fundus images have a dark circular border"
            ),
        )

    # 4. Red-channel dominance (retinal tissue is warm / red-orange)
    mean_r = float(arr[:, :, 0].mean())
    mean_b = float(arr[:, :, 2].mean())
    if mean_r < mean_b:
        return ValidationResult(
            passed=False,
            reason=(
                f"colour profile (R={mean_r:.1f}, B={mean_b:.1f}) is blue-dominant "
                "— retinal fundus images should be red/orange-dominant"
            ),
        )

    logger.debug(
        "Fundus validation passed | %dx%d | ratio=%.2f | corner=%.1f | R=%.1f B=%.1f",
        w, h, ratio, mean_corner, mean_r, mean_b,
    )
    return ValidationResult(passed=True)
