"""
ValidationService — heuristic check to reject non-retinal images.

Fundus (retinal) photographs have three distinctive properties:
  1. Nearly square aspect ratio (ophthalmoscope cameras are circular apertures).
  2. Dark / near-black corners — the circular vignette of the camera lens.
  3. Warm colour profile — retinal tissue is predominantly red-orange.

These three signals together reliably reject screenshots, random photos,
and other non-fundus images while accepting real retinal photographs.

This is NOT a deep-learning classifier; it is a lightweight sanity gate
that runs in < 2 ms and prevents the EfficientNet model from producing
meaningless output on clearly invalid inputs.
"""

import io
import logging

import numpy as np
from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger(__name__)

# ── Tuneable thresholds ───────────────────────────────────────────────────────
_MAX_ASPECT_RATIO   = 1.6   # reject if width/height (or height/width) > this
_MAX_CORNER_MEAN    = 100   # reject if avg corner pixel brightness > this (0-255)
_MIN_IMAGE_DIM      = 64    # reject tiny images (probably not fundus scans)
_CORNER_FRACTION    = 0.10  # sample this fraction of each dimension for corners


def validate_fundus_image(file_bytes: bytes, filename: str) -> None:
    """
    Raise HTTP 422 if the image does not look like a retinal fundus photograph.

    Checks (any single failure triggers rejection):
      • Minimum size
      • Aspect ratio (fundus images are roughly square)
      • Corner darkness (fundus cameras produce a dark circular vignette)
      • Red-channel dominance (retinal tissue is warm-toned)

    Args:
        file_bytes: Raw bytes of the uploaded image.
        filename:   Original filename, used only in error messages.

    Raises:
        HTTPException 422: If the image fails the fundus validation.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"'{filename}': could not decode image.",
        )

    w, h = img.size

    # ── 1. Minimum size ───────────────────────────────────────────────── #
    if w < _MIN_IMAGE_DIM or h < _MIN_IMAGE_DIM:
        _reject(filename, "image is too small to be a fundus photograph")

    # ── 2. Aspect ratio ───────────────────────────────────────────────── #
    ratio = max(w, h) / min(w, h)
    if ratio > _MAX_ASPECT_RATIO:
        _reject(
            filename,
            f"aspect ratio {ratio:.2f}:1 is too wide/tall — "
            "fundus images are roughly square",
        )

    # ── 3. Corner darkness ────────────────────────────────────────────── #
    arr         = np.array(img, dtype=np.float32)  # (H, W, 3)
    H, W        = arr.shape[:2]
    cs          = max(int(min(H, W) * _CORNER_FRACTION), 5)

    corners = [
        arr[:cs,    :cs   ],   # top-left
        arr[:cs,    W-cs: ],   # top-right
        arr[H-cs:,  :cs   ],   # bottom-left
        arr[H-cs:,  W-cs: ],   # bottom-right
    ]
    mean_corner = float(np.mean([c.mean() for c in corners]))

    if mean_corner > _MAX_CORNER_MEAN:
        _reject(
            filename,
            f"image corners are too bright (mean={mean_corner:.1f}) — "
            "fundus photos have dark corners from the camera aperture",
        )

    # ── 4. Red-channel dominance ──────────────────────────────────────── #
    mean_r = float(arr[:, :, 0].mean())
    mean_b = float(arr[:, :, 2].mean())

    if mean_r < mean_b:
        _reject(
            filename,
            f"colour profile (R={mean_r:.1f}, B={mean_b:.1f}) does not match "
            "retinal tissue — fundus images are warm/red-toned",
        )

    logger.debug(
        "Fundus validation passed | %s | ratio=%.2f | corner=%.1f | R=%.1f B=%.1f",
        filename, ratio, mean_corner, mean_r, mean_b,
    )


def _reject(filename: str, reason: str) -> None:
    msg = (
        f"'{filename}' does not appear to be a retinal fundus photograph "
        f"({reason}). Please upload a genuine fundus/retinal scan image."
    )
    logger.warning("Fundus validation failed | %s | %s", filename, reason)
    raise HTTPException(status_code=422, detail=msg)
