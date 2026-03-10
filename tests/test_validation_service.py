"""
tests/test_validation_service.py

Unit tests for app.services.validation_service.validate_fundus_image.

These tests do NOT require a GPU or the ML model — they only test the
lightweight heuristic gate that rejects non-retinal images.

Run with:
    pytest tests/ -v
"""

import io

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image, ImageDraw

from app.services.validation_service import validate_fundus_image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_fundus_like(size: int = 400) -> Image.Image:
    """
    Synthetic fundus-like image:
      • Square
      • Dark corners (black background)
      • Warm/red circular centre
    """
    img  = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    r    = size // 2 - 10
    cx, cy = size // 2, size // 2
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(180, 80, 40),
    )
    return img


def _make_screenshot_like(w: int = 1920, h: int = 1080) -> Image.Image:
    """Wide, bright, non-warm image — like a desktop screenshot."""
    arr = np.full((h, w, 3), fill_value=200, dtype=np.uint8)
    arr[:, :, 2] = 220   # blue-toned (typical UI)
    arr[:, :, 0] = 160
    return Image.fromarray(arr, "RGB")


def _make_bright_corners(size: int = 400) -> Image.Image:
    """Square image with bright (white) corners — not a fundus photo."""
    return Image.new("RGB", (size, size), (230, 230, 230))


def _make_blue_square(size: int = 400) -> Image.Image:
    """Square image with blue dominating — fails colour check."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 2] = 200   # blue
    arr[:, :, 0] = 80    # weak red
    return Image.fromarray(arr, "RGB")


# ── Should PASS ───────────────────────────────────────────────────────────────

class TestValidPassCases:

    def test_fundus_like_jpeg_passes(self):
        img = _make_fundus_like(400)
        validate_fundus_image(_to_bytes(img, "JPEG"), "fundus.jpg")

    def test_fundus_like_png_passes(self):
        img = _make_fundus_like(512)
        validate_fundus_image(_to_bytes(img, "PNG"), "fundus.png")

    def test_slightly_non_square_passes(self):
        """Aspect ratio ~1.3:1 is within tolerance."""
        base = _make_fundus_like(400)
        arr  = np.array(base.resize((480, 380)))
        # Ensure corners are black
        cs = 38
        for r, c in [
            (slice(None, cs),  slice(None, cs)),
            (slice(None, cs),  slice(-cs, None)),
            (slice(-cs, None), slice(None, cs)),
            (slice(-cs, None), slice(-cs, None)),
        ]:
            arr[r, c] = (0, 0, 0)
        validate_fundus_image(_to_bytes(Image.fromarray(arr)), "slightly_rect.jpg")


# ── Should FAIL ───────────────────────────────────────────────────────────────

class TestValidFailCases:

    def test_wide_screenshot_rejected(self):
        img = _make_screenshot_like(1920, 1080)
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "screenshot.jpg")
        assert exc.value.status_code == 422
        assert "aspect ratio" in exc.value.detail.lower()

    def test_portrait_screenshot_rejected(self):
        img = _make_screenshot_like(1080, 1920)
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "portrait.jpg")
        assert exc.value.status_code == 422

    def test_bright_corners_rejected(self):
        img = _make_bright_corners(400)
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "bright.jpg")
        assert exc.value.status_code == 422
        assert "corner" in exc.value.detail.lower()

    def test_blue_dominant_image_rejected(self):
        img = _make_blue_square(400)
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "blue.jpg")
        assert exc.value.status_code == 422
        assert "colour" in exc.value.detail.lower()

    def test_tiny_image_rejected(self):
        img = Image.new("RGB", (30, 30), (180, 80, 40))
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "tiny.jpg")
        assert exc.value.status_code == 422
        assert "too small" in exc.value.detail.lower()

    def test_corrupt_bytes_rejected(self):
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(b"not an image", "garbage.jpg")
        assert exc.value.status_code == 422

    def test_empty_bytes_rejected(self):
        with pytest.raises((HTTPException, Exception)):
            validate_fundus_image(b"", "empty.jpg")


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_just_inside_aspect_limit_passes(self):
        """400×252 → ratio ≈ 1.587, inside the 1.6 limit."""
        arr = np.zeros((252, 400, 3), dtype=np.uint8)
        cs  = 25
        # Centre region warm/red
        arr[cs:-cs, cs:-cs, 0] = 160
        arr[cs:-cs, cs:-cs, 1] = 60
        arr[cs:-cs, cs:-cs, 2] = 20
        validate_fundus_image(_to_bytes(Image.fromarray(arr)), "borderline.jpg")

    def test_just_over_aspect_limit_fails(self):
        """400×248 → ratio ≈ 1.613, just over the 1.6 limit."""
        arr = np.zeros((248, 400, 3), dtype=np.uint8)
        arr[:, :, 0] = 150   # red dominant, corners dark — only aspect fails
        arr[:, :, 2] = 50
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(Image.fromarray(arr)), "over_limit.jpg")
        assert exc.value.status_code == 422

    def test_filename_in_error_message(self):
        img = _make_screenshot_like()
        with pytest.raises(HTTPException) as exc:
            validate_fundus_image(_to_bytes(img), "my_screenshot.png")
        assert "my_screenshot.png" in exc.value.detail
