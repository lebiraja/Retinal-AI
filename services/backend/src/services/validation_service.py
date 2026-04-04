"""
ValidationService — lightweight upload pre-filters.

Only handles basic upload hygiene (size, type). Retinal image verification
is handled entirely by the VLM gate in vlm_service.py.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


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
