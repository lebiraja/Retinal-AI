"""
Model Service — internal Pydantic schemas.

These are the strict contracts between the model service and its callers.
The backend service depends on these exact field names.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """Base64-encoded image payload sent by the backend service."""

    image_b64: str = Field(
        ...,
        description="Base64-encoded raw image bytes (JPEG / PNG / BMP).",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sigmoid probability threshold for positive disease detection.",
    )


# ── Response ───────────────────────────────────────────────────────────────────

class InferenceResponse(BaseModel):
    """Full inference result returned by the model service."""

    probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Raw sigmoid probabilities for ALL 45 disease classes, "
            "regardless of threshold. Keys are disease short codes, "
            "values are in [0.0, 1.0]."
        ),
    )
    detected_labels: List[str] = Field(
        ...,
        description="Disease labels whose probability >= threshold.",
    )
    elapsed_ms: float = Field(
        ...,
        description="Wall-clock GPU/CPU forward-pass time in milliseconds.",
    )


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when service is operational.")
    model_loaded: bool = Field(..., description="True if model weights are in memory.")
    device: str = Field(..., description="Torch device string, e.g. 'cuda:0' or 'cpu'.")
    model_repo: str = Field(..., description="HuggingFace repository identifier.")
