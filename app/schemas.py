"""
app/schemas.py — Pydantic request/response models.

All API response shapes are strictly typed here.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response schema for POST /predict and each item in POST /predict-batch."""

    disease_risk: bool = Field(
        ...,
        description="True if at least one disease was detected above threshold.",
    )
    predictions: Dict[str, float] = Field(
        ...,
        description="Probabilities for every disease detected above threshold.",
    )
    detected_diseases: List[str] = Field(
        ...,
        description="Short labels of all detected diseases e.g. ['DR', 'BRVO'].",
    )
    detected_diseases_full: List[str] = Field(
        ...,
        description="Human-readable names of all detected diseases.",
    )
    num_detected: int = Field(
        ...,
        description="Count of diseases detected above threshold.",
    )
    top_prediction: Optional[str] = Field(
        None,
        description="Short label of the highest-confidence disease. None when nothing detected.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Mean probability across all detected diseases. "
            "Represents overall confidence of the positive findings. "
            "0.0 when no disease is detected."
        ),
    )
    risk_level: str = Field(
        ...,
        description="LOW | MODERATE | HIGH | CRITICAL",
    )
    advisory: str = Field(
        ...,
        description="Non-diagnostic informational advisory message.",
    )
    elapsed_ms: float = Field(
        ...,
        description="Wall-clock model inference time in milliseconds.",
    )
    threshold: float = Field(
        ...,
        description="Sigmoid threshold used for this prediction.",
    )
    disclaimer: str = Field(
        default=(
            "This tool is NOT a medical diagnostic device. "
            "Results are for screening purposes only. "
            "Always consult a qualified ophthalmologist for clinical decisions."
        ),
        description="Fixed medical disclaimer.",
    )


# ── /predict-batch ────────────────────────────────────────────────────────────

class BatchPredictionResponse(BaseModel):
    """Response schema for POST /predict-batch."""

    results: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results, one per uploaded image.",
    )
    total_images: int = Field(
        ...,
        description="Number of images processed in this batch.",
    )
    total_elapsed_ms: float = Field(
        ...,
        description="Total wall-clock time for the full batch in milliseconds.",
    )


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., description="'ok' if the server is healthy.")
    model_loaded: bool = Field(..., description="True if the model is in memory.")
    device: str = Field(..., description="Compute device: 'cuda' or 'cpu'.")


# ── /info ─────────────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    mean_auc:   float
    train_loss: float
    val_loss:   float
    macro_f1:   float
    micro_f1:   float


class ModelInfoResponse(BaseModel):
    """Response schema for GET /info."""

    model_name:   str
    version:      str
    architecture: str
    num_classes:  int
    input_size:   int
    diseases:     List[str]
    metrics:      ModelMetrics
    huggingface:  str
