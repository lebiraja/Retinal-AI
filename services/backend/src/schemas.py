"""
Backend Service Pydantic schemas — public API contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


_DISCLAIMER = (
    "This tool is NOT a medical diagnostic device. "
    "Results are for screening purposes only. "
    "Always consult a qualified ophthalmologist for clinical decisions."
)


# ── /predict ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Structured prediction result for a single fundus image."""

    disease_risk: bool = Field(
        ..., description="True if any disease was detected above threshold."
    )
    predictions: Dict[str, float] = Field(
        ..., description="Probability map for every detected disease (prob >= threshold)."
    )
    detected_diseases: List[str] = Field(
        ..., description="Short codes of detected diseases e.g. ['DR', 'BRVO']."
    )
    detected_diseases_full: List[str] = Field(
        ..., description="Human-readable names of detected diseases."
    )
    num_detected: int = Field(
        ..., description="Number of diseases detected."
    )
    top_prediction: Optional[str] = Field(
        None, description="Label of the highest-confidence detected disease."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Mean probability across all detected diseases (0.0 if none)."
    )
    risk_level: str = Field(
        ..., description="LOW | MODERATE | HIGH | CRITICAL"
    )
    advisory: str = Field(
        ..., description="Non-diagnostic advisory message for the patient."
    )
    elapsed_ms: float = Field(
        ..., description="End-to-end processing time in milliseconds."
    )
    threshold: float = Field(
        ..., description="Sigmoid threshold used for this prediction."
    )
    disclaimer: str = Field(default=_DISCLAIMER)


# ── /predict-batch ─────────────────────────────────────────────────────────────

class BatchPredictionResponse(BaseModel):
    """Prediction results for a batch of fundus images."""

    results: List[PredictionResponse] = Field(
        ..., description="Per-image prediction results."
    )
    total_images: int = Field(
        ..., description="Number of images processed."
    )
    total_elapsed_ms: float = Field(
        ..., description="Total wall-clock time for the full batch."
    )


# ── /health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_service_reachable: bool
    model_service_url: str
    version: str


# ── /info ──────────────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    mean_auc:   float
    train_loss: float
    val_loss:   float
    macro_f1:   float
    micro_f1:   float


class ModelInfoResponse(BaseModel):
    model_name:   str
    version:      str
    architecture: str
    num_classes:  int
    input_size:   int
    diseases:     List[str]
    metrics:      ModelMetrics
    huggingface:  str
