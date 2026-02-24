"""
Routers — all HTTP endpoints.

Endpoints:
    POST /predict        — single-image disease classification
    POST /predict-batch  — multi-image batch classification
    GET  /health         — liveness + model status
    GET  /info           — model metadata and performance metrics
"""

import logging
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import (
    ALLOWED_MIME_TYPES,
    BEST_AUC,
    BEST_EPOCH,
    DEFAULT_THRESHOLD,
    DISEASE_LABELS,
    HF_MODEL_NAME,
    IMAGE_SIZE,
    MACRO_F1,
    MAX_FILE_SIZE_MB,
    MICRO_F1,
    MODEL_VERSION,
    NUM_CLASSES,
    TRAIN_LOSS,
    VAL_LOSS,
)
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetrics,
    PredictionResponse,
)
from app.services.advisory_service import generate_advisory
from app.services.inference_service import run_inference
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retinal Classifier"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_upload(image: UploadFile) -> None:
    """Raise HTTP 400 for unsupported MIME types."""
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{image.content_type}'. "
                f"Accepted: {sorted(ALLOWED_MIME_TYPES)}"
            ),
        )


def _validate_bytes(file_bytes: bytes, filename: str) -> None:
    """Raise HTTP 400/413 for empty or oversized files."""
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail=f"'{filename}': empty file.")
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"'{filename}' exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )


def _validate_threshold(threshold: float) -> None:
    """Raise HTTP 422 for out-of-range thresholds."""
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=422,
            detail="threshold must be between 0.0 and 1.0.",
        )


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify retinal diseases in a fundus image",
    description=(
        "Upload a single fundus image (JPEG/PNG, max 10 MB). "
        "Returns 45-class disease probabilities, risk level, and advisory text.\n\n"
        "**⚠ Not a medical diagnostic device. "
        "Always consult a qualified ophthalmologist.**"
    ),
)
async def predict(
    image: UploadFile = File(..., description="Fundus image (JPEG/PNG/BMP, max 10 MB)"),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Sigmoid probability threshold for disease detection (0–1). "
            "Lower → more sensitive. Higher → more specific. Default: 0.5"
        ),
    ),
) -> PredictionResponse:
    _validate_upload(image)
    file_bytes = await image.read()
    _validate_bytes(file_bytes, image.filename or "upload")

    try:
        predictions, elapsed_ms = run_inference(file_bytes, threshold=threshold)
    except Exception as exc:
        logger.exception("Inference failed for '%s'", image.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        ) from exc

    result = generate_advisory(predictions)

    logger.info(
        "predict | file=%s | threshold=%.2f | risk=%s | diseases=%d "
        "| top=%s | conf=%.3f | %.1f ms",
        image.filename,
        threshold,
        result["risk_level"],
        result["num_detected"],
        result["top_prediction"],
        result["confidence"],
        elapsed_ms,
    )

    return PredictionResponse(
        **result,
        elapsed_ms=round(elapsed_ms, 2),
        threshold=threshold,
    )


# ── POST /predict-batch ───────────────────────────────────────────────────────

@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Classify diseases in multiple fundus images at once",
    description=(
        "Upload up to 10 fundus images in one request. "
        "Each image is processed independently with the same threshold.\n\n"
        "**⚠ Not a medical diagnostic device.**"
    ),
)
async def predict_batch(
    images: list[UploadFile] = File(
        ..., description="List of fundus images (max 10 per request)"
    ),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Detection threshold applied to every image in the batch.",
    ),
) -> BatchPredictionResponse:
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided.")
    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request.",
        )

    t_batch_start = time.perf_counter()
    results: list[PredictionResponse] = []

    for image in images:
        _validate_upload(image)
        file_bytes = await image.read()
        _validate_bytes(file_bytes, image.filename or "upload")

        try:
            predictions, elapsed_ms = run_inference(file_bytes, threshold=threshold)
        except Exception as exc:
            logger.exception("Batch inference failed for '%s'", image.filename)
            raise HTTPException(
                status_code=500,
                detail=f"Inference error on '{image.filename}': {exc}",
            ) from exc

        result = generate_advisory(predictions)
        results.append(
            PredictionResponse(
                **result,
                elapsed_ms=round(elapsed_ms, 2),
                threshold=threshold,
            )
        )
        logger.info(
            "batch item | file=%s | risk=%s | diseases=%d | %.1f ms",
            image.filename,
            result["risk_level"],
            result["num_detected"],
            elapsed_ms,
        )

    total_ms = (time.perf_counter() - t_batch_start) * 1000.0
    return BatchPredictionResponse(
        results=results,
        total_images=len(results),
        total_elapsed_ms=round(total_ms, 2),
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health check",
    description="Returns server status, model load state, and active compute device.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=ModelService.is_loaded(),
        device=ModelService.device_str(),
    )


# ── GET /info ─────────────────────────────────────────────────────────────────

@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model metadata and performance metrics",
    description=(
        "Returns the model architecture, supported disease classes, "
        "training metrics, and HuggingFace source."
    ),
)
async def info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="Retinal Disease Classifier",
        version=MODEL_VERSION,
        architecture="EfficientNet-B4 + Dropout(0.4) + Linear(1792, 45)",
        num_classes=NUM_CLASSES,
        input_size=IMAGE_SIZE,
        diseases=DISEASE_LABELS,
        metrics=ModelMetrics(
            mean_auc=BEST_AUC,
            train_loss=TRAIN_LOSS,
            val_loss=VAL_LOSS,
            macro_f1=MACRO_F1,
            micro_f1=MICRO_F1,
        ),
        huggingface=f"https://huggingface.co/{HF_MODEL_NAME}",
    )
