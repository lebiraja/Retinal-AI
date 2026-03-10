"""
Backend API — prediction endpoints.

Pipeline for each image:
  1. Size check (max upload limit)
  2. VLM stage 1 — silent eye-image gate
  3. Model inference (EfficientNet-B4)
  4. Advisory generation (risk level + base text)
  5. VLM stage 2 — personalised analysis overlaid on advisory
  6. Response

The VLM integration is completely transparent to the user.
Any image format can be uploaded; the VLM decides whether it is retinal.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from services.backend.src.config import (
    ARCHITECTURE,
    BEST_AUC,
    BEST_EPOCH,
    DEFAULT_THRESHOLD,
    DISEASE_LABELS,
    HF_MODEL_NAME,
    IMAGE_SIZE,
    MACRO_F1,
    MAX_BATCH_SIZE,
    MAX_FILE_SIZE_MB,
    MICRO_F1,
    MODEL_SERVICE_URL,
    MODEL_VERSION,
    NUM_CLASSES,
    TRAIN_LOSS,
    VAL_LOSS,
    DISEASE_FULL_NAMES,
)
from services.backend.src.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetrics,
    PredictionResponse,
)
from services.backend.src.services.advisory_service import generate_advisory
from services.backend.src.services.model_client import ModelClient
from services.backend.src.services.validation_service import (
    validate_upload_size,
)
from services.backend.src.services import vlm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retinal Classifier"])


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


async def _process_single_image(
    image: UploadFile,
    threshold: float,
    request_id: str,
) -> PredictionResponse:
    """
    Full pipeline for one image:
      size check → VLM gate → model inference → advisory → VLM analysis → response.
    """
    t_start = time.perf_counter()

    # ── 1. Read bytes and validate size only ──────────────────────────────────
    file_bytes   = await image.read()
    content_type = image.content_type or "image/jpeg"
    validate_upload_size(file_bytes, image.filename or "upload", MAX_FILE_SIZE_MB)

    # ── 2. VLM stage 1 — silent eye-image gate (log only, non-blocking) ─────
    is_eye = await vlm_service.check_is_eye_image(file_bytes, content_type)
    if not is_eye:
        logger.info(
            "[%s] VLM gate returned non-eye for file=%s — proceeding anyway",
            request_id,
            image.filename,
        )

    # ── 3. Model inference ────────────────────────────────────────────────────
    try:
        inference = await ModelClient.infer(file_bytes, threshold=threshold)
    except Exception as exc:
        logger.error("[%s] Model service error: %s", request_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "The model service is temporarily unavailable. "
                "Please try again shortly."
            ),
        ) from exc

    predictions_above_threshold: dict[str, float] = {
        label: prob
        for label, prob in inference["probabilities"].items()
        if label in inference["detected_labels"]
    }

    # ── 4. Structure advisory (risk level + base text) ────────────────────────
    advisory_result = generate_advisory(predictions_above_threshold)

    # ── 5. VLM stage 2 — personalised image-grounded analysis ────────────────
    disease_full_names = [
        DISEASE_FULL_NAMES.get(d, d)
        for d in advisory_result["detected_diseases"]
    ]
    vlm_analysis = await vlm_service.generate_analysis(
        image_bytes=file_bytes,
        content_type=content_type,
        disease_full_names=disease_full_names,
        risk_level=advisory_result["risk_level"],
        probabilities=predictions_above_threshold,
    )
    # Use VLM text when available; fall back to static advisory on failure
    final_advisory = vlm_analysis if vlm_analysis else advisory_result["advisory"]

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        "[%s] predict | file=%s | threshold=%.2f | risk=%s "
        "| detected=%d | top=%s | conf=%.3f | %.1f ms",
        request_id,
        image.filename,
        threshold,
        advisory_result["risk_level"],
        advisory_result["num_detected"],
        advisory_result["top_prediction"],
        advisory_result["confidence"],
        elapsed_ms,
    )

    return PredictionResponse(
        disease_risk=advisory_result["disease_risk"],
        predictions=advisory_result["predictions"],
        detected_diseases=advisory_result["detected_diseases"],
        detected_diseases_full=advisory_result["detected_diseases_full"],
        num_detected=advisory_result["num_detected"],
        top_prediction=advisory_result["top_prediction"],
        confidence=advisory_result["confidence"],
        risk_level=advisory_result["risk_level"],
        advisory=final_advisory,
        elapsed_ms=round(elapsed_ms, 2),
        threshold=threshold,
    )


# ── POST /predict ────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify retinal diseases in an eye image",
    description=(
        "Upload a retinal or eye photograph (any common image format, max 20 MB). "
        "Returns probabilities for 45 disease classes, an overall risk level, "
        "and a personalised advisory message.\n\n"
        "**⚠ Not a medical diagnostic device. "
        "Always consult a qualified ophthalmologist.**"
    ),
)
async def predict(
    request: Request,
    image: UploadFile = File(
        ..., description="Eye or retinal image (any format, max 20 MB)"
    ),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Sigmoid probability threshold for disease detection. "
            "Lower = more sensitive, higher = more specific. Default: 0.5"
        ),
    ),
) -> PredictionResponse:
    req_id = _new_request_id()
    return await _process_single_image(image, threshold, req_id)


# ── POST /predict-batch ────────────────────────────────────────────────────────────────────

@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Classify diseases in multiple eye images",
    description=(
        f"Upload up to {MAX_BATCH_SIZE} eye images in one request. "
        "Each image is processed independently with the same threshold.\n\n"
        "**⚠ Not a medical diagnostic device.**"
    ),
)
async def predict_batch(
    request: Request,
    images: list[UploadFile] = File(
        ..., description=f"List of eye images (max {MAX_BATCH_SIZE} per request)"
    ),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
    ),
) -> BatchPredictionResponse:
    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_SIZE} images per batch request.",
        )

    req_id  = _new_request_id()
    t_start = time.perf_counter()

    tasks = [
        _process_single_image(img, threshold, f"{req_id}-{i}")
        for i, img in enumerate(images)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses: list[PredictionResponse] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("[%s] Batch item %d failed: %s", req_id, i, result)
            raise HTTPException(
                status_code=502,
                detail=f"Image {i + 1} failed: {result}",
            )
        responses.append(result)  # type: ignore[arg-type]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info("[%s] predict-batch | n=%d | %.1f ms", req_id, len(images), total_ms)

    return BatchPredictionResponse(
        results=responses,
        total_images=len(responses),
        total_elapsed_ms=round(total_ms, 2),
    )


# ── GET /health ────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + model service reachability probe",
)
async def health() -> HealthResponse:
    reachable = await ModelClient.is_reachable()
    return HealthResponse(
        status="ok",
        model_service_reachable=reachable,
        model_service_url=MODEL_SERVICE_URL,
        version=MODEL_VERSION,
    )


# ── GET /info ────────────────────────────────────────────────────────────────────────

@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model metadata and performance metrics",
)
async def info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="Retinal Disease Classifier",
        version=MODEL_VERSION,
        architecture=ARCHITECTURE,
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
