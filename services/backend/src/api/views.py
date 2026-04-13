"""
Django async views — replaces services/backend/src/routers/predict.py (FastAPI).

All four endpoints are pure async Django views (Django 4.1+ CBV async support).
They run under Gunicorn + UvicornWorker (ASGI) so asyncio.gather() works natively.

Endpoint map (Nginx strips /api/ prefix):
  FastAPI route          →  Django view
  POST /predict          →  PredictView.post
  POST /predict-batch    →  PredictBatchView.post
  GET  /health           →  HealthView.get
  GET  /info             →  InfoView.get

All service imports are framework-agnostic (plain asyncio / httpx) — zero changes
needed in model_client.py, advisory_service.py, vlm_service.py, validation_service.py.
"""
import asyncio
import logging
import time

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

# ── Internal service imports (framework-agnostic, unchanged from FastAPI version) ──
from config import (
    DEFAULT_THRESHOLD,
    MAX_FILE_SIZE_MB,
    MAX_BATCH_SIZE,
    MODEL_SERVICE_URL,
    MODEL_VERSION,
    HF_MODEL_REPO,
)
from services.advisory_service import generate_advisory
from services.model_client import ModelClient
from services.validation_service import validate_upload_size
from services.vlm_service import (
    VLMUnavailableError,
    check_is_eye_image,
    generate_analysis,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DISCLAIMER = (
    "This tool is NOT a medical diagnostic device. "
    "Results are for screening purposes only. "
    "Always consult a qualified ophthalmologist for clinical decisions."
)

DISEASE_LABELS: list[str] = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CLT",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_threshold(request) -> tuple[float | None, JsonResponse | None]:
    """Return (threshold, None) or (None, error_response)."""
    raw = request.GET.get("threshold", str(DEFAULT_THRESHOLD))
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None, JsonResponse(
            {"detail": f"'threshold' must be a float, got: {raw!r}"},
            status=422,
        )
    if not 0.0 <= value <= 1.0:
        return None, JsonResponse(
            {"detail": "threshold must be in the range [0.0, 1.0]"},
            status=422,
        )
    return value, None


def _build_response(
    advisory: dict,
    infer: dict,
    threshold: float,
    vlm_text: str | None,
) -> dict:
    """Assemble the final PredictionResponse dict (identical structure to FastAPI version)."""
    advisory_text = vlm_text if vlm_text else advisory["advisory"]
    return {
        "disease_risk":           advisory["disease_risk"],
        "predictions":            infer["probabilities"],
        "detected_diseases":      advisory["detected_diseases"],
        "detected_diseases_full": advisory["detected_diseases_full"],
        "num_detected":           advisory["num_detected"],
        "top_prediction":         advisory["top_prediction"],
        "confidence":             advisory["confidence"],
        "risk_level":             advisory["risk_level"],
        "advisory":               advisory_text,
        "elapsed_ms":             infer["elapsed_ms"],
        "threshold":              threshold,
        "disclaimer":             DISCLAIMER,
    }


async def _run_prediction(file_bytes: bytes, content_type: str, filename: str, threshold: float) -> dict:
    """
    Core prediction pipeline — identical logic to FastAPI _process_single_image().

    Raises:
        ValueError          – file validation failed (400 / 413)
        VLMUnavailableError – VLM gate down (503)
        RuntimeError        – "not_eye" sentinel for 422
        Exception           – model inference failed (502)
    """
    # 1. Validate size
    validate_upload_size(file_bytes, filename, MAX_FILE_SIZE_MB)

    # 2. VLM gate — FAIL-CLOSED
    is_eye = await check_is_eye_image(file_bytes, content_type)
    if not is_eye:
        raise RuntimeError("not_eye")

    # 3. Model inference
    infer_result = await ModelClient.infer(file_bytes, threshold)

    # 4. Advisory (pure CPU, sync)
    above_threshold = {
        label: prob
        for label, prob in infer_result["probabilities"].items()
        if prob >= threshold
    }
    advisory_result = generate_advisory(above_threshold)

    # 5. VLM advisory enrichment — FAIL-OPEN
    vlm_text: str | None = None
    try:
        vlm_text = await generate_analysis(
            file_bytes,
            content_type,
            advisory_result.get("detected_diseases_full", []),
            advisory_result["risk_level"],
            infer_result["probabilities"],
        )
    except Exception as exc:
        logger.debug("VLM advisory failed (fail-open): %s", exc)

    return _build_response(advisory_result, infer_result, threshold, vlm_text)


# ── Views ──────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class PredictView(View):
    """POST /predict — single fundus image prediction."""

    async def post(self, request):
        # 1. Parse threshold
        threshold, err = _parse_threshold(request)
        if err:
            return err

        # 2. Get uploaded file
        if "image" not in request.FILES:
            return JsonResponse(
                {"detail": 'Field "image" is required (multipart/form-data).'},
                status=400,
            )

        uploaded = request.FILES["image"]
        file_bytes: bytes = uploaded.read()
        content_type: str = uploaded.content_type or "image/jpeg"
        filename: str = uploaded.name or "image.jpg"

        # 3. Run pipeline
        try:
            result = await _run_prediction(file_bytes, content_type, filename, threshold)
        except ValueError as exc:
            # validate_upload_size raises ValueError with the message
            status = 413 if "too large" in str(exc).lower() else 400
            return JsonResponse({"detail": str(exc)}, status=status)
        except VLMUnavailableError:
            logger.warning("VLM gate unavailable — returning 503")
            return JsonResponse(
                {"detail": "Image validation service temporarily unavailable. Please try again shortly."},
                status=503,
            )
        except RuntimeError as exc:
            if str(exc) == "not_eye":
                return JsonResponse(
                    {"detail": "Image does not appear to be a retinal fundus photograph. "
                               "Please upload a valid fundus image."},
                    status=422,
                )
            logger.exception("Unexpected RuntimeError in predict")
            return JsonResponse({"detail": "Internal server error."}, status=500)
        except Exception:
            logger.exception("Model inference failed")
            return JsonResponse(
                {"detail": "Model inference failed. Please try again."},
                status=502,
            )

        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class PredictBatchView(View):
    """POST /predict-batch — up to MAX_BATCH_SIZE fundus images in one request."""

    async def post(self, request):
        # 1. Parse threshold
        threshold, err = _parse_threshold(request)
        if err:
            return err

        # 2. Collect uploaded files
        files = request.FILES.getlist("images")
        if not files:
            return JsonResponse(
                {"detail": 'Field "images" is required and must contain at least one file.'},
                status=400,
            )
        if len(files) > MAX_BATCH_SIZE:
            return JsonResponse(
                {
                    "detail": (
                        f"Maximum {MAX_BATCH_SIZE} images per batch request. "
                        f"Received {len(files)}."
                    )
                },
                status=422,
            )

        start = time.perf_counter()

        # 3. Read all bytes synchronously before going async (avoids event-loop blocking later)
        batch = [
            (f.read(), f.content_type or "image/jpeg", f.name or f"image_{i}.jpg")
            for i, f in enumerate(files)
        ]

        # 4. Process all images concurrently — same as FastAPI asyncio.gather()
        async def _process_one(file_bytes: bytes, content_type: str, filename: str) -> dict:
            try:
                return await _run_prediction(file_bytes, content_type, filename, threshold)
            except ValueError as exc:
                return {"error": str(exc), "status": 413 if "too large" in str(exc).lower() else 400}
            except VLMUnavailableError:
                return {"error": "VLM gate unavailable", "status": 503}
            except RuntimeError as exc:
                if str(exc) == "not_eye":
                    return {"error": "Not a retinal fundus image", "status": 422}
                return {"error": "Internal error", "status": 500}
            except Exception as exc:
                logger.error("Batch item inference failed: %s", exc)
                return {"error": "Model inference failed", "status": 502}

        tasks = [_process_one(fb, ct, fn) for fb, ct, fn in batch]
        results = await asyncio.gather(*tasks)

        total_ms = round((time.perf_counter() - start) * 1000, 2)

        return JsonResponse(
            {
                "results":          list(results),
                "total_images":     len(files),
                "total_elapsed_ms": total_ms,
            }
        )


class HealthView(View):
    """GET /health — liveness + model-service reachability."""

    async def get(self, request):
        reachable = await ModelClient.is_reachable()
        return JsonResponse(
            {
                "status":                  "ok",
                "model_service_reachable": reachable,
                "model_service_url":       MODEL_SERVICE_URL,
                "version":                 MODEL_VERSION,
            }
        )


class InfoView(View):
    """GET /info — static model metadata."""

    async def get(self, request):
        return JsonResponse(
            {
                "model_name":  "Retinal Disease Classifier",
                "version":     MODEL_VERSION,
                "architecture": "EfficientNet-B4 + Dropout(0.4) + Linear(1792, 45)",
                "num_classes": 45,
                "input_size":  384,
                "diseases":    DISEASE_LABELS,
                "metrics": {
                    "mean_auc":  0.8204,
                    "train_loss": 0.2118,
                    "val_loss":   0.2578,
                    "macro_f1":   0.1517,
                    "micro_f1":   0.4450,
                },
                "huggingface": f"https://huggingface.co/{HF_MODEL_REPO}",
            }
        )
