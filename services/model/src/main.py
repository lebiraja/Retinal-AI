"""
Model Service — FastAPI application.

This service owns ALL GPU resources.  It exposes a minimal, internal-only HTTP
API consumed by the backend service.  It is never exposed directly to the
internet — only through the internal Docker network.

Endpoints:
    POST /inference   — run forward pass on a base64-encoded image
    GET  /health      — liveness probe + model status
    GET  /ready       — readiness probe (only OK after model is fully loaded)
    GET  /metrics     — lightweight counters (requests, errors, mean latency)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Deque

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from services.model.src import config
from services.model.src.model import load_checkpoint
from services.model.src.preprocess import preprocess_b64
from services.model.src.schemas import HealthResponse, InferenceRequest, InferenceResponse

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── In-process metrics (lightweight, no Prometheus dependency) ─────────────────
class _Metrics:
    def __init__(self, window: int = 100) -> None:
        self.total_requests: int = 0
        self.total_errors: int = 0
        self._latencies: Deque[float] = deque(maxlen=window)

    def record(self, elapsed_ms: float, error: bool = False) -> None:
        self.total_requests += 1
        if error:
            self.total_errors += 1
        else:
            self._latencies.append(elapsed_ms)

    @property
    def mean_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p95_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lats = sorted(self._latencies)
        idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        return sorted_lats[idx]


_metrics = _Metrics()


# ── Global model state ─────────────────────────────────────────────────────────
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_ready: asyncio.Event


def _get_model() -> torch.nn.Module:
    if _model is None:
        raise RuntimeError("Model is not loaded yet.")
    return _model


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _model_ready
    _model_ready = asyncio.Event()

    logger.info("Device: %s", _device)
    if _device.type == "cuda":
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Load model in a thread pool so the event loop stays responsive during
    # the potentially multi-second HuggingFace download.
    loop = asyncio.get_event_loop()
    _model = await loop.run_in_executor(None, _load_model_sync)
    _model_ready.set()
    logger.info("Model loaded and ready on %s.", _device)

    yield

    logger.info("Shutting down model service.")
    if _device.type == "cuda":
        torch.cuda.empty_cache()


def _load_model_sync() -> torch.nn.Module:
    """Blocking model load — run in executor to avoid blocking the event loop."""
    # Prefer a locally mounted checkpoint over HuggingFace download so that
    # the container works fully air-gapped if a volume mount is provided.
    if config.LOCAL_CHECKPOINT_PATH and os.path.isfile(config.LOCAL_CHECKPOINT_PATH):
        logger.info("Using local checkpoint: %s", config.LOCAL_CHECKPOINT_PATH)
        return load_checkpoint(config.LOCAL_CHECKPOINT_PATH, _device)

    logger.info(
        "Downloading checkpoint from HuggingFace: %s / %s",
        config.HF_MODEL_REPO,
        config.HF_MODEL_FILE,
    )
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.HF_MODEL_FILE,
        cache_dir=config.HF_CACHE_DIR,
    )
    return load_checkpoint(checkpoint_path, _device)


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retinal Disease — Model Service",
    version="1.0.0",
    description=(
        "Internal GPU inference microservice for EfficientNet-B4. "
        "Not exposed to the public internet — call via the backend service."
    ),
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)


# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def _add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/inference",
    response_model=InferenceResponse,
    summary="Run model inference on a base64-encoded fundus image",
)
async def inference(payload: InferenceRequest) -> InferenceResponse:
    """
    Accepts a base64-encoded image and returns raw sigmoid probabilities for
    all 45 retinal disease classes plus a list of labels above the threshold.
    """
    model = _get_model()

    try:
        tensor = preprocess_b64(payload.image_b64)
    except ValueError as exc:
        _metrics.record(0.0, error=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tensor = tensor.to(_device)

    t0 = time.perf_counter()
    try:
        # Pin to a single inference at a time on GPU to prevent OOM races
        # when multiple requests arrive simultaneously.
        with torch.no_grad():
            logits: torch.Tensor = model(tensor)                         # [1, 45]
            probs: list[float] = (
                torch.sigmoid(logits)[0].cpu().tolist()
            )                                                             # [45]
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        _metrics.record(0.0, error=True)
        logger.error("GPU OOM during inference — cache cleared.")
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory. Retry in a moment.",
        ) from exc
    except Exception as exc:
        _metrics.record(0.0, error=True)
        logger.exception("Unexpected inference error.")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _metrics.record(elapsed_ms)

    probabilities = {
        label: round(prob, 6)
        for label, prob in zip(config.LABEL_COLS, probs)
    }
    detected_labels = [
        label
        for label, prob in probabilities.items()
        if prob >= payload.threshold
    ]

    logger.info(
        "Inference | %.1f ms | threshold=%.2f | %d/%d diseases detected",
        elapsed_ms,
        payload.threshold,
        len(detected_labels),
        config.NUM_CLASSES,
    )

    return InferenceResponse(
        probabilities=probabilities,
        detected_labels=detected_labels,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=str(_device),
        model_repo=config.HF_MODEL_REPO,
    )


@app.get(
    "/ready",
    summary="Readiness probe — only 200 after model is fully loaded",
)
async def ready() -> JSONResponse:
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model is still loading."},
        )
    return JSONResponse(content={"status": "ready"})


@app.get(
    "/metrics",
    summary="Lightweight request counters and latency statistics",
)
async def metrics() -> JSONResponse:
    cuda_info: dict = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved  = torch.cuda.memory_reserved(0)
        cuda_info = {
            "gpu_name":       props.name,
            "vram_total_gb":  round(props.total_memory / 1e9, 2),
            "vram_allocated_gb": round(allocated / 1e9, 3),
            "vram_reserved_gb":  round(reserved  / 1e9, 3),
        }

    return JSONResponse(
        content={
            "total_requests":  _metrics.total_requests,
            "total_errors":    _metrics.total_errors,
            "mean_latency_ms": round(_metrics.mean_latency_ms, 2),
            "p95_latency_ms":  round(_metrics.p95_latency_ms, 2),
            "device":          str(_device),
            **cuda_info,
        }
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.model.src.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL,
        workers=config.WORKERS,
    )
