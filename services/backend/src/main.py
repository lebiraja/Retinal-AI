"""
Backend Service — FastAPI application entry point.

This service is the public-facing API gateway:
  • Accepts image uploads from the frontend
  • Validates images (type, size, fundus heuristics)
  • Delegates GPU inference to the model microservice via HTTP
  • Applies advisory logic and returns structured JSON responses

It has ZERO ML dependencies — torch is not installed here.

Public endpoints (via Nginx → /api/):
    POST /api/predict        — single-image classification
    POST /api/predict-batch  — multi-image batch
    GET  /api/health         — liveness + model service status
    GET  /api/info           — model metadata
    GET  /docs               — Swagger UI
    GET  /redoc              — ReDoc UI
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.backend.src.config import CORS_ORIGINS, HOST, LOG_LEVEL, PORT, WORKERS
from services.backend.src.routers.predict import router
from services.backend.src.services.model_client import ModelClient

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting backend service …")
    await ModelClient.startup()

    # Log readiness of downstream model service (non-blocking)
    reachable = await ModelClient.is_reachable()
    if reachable:
        logger.info("Model service is reachable at startup.")
    else:
        logger.warning(
            "Model service is NOT reachable at startup — "
            "inference requests will fail until it becomes available."
        )

    yield

    logger.info("Shutting down backend service …")
    await ModelClient.shutdown()


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Retinal Disease Classifier — Backend API",
    version="1.0.0",
    description=(
        "Production-grade API gateway for an EfficientNet-B4 retinal disease "
        "classifier trained on the RFMiD dataset. "
        "Detects **45 retinal conditions** with a mean AUC-ROC of **0.82**.\n\n"
        "---\n\n"
        "| Method | Path | Description |\n"
        "|--------|------|-------------|\n"
        "| POST | `/predict` | Single-image classification |\n"
        "| POST | `/predict-batch` | Batch classification (up to 10 images) |\n"
        "| GET  | `/health` | Liveness + model service status |\n"
        "| GET  | `/info` | Model metadata & performance metrics |\n\n"
        "---\n\n"
        "> **Medical Disclaimer:** This system is NOT a medical diagnostic device. "
        "Results are for screening and educational purposes only. "
        "Always consult a qualified ophthalmologist for clinical decisions."
    ),
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Retinal AI Team",
        "url": "https://huggingface.co/lebiraja/retinal-disease-classifier",
    },
    license_info={"name": "MIT"},
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def _request_id_and_timing(request: Request, call_next) -> Response:
    """Attach a unique request ID and processing time to every response."""
    req_id = request.headers.get("X-Request-Id", uuid.uuid4().hex[:12])
    t0     = time.perf_counter()

    response: Response = await call_next(request)

    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-Id"]       = req_id
    response.headers["X-Process-Time-Ms"]  = f"{elapsed:.2f}"

    logger.debug(
        "%s %s → %d | %s | %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        req_id,
        elapsed,
    )
    return response


# ── Exception handler ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred. Please try again."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(router)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.backend.src.main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        workers=WORKERS,
        access_log=True,
    )
