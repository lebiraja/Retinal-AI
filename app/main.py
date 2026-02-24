"""
app/main.py — FastAPI application entry point.

Run (from Team-B-Backend/ directory):
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict        — single-image disease classification
    POST /predict-batch  — multi-image batch classification
    GET  /health         — liveness + model status
    GET  /info           — model metadata and performance metrics
    GET  /docs           — Swagger UI (auto-generated)
    GET  /redoc          — ReDoc UI (auto-generated)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.predict import router
from app.services.model_service import ModelService

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model from HuggingFace ...")
    ModelService.load()
    logger.info("Startup complete. Server ready.")
    yield
    logger.info("Shutting down.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retinal Disease Classifier API",
    lifespan=lifespan,
    description=(
        "Multi-label retinal disease screening powered by **EfficientNet-B4** "
        "trained on the RFMiD dataset (1,920 fundus images). "
        "Detects up to **45 retinal conditions** with a mean AUC-ROC of **0.82**.\n\n"
        "---\n\n"
        "**Endpoints**\n\n"
        "| Method | Path | Description |\n"
        "|--------|------|-------------|\n"
        "| POST | `/predict` | Single-image classification with optional threshold |\n"
        "| POST | `/predict-batch` | Batch classification (up to 10 images) |\n"
        "| GET  | `/health` | Server liveness + model status |\n"
        "| GET  | `/info` | Model metadata and performance metrics |\n\n"
        "---\n\n"
        "> ⚠️ **Medical Disclaimer:** This system is NOT a medical diagnostic device. "
        "Results are for screening and educational purposes only. "
        "Always consult a qualified ophthalmologist for clinical decisions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Team B",
        "url": "https://huggingface.co/lebiraja/retinal-disease-classifier",
    },
    license_info={"name": "MIT"},
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins in development. Restrict to your frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router)
