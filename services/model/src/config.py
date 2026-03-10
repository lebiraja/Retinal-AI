"""
Model Service configuration.
All values are read from environment variables with sane defaults.
"""

from __future__ import annotations

import os
from typing import Final, Tuple

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_MODEL_REPO: Final[str] = os.getenv(
    "HF_MODEL_REPO", "lebiraja/retinal-disease-classifier"
)
HF_MODEL_FILE: Final[str] = os.getenv("HF_MODEL_FILE", "pytorch_model.bin")

# Local checkpoint path (alternative to HuggingFace download).
# Set MODEL_CHECKPOINT_PATH to use a locally mounted .pt file instead.
LOCAL_CHECKPOINT_PATH: Final[str | None] = os.getenv("MODEL_CHECKPOINT_PATH")

# ── Image pre-processing ───────────────────────────────────────────────────────
IMAGE_SIZE: Final[int] = int(os.getenv("IMAGE_SIZE", "384"))
NORM_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
NORM_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)

# ── Model architecture ─────────────────────────────────────────────────────────
NUM_CLASSES: Final[int] = 45

LABEL_COLS: Final[list[str]] = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

# ── Server ─────────────────────────────────────────────────────────────────────
HOST: Final[str] = os.getenv("MODEL_SERVICE_HOST", "0.0.0.0")
PORT: Final[int] = int(os.getenv("MODEL_SERVICE_PORT", "8001"))
WORKERS: Final[int] = int(os.getenv("MODEL_SERVICE_WORKERS", "1"))
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "info")

# Inference timeout (seconds) — protects against runaway GPU jobs
INFERENCE_TIMEOUT_S: Final[float] = float(os.getenv("INFERENCE_TIMEOUT_S", "30.0"))

# ── HuggingFace cache ─────────────────────────────────────────────────────────
HF_CACHE_DIR: Final[str] = os.getenv("HF_HOME", "/app/.cache/huggingface")
