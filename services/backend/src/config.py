"""
Backend Service configuration.
All values are read from environment variables with sane defaults.
"""

from __future__ import annotations

import os
from typing import Final, FrozenSet

# ── Service discovery ──────────────────────────────────────────────────────────
# Internal Docker network URL of the model microservice
MODEL_SERVICE_URL: Final[str] = os.getenv(
    "MODEL_SERVICE_URL", "http://model-service:8001"
)

# HTTP client timeouts (seconds) for calling the model service
MODEL_CLIENT_TIMEOUT_S:  Final[float] = float(os.getenv("MODEL_CLIENT_TIMEOUT_S", "60.0"))
MODEL_CONNECT_TIMEOUT:   Final[float] = float(os.getenv("MODEL_CONNECT_TIMEOUT", "5.0"))
MODEL_READ_TIMEOUT:      Final[float] = float(os.getenv("MODEL_READ_TIMEOUT", "90.0"))
MODEL_MAX_RETRIES:       Final[int]   = int(os.getenv("MODEL_MAX_RETRIES", "2"))

# ── CORS ───────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins.  Use "*" for development only.
CORS_ORIGINS: Final[list[str]] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

# ── Upload limits ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: Final[int] = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_BATCH_SIZE: Final[int] = int(os.getenv("MAX_BATCH_SIZE", "10"))

ALLOWED_MIME_TYPES: Final[FrozenSet[str]] = frozenset(
    os.getenv(
        "ALLOWED_MIME_TYPES",
        "image/jpeg,image/jpg,image/png,image/bmp,image/tiff",
    ).split(",")
)

# ── Detection defaults ─────────────────────────────────────────────────────────
DEFAULT_THRESHOLD: Final[float] = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# ── Model metadata (returned in /info endpoint) ────────────────────────────────
MODEL_VERSION:  Final[str]   = os.getenv("MODEL_VERSION", "1.0")
HF_MODEL_REPO:  Final[str]   = os.getenv("HF_MODEL_REPO", "lebiraja/retinal-disease-classifier")
HF_MODEL_NAME:  Final[str]   = HF_MODEL_REPO   # alias used by router
NUM_CLASSES:    Final[int]   = 45
IMAGE_SIZE:     Final[int]   = 384
ARCHITECTURE:   Final[str]   = "EfficientNet-B4"
BEST_AUC:       Final[float] = 0.8204
BEST_EPOCH:     Final[int]   = 39
TRAIN_LOSS:     Final[float] = 0.2118
VAL_LOSS:       Final[float] = 0.2578
MACRO_F1:       Final[float] = 0.1517
MICRO_F1:       Final[float] = 0.4450

# ── Server ─────────────────────────────────────────────────────────────────────
HOST:      Final[str] = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT:      Final[int] = int(os.getenv("BACKEND_PORT", "8000"))
WORKERS:   Final[int] = int(os.getenv("BACKEND_WORKERS", "2"))
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "info")

# ── Disease labels — ORDER must match model output indices ─────────────────────
DISEASE_LABELS: Final[list[str]] = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

DISEASE_FULL_NAMES: Final[dict[str, str]] = {
    "DR":   "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH":   "Macular Hole",
    "DN":   "Drusen",
    "MYA":  "Myopic Astigmatism",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM":  "Epiretinal Membrane",
    "LS":   "Laser Scar",
    "MS":   "Macular Scar",
    "CSR":  "Central Serous Retinopathy",
    "ODC":  "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV":   "Tire Venture",
    "AH":   "Anterior Chamber Haemorrhage",
    "ODP":  "Optic Disc Pallor",
    "ODE":  "Optic Disc Edema",
    "ST":   "Shunt",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT":   "Parafoveal Telangiectasia",
    "RT":   "Retinal Traction",
    "RS":   "Retinal Scar",
    "CRS":  "Corneal Reflex Shadow",
    "EDN":  "Exudates",
    "RPEC": "RPE Changes",
    "MHL":  "Macular Hole (Large)",
    "RP":   "Retinitis Pigmentosa",
    "CWS":  "Cotton Wool Spots",
    "CB":   "Conjunctival Bleed",
    "ODPM": "Optic Disc Pallor Margin",
    "PRH":  "Peripapillary Retinal Haemorrhage",
    "MNF":  "Macular Neovascularization",
    "HR":   "Hard Retinal Exudate",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD":   "Temporal Disc",
    "CME":  "Cystoid Macular Edema",
    "PTCR": "Posterior Capsular Rent",
    "CF":   "Cotton Fiber",
    "VH":   "Vitreous Haemorrhage",
    "MCA":  "Microaneurysms",
    "VS":   "Vitreous Synchysis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ":  "Placoid Lesion",
    "HPED": "Haemorrhagic Pigment Epithelial Detachment",
    "CL":   "Cotton Lint",
}

HIGH_RISK_DISEASES: Final[frozenset[str]] = frozenset({
    "DR", "ARMD", "CRVO", "CRAO", "VH", "AION",
    "CME", "HPED", "BRAO", "BRVO", "MNF", "RP",
})
