"""
app/config.py — API configuration.

Single source of truth for all FastAPI backend constants.
CNN training constants remain in the root config.py.
"""

# ── HuggingFace model ──────────────────────────────────────────────────────
HF_MODEL_NAME = "lebiraja/retinal-disease-classifier"

# ── Image preprocessing (must match training pipeline exactly) ─────────────
IMAGE_SIZE = 384                             # Resize input to 384×384
NORM_MEAN  = (0.485, 0.456, 0.406)          # ImageNet mean
NORM_STD   = (0.229, 0.224, 0.225)          # ImageNet std

# ── Detection threshold ────────────────────────────────────────────────────
# Probability above this → disease detected. Overridable per-request.
DEFAULT_THRESHOLD = 0.5

# ── Upload limits ──────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 10
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff",
}

# ── Model metadata ─────────────────────────────────────────────────────────
MODEL_VERSION   = "1.0"
BEST_AUC        = 0.8204
BEST_EPOCH      = 39
TRAIN_LOSS      = 0.2118
VAL_LOSS        = 0.2578
MACRO_F1        = 0.1517
MICRO_F1        = 0.4450

# ── Disease labels — ORDER MUST match model output indices ─────────────────
DISEASE_LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]
NUM_CLASSES = len(DISEASE_LABELS)  # 45

# ── Full human-readable disease names ──────────────────────────────────────
DISEASE_FULL_NAMES = {
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
    "AH":   "Anterior Chamber",
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
    "PRH":  "Peripapillary Retinal Hemorrhage",
    "MNF":  "Macular Neovascularization",
    "HR":   "Hard Retinal Exudate",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD":   "Temporal Disc",
    "CME":  "Cystoid Macular Edema",
    "PTCR": "Posterior Capsular Rent",
    "CF":   "Cotton Fiber",
    "VH":   "Vitreous Hemorrhage",
    "MCA":  "Microaneurysms",
    "VS":   "Vitreous Synchysis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ":  "Placoid Lesion",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL":   "Cotton Lint",
}

# ── High-risk diseases (elevated clinical urgency) ─────────────────────────
HIGH_RISK_DISEASES = {
    "DR", "ARMD", "CRVO", "CRAO", "VH", "AION",
    "CME", "HPED", "BRAO", "BRVO", "MNF", "RP",
}
