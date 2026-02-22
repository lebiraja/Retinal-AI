import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = "dataset"

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "Training_Set", "Training_Set", "Training")
VALID_IMG_DIR = os.path.join(DATASET_ROOT, "Evaluation_Set", "Evaluation_Set", "Validation")
TEST_IMG_DIR  = os.path.join(DATASET_ROOT, "Test_Set", "Test_Set", "Test")

TRAIN_CSV = os.path.join(DATASET_ROOT, "Training_Set", "Training_Set", "RFMiD_Training_Labels.csv")
VALID_CSV = os.path.join(DATASET_ROOT, "Evaluation_Set", "Evaluation_Set", "RFMiD_Validation_Labels.csv")
TEST_CSV  = os.path.join(DATASET_ROOT, "Test_Set", "Test_Set", "RFMiD_Testing_Labels.csv")

OUTPUT_DIR      = "outputs"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR        = os.path.join(OUTPUT_DIR, "plots")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")
LOG_CSV_PATH    = os.path.join(LOG_DIR, "training_log.csv")

# ── Image ──────────────────────────────────────────────────────────────────────
IMG_SIZE = 512

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 20       # tuned for RTX 4050 6GB @ 512×512 with EfficientNet-B4
NUM_WORKERS  = 6
EPOCHS       = 50
LR           = 1e-4     # classifier head LR; backbone gets LR * 0.1
WEIGHT_DECAY = 1e-2
POS_WEIGHT_CLIP = 10.0  # cap extreme weights for ultra-rare diseases

# ── Model ──────────────────────────────────────────────────────────────────────
# 45 disease labels (Disease_Risk column excluded — derivable from predictions)
LABEL_COLS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]
NUM_CLASSES = len(LABEL_COLS)  # 45

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
