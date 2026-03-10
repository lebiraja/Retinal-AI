# API Reference

Complete API documentation for all modules.

---

## config.py

Central configuration module. All paths, hyperparameters, and constants defined here.

### Paths

```python
DATASET_ROOT: str = "dataset"
TRAIN_IMG_DIR: str
VALID_IMG_DIR: str
TEST_IMG_DIR: str
TRAIN_CSV: str
VALID_CSV: str
TEST_CSV: str
OUTPUT_DIR: str = "outputs"
CHECKPOINT_DIR: str = "outputs/checkpoints"
LOG_DIR: str = "outputs/logs"
PLOT_DIR: str = "outputs/plots"
BEST_MODEL_PATH: str
LAST_MODEL_PATH: str
LOG_CSV_PATH: str
```

### Image Settings

```python
IMG_SIZE: int = 512  # Resize all images to (512, 512)
```

### Training Hyperparameters

```python
BATCH_SIZE: int = 20
NUM_WORKERS: int = 6
EPOCHS: int = 50
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-2
POS_WEIGHT_CLIP: float = 10.0
```

### Model Configuration

```python
LABEL_COLS: List[str]  # 45 disease names
NUM_CLASSES: int = 45
```

### Device

```python
DEVICE: torch.device  # cuda if available, else cpu
```

**Usage:**
```python
import config
print(f"Training on {config.DEVICE}")
print(f"Image size: {config.IMG_SIZE}")
print(f"Batch size: {config.BATCH_SIZE}")
```

---

## dataset.py

Data loading and preprocessing.

### Function: `_build_transforms(train: bool) -> A.Compose`

Build albumentations transform pipeline.

**Parameters:**
- `train` (bool): If True, apply augmentation. If False, only resize & normalize.

**Returns:** albumentations Compose object.

**Example:**
```python
from dataset import _build_transforms

train_transform = _build_transforms(train=True)
val_transform = _build_transforms(train=False)

# Apply to numpy image
img_array = np.array(Image.open("image.png"))
augmented = train_transform(image=img_array)["image"]  # tensor
```

### Function: `_load_df(csv_path: str, img_dir: str) -> pd.DataFrame`

Load labels CSV and add filepath column.

**Parameters:**
- `csv_path` (str): Path to RFMiD labels CSV
- `img_dir` (str): Path to image directory

**Returns:** DataFrame with columns: ID, Disease_Risk, DR, ..., filepath

**Example:**
```python
from dataset import _load_df
import config

df = _load_df(config.TRAIN_CSV, config.TRAIN_IMG_DIR)
print(df.columns)
# ['ID', 'Disease_Risk', 'DR', 'ARMD', ..., 'filename', 'filepath']
```

### Class: `RetinalDataset(Dataset)`

PyTorch Dataset for retinal images.

**Constructor:**
```python
RetinalDataset(split: str = "train") -> RetinalDataset
```

**Parameters:**
- `split` (str): One of "train", "val", "test"

**Properties:**
```python
dataset.df: pd.DataFrame          # Labels dataframe
dataset.labels: np.ndarray        # (N, 45) float32
dataset.transform: A.Compose      # Transform pipeline
```

**Methods:**

```python
def __len__(self) -> int
    """Return number of images in split."""

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]
    """
    Return single sample.

    Returns:
        image: (3, 512, 512) tensor, normalized to ImageNet stats
        label: (45,) float32 tensor, values in {0, 1}
    """
```

**Example:**
```python
from dataset import RetinalDataset

ds = RetinalDataset(split="train")
print(len(ds))  # 1920

image, labels = ds[0]
print(image.shape)   # torch.Size([3, 512, 512])
print(labels.shape)  # torch.Size([45])
print(labels.sum())  # ~2-5 (avg diseases per image)
```

### Function: `get_pos_weights(dataset: RetinalDataset) -> torch.Tensor`

Compute class imbalance weights.

**Parameters:**
- `dataset` (RetinalDataset): Dataset to compute weights from

**Returns:** (45,) tensor of per-class weights

**Formula:**
```
weight[i] = (total_samples - positive_samples[i]) / positive_samples[i]
capped at max=10.0
```

**Example:**
```python
from dataset import RetinalDataset, get_pos_weights

train_ds = RetinalDataset(split="train")
pos_weights = get_pos_weights(train_ds)
print(pos_weights)  # tensor([1.86, 2.34, 3.12, ...])

# Use in loss
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

---

## model.py

Model architecture definitions.

### Function: `build_model(num_classes: int = 45) -> nn.Module`

Build EfficientNet-B4 classifier.

**Parameters:**
- `num_classes` (int): Number of output classes (default: 45)

**Returns:** EfficientNet-B4 model with custom head

**Example:**
```python
from model import build_model

model = build_model(num_classes=45)
model = model.to("cuda")
model.eval()
```

**Architecture:**
```
EfficientNet-B4 (pretrained on ImageNet1K)
    |
    └─ features: [input] → (N, 1792)
    └─ classifier:
        ├─ Dropout(0.4)
        └─ Linear(1792, 45)

Output: (N, 45) logits
```

### Function: `get_param_groups(model: nn.Module, lr: float) -> List[Dict]`

Create parameter groups for differential learning rates.

**Parameters:**
- `model` (nn.Module): EfficientNet-B4 model
- `lr` (float): Base learning rate for classifier head

**Returns:** List of parameter group dicts for optimizer

**Example:**
```python
from model import build_model, get_param_groups
import torch.optim as optim

model = build_model()
param_groups = get_param_groups(model, lr=1e-4)

optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
# Backbone gets lr=1e-5, head gets lr=1e-4
```

---

## train.py

Main training script and helpers.

### Function: `make_dirs()`

Create output directories if they don't exist.

**Example:**
```python
from train import make_dirs
make_dirs()
# Creates: outputs/checkpoints, outputs/logs, outputs/plots
```

### Function: `compute_metrics(all_targets: np.ndarray, all_probs: np.ndarray) -> dict`

Compute validation metrics.

**Parameters:**
- `all_targets` (np.ndarray): (N, 45) binary labels
- `all_probs` (np.ndarray): (N, 45) sigmoid probabilities

**Returns:** Dict with keys:
```python
{
    "mean_auc": float,      # Average AUC-ROC across classes
    "macro_f1": float,      # Macro-averaged F1
    "micro_f1": float,      # Micro-averaged F1
}
```

**Example:**
```python
from train import compute_metrics
import numpy as np

targets = np.random.randint(0, 2, (100, 45))
probs = np.random.rand(100, 45)

metrics = compute_metrics(targets, probs)
print(metrics)  # {'mean_auc': 0.523, 'macro_f1': 0.123, ...}
```

### Function: `save_plots(log_csv: str)`

Generate and save loss/AUC plots.

**Parameters:**
- `log_csv` (str): Path to training_log.csv

**Output:** Saves `outputs/plots/loss_curve.png`

**Example:**
```python
from train import save_plots
save_plots("outputs/logs/training_log.csv")
```

### Function: `train_one_epoch(...) -> float`

Run one training epoch.

**Signature:**
```python
train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float
```

**Returns:** Average training loss for the epoch

**Example:**
```python
# Called in main training loop
for epoch in range(50):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device
    )
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
```

### Function: `evaluate(...) -> Tuple[float, dict]`

Run validation epoch.

**Signature:**
```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, dict]
```

**Returns:** Tuple of:
- `val_loss` (float): Average validation loss
- `metrics` (dict): Results from `compute_metrics()`

**Example:**
```python
val_loss, metrics = evaluate(model, val_loader, criterion, device)
print(f"Val loss: {val_loss:.4f}, AUC: {metrics['mean_auc']:.4f}")
```

### Function: `main(epochs: int = 50)`

Main training function.

**Parameters:**
- `epochs` (int): Number of epochs to train

**Outputs:**
- Checkpoint files: `outputs/checkpoints/best_model.pt`, `last_model.pt`
- Log file: `outputs/logs/training_log.csv`
- Plot: `outputs/plots/loss_curve.png`

**Example:**
```python
from train import main
main(epochs=50)
```

---

## inference.py

Inference on new images.

### Function: `load_model(checkpoint_path: str) -> nn.Module`

Load model from checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to `.pt` checkpoint

**Returns:** Model on appropriate device, in eval mode

**Example:**
```python
from inference import load_model
import config

model = load_model(config.BEST_MODEL_PATH)
print(model)  # EfficientNet model
```

### Function: `_preprocess(image_path: str) -> torch.Tensor`

Preprocess single image.

**Parameters:**
- `image_path` (str): Path to PNG/JPG image

**Returns:** (1, 3, 512, 512) tensor, normalized

**Internal use only.**

### Function: `predict(...) -> dict`

Run inference on single image.

**Signature:**
```python
@torch.no_grad()
def predict(
    image_path: str,
    model: nn.Module = None,
    threshold: float = 0.5,
    checkpoint_path: str = config.BEST_MODEL_PATH,
) -> dict
```

**Parameters:**
- `image_path` (str): Path to fundus image
- `model` (nn.Module, optional): Pre-loaded model. If None, load from checkpoint.
- `threshold` (float): Probability threshold for disease detection
- `checkpoint_path` (str): Path to checkpoint (if model is None)

**Returns:**
```python
{
    "disease_risk": bool,           # Any disease detected?
    "predictions": {
        "DR": 0.87,
        "ARMD": 0.12,
        ...
    },
    "detected_diseases": ["DR"],     # Diseases >= threshold
    "num_detected": 1,
}
```

> **Note on Microservices API:** 
> When calling the live REST API (via Nginx proxy at `POST /api/predict`), the **Validation Service** performs a heuristic check before inference. If the image uploaded does not look like a retinal fundus scan (e.g. wrong aspect ratio, incorrect colors, or bright corners), the API will immediately return:
> ```json
> HTTP 422 Unprocessable Entity
> {
>   "detail": "'image.png' does not appear to be a retinal fundus photograph (aspect ratio 2.10:1 is too wide/tall). Please upload a genuine fundus/retinal scan image."
> }
> ```

**Example Local CLI Usage:**
```python
from inference import predict, load_model

# Option 1: Load model once, reuse
model = load_model()
result1 = predict("image1.png", model=model)
result2 = predict("image2.png", model=model)

# Option 2: Single prediction (loads model each time)
result = predict("image.png")
```

---

## Module Imports

Quick reference for common imports:

```python
# Config
import config

# Data
from dataset import RetinalDataset, get_pos_weights
from torch.utils.data import DataLoader

# Model
from model import build_model, get_param_groups

# Training
from train import train_one_epoch, evaluate, compute_metrics

# Inference
from inference import predict, load_model

# Standard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
```

---

## Type Hints

Python 3.10+ type annotations used throughout:

```python
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Examples:
def load_model(checkpoint_path: str = "best.pt") -> nn.Module: ...

def predict(
    image_path: str,
    model: Optional[nn.Module] = None,
    threshold: float = 0.5,
) -> Dict[str, any]: ...

def get_pos_weights(dataset: Dataset) -> torch.Tensor: ...
```

---

## Checkpoint Format

Each `.pt` checkpoint is a Python dict:

```python
checkpoint = {
    "epoch": int,                      # Epoch number
    "model_state_dict": dict,          # Model weights (from state_dict())
    "optimizer_state_dict": dict,      # Optimizer state
    "mean_auc": float,                 # Best validation AUC
}
```

**Loading:**
```python
import torch
from model import build_model

model = build_model()
checkpoint = torch.load("best_model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Checkpoint from epoch {checkpoint['epoch']}, AUC={checkpoint['mean_auc']:.4f}")
```

---

## Common Patterns

### Pattern 1: Train from scratch

```python
from model import build_model, get_param_groups
from dataset import RetinalDataset, get_pos_weights
import torch.nn as nn
import torch.optim as optim

model = build_model()
train_ds = RetinalDataset(split="train")
pos_weights = get_pos_weights(train_ds)

optimizer = optim.AdamW(
    get_param_groups(model, lr=1e-4),
    weight_decay=1e-2
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# ... training loop
```

### Pattern 2: Load and evaluate

```python
from inference import load_model, predict

model = load_model()
result = predict("image.png", model=model)
print(f"Disease risk: {result['disease_risk']}")
print(f"Detected: {result['detected_diseases']}")
```

### Pattern 3: Batch inference

```python
from inference import load_model, predict
import os

model = load_model()
results = {}
for filename in os.listdir("images/"):
    if filename.endswith(".png"):
        result = predict(f"images/{filename}", model=model)
        results[filename] = result["num_detected"]
```

---

**Last Updated:** 2026-02-22
