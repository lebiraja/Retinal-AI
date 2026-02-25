# Training Guide

Complete guide to training the retinal disease classifier.

---

## Quick Start

**1. Activate environment:**
```bash
cd ~/projects/mindcraft-2k26
source venv/bin/activate
```

**2. Sanity check (2 epochs, 3-5 min):**
```bash
python3 train.py --epochs 2
```

**3. Full training (50 epochs, 8-12 hours):**
```bash
python3 train.py
```

Monitor progress:
```bash
# In another terminal:
nvidia-smi -l 2          # GPU usage every 2 sec
tail -f outputs/logs/training_log.csv  # Live log
```

---

## Training Script Usage

### Basic: Run 50 epochs (default)

```bash
python3 train.py
```

Trains on GPU if available, saves best model to `outputs/checkpoints/best_model.pt`.

### Custom epoch count

```bash
python3 train.py --epochs 100  # More epochs for better accuracy
python3 train.py --epochs 2    # Sanity check
```

### Command-line options

```bash
python3 train.py --help

usage: train.py [-h] [--epochs EPOCHS]

optional arguments:
  -h, --help       show this help message and exit
  --epochs EPOCHS  Number of training epochs (default: 50)
```

---

## What Happens During Training

### Per Epoch Breakdown

```
Epoch 1/50
├─ Train loop:
│  ├─ Process 96 batches (1920 images / 20 batch_size)
│  ├─ Forward pass → logits
│  ├─ Compute BCEWithLogitsLoss(pos_weight)
│  ├─ Backward pass → gradients
│  └─ Optimizer.step() → update weights
│
├─ Validation loop:
│  ├─ Process 32 batches (640 images / 20 batch_size)
│  ├─ Forward pass → logits
│  ├─ Sigmoid → probabilities
│  ├─ Compute validation loss
│  └─ Compute metrics (AUC-ROC, F1)
│
├─ Print epoch summary:
│  └─ train_loss=0.4523  val_loss=0.3891  mean_auc=0.7234  ...
│
├─ Checkpointing:
│  ├─ If mean_auc improved → save best_model.pt
│  └─ Always save last_model.pt
│
└─ LR scheduling:
   └─ Cosine annealing: decrease LR toward zero
```

### Expected Output

```
Using device: cuda

Epoch 1/50
  train_loss=0.8945  val_loss=0.7234  mean_auc=0.5123  macro_f1=0.1234  micro_f1=0.5678  lr=1.00e-04
  ✓ Best model saved (mean_auc=0.5123)

Epoch 2/50
  train_loss=0.6789  val_loss=0.5821  mean_auc=0.6521  macro_f1=0.1567  micro_f1=0.6234  lr=1.00e-04
  ✓ Best model saved (mean_auc=0.6521)

...

Epoch 50/50
  train_loss=0.1234  val_loss=0.2891  mean_auc=0.8932  macro_f1=0.6123  micro_f1=0.7456  lr=1.23e-05
Training complete. Best mean AUC: 0.8932
Checkpoints: outputs/checkpoints
Plots:       outputs/plots
```

---

## Monitoring Training

### Real-time GPU Usage

```bash
watch -n 1 nvidia-smi
```

Or use `nvidia-smi` with watch:
```bash
while true; do clear; nvidia-smi; sleep 2; done
```

Expected during training:
- **GPU Util:** 90-100%
- **Memory:** ~5.8 GB (out of 6 GB)
- **Power:** 80-100W

### Live Loss Monitoring

```bash
tail -f outputs/logs/training_log.csv
```

Or with column formatting:
```bash
column -t -s ',' outputs/logs/training_log.csv | tail -20
```

### Parse & Plot Results

After training completes:

```bash
python3 << 'EOF'
import csv
import matplotlib.pyplot as plt
import numpy as np

# Read log
rows = []
with open("outputs/logs/training_log.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

epochs = [int(r["epoch"]) for r in rows]
train_loss = [float(r["train_loss"]) for r in rows]
val_loss = [float(r["val_loss"]) for r in rows]
mean_auc = [float(r["mean_auc"]) for r in rows]

# Print best metrics
best_idx = np.argmax(mean_auc)
print(f"Best model at epoch {epochs[best_idx]}:")
print(f"  Train loss: {train_loss[best_idx]:.4f}")
print(f"  Val loss: {val_loss[best_idx]:.4f}")
print(f"  Mean AUC: {mean_auc[best_idx]:.4f}")
EOF
```

---

## Outputs & Checkpoints

### Directory Structure

```
outputs/
├── checkpoints/
│   ├── best_model.pt      # Best model by validation AUC
│   └── last_model.pt      # Last epoch (may not be best)
├── logs/
│   └── training_log.csv   # Epoch-by-epoch metrics
└── plots/
    └── loss_curve.png     # Loss & AUC plots
```

### Checkpoint Format

Each `.pt` file is a PyTorch checkpoint dictionary:

```python
{
    "epoch": 25,
    "model_state_dict": {...},           # Model weights
    "optimizer_state_dict": {...},       # Optimizer state (if resuming)
    "mean_auc": 0.8934,
}
```

### Loading Checkpoint

```python
import torch
from model import build_model

model = build_model()
checkpoint = torch.load("outputs/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Training Log CSV

Example `outputs/logs/training_log.csv`:

```csv
epoch,train_loss,val_loss,mean_auc,macro_f1,micro_f1,lr
1,0.894532,0.723491,0.512345,0.123456,0.567890,1.00e-04
2,0.678901,0.582123,0.652134,0.156789,0.623451,1.00e-04
3,0.545678,0.501234,0.712456,0.198765,0.675123,9.99e-05
...
50,0.123456,0.289123,0.893456,0.612345,0.745678,1.23e-05
```

**Columns:**
- `epoch`: Training epoch (1-50)
- `train_loss`: Average training loss
- `val_loss`: Average validation loss
- `mean_auc`: Mean AUC-ROC across 45 diseases
- `macro_f1`: F1 score (macro-averaged across classes)
- `micro_f1`: F1 score (micro-averaged, global)
- `lr`: Current learning rate

---

## Hyperparameter Tuning

Default settings are tuned for the dataset and hardware. Adjust for your needs:

### Memory Constraints (OOM Error)

If you get `RuntimeError: CUDA out of memory`:

**Option 1: Reduce batch size**
```python
# config.py
BATCH_SIZE = 12  # was 20
```

**Option 2: Reduce image size**
```python
# config.py
IMG_SIZE = 384  # was 512
```

**Option 3: Both**
```python
# config.py
IMG_SIZE = 256
BATCH_SIZE = 32  # can fit larger batches at smaller images
```

### Faster Training

```python
# config.py
EPOCHS = 30        # fewer epochs
BATCH_SIZE = 32    # larger batches (if memory allows)
IMG_SIZE = 384     # smaller images (if detail loss acceptable)
```

### Better Accuracy

```python
# config.py
EPOCHS = 100        # longer training
LR = 5e-5           # lower LR, more careful fine-tuning
WEIGHT_DECAY = 1e-1 # stronger regularization
```

### Aggressive Training (Risky)

```python
# config.py
EPOCHS = 200
LR = 1e-3           # high LR, may diverge or overfit
```

### Learning Rate Schedule

Default is cosine annealing. To use step-based decay instead:

```python
# train.py (modify)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

---

## Training Strategies

### Strategy 1: Quick Exploration (2-3 hours)

For rapid prototyping or debugging:

```python
# config.py
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = 384
```

```bash
python3 train.py --epochs 20
```

Expected: Baseline AUC ~0.70-0.75

### Strategy 2: Production (12-24 hours)

For best accuracy:

```python
# config.py (keep defaults)
EPOCHS = 50
BATCH_SIZE = 20
IMG_SIZE = 512
```

```bash
python3 train.py  # 50 epochs, ~8-12 hours
```

Expected: AUC ~0.85-0.92

### Strategy 3: Conservative (24+ hours)

If time allows:

```python
# config.py
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 512
```

```bash
python3 train.py --epochs 100
```

Expected: AUC ~0.88-0.94, better generalization

---

## Resuming Training

To continue training from a checkpoint (e.g., if interrupted):

```python
# train_resume.py (create new file)
import torch
from train import main
from model import build_model
import config

# Load checkpoint
checkpoint = torch.load(config.BEST_MODEL_PATH)
start_epoch = checkpoint["epoch"]

# Build model and load weights
model = build_model()
model.load_state_dict(checkpoint["model_state_dict"])

# For simplicity, just continue with train.py
# (A proper resume would need to refactor train.py)
```

**Simplified approach:** Just run `python3 train.py` again — it will start from epoch 1 but will overwrite outputs. For true resume functionality, refactor `train.py` to load and continue from best checkpoint.

---

## Troubleshooting Training

### Issue: "CUDA out of memory"

```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX GiB
```

**Solutions:**
1. Reduce `BATCH_SIZE` → 12, then 8
2. Reduce `IMG_SIZE` → 384, then 256
3. Reduce `NUM_WORKERS` → 4, then 2
4. Close other GPU-using applications

### Issue: "Loss is NaN"

```
train_loss=nan  val_loss=nan
```

**Causes & fixes:**
- LR too high → reduce `LR` in config.py
- Exploding gradients → enable gradient clipping:
  ```python
  # train.py
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

### Issue: "Loss doesn't decrease"

**Causes:**
- LR too low → increase `LR` (e.g., 2e-4)
- Model underfitting → reduce dropout or regularization
- Bad data → verify dataset loading with `python3 -c "from dataset import RetinalDataset; ds = RetinalDataset(); print(ds[0])"`

### Issue: Slow training (< 90% GPU utilization)

**Causes:**
- CPU bottleneck (data loading) → increase `NUM_WORKERS`
- Disk I/O bottleneck → ensure dataset is on fast SSD
- Small batch size → increase `BATCH_SIZE` if memory allows

---

## Understanding Metrics

### Mean AUC-ROC

- **Range:** [0, 1]
- **0.5:** Random guessing
- **0.7:** Fair
- **0.8:** Good
- **0.9+:** Excellent

For multi-label, "mean AUC" averages AUC across 45 disease classes.

### F1 Score

- **Macro F1:** Simple average of per-class F1 (0.4-0.7 typical)
- **Micro F1:** Weighted average by instance frequency (usually higher)

### Validation Loss

- **Decreasing:** Good convergence
- **Increasing after epoch X:** Overfitting → may need earlier stopping
- **Flat:** Underfitting or learning rate too low

---

## Best Practices

1. **Always run sanity check first**
   ```bash
   python3 train.py --epochs 2
   ```

2. **Monitor GPU temperature** (keep <80°C)
   ```bash
   watch -n 2 nvidia-smi
   ```

3. **Save logs & models incrementally** (already done in `train.py`)

4. **Check for data leakage** — ensure no overlap between train/val/test splits (already handled)

5. **Validate on held-out test set** after best model is selected

6. **Document hyperparameters** before changing:
   ```bash
   # Before tuning, note current config.py
   cp config.py config_baseline.py
   ```

---

## Expected Performance

After 50 epochs with default settings, expect:

| Metric | Range | Notes |
|--------|-------|-------|
| Mean AUC | 0.85–0.92 | Top classes >0.95, rare <0.60 |
| Macro F1 | 0.55–0.70 | Limited by ultra-rare diseases |
| Micro F1 | 0.60–0.75 | Biased toward common diseases |
| Train loss | 0.10–0.20 | Per-image average |
| Val loss | 0.25–0.35 | Slightly higher than train |

**Variation:** Actual results depend on hardware, random seed, and data augmentation.

---

## Next Steps

After training completes:

1. **Review loss curve:** `cat outputs/plots/loss_curve.png`
2. **Check if overfitting:** Does val loss increase near the end?
3. **Test inference:** `python3 inference.py --image dataset/Training_Set/Training_Set/Training/1.png`
4. **Evaluate on test set:** (code coming in Phase 2)
5. **Deploy:** Load `outputs/checkpoints/best_model.pt` in Phase 2 (web app)

---

**Last Updated:** 2026-02-22
