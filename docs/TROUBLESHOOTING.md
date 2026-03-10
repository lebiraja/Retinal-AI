# Troubleshooting Guide

Common issues and solutions.

---

## Installation & Setup

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Symptom:**
```
Traceback (most recent call last):
  File "train.py", line 1, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**Causes:**
1. Virtual environment not activated
2. PyTorch not installed
3. Wrong Python interpreter

**Solutions:**

**Solution 1: Activate virtual environment**
```bash
source venv/bin/activate
python3 --version  # Should show 3.10+
which python3      # Should show path to venv/bin/python3
```

**Solution 2: Reinstall PyTorch**
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Solution 3: Use correct Python**
```bash
# Ensure venv is activated
source venv/bin/activate

# Try again
python3 train.py --epochs 2
```

---

## Docker & Microservices Issues

### Issue: GPU not detected by Docker

**Symptom:**
```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Cause:** NVIDIA Container Toolkit is not installed on the host machine.

**Solution:** Install `nvidia-docker2` or `nvidia-container-toolkit` for your OS, restart the docker daemon (`sudo systemctl restart docker`), and retry `docker compose up`.

### Issue: 422 Unprocessable Entity on Upload

**Symptom:** The web app immediately rejects an image with a message about aspect ratio, corners, or colors.

**Cause:** The new **Validation Service** heuristics tripped. The model only accepts retinal fundus photographs. 

**Solution:** Ensure you are uploading a genuine retinal scan (roughly square, dark corners, warm/red profile).

---

### Issue: `ImportError: cannot import name 'autocast'`

**Symptom:**
```
ImportError: cannot import name 'autocast' from 'torch.cuda.amp'
```

**Cause:** Old PyTorch version (< 1.10)

**Solution:** Upgrade PyTorch
```bash
pip install --upgrade torch torchvision
```

Verify:
```bash
python3 -c "from torch.cuda.amp import autocast, GradScaler; print('OK')"
```

---

### Issue: CUDA not found after PyTorch install

**Symptom:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**But nvidia-smi works:**
```bash
$ nvidia-smi
NVIDIA-SMI 590.48
```

**Cause:** PyTorch installed for CPU only

**Solution:** Reinstall for CUDA 12.1:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

---

## Training Issues

### Issue: `RuntimeError: CUDA out of memory`

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB.
GPU memory pool has already allocated 1.50 GiB.
```

**Cause:** Batch size too large for GPU VRAM

**Solutions (try in order):**

**Solution 1: Reduce batch size (EASIEST)**
```python
# config.py
BATCH_SIZE = 12  # was 20
```
Then rerun:
```bash
python3 train.py --epochs 2  # Sanity check first
```

**Solution 2: Reduce image size**
```python
# config.py
IMG_SIZE = 384  # was 512 (still reasonable detail)
BATCH_SIZE = 20
```

**Solution 3: Both**
```python
# config.py
IMG_SIZE = 256
BATCH_SIZE = 32
```

**Solution 4: Reduce num_workers (frees CPU memory)**
```python
# config.py
NUM_WORKERS = 2  # was 6 (reduces buffering)
```

**Solution 5: Clear GPU cache**
```bash
# Before training, clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```

**Solution 6: CPU fallback (very slow)**
```bash
# Force CPU training (not recommended, very slow)
python3 << 'EOF'
import config
import torch
config.DEVICE = torch.device("cpu")
EOF
python3 train.py --epochs 2
```

---

### Issue: `Loss is NaN`

**Symptom:**
```
Epoch 1/50
  train_loss=nan  val_loss=nan
```

**Causes:**
1. Learning rate too high (gradients explode)
2. Invalid data (inf, nan in images)
3. pos_weight contains inf or nan

**Solutions:**

**Solution 1: Lower learning rate**
```python
# config.py
LR = 5e-5  # was 1e-4
```

**Solution 2: Add gradient clipping**
```python
# train.py, in train_one_epoch() after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

**Solution 3: Check data**
```python
from dataset import RetinalDataset
ds = RetinalDataset(split="train")
img, label = ds[0]
print(f"Image min: {img.min()}, max: {img.max()}")
print(f"Label: {label}")
print(f"Has nan: {torch.isnan(img).any() or torch.isnan(label).any()}")
```

**Solution 4: Reset pos_weight**
```python
# config.py
POS_WEIGHT_CLIP = 5.0  # was 10.0 (cap extreme weights lower)
```

---

### Issue: Loss doesn't decrease

**Symptom:**
```
Epoch 1: train_loss=0.693, val_loss=0.691
Epoch 2: train_loss=0.692, val_loss=0.691
Epoch 3: train_loss=0.691, val_loss=0.691
```

(Suspicious: BCE loss at random baseline is ~0.693)

**Causes:**
1. Learning rate too low
2. Wrong optimizer settings
3. Model weights not updating
4. Bad random seed

**Solutions:**

**Solution 1: Increase learning rate**
```python
# config.py
LR = 2e-4  # was 1e-4, try higher
```

**Solution 2: Check optimizer is working**
```python
import torch
from train import train_one_epoch
from dataset import RetinalDataset
from model import build_model
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler

model = build_model().cuda()
train_ds = RetinalDataset(split="train")
train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)

# Check if weights update
param_before = model.classifier[1].weight.clone()
print(f"Param before: {param_before[0, 0]:.6f}")

# ... run one batch manually
for img, label in train_loader:
    model.train()
    img, label = img.cuda(), label.cuda()
    logits = model(img)
    loss = nn.BCEWithLogitsLoss()(logits, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Check gradients exist
    print(f"Gradients: {model.classifier[1].weight.grad is not None}")
    break

param_after = model.classifier[1].weight.clone()
print(f"Param after gradient: {param_after[0, 0]:.6f}")
print(f"Changed: {(param_before != param_after).any()}")
```

**Solution 3: Verify data loading**
```python
from dataset import RetinalDataset
ds = RetinalDataset(split="train")
img, label = ds[0]
print(f"Mean image value: {img.mean():.4f}")
print(f"Disease frequency: {label.sum() / 45:.2%}")
# If image is all black or all white, preprocessing is broken
```

---

### Issue: Training is very slow

**Symptom:**
```
Epoch 1/50 (takes 30+ minutes)
```

**Causes:**
1. CPU bottleneck (slow data loading)
2. GPU utilization low (<70%)
3. Dataset on slow storage (HDD)

**Solutions:**

**Solution 1: Increase num_workers**
```python
# config.py
NUM_WORKERS = 10  # was 6 (more parallel data loading)
```

**Solution 2: Check GPU utilization**
```bash
# While training, in another terminal:
nvidia-smi -l 1
# GPU-Util should be >90%
```

**Solution 3: Check disk I/O**
```bash
# Verify dataset is on fast storage
df -h dataset/
# Should show SSD (not /mnt/hdd)
```

**Solution 4: Reduce image size**
```python
# config.py
IMG_SIZE = 384  # was 512 (less data to load per image)
```

**Solution 5: Pin memory for DataLoader**
```python
# train.py, DataLoader creation
DataLoader(..., pin_memory=True)  # Already done
```

---

### Issue: Model overfits (val loss increases after epoch X)

**Symptom:**
```
Epoch 10: train_loss=0.15, val_loss=0.28
Epoch 11: train_loss=0.14, val_loss=0.29
Epoch 12: train_loss=0.13, val_loss=0.31  ← Val loss increasing = overfitting
```

**Causes:**
1. Not enough regularization
2. Training too long
3. Dataset too small (1920 images)

**Solutions:**

**Solution 1: Stop early**
```python
# In train.py, add early stopping:
if epoch_valid_loss > best_valid_loss + 0.05:  # patience of 5 epochs
    print("Early stopping")
    break
```

**Solution 2: Increase dropout**
```python
# model.py
nn.Dropout(p=0.5)  # was 0.4
```

**Solution 3: Increase weight decay**
```python
# config.py
WEIGHT_DECAY = 0.01  # was 1e-2 (stronger L2 regularization)
```

**Solution 4: Reduce epochs**
```bash
python3 train.py --epochs 30  # was 50
```

**Solution 5: More augmentation**
```python
# dataset.py, add more aggressive augmentation
train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.7),       # was 0.5
    A.VerticalFlip(p=0.5),         # was 0.3
    A.RandomBrightnessContrast(p=0.5),  # was 0.3
    ...
])
```

---

## Data Issues

### Issue: `FileNotFoundError: dataset/Training_Set/...`

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/Training_Set/...'
```

**Causes:**
1. Dataset not downloaded
2. Incorrect path in config.py
3. Folder structure wrong

**Solutions:**

**Solution 1: Verify dataset structure**
```bash
ls -la dataset/
# Should show: Training_Set/, Evaluation_Set/, Test_Set/

ls -la dataset/Training_Set/Training_Set/
# Should show: RFMiD_Training_Labels.csv, Training/

ls dataset/Training_Set/Training_Set/Training/ | head -5
# Should show image files: 1.png, 2.png, 3.png, ...
```

**Solution 2: Download dataset**

Follow [SETUP.md](./SETUP.md#dataset-preparation) to download from Kaggle.

**Solution 3: Check config.py paths**
```python
# config.py
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "Training_Set", "Training_Set", "Training")
print(f"Looking for images in: {TRAIN_IMG_DIR}")
print(f"Exists: {os.path.exists(TRAIN_IMG_DIR)}")

import os
print(os.listdir(DATASET_ROOT))  # Debug: what's actually in dataset/
```

---

### Issue: Dataset loading is slow

**Symptom:**
```
DataLoader takes 10+ seconds to load one batch
```

**Cause:** Albumentations operations slow on large images

**Solution:** Reduce image size during preprocessing
```python
# config.py
IMG_SIZE = 384  # or 256, depending on detail needed
```

---

## Inference Issues

### Issue: `FileNotFoundError: best_model.pt not found`

**Symptom:**
```bash
$ python3 inference.py --image image.png
FileNotFoundError: [Errno 2] No such file or directory: 'outputs/checkpoints/best_model.pt'
```

**Cause:** Model not trained yet

**Solution:** Train the model first
```bash
python3 train.py --epochs 2  # Quick training
python3 inference.py --image image.png
```

---

### Issue: All predictions are zeros

**Symptom:**
```python
result = predict("image.png")
print(result["detected_diseases"])  # []
```

**Causes:**
1. Model was randomly initialized, not trained
2. Threshold too high
3. Image is invalid (black/white image)

**Solutions:**

**Solution 1: Verify model exists and is trained**
```bash
ls -lh outputs/checkpoints/best_model.pt
# Should be >70 MB (if trained)

python3 -c "
import torch
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
print(f'Epoch: {checkpoint[\"epoch\"]}')
print(f'Mean AUC: {checkpoint[\"mean_auc\"]}')
"
```

**Solution 2: Lower threshold**
```python
from inference import predict
result = predict("image.png", threshold=0.3)  # was 0.5
print(result["detected_diseases"])
```

**Solution 3: Check image quality**
```python
import numpy as np
from PIL import Image

img = np.array(Image.open("image.png"))
print(f"Shape: {img.shape}")
print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
# Should be actual fundus photo, not black/white/noise
```

---

## GPU Issues

### Issue: GPU detected but training still slow

**Symptom:**
```bash
$ nvidia-smi
GPU 0: NVIDIA GeForce RTX 4050 ...

$ python3 train.py --epochs 2
# Takes 30+ min for 2 epochs (should be 3-5 min)
```

**Cause:** GPU not actually being used (model on CPU)

**Check:**
```python
import torch
from model import build_model
import config

model = build_model()
print(f"DEVICE: {config.DEVICE}")
print(f"Model device: {next(model.parameters()).device}")

# Should both show: cuda:0
```

**Solution:** Ensure model is moved to GPU
```python
model = model.to(config.DEVICE)  # Already done in train.py
```

---

### Issue: GPU memory leak (memory grows over epochs)

**Symptom:**
```
Epoch 1: GPU memory 3.5 GB
Epoch 2: GPU memory 4.2 GB
Epoch 3: GPU memory 4.9 GB
```

**Cause:** Detached tensors still on GPU

**Solution:** Clear cache periodically
```python
# train.py, in main loop:
for epoch in range(epochs):
    ... training ...
    torch.cuda.empty_cache()  # Clear GPU cache
```

---

## Performance Issues

### Issue: Model predictions are inconsistent

**Symptom:**
```python
result1 = predict("image.png")
result2 = predict("image.png")
# Different results each time!
```

**Cause:** Model in training mode (with dropout)

**Check:**
```python
from inference import load_model
model = load_model()
print(f"Training: {model.training}")  # Should be False
```

---

### Issue: Different results on CPU vs GPU

**Cause:** Floating point arithmetic differences

**Solution:** Not a bug, expected behavior. Use same device consistently.

---

## Common Questions

### Q: Can I train on CPU?

**A:** Yes, but **very slow** (~1 hour per epoch). Not recommended. If you must:

```python
import config
config.DEVICE = torch.device("cpu")
python3 train.py --epochs 2
```

---

### Q: Can I train on multiple GPUs?

**A:** Current code supports single GPU. For multi-GPU:

```python
# Use torch.nn.DataParallel
model = torch.nn.DataParallel(model)
```

But requires more setup. Not recommended unless you have multiple GPUs.

---

### Q: How do I adjust the threshold for a specific disease?

**A:** Currently threshold is global. For per-disease thresholds (Phase 2):

```python
result = predict("image.png")
custom_threshold = {"DR": 0.4, "ARMD": 0.6, ...}
detected = [
    disease for disease, prob in result["predictions"].items()
    if prob >= custom_threshold.get(disease, 0.5)
]
```

---

### Q: Can I train on my own dataset?

**A:** Yes. Create CSV with columns: ID, DR, ARMD, ... (45 disease columns).

Then modify `config.py` paths to your dataset location.

---

### Q: How do I export the model for production?

**A:** The `.pt` checkpoint is already production-ready. For other formats:

```python
import torch
from model import build_model

model = build_model()
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Export to ONNX (Phase 2 might use this)
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## Getting Help

If your issue isn't covered:

1. **Check error message carefully** — often contains root cause
2. **Search [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** — similar issues documented
3. **Run sanity check:** `python3 train.py --epochs 2`
4. **Check GPU:** `nvidia-smi`
5. **Verify dataset:** `ls dataset/Training_Set/Training_Set/Training/ | wc -l`
6. **Try from scratch:**
   ```bash
   rm -rf outputs/
   python3 train.py --epochs 2
   ```

---

**Last Updated:** March 2026
