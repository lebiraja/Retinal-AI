# Setup & Installation Guide

This guide walks through setting up the development environment from scratch.

## System Requirements

**Hardware:**
- GPU: NVIDIA GPU with CUDA compute capability ≥ 7.0 (RTX 4050, RTX 3060, RTX 4090, etc.)
- VRAM: ≥ 6 GB (will fit batch_size=20 at 512×512 with EfficientNet-B4)
- CPU: Any modern CPU (training will use 6 workers)
- RAM: ≥ 16 GB recommended

**OS & Software:**
- Linux (Ubuntu 22.04+ recommended) or macOS
- Python 3.10 or higher
- CUDA 12.1+ (if using NVIDIA GPU)
- pip or conda

**Dataset:**
- Retinal disease images already downloaded to `dataset/` folder (~7.6 GB)
- See [Dataset Preparation](#dataset-preparation) if you need to download it

---

## Step 1: Clone & Navigate

```bash
cd ~/projects/mindcraft-2k26
```

Verify the structure:
```bash
ls -la
# Expected output includes: config.py, dataset.py, model.py, train.py, inference.py, dataset/, docs/, etc.
```

---

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Verify activation:
```bash
which python3  # Should show: /path/to/mindcraft-2k26/venv/bin/python3
```

---

## Step 3: Upgrade pip, setuptools, wheel

```bash
pip install --upgrade pip setuptools wheel
```

---

## Step 4: Install PyTorch with CUDA Support

Your GPU driver is CUDA 13.1 compatible, so install PyTorch for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch version: 2.1.x
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 ...
```

---

## Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `albumentations` — image augmentation
- `scikit-learn` — metrics (AUC, F1 score)
- `pandas` — CSV handling
- `numpy` — numerical operations
- `Pillow` — image I/O
- `matplotlib`, `seaborn` — plotting

**Verify installation:**
```bash
python3 -c "import albumentations; import torch; import pandas; import sklearn; print('All imports OK')"
```

---

## Step 6: Verify Dataset Structure

```bash
ls -la dataset/
# Expected:
# Training_Set/
# Evaluation_Set/
# Test_Set/
```

Check label CSVs:
```bash
head -3 dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv
```

Expected first 3 columns: `ID`, `Disease_Risk`, `DR`, ...

---

## Step 7: Test GPU & Dataset Loading

```bash
python3 << 'EOF'
import torch
import config
from dataset import RetinalDataset

print(f"Device: {config.DEVICE}")
print(f"Image size: {config.IMG_SIZE}")
print(f"Batch size: {config.BATCH_SIZE}")

# Load one batch
ds = RetinalDataset(split="train")
print(f"Train dataset size: {len(ds)}")
img, label = ds[0]
print(f"Image shape: {img.shape}")
print(f"Label shape: {label.shape}")
print(f"Label sum (num diseases): {label.sum()}")
print("\n✓ All OK!")
EOF
```

Expected output:
```
Device: cuda
Image size: 512
Batch size: 20
Train dataset size: 1920
Image shape: torch.Size([3, 512, 512])
Label shape: torch.Size([45])
Label sum (num diseases): X.0

✓ All OK!
```

---

## Step 8: Run Sanity Check (2-epoch training)

```bash
python3 train.py --epochs 2
```

This takes ~3-5 minutes and verifies:
- ✓ GPU memory is sufficient
- ✓ Data loading works
- ✓ Model forward pass works
- ✓ Backward pass & optimization work

You should see:
```
Using device: cuda

Epoch 1/2
  train_loss=0.9823  val_loss=0.8945  mean_auc=0.4532  macro_f1=0.1234  micro_f1=0.5678  lr=1.00e-04
  ✓ Best model saved (mean_auc=0.4532)

Epoch 2/2
  train_loss=0.7656  val_loss=0.7234  mean_auc=0.5891  macro_f1=0.1567  micro_f1=0.6123  lr=9.99e-05
  ✓ Best model saved (mean_auc=0.5891)

Training complete. Best mean AUC: 0.5891
Checkpoints: outputs/checkpoints
Plots:       outputs/plots
```

If you see **CUDA out of memory**, reduce `BATCH_SIZE` in `config.py`:
```python
BATCH_SIZE = 12  # Try 12, then 8 if still OOM
```

---

## Troubleshooting Setup

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Virtual environment not activated. Run:
```bash
source venv/bin/activate
```

### Issue: `RuntimeError: CUDA out of memory`
**Solution:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 12  # or 8 for more conservative
```

### Issue: `torch.cuda.is_available()` returns `False`
**Solution:** PyTorch not installed for CUDA. Reinstall:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Dataset not found errors
**Solution:** Verify dataset structure:
```bash
ls dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv
ls dataset/Training_Set/Training_Set/Training/ | head -5
```

If missing, the Kaggle dataset must be downloaded (see [Dataset Preparation](#dataset-preparation)).

---

## Dataset Preparation

If you need to download the dataset from Kaggle:

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Move `~/.kaggle/kaggle.json` to your home directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download dataset:**
   ```bash
   cd ~/projects/mindcraft-2k26
   kaggle datasets download -d -p dataset/ deepblue007/retinal-disease-classification
   unzip -q dataset/retinal-disease-classification.zip -d dataset/
   rm dataset/retinal-disease-classification.zip
   ```

4. **Verify:**
   ```bash
   ls dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv
   ```

---

## Configuration Customization

Before training, you may want to customize `config.py`:

### GPU Memory Constraints

If you have less VRAM, reduce image size and batch size:

```python
# config.py
IMG_SIZE = 384      # or 256 for very tight VRAM
BATCH_SIZE = 12     # or 8
NUM_WORKERS = 4     # reduce for less RAM
```

### Training Duration

Adjust epochs:
```python
# config.py
EPOCHS = 100        # longer training for better accuracy
# or
EPOCHS = 20         # quick test/development run
```

### Learning Rate

For faster convergence (may reduce accuracy):
```python
# config.py
LR = 2e-4  # higher LR, faster convergence but less stable
```

---

## Deactivating Virtual Environment

When done working:
```bash
deactivate
```

To reactivate in a new terminal:
```bash
cd ~/projects/mindcraft-2k26
source venv/bin/activate
```

---

## Docker Setup (Optional)

For reproducible environments across machines:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-venv python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "train.py"]
```

Build and run:
```bash
docker build -t retinal-disease .
docker run --gpus all -v $(pwd)/outputs:/app/outputs retinal-disease
```

---

## Environment Summary

After setup, you should have:

```
✓ Python 3.10+
✓ Virtual environment with venv activated
✓ PyTorch with CUDA support
✓ All dependencies installed
✓ Dataset downloaded to dataset/
✓ GPU detected and working
✓ Sanity check passed (2-epoch training)
```

Next step: See [TRAINING.md](./TRAINING.md) to start training the full model.

---

**Last Updated:** 2026-02-22
