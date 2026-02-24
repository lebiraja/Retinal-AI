# Retinal Disease Classifier — Complete Documentation

**AI-Based Eye Disease Classification and Advisory System** | Phase 1 Complete ✅

---

## 📚 Documentation Index

### **For End Users**
- **[USER_GUIDE.md](./USER_GUIDE.md)** — Installation, basic usage, examples, troubleshooting

### **For Backend Developers**
- **[BACKEND.md](./BACKEND.md)** — FastAPI/Flask integration, API endpoints, deployment

### **For ML Developers & Contributors**
- **[DEVELOPER.md](./DEVELOPER.md)** — Fine-tuning, model modifications, setup

### **Technical Details**
- **[SETUP.md](./SETUP.md)** — Environment setup, dependencies, GPU configuration
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — Model design, loss functions, data pipeline
- **[TRAINING.md](./TRAINING.md)** — How to run training, monitoring, hyperparameter tuning
- **[INFERENCE.md](./INFERENCE.md)** — Predictions, output formats, advanced usage
- **[API_REFERENCE.md](./API_REFERENCE.md)** — Complete code API documentation
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** — Common issues and solutions

### **Model Information**
- **[MODEL_CARD.md](./MODEL_CARD.md)** — Model specifications, performance metrics, supported diseases

## Project Overview

**Goal:** Train a multi-label CNN classifier to detect 45 retinal diseases in fundus images.

**Dataset:** RFMiD (Retinal Fundus Multi-disease Image Dataset)
- 1,920 training images
- 640 validation images
- 640 test images
- 45 disease labels per image (binary classification)
- Total size: ~7.6 GB

**Hardware:**
- GPU: NVIDIA GeForce RTX 4050 Mobile (6GB VRAM)
- CPU: Intel i7-13650HX
- RAM: 15.6 GB

**Model:**
- Base: EfficientNet-B4 (ImageNet pretrained)
- Head: Dropout(0.4) + Linear(1792 → 45)
- Loss: BCEWithLogitsLoss with class-weighted pos_weight
- Optimizer: AdamW with differential learning rates
- Scheduler: CosineAnnealingLR

## File Structure

```
mindcraft-2k26/
├── config.py              # Centralized configuration
├── dataset.py             # RetinalDataset class + transforms
├── model.py               # EfficientNet-B4 builder
├── train.py               # Main training script
├── inference.py           # Prediction/inference script
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
│
├── dataset/               # Dataset (not in git, see SETUP.md)
│   ├── Training_Set/
│   ├── Evaluation_Set/
│   └── Test_Set/
│
├── outputs/               # Auto-generated during training
│   ├── checkpoints/       # Model .pt files (best_model.pt, last_model.pt)
│   ├── logs/              # training_log.csv
│   └── plots/             # loss_curve.png
│
└── docs/                  # This documentation
    ├── README.md          # You are here
    ├── SETUP.md           # Installation & setup
    ├── ARCHITECTURE.md    # Model & design details
    ├── TRAINING.md        # Training workflow
    ├── INFERENCE.md       # Inference & deployment
    ├── API_REFERENCE.md   # Code API docs
    └── TROUBLESHOOTING.md # FAQ & debugging
```

## Quick Start

### 1. Install & Setup (5 min)

```bash
cd ~/projects/mindcraft-2k26
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See [SETUP.md](./SETUP.md) for detailed steps.

### 2. Sanity Check (30 sec)

```bash
python3 train.py --epochs 2
```

Verify no OOM error and loss decreases.

### 3. Full Training (12-24 hours)

```bash
python3 train.py
```

Monitor with `nvidia-smi` and `tail -f outputs/logs/training_log.csv`.

### 4. Test Inference (1 min)

```bash
python3 inference.py --image dataset/Training_Set/Training_Set/Training/1.png
```

See [INFERENCE.md](./INFERENCE.md) for output format.

## Key Concepts

### Multi-Label Classification
Unlike typical image classification (dog vs cat), each image can have **multiple diseases simultaneously**. The model outputs 45 binary predictions.

### Class Imbalance
Diseases have very different prevalence:
- Common: DR (Diabetic Retinopathy)
- Ultra-rare: Some diseases appear in <10 images

**Solution:** Weighted BCEWithLogitsLoss with per-class `pos_weight` calculated from training data.

### GPU Memory Management
EfficientNet-B4 at 512×512 with batch_size=20 uses ~5.8GB. Solutions if you hit OOM:
- Reduce `BATCH_SIZE` in `config.py`
- Enable gradient accumulation
- Use mixed precision (already enabled via `torch.cuda.amp`)

## Next Phase

**Phase 2** (not in this repo yet) will build a web app/CLI that:
1. Loads `outputs/checkpoints/best_model.pt`
2. Calls `inference.py` on user-uploaded images
3. Returns disease predictions + confidence scores + advisory messages

## For New Developers

**Starting points:**
1. Read [SETUP.md](./SETUP.md) and get environment running
2. Skim [ARCHITECTURE.md](./ARCHITECTURE.md) to understand the model
3. Run sanity check (`python3 train.py --epochs 2`)
4. Check [TRAINING.md](./TRAINING.md) for hyperparameter tuning
5. See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) if issues arise

**Questions?** Each doc has examples and code snippets.

---

**Last Updated:** 2026-02-22
**Status:** Phase 1 (Training) — Complete
