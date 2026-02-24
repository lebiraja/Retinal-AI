# Retinal Disease Classifier — Backend API

> **Team B · EfficientNet-B4 · 45 Retinal Diseases · Mean AUC 0.82**

A FastAPI backend that performs multi-label retinal disease screening from fundus images using a fine-tuned **EfficientNet-B4** model trained on the [RFMiD dataset](https://riadd.grand-challenge.org/).

---

## Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Disease Labels](#disease-labels)
- [Risk Level Logic](#risk-level-logic)
- [Configuration](#configuration)
- [Medical Disclaimer](#medical-disclaimer)

---

## Overview

| Property | Value |
|----------|-------|
| Framework | FastAPI + Uvicorn |
| ML Model | EfficientNet-B4 (torchvision) |
| Task | Multi-label image classification |
| Classes | 45 retinal diseases |
| Input size | 384 × 384 px |
| Device | CUDA (falls back to CPU) |
| Model weights | [HuggingFace — lebiraja/retinal-disease-classifier](https://huggingface.co/lebiraja/retinal-disease-classifier) |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Best epoch | 39 / 50 |
| Mean AUC-ROC | **0.8204** |
| Train loss | 0.2118 |
| Val loss | 0.2578 |
| Macro F1 | 0.1517 |
| Micro F1 | 0.4450 |

**Architecture:**
```
EfficientNet-B4 (ImageNet pretrained)
  └── Custom head:
        ├── Dropout(p=0.4)
        └── Linear(1792 → 45)
```

The checkpoint is automatically downloaded from HuggingFace on first startup and cached at `~/.cache/huggingface/`. Subsequent starts load from cache in under 2 seconds.

---

## Project Structure

```
Team-B-Backend/
├── app/
│   ├── main.py                  # FastAPI entry point, CORS, lifespan
│   ├── config.py                # API constants (labels, thresholds, limits)
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   └── predict.py           # All HTTP route handlers
│   └── services/
│       ├── model_service.py     # Singleton model loader (HuggingFace)
│       ├── inference_service.py # Image preprocessing + forward pass
│       └── advisory_service.py  # Risk scoring + advisory text
├── model.py                     # EfficientNet-B4 architecture definition
├── config.py                    # Training constants (paths, hyperparams)
├── dataset.py                   # PyTorch dataset (training only)
├── train.py                     # Training script
├── inference.py                 # Standalone CLI inference script
├── hf_config.json               # HuggingFace repo metadata
├── requirements.txt             # Python dependencies
└── run.sh                       # Helper script to start the server
```

---

## Requirements

- Python **3.10+**
- CUDA-capable GPU recommended (automatically falls back to CPU)
- ~500 MB disk space for model weights (downloaded once, then cached)

---

## Installation

### 1. Navigate to the project directory

```bash
cd Team-B-Backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

### Development (with auto-reload)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1`. The model is a singleton held in memory — multiple workers would each load their own copy, multiplying VRAM usage.

### Using the helper script

```bash
chmod +x run.sh && ./run.sh
```

### Standalone CLI inference (no server required)

```bash
python inference.py --image path/to/fundus.png
python inference.py --image path/to/fundus.png --threshold 0.4
```

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

### `GET /health`

Server liveness check.

**Response `200 OK`**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### `GET /info`

Model metadata and training metrics.

**Response `200 OK`**
```json
{
  "model_name": "Retinal Disease Classifier",
  "version": "1.0",
  "architecture": "EfficientNet-B4 + Dropout(0.4) + Linear(1792, 45)",
  "num_classes": 45,
  "input_size": 384,
  "diseases": ["DR", "ARMD", "MH", "..."],
  "metrics": {
    "mean_auc": 0.8204,
    "train_loss": 0.2118,
    "val_loss": 0.2578,
    "macro_f1": 0.1517,
    "micro_f1": 0.445
  },
  "huggingface": "https://huggingface.co/lebiraja/retinal-disease-classifier"
}
```

---

### `POST /predict`

Classify diseases in a single fundus image.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✅ | Fundus image — JPEG, PNG, BMP, or TIFF (max 10 MB) |
| `threshold` | float | ❌ | Detection threshold `0.0–1.0` (default: `0.5`) |

**Example — curl**
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@fundus.jpg" \
  -F "threshold=0.5"
```

**Response `200 OK`**
```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.7821,
    "MCA": 0.6103
  },
  "detected_diseases": ["DR", "MCA"],
  "detected_diseases_full": ["Diabetic Retinopathy", "Microaneurysms"],
  "num_detected": 2,
  "top_prediction": "DR",
  "confidence": 0.6962,
  "risk_level": "HIGH",
  "advisory": "Detected indicator(s): Diabetic Retinopathy. Signs consistent with significant retinal pathology were detected. This is not a medical diagnosis. Prompt evaluation by a certified ophthalmologist is strongly recommended.",
  "elapsed_ms": 18.43,
  "threshold": 0.5,
  "disclaimer": "This tool is NOT a medical diagnostic device. Results are for screening purposes only. Always consult a qualified ophthalmologist for clinical decisions."
}
```

**Error responses**

| Status | Cause |
|--------|-------|
| `400` | Unsupported file type or empty file |
| `413` | File exceeds 10 MB |
| `422` | `threshold` outside `0.0–1.0` |
| `500` | Inference failure |

---

### `POST /predict-batch`

Classify diseases in multiple fundus images in one request (max 10).

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `images` | file[] | ✅ | 1–10 fundus images |
| `threshold` | float | ❌ | Detection threshold applied to every image (default: `0.5`) |

**Example — curl**
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "threshold=0.4"
```

**Response `200 OK`**
```json
{
  "results": [
    { "...PredictionResponse for img1..." },
    { "...PredictionResponse for img2..." }
  ],
  "total_images": 2,
  "total_elapsed_ms": 35.21
}
```

Each item in `results` has the same shape as the `/predict` response.

---

## Disease Labels

45 conditions supported, in model output order:

| # | Code | Full Name |
|---|------|-----------|
| 0 | DR | Diabetic Retinopathy |
| 1 | ARMD | Age-Related Macular Degeneration |
| 2 | MH | Macular Hole |
| 3 | DN | Drusen |
| 4 | MYA | Myopic Astigmatism |
| 5 | BRVO | Branch Retinal Vein Occlusion |
| 6 | TSLN | Tessellation |
| 7 | ERM | Epiretinal Membrane |
| 8 | LS | Laser Scar |
| 9 | MS | Macular Scar |
| 10 | CSR | Central Serous Retinopathy |
| 11 | ODC | Optic Disc Cupping |
| 12 | CRVO | Central Retinal Vein Occlusion |
| 13 | TV | Tire Venture |
| 14 | AH | Anterior Chamber |
| 15 | ODP | Optic Disc Pallor |
| 16 | ODE | Optic Disc Edema |
| 17 | ST | Shunt |
| 18 | AION | Anterior Ischemic Optic Neuropathy |
| 19 | PT | Parafoveal Telangiectasia |
| 20 | RT | Retinal Traction |
| 21 | RS | Retinal Scar |
| 22 | CRS | Corneal Reflex Shadow |
| 23 | EDN | Exudates |
| 24 | RPEC | RPE Changes |
| 25 | MHL | Macular Hole (Large) |
| 26 | RP | Retinitis Pigmentosa |
| 27 | CWS | Cotton Wool Spots |
| 28 | CB | Conjunctival Bleed |
| 29 | ODPM | Optic Disc Pallor Margin |
| 30 | PRH | Peripapillary Retinal Hemorrhage |
| 31 | MNF | Macular Neovascularization |
| 32 | HR | Hard Retinal Exudate |
| 33 | CRAO | Central Retinal Artery Occlusion |
| 34 | TD | Temporal Disc |
| 35 | CME | Cystoid Macular Edema |
| 36 | PTCR | Posterior Capsular Rent |
| 37 | CF | Cotton Fiber |
| 38 | VH | Vitreous Hemorrhage |
| 39 | MCA | Microaneurysms |
| 40 | VS | Vitreous Synchysis |
| 41 | BRAO | Branch Retinal Artery Occlusion |
| 42 | PLQ | Placoid Lesion |
| 43 | HPED | Hemorrhagic Pigment Epithelial Detachment |
| 44 | CL | Cotton Lint |

---

## Risk Level Logic

| Condition | Risk Level |
|-----------|------------|
| No diseases detected | **LOW** |
| 1–2 diseases, none high-risk | **MODERATE** |
| 3+ diseases OR any 1 high-risk disease | **HIGH** |
| 2+ high-risk diseases | **CRITICAL** |

**High-risk disease codes:** `DR`, `ARMD`, `CRVO`, `CRAO`, `VH`, `AION`, `CME`, `HPED`, `BRAO`, `BRVO`, `MNF`, `RP`

---

## Configuration

All API constants live in [app/config.py](app/config.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_THRESHOLD` | `0.5` | Sigmoid detection threshold |
| `MAX_FILE_SIZE_MB` | `10` | Upload size limit per file |
| `IMAGE_SIZE` | `384` | Input resolution (px) |
| `HF_MODEL_NAME` | `lebiraja/retinal-disease-classifier` | HuggingFace repo ID |
| `NORM_MEAN` | `(0.485, 0.456, 0.406)` | ImageNet normalisation mean |
| `NORM_STD` | `(0.229, 0.224, 0.225)` | ImageNet normalisation std |

Training constants (batch size, LR, paths) are in the root [config.py](config.py).

---

## Medical Disclaimer

> This system is **NOT** a medical diagnostic device. Results are for screening and educational purposes only. Always consult a qualified ophthalmologist for clinical decisions.

---

**Team B · Last updated: February 2026 · MIT License**
