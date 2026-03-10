# Backend Integration Guide

Complete guide for backend developers integrating the retinal disease classifier.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FastAPI Implementation](#fastapi-implementation)
3. [Flask Implementation](#flask-implementation)
4. [API Endpoints](#api-endpoints)
5. [Payload Formats](#payload-formats)
6. [Error Handling](#error-handling)
7. [Deployment](#deployment)
8. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

The system has been refactored from a monolithic API into a Dockerized Microservices architecture:

```
┌─────────────────────┐
│   Client / Browser  │
└──────────┬──────────┘
           │ HTTP (:80)
           ▼
┌─────────────────────┐
│ Nginx Reverse Proxy │ (Routes /api to Backend)
└──────────┬──────────┘
           │ HTTP (:8000)
           ▼
┌─────────────────────┐
│   Backend Gateway   │ (FastAPI - Validation & Logic)
└──────────┬──────────┘
           │ HTTP (:8001)
           ▼
┌─────────────────────┐
│   Model Service     │ (PyTorch/GPU - Inference Only)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   GPU Device (CUDA) │
└─────────────────────┘
```

---

## Microservices Implementation

### 1. Nginx Proxy (`nginx/nginx.conf`)

Directs all traffic:
- `/api/` ➔ Backend Gateway (`:8000`)
- `/` ➔ React Frontend (`:80`)

### 2. Backend API Gateway (`services/backend/`)

A CPU-only FastAPI service. It receives the HTTP requests, runs the lightweight **Validation Service**, and orchestrates HTTP calls to the GPU service. 

**Key Responsibilities:**
- Request parsing & CORS protection
- Blocking non-fundus images heuristics (`validation_service.py`)
- Generating Risk Advisory text (`advisory_service.py`)

### 3. Model Inference Service (`services/model/`)

A dedicated GPU container running FastAPI and PyTorch. 

**Key Responsibilities:**
- Runs a *single worker* to avoid GPU OOM conflicts.
- Loads the EfficientNet-B4 weights into VRAM.
- Dedicated strictly to PyTorch tensor transformations and model prediction.

*(The previous monolithic code examples below are retained for legacy integration reference if you wish to build a bespoke integration).*

# Global model
MODEL = None
DEVICE = None

@app.on_event("startup")
async def load_model():
    global MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = torch.load("pytorch_model.bin", map_location=DEVICE)
    MODEL.eval()
    print(f"Model loaded on {DEVICE}")

# Response schemas
class DiseaseResult(BaseModel):
    disease_risk: bool
    predictions: dict
    detected_diseases: list
    num_detected: int
    confidence: float

# Disease list
DISEASE_NAMES = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
    }

@app.post("/predict", response_model=DiseaseResult)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    """
    Predict diseases in retinal image.

    Args:
        file: PNG/JPG image file
        threshold: Disease detection threshold (0-1)

    Returns:
        DiseaseResult with predictions
    """
    try:
        # Validate input
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail="Threshold must be 0-1")

        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess
        transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        image_array = np.array(image)
        tensor = transform(image=image_array)["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = MODEL(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Parse results
        predictions = {name: float(prob) for name, prob in zip(DISEASE_NAMES, probs)}
        detected = [name for name, prob in predictions.items() if prob >= threshold]

        return DiseaseResult(
            disease_risk=len(detected) > 0,
            predictions=predictions,
            detected_diseases=detected,
            num_detected=len(detected),
            confidence=float(np.mean([predictions[d] for d in detected]) if detected else 0.0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...), threshold: float = 0.5):
    """Batch prediction for multiple images."""
    results = []
    for file in files:
        result = await predict(file, threshold)
        results.append(result)
    return {"results": results}

@app.get("/info")
async def model_info():
    return {
        "model_name": "Retinal Disease Classifier",
        "version": "1.0",
        "num_classes": 45,
        "diseases": DISEASE_NAMES,
        "input_size": 384,
        "metrics": {
            "best_auc": 0.8204,
            "train_loss": 0.2118,
            "val_loss": 0.2578,
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run Server

```bash
# Development
uvicorn app:app --reload

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@image.png" \
  -F "threshold=0.5"

# Model info
curl http://localhost:8000/info

# Swagger docs
http://localhost:8000/docs
```

---

## Flask Implementation

### Installation

```bash
pip install flask torch torchvision pillow albumentations
```

### Basic Server

```python
# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = torch.load("pytorch_model.bin", map_location=DEVICE)
MODEL.eval()

DISEASE_NAMES = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Single image prediction."""
    try:
        # Get parameters
        threshold = float(request.args.get("threshold", 0.5))

        # Validate
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if threshold < 0 or threshold > 1:
            return jsonify({"error": "Threshold must be 0-1"}), 400

        # Load image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess
        transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        image_array = np.array(image)
        tensor = transform(image=image_array)["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = MODEL(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Results
        predictions = {name: float(prob) for name, prob in zip(DISEASE_NAMES, probs)}
        detected = [name for name, prob in predictions.items() if prob >= threshold]

        return jsonify({
            "disease_risk": len(detected) > 0,
            "predictions": predictions,
            "detected_diseases": detected,
            "num_detected": len(detected),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_name": "Retinal Disease Classifier",
        "version": "1.0",
        "num_classes": 45,
        "diseases": DISEASE_NAMES,
        "metrics": {"best_auc": 0.8204},
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)
```

### Run Server

```bash
# Development
python app.py

# Production (with gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## API Endpoints

### POST /predict

Predict diseases in single image.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@fundus.png" \
  -F "threshold=0.5"
```

**Response (200 OK):**
```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.993,
    "CRVO": 0.899,
    "LS": 0.859,
    ...
  },
  "detected_diseases": ["DR", "CRVO", "LS"],
  "num_detected": 3
}
```

**Errors:**
- 400: Invalid file or threshold
- 500: Server error

---

### POST /predict-batch (FastAPI only)

**Request:**
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "threshold=0.5"
```

**Response:**
```json
{
  "results": [
    { "disease_risk": true, ... },
    { "disease_risk": false, ... }
  ]
}
```

---

### GET /health

Check API status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true
}
```

---

### GET /info

Get model information.

**Response:**
```json
{
  "model_name": "Retinal Disease Classifier",
  "version": "1.0",
  "num_classes": 45,
  "diseases": ["DR", "ARMD", ...],
  "metrics": {
    "best_auc": 0.8204,
    "train_loss": 0.2118,
    "val_loss": 0.2578
  }
}
```

---

## Payload Formats

### Input

```
Content-Type: multipart/form-data

- file: Binary image data (PNG/JPG)
- threshold: Float 0-1 (optional, default 0.5)
```

### Output

```json
{
  "disease_risk": boolean,
  "predictions": {
    "disease_name": probability (0-1),
    ...
  },
  "detected_diseases": [disease_names],
  "num_detected": integer,
  "confidence": average_probability (optional)
}
```

---

## Error Handling

```python
class APIError(Exception):
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code

@app.errorhandler(APIError)
def handle_error(error):
    return jsonify({"error": error.message}), error.status_code

# Usage
if not valid:
    raise APIError("Invalid threshold", 400)
```

---

## Deployment

### Docker Compose (Recommended)

The entire architecture is configured in `docker-compose.yml` to automatically handle port routing, environment variable injection, and GPU provisioning.

```yaml
# Extract from docker-compose.yml
services:
  model-service:
    build:
      context: .
      dockerfile: services/model/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  backend:
    build:
      context: .
      dockerfile: services/backend/Dockerfile
    environment:
      - MODEL_SERVICE_URL=http://model-service:8001
    
  nginx:
    # Routes to the backend
    ports:
      - "80:80"
```

Build & Run:
```bash
docker compose up --build -d
```

---

## Performance Optimization

### Batch Processing

```python
@app.post("/predict-batch")
async def batch_predict(files: list[UploadFile]):
    # Load all images
    tensors = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image=np.array(image))["image"]
        tensors.append(tensor)

    # Batch inference
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs  # (N, 45)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_disease_info(disease_name):
    return DISEASE_INFO.get(disease_name, {})
```

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict-async")
async def predict_async(file: UploadFile):
    # Run inference in thread pool
    result = await asyncio.get_event_loop().run_in_executor(
        executor,
        predict_sync,
        file
    )
    return result
```

### GPU Optimization

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    logits = MODEL(tensor)
    probs = torch.sigmoid(logits)

# Batch inference
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # Process batch
```

---

## Testing

```python
# test_api.py
import requests
import io
from PIL import Image

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

def test_predict():
    # Create dummy image
    image = Image.new("RGB", (384, 384))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("test.png", img_bytes, "image/png")}
    response = requests.post(f"{BASE_URL}/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "disease_risk" in data
    assert "predictions" in data

if __name__ == "__main__":
    test_health()
    test_predict()
    print("✅ All tests passed")
```

Run tests:
```bash
python test_api.py
```

---

**Last Updated:** February 22, 2026
**Status:** Production Ready ✅
