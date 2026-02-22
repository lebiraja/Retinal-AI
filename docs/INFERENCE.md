# Inference Guide

How to use the trained model for predictions on new retinal images.

---

## Quick Start

### Via Command Line

```bash
python3 inference.py --image path/to/fundus_image.png
```

**Output:**
```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.8765,
    "ARMD": 0.1234,
    "MH": 0.0456,
    ...
  },
  "detected_diseases": ["DR"],
  "num_detected": 1
}
```

### Via Python (for Phase 2 web app)

```python
from inference import load_model, predict

# Load once (reuse across many predictions)
model = load_model()

# Predict on image
result = predict("path/to/image.png", model=model)

print(result["disease_risk"])      # True/False
print(result["detected_diseases"]) # ["DR", "ARMD"]
print(result["predictions"]["DR"]) # 0.8765
```

---

## Command Line Usage

### Basic

```bash
python3 inference.py --image dataset/Training_Set/Training_Set/Training/1.png
```

### Custom Threshold

Default threshold is 0.5. Change to adjust sensitivity:

```bash
python3 inference.py --image image.png --threshold 0.3
# Lower threshold → more diseases detected (higher sensitivity)

python3 inference.py --image image.png --threshold 0.7
# Higher threshold → fewer diseases detected (higher specificity)
```

### Custom Model Checkpoint

```bash
python3 inference.py --image image.png --checkpoint outputs/checkpoints/last_model.pt
```

### Help

```bash
python3 inference.py --help
```

---

## Python API

### Function: `load_model()`

```python
from inference import load_model
import config

model = load_model(checkpoint_path=config.BEST_MODEL_PATH)
model.eval()  # Already set in load_model()
```

**Returns:** PyTorch model ready for inference.

**Parameters:**
- `checkpoint_path` (str): Path to `.pt` checkpoint (default: `outputs/checkpoints/best_model.pt`)

**Example:**
```python
model = load_model("outputs/checkpoints/best_model.pt")
```

### Function: `predict()`

```python
from inference import predict

result = predict(
    image_path="path/to/image.png",
    model=None,                              # Optional: pre-loaded model
    threshold=0.5,                          # Disease detection threshold
    checkpoint_path="outputs/checkpoints/best_model.pt"  # If model is None
)
```

**Returns:** Dictionary with:

```python
{
    "disease_risk": bool,                    # True if any disease detected
    "predictions": {
        "DR": float,                         # Disease prob for each of 45 diseases
        "ARMD": float,
        ...
    },
    "detected_diseases": [str],              # Diseases above threshold
    "num_detected": int,                     # Number of detected diseases
}
```

**Parameters:**
- `image_path` (str): Path to PNG/JPG retinal fundus image
- `model` (torch.nn.Module, optional): Pre-loaded model to reuse
- `threshold` (float): Probability threshold [0, 1] to classify as disease present
- `checkpoint_path` (str): Path to model checkpoint (if `model=None`)

**Example:**
```python
result = predict("retina.png", threshold=0.4)
if result["disease_risk"]:
    print(f"Detected: {result['detected_diseases']}")
```

---

## Output Format

### Example Response

```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.8765,
    "ARMD": 0.1234,
    "MH": 0.0456,
    "DN": 0.0123,
    "MYA": 0.0089,
    "BRVO": 0.0045,
    "TSLN": 0.0034,
    "ERM": 0.0156,
    "LS": 0.0012,
    "MS": 0.0078,
    "CSR": 0.0234,
    "ODC": 0.3456,
    ... (45 total diseases)
  },
  "detected_diseases": ["DR", "ODC"],
  "num_detected": 2
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `disease_risk` | bool | True if any disease detected above threshold |
| `predictions` | dict | Per-disease sigmoid probabilities [0, 1] |
| `detected_diseases` | list | Diseases above threshold, sorted by probability |
| `num_detected` | int | Count of detected diseases |

---

## Disease List (45 Classes)

```python
LABEL_COLS = [
    "DR",      # Diabetic Retinopathy
    "ARMD",    # Age-Related Macular Degeneration
    "MH",      # Myopia
    "DN",      # Drusen
    "MYA",     # Myopic Astigmatism
    "BRVO",    # Branch Retinal Vein Occlusion
    "TSLN",    # Tessellation
    "ERM",     # Epiretinal Membrane
    "LS",      # Laser Scar
    "MS",      # Macular Scar
    "CSR",     # Central Serous Retinopathy
    "ODC",     # Optic Disc Cupping
    "CRVO",    # Central Retinal Vein Occlusion
    "TV",      # Tire Venture
    "AH",      # Anterior Chamber
    "ODP",     # Optic Disc Pallor
    "ODE",     # Optic Disc Edema
    "ST",      # Shunt
    "AION",    # Anterior Ischemic Optic Neuropathy
    "PT",      # Parafoveal Telangiectasia
    "RT",      # Retinal Traction
    "RS",      # Retinal Scar
    "CRS",     # Corneal Reflex Shadow
    "EDN",     # Exudates
    "RPEC",    # RPE Changes
    "MHL",     # Macular Hole
    "RP",      # Retinitis Pigmentosa
    "CWS",     # Cotton Wool Spots
    "CB",      # Conjunctival Bleed
    "ODPM",    # Optic Disc Pallor Margin
    "PRH",     # Peripapillary Retinal Hemorrhage
    "MNF",     # Macular Neovascularization
    "HR",      # Hard Retinal Exudate
    "CRAO",    # Central Retinal Artery Occlusion
    "TD",      # Temporal Disc
    "CME",     # Cystoid Macular Edema
    "PTCR",    # Posterior Capsular Rent
    "CF",      # Cotton Fiber
    "VH",      # Vitreous Hemorrhage
    "MCA",     # Microaneurysms
    "VS",      # Vitreous Synchysis
    "BRAO",    # Branch Retinal Artery Occlusion
    "PLQ",     # Placoid Lesion
    "HPED",    # Hemorrhagic Pigment Epithelial Detachment
    "CL",      # Cotton Lint
]
```

---

## Threshold Tuning

The default threshold of **0.5** balances sensitivity and specificity. Adjust for your use case:

### Clinical Use Cases

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| Screening (flag for review) | 0.3–0.4 | High sensitivity, okay with false positives |
| Confirmatory diagnosis | 0.6–0.7 | High specificity, okay with false negatives |
| Research | 0.5 | Balanced |
| Ultra-conservative | 0.8+ | Very high confidence |

### Example: Adjust Threshold

```python
from inference import predict

# Sensitive (catch more cases, more false positives)
result = predict("image.png", threshold=0.3)
print(len(result["detected_diseases"]))  # Likely higher

# Specific (fewer false alarms, miss some cases)
result = predict("image.png", threshold=0.7)
print(len(result["detected_diseases"]))  # Likely lower
```

---

## Batch Prediction

For processing multiple images:

```python
import os
from inference import load_model, predict

model = load_model()  # Load once

results = {}
image_dir = "dataset/Training_Set/Training_Set/Training"

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        result = predict(image_path, model=model)
        results[filename] = result

# Print results
for filename, result in results.items():
    print(f"{filename}: {result['num_detected']} diseases detected")
```

---

## Performance Characteristics

### Latency

- **Single image:** ~200–500 ms (depends on GPU utilization)
- **Batch of 10:** ~1000–1500 ms (better throughput)

### Memory

- **Model weights:** ~75 MB
- **Per image:** ~10 MB (for buffering)
- **Total VRAM needed:** ~500 MB (even with 6GB GPU)

### Accuracy (Expected)

After 50 epochs training, expect per-disease performance:

| Disease Prevalence | Mean AUC | Notes |
|-------------------|----------|-------|
| Common (>5%) | 0.90–0.95 | DR, ARMD, MH |
| Moderate (1–5%) | 0.75–0.85 | CWS, HR, etc. |
| Rare (<1%) | 0.60–0.75 | Ultra-rare diseases harder |

**Overall mean AUC:** ~0.85–0.92

---

## Deployment Integration (Phase 2)

### Web API Example

```python
# fastapi_app.py (for Phase 2 web server)
from fastapi import FastAPI, File, UploadFile
from inference import load_model, predict
import tempfile
import json

app = FastAPI()
model = load_model()  # Load at startup

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = predict(tmp_path, model=model, threshold=0.5)
    return result
```

### CLI Integration

```bash
# Batch processing
for image in dataset/Training_Set/Training_Set/Training/*.png; do
    python3 inference.py --image "$image" | jq .detected_diseases
done
```

---

## Error Handling

### Missing Image File

```python
try:
    result = predict("nonexistent.png")
except FileNotFoundError:
    print("Image not found")
```

### Invalid Checkpoint

```python
try:
    result = predict("image.png", checkpoint_path="bad_path.pt")
except (FileNotFoundError, RuntimeError) as e:
    print(f"Checkpoint error: {e}")
```

### GPU Issues

```python
import torch

if not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU (slower)")
    # Model will fall back to CPU automatically
```

---

## Advanced: Custom Preprocessing

If you need different image preprocessing (e.g., custom resolution):

```python
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from model import build_model

# Custom transform
custom_transform = A.Compose([
    A.Resize(256, 256),  # Different size
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load image and transform
image = np.array(Image.open("image.png").convert("RGB"))
tensor = custom_transform(image=image)["image"].unsqueeze(0)

# Inference
model = build_model()
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
```

---

## Troubleshooting

### Issue: "FileNotFoundError: best_model.pt not found"

**Solution:** Run training first or specify correct checkpoint path:
```bash
python3 train.py --epochs 2  # Quick training
python3 inference.py --image image.png  # Now works
```

### Issue: "CUDA out of memory"

**Note:** Inference uses much less VRAM than training. If you still get OOM:
```python
# Force CPU
import config
config.DEVICE = torch.device("cpu")
```

### Issue: "Image dimensions incorrect"

**Note:** Input image is automatically resized to 512×512. Supports any input size.

### Issue: "Model predicts all zeros"

Check:
1. Image is valid PNG/JPG
2. Image is actual fundus photograph (not random image)
3. Model was trained (best_model.pt exists and has reasonable size)

---

## References

- [ImageNet Normalization](https://pytorch.org/vision/main/models.html)
- [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [RFMiD Dataset](https://www.rfmid.org/)

---

**Last Updated:** 2026-02-22
