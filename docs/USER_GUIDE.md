# Retinal Disease Classifier - User Guide

Complete guide for using the retinal disease classifier model.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Quick Start

**Install:**
```bash
pip install torch torchvision pillow albumentations scikit-learn numpy
```

**Use:**
```python
import torch
from PIL import Image
import numpy as np
from model_inference import predict_image

result = predict_image("path/to/fundus_image.png")
print(f"Detected diseases: {result['detected_diseases']}")
```

**Output:**
```json
{
  "disease_risk": true,
  "detected_diseases": ["DR", "CRVO"],
  "num_detected": 2,
  "predictions": { "DR": 0.993, "CRVO": 0.899, ... }
}
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- 4GB RAM (2GB with smaller batch sizes)
- GPU recommended (CUDA 12.1+)

### Step 1: Clone or Download

**Option A: From Hugging Face**
```bash
git clone https://huggingface.co/lebiraja/retinal-disease-classifier
cd retinal-disease-classifier
```

**Option B: Manual Download**
Download `pytorch_model.bin` from Hugging Face and place in your project directory.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch==2.1.0 torchvision==0.16.0 pillow albumentations scikit-learn numpy matplotlib tqdm
```

### Step 3: Verify Installation

```bash
python3 << 'EOF'
import torch
from PIL import Image
import albumentations
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ All dependencies OK")
EOF
```

---

## Basic Usage

### Using Command Line

```bash
# Single image
python3 inference.py --image path/to/image.png

# Batch processing
for img in *.png; do
  python3 inference.py --image "$img"
done

# Custom threshold
python3 inference.py --image image.png --threshold 0.4
```

### Using Python API

```python
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('path/to/model', 'custom', path='pytorch_model.bin')
model = model.to(device)
model.eval()

# Prepare image
transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image = np.array(Image.open("fundus.png").convert("RGB"))
tensor = transform(image=image)["image"].unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

# Parse results
disease_names = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

threshold = 0.5
results = {
    "disease_risk": False,
    "predictions": {name: float(prob) for name, prob in zip(disease_names, probs)},
    "detected_diseases": [name for name, prob in zip(disease_names, probs) if prob >= threshold],
}

results["disease_risk"] = len(results["detected_diseases"]) > 0
print(results)
```

---

## Understanding Results

### Output Format

```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.993,
    "ARMD": 0.042,
    "MH": 0.029,
    ...
  },
  "detected_diseases": ["DR"],
  "num_detected": 1
}
```

### Key Fields

| Field | Type | Meaning |
|-------|------|---------|
| `disease_risk` | bool | Any disease detected (above threshold) |
| `predictions` | dict | Probability [0,1] for all 45 diseases |
| `detected_diseases` | list | Diseases above threshold |
| `num_detected` | int | Count of detected diseases |

### Interpreting Probabilities

- **0.0 - 0.3:** Disease unlikely
- **0.3 - 0.7:** Uncertain (review recommended)
- **0.7 - 1.0:** Disease likely present

### Disease Abbreviations

| Code | Full Name | Severity |
|------|-----------|----------|
| DR | Diabetic Retinopathy | 🔴 High |
| ARMD | Age-Related Macular Degeneration | 🔴 High |
| CRVO | Central Retinal Vein Occlusion | 🔴 High |
| BRVO | Branch Retinal Vein Occlusion | 🟡 Medium |
| LS | Laser Scar | 🟡 Medium |
| MH | Myopia | 🟢 Low |
| CWS | Cotton Wool Spots | 🟢 Low |

---

## Advanced Usage

### Batch Processing with GPU

```python
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class RetinalDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        tensor = self.transform(image=image)["image"]
        return tensor

# Load model
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Create dataset
dataset = RetinalDataset(image_paths, transform)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Batch inference
all_results = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_results.append(probs)

results = np.concatenate(all_results, axis=0)  # (N, 45)
```

### Threshold Optimization

```python
# Adjust threshold based on use case
thresholds = {
    "high_sensitivity": 0.3,      # Catch more cases, more false positives
    "balanced": 0.5,              # Default
    "high_specificity": 0.7,      # Fewer false alarms, miss some cases
    "ultra_conservative": 0.85,   # Only very confident predictions
}

threshold = thresholds["balanced"]
detected = [name for name, prob in predictions.items() if prob >= threshold]
```

### Per-Disease Thresholds

```python
# Different thresholds for different diseases
per_disease_thresholds = {
    "DR": 0.4,      # DR is important, lower threshold
    "ARMD": 0.4,    # ARMD is important, lower threshold
    "MH": 0.6,      # Myopia less critical, higher threshold
    # ... default to 0.5 for others
}

detected = []
for name, prob in predictions.items():
    threshold = per_disease_thresholds.get(name, 0.5)
    if prob >= threshold:
        detected.append(name)
```

---

## Troubleshooting

### Image Issues

**Problem:** "Image has wrong dimensions"
```
Solution: Ensure image is RGB, not RGBA or grayscale
image = Image.open("image.png").convert("RGB")
```

**Problem:** "Out of memory (OOM)"
```
Solution: Reduce batch size or image resolution
batch_size = 8  # or lower
IMG_SIZE = 256  # instead of 384
```

**Problem:** "Image too small"
```
Solution: Model expects 384×384 minimum
from PIL import Image
image = image.resize((384, 384))
```

### Model Issues

**Problem:** "Model file not found"
```bash
# Download from Hugging Face
git clone https://huggingface.co/lebiraja/retinal-disease-classifier
```

**Problem:** "CUDA out of memory"
```python
# Use CPU instead
device = torch.device("cpu")
model = model.to(device)
```

**Problem:** "Predictions are all zeros"
```
Check:
1. Image is valid fundus photo (not blank/black)
2. Model file is not corrupted (check MD5)
3. Image preprocessing is correct
```

---

## FAQ

### Q: What image formats are supported?

**A:** PNG, JPG, JPEG, BMP, TIFF. Convert with:
```python
from PIL import Image
image = Image.open("image.bmp").convert("RGB").save("image.png")
```

### Q: Can I use this for clinical diagnosis?

**A:** **NO.** This is for research/educational purposes only. Always consult qualified ophthalmologists for medical decisions.

### Q: How accurate is the model?

**A:** Mean AUC: 0.8204 (82%). Accuracy varies by disease:
- Common diseases: 90-95% AUC
- Rare diseases: 60-75% AUC

### Q: Can I fine-tune on my own data?

**A:** Yes! See [DEVELOPER.md](./DEVELOPER.md) for fine-tuning instructions.

### Q: What's the difference between probabilities?

**A:**
- **Probability 0.9:** Very confident disease present
- **Probability 0.5:** Uncertain, needs review
- **Probability 0.1:** Very confident disease absent

### Q: How do I batch process images?

**A:** See "Batch Processing with GPU" section above.

### Q: Can I run on CPU only?

**A:** Yes, but ~10x slower. Set device to CPU.

### Q: What if I get different results each time?

**A:** Add `torch.manual_seed(42)` for reproducibility.

---

## Support

- **GitHub Issues:** Report bugs on the project GitHub
- **Hugging Face:** https://huggingface.co/lebiraja/retinal-disease-classifier
- **Documentation:** Check docs/ folder for detailed guides

---

**Last Updated:** February 22, 2026
**Model Version:** 1.0
**Status:** Production Ready ✅
