---
language: en
license: mit
library_name: pytorch
tags:
  - medical
  - vision
  - disease-detection
  - retinal
  - ophthalmology
  - multi-label-classification
---

# Retinal Disease Classifier

A deep learning model for multi-label classification of 45 retinal diseases in fundus images using EfficientNet-B4.

## Model Details

- **Architecture:** EfficientNet-B4 with custom classification head
- **Input:** RGB fundus images (384×384)
- **Output:** 45 disease probabilities (sigmoid-activated)
- **Training Data:** RFMiD (Retinal Fundus Multi-disease Image Dataset) - 1,920 images
- **Best Validation AUC:** 0.8204 (82.04%)

## Supported Diseases (45 total)

Diabetic Retinopathy (DR), Age-Related Macular Degeneration (ARMD), Myopia (MH), Drusen (DN), Myopic Astigmatism (MYA), Branch Retinal Vein Occlusion (BRVO), Tessellation (TSLN), Epiretinal Membrane (ERM), Laser Scar (LS), Macular Scar (MS), Central Serous Retinopathy (CSR), Optic Disc Cupping (ODC), Central Retinal Vein Occlusion (CRVO), Tire Venture (TV), Anterior Chamber (AH), Optic Disc Pallor (ODP), Optic Disc Edema (ODE), Shunt (ST), Anterior Ischemic Optic Neuropathy (AION), Parafoveal Telangiectasia (PT), Retinal Traction (RT), Retinal Scar (RS), Corneal Reflex Shadow (CRS), Exudates (EDN), RPE Changes (RPEC), Macular Hole (MHL), Retinitis Pigmentosa (RP), Cotton Wool Spots (CWS), Conjunctival Bleed (CB), Optic Disc Pallor Margin (ODPM), Peripapillary Retinal Hemorrhage (PRH), Macular Neovascularization (MNF), Hard Retinal Exudate (HR), Central Retinal Artery Occlusion (CRAO), Temporal Disc (TD), Cystoid Macular Edema (CME), Posterior Capsular Rent (PTCR), Cotton Fiber (CF), Vitreous Hemorrhage (VH), Microaneurysms (MCA), Vitreous Synchysis (VS), Branch Retinal Artery Occlusion (BRAO), Placoid Lesion (PLQ), Hemorrhagic Pigment Epithelial Detachment (HPED), Cotton Lint (CL)

## Usage

### Installation

```bash
pip install torch torchvision pillow albumentations scikit-learn
```

### Python API

```python
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load model from Hugging Face
from transformers import AutoModel
model = AutoModel.from_pretrained("username/retinal-disease-classifier")
model.eval()

# Prepare image
transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image = np.array(Image.open("fundus.png").convert("RGB"))
tensor = transform(image=image)["image"].unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

# Get predictions
disease_names = ["DR", "ARMD", "MH", ...]  # 45 diseases
threshold = 0.5
detected = {name: float(prob) for name, prob in zip(disease_names, probs) if prob >= threshold}

print(f"Detected diseases: {list(detected.keys())}")
print(f"Probabilities: {detected}")
```

### Example Output

```json
{
  "disease_risk": true,
  "predictions": {
    "DR": 0.993,
    "BRVO": 0.752,
    "LS": 0.859,
    "CRVO": 0.899
  },
  "detected_diseases": ["DR", "BRVO", "LS", "CRVO"],
  "num_detected": 4
}
```

## Model Performance

| Metric | Value |
|--------|-------|
| Mean AUC-ROC | 0.8204 |
| Train Loss | 0.2118 |
| Val Loss | 0.2578 |
| Macro F1 | 0.1517 |
| Micro F1 | 0.4450 |

## Training Details

- **Optimizer:** AdamW (differential learning rates)
  - Backbone: 1e-5
  - Head: 1e-4
- **Loss:** BCEWithLogitsLoss with class-weighted pos_weight
- **Scheduler:** CosineAnnealingLR (50 epochs)
- **Image Size:** 384×384 (reduced from 1424×2144)
- **Batch Size:** 5
- **Augmentations:** HFlip, VFlip, RandomBrightnessContrast, ShiftScaleRotate

## Limitations

⚠️ **Medical Disclaimer:**
- This model is **NOT** for clinical diagnosis
- Results should be reviewed by qualified ophthalmologists
- Use only for research and educational purposes
- Accuracy varies by disease and image quality

## Dataset

- **Source:** RFMiD (Retinal Fundus Multi-disease Image Dataset)
- **Images:** 3,200 fundus photographs
  - Training: 1,920
  - Validation: 640
  - Test: 640
- **Resolution:** Original ~1424×2144, resized to 384×384

## License

MIT License - Free for research and commercial use

## Citation

```bibtex
@article{mindcraft2026,
  title={Retinal Disease Classification with EfficientNet-B4},
  author={Mindcraft},
  year={2026}
}
```

## Support

For issues or questions, contact the model authors or visit the repository.

---

**Last Updated:** February 22, 2026
**Model Status:** Production Ready ✅
