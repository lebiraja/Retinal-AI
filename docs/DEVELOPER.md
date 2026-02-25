# Developer Guide

Complete guide for developers who want to modify, fine-tune, or extend the model.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Project Structure](#project-structure)
3. [Loading & Inspecting Model](#loading--inspecting-model)
4. [Fine-Tuning on Custom Data](#fine-tuning-on-custom-data)
5. [Model Modifications](#model-modifications)
6. [Contributing](#contributing)

---

## Environment Setup

### Clone Repository

```bash
# From Hugging Face
git clone https://huggingface.co/lebiraja/retinal-disease-classifier
cd retinal-disease-classifier

# Or from your fork
git clone https://github.com/your-username/retinal-disease-classifier.git
cd retinal-disease-classifier
```

### Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Dependencies

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install -r requirements.txt
```

Or manually:
```bash
pip install \
  albumentations==1.3.0 \
  numpy==1.24.0 \
  pandas==2.0.0 \
  pillow==10.0.0 \
  scikit-learn==1.3.0 \
  matplotlib==3.7.0 \
  tqdm==4.65.0
```

### Verify Setup

```bash
python3 << 'EOF'
import torch
import albumentations
import sklearn
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print("✅ Setup OK")
EOF
```

---

## Project Structure

```
retinal-disease-classifier/
├── config.py              # Configuration (paths, hyperparameters)
├── model.py               # EfficientNet-B4 architecture
├── dataset.py             # Data loading & preprocessing
├── train.py               # Training script
├── inference.py           # Inference script
│
├── pytorch_model.bin      # Trained weights (~75 MB)
├── config.json            # Model config for HF
│
├── USER_GUIDE.md          # User documentation
├── BACKEND.md             # Backend integration guide
├── DEVELOPER.md           # This file
│
├── outputs/
│   ├── checkpoints/       # Model checkpoints
│   ├── logs/              # training_log.csv
│   └── plots/             # loss curves
│
└── docs/                  # Additional documentation
    ├── SETUP.md
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    ├── INFERENCE.md
    ├── API_REFERENCE.md
    └── TROUBLESHOOTING.md
```

---

## Loading & Inspecting Model

### Load Pretrained Model

```python
import torch
from model import build_model

# Build model
model = build_model(num_classes=45)

# Load weights
checkpoint = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# For inference
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Inspect Architecture

```python
# Print model summary
print(model)

# Get parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params / 1e6:.2f}M")
print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

# Get layer names
for name, module in model.named_modules():
    print(f"{name}: {module}")
```

### Extract Features

```python
# Get intermediate features (before classifier head)
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features  # EfficientNet backbone

    def forward(self, x):
        return self.features(x)

extractor = FeatureExtractor(model)
with torch.no_grad():
    features = extractor(tensor)  # (1, 1792, 1, 1) for 384×384 input
```

---

## Fine-Tuning on Custom Data

### Prepare Custom Dataset

```python
# custom_dataset.py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomRetinalDataset(Dataset):
    def __init__(self, csv_path, image_dir, disease_columns, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.disease_columns = disease_columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(
            self.image_dir,
            self.df.iloc[idx]["filename"]
        )
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            image = self.transform(image=image)["image"]

        # Get labels
        labels = torch.tensor(
            self.df.iloc[idx][self.disease_columns].values,
            dtype=torch.float32
        )

        return image, labels
```

### Fine-Tune Script

```python
# fine_tune.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model, get_param_groups
from custom_dataset import CustomRetinalDataset
from dataset import get_pos_weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 20
LR = 5e-5  # Lower LR for fine-tuning
IMG_SIZE = 384

# Augmentations
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load dataset
train_ds = CustomRetinalDataset(
    csv_path="custom_data/train_labels.csv",
    image_dir="custom_data/train_images",
    disease_columns=[f"disease_{i}" for i in range(45)],
    transform=transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
checkpoint = torch.load("pytorch_model.bin", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Setup training
pos_weights = get_pos_weights(train_ds).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = AdamW(
    get_param_groups(model, LR),
    weight_decay=1e-2
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    scheduler.step()

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"fine_tuned_epoch_{epoch}.pt")

print("✅ Fine-tuning complete")
```

### Run Fine-Tuning

```bash
python fine_tune.py
```

### Use Fine-Tuned Model

```python
import torch
from model import build_model

# Load fine-tuned weights
model = build_model()
checkpoint = torch.load("fine_tuned_epoch_19.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use for inference
# ... same as normal inference
```

---

## Model Modifications

### Change Number of Output Classes

```python
# model.py
def build_model(num_classes=50):  # Change from 45 to 50
    model = models.efficientnet_b4(weights=...)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),  # 50 outputs
    )
    return model
```

### Add Dropout Regularization

```python
# Increase dropout
nn.Dropout(p=0.6)  # from 0.4

# Or add dropout to backbone
for module in model.features.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.3
```

### Freeze Backbone for Transfer Learning

```python
# Freeze all backbone parameters
for param in model.features.parameters():
    param.requires_grad = False

# Only head is trainable
for param in model.classifier.parameters():
    param.requires_grad = True
```

### Use Different Backbone

```python
# Try EfficientNet-B3 (smaller)
import torchvision.models as models

model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features  # 1536 for B3
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 45),
)
```

### Add Custom Layers

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=...)
        self.features_dim = 1792

        # Add custom layers
        self.classifier = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),

            nn.Linear(256, 45),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

---

## Contributing

### Code Style

```python
# Follow PEP 8
# - Use 4 spaces for indentation
# - Max line length: 88 characters
# - Use type hints

def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    pass
```

### Create Pull Request

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test
4. Commit: `git commit -m "Add feature description"`
5. Push: `git push origin feature/my-feature`
6. Create PR on GitHub

### Testing

```python
# test_model.py
import torch
from model import build_model

def test_model_output_shape():
    model = build_model()
    model.eval()

    # Test input
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 45), f"Expected (1, 45), got {output.shape}"
    print("✅ Test passed")

if __name__ == "__main__":
    test_model_output_shape()
```

Run tests:
```bash
python -m pytest test_model.py
```

---

## Troubleshooting

### Model weights mismatch

```
RuntimeError: Error(s) in loading state_dict...
```

**Solution:** Ensure model architecture matches checkpoint:
```python
model = build_model(num_classes=45)  # Must match saved checkpoint
```

### Out of memory during fine-tuning

**Solution:** Reduce batch size or image size:
```python
BATCH_SIZE = 4  # or lower
IMG_SIZE = 256  # instead of 384
```

### Loss not decreasing

**Solution:** Check learning rate and data:
```python
LR = 1e-4  # Increase if too low
# Verify data loading works
for img, label in train_loader:
    print(img.shape, label.shape)
    break
```

---

## Resources

- **PyTorch Docs:** https://pytorch.org/docs
- **EfficientNet Paper:** https://arxiv.org/abs/1905.11946
- **Transfer Learning:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

**Last Updated:** February 22, 2026
**Status:** Production Ready ✅
