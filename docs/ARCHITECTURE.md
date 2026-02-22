# Architecture & Design

This document explains the model architecture, loss functions, and data processing pipeline.

## Overview

The system is a **multi-label image classifier** that predicts the presence or absence of 45 retinal diseases in a single fundus (retinal) image.

```
Input Image (512×512)
        ↓
[Preprocessing & Augmentation]
        ↓
[EfficientNet-B4 Backbone]
        ↓
[Classification Head: FC(1792→45)]
        ↓
Output: 45 logits → sigmoid → [0, 1] probabilities
```

---

## Model Architecture

### Backbone: EfficientNet-B4

**Why EfficientNet-B4?**

- **Efficiency:** Best accuracy-to-VRAM ratio for medical imaging
- **Proven:** State-of-the-art on ImageNet, pretrained weights available
- **Medical imaging track record:** Used in multiple published retinal disease papers
- **VRAM friendly:** ~5.8 GB at 512×512 with batch_size=20 on RTX 4050

**Comparison with alternatives:**

| Model | Params | VRAM@512 | Accuracy | Speed | Notes |
|-------|--------|----------|----------|-------|-------|
| ResNet-50 | 25.5M | ~6.2GB | Good | Slow | Older, heavier |
| **EfficientNet-B4** | **19M** | **~5.8GB** | **Best** | **Fast** | **✓ Recommended** |
| EfficientNet-B3 | 12M | ~5.0GB | Slightly lower | Fastest | Conservative choice |
| MobileNetV3 | 5M | ~3GB | Lower | Fastest | Limited capacity |

### Classification Head

```python
nn.Sequential(
    nn.Dropout(p=0.4),           # 40% dropout for regularization
    nn.Linear(1792, 45),          # EfficientNet-B4 output is 1792-dim
)
```

**Design choices:**
- **Dropout(0.4):** Prevents overfitting on 1920 training samples
- **No batch norm:** Not needed after dropout in final layer
- **45 outputs:** One sigmoid-activated output per disease (multi-label, not softmax)

### Full Model Code

```python
# model.py
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=45):
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model
```

---

## Loss Function: BCEWithLogitsLoss with pos_weight

### Why Binary Cross-Entropy?

Unlike single-label classification (softmax + CrossEntropyLoss), we use **Binary Cross-Entropy** because:

- Each disease is **independent** (image can have both DR and ARMD)
- Not mutually exclusive (unlike dog vs cat)
- Each disease → separate binary (0/1) prediction

### Class Imbalance Problem

The dataset is heavily imbalanced:

```
Disease Prevalence:
  DR:         ~35% of images
  ARMD:       ~12% of images
  MH:         ~8% of images
  ...
  Rare disease: ~0.5% of images
```

**Problem:** Without correction, the model learns to predict all-zeros (high accuracy on common negatives).

### Solution: pos_weight

```python
pos_weight[i] = (total_samples - positive_samples[i]) / positive_samples[i]
```

**Example:** If DR appears in 672 / 1920 images:
```
pos_weight[DR] = (1920 - 672) / 672 = 1.86
```

This **upweights** the loss for missing positive predictions.

**In code:**

```python
# dataset.py
def get_pos_weights(dataset):
    n = len(dataset)
    n_pos = dataset.labels.sum(axis=0)    # (45,)
    n_neg = n - n_pos
    weights = n_neg / np.clip(n_pos, a_min=1, a_max=None)
    weights = np.clip(weights, a_min=1.0, a_max=10.0)  # cap at 10
    return torch.tensor(weights, dtype=torch.float32)

# train.py
pos_weights = get_pos_weights(train_ds)  # shape (45,)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
loss = criterion(logits, labels)
```

**Why cap at 10?** Ultra-rare diseases (appearing in <50 images) would have weight >10, which can destabilize training.

---

## Optimization Strategy

### Optimizer: AdamW with Differential Learning Rates

```python
optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": LR * 0.1},      # backbone: 1e-5
    {"params": model.classifier.parameters(), "lr": LR},          # head: 1e-4
], weight_decay=1e-2)
```

**Why differential LR?**

- **Backbone (EfficientNet):** Already pretrained on ImageNet. Only needs fine-tuning → low LR
- **Head (classification layer):** Trained from scratch for our 45-disease task → higher LR

**Why AdamW?**

- Better generalization than SGD for fine-tuning
- Decoupled weight decay (L2 regularization)
- Adaptive per-parameter learning rates

### Learning Rate Schedule: CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50  # 50 epochs
)
```

**Why cosine annealing?**

- Smoothly decreases LR from initial → near zero over 50 epochs
- Formula: `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / T_max))`
- Better convergence than step-based decay

---

## Data Pipeline

### Image Processing

**Input:** 1424×2144 PNG fundus images (~2.4 MB each)
**Output:** 512×512 tensors normalized to ImageNet statistics

### Transforms Pipeline

#### Training Augmentation (dataset.py)

```python
train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),              # 50% prob
    A.VerticalFlip(p=0.3),                # 30% prob
    A.RandomBrightnessContrast(p=0.3),    # 30% prob
    A.ShiftScaleRotate(
        shift_limit=0.05,     # 5% translate
        scale_limit=0.1,      # 10% zoom
        rotate_limit=15,      # ±15 degrees
        p=0.4
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),       # ImageNet mean
        std=(0.229, 0.224, 0.225),        # ImageNet std
    ),
    ToTensorV2(),                         # → torch tensor [0, 1]
])
```

**Augmentation rationale:**

- **Flip:** Retinal anatomy is roughly symmetric; flips don't change disease presence
- **Brightness/Contrast:** Camera gain variations between different devices
- **Shift/Scale/Rotate:** Fundus images can be centered differently; small rotations occur in practice
- **Normalization:** ImageNet stats improve pretrained weight transfer

#### Validation/Test (No Augmentation)

```python
val_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

No augmentation on validation → consistent metrics across runs.

### Dataset Class

```python
# dataset.py
class RetinalDataset(Dataset):
    def __init__(self, split="train"):
        self.df = _load_df(csv_path, img_dir)  # Load CSV
        self.labels = self.df[LABEL_COLS].values.astype(np.float32)
        self.transform = _build_transforms(split == "train")

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]      # (3, 512, 512)
        label = torch.tensor(self.labels[idx])           # (45,)
        return image, label
```

**Key points:**

- CSV columns: ID, Disease_Risk, DR, ARMD, ..., CL (48 total)
- We ignore Disease_Risk and predict only the 45 diseases
- Albumentations operates on numpy arrays (faster than PIL transforms)

---

## Data Splits

| Split | Images | Ratio | Purpose |
|-------|--------|-------|---------|
| Train | 1,920 | 60% | Learn weights |
| Val | 640 | 20% | Monitor overfitting, select best model |
| Test | 640 | 20% | Final evaluation (not used during training) |

**Note:** No overlap between splits — each image appears exactly once.

---

## Training Loop Pseudocode

```
for epoch in 1..50:
    # Training
    for batch in train_loader:
        images, labels = batch
        logits = model(images)
        loss = BCEWithLogitsLoss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    for batch in val_loader:
        logits = model(images)
        loss = BCEWithLogitsLoss(logits, labels)
        probs = sigmoid(logits)
        preds = (probs >= 0.5).int()

    # Metrics
    mean_auc = average AUC-ROC across all 45 classes
    macro_f1 = F1 averaging label-wise
    micro_f1 = F1 averaging instance-wise

    # Checkpointing
    if mean_auc > best_auc:
        save best_model.pt

    # Schedule
    scheduler.step()
```

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **EfficientNet-B4** | Best VRAM/accuracy tradeoff for 6GB GPU |
| **512×512 images** | Original 1424×2144 too large; 512 preserves retinal detail |
| **Multi-label (45 outputs)** | 45 independent diseases, not mutually exclusive |
| **BCEWithLogitsLoss** | Proper loss for multi-label classification |
| **pos_weight** | Handles severe class imbalance |
| **Differential LR** | Fine-tune pretrained features carefully |
| **Cosine annealing** | Smooth LR decay improves convergence |
| **Augmentation** | Realistic variations in fundus imaging |
| **Dropout(0.4)** | 1920 images → regularization critical |

---

## Model Size & Computational Cost

### Memory

```
Model weights:    ~75 MB (EfficientNet-B4)
Batch (size 20):  ~5.8 GB
```

Total GPU memory: ~5.8 GB (fits in RTX 4050's 6GB with margin)

### Compute

- **Forward pass:** ~150ms per batch
- **Backward pass:** ~300ms per batch
- **Per epoch:** ~1920/20 = 96 batches → ~7 min
- **50 epochs:** ~350 min = ~6 hours

Actual time on RTX 4050 mobile: ~8-10 hours (lower clocks than desktop RTX 4050).

---

## Metrics Explained

### AUC-ROC (Area Under Receiver Operating Characteristic)

- **Range:** [0, 1]
- **Interpretation:** Probability that model ranks a random positive higher than random negative
- **Medical imaging standard:** Primary metric for disease classification

**Why per-class?** Each disease has different characteristics (DR easier to detect than ultra-rare disease).

### F1 Score

- **Macro F1:** Average F1 across classes (weights all diseases equally)
- **Micro F1:** F1 computed globally (weights by instance frequency)

**Interpretation:** Harmonic mean of precision & recall. Good for imbalanced data.

### Threshold

- **Default:** 0.5 (standard for binary classification)
- **Tunable:** Can adjust for higher sensitivity (lower threshold) or specificity (higher threshold)

---

## Forward Pass Data Flow

```
Input image (PIL):
    ↓ [Convert to numpy]
numpy array (H, W, 3):
    ↓ [Albumentations transform]
tensor [0, 1] (3, 512, 512):
    ↓ [Batch with others → (20, 3, 512, 512)]
batch tensor:
    ↓ [EfficientNet backbone]
features (20, 1792):
    ↓ [Dropout]
features (20, 1792):
    ↓ [Linear layer]
logits (20, 45):
    ↓ [Training: loss.backward()]
    ↓ [Inference: sigmoid → probs]
predictions (20, 45) ∈ [0, 1]
```

---

## References & Further Reading

- **EfficientNet paper:** Tan & Le (2019) — "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **Multi-label learning:** Zhang & Zhou (2014) — "A Review on Multi-Label Learning Algorithms"
- **Class imbalance:** He & Garcia (2009) — "Learning from Imbalanced Data"
- **Retinal disease imaging:** https://www.rfmid.org/ (RFMiD dataset)

---

**Last Updated:** 2026-02-22
