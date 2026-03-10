"""
Model architecture — EfficientNet-B4 for multi-label retinal disease classification.

Architecture details:
  • Backbone : EfficientNet-B4 pretrained on ImageNet
  • Head     : Dropout(0.4) → Linear(1792 → 45)
  • Loss      : BCEWithLogitsLoss with per-class positive weights
  • Output   : Raw logits (45-dim); sigmoid applied at inference time

The build_model() function must produce an architecture that is byte-for-byte
compatible with checkpoints saved during training (model.py in the repo root).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from services.model.src import config


def build_model(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """
    Construct an EfficientNet-B4 with a custom classification head.

    Args:
        num_classes: Number of output classes (default: 45).

    Returns:
        Initialised model with ImageNet-pretrained features and a fresh head.
    """
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    in_features: int = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a training checkpoint and return an eval-mode model.

    Checkpoint format (saved by train.py):
        {
            'epoch'              : int,
            'model_state_dict'   : OrderedDict,
            'optimizer_state_dict': ...,
            'mean_auc'           : float,
        }

    Args:
        checkpoint_path: Absolute path to the .bin / .pt checkpoint file.
        device         : Target device for the model.

    Returns:
        Model in eval mode, moved to `device`.
    """
    checkpoint: dict = torch.load(
        checkpoint_path,
        map_location="cpu",   # load to CPU first, then move to target device
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]
    epoch    = checkpoint.get("epoch",    "?")
    mean_auc = checkpoint.get("mean_auc", 0.0)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Checkpoint loaded — epoch=%s | mean_auc=%.4f", epoch, mean_auc)

    model = build_model()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
