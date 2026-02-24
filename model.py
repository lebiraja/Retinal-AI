import torch.nn as nn
import torchvision.models as models

import config


def build_model(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """EfficientNet-B4 with custom multi-label classification head."""
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_param_groups(model: nn.Module, lr: float) -> list:
    """Differential learning rates: backbone at lr*0.1, head at lr."""
    return [
        {"params": model.features.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
