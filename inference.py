"""
inference.py — Single-image inference for the retinal disease classifier.

Usage (CLI):
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --threshold 0.4

Usage (as module, for Phase 2 web app):
    from inference import load_model, predict
    model = load_model()
    result = predict("path/to/image.png", model=model)
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image

import config
from model import build_model


def load_model(checkpoint_path: str = config.BEST_MODEL_PATH) -> torch.nn.Module:
    """Load model from checkpoint. Call once and reuse the returned model."""
    model = build_model(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()
    return model


def _preprocess(image_path: str) -> torch.Tensor:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def predict(
    image_path: str,
    model: torch.nn.Module = None,
    threshold: float = 0.5,
    checkpoint_path: str = config.BEST_MODEL_PATH,
) -> dict:
    """
    Run inference on a single retinal image.

    Args:
        image_path:      Path to PNG/JPG retinal fundus image.
        model:           Pre-loaded model (pass to avoid reloading on every call).
        threshold:       Sigmoid probability threshold to call a disease positive.
        checkpoint_path: Path to .pt checkpoint (used only if model is None).

    Returns:
        {
            "disease_risk": True/False,
            "predictions": {"DR": 0.87, "ARMD": 0.03, ...},   # all 45 probs
            "detected_diseases": ["DR"],                        # above threshold
            "num_detected": 1,
        }
    """
    if model is None:
        model = load_model(checkpoint_path)

    tensor = _preprocess(image_path).to(config.DEVICE)
    logits = model(tensor)                          # (1, 45)
    probs  = torch.sigmoid(logits).cpu().numpy()[0] # (45,)

    predictions = {name: float(f"{prob:.4f}") for name, prob in zip(config.LABEL_COLS, probs)}
    detected    = [name for name, prob in zip(config.LABEL_COLS, probs) if prob >= threshold]

    return {
        "disease_risk": len(detected) > 0,
        "predictions": predictions,
        "detected_diseases": detected,
        "num_detected": len(detected),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retinal disease inference")
    parser.add_argument("--image",     required=True,        help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH, help="Model checkpoint path")
    args = parser.parse_args()

    result = predict(args.image, threshold=args.threshold, checkpoint_path=args.checkpoint)
    print(json.dumps(result, indent=2))
