"""
ModelService — Singleton model loader.

The model is downloaded once from HuggingFace on startup and kept in
GPU/CPU memory for the entire lifetime of the server.  Never call
ModelService.load() inside a request handler.

Checkpoint format (from Team-B-Backend training):
    { 'epoch', 'model_state_dict', 'optimizer_state_dict', 'mean_auc' }

Architecture:
    EfficientNet-B4 (ImageNet pretrained) with a custom head:
        Sequential(Dropout(0.4), Linear(1792, 45))
    Defined in Team-B-Backend/model.py — imported directly.

Note: Run via `uvicorn app.main:app --app-dir <Team-B-Backend path>` so
that the Team-B-Backend root is on sys.path and `model.py` is importable.
"""

import logging

import torch
from huggingface_hub import hf_hub_download

from app.config import HF_MODEL_NAME

# model.py lives at Team-B-Backend/model.py.
# uvicorn --app-dir ensures Team-B-Backend/ is on sys.path.
from model import build_model  # noqa: E402  (Team-B-Backend/model.py)

logger = logging.getLogger(__name__)


class ModelService:
    _model: torch.nn.Module | None = None
    _device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Public API ────────────────────────────────────────────────────── #

    @classmethod
    def load(cls) -> None:
        """
        Download the training checkpoint from HuggingFace (cached after
        first run), extract model_state_dict, and load into EfficientNet-B4.

        Called exactly once from the FastAPI startup event.
        """
        if cls._model is not None:
            logger.info("Model already loaded — skipping.")
            return

        logger.info("Downloading checkpoint for '%s' ...", HF_MODEL_NAME)
        checkpoint_path = hf_hub_download(
            repo_id=HF_MODEL_NAME,
            filename="pytorch_model.bin",
        )

        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        state_dict = checkpoint["model_state_dict"]
        epoch    = checkpoint.get("epoch",    "?")
        mean_auc = checkpoint.get("mean_auc", 0.0)
        logger.info("Checkpoint  epoch=%s | mean_auc=%.4f", epoch, mean_auc)

        model = build_model()
        model.load_state_dict(state_dict, strict=True)
        model.to(cls._device)
        model.eval()

        cls._model = model
        logger.info(
            "EfficientNet-B4 ready on device: %s", cls._device
        )

    @classmethod
    def get_model(cls) -> torch.nn.Module:
        if cls._model is None:
            raise RuntimeError(
                "Model has not been loaded. Call ModelService.load() at startup."
            )
        return cls._model

    @classmethod
    def get_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def device_str(cls) -> str:
        return str(cls._device)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None
