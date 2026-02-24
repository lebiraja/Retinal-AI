import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def _build_transforms(train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _load_df(csv_path: str, img_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["filename"] = df["ID"].astype(str) + ".png"
    df["filepath"] = df["filename"].apply(lambda f: os.path.join(img_dir, f))
    return df


class RetinalDataset(Dataset):
    def __init__(self, split: str = "train"):
        """
        Args:
            split: one of 'train', 'val', 'test'
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.train = split == "train"
        self.transform = _build_transforms(self.train)

        if split == "train":
            self.df = _load_df(config.TRAIN_CSV, config.TRAIN_IMG_DIR)
        elif split == "val":
            self.df = _load_df(config.VALID_CSV, config.VALID_IMG_DIR)
        else:
            self.df = _load_df(config.TEST_CSV, config.TEST_IMG_DIR)

        self.labels = self.df[config.LABEL_COLS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.iloc[idx]["filepath"]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_pos_weights(dataset: RetinalDataset) -> torch.Tensor:
    """Compute per-class positive weights for BCEWithLogitsLoss."""
    n = len(dataset)
    n_pos = dataset.labels.sum(axis=0)           # shape (45,)
    n_neg = n - n_pos
    weights = n_neg / np.clip(n_pos, a_min=1, a_max=None)
    weights = np.clip(weights, a_min=1.0, a_max=config.POS_WEIGHT_CLIP)
    return torch.tensor(weights, dtype=torch.float32)
