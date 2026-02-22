"""
train.py — Retinal Disease Multi-Label Classification
EfficientNet-B4 | AdamW | CosineAnnealingLR | AMP | BCEWithLogitsLoss(pos_weight)

Usage:
    python train.py               # full training run
    python train.py --epochs 2    # sanity-check run
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from dataset import RetinalDataset, get_pos_weights
from model import build_model, get_param_groups


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_dirs():
    for d in (config.CHECKPOINT_DIR, config.LOG_DIR, config.PLOT_DIR):
        os.makedirs(d, exist_ok=True)


def compute_metrics(all_targets: np.ndarray, all_probs: np.ndarray):
    """
    Args:
        all_targets: (N, 45) ground truth binary labels
        all_probs:   (N, 45) sigmoid probabilities
    Returns dict with mean_auc, macro_f1, micro_f1
    """
    preds = (all_probs >= 0.5).astype(int)

    # AUC — skip classes with only one label present (undefined AUC)
    aucs = []
    for i in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, i])) > 1:
            aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, preds, average="micro", zero_division=0)

    return {"mean_auc": mean_auc, "macro_f1": macro_f1, "micro_f1": micro_f1}


def save_plots(log_csv: str):
    rows = []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    epochs   = [int(r["epoch"]) for r in rows]
    tr_loss  = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    mean_auc = [float(r["mean_auc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, tr_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, mean_auc, color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean AUC-ROC")
    axes[1].set_title("Validation Mean AUC-ROC")

    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOT_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)


# ── Training / Validation loops ────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets, all_probs = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs   = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(all_targets, all_probs)
    return total_loss / len(loader), metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main(epochs: int = config.EPOCHS):
    make_dirs()
    device = config.DEVICE
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = RetinalDataset(split="train")
    val_ds   = RetinalDataset(split="val")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = build_model().to(device)

    # Loss with pos_weight for class imbalance
    pos_weights = get_pos_weights(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer — differential LR
    optimizer = torch.optim.AdamW(
        get_param_groups(model, config.LR),
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # CSV log header
    log_exists = os.path.exists(config.LOG_CSV_PATH)
    log_file = open(config.LOG_CSV_PATH, "a", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=["epoch", "train_loss", "val_loss", "mean_auc", "macro_f1", "micro_f1", "lr"],
    )
    if not log_exists:
        log_writer.writeheader()

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mean_auc={metrics['mean_auc']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  micro_f1={metrics['micro_f1']:.4f}  "
            f"lr={current_lr:.2e}"
        )

        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "mean_auc": f"{metrics['mean_auc']:.6f}",
            "macro_f1": f"{metrics['macro_f1']:.6f}",
            "micro_f1": f"{metrics['micro_f1']:.6f}",
            "lr": f"{current_lr:.2e}",
        })
        log_file.flush()

        # Save best checkpoint
        if metrics["mean_auc"] > best_auc:
            best_auc = metrics["mean_auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_auc": best_auc,
            }, config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (mean_auc={best_auc:.4f})")

        # Always save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_auc": metrics["mean_auc"],
        }, config.LAST_MODEL_PATH)

    log_file.close()
    save_plots(config.LOG_CSV_PATH)
    print(f"\nTraining complete. Best mean AUC: {best_auc:.4f}")
    print(f"Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"Plots:       {config.PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    args = parser.parse_args()
    main(epochs=args.epochs)
