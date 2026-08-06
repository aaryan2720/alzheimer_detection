# =============================================================================
# train.py — Training Loop for GAT Alzheimer's Detection
# =============================================================================

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as GeoDataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import save_checkpoint, save_history, logger, Timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : int   — number of epochs to wait
    mode      : str   — 'min' (loss) or 'max' (accuracy)
    delta     : float — minimum improvement threshold
    """

    def __init__(self, patience: int = C.EARLY_STOP_PAT, mode: str = "min", delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.stop      = False

    def __call__(self, value: float) -> bool:
        improved = (
            value < self.best - self.delta if self.mode == "min"
            else value > self.best + self.delta
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model:       nn.Module,
    loader:      GeoDataLoader,
    criterion:   nn.Module,
    optimizer:   optim.Optimizer | None,
    device:      torch.device,
    phase:       str = "train",
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    is_train = (phase == "train")
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss   = criterion(logits, batch.y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds       = logits.argmax(dim=-1)
            correct    += (preds == batch.y).sum().item()
            total      += batch.num_graphs

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training curves plot
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, figures_dir: str) -> None:
    """Save training / validation accuracy and loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GAT Training History", fontsize=16, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, history["train_acc"],  label="Train Accuracy",  color="#2196F3")
    axes[0].plot(epochs, history["val_acc"],    label="Val Accuracy",    color="#FF5722", linestyle="--")
    axes[0].set_title("Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Loss
    axes[1].plot(epochs, history["train_loss"], label="Train Loss",      color="#4CAF50")
    axes[1].plot(epochs, history["val_loss"],   label="Val Loss",        color="#FF9800", linestyle="--")
    axes[1].set_title("Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    acc_path  = os.path.join(figures_dir, "training_accuracy.png")
    loss_path = os.path.join(figures_dir, "training_loss.png")
    fig.tight_layout()
    fig.savefig(acc_path,  dpi=300, bbox_inches="tight")
    logger.info(f"Accuracy curve saved → {acc_path}")
    fig.clf()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, history["train_loss"], label="Train Loss", color="#4CAF50")
    ax2.plot(epochs, history["val_loss"],   label="Val Loss",   color="#FF9800", linestyle="--")
    ax2.set_title("GAT Loss Curves", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Loss curve saved → {loss_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: GeoDataLoader,
    val_loader:   GeoDataLoader,
    device:       torch.device,
    best_model_path: str = C.BEST_MODEL_PATH,
    history_path:    str = C.HISTORY_PKL,
    figures_dir:     str = C.FIGURES_DIR,
) -> dict:
    """
    Full training loop with:
      • Adam optimizer
      • ReduceLROnPlateau scheduler
      • EarlyStopping
      • ModelCheckpoint (best val loss)

    Returns
    -------
    history : dict with keys train_loss, val_loss, train_acc, val_acc, lr
    """
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        model.parameters(),
        lr=C.LEARNING_RATE,
        weight_decay=C.WEIGHT_DECAY
    )
    scheduler  = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=C.LR_MIN
    )
    stopper    = EarlyStopping(patience=C.EARLY_STOP_PAT, mode="min")

    history    = defaultdict(list)
    best_val_loss = float("inf")
    total_timer = Timer("Total Training")
    total_timer.start()

    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting GAT Training — {C.MAX_EPOCHS} max epochs")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*60}\n")

    for epoch in range(1, C.MAX_EPOCHS + 1):
        epoch_timer = Timer(f"Epoch {epoch}")
        epoch_timer.start()

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, "train")
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion, None,      device, "val")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = epoch_timer.stop()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        logger.info(
            f"Epoch [{epoch:03d}/{C.MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # ModelCheckpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_model_path,
            )
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # EarlyStopping
        if stopper(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    history["total_time"] = total_timer.stop()
    history = dict(history)

    logger.info(f"\nTraining complete — {history['total_time']:.1f}s total")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")

    save_history(history, history_path)
    plot_training_curves(history, figures_dir)

    return history
