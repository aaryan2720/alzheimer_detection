# =============================================================================
# evaluate.py — Comprehensive Evaluation & Visualizations for GAT Model
# =============================================================================
#
# Generates ALL metrics expected in a high-quality IEEE/Springer research paper
# and publication-quality figures.
# =============================================================================

from __future__ import annotations

import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import logger, save_fig

# Publication palette
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model:  torch.nn.Module,
    loader: GeoDataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run model on loader and collect y_true, y_pred, y_probs.

    Returns
    -------
    y_true      : [N]        integer class labels
    y_pred      : [N]        predicted class indices
    y_probs     : [N, C]     softmax probabilities
    inference_s : float      total inference time (seconds)
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    t0 = time.perf_counter()
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_true.extend(batch.y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    inference_s = time.perf_counter() - t0

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_probs),
        inference_s,
    )


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    y_probs:     np.ndarray,
    inference_s: float,
    model:       torch.nn.Module,
    history:     dict,
) -> dict:
    """
    Compute comprehensive evaluation metrics for IEEE/Springer report.
    """
    n = len(y_true)

    # --- Basic ---
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)

    # --- Per-class Sensitivity / Specificity ---
    cm = confusion_matrix(y_true, y_pred)
    sensitivity_list = []
    specificity_list = []
    for i in range(C.NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity_list.append(tp / (tp + fn + 1e-8))
        specificity_list.append(tn / (tn + fp + 1e-8))

    # --- Macro / Weighted ---
    mac_p  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    mac_r  = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    mac_f1 = f1_score(       y_true, y_pred, average="macro",    zero_division=0)
    wgt_p  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    wgt_r  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    wgt_f1 = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    # --- ROC AUC ---
    try:
        roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    # --- Log Loss ---
    ll = log_loss(y_true, y_probs)

    # --- MCC + Kappa ---
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # --- Per-class accuracy ---
    per_class_acc = {
        C.DISPLAY_CLASS_NAMES[i]: float(cm[i, i] / (cm[i, :].sum() + 1e-8))
        for i in range(C.NUM_CLASSES)
    }

    # --- Model size ---
    from utils import count_parameters
    param_info = count_parameters(model)

    # --- Training summary ---
    total_epochs = len(history.get("train_loss", []))
    total_time_s = history.get("total_time", 0.0)

    metrics = {
        # Accuracy family
        "Accuracy":             round(acc, 4),
        "Balanced Accuracy":    round(bal_acc, 4),

        # Precision
        "Macro Precision":      round(mac_p, 4),
        "Weighted Precision":   round(wgt_p, 4),

        # Recall / Sensitivity
        "Macro Recall":         round(mac_r, 4),
        "Weighted Recall":      round(wgt_r, 4),
        "Mean Sensitivity":     round(float(np.mean(sensitivity_list)), 4),
        "Mean Specificity":     round(float(np.mean(specificity_list)), 4),

        # F1
        "Macro F1":             round(mac_f1, 4),
        "Weighted F1":          round(wgt_f1, 4),

        # AUC / Curves
        "ROC AUC (Macro OvR)":  round(roc_auc, 4),

        # Information-theoretic
        "Log Loss":             round(ll, 4),

        # Agreement
        "MCC":                  round(mcc, 4),
        "Cohen Kappa":          round(kappa, 4),

        # Efficiency
        "Parameters":           param_info["total"],
        "Model Size (MB)":      param_info["size_mb"],
        "Inference Time (s)":   round(inference_s, 3),
        "Inference/Sample (ms)": round(inference_s * 1000 / n, 3),
        "Training Time (s)":    round(total_time_s, 1),
        "Training Epochs":      total_epochs,

        # Per class
        "Per Class Accuracy":   per_class_acc,
        "Sensitivity per class": {C.DISPLAY_CLASS_NAMES[i]: round(sensitivity_list[i], 4) for i in range(C.NUM_CLASSES)},
        "Specificity per class": {C.DISPLAY_CLASS_NAMES[i]: round(specificity_list[i], 4) for i in range(C.NUM_CLASSES)},
    }

    return metrics, cm


# ---------------------------------------------------------------------------
# Print evaluation summary
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  GAT MODEL — EVALUATION RESULTS")
    print(sep)
    skip = {"Per Class Accuracy", "Sensitivity per class", "Specificity per class"}
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<35} {v}")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=C.DISPLAY_CLASS_NAMES))

    print("  Per-Class Accuracy:")
    for cls, acc in metrics["Per Class Accuracy"].items():
        print(f"    {cls:<25} {acc:.4f}")

    print("  Per-Class Sensitivity:")
    for cls, s in metrics["Sensitivity per class"].items():
        print(f"    {cls:<25} {s:.4f}")

    print("  Per-Class Specificity:")
    for cls, s in metrics["Specificity per class"].items():
        print(f"    {cls:<25} {s:.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Save classification report
# ---------------------------------------------------------------------------

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=C.DISPLAY_CLASS_NAMES)
    with open(filepath, "w") as f:
        f.write("GAT Model — Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    logger.info(f"Classification report saved → {filepath}")


# ---------------------------------------------------------------------------
# Save metrics CSV
# ---------------------------------------------------------------------------

def save_metrics_csv(metrics: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    flat = {k: v for k, v in metrics.items()
            if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(filepath, index=False)
    logger.info(f"Metrics CSV saved → {filepath}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_dataset_distribution(figures_dir: str) -> None:
    """Bar chart of class distribution across train, val, test splits."""
    from dataset.alzheimer_dataset import _discover_images, _train_val_split
    train_paths, train_labels = _discover_images(C.TRAIN_DIR)
    tr_p, tr_l, va_p, va_l   = _train_val_split(train_paths, train_labels)
    te_p, te_l                = _discover_images(C.TEST_DIR)

    splits = {"Train": tr_l, "Val": va_l, "Test": te_l}
    counts = {s: [lbs.count(i) for i in range(C.NUM_CLASSES)] for s, lbs in splits.items()}

    x    = np.arange(C.NUM_CLASSES)
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (split, cnts) in enumerate(counts.items()):
        ax.bar(x + j * w, cnts, width=w, label=split, color=PALETTE[j], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(C.DISPLAY_CLASS_NAMES, rotation=15, ha="right")
    ax.set_xlabel("Class"); ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Distribution Across Splits", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    os.makedirs(figures_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "dataset_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dataset distribution plot saved")


def plot_confusion_matrices(cm: np.ndarray, figures_dir: str) -> None:
    """Raw and normalized confusion matrices."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=C.DISPLAY_CLASS_NAMES,
        yticklabels=C.DISPLAY_CLASS_NAMES,
        ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].tick_params(axis="y", rotation=0)

    # Normalized
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=C.DISPLAY_CLASS_NAMES,
        yticklabels=C.DISPLAY_CLASS_NAMES,
        ax=axes[1], linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=0, vmax=1
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle("GAT Model — Confusion Matrices", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved")


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, figures_dir: str) -> None:
    """Per-class + Macro ROC curves."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    all_auc = []
    for i, (cname, color) in enumerate(zip(C.DISPLAY_CLASS_NAMES, PALETTE)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cname} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest) — GAT Model", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.text(0.5, 0.05, f"Macro AUC = {np.mean(all_auc):.3f}",
            transform=ax.transAxes, fontsize=12, ha="center",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved")


def plot_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray, figures_dir: str) -> None:
    """Per-class Precision-Recall curves."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (cname, color) in enumerate(zip(C.DISPLAY_CLASS_NAMES, PALETTE)):
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
        ap = average_precision_score((y_true == i).astype(int), y_probs[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"{cname} (AP = {ap:.3f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — GAT Model", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "precision_recall_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved")


def plot_prediction_samples(
    model:     torch.nn.Module,
    test_loader: GeoDataLoader,
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    figures_dir: str,
    n_show:    int = 10,
) -> None:
    """Show random, correct, and misclassified sample predictions."""
    os.makedirs(figures_dir, exist_ok=True)

    # Collect all image paths from test set
    all_paths  = []
    all_labels = []
    ds = test_loader.dataset
    for i in range(len(ds)):
        all_paths.append(ds.paths[i])
        all_labels.append(ds.labels[i])

    import cv2
    def load_img(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((C.IMG_HEIGHT, C.IMG_WIDTH, 3), dtype=np.uint8)
        img = cv2.resize(img, (C.IMG_WIDTH, C.IMG_HEIGHT))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    n = min(n_show, len(all_paths))
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(all_paths), n, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    fig.suptitle("Sample Predictions — GAT Model", fontsize=14, fontweight="bold")

    for i, sample_idx in enumerate(idx[:10]):
        img = load_img(all_paths[sample_idx])
        axes[i].imshow(img)
        t = C.DISPLAY_CLASS_NAMES[y_true[sample_idx]]
        p = C.DISPLAY_CLASS_NAMES[y_pred[sample_idx]]
        color = "green" if y_true[sample_idx] == y_pred[sample_idx] else "red"
        axes[i].set_title(f"T: {t}\nP: {p}", color=color, fontsize=8)
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "prediction_samples.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction samples saved")

    # Correct predictions
    correct_idx = np.where(y_true == y_pred)[0]
    if len(correct_idx) > 0:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        fig.suptitle("Correctly Classified Samples", fontsize=13, fontweight="bold")
        chosen = rng.choice(correct_idx, min(5, len(correct_idx)), replace=False)
        for i, cidx in enumerate(chosen):
            axes[i].imshow(load_img(all_paths[cidx]))
            axes[i].set_title(f"T: {C.DISPLAY_CLASS_NAMES[y_true[cidx]]}\nP: {C.DISPLAY_CLASS_NAMES[y_pred[cidx]]}", color="green", fontsize=8)
            axes[i].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "correct_predictions.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Misclassified
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) > 0:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        fig.suptitle("Misclassified Samples", fontsize=13, fontweight="bold")
        chosen = rng.choice(wrong_idx, min(5, len(wrong_idx)), replace=False)
        for i, widx in enumerate(chosen):
            axes[i].imshow(load_img(all_paths[widx]))
            axes[i].set_title(f"T: {C.DISPLAY_CLASS_NAMES[y_true[widx]]}\nP: {C.DISPLAY_CLASS_NAMES[y_pred[widx]]}", color="red", fontsize=8)
            axes[i].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "misclassified_samples.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info("Prediction sample plots saved")


def plot_cnn_vs_gat(metrics: dict, figures_dir: str) -> None:
    """
    Side-by-side comparison bar chart: CNN Baseline vs GAT.
    """
    os.makedirs(figures_dir, exist_ok=True)

    keys = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
            "Weighted Precision", "Weighted Recall", "Weighted F1", "ROC AUC (Macro OvR)"]

    cnn_vals = [
        C.CNN_METRICS.get("Accuracy",         0),
        C.CNN_METRICS.get("Macro Precision",   0),
        C.CNN_METRICS.get("Macro Recall",      0),
        C.CNN_METRICS.get("Macro F1",          0),
        C.CNN_METRICS.get("Weighted Precision",0),
        C.CNN_METRICS.get("Weighted Recall",   0),
        C.CNN_METRICS.get("Weighted F1",       0),
        C.CNN_METRICS.get("ROC AUC",           0),
    ]
    gat_vals = [metrics.get(k, 0) for k in keys]

    x  = np.arange(len(keys))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - w/2, cnn_vals, w, label="CNN Baseline", color="#FF5722", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, gat_vals, w, label="GAT (Ours)",  color="#2196F3", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    short_keys = ["Accuracy", "Mac. Prec", "Mac. Rec", "Mac. F1",
                  "Wgt. Prec", "Wgt. Rec", "Wgt. F1", "ROC AUC"]
    ax.set_xticklabels(short_keys, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("CNN Baseline vs. GAT — Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color="#0D47A1")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "cnn_vs_gat_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("CNN vs GAT comparison plot saved")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model:        torch.nn.Module,
    test_loader:  GeoDataLoader,
    device:       torch.device,
    history:      dict,
    figures_dir:  str  = C.FIGURES_DIR,
    results_dir:  str  = C.RESULTS_DIR,
    metrics_csv:  str  = C.METRICS_CSV,
    class_report: str  = C.CLASS_REPORT,
) -> dict:
    """
    Run full evaluation pipeline.
    Produces all metrics, figures, CSV, and classification report.
    """
    logger.info("\nStarting evaluation …")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Collect predictions ---
    y_true, y_pred, y_probs, inference_s = collect_predictions(model, test_loader, device)

    # --- Metrics ---
    metrics, cm = compute_metrics(y_true, y_pred, y_probs, inference_s, model, history)
    print_metrics(metrics, y_true, y_pred)

    # --- Save ---
    save_metrics_csv(metrics, metrics_csv)
    save_classification_report(y_true, y_pred, class_report)

    # --- Figures ---
    plot_confusion_matrices(cm, figures_dir)
    plot_roc_curves(y_true, y_probs, figures_dir)
    plot_precision_recall_curves(y_true, y_probs, figures_dir)
    plot_cnn_vs_gat(metrics, figures_dir)
    plot_prediction_samples(model, test_loader, y_true, y_pred, figures_dir)

    logger.info("Evaluation complete")
    return metrics
