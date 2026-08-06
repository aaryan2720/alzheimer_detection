# =============================================================================
# utils.py — Utility Functions for GAT Alzheimer's Detection
# =============================================================================

import os
import random
import logging
import time
import pickle
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "alzheimer_gat", log_file: str | None = None) -> logging.Logger:
    """Configure and return a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect and return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({mem:.1f} GB VRAM) — using CUDA")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected — using CPU")
    return device


# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------

def create_output_dirs(*dirs: str) -> None:
    """Create output directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Output directories ready: {dirs}")


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved → {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None) -> dict:
    """Load a training checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded ← {filepath}")
    return checkpoint


def save_history(history: dict, filepath: str) -> None:
    """Persist training history as a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"Training history saved → {filepath}")


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> dict:
    """Return trainable and non-trainable parameter counts + estimated size."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    # Estimate size in MB (float32 = 4 bytes)
    size_mb = total * 4 / 1e6
    return {
        "trainable":     trainable,
        "non_trainable": non_trainable,
        "total":         total,
        "size_mb":       round(size_mb, 3),
    }


def print_model_summary(model: torch.nn.Module, name: str = "Model") -> None:
    """Print a compact model summary."""
    info = count_parameters(model)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name} Summary")
    print(sep)
    print(model)
    print(sep)
    print(f"  Trainable Parameters  : {info['trainable']:,}")
    print(f"  Non-Trainable Params  : {info['non_trainable']:,}")
    print(f"  Total Parameters      : {info['total']:,}")
    print(f"  Estimated Model Size  : {info['size_mb']} MB")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer with context manager support."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        logger.info(f"{label}Elapsed time: {self.elapsed:.2f}s")

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed


# ---------------------------------------------------------------------------
# Plot Helpers
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "lines.linewidth":   2.0,
}

plt.rcParams.update(PLOT_STYLE)


def save_fig(fig: plt.Figure, filepath: str, tight: bool = True) -> None:
    """Save a matplotlib figure at publication quality."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {filepath}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
