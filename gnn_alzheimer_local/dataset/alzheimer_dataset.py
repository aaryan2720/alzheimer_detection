# =============================================================================
# dataset/alzheimer_dataset.py — PyTorch Geometric Dataset + DataLoaders
# =============================================================================
#
# Mirrors the CNN baseline data pipeline exactly:
#   • Same folder structure: AugmentedAlzheimerDataset/ and OriginalDataset/
#   • Same 80/20 train/validation split
#   • Same 4 class labels
#   • Same image size (224×224)
#   • Normalization: pixel values rescaled to [0, 1]  (identical to CNN)
#
# The dataset is built once and optionally cached to disk to avoid
# re-computing graphs on every run.
# =============================================================================

from __future__ import annotations

import os
import glob
import pickle
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from dataset.graph_builder import MRIGraphBuilder

logger = logging.getLogger("alzheimer_gat")


# ---------------------------------------------------------------------------
# Image path discovery (mirrors CNN flow_from_directory logic)
# ---------------------------------------------------------------------------

def _discover_images(root_dir: str) -> tuple[list[str], list[int]]:
    """
    Scan root_dir for sub-folders matching C.CLASS_NAMES and collect
    (image_path, label_index) pairs.

    Returns
    -------
    paths  : list of str
    labels : list of int
    """
    paths, labels = [], []

    # Map sub-folder name → class index (case-insensitive, strip spaces)
    name_map = {n.lower().replace(" ", ""): i for i, n in enumerate(C.CLASS_NAMES)}
    alt_map = {n.lower().replace(" ", ""): i for i, n in enumerate(C.DISPLAY_CLASS_NAMES)}

    for entry in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, entry)
        if not os.path.isdir(folder):
            continue

        key = entry.lower().replace(" ", "")
        label = name_map.get(key, alt_map.get(key, None))
        if label is None:
            logger.warning(f"Skipping unrecognised folder: {entry}")
            continue

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for ext in exts:
            for p in glob.glob(os.path.join(folder, ext)):
                paths.append(p)
                labels.append(label)

    logger.info(f"Found {len(paths)} images in {root_dir}")
    return paths, labels


def _train_val_split(
    paths: list[str],
    labels: list[int],
    val_split: float = C.VAL_SPLIT,
    seed: int = C.SEED,
) -> tuple[list, list, list, list]:
    """Stratified split mirroring ImageDataGenerator(validation_split=val_split)."""
    rng = random.Random(seed)

    # Group by class
    class_paths = {i: [] for i in range(C.NUM_CLASSES)}
    for p, l in zip(paths, labels):
        class_paths[l].append(p)

    train_p, train_l, val_p, val_l = [], [], [], []
    for cls_idx, cls_paths in class_paths.items():
        rng.shuffle(cls_paths)
        
        # Apply sub-sampling ratio to reduce dataset size if configured
        if hasattr(C, 'SUB_SAMPLE_RATIO') and C.SUB_SAMPLE_RATIO < 1.0:
            num_keep = max(1, int(len(cls_paths) * C.SUB_SAMPLE_RATIO))
            cls_paths = cls_paths[:num_keep]
            
        n_val = int(len(cls_paths) * val_split)
        val_list   = cls_paths[:n_val]
        train_list = cls_paths[n_val:]
        val_p   += val_list
        val_l   += [cls_idx] * len(val_list)
        train_p += train_list
        train_l += [cls_idx] * len(train_list)

    return train_p, train_l, val_p, val_l


# ---------------------------------------------------------------------------
# AlzheimerGraphDataset — single split (train / val / test)
# ---------------------------------------------------------------------------

class AlzheimerGraphDataset(Dataset):
    """
    PyTorch Dataset that converts MRI images to graphs on the fly or from cache.

    Parameters
    ----------
    paths    : list of image file paths
    labels   : list of integer class labels
    builder  : MRIGraphBuilder (with fitted PCA)
    split    : str — "train", "val", or "test" (for logging only)
    """

    def __init__(
        self,
        paths:   list[str],
        labels:  list[int],
        builder: MRIGraphBuilder,
        split:   str = "train",
        cache_dir: str = C.GRAPH_CACHE_DIR,
    ):
        self.paths   = paths
        self.labels  = labels
        self.builder = builder
        self.split   = split
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # In-memory cache to avoid disk I/O on subsequent epochs
        self.mem_cache = [None] * len(paths)
        logger.info(f"[{split}] Dataset initialised with {len(paths)} samples (caching to disk and RAM)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Data:
        # Check in-memory RAM cache first (virtual instantaneous)
        if self.mem_cache[idx] is not None:
            return self.mem_cache[idx]

        import hashlib
        path = self.paths[idx]
        label = self.labels[idx]
        # Generate safe cache filename based on hash of image path
        hash_str = hashlib.md5(path.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{self.split}_{hash_str}.pt")

        data = None
        if os.path.exists(cache_file):
            try:
                data = torch.load(cache_file)
            except Exception:
                pass # Fall back to constructing if load fails

        if data is None:
            data = self.builder.build(path, label)
            torch.save(data, cache_file)

        # Store in RAM cache for subsequent epochs
        self.mem_cache[idx] = data
        return data

    def class_counts(self) -> dict:
        counts = {n: 0 for n in C.DISPLAY_CLASS_NAMES}
        for l in self.labels:
            counts[C.DISPLAY_CLASS_NAMES[l]] += 1
        return counts


# ---------------------------------------------------------------------------
# PCA fitting over a sample of training images
# ---------------------------------------------------------------------------

def _fit_pca(
    paths: list[str],
    builder: MRIGraphBuilder,
    n_components: int = C.NODE_FEAT_DIM,
    sample_n: int = 1000,
    seed: int = C.SEED,
) -> PCA:
    """
    Fit PCA on a random sample of training patches.

    Sampling is done for speed — 1000 images × 196 nodes = 196 000 vectors,
    which is enough to faithfully capture the principal components of the
    patch feature space.
    """
    logger.info(f"Fitting PCA (n_components={n_components}) on {sample_n} training images …")
    rng = random.Random(seed)
    sampled = rng.sample(paths, min(sample_n, len(paths)))

    all_patches = []
    for p in sampled:
        try:
            img  = builder._load_image(p)
            feats = builder._extract_patches(img)   # (N, raw_dim)
            all_patches.append(feats)
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")

    X = np.vstack(all_patches)                      # (n_samples*N, raw_dim)
    logger.info(f"PCA input shape: {X.shape}")

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X)
    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"PCA fitted — {explained:.1f}% variance explained by {n_components} components")
    return pca


# ---------------------------------------------------------------------------
# build_dataloaders — main entry point
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size:    int = C.BATCH_SIZE,
    num_workers:   int = 0,
    cache_dir:     str = C.GRAPH_CACHE_DIR,
    pca_cache_path: str | None = None,
) -> tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader, MRIGraphBuilder]:
    """
    Build train, validation, and test DataLoaders.

    Process
    -------
    1. Discover images under TRAIN_DIR (augmented)
    2. Stratified 80/20 train/val split (identical to CNN baseline)
    3. Discover images under TEST_DIR  (original)
    4. Fit PCA on training patches
    5. Build graph datasets
    6. Return DataLoaders

    Parameters
    ----------
    batch_size    : int
    num_workers   : int — set to 0 on Windows to avoid multiprocessing errors
    cache_dir     : str — directory to cache pre-built graphs
    pca_cache_path: str — optional path to cache fitted PCA transformer

    Returns
    -------
    train_loader, val_loader, test_loader, builder
    """
    # ---- 1. Discover images ------------------------------------------------
    train_all_paths, train_all_labels = _discover_images(C.TRAIN_DIR)
    test_paths,  test_labels  = _discover_images(C.TEST_DIR)

    # ---- 2. Train / Val split ----------------------------------------------
    train_paths, train_labels, val_paths, val_labels = _train_val_split(
        train_all_paths, train_all_labels
    )

    logger.info(f"Split → Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # ---- 3. Build / load PCA -----------------------------------------------
    builder_tmp = MRIGraphBuilder()   # no PCA yet — used only for raw patch extraction

    pca_path = pca_cache_path or os.path.join(cache_dir, "pca.pkl")
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)

    if os.path.exists(pca_path):
        logger.info(f"Loading cached PCA ← {pca_path}")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    else:
        pca = _fit_pca(train_paths, builder_tmp)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)
        logger.info(f"PCA saved → {pca_path}")

    # ---- 4. Build MRIGraphBuilder with fitted PCA --------------------------
    builder = MRIGraphBuilder(pca=pca)

    # ---- 5. Datasets -------------------------------------------------------
    train_ds = AlzheimerGraphDataset(train_paths, train_labels, builder, "train")
    val_ds   = AlzheimerGraphDataset(val_paths,   val_labels,   builder, "val")
    test_ds  = AlzheimerGraphDataset(test_paths,  test_labels,  builder, "test")

    # ---- 6. DataLoaders ----------------------------------------------------
    train_loader = GeoDataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = GeoDataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = GeoDataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    logger.info("DataLoaders created successfully")
    return train_loader, val_loader, test_loader, builder
