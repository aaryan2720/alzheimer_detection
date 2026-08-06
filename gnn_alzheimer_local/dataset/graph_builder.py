# =============================================================================
# dataset/graph_builder.py — MRI Image → Graph Converter
# =============================================================================
#
# SCIENTIFIC RATIONALE
# --------------------
# Alzheimer's Disease (AD) affects spatially distributed brain regions in
# an interconnected manner.  Cortical atrophy, white-matter loss and
# hippocampal shrinkage are not isolated — they form structured patterns
# across anatomical neighbourhoods.
#
# A CNN treats the MRI as a rectangular pixel grid and learns local
# spatial filters. It does NOT explicitly model the relationships between
# non-adjacent brain regions.
#
# A Graph representation lets us model:
#   • Nodes  → local brain patches (regions of interest)
#   • Edges  → structural/functional similarity between regions
#   • GNN    → message-passing over this relational structure
#
# This is biologically motivated: connected brain regions often degenerate
# together in AD, so relational learning can capture group-level patterns
# that pixel-level convolution misses.
#
# GRAPH CONSTRUCTION PIPELINE
# ----------------------------
#   1. Load & resize MRI to 224×224
#   2. Divide into PATCH_GRID × PATCH_GRID non-overlapping patches
#      → Each patch = one graph node
#   3. Extract node features:
#        Raw patch pixels  (H_p × W_p × 3) flattened
#        + Optional: HOG descriptor, LBP texture features
#        → PCA-reduce to NODE_FEAT_DIM (128)
#   4. Construct edges:
#        a) Spatial adjacency: connect 8-connected grid neighbours
#        b) Feature k-NN   : connect KNN_K most similar nodes by
#                            cosine similarity of node features
#        (Union of both edge sets)
#   5. Edge attributes: cosine similarity scores
#   6. Return PyG Data(x, edge_index, edge_attr, y)
# =============================================================================

from __future__ import annotations

import os
import numpy as np
import cv2
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# PCA singleton — fitted once per dataset build, reused for all images
# ---------------------------------------------------------------------------

class _PCACache:
    _pca: PCA | None = None

    @classmethod
    def get(cls, n_components: int) -> PCA:
        if cls._pca is None or cls._pca.n_components != n_components:
            cls._pca = PCA(n_components=n_components, random_state=C.SEED)
        return cls._pca

    @classmethod
    def reset(cls):
        cls._pca = None


# ---------------------------------------------------------------------------
# MRIGraphBuilder
# ---------------------------------------------------------------------------

class MRIGraphBuilder:
    """
    Converts a single MRI image file into a PyTorch Geometric Data object.

    Parameters
    ----------
    patch_grid  : int  — number of patches along each axis (default 14 → 196 nodes)
    feat_dim    : int  — PCA output dimension for node features
    knn_k       : int  — number of k-NN feature neighbours per node
    pca         : PCA | None — fitted PCA transformer (None = fit on first call)
    """

    def __init__(
        self,
        patch_grid: int = C.PATCH_GRID,
        feat_dim: int   = C.NODE_FEAT_DIM,
        knn_k: int      = C.KNN_K,
        pca: PCA | None = None,
    ):
        self.patch_grid  = patch_grid
        self.feat_dim    = feat_dim
        self.knn_k       = knn_k
        self.num_nodes   = patch_grid * patch_grid
        self.patch_h     = C.IMG_HEIGHT // patch_grid
        self.patch_w     = C.IMG_WIDTH  // patch_grid
        self.pca         = pca          # injected after dataset-level PCA fit
        self._raw_dim    = self.patch_h * self.patch_w * C.IMG_CHANNELS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, image_path: str, label: int) -> Data:
        """
        Read an MRI image and return a PyG Data object.

        Parameters
        ----------
        image_path : str  — absolute or relative path to the image file
        label      : int  — class index

        Returns
        -------
        torch_geometric.data.Data with fields:
            x          : float tensor [num_nodes, feat_dim]
            edge_index : long  tensor [2, num_edges]
            edge_attr  : float tensor [num_edges]
            y          : long  tensor [1]
            pos        : float tensor [num_nodes, 2]  (patch centre coords)
        """
        img = self._load_image(image_path)
        raw_features = self._extract_patches(img)          # [N, raw_dim]
        node_features = self._reduce_features(raw_features) # [N, feat_dim]
        edge_index, edge_attr = self._build_edges(node_features)
        pos = self._patch_centres()

        data = Data(
            x          = torch.tensor(node_features, dtype=torch.float),
            edge_index = edge_index,
            edge_attr  = torch.tensor(edge_attr,    dtype=torch.float),
            y          = torch.tensor([label],       dtype=torch.long),
            pos        = torch.tensor(pos,           dtype=torch.float),
        )
        return data

    def build_batch(self, paths: list[str], labels: list[int]) -> list[Data]:
        """Build graphs for a list of (path, label) pairs."""
        return [self.build(p, l) for p, l in zip(paths, labels)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess an MRI image to float32 [0,1] RGB array."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.resize(img, (C.IMG_WIDTH, C.IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img  # shape (H, W, 3)

    def _extract_patches(self, img: np.ndarray) -> np.ndarray:
        """
        Divide the image into patch_grid × patch_grid non-overlapping patches.
        Each patch is flattened to a 1-D feature vector.

        Returns
        -------
        raw_features : np.ndarray shape [num_nodes, raw_dim]
        """
        patches = []
        for r in range(self.patch_grid):
            for c in range(self.patch_grid):
                y0 = r * self.patch_h
                y1 = y0 + self.patch_h
                x0 = c * self.patch_w
                x1 = x0 + self.patch_w
                patch = img[y0:y1, x0:x1]          # (patch_h, patch_w, 3)
                patches.append(patch.flatten())     # → 1-D
        return np.stack(patches, axis=0)            # (N, raw_dim)

    def _reduce_features(self, raw: np.ndarray) -> np.ndarray:
        """
        Reduce raw patch features to feat_dim via PCA.
        If PCA not fitted yet, return raw features (caller fits PCA externally).
        """
        if self.pca is not None and hasattr(self.pca, "components_"):
            return self.pca.transform(raw)
        # Fallback: simple normalisation (used when PCA not yet fitted)
        norm = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        reduced = (raw / norm)[:, : self.feat_dim]
        if reduced.shape[1] < self.feat_dim:
            pad = np.zeros((reduced.shape[0], self.feat_dim - reduced.shape[1]), dtype=np.float32)
            reduced = np.concatenate([reduced, pad], axis=1)
        return reduced.astype(np.float32)

    def _build_edges(self, features: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        Construct edge_index and edge_attr tensors.

        Two types of edges are combined:
        (a) Spatial adjacency — 8-connected neighbours in the patch grid
        (b) Feature k-NN     — KNN_K closest nodes by cosine similarity

        Returns
        -------
        edge_index : LongTensor  [2, E]
        edge_attr  : ndarray     [E]   (cosine similarity, ∈ [-1, 1])
        """
        N = self.num_nodes
        edges = set()

        # (a) Spatial 8-connected adjacency
        for r in range(self.patch_grid):
            for c in range(self.patch_grid):
                u = r * self.patch_grid + c
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.patch_grid and 0 <= cc < self.patch_grid:
                            v = rr * self.patch_grid + cc
                            edges.add((min(u, v), max(u, v)))

        # (b) Feature k-NN edges
        nn = NearestNeighbors(n_neighbors=min(self.knn_k + 1, N), metric="cosine")
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        for i in range(N):
            for j, dist in zip(indices[i][1:], distances[i][1:]):   # skip self
                u, v = min(i, j), max(i, j)
                edges.add((u, v))

        # Build symmetric edge_index
        src_list, dst_list = [], []
        for u, v in edges:
            src_list.extend([u, v])
            dst_list.extend([v, u])

        # Self-loops
        if C.EDGE_SELF_LOOP:
            src_list.extend(range(N))
            dst_list.extend(range(N))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Compute cosine similarity as edge attribute
        feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        edge_attr = np.array([
            float(np.dot(feat_norm[s], feat_norm[d]))
            for s, d in zip(src_list, dst_list)
        ], dtype=np.float32)

        return edge_index, edge_attr

    def _patch_centres(self) -> np.ndarray:
        """Return [num_nodes, 2] array of (row_centre, col_centre) in pixels."""
        centres = []
        for r in range(self.patch_grid):
            for c in range(self.patch_grid):
                cy = r * self.patch_h + self.patch_h / 2
                cx = c * self.patch_w + self.patch_w / 2
                centres.append([cy, cx])
        return np.array(centres, dtype=np.float32)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def visualize_graph(
        self,
        image_path: str,
        label: int,
        save_path: str | None = None,
        max_edges: int = 300,
    ) -> None:
        """
        Side-by-side visualization: MRI image + constructed graph.

        Requires networkx and matplotlib.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        data = self.build(image_path, label)
        img  = self._load_image(image_path)
        pos_arr = data.pos.numpy()           # (N, 2) — (row, col)
        pos_dict = {i: (pos_arr[i, 1], C.IMG_HEIGHT - pos_arr[i, 0])
                    for i in range(data.num_nodes)}

        # Build networkx graph (undirected, sample edges for clarity)
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        ei = data.edge_index.numpy()
        ea = data.edge_attr.numpy()
        for k in range(0, min(ei.shape[1], max_edges * 2), 2):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                G.add_edge(u, v, weight=float(ea[k]))

        # Node colours by row (simulate spatial gradient)
        node_colors = [plt.cm.viridis(i / data.num_nodes) for i in range(data.num_nodes)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: original MRI
        axes[0].imshow(img)
        axes[0].set_title(
            f"Original MRI\nClass: {C.DISPLAY_CLASS_NAMES[label]}",
            fontsize=13, fontweight="bold"
        )
        # Overlay patch grid
        for r in range(self.patch_grid + 1):
            axes[0].axhline(r * self.patch_h - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
        for c in range(self.patch_grid + 1):
            axes[0].axvline(c * self.patch_w - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
        axes[0].axis("off")

        # Right: graph
        nx.draw_networkx_nodes(G, pos_dict, ax=axes[1],
                               node_color=node_colors, node_size=20, alpha=0.9)
        edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos_dict, ax=axes[1],
                               width=0.4, alpha=0.4, edge_color=edge_weights,
                               edge_cmap=plt.cm.plasma)
        axes[1].set_title(
            f"Constructed Graph\n{data.num_nodes} nodes · {G.number_of_edges()} edges",
            fontsize=13, fontweight="bold"
        )
        axes[1].set_xlim(0, C.IMG_WIDTH)
        axes[1].set_ylim(0, C.IMG_HEIGHT)
        axes[1].set_aspect("equal")
        axes[1].axis("off")

        fig.suptitle(
            "MRI to Graph Conversion — Brain Region Relational Modelling",
            fontsize=14, fontweight="bold", y=1.02
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.show()
        plt.close(fig)
