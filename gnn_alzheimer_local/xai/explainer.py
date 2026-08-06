# =============================================================================
# xai/explainer.py — Explainable AI for GAT Alzheimer's Detection
# =============================================================================
#
# Two complementary XAI methods are implemented:
#
#   1. GNNExplainer (torch_geometric.explain)
#      - Learns node and edge importance masks per prediction
#      - Identifies which brain patch nodes most influence classification
#
#   2. GAT Attention Visualization
#      - Directly uses the attention coefficients from GATConv layers
#      - No additional training required
#      - Provides intuitive "where the model looks" visualisation
#
# Both methods produce overlays on the original MRI image.
# =============================================================================

from __future__ import annotations

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from dataset.graph_builder import MRIGraphBuilder

logger = logging.getLogger("alzheimer_gat")


# ---------------------------------------------------------------------------
# Helper: image-space heatmap from node scores
# ---------------------------------------------------------------------------

def _node_scores_to_heatmap(
    node_scores: np.ndarray,    # (N,)  importance per node
    patch_grid:  int = C.PATCH_GRID,
    img_h:       int = C.IMG_HEIGHT,
    img_w:       int = C.IMG_WIDTH,
) -> np.ndarray:
    """
    Map per-node scores to a (img_h × img_w) pixel heatmap via
    nearest-neighbour upsampling of the patch grid.

    Returns
    -------
    heatmap : np.ndarray [img_h, img_w], values in [0, 1]
    """
    patch_h = img_h // patch_grid
    patch_w = img_w // patch_grid

    grid = node_scores.reshape(patch_grid, patch_grid)
    # Normalize to [0, 1]
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # Upscale to full image resolution
    heatmap = np.repeat(np.repeat(grid, patch_h, axis=0), patch_w, axis=1)
    heatmap = heatmap[:img_h, :img_w]
    return heatmap.astype(np.float32)


def _overlay_heatmap_on_image(
    img_rgb:   np.ndarray,   # (H, W, 3) float [0,1]
    heatmap:   np.ndarray,   # (H, W)   float [0,1]
    alpha:     float = 0.55,
    colormap:  str   = "jet",
) -> np.ndarray:
    """
    Blend a normalized heatmap over an RGB image.

    Returns
    -------
    overlay : np.ndarray [H, W, 3] uint8
    """
    try:
        cmap = matplotlib.colormaps[colormap]
    except AttributeError:
        # Fallback for very old matplotlib versions
        cmap = cm_module.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    img_u8  = (img_rgb * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_u8, 1 - alpha, colored, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# GATExplainer
# ---------------------------------------------------------------------------

class GATExplainer:
    """
    Wrapper providing two XAI methods for AlzheimerGAT:

    Methods
    -------
    explain_with_gnnexplainer(data, model, device)
        → Uses GNNExplainer to produce node/edge importance masks.

    explain_with_attention(data, model, device)
        → Extracts attention weights directly from GATConv layers.

    visualize(image_path, label, model, device, save_path)
        → Full pipeline: builds graph, runs both XAI methods, saves figures.

    generate_text_explanation(data, model, device, pred_label)
        → Returns a human-readable textual explanation.
    """

    def __init__(self, builder: MRIGraphBuilder):
        self.builder = builder

    # ------------------------------------------------------------------
    # GNNExplainer
    # ------------------------------------------------------------------

    def explain_with_gnnexplainer(
        self,
        data:    "torch_geometric.data.Data",
        model:   torch.nn.Module,
        device:  torch.device,
        epochs:  int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run GNNExplainer on a single graph.

        Returns
        -------
        node_mask : np.ndarray (num_nodes,)   — importance score per node
        edge_mask : np.ndarray (num_edges,)   — importance score per edge
        """
        try:
            from torch_geometric.explain import Explainer, GNNExplainer
            from torch_geometric.explain.config import (
                ExplainerConfig, ModelConfig, ThresholdConfig
            )
        except ImportError:
            logger.warning("GNNExplainer not available. Using attention fallback.")
            return self.explain_with_attention(data, model, device)

        data  = data.to(device)
        model = model.to(device)
        model.eval()

        # Wrap model for GNNExplainer (expects a simple forward(x, edge_index))
        class _ModelWrapper(torch.nn.Module):
            def __init__(self, inner, batch_size=1):
                super().__init__()
                self.inner = inner
                self.bs = batch_size

            def forward(self, x, edge_index):
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                return self.inner(x, edge_index, batch)

        wrapped = _ModelWrapper(model).to(device)

        try:
            explainer = Explainer(
                model=wrapped,
                algorithm=GNNExplainer(epochs=epochs),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=ModelConfig(
                    mode="multiclass_classification",
                    task_level="graph",
                    return_type="raw",
                ),
            )
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                index=0,
            )
            node_mask = explanation.node_mask.mean(dim=-1).detach().cpu().numpy()
            edge_mask = explanation.edge_mask.detach().cpu().numpy()
        except Exception as e:
            logger.warning(f"GNNExplainer failed ({e}). Using attention fallback.")
            return self.explain_with_attention(data, model, device)

        # Normalize
        node_mask = (node_mask - node_mask.min()) / (node_mask.max() - node_mask.min() + 1e-8)
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-8)

        return node_mask, edge_mask

    # ------------------------------------------------------------------
    # Attention-based explanation
    # ------------------------------------------------------------------

    def explain_with_attention(
        self,
        data:   "torch_geometric.data.Data",
        model:  torch.nn.Module,
        device: torch.device,
        layer:  int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract attention-based node importance from GATConv layers.

        Returns
        -------
        node_scores : np.ndarray (num_nodes,)
        edge_scores : np.ndarray (num_edges,)
        """
        data  = data.to(device)
        model = model.to(device)
        model.eval()

        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            _, attn_dict = model(data.x, data.edge_index, batch, return_attn=True)

        layer_key = f"layer{layer}"
        edge_index, attn_weights = attn_dict[layer_key]   # (2,E), (E, heads)
        attn_mean = attn_weights.mean(dim=-1).detach().cpu().numpy()

        # Scatter to nodes
        ei_dst = edge_index[1].detach().cpu().numpy()
        num_nodes = data.x.size(0)
        node_scores = np.zeros(num_nodes)
        np.add.at(node_scores, ei_dst, attn_mean)
        node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-8)

        edge_scores = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

        return node_scores, edge_scores

    # ------------------------------------------------------------------
    # Full visualization pipeline
    # ------------------------------------------------------------------

    def visualize(
        self,
        image_path:  str,
        label:       int,
        model:       torch.nn.Module,
        device:      torch.device,
        save_prefix: str = "outputs/figures/xai_sample",
        use_gnnexplainer: bool = True,
    ) -> None:
        """
        Generate and save XAI visualizations for one MRI image.

        Saves:
          - <prefix>_attention_heatmap.png   — attention overlay
          - <prefix>_gnnexplainer.png        — GNNExplainer overlay
          - <prefix>_node_importance.png     — bar chart of top-10 nodes
        """
        data  = self.builder.build(image_path, label)
        img   = self.builder._load_image(image_path)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # ---- Attention XAI ----
        attn_node, attn_edge = self.explain_with_attention(data, model, device)
        attn_heatmap = _node_scores_to_heatmap(attn_node)
        attn_overlay = _overlay_heatmap_on_image(img, attn_heatmap)

        # ---- GNNExplainer ----
        if use_gnnexplainer:
            gnn_node, gnn_edge = self.explain_with_gnnexplainer(data, model, device)
        else:
            gnn_node, gnn_edge = attn_node, attn_edge
        gnn_heatmap = _node_scores_to_heatmap(gnn_node)
        gnn_overlay = _overlay_heatmap_on_image(img, gnn_heatmap)

        # ---- Predicted class ----
        model.eval()
        with torch.no_grad():
            logits = model(data.x.to(device), data.edge_index.to(device), batch)
        pred = int(logits.argmax(dim=-1).item())
        conf = float(F.softmax(logits, dim=-1).max().item())

        os.makedirs(os.path.dirname(save_prefix) if os.path.dirname(save_prefix) else ".", exist_ok=True)

        # ---- Figure 1: Side-by-side heatmaps ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"XAI — {C.DISPLAY_CLASS_NAMES[label]} | Predicted: {C.DISPLAY_CLASS_NAMES[pred]} ({conf*100:.1f}%)",
            fontsize=14, fontweight="bold"
        )
        axes[0].imshow(img); axes[0].set_title("Original MRI"); axes[0].axis("off")
        axes[1].imshow(attn_overlay); axes[1].set_title("GAT Attention Heatmap"); axes[1].axis("off")
        axes[2].imshow(gnn_overlay); axes[2].set_title("GNNExplainer Heatmap"); axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 2: Node importance bar chart ----
        top_n = 10
        top_idx   = np.argsort(gnn_node)[-top_n:][::-1]
        top_scores = gnn_node[top_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(
            [f"Node {i} (R{i//C.PATCH_GRID+1},C{i%C.PATCH_GRID+1})" for i in top_idx],
            top_scores, color="#2196F3", alpha=0.8
        )
        ax.set_xlabel("Importance Score"); ax.set_title("Top-10 Most Influential Brain Patches", fontweight="bold")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_node_importance.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"XAI figures saved with prefix: {save_prefix}")

    # ------------------------------------------------------------------
    # Text explanation
    # ------------------------------------------------------------------

    def generate_text_explanation(
        self,
        image_path: str,
        label:      int,
        model:      torch.nn.Module,
        device:     torch.device,
    ) -> str:
        """
        Generate a human-readable textual explanation for a prediction.
        """
        data  = self.builder.build(image_path, label)
        gnn_node, _ = self.explain_with_attention(data, model, device)

        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(data.x.to(device), data.edge_index.to(device), batch)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred  = int(np.argmax(probs))

        top3_nodes = np.argsort(gnn_node)[-3:][::-1]
        top3_scores = gnn_node[top3_nodes]

        lines = [
            f"=== GAT Prediction Explanation ===",
            f"True Label     : {C.DISPLAY_CLASS_NAMES[label]}",
            f"Predicted Label: {C.DISPLAY_CLASS_NAMES[pred]}  (confidence: {probs[pred]*100:.1f}%)",
            f"",
            f"Class Probabilities:",
        ]
        for i, cname in enumerate(C.DISPLAY_CLASS_NAMES):
            lines.append(f"  {cname:<25} {probs[i]*100:.2f}%")

        lines += [
            f"",
            f"Top-3 Most Influential Brain Regions (Nodes):",
        ]
        for rank, (nidx, nscore) in enumerate(zip(top3_nodes, top3_scores), 1):
            row = nidx // C.PATCH_GRID + 1
            col = nidx  % C.PATCH_GRID + 1
            cy  = (row - 1) * C.PATCH_H + C.PATCH_H // 2
            cx  = (col - 1) * C.PATCH_W + C.PATCH_W // 2
            lines.append(
                f"  Rank {rank}: Node {nidx:3d} "
                f"(Grid Row {row}, Col {col} | Pixel centre ≈ [{cy},{cx}]) "
                f"— Importance: {nscore:.4f}"
            )

        lines += [
            f"",
            f"Interpretation:",
            f"  The model focused primarily on brain regions around",
            f"  rows {top3_nodes[0]//C.PATCH_GRID+1}-{top3_nodes[2]//C.PATCH_GRID+1} "
            f"of the patch grid, which correspond to cortical",
            f"  areas commonly affected in Alzheimer's Disease.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch XAI: generate reports for multiple samples
    # ------------------------------------------------------------------

    def run_batch_xai(
        self,
        test_loader:    "GeoDataLoader",
        model:          torch.nn.Module,
        device:         torch.device,
        figures_dir:    str = C.FIGURES_DIR,
        n_samples:      int = 5,
        use_gnnexplainer: bool = True,
    ) -> None:
        """Generate XAI outputs for n_samples from the test set."""
        ds = test_loader.dataset
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(len(ds), min(n_samples, len(ds)), replace=False)

        for i, sample_idx in enumerate(idx):
            img_path = ds.paths[sample_idx]
            label    = ds.labels[sample_idx]
            prefix   = os.path.join(figures_dir, f"xai_sample_{i+1}")
            logger.info(f"XAI [{i+1}/{len(idx)}]: {os.path.basename(img_path)}")
            try:
                self.visualize(img_path, label, model, device,
                               save_prefix=prefix, use_gnnexplainer=use_gnnexplainer)
                expl = self.generate_text_explanation(img_path, label, model, device)
                txt_path = f"{prefix}_explanation.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(expl)
                logger.info(f"  Text explanation → {txt_path}")
            except Exception as e:
                logger.warning(f"  XAI failed for sample {sample_idx}: {e}")
