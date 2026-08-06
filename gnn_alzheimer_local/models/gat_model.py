# =============================================================================
# models/gat_model.py — Graph Attention Network for Alzheimer's Detection
# =============================================================================
#
# ARCHITECTURE OVERVIEW
# ----------------------
#
#   Input: Graph with N=196 nodes, each with feat_dim=128 features
#
#   Layer 1: GATConv(128 → 64, heads=8, concat=True)   → 512-dim
#             BatchNorm(512) + ELU + Dropout(0.3)
#
#   Layer 2: GATConv(512 → 64, heads=4, concat=True)   → 256-dim
#             BatchNorm(256) + ELU + Dropout(0.3)
#
#   Layer 3: GATConv(256 → 32, heads=1, concat=False)  → 32-dim
#             BatchNorm(32)  + ELU
#
#   Global Mean Pooling  →  graph-level embedding [32]
#
#   Dense: Linear(32→128) + ReLU + Dropout(0.5)
#   Output: Linear(128→4) + Softmax
#
# WHY GAT OVER GCN?
# -----------------
# • GCN uses fixed, normalized adjacency weights — every neighbour
#   contributes equally regardless of relevance.
# • GAT learns attention coefficients α_ij — the model decides which
#   brain region neighbours are most informative for each node.
# • This produces interpretable attention heatmaps showing which
#   brain regions drive the classification decision.
# • Multiple attention heads capture different types of relationships
#   simultaneously (spectral diversity).
# =============================================================================

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

import config as C


# ---------------------------------------------------------------------------
# AlzheimerGAT
# ---------------------------------------------------------------------------

class AlzheimerGAT(nn.Module):
    """
    Graph Attention Network for 4-class Alzheimer's Disease classification.

    Parameters
    ----------
    in_channels   : int — input node feature dimension (default: NODE_FEAT_DIM=128)
    hidden_dim    : int — hidden channels per attention head (default: 64)
    heads_1       : int — attention heads in layer 1 (default: 8)
    heads_2       : int — attention heads in layer 2 (default: 4)
    heads_3       : int — attention heads in layer 3 (default: 1)
    num_classes   : int — output classes (default: 4)
    dropout       : float — graph dropout rate (default: 0.3)
    dense_dim     : int — intermediate dense layer size (default: 128)
    dense_dropout : float — classifier dropout rate (default: 0.5)
    """

    def __init__(
        self,
        in_channels:   int   = C.NODE_FEAT_DIM,
        hidden_dim:    int   = C.GAT_HIDDEN_DIM,
        heads_1:       int   = C.GAT_HEADS_1,
        heads_2:       int   = C.GAT_HEADS_2,
        heads_3:       int   = C.GAT_HEADS_3,
        num_classes:   int   = C.NUM_CLASSES,
        dropout:       float = C.GAT_DROPOUT,
        dense_dim:     int   = C.DENSE_DIM,
        dense_dropout: float = C.DENSE_DROPOUT,
    ):
        super().__init__()

        self.dropout = dropout

        # ---- Graph Attention Layers ----------------------------------------
        # Layer 1: 128 → 64×8 = 512 (concat=True)
        self.gat1 = GATConv(
            in_channels, hidden_dim,
            heads=heads_1, concat=True,
            dropout=dropout, add_self_loops=True
        )
        self.bn1 = BatchNorm(hidden_dim * heads_1)

        # Layer 2: 512 → 64×4 = 256 (concat=True)
        self.gat2 = GATConv(
            hidden_dim * heads_1, hidden_dim,
            heads=heads_2, concat=True,
            dropout=dropout, add_self_loops=True
        )
        self.bn2 = BatchNorm(hidden_dim * heads_2)

        # Layer 3: 256 → 32×1 = 32 (concat=False → mean over heads)
        self.gat3 = GATConv(
            hidden_dim * heads_2, hidden_dim // 2,
            heads=heads_3, concat=False,
            dropout=dropout, add_self_loops=True
        )
        self.bn3 = BatchNorm(hidden_dim // 2)

        # ---- Classifier Head -----------------------------------------------
        graph_embed_dim = hidden_dim // 2   # = 32
        self.fc1    = nn.Linear(graph_embed_dim, dense_dim)
        self.bn_fc  = nn.BatchNorm1d(dense_dim)
        self.fc2    = nn.Linear(dense_dim, num_classes)

        self.dense_dropout = dense_dropout

        # Storage for attention weights (used by XAI)
        self._attn1: torch.Tensor | None = None
        self._attn2: torch.Tensor | None = None
        self._attn3: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x          : [total_nodes, in_channels]
        edge_index : [2, total_edges]
        batch      : [total_nodes]  — batch assignment vector
        return_attn: bool — if True, also return attention weight tensors

        Returns
        -------
        logits : [batch_size, num_classes]
        (optionally) attn_dict : dict with keys "layer1", "layer2", "layer3"
        """

        # ---- Layer 1 -------------------------------------------------------
        x, (ei1, a1) = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        self._attn1 = (ei1, a1)

        # ---- Layer 2 -------------------------------------------------------
        x, (ei2, a2) = self.gat2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        self._attn2 = (ei2, a2)

        # ---- Layer 3 -------------------------------------------------------
        x, (ei3, a3) = self.gat3(x, edge_index, return_attention_weights=True)
        x = self.bn3(x)
        x = F.elu(x)
        self._attn3 = (ei3, a3)

        # ---- Global Pooling ------------------------------------------------
        x = global_mean_pool(x, batch)   # [batch_size, 32]

        # ---- Classifier Head -----------------------------------------------
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dense_dropout, training=self.training)
        logits = self.fc2(x)              # [batch_size, num_classes]

        if return_attn:
            attn_dict = {
                "layer1": (ei1, a1),
                "layer2": (ei2, a2),
                "layer3": (ei3, a3),
            }
            return logits, attn_dict
        return logits

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities [batch_size, num_classes]."""
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch)
        return F.softmax(logits, dim=-1)

    def get_node_attention_scores(self, layer: int = 1) -> torch.Tensor | None:
        """
        Return per-node attention importance aggregated from stored attention weights.

        Aggregates by summing attention coefficients flowing into each node.

        Parameters
        ----------
        layer : int — 1, 2, or 3

        Returns
        -------
        node_scores : Tensor [num_nodes] or None
        """
        layer_map = {1: self._attn1, 2: self._attn2, 3: self._attn3}
        attn_data = layer_map.get(layer)
        if attn_data is None:
            return None
        edge_index, attn_weights = attn_data
        # attn_weights: [num_edges, heads]
        # Sum over heads then scatter-add to destination nodes
        attn_mean = attn_weights.mean(dim=-1)   # [num_edges]
        num_nodes = edge_index.max().item() + 1
        scores = torch.zeros(num_nodes, device=attn_weights.device)
        scores.scatter_add_(0, edge_index[1], attn_mean)
        return scores


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(device: torch.device | None = None) -> AlzheimerGAT:
    """Instantiate, move to device, and return AlzheimerGAT."""
    model = AlzheimerGAT()
    if device is not None:
        model = model.to(device)
    return model
