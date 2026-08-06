# GAT-Based Alzheimer's Disease Detection — Phase 2

> **A Graph Attention Network framework for 4-class Alzheimer's Disease classification from brain MRI, with GNNExplainer-based interpretability and full CNN vs. GAT comparison.**

---

## Project Overview

This is **Phase 2** of an IEEE/Springer research project. Phase 1 established a CNN baseline. Phase 2 replaces the CNN architecture with a **Graph Attention Network (GAT)** while keeping all other aspects of the pipeline identical, enabling a fair head-to-head comparison.

### Key Contributions
- MRI images are converted into **196-node graphs** (14×14 patch grid)
- **3-layer GAT** with 8-4-1 multi-head attention learns brain region relationships
- **GNNExplainer + Attention Heatmaps** provide node-level and edge-level interpretability
- Automatic generation of **report.pdf** and **report.docx** in IEEE format
- Complete CNN vs. GAT comparison across 20+ metrics

---

## Project Structure

```
gnn_alzheimer_local/
├── main.py                     # ← RUN THIS: python main.py
├── config.py                   # All hyperparameters and paths
├── train.py                    # Training loop (Adam, ReduceLROnPlateau, EarlyStopping)
├── evaluate.py                 # All 20+ metrics + publication-quality figures
├── utils.py                    # Helpers (seed, device, logging, checkpointing)
│
├── models/
│   └── gat_model.py            # 3-layer AlzheimerGAT architecture
│
├── dataset/
│   ├── graph_builder.py        # MRI → Graph (patch extraction, PCA, k-NN edges)
│   └── alzheimer_dataset.py    # PyG Dataset + DataLoaders (mirrors CNN splits exactly)
│
├── xai/
│   └── explainer.py            # GNNExplainer + attention visualization
│
├── reporting/
│   └── report_generator.py    # Auto-generate report.pdf + report.docx
│
├── outputs/
│   ├── figures/                # All publication-quality figures (PNG, 300 DPI)
│   ├── models/                 # Best model checkpoint (.pth)
│   └── graph_cache/            # Cached PCA transformer (speeds up re-runs)
│
├── results/
│   ├── gat_metrics.csv         # All evaluation metrics
│   ├── classification_report.txt
│   ├── training_history.pkl
│   ├── GAT_Alzheimer_Report.pdf
│   └── GAT_Alzheimer_Report.docx
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Quick Start

### 1. Set Dataset Paths

Open `config.py` and update:

```python
TRAIN_DIR = r"D:\path\to\AugmentedAlzheimerDataset"   # Same as CNN Phase 1
TEST_DIR  = r"D:\path\to\OriginalDataset"
```

The dataset structure should be:

```
AugmentedAlzheimerDataset/
    MildDemented/
    ModerateDemented/
    NonDemented/
    VeryMildDemented/

OriginalDataset/
    MildDemented/
    ModerateDemented/
    NonDemented/
    VeryMildDemented/
```

### 2. Install Dependencies

#### Option A — pip (recommended for quick start)

```bash
# Install PyTorch first (visit https://pytorch.org for CUDA-specific commands)
pip install torch torchvision

# Install PyG
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option B — conda

```bash
conda env create -f environment.yml
conda activate gnn_alzheimer
```

> **Windows + CUDA users**: For GPU acceleration, install the CUDA-enabled torch and PyG wheels.  
> Visit https://pytorch.org/get-started/locally/ and https://pyg.org/whl/ for the correct wheels.

### 3. Run the Pipeline

```bash
python main.py
```

This runs all 9 steps automatically:

| Step | Description |
|---|---|
| 1 | Dataset discovery + stratified 80/20 train/val split (identical to CNN) |
| 2 | PCA fitting on 1000 training images (cached for future runs) |
| 3 | Graph construction: 14×14 patches → 196 nodes per image |
| 4 | Model instantiation + summary |
| 5 | Training with Adam + ReduceLROnPlateau + EarlyStopping |
| 6 | Load best model checkpoint |
| 7 | Full evaluation: 20+ metrics + all figures |
| 8 | XAI: GNNExplainer + attention heatmaps |
| 9 | Report generation: PDF + DOCX |

---

## Graph Construction

Each MRI image is converted into a graph as follows:

```
224×224 MRI
    ↓
Divide into 14×14 = 196 non-overlapping patches
    ↓
Node Features: PCA(raw pixels) → 128-dim per node
    ↓
Edges: 8-connected spatial adjacency + 8-NN feature similarity
    ↓
PyG Data(x=[196,128], edge_index=[2,E], edge_attr=[E], y=[1])
    ↓
GAT Model → Prediction
```

**Why patches become nodes?**  
Brain regions affected by Alzheimer's Disease (hippocampus, temporal cortex, frontal lobe) are spatially distributed. A graph where nodes represent brain patches and edges capture structural/functional similarity allows the GNN to model the *relational structure* of neurodegeneration — something a CNN cannot do explicitly.

---

## GAT Architecture

```
Input: [196 nodes × 128 features]
       ↓
GATConv(128→64, heads=8, concat)  → [196 × 512]
BatchNorm + ELU + Dropout(0.3)
       ↓
GATConv(512→64, heads=4, concat)  → [196 × 256]
BatchNorm + ELU + Dropout(0.3)
       ↓
GATConv(256→32, heads=1, mean)    → [196 × 32]
BatchNorm + ELU
       ↓
Global Mean Pooling → [32]
       ↓
Linear(32→128) + BN + ReLU + Dropout(0.5)
       ↓
Linear(128→4) → Softmax
```

---

## CNN Baseline Metrics (Phase 1)

| Metric | CNN |
|---|---|
| Test Accuracy | 0.3953 |
| Macro F1 | 0.2458 |
| Weighted F1 | 0.3923 |
| ROC AUC | 0.4902 |
| Parameters | 129,220 |
| Training Time | 3,320 s |

> **Note**: The CNN shows clear overfitting (train acc 92.4% vs test acc 39.5%). The GAT's relational representation is expected to generalize significantly better.

---

## Output Files

After running `python main.py`, all outputs are saved automatically:

| File | Location |
|---|---|
| Best model | `outputs/models/best_gat_model.pth` |
| Metrics CSV | `results/gat_metrics.csv` |
| Classification report | `results/classification_report.txt` |
| Training curves | `outputs/figures/training_accuracy.png` |
| Confusion matrices | `outputs/figures/confusion_matrix.png` |
| ROC curves | `outputs/figures/roc_curve.png` |
| PR curves | `outputs/figures/precision_recall_curve.png` |
| CNN vs GAT comparison | `outputs/figures/cnn_vs_gat_comparison.png` |
| XAI heatmaps | `outputs/figures/xai_sample_*.png` |
| Node importance | `outputs/figures/xai_sample_*_node_importance.png` |
| PDF report | `results/GAT_Alzheimer_Report.pdf` |
| DOCX report | `results/GAT_Alzheimer_Report.docx` |

---

## Google Colab Version

See `GNN_Alzheimer_GAT_Colab.ipynb` in the parent directory for the complete notebook version.

---

## Research Paper Information

- **Dataset**: Augmented Alzheimer MRI Dataset (Kaggle: `uraninjo/augmented-alzheimer-mri-dataset`)
- **Framework**: PyTorch + PyTorch Geometric
- **Model**: Graph Attention Network (Veličković et al., 2018)
- **XAI**: GNNExplainer (Ying et al., 2019)
- **Comparison**: CNN Baseline (Phase 1) vs. GAT (Phase 2)

---

## Citation

If you use this code in your research, please cite:

```
@article{alzheimer_gat_2026,
  title     = {Graph Attention Networks for Alzheimer's Disease Detection from Brain MRI},
  author    = {[Your Name]},
  journal   = {IEEE/Springer [Journal Name]},
  year      = {2026}
}
```
