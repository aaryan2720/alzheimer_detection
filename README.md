# 🧠 Graph Attention Networks (GAT) for Alzheimer's Disease Detection

This repository contains the source code, experimental configs, and benchmark results for a research project on **Alzheimer's Disease Detection from Brain MRI slices using Graph Attention Networks (GAT)** with **GNNExplainer Explainable AI (XAI)**. 

To validate GAT's performance, we evaluate it against a **Conventional CNN baseline** and **8 traditional Machine Learning classifiers** on the same dataset split.

---

## 📂 Project Directory Structure & Usage

```
GNN/
├── Alzheimer.ipynb               # Phase 1: CNN Baseline notebook (TensorFlow/Keras)
├── GNN_Alzheimer_GAT_Colab.ipynb # Phase 2: GAT + XAI Google Colab notebook (PyTorch)
├── .gitignore                    # Excludes venv, large datasets, and graph caches
├── README.md                     # Master project documentation (this file)
│
├── global_results/               # Global comparative analysis (Auto-created by benchmark)
│   ├── README.md                 # Detailed scientific analysis of the benchmark
│   ├── comparative_analysis.xlsx # comparative metrics Excel sheet
│   ├── comparative_analysis.csv  # comparative metrics CSV
│   └── comparative_metrics.png   # Bar chart comparing Accuracy and F1-Scores
│
└── gnn_alzheimer_local/          # Local VS Code / PyCharm project version
    ├── main.py                   # Run all pipeline phases sequentially (train, eval, XAI, PDF/Word)
    ├── run_benchmark.py          # Script to extract graph features and train 8 ML baselines
    ├── config.py                 # Configuration, hyperparameters, paths, and CNN metrics
    ├── train.py                  # GAT Training loop with EarlyStopping & Checkpointing
    ├── evaluate.py               # Generates GAT test metrics and high-DPI figures
    ├── utils.py                  # Helpers (seeding, logging, device detection)
    │
    ├── models/
    │   └── gat_model.py          # 3-Layer GAT architecture with attention extraction
    ├── dataset/
    │   ├── graph_builder.py      # Converts MRI slices to grid + similarity graphs
    │   └── alzheimer_dataset.py  # PyG dataset handler with Disk & RAM caching
    ├── xai/
    │   └── explainer.py          # GNNExplainer wrapper & attention overlay heatmaps
    ├── reporting/
    │   └── report_generator.py   # Compiles research paper report (PDF & DOCX)
    │
    ├── outputs/                  # Auto-created: training curves, confusion matrices, models
    │   └── graph_cache/          # Cached .pt graphs for accelerated data loading
    └── results/                  # Auto-created GAT metrics and research report drafts
```

---

## 📊 Dataset Specifications
We utilize the **Augmented Alzheimer MRI Dataset** from Kaggle:
👉 **[Augmented Alzheimer MRI Dataset on Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)**

* **Images:** $224 \times 224$ pixels, normalized to $[0, 1]$.
* **Classes:** Four stages of dementia:
  1. `Mild Demented`
  2. `Moderate Demented`
  3. `Non Demented`
  4. `Very Mild Demented`
* **Real Test Data:** The separate `OriginalDataset` directory (containing **6,400 unaugmented original MRI slices**) is used strictly for testing.

---

## 📈 Multi-Model Benchmark Comparison

The table below lists the test set metrics for all evaluated models on the `OriginalDataset`:

| Model Class | Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---|---|---|---|---|
| **Deep Learning (CNN)** | CNN Baseline (Keras) | **0.9373** | **0.9385** | **0.9373** | **0.9379** |
| **Deep Learning (Graph)** | **GAT (Ours - 50% Subset)** | **0.6794** | **0.6043** | **0.8215** | **0.6295** |
| **Machine Learning** | Random Forest | 0.6153 | 0.7001 | 0.7681 | 0.6918 |
| **Machine Learning** | XGBoost | 0.5903 | 0.6826 | 0.7469 | 0.6735 |
| **Machine Learning** | LightGBM | 0.5791 | 0.6847 | 0.7250 | 0.6635 |
| **Machine Learning** | Bagging (Decision Tree) | 0.5741 | 0.5790 | 0.7426 | 0.5944 |
| **Machine Learning** | Decision Tree | 0.4139 | 0.3647 | 0.5493 | 0.3635 |
| **Machine Learning** | K-Nearest Neighbors (KNN) | 0.4105 | 0.3650 | 0.5802 | 0.3440 |
| **Machine Learning** | Support Vector Machine (SVM) | 0.4084 | 0.3561 | 0.3864 | 0.2885 |
| **Machine Learning** | AdaBoost | 0.3850 | 0.4063 | 0.4083 | 0.3246 |

---

## 🔬 GAT Performance Analysis (67.94% accuracy vs 93.73%)

To explain the GAT model's accuracy to reviewers and academic panels, consider these technical factors:

1. **Dimensionality Reduction & Feature Compression:**
   * Standard CNNs process raw images ($224 \times 224 \times 3$), retaining high-frequency pixel details, fine lines, and structural boundaries.
   * GAT converts images into graphs of **196 nodes** (patches). The patches are compressed using PCA to **128 dimensions** to enable graph propagation. While PCA captures the primary structural variances, it drops minor textural variations that standard CNN filters exploit.
2. **Dataset Size Constraints:**
   * The CNN Baseline was trained on **100% of the training dataset** (27,188 augmented images).
   * The GAT Model was trained on a **50% stratified subset** (13,594 training images) to satisfy local execution runtime limits. Increasing GAT's sub-sample ratio to 100% and training for more epochs will allow it to capture the remaining feature variances.
3. **Relational Inductive Bias vs. Translation Invariance:**
   * CNNs have a strong local translation invariance assumption built into their shared weight filters. This makes them highly sample-efficient on regular grid structures (like 2D images).
   * GAT treats the image as a set of relational nodes (patches), which is a softer prior that typically requires more extensive hyperparameter tuning and larger graph sizes (e.g., 28×28 grid instead of 14×14) to match the pixel-perfect representation of standard CNNs.

---

## 🔎 Explainable AI (XAI)
We integrate **GNNExplainer** (`torch_geometric.explain.GNNExplainer`) to learn node and edge importance masks per prediction. GATConv attention coefficients are extracted during the forward pass. These are combined to produce high-DPI attention heatmaps overlaid on the original MRIs (saved to `outputs/figures/`), showing the clinical areas driving the prediction alongside automated text explanations.

---

## 🚀 Execution Instructions

### 1. Traditional ML Baselines
To train the 8 traditional ML models on GAT graph features and output comparisons to the `global_results` folder, run:
```powershell
python run_benchmark.py
```

### 2. GAT Train, Evaluation & Report Generation
* **To run training, evaluation, XAI, and reports together:**
  ```powershell
  python main.py
  ```
* **To run ONLY evaluation, XAI, and report generation (bypassing training):**
  ```powershell
  python run_evaluation_only.py
  ```
