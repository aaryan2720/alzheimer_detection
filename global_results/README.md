# 📊 Global Comparative Benchmark Results

This folder contains the comparative benchmark results of **ten different models** trained and evaluated on the same dataset split for Alzheimer's Disease detection.

## 📈 Comparative Metrics

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

## 🔬 Scientific Analysis: GAT (67.94%) vs. CNN Baseline (93.73%)

When presenting these results to reviewers or advisors, the difference in accuracy between the standard CNN baseline and the GAT model should be discussed through these key technical factors:

### 1. Spatial Resolution & Information Loss (PCA Node Features)
* **Standard CNN:** Processes raw $224 \times 224 \times 3$ pixel grids directly, retaining high-frequency textures, edge details, and localized contrast boundaries.
* **GAT Model:** Divides the image into a $14 \times 14$ patch grid (**196 nodes**). The patches are flattened and projected using PCA to a **128-dimensional space** to keep GNN training computationally feasible. While PCA captures the primary structural variances, it inevitably discards high-frequency details (fine textures, tissue boundaries) that standard CNN filters exploit.

### 2. Dataset Volume & Sub-Sampling
* **Standard CNN:** Trained on **100% of the training dataset** (27,188 augmented images).
* **GAT Model:** Trained on a **50% stratified subset** (13,594 training images) to ensure training completed within local hardware time constraints (~8 minutes). Standard GATs typically require larger volumes of training graphs to generalize fully compared to translation-invariant CNNs.

### 3. Translation Invariance Inductive Bias
* CNNs have a strong local translation invariance assumption built into their shared weight filters. This makes them highly sample-efficient on regular grid structures (like 2D images). 
* GNNs, by contrast, treat the image as a set of relational nodes (patches). This is a softer prior that allows the model to learn arbitrary structural connections, but it generally requires longer training runs and finer patch resolutions (e.g., $28 \times 28$ grid = 784 nodes) to match the spatial accuracy of CNNs.

---

## 📂 Folder Artifacts

* 📊 **[comparative_analysis.xlsx](comparative_analysis.xlsx):** Complete side-by-side metrics spreadsheet for academic reporting.
* 📋 **[comparative_analysis.csv](comparative_analysis.csv):** Fallback plaintext CSV version of the metrics.
* 🖼️ **[comparative_metrics.png](comparative_metrics.png):** High-DPI bar chart comparing model Accuracies and F1-Scores.
