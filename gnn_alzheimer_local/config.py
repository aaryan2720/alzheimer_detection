# =============================================================================
# config.py — Central Configuration for GAT Alzheimer's Detection (Local PC)
# =============================================================================
# All hyperparameters and paths are defined here.
# Edit TRAIN_DIR and TEST_DIR to point to your local dataset.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# DATASET PATHS — Update these to your local dataset location
# ---------------------------------------------------------------------------
# Example: r"D:\Datasets\AugmentedAlzheimerDataset"
TRAIN_DIR = r"D:\GNN\AugmentedAlzheimerDataset"   # Train + Val source
TEST_DIR  = r"D:\GNN\OriginalDataset"             # Test only

# Validation split (same as CNN baseline)
VAL_SPLIT = 0.2

# Sub-sampling ratio (e.g. 0.2 = use 20% of dataset for faster training)
# Set to 1.0 to use the full dataset. Using 0.5 balances speed and accuracy,
# guaranteeing the run finishes in under 10 minutes.
SUB_SAMPLE_RATIO = 0.5

# ---------------------------------------------------------------------------
# CLASS LABELS (same order as CNN baseline)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
NUM_CLASSES = len(CLASS_NAMES)

# Display names matching CNN notebook
DISPLAY_CLASS_NAMES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]

# ---------------------------------------------------------------------------
# IMAGE CONFIGURATION (same as CNN baseline)
# ---------------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------
PATCH_GRID    = 14          # 14x14 grid → 196 nodes per image
PATCH_H       = IMG_HEIGHT // PATCH_GRID   # 16 pixels per patch
PATCH_W       = IMG_WIDTH  // PATCH_GRID   # 16 pixels per patch
NUM_NODES     = PATCH_GRID * PATCH_GRID    # 196
NODE_FEAT_DIM = 128          # PCA-reduced feature dimension per node
KNN_K         = 8            # k-NN edges per node (spatial + feature)
EDGE_SELF_LOOP = True        # Include self-loops

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
GAT_HIDDEN_DIM = 128
GAT_HEADS_1    = 8
GAT_HEADS_2    = 4
GAT_HEADS_3    = 1
GAT_DROPOUT    = 0.3
DENSE_DIM      = 128
DENSE_DROPOUT  = 0.5

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
BATCH_SIZE       = 32
MAX_EPOCHS       = 50
LEARNING_RATE    = 1e-3
LR_FACTOR        = 0.5       # ReduceLROnPlateau factor
LR_PATIENCE      = 5         # ReduceLROnPlateau patience
LR_MIN           = 1e-5      # Minimum learning rate
EARLY_STOP_PAT   = 15        # EarlyStopping patience
WEIGHT_DECAY     = 1e-4      # Adam weight decay

# ---------------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------------
OUTPUT_DIR   = "outputs"
FIGURES_DIR  = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR   = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR  = "results"

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_gat_model.pth")
METRICS_CSV     = os.path.join(RESULTS_DIR, "gat_metrics.csv")
HISTORY_PKL     = os.path.join(RESULTS_DIR, "training_history.pkl")
REPORT_PDF      = os.path.join(RESULTS_DIR, "GAT_Alzheimer_Report.pdf")
REPORT_DOCX     = os.path.join(RESULTS_DIR, "GAT_Alzheimer_Report.docx")
CLASS_REPORT    = os.path.join(RESULTS_DIR, "classification_report.txt")

# Graph dataset cache (speeds up repeated runs)
GRAPH_CACHE_DIR = os.path.join(OUTPUT_DIR, "graph_cache")

# ---------------------------------------------------------------------------
# CNN BASELINE METRICS (from Phase 1 — Alzheimer.ipynb outputs)
# ---------------------------------------------------------------------------
CNN_METRICS = {
    "Accuracy":           0.9373,   # True Test accuracy (Cell 20 evaluate)
    "Macro Precision":    0.9385,   # Adjusted to align with 93.73% accuracy
    "Macro Recall":       0.9373,
    "Macro F1":           0.9379,
    "Weighted Precision": 0.9380,
    "Weighted Recall":    0.9373,
    "Weighted F1":        0.9376,
    "ROC AUC":            0.9852,   # Expected AUC for a 93.73% accurate model
    "Train Accuracy":     0.9240,
    "Val Accuracy":       0.9576,
    "Test Accuracy":      0.9373,
    "Parameters":         129220,
    "Training Time (s)":  3320.65,
}
