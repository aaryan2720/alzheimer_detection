# =============================================================================
# run_benchmark.py — Run Comparative Benchmark Analysis (ML Baselines)
# =============================================================================
# Trains & evaluates 8 traditional ML models on the exact same dataset splits,
# extracts GAT and CNN metrics, and compiles a comprehensive comparison table
# (Excel/CSV) and bar charts in the global_results/ folder.
# =============================================================================

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from dataset import build_dataloaders
from utils import set_seed, logger

# Import PyTorch & PyG
import torch
from torch_geometric.nn import global_mean_pool

# Import traditional ML classifiers
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def extract_features(loader, device):
    """Extract average-pooled node features for traditional ML models."""
    all_feats = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Global average pool GAT PCA node features to get a single vector per graph
            pooled = global_mean_pool(batch.x, batch.batch)
            all_feats.append(pooled.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            
    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)
    return X, y


def main():
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create global results folder
    global_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "global_results")
    os.makedirs(global_results_dir, exist_ok=True)
    
    logger.info("="*70)
    logger.info("  STARTING COMPARATIVE BENCHMARK ANALYSIS")
    logger.info(f"  Target Folder: {global_results_dir}")
    logger.info("="*70 + "\n")
    
    # ---- 1. Build DataLoaders & Extract Features ----
    logger.info("Loading PyTorch Geometric Datasets ...")
    train_loader, val_loader, test_loader, _ = build_dataloaders(batch_size=C.BATCH_SIZE)
    
    logger.info("Extracting flat feature representations (Average Pooled 128D PCA) ...")
    X_train, y_train = extract_features(train_loader, device)
    X_val, y_val     = extract_features(val_loader, device)
    X_test, y_test   = extract_features(test_loader, device)
    
    # Combine Train & Val sets for training traditional ML classifiers to give them more data
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    logger.info(f"  Features Extracted Successfully:")
    logger.info(f"    Train size: {X_train_full.shape}")
    logger.info(f"    Test size : {X_test.shape}\n")
    
    # ---- 2. Initialize Models ----
    classifiers = {
        "SVM": SVC(kernel="rbf", C=1.0, random_state=C.SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=C.SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=C.SEED),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=C.SEED, verbosity=0),
        "Bagging (DT)": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=C.SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=C.SEED),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=C.SEED, verbose=-1),
    }
    
    results = []
    
    # ---- 3. Train & Evaluate Traditional ML Classifiers ----
    for name, clf in classifiers.items():
        logger.info(f"Training {name} Classifier ...")
        t0 = time.time()
        clf.fit(X_train_full, y_train_full)
        train_time = time.time() - t0
        
        # Inference
        y_pred = clf.predict(X_test)
        
        # Calculate Metrics
        acc    = accuracy_score(y_test, y_pred)
        prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        logger.info(f"  {name} test results: Acc={acc:.4f} | Macro F1={f1:.4f} (trained in {train_time:.1f}s)")
        results.append({
            "Model": name,
            "Test Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1-Score": f1,
        })
        
    # ---- 4. Append CNN Baseline & GAT Model Results ----
    logger.info("\nLoading CNN Baseline & GAT metrics ...")
    
    # CNN Baseline
    results.append({
        "Model": "CNN Baseline",
        "Test Accuracy": float(C.CNN_METRICS["Accuracy"]),
        "Macro Precision": float(C.CNN_METRICS["Macro Precision"]),
        "Macro Recall": float(C.CNN_METRICS["Macro Recall"]),
        "Macro F1-Score": float(C.CNN_METRICS["Macro F1"]),
    })
    
    # GAT Model
    gat_csv_path = C.METRICS_CSV
    if os.path.exists(gat_csv_path):
        df_gat = pd.read_csv(gat_csv_path)
        results.append({
            "Model": "GAT (Ours)",
            "Test Accuracy": float(df_gat["Accuracy"].iloc[0]),
            "Macro Precision": float(df_gat["Macro Precision"].iloc[0]),
            "Macro Recall": float(df_gat["Macro Recall"].iloc[0]),
            "Macro F1-Score": float(df_gat["Macro F1"].iloc[0]),
        })
    else:
        logger.warning(f"GAT metrics CSV not found at {gat_csv_path}. Skipping GAT in comparison.")

    # ---- 5. Compile into DataFrames & Save ----
    df_comparison = pd.DataFrame(results)
    
    # Re-order to put GAT and CNN at the top for emphasis
    df_comparison["Sort_Key"] = df_comparison["Model"].apply(
        lambda x: 0 if "GAT" in x else (1 if "CNN" in x else 2)
    )
    df_comparison = df_comparison.sort_values(by=["Sort_Key", "Test Accuracy"], ascending=[True, False]).drop(columns="Sort_Key")
    
    # Save CSV
    csv_out = os.path.join(global_results_dir, "comparative_analysis.csv")
    df_comparison.to_csv(csv_out, index=False)
    logger.info(f"Saved CSV comparison table → {csv_out}")
    
    # Save Excel
    xlsx_out = os.path.join(global_results_dir, "comparative_analysis.xlsx")
    try:
        df_comparison.to_excel(xlsx_out, index=False)
        logger.info(f"Saved Excel comparison table → {xlsx_out}")
    except Exception as e:
        logger.warning(f"Failed to write Excel file: {e}")

    # ---- 6. Generate Comparative Bar Chart ----
    logger.info("Generating comparative visualization chart ...")
    
    # Melt dataframe for seaborn plotting
    df_plot = df_comparison.melt(id_vars="Model", value_vars=["Test Accuracy", "Macro F1-Score"], 
                                 var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Palette definition
    custom_colors = {"Test Accuracy": "#2196F3", "Macro F1-Score": "#4CAF50"}
    
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="Score",
        hue="Metric",
        palette=custom_colors,
        edgecolor="black",
        linewidth=0.5
    )
    
    # Highlight values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height + 0.01),
                        ha="center", va="bottom", fontsize=8, color="#212121", fontweight="semibold")
            
    plt.title("Comparative Performance Analysis: CNN vs. GAT vs. Machine Learning Baselines", 
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classifier Model", fontsize=11, labelpad=10)
    plt.ylabel("Score", fontsize=11, labelpad=10)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    
    plot_out = os.path.join(global_results_dir, "comparative_metrics.png")
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparative metrics visualization → {plot_out}")
    
    logger.info("\n" + "="*70)
    logger.info("  BENCHMARK COMPLETE")
    logger.info("="*70)
    print(df_comparison.to_string(index=False))
    logger.info("="*70)


if __name__ == "__main__":
    main()
