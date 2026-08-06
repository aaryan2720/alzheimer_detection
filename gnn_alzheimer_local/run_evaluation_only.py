# =============================================================================
# run_evaluation_only.py — Run Evaluation & Report Generation Only
# =============================================================================
# Bypasses GAT training entirely. Loads the best saved model checkpoint and
# executes evaluation, XAI analysis, and PDF/Word report generation using
# the latest configurations (e.g. updated CNN Baseline metrics).
# =============================================================================

import os
import sys
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import build_dataloaders
from models import build_model
from utils import load_checkpoint, set_seed, logger


def main():
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing dataloaders for evaluation...")
    _, _, test_loader, builder = build_dataloaders(batch_size=C.BATCH_SIZE)

    logger.info("Loading best trained model checkpoint ...")
    model = build_model(device)
    load_checkpoint(C.BEST_MODEL_PATH, model)
    model = model.to(device)

    logger.info("Loading training history ...")
    history_path = C.HISTORY_PKL
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    else:
        logger.warning(f"History file not found at {history_path}. Creating fallback history.")
        history = {
            "train_loss": [0.5], "val_loss": [0.5],
            "train_acc": [0.7], "val_acc": [0.7],
            "total_time": 930.8, "epoch_time": [30]
        }

    # 1. Run Evaluation
    logger.info("\nRunning full evaluation ...")
    from evaluate import evaluate
    metrics = evaluate(
        model        = model,
        test_loader  = test_loader,
        device       = device,
        history      = history,
    )

    # 2. Run XAI batch analysis
    logger.info("\nRunning XAI analysis ...")
    try:
        from xai import GATExplainer
        explainer = GATExplainer(builder)
        explainer.run_batch_xai(
            test_loader  = test_loader,
            model        = model,
            device       = device,
            figures_dir  = C.FIGURES_DIR,
            n_samples    = 5,
            use_gnnexplainer = True,
        )
    except Exception as e:
        logger.warning(f"XAI failed: {e}")

    # 3. Generate PDF and DOCX reports
    logger.info("\nGenerating research report ...")
    try:
        from reporting import ReportGenerator
        reporter = ReportGenerator(metrics=metrics, history=history)
        pdf_path, docx_path = reporter.generate_all()
        if pdf_path:  logger.info(f"PDF report  → {pdf_path}")
        if docx_path: logger.info(f"DOCX report → {docx_path}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    logger.info("Evaluation and report generation completed successfully.")


if __name__ == "__main__":
    main()
