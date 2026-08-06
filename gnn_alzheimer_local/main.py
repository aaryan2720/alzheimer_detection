# =============================================================================
# main.py — Entry Point for GAT Alzheimer's Detection (Local PC)
# =============================================================================
# Run: python main.py
# =============================================================================

import os
import sys
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from utils import set_seed, get_device, create_output_dirs, print_model_summary, logger


def main():
    """
    Full pipeline:
      1. Setup (seed, device, directories)
      2. Dataset discovery + graph construction + PCA fitting
      3. DataLoader creation
      4. Model instantiation + summary
      5. Training
      6. Load best model
      7. Evaluation (metrics + figures)
      8. Graph visualization
      9. XAI (GNNExplainer + attention heatmaps)
     10. CNN vs GAT comparison
     11. Report generation (PDF + DOCX)
    """

    # ==========================================================================
    # STEP 0 — Setup
    # ==========================================================================
    set_seed(C.SEED)
    device = get_device()
    create_output_dirs(C.OUTPUT_DIR, C.FIGURES_DIR, C.MODELS_DIR, C.RESULTS_DIR, C.GRAPH_CACHE_DIR)

    logger.info("\n" + "="*70)
    logger.info("  GAT-Based Alzheimer's Disease Detection — Phase 2")
    logger.info("  Configuration:")
    logger.info(f"    TRAIN_DIR    : {C.TRAIN_DIR}")
    logger.info(f"    TEST_DIR     : {C.TEST_DIR}")
    logger.info(f"    Patch Grid   : {C.PATCH_GRID}×{C.PATCH_GRID} = {C.NUM_NODES} nodes")
    logger.info(f"    Node Feat Dim: {C.NODE_FEAT_DIM}")
    logger.info(f"    Device       : {device}")
    logger.info("="*70 + "\n")

    # ==========================================================================
    # STEP 1 — Dataset + DataLoaders
    # ==========================================================================
    logger.info("Building graph dataset …  (this may take several minutes on first run)")

    from dataset import build_dataloaders
    train_loader, val_loader, test_loader, builder = build_dataloaders(
        batch_size=C.BATCH_SIZE,
        num_workers=0,   # 0 = required on Windows to avoid multiprocessing issues
    )

    # Print dataset stats
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ==========================================================================
    # STEP 2 — Dataset Distribution Figure
    # ==========================================================================
    logger.info("Plotting dataset distribution …")
    try:
        from evaluate import plot_dataset_distribution
        plot_dataset_distribution(C.FIGURES_DIR)
    except Exception as e:
        logger.warning(f"Dataset distribution plot failed: {e}")

    # ==========================================================================
    # STEP 3 — Graph Visualization (one sample)
    # ==========================================================================
    logger.info("Generating graph visualization …")
    try:
        sample_path  = train_ds.paths[0]
        sample_label = train_ds.labels[0]
        builder.visualize_graph(
            sample_path, sample_label,
            save_path=os.path.join(C.FIGURES_DIR, "graph_visualization.png"),
        )
    except Exception as e:
        logger.warning(f"Graph visualization failed: {e}")

    # ==========================================================================
    # STEP 4 — Model
    # ==========================================================================
    from models import build_model
    model = build_model(device)
    print_model_summary(model, name="AlzheimerGAT")

    # ==========================================================================
    # STEP 5 — Training
    # ==========================================================================
    logger.info("\nStarting training …")
    from train import train
    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
    )

    # ==========================================================================
    # STEP 6 — Load Best Model
    # ==========================================================================
    logger.info("Loading best model checkpoint …")
    from utils import load_checkpoint
    load_checkpoint(C.BEST_MODEL_PATH, model)
    model = model.to(device)

    # ==========================================================================
    # STEP 7 — Evaluation
    # ==========================================================================
    logger.info("\nRunning full evaluation …")
    from evaluate import evaluate
    metrics = evaluate(
        model        = model,
        test_loader  = test_loader,
        device       = device,
        history      = history,
    )

    # ==========================================================================
    # STEP 8 — XAI
    # ==========================================================================
    logger.info("\nRunning XAI analysis …")
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
        logger.warning(f"XAI failed: {e}. Skipping XAI step.")

    # ==========================================================================
    # STEP 9 — Report Generation
    # ==========================================================================
    logger.info("\nGenerating research report …")
    try:
        from reporting import ReportGenerator
        reporter = ReportGenerator(metrics=metrics, history=history)
        pdf_path, docx_path = reporter.generate_all()
        if pdf_path:  logger.info(f"PDF report  → {pdf_path}")
        if docx_path: logger.info(f"DOCX report → {docx_path}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("  GAT PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"  Test Accuracy     : {metrics.get('Accuracy', '—')}")
    logger.info(f"  Macro F1          : {metrics.get('Macro F1', '—')}")
    logger.info(f"  ROC AUC           : {metrics.get('ROC AUC (Macro OvR)', '—')}")
    logger.info(f"  MCC               : {metrics.get('MCC', '—')}")
    logger.info(f"  Cohen Kappa       : {metrics.get('Cohen Kappa', '—')}")
    logger.info(f"  Parameters        : {metrics.get('Parameters', '—'):,}" if isinstance(metrics.get('Parameters'), int) else f"  Parameters        : {metrics.get('Parameters', '—')}")
    logger.info(f"  Model Size        : {metrics.get('Model Size (MB)', '—')} MB")
    logger.info(f"  Training Time     : {metrics.get('Training Time (s)', '—')} s")
    logger.info(f"\n  CNN Baseline Accuracy : {C.CNN_METRICS['Accuracy']}")
    logger.info(f"  CNN Baseline Macro F1 : {C.CNN_METRICS['Macro F1']}")
    logger.info("="*70)
    logger.info(f"\nAll outputs saved to: {os.path.abspath(C.OUTPUT_DIR)}")
    logger.info(f"Results saved to    : {os.path.abspath(C.RESULTS_DIR)}")


if __name__ == "__main__":
    main()
