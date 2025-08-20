import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from data_loader import DataProcessor
from model import AdvancedSpatioTemporalGNN
from trainer import ModelTrainer
from interpreter import InterpretabilityAnalyzer
from visualization import GeoVisualizer, plot_and_save_losses, plot_fidelity_curves


def save_to_csv(data, filename, out_dir):
    """Utility function to save dataframes to CSV."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    data.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Data saved: {path}")


def print_metrics(metrics_dict, stage_name=""):
    """Prints performance metrics in a formatted way."""
    print(f"\n--- {stage_name} Performance Metrics ---")
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
    print(metrics_str)


def main():
    """Main execution function to coordinate the entire workflow."""
    start_time = time.time()
    cfg = Config()

    # Create output directories
    for sub_dir in ['checkpoints', 'figures', 'logs', 'results']:
        (Path(cfg.OUTPUT_DIR) / sub_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting Workflow. Output Directory: {cfg.OUTPUT_DIR}, Device: {cfg.DEVICE}")

    try:
        # 1. Data Processing
        dp = DataProcessor(cfg)
        adj_matrix, soc_vals = dp.load_data()
        node_ids, num_nodes = dp.node_ids, dp.num_nodes
        tr_dl, vl_dl, te_dl, data_shape = dp.prepare_datasets(soc_vals)

        # 2. Model Initialization
        gnn = AdvancedSpatioTemporalGNN(
            num_nodes=num_nodes, seq_len=cfg.SEQ_LEN, pred_len=cfg.PRED_LEN,
            rnn_units=cfg.RNN_UNITS, num_rnn_layers=cfg.NUM_RNN_LAYERS,
            max_diffusion_step=cfg.MAX_DIFFUSION_STEP, input_dim=data_shape[0][-1]
        ).to(cfg.DEVICE)

        # 3. GNN Training and Evaluation
        trainer = ModelTrainer(gnn, cfg, dp.scaler)
        tr_loss, vl_loss, _ = trainer.train_gnn(tr_dl, vl_dl, adj_matrix)
        plot_and_save_losses(tr_loss, vl_loss, "GNN Training & Validation Loss", "gnn_losses.png", cfg.OUTPUT_DIR)

        gnn_metrics, _, _ = trainer.evaluate_model(te_dl, adj_matrix, load_best=True)
        print_metrics(gnn_metrics, "Original GNN Model")
        save_to_csv(pd.DataFrame([gnn_metrics]), "metrics_gnn_original.csv", Path(cfg.OUTPUT_DIR) / 'results')

        # 4. Interpretability Analysis
        interpreter = InterpretabilityAnalyzer(cfg, gnn)

        # 4a. Perturbation-based method
        sm1_p, sm2_p, tm_p, mask_loss = trainer.train_masks(tr_dl, adj_matrix, num_nodes)
        plot_and_save_losses(mask_loss, None, "Mask Training Loss", "mask_losses.png", cfg.OUTPUT_DIR)
        act_sm1, _, _ = interpreter.analyze_masks(sm1_p, sm2_p, tm_p)

        # 4b. Gradient-based methods
        saliency_map = interpreter.calculate_gradient_based_importance(te_dl, adj_matrix, 'saliency')
        ig_map = interpreter.calculate_gradient_based_importance(te_dl, adj_matrix, 'ig')

        # 5. Fidelity Analysis to Compare Methods
        fid_df = interpreter.calculate_fidelity_sparsity(
            trainer.evaluate_model, te_dl, adj_matrix, act_sm1,
            gnn_metrics.get('RMSE', float('inf')), saliency_map, ig_map
        )
        if not fid_df.empty:
            save_to_csv(fid_df, "fidelity_sparsity_analysis.csv", Path(cfg.OUTPUT_DIR) / 'results')
            plot_fidelity_curves(fid_df, cfg.OUTPUT_DIR)
            print("Fidelity analysis complete and curve saved.")

        # 6. Visualization
        top_n_ids = [node_ids[i] for i in np.argsort(soc_vals.sum(axis=0))[-cfg.TOP_N_NODES_ANALYSIS:]]
        viz = GeoVisualizer(cfg, node_ids)
        viz.generate_focused_area_maps_for_top_n(act_sm1, top_n_ids)

    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\nWorkflow finished. Total elapsed time: {total_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()