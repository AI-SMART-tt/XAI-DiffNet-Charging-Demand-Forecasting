
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config import Config


class InterpretabilityAnalyzer:
    """
    Handles all interpretability analyses:
    1. Perturbation-based mask statistics.
    2. Gradient-based importance (Saliency, Integrated Gradients).
    3. Fidelity analysis to compare the faithfulness of different methods.
    """

    def __init__(self, config: Config, model: nn.Module):
        self.config = config
        self.model = model

    def _calc_mask_stats(self, ms1_np, ms2_np, mt_np):
        """Helper function to calculate detailed statistics for the activated masks."""
        print("\nMask Statistics:")
        stats_list, thr = [], self.config.IMPORTANCE_THRESHOLD
        for name, data_np in [("Spatial Mask 1", ms1_np), ("Spatial Mask 2", ms2_np), ("Temporal Mask", mt_np)]:
            n_strong = (data_np > (1 + thr)).sum()
            n_weak = (data_np < (1 - thr)).sum()
            activity_ratio = (n_strong + n_weak) / data_np.size if data_np.size > 0 else 0.0
            stats = {
                "Mask": name, "Min": data_np.min(), "Max": data_np.max(),
                "Mean": data_np.mean(), "Std": data_np.std(),
                "N_Strong": n_strong, "N_Weak": n_weak,
                "Activity": activity_ratio, "Total": data_np.size
            }
            stats_list.append(stats)
            print(f"  {name}: Min {stats['Min']:.3f}, Max {stats['Max']:.3f}, Mean {stats['Mean']:.3f}, "
                  f"Std {stats['Std']:.3f}, Activity {stats['Activity']:.3f}")
        return pd.DataFrame(stats_list)

    def analyze_masks(self, sm1_p, sm2_p, tm_p):
        """Analyzes the learned perturbation masks and computes their statistics."""
        print("Analyzing learned perturbation masks...")
        with torch.no_grad():
            act_sm1 = (torch.tanh(sm1_p) + torch.tanh(sm1_p.T)) / 2 + 1
            act_sm2 = (torch.tanh(sm2_p) + torch.tanh(sm2_p.T)) / 2 + 1
            act_tm = torch.tanh(tm_p) + 1

        act_sm1_np = act_sm1.cpu().numpy()
        act_sm2_np = act_sm2.cpu().numpy()
        act_tm_np = act_tm.cpu().numpy()

        stats_df = self._calc_mask_stats(act_sm1_np, act_sm2_np, act_tm_np)
        return act_sm1_np, act_sm2_np, act_tm_np, stats_df

    def calculate_gradient_based_importance(self, loader, adj_matrix_np, method='saliency'):
        """Calculates gradient-based importance maps for the graph structure."""
        print(f"Calculating '{method}' importance map...")
        self.model.eval()
        criterion = nn.MSELoss()
        accumulated_gradients = np.zeros_like(adj_matrix_np, dtype=np.float32)

        for x_b, y_b in loader:
            x_b, y_b = x_b.to(self.config.DEVICE), y_b.to(self.config.DEVICE)
            adj_tensor = torch.from_numpy(adj_matrix_np).float().to(self.config.DEVICE)

            if method == 'ig':
                baseline = torch.zeros_like(adj_tensor)
                batch_total_gradients = torch.zeros_like(adj_tensor)
                for alpha in np.linspace(0.0, 1.0, self.config.IG_STEPS, endpoint=False):
                    adj_interpolated = (baseline + alpha * (adj_tensor - baseline)).requires_grad_(True)
                    self.model.zero_grad()
                    output = self.model(x_b, adj_interpolated)
                    loss = criterion(output, y_b.squeeze(-1))
                    loss.backward()
                    if adj_interpolated.grad is not None:
                        batch_total_gradients += adj_interpolated.grad
                ig_batch = (adj_tensor - baseline) * (batch_total_gradients / self.config.IG_STEPS)
                accumulated_gradients += ig_batch.abs().cpu().detach().numpy()
            else:  # Saliency (Vanilla Gradients)
                adj_tensor_grad = adj_tensor.clone().detach().requires_grad_(True)
                self.model.zero_grad()
                output = self.model(x_b, adj_tensor_grad)
                loss = criterion(output, y_b.squeeze(-1))
                loss.backward()
                if adj_tensor_grad.grad is not None:
                    accumulated_gradients += adj_tensor_grad.grad.abs().cpu().detach().numpy()

        return accumulated_gradients / len(loader) if loader else accumulated_gradients

    def calculate_fidelity_sparsity(self, eval_fn, loader, adj_phys_np, mask_rank, rmse_orig, saliency_map, ig_map):
        """Calculates Fidelity vs. Sparsity, comparing three explanation methods."""
        print("\nCalculating Fidelity vs. Sparsity (comparing methods)...")
        num_nodes = adj_phys_np.shape[0]

        edges = []
        for r in range(num_nodes):
            for c in range(r + 1, num_nodes):
                if adj_phys_np[r, c] > 0:
                    edges.append({
                        "coords": (r, c),
                        "traffexplainer": mask_rank[r, c],
                        "saliency": saliency_map[r, c],
                        "ig": ig_map[r, c]
                    })

        edges.sort(key=lambda x: abs(x["traffexplainer"] - 1), reverse=True)
        M = len(edges)
        if M == 0:
            print("Warning: No edges in graph to calculate fidelity.")
            return pd.DataFrame()

        results = []
        for s_target in self.config.FIDELITY_SPARSITY_LEVELS:
            m_remove = int(round((1 - s_target) * M))
            adj_p_pert_np = adj_phys_np.copy()
            removed_edges_info = edges[:m_remove]

            for edge_info in removed_edges_info:
                r, c = edge_info["coords"]
                adj_p_pert_np[r, c] = adj_p_pert_np[c, r] = 0.

            metrics, _, _ = eval_fn(loader, adj_p_pert_np, load_best=True)
            rmse_pert = metrics.get('RMSE', float('inf'))
            m_actual = len(removed_edges_info)

            results.append({
                "S_Target": s_target,
                "S_Actual": (M - m_actual) / M if M > 0 else 1.0,
                "Fidelity_Absolute": rmse_pert - rmse_orig,
                "Fidelity_Relative": (rmse_pert - rmse_orig) / rmse_orig if abs(rmse_orig) > 1e-6 else 0,
                "Avg_Saliency_Removed": np.mean([e['saliency'] for e in removed_edges_info]) if m_actual > 0 else 0,
                "Avg_IG_Removed": np.mean([e['ig'] for e in removed_edges_info]) if m_actual > 0 else 0,
                "RMSE_Pert": rmse_pert,
                "RMSE_Orig": rmse_orig
            })
        return pd.DataFrame(results)