import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from itertools import product
import time
import traceback
import warnings

warnings.filterwarnings('ignore')


class Config:

    INPUT_GRAPH_DIR = "data_graph/graph2/"
    ADJACENCY_FILE = "adjacency_matrix.csv"
    SOC_DATA_FILE = './data0/charging_energy_all.csv'
    TOWN_IDS_FILE = "data_graph/graph2/town_ids.csv"
    SHAPEFILE_PATH = "data_shap/Shanghai_with_ID_merged.shp"

    SEMANTIC_MATRICES = {
        "distance": {"file": "distance_matrix.csv",             "name": "Spatial Distance",    "is_distance": True},
        "poi":      {"file": "POI_matrix.csv",                  "name": "Functional Similarity","is_distance": False},
        "od":       {"file": "OD_matrix.csv",                   "name": "Travel Orders",        "is_distance": False},
        "cosine":   {"file": "similarity_matrix_cosine.csv",    "name": "Demand Pattern",       "is_distance": False},
    }

    EXECUTION_MODE = 'fused'
    REMOTE_EDGE_STRATEGY = 'top_k'
    SW_K = 3
    SW_PERCENTILE = 89
    GRID_SEARCH_STEP = 0.1
    GRID_SEARCH_EPOCHS = 15

    SEQ_LEN = 12
    PRED_LEN = 3
    RNN_UNITS = 64
    NUM_RNN_LAYERS = 2
    MAX_DIFFUSION_STEP = 3

    NUM_EPOCHS_GNN = 100
    NUM_EPOCHS_MASK = 50
    BATCH_SIZE = 64
    LEARNING_RATE_GNN = 0.001
    LEARNING_RATE_MASK = 0.005

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = './output_xai_diffnet'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2


class SmallWorldNetworkBuilder:

    def __init__(self, config: Config):
        self.config = config
        self.preprocessed_matrices = {}
        self.adj_df = None

    def load_adjacency_matrix(self):
        file_path = os.path.join(self.config.INPUT_GRAPH_DIR, self.config.ADJACENCY_FILE)
        df = self._read_csv_with_encoding(file_path, index_col=0)
        df.index = df.index.astype(int)
        df.columns = df.columns.astype(int)
        self.adj_df = df
        print(f"Adjacency matrix loaded: {df.shape[0]} nodes")
        return df

    def load_all_semantic_matrices(self):
        print("\n--- Loading and preprocessing semantic matrices ---")
        for key, params in self.config.SEMANTIC_MATRICES.items():
            file_path = os.path.join(self.config.INPUT_GRAPH_DIR, params["file"])
            df = self._load_and_preprocess_matrix(file_path, params["is_distance"])
            if df is not None:
                self.preprocessed_matrices[key] = df
                print(f"  + {params['name']} ({key}) loaded successfully")
            else:
                print(f"  - {params['name']} ({key}) failed to load")
        return self.preprocessed_matrices

    def fuse_semantic_matrices(self, weights: dict):
        fused = pd.DataFrame(
            np.zeros_like(self.adj_df.values, dtype=float),
            index=self.adj_df.index, columns=self.adj_df.columns
        )
        for key, weight in weights.items():
            if key in self.preprocessed_matrices and self.preprocessed_matrices[key] is not None:
                fused += self.preprocessed_matrices[key] * weight
        return fused

    def construct_small_world_network(self, S_fused: pd.DataFrame):
        A = self.adj_df
        strategy = self.config.REMOTE_EDGE_STRATEGY

        if strategy == 'percentile':
            W_remote = self._create_remote_edges_percentile(S_fused, A, self.config.SW_PERCENTILE)
        elif strategy == 'top_k':
            W_remote = self._create_remote_edges_top_k(S_fused, A, self.config.SW_K)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        W_combined = A.values.astype(float) + W_remote.values
        W_star = (W_combined + W_combined.T) / 2.0
        W_star_df = pd.DataFrame(W_star, index=A.index, columns=A.columns)

        return W_star_df, W_remote

    def _create_remote_edges_percentile(self, S, A, percentile):
        S_values = S.values.copy()
        A_values = A.values.astype(float)
        mask = (1 - A_values).astype(bool)
        np.fill_diagonal(mask, False)
        candidate_values = S_values[mask]
        if len(candidate_values) == 0:
            return pd.DataFrame(np.zeros_like(A_values), index=S.index, columns=S.columns)
        threshold = np.percentile(candidate_values[candidate_values > 0], percentile)
        W = np.zeros_like(S_values)
        select = mask & (S_values >= threshold)
        W[select] = S_values[select]
        return pd.DataFrame(W, index=S.index, columns=S.columns)

    def _create_remote_edges_top_k(self, S, A, k):
        S_values = S.values.copy()
        A_values = A.values.astype(float)
        mask = (1 - A_values).astype(bool)
        np.fill_diagonal(mask, False)
        S_masked = S_values * mask

        W = np.zeros_like(S_values)
        for i in range(S_values.shape[0]):
            row = S_masked[i, :]
            if np.sum(row > 0) == 0:
                continue
            top_k_indices = np.argsort(row)[-k:]
            for j in top_k_indices:
                if row[j] > 0:
                    W[i, j] = row[j]
        return pd.DataFrame(W, index=S.index, columns=S.columns)

    def grid_search_fusion_weights(self, evaluate_fn):
        keys = list(self.preprocessed_matrices.keys())
        m = len(keys)
        step = self.config.GRID_SEARCH_STEP
        weight_values = np.arange(0, 1 + step / 2, step)
        weight_values = np.round(weight_values, 2)

        best_mae = float('inf')
        best_weights = None

        print(f"\n{'='*60}")
        print(f"  Grid search for fusion weights (step={step}, num_matrices={m})")
        print(f"{'='*60}")

        all_combos = list(product(weight_values, repeat=m))
        valid_combos = [(combo, dict(zip(keys, combo)))
                        for combo in all_combos
                        if abs(sum(combo) - 1.0) < 1e-6]

        print(f"Total combinations: {len(all_combos)}, Valid combinations: {len(valid_combos)}")

        for idx, (combo, weights_dict) in enumerate(valid_combos):
            S_fused = self.fuse_semantic_matrices(weights_dict)
            W_star_df, _ = self.construct_small_world_network(S_fused)
            W_star = W_star_df.values

            mae = evaluate_fn(W_star)

            if mae < best_mae:
                best_mae = mae
                best_weights = weights_dict.copy()
                print(f"  [New best #{idx+1}/{len(valid_combos)}] "
                      f"weights={weights_dict}, MAE={mae:.4f}")

            if (idx + 1) % 20 == 0:
                print(f"  ... Evaluated {idx+1}/{len(valid_combos)} combinations, "
                      f"current best MAE={best_mae:.4f}")

        print(f"\nGrid search complete. Best weights: {best_weights}, Best MAE: {best_mae:.4f}")
        return best_weights, best_mae

    @staticmethod
    def _read_csv_with_encoding(file_path, **kwargs):
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file: {file_path}")

    def _load_and_preprocess_matrix(self, file_path, is_distance=False):
        try:
            df = self._read_csv_with_encoding(file_path, index_col=0)
            df.index = df.index.astype(int)
            df.columns = df.columns.astype(int)
            matrix = df.values.astype(float)

            scaler = MinMaxScaler()
            matrix = scaler.fit_transform(matrix)

            if is_distance:
                matrix = 1 - matrix

            np.fill_diagonal(matrix, 0)
            return pd.DataFrame(matrix, index=df.index, columns=df.columns)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"  Load error ({file_path}): {e}")
            return None


class DataProcessor:

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.node_ids = None
        self.num_nodes = None

    def load_soc_data(self, node_ids: list):
        self.node_ids = [str(nid) for nid in node_ids]
        self.num_nodes = len(self.node_ids)

        soc_df_raw = pd.read_csv(self.config.SOC_DATA_FILE, index_col=0)
        soc_df_raw.columns = soc_df_raw.columns.map(str)

        soc_df_aligned = pd.DataFrame(np.nan, index=soc_df_raw.index, columns=self.node_ids)
        common_cols = [col for col in self.node_ids if col in soc_df_raw.columns]
        if not common_cols:
            raise ValueError("SOC data columns do not match any node IDs")
        soc_df_aligned[common_cols] = soc_df_raw[common_cols]
        soc_values = soc_df_aligned.values.astype(np.float32)

        if np.isnan(soc_values).any():
            col_means = np.nanmean(soc_values, axis=0)
            global_mean = np.nanmean(col_means) if not np.all(np.isnan(col_means)) else 0
            for i in range(soc_values.shape[1]):
                if np.isnan(col_means[i]):
                    soc_values[:, i] = global_mean
                else:
                    soc_values[np.isnan(soc_values[:, i]), i] = col_means[i]
            soc_values = np.nan_to_num(soc_values, nan=0.0)

        print(f"SOC data loaded: {soc_values.shape}")
        return soc_values

    def create_sequences(self, data: np.ndarray):
        X, Y = [], []
        seq_len, pred_len = self.config.SEQ_LEN, self.config.PRED_LEN
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i: i + seq_len])
            Y.append(data[i + seq_len: i + seq_len + pred_len])
        return np.array(X), np.array(Y)

    def prepare_datasets(self, soc_values: np.ndarray):
        soc_scaled = self.scaler.fit_transform(soc_values)
        X_data, Y_data = self.create_sequences(soc_scaled)

        if X_data.shape[0] == 0:
            raise ValueError("No valid sequences could be created")

        X_data = X_data[..., np.newaxis]
        Y_data = Y_data[..., np.newaxis]

        X_tensor = torch.FloatTensor(X_data).to(self.config.DEVICE)
        Y_tensor = torch.FloatTensor(Y_data).to(self.config.DEVICE)

        total = len(X_tensor)
        train_size = int(self.config.TRAIN_RATIO * total)
        val_size = int(self.config.VAL_RATIO * total)

        bs = self.config.BATCH_SIZE
        train_loader = DataLoader(
            TensorDataset(X_tensor[:train_size], Y_tensor[:train_size]),
            batch_size=bs, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_tensor[train_size:train_size + val_size],
                          Y_tensor[train_size:train_size + val_size]),
            batch_size=bs, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(X_tensor[train_size + val_size:],
                          Y_tensor[train_size + val_size:]),
            batch_size=bs, shuffle=False)

        print(f"Dataset splits: train={train_size}, val={val_size}, "
              f"test={total - train_size - val_size}")
        return train_loader, val_loader, test_loader


class DiffusionGraphConv(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, max_diffusion_step: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_diffusion_step = max_diffusion_step

        self.weights_forward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            for _ in range(max_diffusion_step)
        ])
        self.weights_backward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            for _ in range(max_diffusion_step)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in self.weights_forward:
            nn.init.xavier_uniform_(w)
        for w in self.weights_backward:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, support_fwd: torch.Tensor, support_bwd: torch.Tensor):
        output = torch.zeros(x.size(0), x.size(1), self.output_dim, device=x.device)

        x_fwd = x
        x_bwd = x

        for k in range(self.max_diffusion_step):
            if k > 0:
                x_fwd = torch.einsum('mn,bnd->bmd', support_fwd, x_fwd)
                x_bwd = torch.einsum('mn,bnd->bmd', support_bwd, x_bwd)

            output += torch.einsum('bnd,do->bno', x_fwd, self.weights_forward[k])
            output += torch.einsum('bnd,do->bno', x_bwd, self.weights_backward[k])

        output += self.bias
        return output


class DCRNNCell(nn.Module):

    def __init__(self, input_dim: int, rnn_units: int, max_diffusion_step: int, num_nodes: int):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.num_nodes = num_nodes

        combined_dim = input_dim + rnn_units

        self.dgc_path1_gate = DiffusionGraphConv(combined_dim, rnn_units * 2, max_diffusion_step)
        self.dgc_path1_candidate = DiffusionGraphConv(combined_dim, rnn_units, max_diffusion_step)

        self.dgc_path2_gate = DiffusionGraphConv(combined_dim, rnn_units * 2, max_diffusion_step)
        self.dgc_path2_candidate = DiffusionGraphConv(combined_dim, rnn_units, max_diffusion_step)

    def forward(self, x, h, supports1, supports2):
        xh = torch.cat([x, h], dim=-1)

        ru1 = torch.sigmoid(self.dgc_path1_gate(xh, supports1[0], supports1[1]))
        r1, u1 = torch.chunk(ru1, 2, dim=-1)
        c1 = torch.tanh(self.dgc_path1_candidate(
            torch.cat([x, r1 * h], dim=-1), supports1[0], supports1[1]))

        ru2 = torch.sigmoid(self.dgc_path2_gate(xh, supports2[0], supports2[1]))
        r2, u2 = torch.chunk(ru2, 2, dim=-1)
        c2 = torch.tanh(self.dgc_path2_candidate(
            torch.cat([x, r2 * h], dim=-1), supports2[0], supports2[1]))

        u = (u1 + u2) / 2.0
        c = (c1 + c2) / 2.0

        new_h = (1.0 - u) * h + u * c
        return new_h


class XAIDiffNet(nn.Module):

    def __init__(self, num_nodes: int, seq_len: int, pred_len: int,
                 rnn_units: int, num_rnn_layers: int, max_diffusion_step: int,
                 input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rnn_units = rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_fc = nn.Linear(input_dim, rnn_units)
        self.output_fc = nn.Linear(rnn_units, output_dim)

        self.encoder_cells = nn.ModuleList([
            DCRNNCell(rnn_units, rnn_units, max_diffusion_step, num_nodes)
            for _ in range(num_rnn_layers)
        ])
        self.decoder_cells = nn.ModuleList([
            DCRNNCell(rnn_units, rnn_units, max_diffusion_step, num_nodes)
            for _ in range(num_rnn_layers)
        ])

    @staticmethod
    def _compute_supports(adj_tensor: torch.Tensor):
        device = adj_tensor.device
        N = adj_tensor.size(0)
        adj_hat = adj_tensor + torch.eye(N, device=device)

        d_out = adj_hat.sum(dim=1)
        d_out_inv = torch.where(d_out > 0, 1.0 / d_out, torch.zeros_like(d_out))
        P_fwd = torch.diag(d_out_inv) @ adj_hat

        adj_hat_T = adj_hat.T
        d_in = adj_hat_T.sum(dim=1)
        d_in_inv = torch.where(d_in > 0, 1.0 / d_in, torch.zeros_like(d_in))
        P_bwd = torch.diag(d_in_inv) @ adj_hat_T

        return P_fwd, P_bwd

    def forward(self, x, adj_matrix: np.ndarray,
                spatial_mask_1=None, spatial_mask_2=None,
                temporal_mask=None):
        batch_size = x.size(0)
        device = x.device

        if temporal_mask is not None:
            x = x * (torch.tanh(temporal_mask) + 1).unsqueeze(0).unsqueeze(-1)

        adj_torch = torch.from_numpy(adj_matrix).float().to(device)

        adj_path1 = adj_torch.clone()
        if spatial_mask_1 is not None:
            sym_mask1 = (spatial_mask_1 + spatial_mask_1.T) / 2
            adj_path1 = adj_path1 * (torch.tanh(sym_mask1) + 1)
        supports1 = self._compute_supports(adj_path1)

        adj_path2 = adj_torch.clone()
        if spatial_mask_2 is not None:
            sym_mask2 = (spatial_mask_2 + spatial_mask_2.T) / 2
            adj_path2 = adj_path2 * (torch.tanh(sym_mask2) + 1)
        supports2 = self._compute_supports(adj_path2)

        h_enc = [torch.zeros(batch_size, self.num_nodes, self.rnn_units, device=device)
                 for _ in range(self.num_rnn_layers)]

        for t in range(self.seq_len):
            x_t = x[:, t, :, :]
            x_t_emb = self.input_fc(x_t)

            h_in = x_t_emb
            for i in range(self.num_rnn_layers):
                h_enc[i] = self.encoder_cells[i](h_in, h_enc[i], supports1, supports2)
                h_in = h_enc[i]

        outputs = []
        h_dec = h_enc
        go_symbol = torch.zeros(batch_size, self.num_nodes, self.input_dim, device=device)

        for t in range(self.pred_len):
            x_dec = self.input_fc(go_symbol)
            h_in = x_dec
            for i in range(self.num_rnn_layers):
                h_dec[i] = self.decoder_cells[i](h_in, h_dec[i], supports1, supports2)
                h_in = h_dec[i]
            output_t = self.output_fc(h_dec[-1])
            outputs.append(output_t)
            go_symbol = output_t

        outputs = torch.stack(outputs, dim=1)
        return outputs.squeeze(-1)


class ModelTrainer:

    def __init__(self, model: XAIDiffNet, config: Config, scaler: StandardScaler):
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.scaler = scaler
        self.best_model_path = Path(config.OUTPUT_DIR) / 'best_gnn_model.pth'

    def train_gnn(self, train_loader, val_loader, adj_matrix, num_epochs=None):
        epochs = num_epochs or self.config.NUM_EPOCHS_GNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE_GNN)
        best_val_loss = float('inf')
        total_time = 0

        for epoch in range(epochs):
            t0 = time.time()

            self.model.train()
            train_loss = 0
            for x_b, y_b in train_loader:
                optimizer.zero_grad()
                out = self.model(x_b, adj_matrix)
                loss = self.criterion(out, y_b.squeeze(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    out = self.model(x_b, adj_matrix)
                    val_loss += self.criterion(out, y_b.squeeze(-1)).item()
            avg_val = val_loss / max(len(val_loader), 1)

            dt = time.time() - t0
            total_time += dt

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), self.best_model_path)

            if (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"  Ep {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | {dt:.1f}s")

        avg_epoch_time = total_time / max(epochs, 1)
        print(f"  Training complete. Best val loss: {best_val_loss:.5f}")
        return avg_epoch_time

    def evaluate_on_val_mae(self, val_loader, adj_matrix):
        self.model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                out = self.model(x_b, adj_matrix)
                preds.append(out.cpu().numpy())
                actuals.append(y_b.cpu().numpy())

        if not preds:
            return float('inf')

        preds_np = np.concatenate(preds, axis=0)
        actuals_np = np.concatenate(actuals, axis=0).squeeze(-1)

        n_samples, pred_len, n_nodes = preds_np.shape
        preds_inv = self.scaler.inverse_transform(
            preds_np.reshape(-1, n_nodes)).reshape(n_samples, pred_len, n_nodes)
        actuals_inv = self.scaler.inverse_transform(
            actuals_np.reshape(-1, n_nodes)).reshape(n_samples, pred_len, n_nodes)

        mae = mean_absolute_error(actuals_inv.flatten(), preds_inv.flatten())
        return mae

    def train_masks(self, train_loader, adj_matrix, num_nodes):
        print(f"\nStarting mask training (Epochs: {self.config.NUM_EPOCHS_MASK})...")
        device = self.config.DEVICE

        sm1 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=device))
        sm2 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=device))
        tm = nn.Parameter(torch.zeros(self.config.SEQ_LEN, num_nodes, device=device))
        optimizer = optim.Adam([sm1, sm2, tm], lr=self.config.LEARNING_RATE_MASK)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.train()
        for epoch in range(self.config.NUM_EPOCHS_MASK):
            epoch_loss = 0
            for x_b, y_b in train_loader:
                optimizer.zero_grad()
                out = self.model(x_b, adj_matrix, sm1, sm2, tm)
                loss = self.criterion(out, y_b.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Mask Ep {epoch+1:3d} | Loss: {epoch_loss/max(len(train_loader),1):.5f}")

        for param in self.model.parameters():
            param.requires_grad = True

        print("Mask training complete.")
        return sm1, sm2, tm

    def evaluate_model(self, test_loader, adj_matrix,
                       sm1=None, sm2=None, tm=None, load_best=True):
        if load_best and self.best_model_path.exists():
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.config.DEVICE))

        self.model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                out = self.model(x_b, adj_matrix, sm1, sm2, tm)
                preds.append(out.cpu().numpy())
                actuals.append(y_b.cpu().numpy())

        if not preds:
            return {}

        preds_np = np.concatenate(preds, axis=0)
        actuals_np = np.concatenate(actuals, axis=0).squeeze(-1)

        n_samples, pred_len, n_nodes = preds_np.shape
        preds_inv = self.scaler.inverse_transform(
            preds_np.reshape(-1, n_nodes)).reshape(n_samples, pred_len, n_nodes)
        actuals_inv = self.scaler.inverse_transform(
            actuals_np.reshape(-1, n_nodes)).reshape(n_samples, pred_len, n_nodes)

        metrics = self._calculate_metrics(actuals_inv, preds_inv)
        return metrics

    @staticmethod
    def _calculate_metrics(actuals, preds, epsilon=1.0):
        a = actuals.flatten()
        p = preds.flatten()
        return {
            'MAE': mean_absolute_error(a, p),
            'RMSE': np.sqrt(mean_squared_error(a, p)),
            'MAPE': np.mean(np.abs((a - p) / (np.maximum(np.abs(a), epsilon)))) * 100,
            'R2': r2_score(a, p)
        }


def print_metrics(metrics, stage_name=""):
    print(f"\n--- {stage_name} ---")
    print(f"  MAE: {metrics.get('MAE', 0):.4f}, "
          f"RMSE: {metrics.get('RMSE', 0):.4f}, "
          f"MAPE: {metrics.get('MAPE', 0):.2f}%, "
          f"R2: {metrics.get('R2', 0):.4f}")


def main():
    config = Config()
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {config.DEVICE}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    total_start = time.time()

    try:
        print("\n" + "=" * 60)
        print("  Stage 1: Small-World Network Construction")
        print("=" * 60)

        sw_builder = SmallWorldNetworkBuilder(config)
        adj_df = sw_builder.load_adjacency_matrix()
        sw_builder.load_all_semantic_matrices()

        node_ids = adj_df.index.tolist()
        num_nodes = len(node_ids)

        print("\n" + "=" * 60)
        print("  Stage 2: Data Preparation")
        print("=" * 60)

        dp = DataProcessor(config)
        soc_values = dp.load_soc_data(node_ids)
        train_loader, val_loader, test_loader = dp.prepare_datasets(soc_values)

        print("\n" + "=" * 60)
        print("  Stage 3: Grid Search for Optimal Fusion Weights")
        print("=" * 60)

        if config.EXECUTION_MODE == 'fused':

            def evaluate_weight_combination(W_star: np.ndarray):
                model = XAIDiffNet(
                    num_nodes=num_nodes,
                    seq_len=config.SEQ_LEN,
                    pred_len=config.PRED_LEN,
                    rnn_units=config.RNN_UNITS,
                    num_rnn_layers=config.NUM_RNN_LAYERS,
                    max_diffusion_step=config.MAX_DIFFUSION_STEP,
                    input_dim=1
                ).to(config.DEVICE)

                trainer = ModelTrainer(model, config, dp.scaler)
                trainer.train_gnn(train_loader, val_loader, W_star,
                                  num_epochs=config.GRID_SEARCH_EPOCHS)
                mae = trainer.evaluate_on_val_mae(val_loader, W_star)
                return mae

            best_weights, best_mae = sw_builder.grid_search_fusion_weights(
                evaluate_weight_combination)

            S_fused = sw_builder.fuse_semantic_matrices(best_weights)
            W_star_df, W_remote = sw_builder.construct_small_world_network(S_fused)
            W_star = W_star_df.values.astype(np.float64)

            W_star_df.to_csv(Path(config.OUTPUT_DIR) / 'W_star_optimal.csv')
            print(f"\nOptimal small-world network saved. Weights: {best_weights}")

        elif config.EXECUTION_MODE == 'single':
            print("Single-semantic mode: building network for each semantic matrix individually")
            for key, S_df in sw_builder.preprocessed_matrices.items():
                if S_df is not None:
                    W_star_df, _ = sw_builder.construct_small_world_network(S_df)
                    W_star = W_star_df.values.astype(np.float64)
                    print(f"Using '{key}' semantic matrix to build small-world network")
                    break
        else:
            raise ValueError(f"Unknown execution mode: {config.EXECUTION_MODE}")

        print("\n" + "=" * 60)
        print("  Stage 4: Full Training of XAI-DiffNet")
        print("=" * 60)

        model = XAIDiffNet(
            num_nodes=num_nodes,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            rnn_units=config.RNN_UNITS,
            num_rnn_layers=config.NUM_RNN_LAYERS,
            max_diffusion_step=config.MAX_DIFFUSION_STEP,
            input_dim=1
        ).to(config.DEVICE)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        trainer = ModelTrainer(model, config, dp.scaler)

        avg_epoch_time = trainer.train_gnn(train_loader, val_loader, W_star)

        gnn_metrics = trainer.evaluate_model(test_loader, W_star, load_best=True)
        print_metrics(gnn_metrics, "Base GNN Model - Test Set Performance")

        print("\n" + "=" * 60)
        print("  Stage 5: Interpretability Mask Training")
        print("=" * 60)

        model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=config.DEVICE))

        sm1, sm2, tm = trainer.train_masks(train_loader, W_star, num_nodes)

        masked_metrics = trainer.evaluate_model(
            test_loader, W_star, sm1, sm2, tm, load_best=False)
        print_metrics(masked_metrics, "Masked GNN Model - Test Set Performance")

        results = pd.DataFrame([
            {'Model_Type': 'Original_GNN', **gnn_metrics,
             'Avg_Epoch_Time_s': avg_epoch_time},
            {'Model_Type': 'Masked_GNN', **masked_metrics}
        ])
        output_file = Path(config.OUTPUT_DIR) / f'results_pred{config.PRED_LEN}.csv'
        results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved: {output_file}")

    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
    finally:
        total_time = time.time() - total_start
        print(f"\nTotal runtime: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.1f}s")


if __name__ == "__main__":
    main()
