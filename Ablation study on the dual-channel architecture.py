import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import time
import traceback
import warnings

warnings.filterwarnings('ignore')


class Config:

    ADJACENCY_MATRIX_FILE = './output_xai_diffnet/W_star_optimal.csv'
    SOC_DATA_FILE = './data0/charging_energy_all.csv'

    SEQ_LEN = 12
    RNN_UNITS = 64
    NUM_RNN_LAYERS = 2
    MAX_DIFFUSION_STEP = 3

    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    PRED_LENS = [3, 6, 12]
    PATH_MODES = ['path1_only', 'path2_only', 'dual']
    PATH_MODE_NAMES = {
        'path1_only': 'w/o Path 2',
        'path2_only': 'w/o Path 1',
        'dual': 'Dual-path (ours)'
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = './output_ablation'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2


class DataProcessor:

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.node_ids = None
        self.num_nodes = None
        self._soc_scaled = None

    def load_data(self):
        print("  Loading adjacency matrix...")
        adj_df = self._read_csv(self.config.ADJACENCY_MATRIX_FILE, index_col=0)
        adj_df.index = adj_df.index.map(str)
        adj_df.columns = adj_df.columns.map(str)
        self.node_ids = adj_df.index.tolist()
        self.num_nodes = len(self.node_ids)
        adj_matrix = adj_df.reindex(index=self.node_ids, columns=self.node_ids).fillna(0).values
        print(f"    Nodes: {self.num_nodes}, Edges: {int(adj_matrix.sum())}")

        print("  Loading SOC data...")
        soc_raw = pd.read_csv(self.config.SOC_DATA_FILE, index_col=0)
        soc_raw.columns = soc_raw.columns.map(str)
        soc_aligned = pd.DataFrame(np.nan, index=soc_raw.index, columns=self.node_ids)
        common = [c for c in self.node_ids if c in soc_raw.columns]
        if not common:
            raise ValueError("SOC data columns do not match adjacency matrix node IDs")
        soc_aligned[common] = soc_raw[common]
        soc_values = soc_aligned.values.astype(np.float32)

        if np.isnan(soc_values).any():
            col_means = np.nanmean(soc_values, axis=0)
            g_mean = np.nanmean(col_means) if not np.all(np.isnan(col_means)) else 0
            for i in range(soc_values.shape[1]):
                mask = np.isnan(soc_values[:, i])
                soc_values[mask, i] = col_means[i] if not np.isnan(col_means[i]) else g_mean
            soc_values = np.nan_to_num(soc_values, nan=0.0)

        self._soc_scaled = self.scaler.fit_transform(soc_values)
        print(f"    SOC data shape: {soc_values.shape}")

        return adj_matrix, soc_values

    def prepare_datasets(self, pred_len: int):
        X, Y = [], []
        sl = self.config.SEQ_LEN
        data = self._soc_scaled
        for i in range(len(data) - sl - pred_len + 1):
            X.append(data[i: i + sl])
            Y.append(data[i + sl: i + sl + pred_len])
        X = np.array(X)[..., np.newaxis]
        Y = np.array(Y)[..., np.newaxis]

        X_t = torch.FloatTensor(X).to(self.config.DEVICE)
        Y_t = torch.FloatTensor(Y).to(self.config.DEVICE)

        total = len(X_t)
        tr = int(self.config.TRAIN_RATIO * total)
        va = int(self.config.VAL_RATIO * total)
        bs = self.config.BATCH_SIZE

        train_dl = DataLoader(TensorDataset(X_t[:tr], Y_t[:tr]),
                              batch_size=bs, shuffle=True)
        val_dl = DataLoader(TensorDataset(X_t[tr:tr+va], Y_t[tr:tr+va]),
                            batch_size=bs, shuffle=False)
        test_dl = DataLoader(TensorDataset(X_t[tr+va:], Y_t[tr+va:]),
                             batch_size=bs, shuffle=False)

        print(f"    pred_len={pred_len}: train={tr}, val={va}, test={total-tr-va}")
        return train_dl, val_dl, test_dl

    @staticmethod
    def _read_csv(path, **kwargs):
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file: {path}")


class DiffusionGraphConv(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, max_diffusion_step: int):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        self.output_dim = output_dim

        self.W_fwd = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            for _ in range(max_diffusion_step)
        ])
        self.W_bwd = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            for _ in range(max_diffusion_step)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self._init()

    def _init(self):
        for w in self.W_fwd:
            nn.init.xavier_uniform_(w)
        for w in self.W_bwd:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, P_fwd: torch.Tensor, P_bwd: torch.Tensor):
        out = torch.zeros(x.size(0), x.size(1), self.output_dim, device=x.device)
        x_f, x_b = x, x

        for k in range(self.max_diffusion_step):
            if k > 0:
                x_f = torch.einsum('mn,bnd->bmd', P_fwd, x_f)
                x_b = torch.einsum('mn,bnd->bmd', P_bwd, x_b)
            out = out + x_f @ self.W_fwd[k] + x_b @ self.W_bwd[k]

        return out + self.bias


class DCRNNCell(nn.Module):

    def __init__(self, input_dim: int, rnn_units: int,
                 max_diffusion_step: int, num_nodes: int,
                 path_mode: str = 'dual'):
        super().__init__()
        self.rnn_units = rnn_units
        self.path_mode = path_mode
        combined = input_dim + rnn_units

        if path_mode in ('dual', 'path1_only'):
            self.p1_gate = DiffusionGraphConv(combined, rnn_units * 2, max_diffusion_step)
            self.p1_cand = DiffusionGraphConv(combined, rnn_units, max_diffusion_step)

        if path_mode in ('dual', 'path2_only'):
            self.p2_gate = DiffusionGraphConv(combined, rnn_units * 2, max_diffusion_step)
            self.p2_cand = DiffusionGraphConv(combined, rnn_units, max_diffusion_step)

    def forward(self, x, h, supports1, supports2):
        xh = torch.cat([x, h], dim=-1)

        if self.path_mode == 'dual':
            ru1 = torch.sigmoid(self.p1_gate(xh, supports1[0], supports1[1]))
            r1, u1 = ru1.chunk(2, dim=-1)
            c1 = torch.tanh(self.p1_cand(
                torch.cat([x, r1 * h], dim=-1), supports1[0], supports1[1]))
            ru2 = torch.sigmoid(self.p2_gate(xh, supports2[0], supports2[1]))
            r2, u2 = ru2.chunk(2, dim=-1)
            c2 = torch.tanh(self.p2_cand(
                torch.cat([x, r2 * h], dim=-1), supports2[0], supports2[1]))
            u = (u1 + u2) / 2.0
            c = (c1 + c2) / 2.0

        elif self.path_mode == 'path1_only':
            ru1 = torch.sigmoid(self.p1_gate(xh, supports1[0], supports1[1]))
            r1, u = ru1.chunk(2, dim=-1)
            c = torch.tanh(self.p1_cand(
                torch.cat([x, r1 * h], dim=-1), supports1[0], supports1[1]))

        elif self.path_mode == 'path2_only':
            ru2 = torch.sigmoid(self.p2_gate(xh, supports2[0], supports2[1]))
            r2, u = ru2.chunk(2, dim=-1)
            c = torch.tanh(self.p2_cand(
                torch.cat([x, r2 * h], dim=-1), supports2[0], supports2[1]))

        else:
            raise ValueError(f"Unknown path_mode: {self.path_mode}")

        return (1.0 - u) * h + u * c


class XAIDiffNet(nn.Module):

    def __init__(self, num_nodes, seq_len, pred_len, rnn_units,
                 num_rnn_layers, max_diffusion_step,
                 path_mode='dual', input_dim=1, output_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rnn_units = rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.input_dim = input_dim

        self.input_fc = nn.Linear(input_dim, rnn_units)
        self.output_fc = nn.Linear(rnn_units, output_dim)

        self.encoder = nn.ModuleList([
            DCRNNCell(rnn_units, rnn_units, max_diffusion_step, num_nodes, path_mode)
            for _ in range(num_rnn_layers)
        ])
        self.decoder = nn.ModuleList([
            DCRNNCell(rnn_units, rnn_units, max_diffusion_step, num_nodes, path_mode)
            for _ in range(num_rnn_layers)
        ])

    @staticmethod
    def _compute_supports(adj: torch.Tensor):
        N = adj.size(0)
        dev = adj.device
        A = adj + torch.eye(N, device=dev)

        d_out = A.sum(1)
        d_out_inv = torch.where(d_out > 0, 1.0 / d_out, torch.zeros_like(d_out))
        P_f = torch.diag(d_out_inv) @ A

        At = A.T
        d_in = At.sum(1)
        d_in_inv = torch.where(d_in > 0, 1.0 / d_in, torch.zeros_like(d_in))
        P_b = torch.diag(d_in_inv) @ At

        return (P_f, P_b)

    def forward(self, x, adj_matrix: np.ndarray):
        B = x.size(0)
        dev = x.device
        adj_t = torch.from_numpy(adj_matrix).float().to(dev)

        supports1 = self._compute_supports(adj_t)
        supports2 = self._compute_supports(adj_t)

        h = [torch.zeros(B, self.num_nodes, self.rnn_units, device=dev)
             for _ in range(self.num_rnn_layers)]

        for t in range(self.seq_len):
            inp = self.input_fc(x[:, t, :, :])
            for i in range(self.num_rnn_layers):
                h[i] = self.encoder[i](inp, h[i], supports1, supports2)
                inp = h[i]

        outputs = []
        h_dec = [hi.clone() for hi in h]
        go = torch.zeros(B, self.num_nodes, self.input_dim, device=dev)

        for t in range(self.pred_len):
            inp = self.input_fc(go)
            for i in range(self.num_rnn_layers):
                h_dec[i] = self.decoder[i](inp, h_dec[i], supports1, supports2)
                inp = h_dec[i]
            out_t = self.output_fc(h_dec[-1])
            outputs.append(out_t)
            go = out_t

        return torch.stack(outputs, dim=1).squeeze(-1)


class Trainer:

    def __init__(self, model, config, scaler, experiment_tag=''):
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.scaler = scaler
        tag = experiment_tag.replace(' ', '_').replace('/', '_')
        self.best_path = Path(config.OUTPUT_DIR) / f'best_{tag}.pth'

    def train(self, train_dl, val_dl, adj):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        best_val = float('inf')
        t_total = 0

        for ep in range(self.config.NUM_EPOCHS):
            t0 = time.time()

            self.model.train()
            tr_loss = 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = self.criterion(self.model(xb, adj), yb.squeeze(-1))
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()

            self.model.eval()
            va_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    va_loss += self.criterion(
                        self.model(xb, adj), yb.squeeze(-1)).item()

            avg_tr = tr_loss / max(len(train_dl), 1)
            avg_va = va_loss / max(len(val_dl), 1)
            dt = time.time() - t0
            t_total += dt

            if avg_va < best_val:
                best_val = avg_va
                torch.save(self.model.state_dict(), self.best_path)

            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"      Ep {ep+1:3d}/{self.config.NUM_EPOCHS} | "
                      f"Train: {avg_tr:.5f} | Val: {avg_va:.5f} | {dt:.1f}s")

        avg_ep = t_total / max(self.config.NUM_EPOCHS, 1)
        print(f"      Done. Best Val Loss: {best_val:.5f} | "
              f"Avg Epoch Time: {avg_ep:.1f}s")
        return avg_ep

    def evaluate(self, test_dl, adj):
        if self.best_path.exists():
            self.model.load_state_dict(
                torch.load(self.best_path, map_location=self.config.DEVICE))

        self.model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds.append(self.model(xb, adj).cpu().numpy())
                acts.append(yb.cpu().numpy())

        p = np.concatenate(preds, 0)
        a = np.concatenate(acts, 0).squeeze(-1)
        ns, pl, nn_ = p.shape

        p_inv = self.scaler.inverse_transform(p.reshape(-1, nn_)).reshape(ns, pl, nn_)
        a_inv = self.scaler.inverse_transform(a.reshape(-1, nn_)).reshape(ns, pl, nn_)

        af, pf = a_inv.flatten(), p_inv.flatten()
        return {
            'MAE': mean_absolute_error(af, pf),
            'RMSE': np.sqrt(mean_squared_error(af, pf)),
            'MAPE': np.mean(np.abs((af - pf) / np.maximum(np.abs(af), 1.0))) * 100,
            'R2': r2_score(af, pf)
        }


def main():
    config = Config()
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {config.DEVICE}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"Adjacency matrix: {config.ADJACENCY_MATRIX_FILE}")
    t_start = time.time()

    print(f"\n{'='*70}")
    print(f"  Data Loading")
    print(f"{'='*70}")
    dp = DataProcessor(config)
    adj_matrix, _ = dp.load_data()
    num_nodes = dp.num_nodes

    all_results = []

    print(f"\n{'='*70}")
    print(f"  Dual-Channel Ablation Study")
    print(f"  Path configurations: {list(config.PATH_MODE_NAMES.values())}")
    print(f"  Prediction horizons: {config.PRED_LENS}")
    print(f"  Total experiments: {len(config.PATH_MODES) * len(config.PRED_LENS)}")
    print(f"{'='*70}")

    exp_count = 0
    total_exps = len(config.PATH_MODES) * len(config.PRED_LENS)

    for pred_len in config.PRED_LENS:
        print(f"\n{'─'*60}")
        print(f"  Prediction horizon: {pred_len} steps ({pred_len * 5} minutes)")
        print(f"{'─'*60}")

        train_dl, val_dl, test_dl = dp.prepare_datasets(pred_len)

        for path_mode in config.PATH_MODES:
            exp_count += 1
            mode_name = config.PATH_MODE_NAMES[path_mode]

            print(f"\n    > [{exp_count}/{total_exps}] {mode_name} | pred_len={pred_len}")

            model = XAIDiffNet(
                num_nodes=num_nodes,
                seq_len=config.SEQ_LEN,
                pred_len=pred_len,
                rnn_units=config.RNN_UNITS,
                num_rnn_layers=config.NUM_RNN_LAYERS,
                max_diffusion_step=config.MAX_DIFFUSION_STEP,
                path_mode=path_mode
            ).to(config.DEVICE)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"      Parameters: {n_params:,}")

            tag = f"{path_mode}_pred{pred_len}"
            trainer = Trainer(model, config, dp.scaler, experiment_tag=tag)
            avg_ep_time = trainer.train(train_dl, val_dl, adj_matrix)

            metrics = trainer.evaluate(test_dl, adj_matrix)

            print(f"      MAE={metrics['MAE']:.4f}  "
                  f"RMSE={metrics['RMSE']:.4f}  "
                  f"MAPE={metrics['MAPE']:.2f}%  "
                  f"R2={metrics['R2']:.4f}")

            all_results.append({
                'Configuration': mode_name,
                'path_mode': path_mode,
                'Horizon': f"{pred_len * 5}-min",
                'pred_len': pred_len,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': f"{metrics['MAPE']:.2f}%",
                'R2': metrics['R2'],
                'Params': n_params,
                'Avg_Epoch_Time': f"{avg_ep_time:.1f}s"
            })

            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n\n{'='*70}")
    print(f"  Ablation Study Results Summary")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)

    print(f"\n{'─'*70}")
    header = f"{'Configuration':<20}"
    for pl in config.PRED_LENS:
        header += f" | {'MAE':>7} {'RMSE':>7} {'MAPE':>8}"
    print(header)
    print(f"{'─'*70}")

    for pm in config.PATH_MODES:
        name = config.PATH_MODE_NAMES[pm]
        row = f"{name:<20}"
        for pl in config.PRED_LENS:
            r = results_df[(results_df['path_mode'] == pm) &
                           (results_df['pred_len'] == pl)].iloc[0]
            row += f" | {r['MAE']:7.3f} {r['RMSE']:7.3f} {r['MAPE']:>8}"
        if pm == 'dual':
            row += "  *"
        print(row)

    print(f"{'─'*70}")

    output_path = Path(config.OUTPUT_DIR) / 'ablation_dual_channel_results.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  Full results saved: {output_path}")

    pivot_data = []
    for pm in config.PATH_MODES:
        entry = {'Configuration': config.PATH_MODE_NAMES[pm]}
        for pl in config.PRED_LENS:
            r = results_df[(results_df['path_mode'] == pm) &
                           (results_df['pred_len'] == pl)].iloc[0]
            h = f"{pl*5}min"
            entry[f'{h}_MAE'] = r['MAE']
            entry[f'{h}_RMSE'] = r['RMSE']
            entry[f'{h}_MAPE'] = r['MAPE']
        pivot_data.append(entry)

    pivot_df = pd.DataFrame(pivot_data)
    pivot_path = Path(config.OUTPUT_DIR) / 'ablation_table8_format.csv'
    pivot_df.to_csv(pivot_path, index=False, encoding='utf-8-sig')
    print(f"  Paper table format saved: {pivot_path}")

    dt = time.time() - t_start
    print(f"\n  Total runtime: {dt//3600:.0f}h {(dt%3600)//60:.0f}m {dt%60:.1f}s")


if __name__ == "__main__":
    main()