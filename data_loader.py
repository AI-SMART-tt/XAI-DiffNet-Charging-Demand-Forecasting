

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from config import Config


class DataProcessor:
    """Handles loading, aligning, cleaning, and preparing all input data for the model."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.node_ids = None
        self.num_nodes = None

    def load_data(self):
        """Loads the adjacency matrix and the time-series charging data."""
        print("Loading data...")
        try:
            adj_df = pd.read_csv(self.config.ADJACENCY_MATRIX_FILE, index_col=0)
            adj_df.index = adj_df.index.map(str)
            adj_df.columns = adj_df.columns.map(str)
            self.node_ids = adj_df.index.tolist()
            self.num_nodes = adj_df.shape[0]
            adj_df_reordered = adj_df.reindex(index=self.node_ids, columns=self.node_ids).fillna(0)
            adj_matrix = adj_df_reordered.values
            print(f"Adjacency matrix loaded successfully: {self.num_nodes} nodes.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load adjacency matrix: {e} @ {self.config.ADJACENCY_MATRIX_FILE}")

        try:
            soc_df_raw = pd.read_csv(self.config.SOC_DATA_FILE, index_col=0)
            soc_df_raw.columns = soc_df_raw.columns.map(str)
            soc_df_final_aligned = pd.DataFrame(np.nan, index=soc_df_raw.index, columns=self.node_ids)
            common_columns = [col for col in self.node_ids if col in soc_df_raw.columns]
            if not common_columns:
                raise ValueError("No matching node IDs found between SOC data and adjacency matrix.")
            soc_df_final_aligned[common_columns] = soc_df_raw[common_columns]
            soc_values = soc_df_final_aligned.values.astype(np.float32)

            if np.isnan(soc_values).any():
                print("Missing values detected in SOC data. Performing imputation...")
                col_means = np.nanmean(soc_values, axis=0)
                global_mean = np.nanmean(col_means) if not np.all(np.isnan(col_means)) else 0
                for i in range(soc_values.shape[1]):
                    if np.isnan(col_means[i]):
                        soc_values[:, i] = global_mean
                    else:
                        soc_values[np.isnan(soc_values[:, i]), i] = col_means[i]
                if np.isnan(soc_values).any():
                    soc_values = np.nan_to_num(soc_values, nan=0.0)
                print(f"SOC data loaded, aligned, and imputed successfully. Shape: {soc_values.shape}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to process SOC data: {e} @ {self.config.SOC_DATA_FILE}")

        return adj_matrix, soc_values

    def create_sequences(self, data: np.ndarray):
        """Converts time-series data into a supervised learning format (X, Y)."""
        X, Y = [], []
        seq_len, pred_len = self.config.SEQ_LEN, self.config.PRED_LEN
        if len(data) < seq_len + pred_len:
            raise ValueError(f"Total data length ({len(data)}) is too short for sequences.")
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i: i + seq_len])
            Y.append(data[i + seq_len: i + seq_len + pred_len])
        return np.array(X), np.array(Y)

    def prepare_datasets(self, soc_values: np.ndarray):
        """Prepares and splits data into PyTorch DataLoaders."""
        print("Preparing datasets...")
        soc_scaled = self.scaler.fit_transform(soc_values)
        X_data, Y_data = self.create_sequences(soc_scaled)

        if X_data.shape[0] == 0:
            raise ValueError("Failed to create any valid sequences from the data.")

        X_data = X_data[..., np.newaxis]
        Y_data = Y_data[..., np.newaxis]

        X_tensor = torch.FloatTensor(X_data)
        Y_tensor = torch.FloatTensor(Y_data)

        total_samples = len(X_tensor)
        train_size = int(self.config.TRAIN_RATIO * total_samples)
        val_size = int(self.config.VAL_RATIO * total_samples)

        train_X, train_Y = X_tensor[:train_size], Y_tensor[:train_size]
        val_X, val_Y = X_tensor[train_size:train_size + val_size], Y_tensor[train_size:train_size + val_size]
        test_X, test_Y = X_tensor[train_size + val_size:], Y_tensor[train_size + val_size:]

        bs = self.config.BATCH_SIZE
        train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=bs, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=bs, shuffle=False)

        print(f"Dataset preparation complete - Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
        return train_loader, val_loader, test_loader, (X_tensor.shape, Y_tensor.shape)