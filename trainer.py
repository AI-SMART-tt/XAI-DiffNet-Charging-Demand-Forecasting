
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Custom module imports
from config import Config
from model import AdvancedSpatioTemporalGNN


class MetricCalculator:
    """Utility class for calculating performance metrics with safe division."""

    @staticmethod
    def calculate_all_metrics(actuals: np.ndarray, preds: np.ndarray, epsilon: float = 1e-6):
        """Calculates and returns a dictionary of performance metrics."""
        actuals_flat = actuals.flatten()
        preds_flat = preds.flatten()

        # Prepare safe denominators to avoid division by zero
        safe_actuals = np.where(actuals_flat == 0, epsilon, actuals_flat)
        denominator_smape = (np.abs(actuals_flat) + np.abs(preds_flat)) / 2
        safe_denominator_smape = np.where(denominator_smape == 0, epsilon, denominator_smape)
        sum_abs_actuals = np.sum(np.abs(actuals_flat))

        metrics = {
            'MAE': mean_absolute_error(actuals_flat, preds_flat),
            'RMSE': np.sqrt(mean_squared_error(actuals_flat, preds_flat)),
            'MAPE': np.mean(np.abs((actuals_flat - preds_flat) / safe_actuals)) * 100,
            'SMAPE': np.mean(np.abs(preds_flat - actuals_flat) / safe_denominator_smape) * 100,
            'WAPE': np.nan if sum_abs_actuals == 0 else np.sum(
                np.abs(actuals_flat - preds_flat)) / sum_abs_actuals * 100,
            'R2': r2_score(actuals_flat, preds_flat)
        }
        return metrics


class ModelTrainer:
    """Handles training/evaluation for the GNN and the interpretability masks."""

    def __init__(self, model: nn.Module, config: Config, scaler: StandardScaler):
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.scaler = scaler

    def train_gnn(self, train_loader: DataLoader, val_loader: DataLoader, adj_matrix: np.ndarray):
        """Trains the main GNN forecasting model."""
        print(f"Starting GNN training (Epochs: {self.config.NUM_EPOCHS_GNN})...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE_GNN)
        losses = {'train': [], 'val': []}
        best_val_loss = float('inf')
        total_epoch_time = 0

        for epoch in range(self.config.NUM_EPOCHS_GNN):
            epoch_start_time = time.time()
            self.model.train()
            epoch_train_loss = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.config.DEVICE), y_b.to(self.config.DEVICE)
                optimizer.zero_grad()
                out = self.model(x_b, adj_matrix)
                loss = self.criterion(out, y_b.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            avg_tr_loss = epoch_train_loss / len(train_loader) if train_loader else 0
            losses['train'].append(avg_tr_loss)

            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b, y_b = x_b.to(self.config.DEVICE), y_b.to(self.config.DEVICE)
                    out = self.model(x_b, adj_matrix)
                    epoch_val_loss += self.criterion(out, y_b.squeeze(-1)).item()
            avg_val_loss = epoch_val_loss / len(val_loader) if val_loader else 0
            losses['val'].append(avg_val_loss)

            epoch_duration = time.time() - epoch_start_time
            total_epoch_time += epoch_duration

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = Path(self.config.OUTPUT_DIR) / 'checkpoints/best_gnn_model.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

            if (epoch + 1) % 10 == 0:
                print(
                    f"GNN Ep {epoch + 1:3d} | Train Loss: {avg_tr_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Time: {epoch_duration:.2f}s")

        avg_epoch_time = total_epoch_time / self.config.NUM_EPOCHS_GNN if self.config.NUM_EPOCHS_GNN > 0 else 0
        print(f"GNN training complete. Best Val Loss: {best_val_loss:.5f}")
        return losses['train'], losses['val'], avg_epoch_time

    def train_masks(self, train_loader: DataLoader, adj_matrix: np.ndarray, num_nodes: int):
        """Trains the learnable perturbation masks while freezing the GNN model."""
        print(f"Starting mask training (Epochs: {self.config.NUM_EPOCHS_MASK})...")
        sm1 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=self.config.DEVICE))
        sm2 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=self.config.DEVICE))
        tm = nn.Parameter(torch.zeros(self.config.SEQ_LEN, num_nodes, device=self.config.DEVICE))
        optimizer = optim.Adam([sm1, sm2, tm], lr=self.config.LEARNING_RATE_MASK)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train()

        mask_losses = []
        for epoch in range(self.config.NUM_EPOCHS_MASK):
            epoch_loss = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.config.DEVICE), y_b.to(self.config.DEVICE)
                optimizer.zero_grad()
                out = self.model(x_b, adj_matrix, sm1, sm2, tm)
                loss = self.criterion(out, y_b.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader) if train_loader else 0
            mask_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Mask Ep {epoch + 1:3d} | Loss: {avg_loss:.5f}")

        print("Mask training complete.")
        return sm1, sm2, tm, mask_losses

    def evaluate_model(self, test_loader: DataLoader, adj_matrix: np.ndarray, sm1=None, sm2=None, tm=None,
                       load_best=True):
        """Evaluates the model on the test set, optionally applying masks."""
        if load_best:
            model_path = Path(self.config.OUTPUT_DIR) / 'checkpoints/best_gnn_model.pth'
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
            else:
                print(f"Warning: Best model not found at {model_path}. Using current model state.")

        self.model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(self.config.DEVICE), y_b.to(self.config.DEVICE)
                out = self.model(x_b, adj_matrix, sm1, sm2, tm)
                preds.append(out.cpu().numpy())
                actuals.append(y_b.cpu().numpy())

        if not preds:
            return {}, np.array([]), np.array([])

        preds_np = np.concatenate(preds, axis=0)
        actuals_np = np.concatenate(actuals, axis=0).squeeze(-1)

        num_nodes = preds_np.shape[-1]
        preds_original = self.scaler.inverse_transform(preds_np.reshape(-1, num_nodes)).reshape(preds_np.shape)
        actuals_original = self.scaler.inverse_transform(actuals_np.reshape(-1, num_nodes)).reshape(actuals_np.shape)

        metrics = MetricCalculator.calculate_all_metrics(actuals_original, preds_original)
        return metrics, actuals_original, preds_original