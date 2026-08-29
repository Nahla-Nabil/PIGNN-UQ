"""Shared graph-loading / model / training utilities for the baseline and
physics-informed GNN variants, so both use an identical data split, feature
standardization, architecture, and training procedure and differ ONLY in
the loss function -- required for the baseline-vs-physics-informed
comparison (paper's central empirical claim) to be a fair, single-variable
comparison.

Fixed seed = 42 everywhere in this project (declared once, here).
"""
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

SEED = 42
DATA_PROCESSED = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURE_COLS = ["ndvi", "b04", "b08", "b11", "dist_to_tide_km"]
TARGET_COL = "soc_tha_mean"


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graph():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")
    return df, edges


def make_pyg_data(df: pd.DataFrame, edges: np.ndarray):
    """Standardize features/target using TRAIN rows only (no test-set
    leakage into the scaler), return a PyG Data object plus the scaler
    stats (needed to invert predictions back to t C/ha for reporting)."""
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    feat_mean = df.loc[train_mask, FEATURE_COLS].mean()
    feat_std = df.loc[train_mask, FEATURE_COLS].std().replace(0, 1.0)
    X = ((df[FEATURE_COLS] - feat_mean) / feat_std).values.astype(np.float32)

    y_mean = df.loc[train_mask, TARGET_COL].mean()
    y_std = df.loc[train_mask, TARGET_COL].std()
    y = ((df[TARGET_COL] - y_mean) / y_std).values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32)

    data = Data(
        x=torch.tensor(X),
        y=torch.tensor(y).unsqueeze(1),
        edge_index=torch.tensor(edges, dtype=torch.long),
    )
    data.y_raw = torch.tensor(y_raw).unsqueeze(1)
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    scaler = {"y_mean": float(y_mean), "y_std": float(y_std),
              "feat_mean": feat_mean.to_dict(), "feat_std": feat_std.to_dict()}
    return data, scaler


class SAGERegressor(nn.Module):
    """2-layer GraphSAGE regressor. Fixed architecture for every variant
    trained in this project (baseline, physics-informed) -- only the loss
    function differs between variants, per PLAN.md's single-architecture
    scope cut."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.dropout(h, p=self.dropout, train=self.training)
        h = torch.relu(self.conv2(h, edge_index))
        return self.head(h)


def unscale(y_scaled: torch.Tensor, scaler: dict) -> np.ndarray:
    return (y_scaled.detach().cpu().numpy().flatten() * scaler["y_std"]) + scaler["y_mean"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true)), "mean_true": float(y_true.mean())}
