"""
Journal-extension sprint (post-submission, temporal dC/dt experiment) --
extension-only counterpart to `src/models/gnn_common.py`. Re-imports every
piece of the submitted paper's architecture/scaler/metric logic UNCHANGED
(same 2-layer GraphSAGE, same standardization procedure, same regression
metrics) so the model itself is identical; the only things this module
does differently are (a) load the extension's own graph files
(`graph_nodes_ext_temporal.csv` / `graph_edges_ext_temporal.npy`, written
by `build_graph_temporal.py`, never the submitted paper's files) and (b)
read the `split_ext` column instead of `split` -- so this module can never
accidentally read or touch the submitted paper's train/val/test partition.

`gnn_common.py` itself is imported read-only and not modified.
"""
import pathlib

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.models.gnn_common import (  # noqa: F401
    SEED, FEATURE_COLS, TARGET_COL, SAGERegressor, set_seed, unscale, regression_metrics,
)

DATA_PROCESSED = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"


def load_graph_temporal():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges_ext_temporal.npy")
    return df, edges


def make_pyg_data_temporal(df: pd.DataFrame, edges: np.ndarray, feature_cols=None):
    """Identical standardization logic to gnn_common.make_pyg_data, reading
    `split_ext` (this extension's own column) instead of `split`."""
    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    train_mask = (df["split_ext"] == "train").values
    val_mask = (df["split_ext"] == "val").values
    test_mask = (df["split_ext"] == "test").values

    feat_mean = df.loc[train_mask, feature_cols].mean()
    feat_std = df.loc[train_mask, feature_cols].std().replace(0, 1.0)
    X = ((df[feature_cols] - feat_mean) / feat_std).values.astype(np.float32)

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
