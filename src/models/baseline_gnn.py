"""
Baseline GNN: predicts per-node soil organic carbon (SOC, t C/ha, 0-100cm)
from Sentinel-2-derived features on the pilot-region graph. NO physics term
-- this is the Day-1 comparison point for the Day-2 physics-informed
variant (see PLAN.md experiment-to-claim mapping, Table 2).

Model selection (early stopping) uses ONLY the val split, carved from
train's superblocks. The test split is touched exactly once, after
training is complete, for the single reported metric -- test performance
is never used to pick epochs/hyperparameters (an earlier version of this
script did watch test loss during training, which is a methodological
error -- see paper/NUMBER_SOURCE_LOG.md Day 1 entry for what changed and
why).

Writes results/baseline_gnn_metrics.csv and
results/baseline_gnn_predictions.csv.
"""
import copy
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import (  # noqa: E402
    SEED, FEATURE_COLS, load_graph, make_pyg_data, SAGERegressor, set_seed,
    unscale, regression_metrics,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"
MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-3


def train():
    set_seed(SEED)
    df, edges = load_graph()
    data, scaler = make_pyg_data(df, edges)

    model = SAGERegressor(in_dim=len(FEATURE_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_loss = loss_fn(out_eval[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 20 == 0 or epoch == MAX_EPOCHS:
            train_loss = loss_fn(out_eval[data.train_mask], data.y[data.train_mask]).item()
            print(f"epoch {epoch:4d}  train_mse(std)={train_loss:.4f}  val_mse(std)={val_loss:.4f}  best_val={best_val:.4f}@{best_epoch}")

        if epochs_since_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"restored best checkpoint from epoch {best_epoch} (val_mse(std)={best_val:.4f})")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(data.x, data.edge_index)
    preds_raw = unscale(preds_scaled, scaler)
    y_true_raw = data.y_raw.numpy().flatten()

    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    metrics_train = regression_metrics(y_true_raw[train_mask], preds_raw[train_mask])
    metrics_val = regression_metrics(y_true_raw[val_mask], preds_raw[val_mask])
    metrics_test = regression_metrics(y_true_raw[test_mask], preds_raw[test_mask])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    metrics_df = pd.DataFrame([
        {"model": "baseline_gnn_no_physics", "split": "train", "seed": SEED, "best_epoch": best_epoch, **metrics_train},
        {"model": "baseline_gnn_no_physics", "split": "val", "seed": SEED, "best_epoch": best_epoch, **metrics_val},
        {"model": "baseline_gnn_no_physics", "split": "test_spatial_block", "seed": SEED, "best_epoch": best_epoch, **metrics_test},
    ])
    metrics_df.to_csv(RESULTS_DIR / "baseline_gnn_metrics.csv", index=False)

    pred_df = df.copy()
    pred_df["y_pred_soc_tha"] = preds_raw
    pred_df["residual"] = pred_df["y_pred_soc_tha"] - pred_df["soc_tha_mean"]
    pred_df.to_csv(RESULTS_DIR / "baseline_gnn_predictions.csv", index=False)

    torch.save(model.state_dict(), RESULTS_DIR / "baseline_gnn_weights.pt")

    print()
    print("=== FINAL (raw t C/ha units), test evaluated once, after model selection ===")
    print("train:", metrics_train)
    print("val:", metrics_val)
    print("test (spatial block, held out entirely during training/selection):", metrics_test)
    print(f"wrote -> {RESULTS_DIR / 'baseline_gnn_metrics.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'baseline_gnn_predictions.csv'}")
    return metrics_train, metrics_val, metrics_test


if __name__ == "__main__":
    train()
