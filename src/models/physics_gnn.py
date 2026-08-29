"""
Physics-informed GNN: IDENTICAL architecture, split, seed, and early-
stopping rule as src/models/baseline_gnn.py (SAGERegressor, same
hyperparameters) -- the ONLY difference is the addition of the physics
loss term from src/models/physics_terms.py. This is deliberate: per
PLAN.md's scope cut, the baseline-vs-physics-informed comparison must be a
single-variable comparison (loss function only) for Table 2 to mean
anything.

Model selection (early stopping) still uses ONLY the val DATA loss (not
the physics loss) -- so the stopping RULE is identical to the baseline's,
isolating the physics term's effect to training dynamics, not to a
different selection criterion.

Writes results/physics_gnn_metrics.csv and
results/physics_gnn_predictions.csv.
"""
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import (  # noqa: E402
    SEED, FEATURE_COLS, load_graph, make_pyg_data, SAGERegressor, set_seed,
    unscale, regression_metrics,
)
from src.models.physics_terms import compute_mass_balance_proxy, physics_loss  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"
MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-3
W_PRIOR = 0.3
W_NONNEG = 0.1


def train():
    set_seed(SEED)
    df, edges = load_graph()
    data, scaler = make_pyg_data(df, edges)

    train_mask_np = (df["split"] == "train").values
    proxy_raw, proxy_stats = compute_mass_balance_proxy(df, train_mask_np)
    proxy_z = torch.tensor(
        ((proxy_raw - proxy_stats["proxy_mean"]) / proxy_stats["proxy_std"]).astype(np.float32)
    ).unsqueeze(1)
    y_std = scaler["y_std"]

    model = SAGERegressor(in_dim=len(FEATURE_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    data_loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        out_raw = out * y_std + scaler["y_mean"]  # differentiable unscale (keeps autograd graph intact)

        data_term = data_loss_fn(out[data.train_mask], data.y[data.train_mask])
        phys_term, prior_val, nonneg_val = physics_loss(
            out[data.train_mask], out_raw[data.train_mask], proxy_z[data.train_mask],
            y_std, W_PRIOR, W_NONNEG,
        )
        loss = data_term + phys_term
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_loss = data_loss_fn(out_eval[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 20 == 0 or epoch == MAX_EPOCHS:
            print(f"epoch {epoch:4d}  data_mse(std)={data_term.item():.4f}  "
                  f"phys_prior={prior_val:.4f}  phys_nonneg={nonneg_val:.6f}  "
                  f"val_mse(std)={val_loss:.4f}  best_val={best_val:.4f}@{best_epoch}")

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
    metrics_df = pd.DataFrame([
        {"model": "physics_gnn", "split": "train", "seed": SEED, "best_epoch": best_epoch,
         "w_prior": W_PRIOR, "w_nonneg": W_NONNEG, **metrics_train},
        {"model": "physics_gnn", "split": "val", "seed": SEED, "best_epoch": best_epoch,
         "w_prior": W_PRIOR, "w_nonneg": W_NONNEG, **metrics_val},
        {"model": "physics_gnn", "split": "test_spatial_block", "seed": SEED, "best_epoch": best_epoch,
         "w_prior": W_PRIOR, "w_nonneg": W_NONNEG, **metrics_test},
    ])
    metrics_df.to_csv(RESULTS_DIR / "physics_gnn_metrics.csv", index=False)

    pred_df = df.copy()
    pred_df["y_pred_soc_tha"] = preds_raw
    pred_df["residual"] = pred_df["y_pred_soc_tha"] - pred_df["soc_tha_mean"]
    pred_df["mass_balance_proxy_raw"] = proxy_raw
    pred_df.to_csv(RESULTS_DIR / "physics_gnn_predictions.csv", index=False)

    torch.save(model.state_dict(), RESULTS_DIR / "physics_gnn_weights.pt")
    pd.Series(proxy_stats).to_csv(RESULTS_DIR / "physics_proxy_scaler_stats.csv")

    print()
    print("=== FINAL (raw t C/ha units), test evaluated once, after model selection ===")
    print("train:", metrics_train)
    print("val:", metrics_val)
    print("test (spatial block):", metrics_test)
    print(f"wrote -> {RESULTS_DIR / 'physics_gnn_metrics.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'physics_gnn_predictions.csv'}")
    return metrics_train, metrics_val, metrics_test


if __name__ == "__main__":
    train()
