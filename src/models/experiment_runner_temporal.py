"""
Journal-extension sprint (post-submission, temporal dC/dt experiment) --
extension-only counterpart to `src/models/experiment_runner.py`. Same
training loop, architecture, hyperparameters, and early-stopping rule
(copied, not altered) -- the only differences are (a) the graph loader
(`gnn_common_temporal.make_pyg_data_temporal`, reading `split_ext`) and
(b) the physics proxy (`physics_terms_temporal.compute_mass_balance_proxy_
dcdt`, the genuine two-date dC/dt proxy) in place of the submitted paper's
steady-state one. `physics_loss()` itself is imported UNCHANGED from
`physics_terms.py` -- only the proxy fed into it differs.

`experiment_runner.py` is imported nowhere here and not modified; this is
a parallel module, not a patch.
"""
import copy

import numpy as np
import pandas as pd
import torch

from src.models.gnn_common_temporal import (
    FEATURE_COLS, TARGET_COL, make_pyg_data_temporal, SAGERegressor, set_seed, unscale, regression_metrics,
)
from src.models.physics_terms import physics_loss, compute_mass_balance_proxy  # unchanged, reused directly
from src.models.physics_terms_temporal import compute_mass_balance_proxy_dcdt
from src.eval.plausibility import negative_rate, physical_consistency_corr

MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-3


def train_gnn_temporal(df: pd.DataFrame, edges: np.ndarray, seed: int, w_prior: float = 0.0, w_nonneg: float = 0.0,
                        feature_cols=None, verbose: bool = False, proxy_fn=compute_mass_balance_proxy_dcdt) -> dict:
    """Unified GNN trainer for the temporal-extension track. w_prior=
    w_nonneg=0 reproduces "baseline GNN (no physics)" on the NEW split/
    graph. `proxy_fn` defaults to the real dC/dt proxy
    (physics_terms_temporal.compute_mass_balance_proxy_dcdt); passing
    `physics_terms.compute_mass_balance_proxy` (the submitted paper's
    original steady-state proxy, imported above UNCHANGED) instead trains
    a same-split, same-seed, same-architecture CONTROL variant, so the
    dC/dt-vs-steady-state comparison in `extension_temporal_notes.md` is a
    genuine single-variable (proxy formula only) comparison, not confounded
    by the split also having changed. Identical structure to
    experiment_runner.train_gnn otherwise, only the loader/proxy source
    differ."""
    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    set_seed(seed)
    data, scaler = make_pyg_data_temporal(df, edges, feature_cols=feature_cols)

    train_mask_np = (df["split_ext"] == "train").values
    proxy_raw, proxy_stats = proxy_fn(df, train_mask_np)
    proxy_z = torch.tensor(
        ((proxy_raw - proxy_stats["proxy_mean"]) / proxy_stats["proxy_std"]).astype(np.float32)
    ).unsqueeze(1)
    y_std = scaler["y_std"]

    model = SAGERegressor(in_dim=len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    data_loss_fn = torch.nn.MSELoss()

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        out_raw = out * y_std + scaler["y_mean"]

        data_term = data_loss_fn(out[data.train_mask], data.y[data.train_mask])
        phys_term, prior_val, nonneg_val = physics_loss(
            out[data.train_mask], out_raw[data.train_mask], proxy_z[data.train_mask],
            y_std, w_prior, w_nonneg,
        )
        loss = data_term + phys_term
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_loss = data_loss_fn(out_eval[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, no_improve = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1

        if verbose and (epoch % 50 == 0 or epoch == MAX_EPOCHS):
            print(f"  epoch {epoch:4d} data_mse(std)={data_term.item():.4f} "
                  f"phys_prior={prior_val:.4f} val_mse(std)={val_loss:.4f} best={best_val:.4f}@{best_epoch}")
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(data.x, data.edge_index)
    preds_raw = unscale(preds_scaled, scaler)
    y_true_raw = data.y_raw.numpy().flatten()

    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    metrics = {
        "train": regression_metrics(y_true_raw[train_mask], preds_raw[train_mask]),
        "val": regression_metrics(y_true_raw[val_mask], preds_raw[val_mask]),
        "test": regression_metrics(y_true_raw[test_mask], preds_raw[test_mask]),
    }
    plaus_val = {
        "neg_rate": negative_rate(preds_raw[val_mask]),
        "prior_gen_corr": physical_consistency_corr(preds_raw, proxy_raw, val_mask),
    }
    plaus_test = {
        "neg_rate": negative_rate(preds_raw[test_mask]),
        "prior_gen_corr": physical_consistency_corr(preds_raw, proxy_raw, test_mask),
    }
    return {
        "metrics": metrics, "plaus_val": plaus_val, "plaus_test": plaus_test,
        "best_epoch": best_epoch, "preds_raw": preds_raw, "y_true_raw": y_true_raw,
        "proxy_raw": proxy_raw, "model_state": best_state, "scaler": scaler,
        "feature_cols": feature_cols, "seed": seed, "w_prior": w_prior, "w_nonneg": w_nonneg,
    }


def train_linear_temporal(df: pd.DataFrame, feature_cols=None) -> dict:
    """Deterministic OLS, identical to experiment_runner.train_linear, on
    the temporal-extension split/graph. Its own plausibility numbers are
    computed against the NEW dC/dt proxy (not the submitted paper's
    steady-state one) so the three-way comparison table's prior-gen-corr
    column is internally consistent within this extension."""
    from sklearn.linear_model import LinearRegression

    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    train_df = df[df["split_ext"] == "train"]
    val_df = df[df["split_ext"] == "val"]
    test_df = df[df["split_ext"] == "test"]

    model = LinearRegression()
    model.fit(train_df[feature_cols].values, train_df[TARGET_COL].values)
    preds_raw = model.predict(df[feature_cols].values)

    split_col = df["split_ext"].values
    metrics = {
        "train": regression_metrics(train_df[TARGET_COL].values, preds_raw[split_col == "train"]),
        "val": regression_metrics(val_df[TARGET_COL].values, preds_raw[split_col == "val"]),
        "test": regression_metrics(test_df[TARGET_COL].values, preds_raw[split_col == "test"]),
    }

    train_mask_np = (df["split_ext"] == "train").values
    proxy_raw, _ = compute_mass_balance_proxy_dcdt(df, train_mask_np)
    val_mask = split_col == "val"
    test_mask = split_col == "test"
    plaus_val = {"neg_rate": negative_rate(preds_raw[val_mask]),
                 "prior_gen_corr": physical_consistency_corr(preds_raw, proxy_raw, val_mask)}
    plaus_test = {"neg_rate": negative_rate(preds_raw[test_mask]),
                  "prior_gen_corr": physical_consistency_corr(preds_raw, proxy_raw, test_mask)}

    return {
        "metrics": metrics, "plaus_val": plaus_val, "plaus_test": plaus_test,
        "preds_raw": preds_raw, "proxy_raw": proxy_raw, "feature_cols": feature_cols, "model": model,
    }
