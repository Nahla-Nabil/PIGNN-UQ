"""
Unified, parameterized training functions for Phase 1's pre-registered
experiments (multi-seed stability, lambda sweep, feature ablation) --
factored out of baseline_gnn.py / physics_gnn.py / linear_baseline.py so
every Phase 1 driver script (scripts/run_phase1*.py) calls the SAME
training code Day 1-2's individual scripts used, just parameterized over
seed / w_prior / feature_cols instead of hardcoded. With seed=42,
w_prior=0.3, w_nonneg=0.1, feature_cols=FEATURE_COLS this reproduces Day
1-2's training procedure exactly (the only Day-1-2 numbers that changed are
the ones affected by the Phase 0 split fix, which changed which nodes are
train/val/test, not how a model is trained on them).

Day 1-2's baseline_gnn.py / physics_gnn.py / linear_baseline.py are left in
place, unmodified, as the historical record of exactly what Day 1-2 ran --
they are not imported by anything below.
"""
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import (  # noqa: E402
    FEATURE_COLS, TARGET_COL, make_pyg_data, SAGERegressor, set_seed, unscale, regression_metrics,
)
from src.models.physics_terms import compute_mass_balance_proxy, physics_loss  # noqa: E402
from src.eval.plausibility import negative_rate, edge_incoherence_rate, physical_consistency_corr  # noqa: E402

MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-3


def train_gnn(df: pd.DataFrame, edges: np.ndarray, seed: int, w_prior: float = 0.0, w_nonneg: float = 0.0,
              feature_cols=None, verbose: bool = False) -> dict:
    """Unified GNN trainer. w_prior=w_nonneg=0 reproduces the "baseline GNN
    (no physics)" variant (physics_loss's two terms are still computed, for
    logging, but contribute exactly 0 to the gradient since they're
    multiplied by 0) -- this keeps baseline-vs-physics a true
    single-variable (loss-weight) comparison without maintaining two copies
    of the training loop, and lets the lambda sweep (Phase 1B) just vary
    w_prior through this one function.
    """
    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    set_seed(seed)
    data, scaler = make_pyg_data(df, edges, feature_cols=feature_cols)

    train_mask_np = (df["split"] == "train").values
    proxy_raw, proxy_stats = compute_mass_balance_proxy(df, train_mask_np)
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
        out_raw = out * y_std + scaler["y_mean"]  # differentiable unscale

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


def train_linear(df: pd.DataFrame, feature_cols=None) -> dict:
    """Deterministic OLS -- no seed dependence (closed-form solution), so
    Phase 1A reports it with std=0 by construction, not simulated
    variability. See scripts/run_phase1a_multiseed.py for how this is
    surfaced in the multi-seed table."""
    from sklearn.linear_model import LinearRegression

    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    model = LinearRegression()
    model.fit(train_df[feature_cols].values, train_df[TARGET_COL].values)
    preds_raw = model.predict(df[feature_cols].values)

    split_col = df["split"].values
    metrics = {
        "train": regression_metrics(train_df[TARGET_COL].values, preds_raw[split_col == "train"]),
        "val": regression_metrics(val_df[TARGET_COL].values, preds_raw[split_col == "val"]),
        "test": regression_metrics(test_df[TARGET_COL].values, preds_raw[split_col == "test"]),
    }

    train_mask_np = (df["split"] == "train").values
    proxy_raw, _ = compute_mass_balance_proxy(df, train_mask_np)
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
