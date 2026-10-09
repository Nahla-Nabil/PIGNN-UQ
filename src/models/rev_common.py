"""
Shared code for the revision experiments (paper/revision_preregistration.md).

- load_rev(): graph_nodes.csv (frozen nodes/splits/labels) joined with the
  corrected + extended features in graph_nodes_rev_features.csv (Deviation
  D1). The base columns (b03/b04/b08/b11/ndvi/ndwi/ndmi) are REPLACED by
  their corrected versions.
- Leakage-safe graph regime (§1): every fit/predict uses the subgraph
  induced by one node set, so no edge crosses a split or CV-fold boundary.
- fit_predict(): one entry point for every model family, so CV (§4) and
  the final test evaluation call identical code.
"""
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import SAGERegressor, TARGET_COL, DATA_PROCESSED  # noqa: E402
from src.models.physics_terms import physics_loss  # noqa: E402

F0 = ["ndvi", "b04", "b08", "b11", "dist_to_tide_km"]
F1 = F0 + ["b02", "b03", "b05", "b06", "b07", "b8a", "b12", "ndre", "mndwi", "ndmi"]
F2 = F1 + ["elevation", "elevation_std"]
F3 = F2 + ["t1_ndvi", "t1_ndre", "t1_ndmi", "t1_mndwi", "d_ndvi", "d_ndmi"]
FEATURE_SETS = {"F0_base": F0, "F1_spectral": F1, "F2_spec_dem": F2, "F3_spec_dem_temporal": F3}
FS_ORDER = list(FEATURE_SETS)  # simplicity order for the 1-SE rule

# Deviation D2: patience 30 -> 100, min 50 epochs, max 500 (both GNN variants);
# the inner-holdout curve is noisy for the first ~40 epochs, so patience 30
# stopped runs at an essentially random early epoch.
GNN_MAX_EPOCHS, GNN_PATIENCE, GNN_MIN_EPOCHS, GNN_LR, GNN_WD, GNN_DROPOUT = 500, 100, 50, 0.01, 1e-3, 0.3
W_NONNEG = 0.1


def load_rev():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    rev = pd.read_csv(DATA_PROCESSED / "graph_nodes_rev_features.csv")
    base_cols = ["b03", "b04", "b08", "b11", "ndvi", "ndwi", "ndmi"]
    df = df.drop(columns=base_cols).merge(rev, on=["row", "col"], how="left", validate="1:1")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")
    return df, edges


def induced_edges(edges: np.ndarray, node_idx: np.ndarray, n_total: int):
    """Edges with both endpoints in node_idx, re-indexed to 0..len(node_idx)-1."""
    pos = np.full(n_total, -1, dtype=np.int64)
    pos[node_idx] = np.arange(len(node_idx))
    s, d = pos[edges[0]], pos[edges[1]]
    keep = (s >= 0) & (d >= 0)
    return np.vstack([s[keep], d[keep]])


def mass_balance_proxy(df: pd.DataFrame, fit_idx: np.ndarray) -> np.ndarray:
    """Same steady-state proxy as physics_terms.compute_mass_balance_proxy
    (equal-weight min-max NDVI + tide distance), but scaled on the fit nodes
    and using the corrected NDVI."""
    out = np.zeros(len(df))
    for col in ("ndvi", "dist_to_tide_km"):
        v = df[col].values
        lo, hi = np.nanmin(v[fit_idx]), np.nanmax(v[fit_idx])
        out += 0.5 * (v - lo) / (hi - lo + 1e-9)
    return out


def _design(df, cols, fit_idx, pred_df=None):
    """Standardized design matrix with fit-set median imputation (only the
    cloud-masked t1 columns have NaNs) and fit-set mean/SD. If pred_df is
    given, the returned matrix is pred_df's inputs transformed with df's
    fit-set statistics (used by E8 to swap in second-date imagery)."""
    X = df[cols].values.astype(np.float64)
    med = np.nanmedian(X[fit_idx], axis=0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    if pred_df is None:
        return X, Xz
    Xp = pred_df[cols].values.astype(np.float64)
    return X, Xz, (np.where(np.isnan(Xp), med, Xp) - mu) / sd


def spatial_lag(Xz: np.ndarray, edges: np.ndarray, groups: list) -> np.ndarray:
    """1-hop neighbor mean of each column, computed within each node group's
    induced subgraph (no cross-group information)."""
    lag = np.zeros_like(Xz)
    n = len(Xz)
    for g in groups:
        e = induced_edges(edges, g, n)
        acc = np.zeros((len(g), Xz.shape[1]))
        cnt = np.zeros(len(g))
        np.add.at(acc, e[1], Xz[g][e[0]])
        np.add.at(cnt, e[1], 1)
        own = Xz[g]
        lag[g] = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], own)
    return lag


def _train_gnn(Xz, y, edges, fit_idx, stop_idx, pred_groups, seed, hidden, w_prior, proxy, Xz_pred=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(Xz)
    y_mu, y_sd = y[fit_idx].mean(), y[fit_idx].std()
    X_t = torch.tensor(Xz, dtype=torch.float32)

    def sub(idx):
        return X_t[idx], torch.tensor(induced_edges(edges, idx, n), dtype=torch.long)

    x_fit, e_fit = sub(fit_idx)
    x_stop, e_stop = sub(stop_idx)
    y_fit = torch.tensor((y[fit_idx] - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)
    y_stop = torch.tensor((y[stop_idx] - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)
    p_mu, p_sd = proxy[fit_idx].mean(), proxy[fit_idx].std() + 1e-9
    proxy_fit = torch.tensor((proxy[fit_idx] - p_mu) / p_sd, dtype=torch.float32).unsqueeze(1)

    model = SAGERegressor(Xz.shape[1], hidden_dim=hidden, dropout=GNN_DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
    mse = torch.nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(GNN_MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(x_fit, e_fit)
        phys, _, _ = physics_loss(out, out * y_sd + y_mu, proxy_fit, y_sd, w_prior, W_NONNEG if w_prior > 0 else 0.0)
        loss = mse(out, y_fit) + phys
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(x_stop, e_stop), y_stop).item()
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= GNN_PATIENCE and epoch >= GNN_MIN_EPOCHS:
                break
    model.load_state_dict(best_state)
    model.eval()
    pred = np.full(n, np.nan)
    if Xz_pred is not None:
        X_t = torch.tensor(Xz_pred, dtype=torch.float32)
    with torch.no_grad():
        for g in pred_groups:
            xg, eg = sub(g)
            pred[g] = model(xg, eg).numpy().ravel() * y_sd + y_mu
    return pred


def fit_predict(df, edges, family, cols, fit_idx, pred_groups, seed, params=None, stop_idx=None, pred_df=None):
    """Fit `family` on fit_idx; return predictions for every node in
    pred_groups (a list of index arrays, each predicted on its own induced
    subgraph). stop_idx: early-stopping nodes (GNN only), disjoint from
    fit_idx."""
    params = params or {}
    y = df[TARGET_COL].values.astype(np.float64)
    if pred_df is None:
        X, Xz = _design(df, cols, fit_idx)
        Xp = Xz
    else:
        X, Xz, Xp = _design(df, cols, fit_idx, pred_df)
    pred_idx = np.concatenate(pred_groups)
    pred = np.full(len(df), np.nan)

    if family == "linear":
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(Xz[fit_idx], y[fit_idx])
        pred[pred_idx] = m.predict(Xp[pred_idx])
    elif family in ("rf", "rf_lag", "rf_coords"):
        from sklearn.ensemble import RandomForestRegressor
        if family == "rf_lag":
            groups = [fit_idx] + list(pred_groups)
            if pred_df is not None:
                raise NotImplementedError("pred_df is not supported for rf_lag")
            Xz = np.hstack([Xz, spatial_lag(Xz, edges, groups)])
            Xp = Xz
        m = RandomForestRegressor(n_estimators=500, max_features=params.get("max_features", 1.0),
                                  min_samples_leaf=params.get("min_samples_leaf", 5),
                                  n_jobs=-1, random_state=seed).fit(Xz[fit_idx], y[fit_idx])
        pred[pred_idx] = m.predict(Xp[pred_idx])
    elif family == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=params.get("max_depth", 3),
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                             random_state=seed, n_jobs=-1).fit(Xz[fit_idx], y[fit_idx])
        pred[pred_idx] = m.predict(Xp[pred_idx])
    elif family in ("gnn", "phys_gnn"):
        w_prior = params.get("lambda", 0.0) if family == "phys_gnn" else 0.0
        proxy = mass_balance_proxy(df, fit_idx)
        pred = _train_gnn(Xz, y, edges, fit_idx, stop_idx, pred_groups, seed,
                          params.get("hidden", 16), w_prior, proxy, Xz_pred=None if pred_df is None else Xp)
    else:
        raise ValueError(family)
    return pred


def metrics(y, p, y_ref_mean):
    """R2 (vs. the evaluated set's own mean), MAE, RMSE, Spearman rho, and
    skill vs. the fit-set-mean predictor (1 - MSE/MSE_const)."""
    from scipy.stats import spearmanr
    err = p - y
    mse = np.mean(err ** 2)
    return {"r2": 1 - mse / np.var(y), "mae": np.mean(np.abs(err)), "rmse": np.sqrt(mse),
            "spearman": spearmanr(y, p).statistic,
            "skill_vs_mean": 1 - mse / np.mean((y - y_ref_mean) ** 2)}
