"""
Revision UQ methods (paper/revision_preregistration.md §5).

Every method returns, for each node in `pred_idx`, a pair (lo, hi) for each
nominal level BEFORE conformal adjustment, plus a scoring rule:

- "cqr" methods (CQR-GNN, CQR-PhysGNN, CQR-XGB): raw quantile bounds; score
  E = max(lo - y, y - hi); adjusted interval [lo - Q, hi + Q].
- "ncp" (NCP-RF): point yhat and scale s; score |y - yhat| / s; adjusted
  interval yhat ± Q·s.
- "trivial": constant fit-set mean, score |y - mean|; interval mean ± Q.

Quantile GNNs are trained on the fit nodes' induced subgraph and
early-stopped on an inner holdout of fit-set superblock groups (15%,
seeded), so the calibration nodes are never used for training or stopping.
(Clarification C1: the submitted paper early-stopped on val and then
calibrated on the same val; this removes that overlap.)
"""
import copy
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.rev_common import (_design, induced_edges, mass_balance_proxy, GNN_MAX_EPOCHS,  # noqa: E402
                                   GNN_PATIENCE, GNN_MIN_EPOCHS, GNN_LR, GNN_WD, GNN_DROPOUT, W_NONNEG)
from src.models.gnn_common import TARGET_COL  # noqa: E402
from src.models.physics_terms import physics_loss  # noqa: E402

QUANTILES = [0.025, 0.05, 0.5, 0.95, 0.975]
LEVELS = {0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}
INNER_FRAC = 0.15


class QuantileSAGE(nn.Module):
    def __init__(self, in_dim, hidden, n_q):
        super().__init__()
        self.conv1, self.conv2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, n_q)

    def forward(self, x, e):
        h = torch.relu(self.conv1(x, e))
        h = torch.dropout(h, p=GNN_DROPOUT, train=self.training)
        return self.head(torch.relu(self.conv2(h, e)))


def pinball(pred, y, qs):
    q = torch.tensor(qs, dtype=pred.dtype)[None, :]
    err = y.expand(-1, pred.shape[1]) - pred
    return torch.max(q * err, (q - 1) * err).mean()


def inner_split(df, fit_idx, seed):
    rng = np.random.default_rng(seed)
    g = df["phase1_superblock"].values[fit_idx]
    ug = np.unique(g)
    sg = rng.choice(ug, size=max(1, int(round(INNER_FRAC * len(ug)))), replace=False)
    return fit_idx[~np.isin(g, sg)], fit_idx[np.isin(g, sg)]


def quantile_gnn(df, edges, cols, fit_idx, pred_groups, seed, hidden, lam=0.0, proxy_kind="P0"):
    """Returns array (N, len(QUANTILES)) of raw-unit quantile predictions
    (NaN outside pred_groups). lam > 0 adds the physics prior on the 0.5
    head (CQR-PhysGNN)."""
    torch.manual_seed(seed)
    n = len(df)
    y = df[TARGET_COL].values.astype(np.float64)
    fit, stop = inner_split(df, fit_idx, seed)
    _, Xz = _design(df, cols, fit_idx)
    X_t = torch.tensor(Xz, dtype=torch.float32)
    mu, sd = y[fit].mean(), y[fit].std()
    proxy = mass_balance_proxy(df, fit, proxy_kind)
    pz = torch.tensor((proxy[fit] - proxy[fit].mean()) / (proxy[fit].std() + 1e-9), dtype=torch.float32)[:, None]

    def sub(idx):
        return X_t[idx], torch.tensor(induced_edges(edges, idx, n), dtype=torch.long)

    xf, ef = sub(fit)
    xs, es = sub(stop)
    yf = torch.tensor((y[fit] - mu) / sd, dtype=torch.float32)[:, None]
    ys = torch.tensor((y[stop] - mu) / sd, dtype=torch.float32)[:, None]
    med = QUANTILES.index(0.5)
    model = QuantileSAGE(Xz.shape[1], hidden, len(QUANTILES))
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(GNN_MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(xf, ef)
        loss = pinball(out, yf, QUANTILES)
        if lam > 0:
            m = out[:, med:med + 1]
            loss = loss + physics_loss(m, m * sd + mu, pz, sd, lam, W_NONNEG)[0]
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = pinball(model(xs, es), ys, QUANTILES).item()
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= GNN_PATIENCE and epoch >= GNN_MIN_EPOCHS:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = np.full((n, len(QUANTILES)), np.nan)
    with torch.no_grad():
        for g in pred_groups:
            xg, eg = sub(g)
            out[g] = model(xg, eg).numpy() * sd + mu
    return np.sort(out, axis=1)  # enforce non-crossing quantiles


def quantile_xgb(df, cols, fit_idx, pred_idx, seed, max_depth):
    import xgboost as xgb
    y = df[TARGET_COL].values
    _, Xz = _design(df, cols, fit_idx)
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES),
                         n_estimators=600, learning_rate=0.03, max_depth=max_depth, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, random_state=seed, n_jobs=-1)
    m.fit(Xz[fit_idx], y[fit_idx])
    out = np.full((len(df), len(QUANTILES)), np.nan)
    out[pred_idx] = np.sort(m.predict(Xz[pred_idx]), axis=1)
    return out


def ncp_rf(df, cols, fit_idx, pred_idx, seed, max_features, min_samples_leaf):
    """Normalized conformal: RF point model; scale = RF regressor fit on the
    point model's out-of-bag absolute residuals (no extra holdout needed)."""
    from sklearn.ensemble import RandomForestRegressor
    y = df[TARGET_COL].values
    _, Xz = _design(df, cols, fit_idx)
    rf = RandomForestRegressor(n_estimators=500, max_features=max_features, min_samples_leaf=min_samples_leaf,
                               oob_score=True, n_jobs=-1, random_state=seed).fit(Xz[fit_idx], y[fit_idx])
    abs_res = np.abs(y[fit_idx] - rf.oob_prediction_)
    sc = RandomForestRegressor(n_estimators=500, max_features=max_features, min_samples_leaf=20,
                               n_jobs=-1, random_state=seed).fit(Xz[fit_idx], abs_res)
    yhat, s = np.full(len(df), np.nan), np.full(len(df), np.nan)
    yhat[pred_idx] = rf.predict(Xz[pred_idx])
    s[pred_idx] = np.maximum(sc.predict(Xz[pred_idx]), 1e-6)
    return yhat, s


Z = {0.025: -1.959964, 0.05: -1.644854, 0.5: 0.0, 0.95: 1.644854, 0.975: 1.959964}


def ensemble_quantiles(q_gnn, q_xgb, ncp):
    """Addendum A2 Ens-Q: element-wise mean of CQR-GNN and CQR-XGB raw
    quantiles and NCP-RF's yhat + z_q * s (normal quantiles); conformalized
    afterwards like any CQR base."""
    yhat, s = ncp
    q_rf = np.column_stack([yhat + Z[q] * s for q in QUANTILES])
    return np.sort((q_gnn + q_xgb + q_rf) / 3.0, axis=1)


# ---------- conformal calibration ----------

def _q(scores, alpha):
    n = len(scores)
    return np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0))


def scores_and_interval(kind, base, y, idx, level):
    """kind: 'cqr' (base = quantile array), 'ncp' (base = (yhat, s)),
    'trivial' (base = constant mean). Returns (scores on idx, fn(Q)->(lo,hi) on idx)."""
    if kind == "cqr":
        qi = {q: i for i, q in enumerate(QUANTILES)}
        lo, hi = base[idx, qi[LEVELS[level][0]]], base[idx, qi[LEVELS[level][1]]]
        return np.maximum(lo - y[idx], y[idx] - hi), (lambda Q: (lo - Q, hi + Q))
    if kind == "ncp":
        yhat, s = base[0][idx], base[1][idx]
        return np.abs(y[idx] - yhat) / s, (lambda Q: (yhat - Q * s, yhat + Q * s))
    m = np.full(len(idx), base)
    return np.abs(y[idx] - m), (lambda Q: (m - Q, m + Q))


def calibrate(kind, base, y, cal_idx, eval_idx, level, superblock, mondrian=False, min_n=30):
    alpha = 1 - level
    cal_s, _ = scores_and_interval(kind, base, y, cal_idx, level)
    _, make = scores_and_interval(kind, base, y, eval_idx, level)
    Qg = _q(cal_s, alpha)
    if not mondrian:
        return make(Qg)
    sb_cal, sb_eval = superblock[cal_idx], superblock[eval_idx]
    Qb = {b: _q(cal_s[sb_cal == b], alpha) for b in np.unique(sb_cal) if (sb_cal == b).sum() >= min_n}
    Q = np.array([Qb.get(b, Qg) for b in sb_eval])
    return make(Q)


def interval_metrics(y, lo, hi, level, n_boot=1000, seed=0):
    from scipy.stats import spearmanr
    alpha = 1 - level
    width = np.maximum(hi - lo, 0)
    cover = (y >= lo) & (y <= hi)
    wink = width + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    abs_err = np.abs(y - (lo + hi) / 2)
    rho = spearmanr(width, abs_err).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        boots.append(spearmanr(width[b], abs_err[b]).statistic)
    terc = np.quantile(width, [1 / 3, 2 / 3])
    bins = np.digitize(width, terc)
    out = {"coverage": cover.mean(), "mean_width": width.mean(), "median_width": np.median(width),
           "winkler": wink.mean(), "rho_width_abserr": rho,
           "rho_ci_lo": np.nanpercentile(boots, 2.5), "rho_ci_hi": np.nanpercentile(boots, 97.5)}
    for t in range(3):
        out[f"coverage_tercile{t + 1}"] = cover[bins == t].mean() if np.any(bins == t) else np.nan
    return out
