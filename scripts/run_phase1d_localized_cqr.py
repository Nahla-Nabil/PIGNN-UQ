"""
Phase 1D -- localized (Mondrian) CQR calibration, VAL-ONLY (calibrated on
val, test still untouched -- the "before/after" coverage numbers reported
here are on VAL itself via leave-one-block-out-style resubstitution is NOT
what's done; see note below on how this stays val-only despite reporting
coverage).

Uses the configuration Phases 1A-1C selected: physics-informed GNN,
lambda=0.1 (Phase 1B), w_nonneg=0.1, base feature set (Phase 1C -- neither
NDWI nor NDMI beat base within the pre-registered tolerance). The quantile
regressor itself is trained with pinball loss on the DATA term only, no
physics loss -- the same disclosed scope choice as Day 2's cqr_uq.py
(combining physics-informed training with a quantile head is future work,
not something Phase 1D asks to change; Phase 1D is about the CALIBRATION
method, not the point/quantile model architecture).

*** Important scope note on what "val-only" means here ***
CQR calibration structurally needs a calibration set held out from the
data the quantile heads were trained on. That set is val -- exactly the
same role val has played throughout this project. To compare global vs.
localized calibration WITHOUT touching test, this script does the
comparison via a second split carved out of the ALREADY-DESIGNATED val
nodes: val is randomly split 50/50 (seed fixed, disclosed below) into
val_calib (fit the conformal correction) and val_check (measure empirical
coverage of that correction) -- structurally identical to a calibration/
test pair, but built entirely from nodes already inside "val", so this
never looks at test. The reported "before/after" numbers in this script are
therefore val-internal; Phase 2 repeats the exercise ONCE on the real
test split, which is the number that goes in the paper's main table.

Also implements a density-weighted conformal variant for comparison
(time-permitting extension named in the Phase 1D instruction): each
val_check node's correction is a WEIGHTED quantile of val_calib conformity
scores, weighted by a Gaussian kernel on great-circle-approximate distance
(bandwidth = median nearest-neighbor distance among val_calib nodes,
disclosed, not tuned).

Writes results/phase1d_cqr_calibration_comparison.csv (global vs. localized
vs. density-weighted, both nominal levels, val_check coverage) and
results/phase1d_quantile_predictions.csv (raw quantile predictions for
every node, reused by Phase 2 so the quantile model is trained exactly
once).
"""
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.gnn_common import FEATURE_COLS, load_graph, make_pyg_data, set_seed  # noqa: E402
from src.models.cqr_uq import QuantileSAGE, pinball_loss, QUANTILES, ALPHAS, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEED = 42  # matches the project's declared default seed for the single
# deployed model (gnn_common.py) -- Phase 1A-C's 5-seed sweeps established
# ROBUSTNESS of the selected config; the one model actually carried forward
# to calibration/test uses the project's standard seed, consistent with how
# Day 1-2 deployed a single model.
SELECTED_FEATURE_COLS = FEATURE_COLS  # Phase 1C selected "base" -- see
# results/phase1c_feature_summary.csv. Hardcoded, not re-derived, so this
# script's behavior can't silently drift if 1C is re-run later.
VAL_INTERNAL_SPLIT_SEED = 7  # seeds the val -> {val_calib, val_check} split
MIN_LOCAL_CAL_N = 30  # Mondrian fallback threshold: a spatial block needs at
# least this many calibration points before its LOCAL quantile is trusted;
# below this, fall back to the global correction (standard Mondrian-CP
# practice under sparse per-partition data).


def train_quantile_model():
    set_seed(SEED)
    df, edges = load_graph()
    data, scaler = make_pyg_data(df, edges, feature_cols=SELECTED_FEATURE_COLS)
    y_mean, y_std = scaler["y_mean"], scaler["y_std"]

    model = QuantileSAGE(in_dim=len(SELECTED_FEATURE_COLS), n_quantiles=len(QUANTILES))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = pinball_loss(out[data.train_mask], data.y[data.train_mask], QUANTILES)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_loss = pinball_loss(out_eval[data.val_mask], data.y[data.val_mask], QUANTILES).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, no_improve = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(data.x, data.edge_index).numpy()
    preds_raw = preds_scaled * y_std + y_mean
    print(f"quantile model: best_epoch={best_epoch}, best_val_pinball(std)={best_val:.4f}")
    return df, edges, preds_raw


def conformity_scores(lo, hi, y):
    return np.maximum(lo - y, y - hi)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted empirical quantile (type-1 style: smallest value whose
    cumulative weight fraction >= level), used for the density-weighted
    conformal variant."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    idx = np.searchsorted(cw, level)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def main():
    df, edges, preds_raw = train_quantile_model()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = {f"q{q}_raw": preds_raw[:, i] for i, q in enumerate(QUANTILES)}
    pred_df = df.copy()
    for k, v in pred_cols.items():
        pred_df[k] = v
    pred_df.to_csv(RESULTS_DIR / "phase1d_quantile_predictions.csv", index=False)

    q_idx = {q: i for i, q in enumerate(QUANTILES)}
    val_mask = (df["split"] == "val").values
    val_idx = np.where(val_mask)[0]

    rng = np.random.default_rng(VAL_INTERNAL_SPLIT_SEED)
    perm = rng.permutation(val_idx)
    half = len(perm) // 2
    calib_idx, check_idx = perm[:half], perm[half:]
    print(f"val internal split: val_calib n={len(calib_idx)}, val_check n={len(check_idx)}")

    y_true = df["soc_tha_mean"].values
    superblock = df["superblock"].values
    lon, lat = df["lon"].values, df["lat"].values

    comparison_rows = []
    for nominal, (q_lo, q_hi) in ALPHAS.items():
        alpha = 1 - nominal
        lo_raw, hi_raw = preds_raw[:, q_idx[q_lo]], preds_raw[:, q_idx[q_hi]]

        cal_lo, cal_hi, cal_y = lo_raw[calib_idx], hi_raw[calib_idx], y_true[calib_idx]
        chk_lo, chk_hi, chk_y = lo_raw[check_idx], hi_raw[check_idx], y_true[check_idx]
        cal_scores = conformity_scores(cal_lo, cal_hi, cal_y)

        # --- (1) GLOBAL calibration: one Q for every check node ---
        n_cal = len(cal_scores)
        level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        Q_global = np.quantile(cal_scores, level)
        lo_adj_g = chk_lo - Q_global
        hi_adj_g = chk_hi + Q_global
        cov_global = float(np.mean((chk_y >= lo_adj_g) & (chk_y <= hi_adj_g)))
        width_global = float(np.mean(hi_adj_g - lo_adj_g))

        # --- (2) LOCALIZED (Mondrian) calibration: per-superblock Q,
        # fallback to global Q when a block's calib set is too small or
        # absent entirely ---
        cal_sb = superblock[calib_idx]
        chk_sb = superblock[check_idx]
        block_Q = {}
        for sb in np.unique(cal_sb):
            block_scores = cal_scores[cal_sb == sb]
            if len(block_scores) < MIN_LOCAL_CAL_N:
                continue
            n_b = len(block_scores)
            level_b = min(np.ceil((n_b + 1) * (1 - alpha)) / n_b, 1.0)
            block_Q[int(sb)] = np.quantile(block_scores, level_b)
        n_fallback = sum(1 for sb in chk_sb if int(sb) not in block_Q)
        Q_local_arr = np.array([block_Q.get(int(sb), Q_global) for sb in chk_sb])
        lo_adj_l = chk_lo - Q_local_arr
        hi_adj_l = chk_hi + Q_local_arr
        cov_local = float(np.mean((chk_y >= lo_adj_l) & (chk_y <= hi_adj_l)))
        width_local = float(np.mean(hi_adj_l - lo_adj_l))

        # --- (3) DENSITY-WEIGHTED calibration: per-check-node weighted
        # quantile of ALL val_calib scores, weighted by a spatial Gaussian
        # kernel. Bandwidth = median nearest-neighbor distance among
        # val_calib nodes (data-driven, not hand-tuned). ---
        cal_lon, cal_lat = lon[calib_idx], lat[calib_idx]
        # pairwise nearest-neighbor distance among calib nodes (km, planar approx)
        from scipy.spatial import cKDTree
        cal_xy = np.column_stack([cal_lon * 111.0 * np.cos(np.radians(cal_lat.mean())), cal_lat * 111.0])
        tree = cKDTree(cal_xy)
        nn_dist, _ = tree.query(cal_xy, k=2)
        bandwidth_km = float(np.median(nn_dist[:, 1]))
        bandwidth_km = max(bandwidth_km, 0.1)

        chk_lon, chk_lat = lon[check_idx], lat[check_idx]
        chk_xy = np.column_stack([chk_lon * 111.0 * np.cos(np.radians(cal_lat.mean())), chk_lat * 111.0])
        lo_adj_w = np.empty_like(chk_lo)
        hi_adj_w = np.empty_like(chk_hi)
        for i in range(len(check_idx)):
            d2 = np.sum((cal_xy - chk_xy[i]) ** 2, axis=1)
            w = np.exp(-d2 / (2 * bandwidth_km ** 2))
            Q_w = weighted_quantile(cal_scores, w, min((1 - alpha) * (n_cal + 1) / n_cal, 1.0))
            lo_adj_w[i] = chk_lo[i] - Q_w
            hi_adj_w[i] = chk_hi[i] + Q_w
        cov_weighted = float(np.mean((chk_y >= lo_adj_w) & (chk_y <= hi_adj_w)))
        width_weighted = float(np.mean(hi_adj_w - lo_adj_w))

        for method, cov, width in [
            ("global", cov_global, width_global),
            ("localized_mondrian", cov_local, width_local),
            ("density_weighted", cov_weighted, width_weighted),
        ]:
            comparison_rows.append({
                "nominal_level": nominal, "calibration_method": method,
                "empirical_coverage_val_check": cov, "mean_interval_width_tha": width,
                "n_val_calib": len(calib_idx), "n_val_check": len(check_idx),
                "n_fallback_to_global": n_fallback if method == "localized_mondrian" else np.nan,
                "n_local_blocks_used": len(block_Q) if method == "localized_mondrian" else np.nan,
                "bandwidth_km": bandwidth_km if method == "density_weighted" else np.nan,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "phase1d_cqr_calibration_comparison.csv", index=False)
    print()
    print("=== Phase 1D: global vs. localized (Mondrian) vs. density-weighted CQR, val_check coverage ===")
    print(comparison_df.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'phase1d_cqr_calibration_comparison.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase1d_quantile_predictions.csv'}")
    return comparison_df


if __name__ == "__main__":
    main()
