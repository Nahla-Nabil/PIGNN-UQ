"""
Phase 2 -- THE single test-set evaluation.

Everything above this script (Phase 0 split fix, Phase 1A-D) was decided on
VAL data only. This script trains the ONE configuration Phase 1's
mechanical rules selected:
    - lambda (w_prior) = 0.1              (Phase 1B, results/phase1b_lambda_summary.csv)
    - w_nonneg = 0.1                       (held fixed, Day 2 value, never swept)
    - feature set = base (FEATURE_COLS)    (Phase 1C, results/phase1c_feature_summary.csv --
                                             neither NDWI nor NDMI beat base within tolerance)
    - CQR method = localized (Mondrian)    (Phase 1D, results/phase1d_cqr_calibration_comparison.csv)
    - seed = 42                            (project's declared standard seed for the one
                                             deployed model; Phase 1A already established
                                             cross-seed robustness/instability context)

...and evaluates it EXACTLY ONCE on the test split, which no script before
this one has read predictions or metrics from. Also retrains linear
regression and the no-physics baseline GNN (both already fully determined,
no selection needed) on the same split, for the three-way comparison table.

This IS the number that goes in the paper's main results table.

Writes:
  results/phase2_test_comparison.csv       (three-way model comparison, TEST)
  results/phase2_test_predictions_physics_gnn.csv
  results/phase2_cqr_test_coverage.csv     (global vs. localized Mondrian, TEST)
  results/phase2_cqr_test_predictions.csv  (per-node adjusted intervals, both methods)
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.gnn_common import FEATURE_COLS  # noqa: E402
from src.models.experiment_runner import train_gnn, train_linear  # noqa: E402
from src.eval.plausibility import train_edge_jump_threshold, edge_incoherence_rate, negative_rate, physical_consistency_corr  # noqa: E402
from scripts.run_phase1d_localized_cqr import (  # noqa: E402
    train_quantile_model, conformity_scores, weighted_quantile, MIN_LOCAL_CAL_N,
)
from src.models.cqr_uq import QUANTILES, ALPHAS  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEED = 42
SELECTED_LAMBDA = 0.1
W_NONNEG = 0.1
SELECTED_FEATURE_COLS = FEATURE_COLS


def three_way_comparison(df, edges):
    edge_threshold = train_edge_jump_threshold(df, edges, percentile=95.0)

    lin = train_linear(df, feature_cols=SELECTED_FEATURE_COLS)
    base_gnn = train_gnn(df, edges, seed=SEED, w_prior=0.0, w_nonneg=0.0, feature_cols=SELECTED_FEATURE_COLS)
    phys_gnn = train_gnn(df, edges, seed=SEED, w_prior=SELECTED_LAMBDA, w_nonneg=W_NONNEG,
                          feature_cols=SELECTED_FEATURE_COLS)

    test_mask = (df["split"] == "test").values
    rows = []
    for name, res, preds in [
        ("linear_regression", lin, lin["preds_raw"]),
        ("baseline_gnn_no_physics", base_gnn, base_gnn["preds_raw"]),
        ("physics_gnn", phys_gnn, phys_gnn["preds_raw"]),
    ]:
        m = res["metrics"]
        rows.append({
            "model": name,
            "train_r2": m["train"]["r2"], "train_rmse": m["train"]["rmse"],
            "test_r2": m["test"]["r2"], "test_rmse": m["test"]["rmse"], "test_mae": m["test"]["mae"],
            "test_n": m["test"]["n"],
            "neg_rate_test": negative_rate(preds[test_mask]),
            "edge_incoherence_rate_test": edge_incoherence_rate(preds, df, edges, edge_threshold),
            "edge_jump_threshold_tha": edge_threshold,
            "prior_gen_corr_test": res["plaus_test"]["prior_gen_corr"],
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "phase2_test_comparison.csv", index=False)

    pred_df = df.copy()
    pred_df["y_pred_soc_tha"] = phys_gnn["preds_raw"]
    pred_df["residual"] = pred_df["y_pred_soc_tha"] - pred_df["soc_tha_mean"]
    pred_df["mass_balance_proxy_raw"] = phys_gnn["proxy_raw"]
    pred_df.to_csv(RESULTS_DIR / "phase2_test_predictions_physics_gnn.csv", index=False)

    print("=== Phase 2: three-way comparison, TEST (evaluated once) ===")
    print(comparison.to_string(index=False))
    return comparison, phys_gnn


def cqr_test_evaluation(df):
    df_q, edges_q, preds_raw = train_quantile_model()
    assert df_q["split"].equals(df["split"]), "quantile-model graph must match the main graph"

    q_idx = {q: i for i, q in enumerate(QUANTILES)}
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values
    y_true = df["soc_tha_mean"].values
    superblock = df["superblock"].values

    coverage_rows = []
    all_pred_cols = {}
    for nominal, (q_lo, q_hi) in ALPHAS.items():
        alpha = 1 - nominal
        lo_raw, hi_raw = preds_raw[:, q_idx[q_lo]], preds_raw[:, q_idx[q_hi]]

        cal_lo, cal_hi, cal_y = lo_raw[val_mask], hi_raw[val_mask], y_true[val_mask]
        cal_sb = superblock[val_mask]
        cal_scores = conformity_scores(cal_lo, cal_hi, cal_y)
        n_cal = len(cal_scores)

        # global
        level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        Q_global = np.quantile(cal_scores, level)
        lo_g, hi_g = lo_raw - Q_global, hi_raw + Q_global

        # localized (Mondrian), fallback to global for sparse/absent blocks
        block_Q = {}
        for sb in np.unique(cal_sb):
            block_scores = cal_scores[cal_sb == sb]
            if len(block_scores) < MIN_LOCAL_CAL_N:
                continue
            n_b = len(block_scores)
            level_b = min(np.ceil((n_b + 1) * (1 - alpha)) / n_b, 1.0)
            block_Q[int(sb)] = np.quantile(block_scores, level_b)
        Q_local_arr = np.array([block_Q.get(int(sb), Q_global) for sb in superblock])
        lo_l, hi_l = lo_raw - Q_local_arr, hi_raw + Q_local_arr

        test_y = y_true[test_mask]
        for method, lo_adj, hi_adj in [("global", lo_g, hi_g), ("localized_mondrian", lo_l, hi_l)]:
            t_lo, t_hi = lo_adj[test_mask], hi_adj[test_mask]
            cov = float(np.mean((test_y >= t_lo) & (test_y <= t_hi)))
            raw_width = t_hi - t_lo
            n_degenerate = int((raw_width < 0).sum())
            # A block-local Q can over-tighten enough to invert hi<lo (an
            # empty conformal set for that node) -- rare (verified: 1/2076
            # test nodes under Mondrian at both nominal levels, 0/2076
            # under global), but a raw negative "width" is meaningless for
            # reporting/pricing, so it's floored at 0 here (a coverage MISS
            # either way -- y can never fall in an empty interval, so
            # flooring doesn't change the coverage number above, only the
            # width summary). Disclosed via n_degenerate, not silently fixed.
            width = float(np.mean(np.clip(raw_width, 0, None)))
            coverage_rows.append({
                "nominal_level": nominal, "calibration_method": method, "n_calibration": n_cal,
                "n_test": int(test_mask.sum()), "empirical_coverage_test": cov, "mean_interval_width_tha": width,
                "n_degenerate_interval_test": n_degenerate,
            })
        all_pred_cols[f"q{q_lo}_raw"] = lo_raw
        all_pred_cols[f"q{q_hi}_raw"] = hi_raw
        all_pred_cols[f"lo_adj_global_{nominal}"] = lo_g
        all_pred_cols[f"hi_adj_global_{nominal}"] = hi_g
        all_pred_cols[f"lo_adj_mondrian_{nominal}"] = lo_l
        all_pred_cols[f"hi_adj_mondrian_{nominal}"] = hi_l

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(RESULTS_DIR / "phase2_cqr_test_coverage.csv", index=False)

    pred_df = df.copy()
    pred_df["y_true_soc_tha"] = y_true
    for k, v in all_pred_cols.items():
        pred_df[k] = v
    pred_df.to_csv(RESULTS_DIR / "phase2_cqr_test_predictions.csv", index=False)

    print()
    print("=== Phase 2: CQR coverage, TEST (calibrated on val, evaluated once) ===")
    print(coverage_df.to_string(index=False))
    return coverage_df


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")

    comparison, phys_gnn = three_way_comparison(df, edges)
    coverage_df = cqr_test_evaluation(df)

    print()
    print(f"wrote -> {RESULTS_DIR / 'phase2_test_comparison.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase2_test_predictions_physics_gnn.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase2_cqr_test_coverage.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase2_cqr_test_predictions.csv'}")
    return comparison, coverage_df


if __name__ == "__main__":
    main()
