"""
Extension Phase 1B -- pre-registered lambda (physics-loss weight) sweep,
temporal dC/dt track, VAL-ONLY (mirrors scripts/run_phase1b_lambda_sweep.py
exactly, on the extension's own graph/split). Re-swept here because the
loss formulation changed meaningfully: the proxy target's accretion input
is now a rate (`ndvi_rate_per_day`, scale ~1e-3 to 1e-4 per day before
scaling) instead of a level (`ndvi`, scale ~0-1) -- the submitted paper's
selected lambda=0.1 (results/phase1b_lambda_summary.csv) was tuned against
a different proxy distribution and is not assumed to transfer.

lambda values (FIXED IN ADVANCE, identical grid to the submitted paper's
Phase 1B, no new values added after seeing results): {0.01, 0.05, 0.1, 0.3,
0.5, 1.0, 2.0}. Each x 5 seeds {1..5}, val-only. ALL 35 rows reported.

DECISION RULE (fixed BEFORE running this script, identical mechanical rule
to the submitted paper's Phase 1B -- same tolerance, same tie-break):
    1. For each lambda, compute mean val R^2 across the 5 seeds.
    2. best_r2 = max over lambda of that mean.
    3. Candidate set = every lambda with mean val R^2 >= best_r2 - 0.05.
    4. Selected lambda = the candidate with the highest mean val
       prior-generalization correlation.

Writes results/ext_phase1b_lambda_sweep.csv (all 35 rows) and
results/ext_phase1b_lambda_summary.csv (mean +/- std per lambda, plus the
columns used by the decision rule).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph_temporal import DATA_PROCESSED  # noqa: E402
from src.models.experiment_runner_temporal import train_gnn_temporal  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]
LAMBDAS = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
W_NONNEG_FIXED = 0.1
R2_TOLERANCE = 0.05


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges_ext_temporal.npy")

    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            r = train_gnn_temporal(df, edges, seed=seed, w_prior=lam, w_nonneg=W_NONNEG_FIXED)
            rows.append({
                "lambda_w_prior": lam, "w_nonneg": W_NONNEG_FIXED, "seed": seed,
                "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
                "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"], "best_epoch": r["best_epoch"],
            })
            print(f"lambda={lam:<5} seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
                  f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "ext_phase1b_lambda_sweep.csv", index=False)

    summary = full_df.groupby("lambda_w_prior").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_prior_gen_corr_mean=("val_prior_gen_corr", "mean"), val_prior_gen_corr_std=("val_prior_gen_corr", "std"),
    ).reset_index()

    best_r2 = summary["val_r2_mean"].max()
    summary["within_tolerance"] = summary["val_r2_mean"] >= (best_r2 - R2_TOLERANCE)
    candidates = summary[summary["within_tolerance"]]
    selected_row = candidates.loc[candidates["val_prior_gen_corr_mean"].idxmax()]
    selected_lambda = float(selected_row["lambda_w_prior"])

    summary.to_csv(RESULTS_DIR / "ext_phase1b_lambda_summary.csv", index=False)

    print()
    print("=== Extension Phase 1B summary (val, mean +/- std across 5 seeds) ===")
    print(summary.to_string(index=False))
    print()
    print(f"best mean val R2 = {best_r2:.4f} (tolerance band: >= {best_r2 - R2_TOLERANCE:.4f})")
    print(f"candidates within tolerance: {candidates['lambda_w_prior'].tolist()}")
    print(f"SELECTED lambda (mechanical rule): {selected_lambda} "
          f"(mean val R2={selected_row['val_r2_mean']:.4f}, "
          f"mean val prior-gen corr={selected_row['val_prior_gen_corr_mean']:.4f})")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1b_lambda_sweep.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1b_lambda_summary.csv'}")
    return full_df, summary, selected_lambda


if __name__ == "__main__":
    main()
