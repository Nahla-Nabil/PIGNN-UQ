"""
Phase 1B -- pre-registered lambda (physics-loss weight) sweep, VAL-ONLY.

lambda == w_prior in src/models/physics_terms.py's physics_loss(). The
non-negativity weight (w_nonneg) is held fixed at Day 2's value (0.1) --
the instruction names ONE swept quantity, "lambda (physics-loss weight)",
and PLAN.md's own scope cut already treats the non-negativity term as a
corollary of the same physics constraint, not a second free parameter to
also sweep. This is a scope choice, stated here, not a silent one.

lambda values (FIXED IN ADVANCE, no additions/removals after seeing
results): {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0}. Each x 5 seeds {1..5},
val-only. ALL 35 rows are reported, including "obviously bad" ones -- no
row is dropped from the CSV or the log.

DECISION RULE (fixed BEFORE running this script, applied mechanically
after -- not adjusted once numbers were seen):
    1. For each lambda, compute mean val R^2 across the 5 seeds.
    2. best_r2 = max over lambda of that mean.
    3. Candidate set = every lambda with mean val R^2 >= best_r2 - 0.05.
    4. Selected lambda = the candidate with the highest mean val
       prior-generalization correlation.

Writes results/phase1b_lambda_sweep.csv (all 35 rows) and
results/phase1b_lambda_summary.csv (mean +/- std per lambda, plus the
columns used by the decision rule).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.experiment_runner import train_gnn  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]
LAMBDAS = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]  # fixed in advance, see docstring
W_NONNEG_FIXED = 0.1
R2_TOLERANCE = 0.05


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")

    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            r = train_gnn(df, edges, seed=seed, w_prior=lam, w_nonneg=W_NONNEG_FIXED)
            rows.append({
                "lambda_w_prior": lam, "w_nonneg": W_NONNEG_FIXED, "seed": seed,
                "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
                "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"], "best_epoch": r["best_epoch"],
            })
            print(f"lambda={lam:<5} seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
                  f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "phase1b_lambda_sweep.csv", index=False)

    summary = full_df.groupby("lambda_w_prior").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_prior_gen_corr_mean=("val_prior_gen_corr", "mean"), val_prior_gen_corr_std=("val_prior_gen_corr", "std"),
    ).reset_index()

    # --- Apply the pre-registered decision rule mechanically ---
    best_r2 = summary["val_r2_mean"].max()
    summary["within_tolerance"] = summary["val_r2_mean"] >= (best_r2 - R2_TOLERANCE)
    candidates = summary[summary["within_tolerance"]]
    selected_row = candidates.loc[candidates["val_prior_gen_corr_mean"].idxmax()]
    selected_lambda = float(selected_row["lambda_w_prior"])

    summary.to_csv(RESULTS_DIR / "phase1b_lambda_summary.csv", index=False)

    print()
    print("=== Phase 1B summary (val, mean +/- std across 5 seeds) ===")
    print(summary.to_string(index=False))
    print()
    print(f"best mean val R2 = {best_r2:.4f} (tolerance band: >= {best_r2 - R2_TOLERANCE:.4f})")
    print(f"candidates within tolerance: {candidates['lambda_w_prior'].tolist()}")
    print(f"SELECTED lambda (mechanical rule): {selected_lambda} "
          f"(mean val R2={selected_row['val_r2_mean']:.4f}, "
          f"mean val prior-gen corr={selected_row['val_prior_gen_corr_mean']:.4f})")
    print(f"wrote -> {RESULTS_DIR / 'phase1b_lambda_sweep.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase1b_lambda_summary.csv'}")
    return full_df, summary, selected_lambda


if __name__ == "__main__":
    main()
