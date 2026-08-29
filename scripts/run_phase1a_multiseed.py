"""
Phase 1A -- multi-seed baseline stability (pre-registered, VAL-ONLY).

Re-runs the three-way model comparison (linear regression / baseline GNN /
physics-informed GNN) across 5 fixed seeds {1,2,3,4,5} on the Phase 0 split
(train/val; test is never touched here). Reports mean +/- std on VAL for
R^2, RMSE, and the prior-generalization correlation -- checks whether Day
2's finding (physics term helps R^2 but doesn't close the gap to linear;
physics term's predictions correlate much more strongly with the
mass-balance proxy) is a stable phenomenon or a single-seed artifact.

Linear regression is closed-form OLS -- no seed dependence -- so it is run
once and reported with std=0 BY CONSTRUCTION, not simulated. This is
disclosed in the output, not hidden by fabricating 5 near-identical runs.

Writes results/phase1a_multiseed_val.csv (every row, every seed) and
results/phase1a_multiseed_summary.csv (mean +/- std per model).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.experiment_runner import train_gnn, train_linear  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]

# Same fixed hyperparameters as Day 2 (paper/method_notes.md), reproduced
# here as the FIXED starting configuration Phase 1B/1C then perturb.
W_PRIOR_DEFAULT = 0.3
W_NONNEG_DEFAULT = 0.1


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")

    rows = []

    # --- Linear regression: deterministic, run once ---
    lin = train_linear(df)
    for seed in SEEDS:
        rows.append({
            "model": "linear_regression", "seed": seed,
            "val_r2": lin["metrics"]["val"]["r2"], "val_rmse": lin["metrics"]["val"]["rmse"],
            "val_mae": lin["metrics"]["val"]["mae"], "val_prior_gen_corr": lin["plaus_val"]["prior_gen_corr"],
            "best_epoch": np.nan, "note": "deterministic OLS -- identical across seeds, not re-fit per seed",
        })
    print(f"linear regression (deterministic): val R2={lin['metrics']['val']['r2']:.3f}, "
          f"val RMSE={lin['metrics']['val']['rmse']:.1f}, prior-gen corr={lin['plaus_val']['prior_gen_corr']:.3f}")

    # --- Baseline GNN (no physics): w_prior=w_nonneg=0 ---
    for seed in SEEDS:
        r = train_gnn(df, edges, seed=seed, w_prior=0.0, w_nonneg=0.0)
        rows.append({
            "model": "baseline_gnn_no_physics", "seed": seed,
            "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
            "val_mae": r["metrics"]["val"]["mae"], "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"],
            "best_epoch": r["best_epoch"], "note": "",
        })
        print(f"baseline GNN seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
              f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}, "
              f"best_epoch={r['best_epoch']}")

    # --- Physics-informed GNN: w_prior=0.3, w_nonneg=0.1 (Day 2 fixed weights) ---
    for seed in SEEDS:
        r = train_gnn(df, edges, seed=seed, w_prior=W_PRIOR_DEFAULT, w_nonneg=W_NONNEG_DEFAULT)
        rows.append({
            "model": "physics_gnn", "seed": seed,
            "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
            "val_mae": r["metrics"]["val"]["mae"], "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"],
            "best_epoch": r["best_epoch"], "note": "",
        })
        print(f"physics GNN seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
              f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}, "
              f"best_epoch={r['best_epoch']}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "phase1a_multiseed_val.csv", index=False)

    summary = full_df.groupby("model").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_prior_gen_corr_mean=("val_prior_gen_corr", "mean"), val_prior_gen_corr_std=("val_prior_gen_corr", "std"),
    ).reset_index()
    # Linear regression's std is exactly 0 by construction (5 identical
    # rows from one deterministic fit) -- groupby.std() already returns 0.0
    # here, not NaN, since n=5 identical values; left as computed, not
    # special-cased, so the CSV is honest about why it's 0.
    summary.to_csv(RESULTS_DIR / "phase1a_multiseed_summary.csv", index=False)

    print()
    print("=== Phase 1A summary (val, mean +/- std across 5 seeds) ===")
    print(summary.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'phase1a_multiseed_val.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase1a_multiseed_summary.csv'}")
    return full_df, summary


if __name__ == "__main__":
    main()
