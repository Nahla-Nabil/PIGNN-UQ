"""
Extension Phase 1A -- multi-seed stability, temporal dC/dt track
(pre-registered, VAL-ONLY, mirrors scripts/run_phase1a_multiseed.py exactly,
on the extension's own graph/split instead of the submitted paper's).

Re-runs the three-way model comparison (linear regression / baseline GNN /
physics-GNN-with-real-dC/dt) across the SAME 5 fixed seeds {1,2,3,4,5} used
throughout this project, on `data/processed/graph_nodes_ext_temporal.csv`'s
`split_ext` train/val (test untouched here). Physics GNN uses Day-2's
original fixed starting weights (w_prior=0.3, w_nonneg=0.1) as the
FIXED starting configuration -- Phase 1B (below) then pre-registers a
lambda re-sweep around it, exactly mirroring the submitted paper's own
Phase 1A -> 1B sequence.

Writes results/ext_phase1a_multiseed_val.csv and
results/ext_phase1a_multiseed_summary.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph_temporal import DATA_PROCESSED  # noqa: E402
from src.models.experiment_runner_temporal import train_gnn_temporal, train_linear_temporal  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]
W_PRIOR_DEFAULT = 0.3
W_NONNEG_DEFAULT = 0.1


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges_ext_temporal.npy")

    rows = []

    lin = train_linear_temporal(df)
    for seed in SEEDS:
        rows.append({
            "model": "linear_regression", "seed": seed,
            "val_r2": lin["metrics"]["val"]["r2"], "val_rmse": lin["metrics"]["val"]["rmse"],
            "val_mae": lin["metrics"]["val"]["mae"], "val_prior_gen_corr": lin["plaus_val"]["prior_gen_corr"],
            "best_epoch": np.nan, "note": "deterministic OLS -- identical across seeds, not re-fit per seed",
        })
    print(f"linear regression (deterministic): val R2={lin['metrics']['val']['r2']:.3f}, "
          f"val RMSE={lin['metrics']['val']['rmse']:.1f}, prior-gen corr={lin['plaus_val']['prior_gen_corr']:.3f}")

    for seed in SEEDS:
        r = train_gnn_temporal(df, edges, seed=seed, w_prior=0.0, w_nonneg=0.0)
        rows.append({
            "model": "baseline_gnn_no_physics", "seed": seed,
            "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
            "val_mae": r["metrics"]["val"]["mae"], "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"],
            "best_epoch": r["best_epoch"], "note": "",
        })
        print(f"baseline GNN seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
              f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}, "
              f"best_epoch={r['best_epoch']}")

    for seed in SEEDS:
        r = train_gnn_temporal(df, edges, seed=seed, w_prior=W_PRIOR_DEFAULT, w_nonneg=W_NONNEG_DEFAULT)
        rows.append({
            "model": "physics_gnn_dcdt", "seed": seed,
            "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
            "val_mae": r["metrics"]["val"]["mae"], "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"],
            "best_epoch": r["best_epoch"], "note": "",
        })
        print(f"physics GNN (real dC/dt) seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
              f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}, "
              f"best_epoch={r['best_epoch']}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "ext_phase1a_multiseed_val.csv", index=False)

    summary = full_df.groupby("model").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_prior_gen_corr_mean=("val_prior_gen_corr", "mean"), val_prior_gen_corr_std=("val_prior_gen_corr", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "ext_phase1a_multiseed_summary.csv", index=False)

    print()
    print("=== Extension Phase 1A summary (val, mean +/- std across 5 seeds) ===")
    print(summary.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1a_multiseed_val.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1a_multiseed_summary.csv'}")
    return full_df, summary


if __name__ == "__main__":
    main()
