"""
Extension Phase 2B -- multi-seed TEST characterization, temporal dC/dt
track. Requested explicitly by the user after reviewing Phase 2's
single-seed (seed=42) test table, to separate two possible explanations
for the ~0.2-0.3 test R^2 jump relative to the submitted paper's test
split: (a) the real-dC/dt physics proxy specifically helping, or (b) the
new split's superblocks simply being easier/more homogeneous, in which
case even the no-physics baseline (and linear regression) would show a
similar jump regardless of any physics term.

**Deliberate, disclosed departure from this project's "evaluate test
exactly once" norm (Phase 2's own docstring; `scripts/run_phase2_test_
eval.py`).** This script evaluates the SAME 5 seeds {1,2,3,4,5} used
throughout this project's val-only stability checks (Phase 1A / Extension
Phase 1A) on TEST, for each of the four model configurations Extension
Phase 1B/1B-control ALREADY fixed via a val-only, pre-registered,
mechanical selection rule:
    - linear regression            (deterministic, no seed dependence)
    - baseline_gnn_no_physics      (w_prior=0, w_nonneg=0)
    - physics_gnn_steady_state_proxy_control  (lambda=1.0, control proxy)
    - physics_gnn_dcdt             (lambda=2.0, real dC/dt proxy)

No hyperparameter, feature-set, or lambda choice made anywhere in this
project is informed by any number this script produces -- every one of
those choices was already fixed by Extension Phase 1B/1B-control's
val-only sweeps before this script existed. This script ONLY adds
variance characterization to an already-fixed test-set comparison, which
is what makes it defensible to run test five times per config here even
though the project's general norm is "test, once": nothing downstream of
this script's output selects or tunes anything further.

Writes results/ext_phase2b_multiseed_test.csv (every row, every seed) and
results/ext_phase2b_multiseed_test_summary.csv (mean +/- std per model).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph_temporal import DATA_PROCESSED  # noqa: E402
from src.models.gnn_common_temporal import FEATURE_COLS  # noqa: E402
from src.models.experiment_runner_temporal import train_gnn_temporal, train_linear_temporal  # noqa: E402
from src.models.physics_terms import compute_mass_balance_proxy  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]  # same 5 seeds as every other multi-seed check in this project
SELECTED_LAMBDA_DCDT = 2.0          # Extension Phase 1B, results/ext_phase1b_lambda_summary.csv
SELECTED_LAMBDA_STEADYSTATE = 1.0   # Extension Phase 1B control, results/ext_phase1b_steadystate_control_summary.csv
W_NONNEG = 0.1
SELECTED_FEATURE_COLS = FEATURE_COLS


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges_ext_temporal.npy")

    rows = []

    # --- linear regression: deterministic, run once, replicated across the
    # 5 seed slots (std=0 BY CONSTRUCTION), identical convention to
    # Extension Phase 1A / the submitted paper's Phase 1A ---
    lin = train_linear_temporal(df, feature_cols=SELECTED_FEATURE_COLS)
    for seed in SEEDS:
        rows.append({
            "model": "linear_regression", "seed": seed,
            "test_r2": lin["metrics"]["test"]["r2"], "test_rmse": lin["metrics"]["test"]["rmse"],
            "test_mae": lin["metrics"]["test"]["mae"], "test_prior_gen_corr": lin["plaus_test"]["prior_gen_corr"],
            "note": "deterministic OLS -- identical across seeds, not re-fit per seed",
        })
    print(f"linear regression (deterministic): test R2={lin['metrics']['test']['r2']:.4f}, "
          f"test RMSE={lin['metrics']['test']['rmse']:.2f}, prior-gen corr={lin['plaus_test']['prior_gen_corr']:.4f}")

    # --- baseline GNN, no physics: 5 seeds, TEST ---
    for seed in SEEDS:
        r = train_gnn_temporal(df, edges, seed=seed, w_prior=0.0, w_nonneg=0.0, feature_cols=SELECTED_FEATURE_COLS)
        rows.append({
            "model": "baseline_gnn_no_physics", "seed": seed,
            "test_r2": r["metrics"]["test"]["r2"], "test_rmse": r["metrics"]["test"]["rmse"],
            "test_mae": r["metrics"]["test"]["mae"], "test_prior_gen_corr": r["plaus_test"]["prior_gen_corr"],
            "note": "",
        })
        print(f"baseline GNN seed={seed}: test R2={r['metrics']['test']['r2']:.4f}, "
              f"test RMSE={r['metrics']['test']['rmse']:.2f}, prior-gen corr={r['plaus_test']['prior_gen_corr']:.4f}")

    # --- physics GNN, steady-state-proxy control (lambda=1.0): 5 seeds, TEST ---
    for seed in SEEDS:
        r = train_gnn_temporal(df, edges, seed=seed, w_prior=SELECTED_LAMBDA_STEADYSTATE, w_nonneg=W_NONNEG,
                                feature_cols=SELECTED_FEATURE_COLS, proxy_fn=compute_mass_balance_proxy)
        rows.append({
            "model": "physics_gnn_steady_state_proxy_control", "seed": seed,
            "test_r2": r["metrics"]["test"]["r2"], "test_rmse": r["metrics"]["test"]["rmse"],
            "test_mae": r["metrics"]["test"]["mae"], "test_prior_gen_corr": r["plaus_test"]["prior_gen_corr"],
            "note": "",
        })
        print(f"physics GNN (steady-state control) seed={seed}: test R2={r['metrics']['test']['r2']:.4f}, "
              f"test RMSE={r['metrics']['test']['rmse']:.2f}, prior-gen corr={r['plaus_test']['prior_gen_corr']:.4f}")

    # --- physics GNN, real dC/dt (lambda=2.0): 5 seeds, TEST ---
    for seed in SEEDS:
        r = train_gnn_temporal(df, edges, seed=seed, w_prior=SELECTED_LAMBDA_DCDT, w_nonneg=W_NONNEG,
                                feature_cols=SELECTED_FEATURE_COLS)
        rows.append({
            "model": "physics_gnn_dcdt", "seed": seed,
            "test_r2": r["metrics"]["test"]["r2"], "test_rmse": r["metrics"]["test"]["rmse"],
            "test_mae": r["metrics"]["test"]["mae"], "test_prior_gen_corr": r["plaus_test"]["prior_gen_corr"],
            "note": "",
        })
        print(f"physics GNN (real dC/dt) seed={seed}: test R2={r['metrics']['test']['r2']:.4f}, "
              f"test RMSE={r['metrics']['test']['rmse']:.2f}, prior-gen corr={r['plaus_test']['prior_gen_corr']:.4f}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "ext_phase2b_multiseed_test.csv", index=False)

    summary = full_df.groupby("model", sort=False).agg(
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        test_rmse_mean=("test_rmse", "mean"), test_rmse_std=("test_rmse", "std"),
        test_mae_mean=("test_mae", "mean"), test_mae_std=("test_mae", "std"),
        test_prior_gen_corr_mean=("test_prior_gen_corr", "mean"), test_prior_gen_corr_std=("test_prior_gen_corr", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "ext_phase2b_multiseed_test_summary.csv", index=False)

    print()
    print("=== Extension Phase 2B summary (TEST, mean +/- std across 5 seeds) ===")
    print(summary.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'ext_phase2b_multiseed_test.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase2b_multiseed_test_summary.csv'}")
    return full_df, summary


if __name__ == "__main__":
    main()
