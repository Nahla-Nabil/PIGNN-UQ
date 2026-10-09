"""
Extension Phase 1B, CONTROL ARM -- pre-registered lambda sweep for the
SUBMITTED PAPER'S ORIGINAL steady-state proxy (single-date NDVI + static
tide-distance, `physics_terms.compute_mass_balance_proxy`, imported
UNCHANGED), but trained on the extension's NEW split
(`graph_nodes_ext_temporal.csv`, `split_ext`) -- i.e. everything about this
run is identical to `run_ext_phase1b_lambda_sweep.py` EXCEPT which proxy
formula the physics loss targets.

Purpose, stated before running (mechanical, not post-hoc): this isolates
the proxy-formula effect (real dC/dt vs. steady-state) from the split
effect (new superblocks vs. the submitted paper's). Without this control,
"does dC/dt outperform steady-state" cannot be answered cleanly, because
the extension's fresh test partition also differs from the submitted
paper's -- comparing dC/dt-on-new-split against steady-state-on-OLD-split
(results/phase1b_lambda_summary.csv) would confound two changes at once.
This script trains steady-state-on-NEW-split so
`extension_temporal_notes.md`'s headline comparison is single-variable.

Same lambda grid, same 5 seeds, same mechanical selection rule as every
other lambda sweep in this project (see run_ext_phase1b_lambda_sweep.py's
docstring for the exact rule text -- reproduced verbatim here).

Writes results/ext_phase1b_steadystate_control_sweep.csv and
results/ext_phase1b_steadystate_control_summary.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph_temporal import DATA_PROCESSED  # noqa: E402
from src.models.experiment_runner_temporal import train_gnn_temporal  # noqa: E402
from src.models.physics_terms import compute_mass_balance_proxy  # noqa: E402

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
            r = train_gnn_temporal(df, edges, seed=seed, w_prior=lam, w_nonneg=W_NONNEG_FIXED,
                                    proxy_fn=compute_mass_balance_proxy)
            rows.append({
                "lambda_w_prior": lam, "w_nonneg": W_NONNEG_FIXED, "seed": seed,
                "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
                "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"], "best_epoch": r["best_epoch"],
            })
            print(f"[steady-state-proxy control] lambda={lam:<5} seed={seed}: "
                  f"val R2={r['metrics']['val']['r2']:.3f}, val RMSE={r['metrics']['val']['rmse']:.1f}, "
                  f"prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "ext_phase1b_steadystate_control_sweep.csv", index=False)

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

    summary.to_csv(RESULTS_DIR / "ext_phase1b_steadystate_control_summary.csv", index=False)

    print()
    print("=== Extension Phase 1B CONTROL (steady-state proxy, new split) summary ===")
    print(summary.to_string(index=False))
    print()
    print(f"best mean val R2 = {best_r2:.4f} (tolerance band: >= {best_r2 - R2_TOLERANCE:.4f})")
    print(f"candidates within tolerance: {candidates['lambda_w_prior'].tolist()}")
    print(f"SELECTED lambda (mechanical rule): {selected_lambda} "
          f"(mean val R2={selected_row['val_r2_mean']:.4f}, "
          f"mean val prior-gen corr={selected_row['val_prior_gen_corr_mean']:.4f})")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1b_steadystate_control_sweep.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase1b_steadystate_control_summary.csv'}")
    return full_df, summary, selected_lambda


if __name__ == "__main__":
    main()
