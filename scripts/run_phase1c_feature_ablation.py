"""
Phase 1C -- feature ablation (base+NDWI vs. base+NDMI), VAL-ONLY,
pre-registered.

Two separate configurations, each tested independently (not combined into
one run, per the instruction): base features + NDWI, and base features +
NDMI. Both real Sentinel-2-derived indices (see
src/data_pipeline/build_graph.py's docstring on the NDWI/NDMI formulas and
the NUMBER_SOURCE_LOG.md entry on why a small, disclosed B03 pull was
needed to make NDWI non-degenerate with NDMI). Model = the physics-informed
GNN at the lambda Phase 1B selected mechanically (0.1) and the fixed
w_nonneg (0.1) -- this is the config Phase 1C is extending, not a fresh
model choice.

DECISION RULE (fixed BEFORE running this script, applied mechanically
after -- symmetric with Phase 1B's rule, not invented post-hoc):
    Three candidates: base (already run in Phase 1B at lambda=0.1, reused
    here, not re-run), base+NDWI, base+NDMI.
    1. best_r2 = max over the 3 candidates of mean val R^2.
    2. Candidate set = every config with mean val R^2 >= best_r2 - 0.05.
    3. Selected feature set = the candidate with the highest mean val
       prior-generalization correlation among those.

Writes results/phase1c_feature_ablation.csv (all 10 new rows: 2 configs x
5 seeds) and results/phase1c_feature_summary.csv (all 3 candidates,
including the reused base row, plus the decision columns).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.gnn_common import FEATURE_COLS  # noqa: E402
from src.models.experiment_runner import train_gnn  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = [1, 2, 3, 4, 5]
SELECTED_LAMBDA = 0.1  # Phase 1B's mechanically-selected value -- see
# results/phase1b_lambda_summary.csv. Hardcoded (not re-derived) so this
# script's behavior can't silently drift if 1B is re-run later.
W_NONNEG_FIXED = 0.1
R2_TOLERANCE = 0.05

FEATURE_SETS = {
    "base+NDWI": FEATURE_COLS + ["ndwi"],
    "base+NDMI": FEATURE_COLS + ["ndmi"],
}


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")

    rows = []
    for config_name, feat_cols in FEATURE_SETS.items():
        for seed in SEEDS:
            r = train_gnn(df, edges, seed=seed, w_prior=SELECTED_LAMBDA, w_nonneg=W_NONNEG_FIXED,
                           feature_cols=feat_cols)
            rows.append({
                "feature_config": config_name, "seed": seed,
                "val_r2": r["metrics"]["val"]["r2"], "val_rmse": r["metrics"]["val"]["rmse"],
                "val_prior_gen_corr": r["plaus_val"]["prior_gen_corr"], "best_epoch": r["best_epoch"],
            })
            print(f"{config_name} seed={seed}: val R2={r['metrics']['val']['r2']:.3f}, "
                  f"val RMSE={r['metrics']['val']['rmse']:.1f}, prior-gen corr={r['plaus_val']['prior_gen_corr']:.3f}")

    full_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "phase1c_feature_ablation.csv", index=False)

    summary = full_df.groupby("feature_config").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_prior_gen_corr_mean=("val_prior_gen_corr", "mean"), val_prior_gen_corr_std=("val_prior_gen_corr", "std"),
    ).reset_index()

    # Pull in the "base" candidate from Phase 1B's lambda=0.1 rows (reused,
    # not re-run -- identical config, identical seeds).
    b1b = pd.read_csv(RESULTS_DIR / "phase1b_lambda_sweep.csv")
    base_rows = b1b[np.isclose(b1b["lambda_w_prior"], SELECTED_LAMBDA)]
    base_row = pd.DataFrame([{
        "feature_config": "base", "val_r2_mean": base_rows["val_r2"].mean(), "val_r2_std": base_rows["val_r2"].std(),
        "val_rmse_mean": base_rows["val_rmse"].mean(), "val_rmse_std": base_rows["val_rmse"].std(),
        "val_prior_gen_corr_mean": base_rows["val_prior_gen_corr"].mean(),
        "val_prior_gen_corr_std": base_rows["val_prior_gen_corr"].std(),
    }])
    summary = pd.concat([base_row, summary], ignore_index=True)

    # --- Apply the pre-registered decision rule mechanically ---
    best_r2 = summary["val_r2_mean"].max()
    summary["within_tolerance"] = summary["val_r2_mean"] >= (best_r2 - R2_TOLERANCE)
    candidates = summary[summary["within_tolerance"]]
    selected_row = candidates.loc[candidates["val_prior_gen_corr_mean"].idxmax()]
    selected_config = str(selected_row["feature_config"])

    summary.to_csv(RESULTS_DIR / "phase1c_feature_summary.csv", index=False)

    print()
    print("=== Phase 1C summary (val, mean +/- std across 5 seeds; 'base' reused from Phase 1B) ===")
    print(summary.to_string(index=False))
    print()
    print(f"best mean val R2 = {best_r2:.4f} (tolerance band: >= {best_r2 - R2_TOLERANCE:.4f})")
    print(f"candidates within tolerance: {candidates['feature_config'].tolist()}")
    print(f"SELECTED feature config (mechanical rule): {selected_config} "
          f"(mean val R2={selected_row['val_r2_mean']:.4f}, "
          f"mean val prior-gen corr={selected_row['val_prior_gen_corr_mean']:.4f})")
    print(f"wrote -> {RESULTS_DIR / 'phase1c_feature_ablation.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'phase1c_feature_summary.csv'}")
    return full_df, summary, selected_config


if __name__ == "__main__":
    main()
