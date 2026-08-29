"""
Assembles the Day 2 three-way comparison table (linear regression /
baseline GNN / physics-informed GNN), combining each model's regression
metrics (already in results/*_metrics.csv) with plausibility metrics
computed HERE, uniformly, from each model's results/*_predictions.csv --
so the plausibility numbers are computed the exact same way for every
model, not reimplemented three times.

Writes:
  results/plausibility_comparison.csv
  results/day2_model_comparison.csv  (the actual table for the paper)
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.build_graph import DATA_PROCESSED  # noqa: E402
from src.models.physics_terms import compute_mass_balance_proxy  # noqa: E402
from src.eval.plausibility import full_plausibility_report, train_edge_jump_threshold  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"

MODELS = {
    "linear_regression": RESULTS_DIR / "linear_baseline_predictions.csv",
    "baseline_gnn_no_physics": RESULTS_DIR / "baseline_gnn_predictions.csv",
    "physics_gnn": RESULTS_DIR / "physics_gnn_predictions.csv",
}
METRICS_FILES = {
    "linear_regression": RESULTS_DIR / "linear_baseline_metrics.csv",
    "baseline_gnn_no_physics": RESULTS_DIR / "baseline_gnn_metrics.csv",
    "physics_gnn": RESULTS_DIR / "physics_gnn_metrics.csv",
}


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges.npy")
    train_mask = (df["split"] == "train").values

    proxy_raw, proxy_stats = compute_mass_balance_proxy(df, train_mask)
    edge_threshold = train_edge_jump_threshold(df, edges, percentile=95.0)
    print(f"edge jump threshold (95th pct of real train-side neighbor diffs): {edge_threshold:.2f} t/ha")

    plaus_rows = []
    for model_name, pred_path in MODELS.items():
        pred_df = pd.read_csv(pred_path)
        y_pred = pred_df["y_pred_soc_tha"].values
        report = full_plausibility_report(y_pred, df, edges, proxy_raw, edge_threshold)
        plaus_rows.append({"model": model_name, **report})
    plaus_df = pd.DataFrame(plaus_rows)
    plaus_df.to_csv(RESULTS_DIR / "plausibility_comparison.csv", index=False)
    print(plaus_df.to_string(index=False))

    metrics_rows = []
    for model_name, mpath in METRICS_FILES.items():
        m = pd.read_csv(mpath)
        test_row = m[m["split"] == "test_spatial_block"].iloc[0]
        train_row = m[m["split"] == "train"].iloc[0]
        metrics_rows.append({
            "model": model_name,
            "train_r2": train_row["r2"], "train_rmse": train_row["rmse"],
            "test_r2": test_row["r2"], "test_rmse": test_row["rmse"], "test_mae": test_row["mae"],
            "test_n": test_row["n"],
        })
    metrics_df = pd.DataFrame(metrics_rows)

    combined = metrics_df.merge(plaus_df, on="model")
    combined.to_csv(RESULTS_DIR / "day2_model_comparison.csv", index=False)
    print()
    print("=== Day 2 three-way comparison table ===")
    print(combined.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'day2_model_comparison.csv'}")
    return combined


if __name__ == "__main__":
    main()
