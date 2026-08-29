"""
Formal linear regression baseline -- same features, same graph_nodes.csv
train/val/test split as the GNN variants (val is unused here since linear
regression needs no early stopping, but is excluded from the fit exactly
like it's excluded from GNN training, for an apples-to-apples split).

This was originally an ad hoc diagnostic (src/models/gnn_common.py-adjacent
script run inline during the Day 1 checkpoint, see
paper/NUMBER_SOURCE_LOG.md #13) that revealed the baseline GNN was
overfitting relative to a plain linear model. Promoted to a first-class,
formally saved result per the user's Day 2 instruction: linear regression
is now a required third row in the model comparison table, not just a
diagnostic aside.

Writes results/linear_baseline_metrics.csv and
results/linear_baseline_predictions.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import SEED, FEATURE_COLS, TARGET_COL, load_graph, regression_metrics  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"


def train():
    df, _edges = load_graph()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    model = LinearRegression()
    model.fit(train_df[FEATURE_COLS].values, train_df[TARGET_COL].values)

    preds_raw = model.predict(df[FEATURE_COLS].values)

    metrics_train = regression_metrics(train_df[TARGET_COL].values, preds_raw[df["split"] == "train"])
    metrics_val = regression_metrics(val_df[TARGET_COL].values, preds_raw[df["split"] == "val"])
    metrics_test = regression_metrics(test_df[TARGET_COL].values, preds_raw[df["split"] == "test"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([
        {"model": "linear_regression", "split": "train", "seed": SEED, **metrics_train},
        {"model": "linear_regression", "split": "val", "seed": SEED, **metrics_val},
        {"model": "linear_regression", "split": "test_spatial_block", "seed": SEED, **metrics_test},
    ])
    metrics_df.to_csv(RESULTS_DIR / "linear_baseline_metrics.csv", index=False)

    pred_df = df.copy()
    pred_df["y_pred_soc_tha"] = preds_raw
    pred_df["residual"] = pred_df["y_pred_soc_tha"] - pred_df[TARGET_COL]
    pred_df.to_csv(RESULTS_DIR / "linear_baseline_predictions.csv", index=False)

    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coefficient": model.coef_})
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]
    coef_df.to_csv(RESULTS_DIR / "linear_baseline_coefficients.csv", index=False)

    print("train:", metrics_train)
    print("val:", metrics_val)
    print("test (spatial block):", metrics_test)
    print(f"wrote -> {RESULTS_DIR / 'linear_baseline_metrics.csv'}")
    return metrics_train, metrics_val, metrics_test


if __name__ == "__main__":
    train()
