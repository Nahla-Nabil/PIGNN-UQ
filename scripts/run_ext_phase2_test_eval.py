"""
Extension Phase 2 -- THE single test-set evaluation, temporal dC/dt track
(mirrors scripts/run_phase2_test_eval.py exactly in structure/discipline,
on the extension's own graph/split/proxy).

Everything above this script (build_graph_temporal.py's split-carving,
Extension Phase 1A/1B) was decided on VAL data only, over the extension's
own train/val pool -- the extension's test partition (`split_ext=="test"`)
has not been read by any script before this one, and (per
build_graph_temporal.py's own construction) is drawn entirely from coarse
superblocks the SUBMITTED paper's own operative test never touched.

Configuration evaluated (fixed by Extension Phase 1B's mechanical rule,
no test-informed tuning):
    lambda (w_prior) = 2.0     (results/ext_phase1b_lambda_summary.csv,
                                 dC/dt-proxy sweep)
    w_nonneg = 0.1             (held fixed, same convention as the
                                 submitted paper -- never swept here either)
    feature set = base FEATURE_COLS, UNCHANGED from the submitted paper
                                 (this extension changes the physics-loss
                                 PROXY, not the model's input features --
                                 see physics_terms_temporal.py's docstring
                                 for why that is a deliberate, disclosed
                                 scope choice, keeping this a single-
                                 variable comparison against Phase 2)
    seed = 42                  (project's declared standard seed)

Also retrains linear regression, the no-physics baseline GNN, AND the
steady-state-proxy CONTROL variant (submitted paper's original proxy
formula, run through its own pre-registered lambda sweep on this SAME new
split -- lambda=1.0, results/ext_phase1b_steadystate_control_summary.csv)
on the same extension split, for a four-way comparison table. The
dC/dt-vs-steady-state pair is the genuine single-variable comparison this
extension's headline claim rests on: same split, same seed, same
architecture, same feature set -- only the physics-loss proxy formula (and
each arm's OWN mechanically-selected lambda) differs.

Writes:
  results/ext_phase2_test_comparison.csv
  results/ext_phase2_test_predictions_physics_gnn_dcdt.csv
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
from src.eval.plausibility import negative_rate  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SEED = 42
SELECTED_LAMBDA_DCDT = 2.0
SELECTED_LAMBDA_STEADYSTATE = 1.0
W_NONNEG = 0.1
SELECTED_FEATURE_COLS = FEATURE_COLS


def main():
    df = pd.read_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv")
    edges = np.load(DATA_PROCESSED / "graph_edges_ext_temporal.npy")

    lin = train_linear_temporal(df, feature_cols=SELECTED_FEATURE_COLS)
    base_gnn = train_gnn_temporal(df, edges, seed=SEED, w_prior=0.0, w_nonneg=0.0, feature_cols=SELECTED_FEATURE_COLS)
    phys_gnn = train_gnn_temporal(df, edges, seed=SEED, w_prior=SELECTED_LAMBDA_DCDT, w_nonneg=W_NONNEG,
                                   feature_cols=SELECTED_FEATURE_COLS)
    phys_gnn_ss = train_gnn_temporal(df, edges, seed=SEED, w_prior=SELECTED_LAMBDA_STEADYSTATE, w_nonneg=W_NONNEG,
                                      feature_cols=SELECTED_FEATURE_COLS, proxy_fn=compute_mass_balance_proxy)

    test_mask = (df["split_ext"] == "test").values
    rows = []
    for name, res, preds in [
        ("linear_regression", lin, lin["preds_raw"]),
        ("baseline_gnn_no_physics", base_gnn, base_gnn["preds_raw"]),
        ("physics_gnn_steady_state_proxy_control", phys_gnn_ss, phys_gnn_ss["preds_raw"]),
        ("physics_gnn_dcdt", phys_gnn, phys_gnn["preds_raw"]),
    ]:
        m = res["metrics"]
        rows.append({
            "model": name,
            "train_r2": m["train"]["r2"], "train_rmse": m["train"]["rmse"],
            "val_r2": m["val"]["r2"], "val_rmse": m["val"]["rmse"],
            "test_r2": m["test"]["r2"], "test_rmse": m["test"]["rmse"], "test_mae": m["test"]["mae"],
            "test_n": m["test"]["n"],
            "neg_rate_test": negative_rate(preds[test_mask]),
            "prior_gen_corr_test": res["plaus_test"]["prior_gen_corr"],
        })

    comparison = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "ext_phase2_test_comparison.csv", index=False)

    pred_df = df.copy()
    pred_df["y_pred_soc_tha"] = phys_gnn["preds_raw"]
    pred_df["residual"] = pred_df["y_pred_soc_tha"] - pred_df["soc_tha_mean"]
    pred_df["mass_balance_proxy_dcdt_raw"] = phys_gnn["proxy_raw"]
    pred_df.to_csv(RESULTS_DIR / "ext_phase2_test_predictions_physics_gnn_dcdt.csv", index=False)

    print("=== Extension Phase 2: three-way comparison, TEST (evaluated once) ===")
    print(comparison.to_string(index=False))
    print()
    print(f"wrote -> {RESULTS_DIR / 'ext_phase2_test_comparison.csv'}")
    print(f"wrote -> {RESULTS_DIR / 'ext_phase2_test_predictions_physics_gnn_dcdt.csv'}")
    return comparison


if __name__ == "__main__":
    main()
