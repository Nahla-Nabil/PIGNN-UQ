"""Shared helpers for Addendum F scripts: predict with any F1 candidate
(including the unweighted `ens`) through one call."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import FEATURE_SETS, fit_predict  # noqa: E402
from scripts.run_rev_f_cv import candidates  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def winner_name():
    return json.loads((RES / "rev_f1_selection.json").read_text())["winner"]


def predict(df, edges, name, fit_idx, pred_groups, seed, stop_idx=None, drop_cols=()):
    """Point predictions of candidate `name` (xgb, phys_gnn, ens, stack_gnn,
    phys_gnn_f3aug) trained on fit_idx."""
    sel, _ = selections()
    c = candidates(sel)
    if name == "ens":
        a = predict(df, edges, "xgb", fit_idx, pred_groups, seed, stop_idx, drop_cols)
        b = predict(df, edges, "phys_gnn", fit_idx, pred_groups, seed, stop_idx, drop_cols)
        return (a + b) / 2
    fam, fs, params = c[name]
    cols = [x for x in FEATURE_SETS[fs] if x not in drop_cols]
    if "xgb_cols" in params:
        params = {**params, "xgb_cols": [x for x in params["xgb_cols"] if x not in drop_cols]}
    return fit_predict(df, edges, fam, cols, fit_idx, pred_groups, seed, params, stop_idx=stop_idx)


def inner_stop(df, fit_idx, seed, frac=0.15):
    """15% of fit-set fine-grid groups held out for GNN early stopping."""
    g = df["phase1_superblock"].values[fit_idx]
    ug = np.unique(g)
    sg = np.random.default_rng(seed).choice(ug, size=max(1, int(round(frac * len(ug)))), replace=False)
    return fit_idx[~np.isin(g, sg)], fit_idx[np.isin(g, sg)]
