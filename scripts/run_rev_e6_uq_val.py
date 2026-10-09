"""
Revision E5/E6: UQ method comparison and selection, VAL-INTERNAL ONLY
(paper/revision_preregistration.md §5). Test nodes are never predicted here.

Models are fit on the train split (leakage-safe induced subgraph), val nodes
are predicted on the val-induced subgraph, and val is split 50/50 (seed 7)
into calibrate/check. Selection: lowest mean Winkler score at 90% on
val-check among methods with val-check coverage >= 88% at 90%.

Writes results/rev_e6_uq_val.csv and results/rev_e6_uq_selection.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import FEATURE_SETS, load_rev  # noqa: E402
from src.models.rev_uq import (quantile_gnn, quantile_xgb, ncp_rf, calibrate, interval_metrics,  # noqa: E402
                               LEVELS, ensemble_quantiles)
from src.models.gnn_common import TARGET_COL  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SEED, VAL_SPLIT_SEED = 1, 7


def parse(key):
    fam, fs, params = key.split("|", 2)
    return fam, fs, json.loads(params)


def build_bases(df, edges, sel, fit_idx, pred_groups):
    """phys_gnn: the W2 (Addendum A1) selection if it exists, else §4's."""
    pred_idx = np.concatenate(pred_groups)
    _, gfs, gp = parse(sel["gnn"])
    w2 = RESULTS / "rev_w2_selection.json"
    phys_key = json.loads(w2.read_text())["selected_phys"] if w2.exists() else sel["phys_gnn"]
    _, pfs, pp = parse(phys_key)
    _, xfs, xp = parse(sel["xgb"])
    _, rfs, rp = parse(sel["rf"])
    y = df[TARGET_COL].values
    q_gnn = quantile_gnn(df, edges, FEATURE_SETS[gfs], fit_idx, pred_groups, SEED, gp["hidden"])
    q_xgb = quantile_xgb(df, FEATURE_SETS[xfs], fit_idx, pred_idx, SEED, xp["max_depth"])
    ncp = ncp_rf(df, FEATURE_SETS[rfs], fit_idx, pred_idx, SEED, rp["max_features"], rp["min_samples_leaf"])
    return {
        "CQR-GNN": ("cqr", q_gnn),
        "CQR-PhysGNN": ("cqr", quantile_gnn(df, edges, FEATURE_SETS[pfs], fit_idx, pred_groups, SEED,
                                            pp["hidden"], lam=pp.get("lambda", 0.0),
                                            proxy_kind=pp.get("proxy", "P0"))),
        "CQR-XGB": ("cqr", q_xgb),
        "NCP-RF": ("ncp", ncp),
        "Ens-Q": ("cqr", ensemble_quantiles(q_gnn, q_xgb, ncp)),
        "Trivial": ("trivial", float(y[fit_idx].mean())),
    }


def main():
    df, edges = load_rev()
    sel = json.loads((RESULTS / "rev_e1_selection.json").read_text())
    y = df[TARGET_COL].values
    sb = df["superblock"].values
    train = np.where(df["split"] == "train")[0]
    val = np.where(df["split"] == "val")[0]
    perm = np.random.default_rng(VAL_SPLIT_SEED).permutation(val)
    cal, chk = perm[: len(perm) // 2], perm[len(perm) // 2:]

    bases = build_bases(df, edges, sel, train, [val])
    rows = []
    for name, (kind, base) in bases.items():
        for level in LEVELS:
            for mondrian in (False, True):
                lo, hi = calibrate(kind, base, y, cal, chk, level, sb, mondrian=mondrian)
                rows.append({"method": name, "calibration": "mondrian" if mondrian else "global",
                             "level": level, **interval_metrics(y[chk], lo, hi, level)})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "rev_e6_uq_val.csv", index=False)

    g90 = res[(res.level == 0.90) & (res.calibration == "global") & (res.method != "Trivial")]
    ok = g90[g90.coverage >= 0.88]
    chosen = ok.loc[ok.winkler.idxmin(), "method"] if len(ok) else None
    (RESULTS / "rev_e6_uq_selection.json").write_text(json.dumps({"selected_uq": chosen}, indent=2))
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("selected:", chosen)


if __name__ == "__main__":
    main()
