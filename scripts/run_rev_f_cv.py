"""
Addendum F1: hybrid / ensemble / feature-parity candidates vs the selected
XGBoost and Phys-GNN, spatial CV on train ∪ val in BOTH regions (test unused).
Writes results/rev_f1_cv_summary.csv, rev_f1_cv_oof_<region>.csv, rev_f1_selection.json.
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import FEATURE_SETS, load_rev  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e1_cv import folds, run_config, SEEDS  # noqa: E402
from scripts.run_rev_e6_uq_val import parse  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
ORDER = ["xgb", "phys_gnn", "ens", "stack_gnn", "phys_gnn_f3aug"]


def candidates(sel):
    _, xfs, xp = parse(sel["xgb"])
    _, pfs, pp = parse(sel["phys_gnn"])
    stack = {**pp, "xgb_cols": FEATURE_SETS[xfs], "xgb_depth": xp["max_depth"]}
    return {"xgb": ("xgb", xfs, xp), "phys_gnn": ("phys_gnn", pfs, pp),
            "stack_gnn": ("stack_gnn", pfs, stack), "phys_gnn_f3aug": ("phys_gnn_f3aug", "F3_spec_dem_temporal", pp)}


def main():
    sel, _ = selections()
    cands = candidates(sel)
    rows, t0 = [], time.time()
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        pool = np.where(df["split"].isin(["train", "val"]).values)[0]
        fl = folds(df, pool)
        y = df[TARGET_COL].values[pool]
        oof, frows = {}, []
        for name, (fam, fs, params) in cands.items():
            r, o = [], {}
            run_config(df, edges, pool, fl, fam, fs, params, SEEDS, r, o)
            for s in SEEDS:
                oof[(name, s)] = o[(r[0]["key"], s)]
            frows += [{**x, "cand": name} for x in r]
            print(f"[{time.time() - t0:5.0f}s] {rk} {name}", flush=True)
        for s in SEEDS:  # ensemble = unweighted mean of the two references
            oof[("ens", s)] = (oof[("xgb", s)] + oof[("phys_gnn", s)]) / 2
        # fold-level R2 for ens
        fold_of = np.full(len(df), -1)
        for k, (_, ev) in enumerate(fl):
            fold_of[ev] = k
        fp = fold_of[pool]
        for s in SEEDS:
            for k in range(len(fl)):
                m = fp == k
                pk, yk = oof[("ens", s)][m], y[m]
                frows.append({"cand": "ens", "seed": s, "fold": k, "r2": 1 - np.mean((pk - yk) ** 2) / np.var(yk)})
        fr = pd.DataFrame(frows)
        for name in ORDER:
            r2 = [1 - np.mean((oof[(name, s)] - y) ** 2) / np.var(y) for s in SEEDS]
            mae = [np.mean(np.abs(oof[(name, s)] - y)) for s in SEEDS]
            g = fr[fr.cand == name]
            rows.append({"region": rk, "cand": name, "oof_r2": np.mean(r2), "oof_r2_seed_sd": np.std(r2),
                         "fold_r2_se": g.r2.std(ddof=1) / np.sqrt(len(g)), "oof_mae": np.mean(mae)})
        pd.DataFrame({f"{n}|seed{s}": v for (n, s), v in oof.items()}, index=pool).assign(y=y).to_csv(
            RES / f"rev_f1_cv_oof_{rk}.csv")
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "rev_f1_cv_summary.csv", index=False)
    agg = summ.groupby("cand").agg(mean_r2=("oof_r2", "mean"), se=("fold_r2_se", "mean")).reindex(ORDER)
    best = agg.mean_r2.idxmax()
    tied = agg[agg.mean_r2 >= agg.loc[best, "mean_r2"] - agg.loc[best, "se"]]
    winner = [c for c in ORDER if c in tied.index][0]
    out = {"winner": winner, "best_by_mean": best, "tied_within_1se": list(tied.index),
           "agg": agg.round(4).to_dict(orient="index"), "params": {k: list(map(str, v)) for k, v in cands.items()}}
    (RES / "rev_f1_selection.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(agg.round(3))
    print("winner:", winner, "| best by mean:", best, "| tied:", list(tied.index))


if __name__ == "__main__":
    main()
