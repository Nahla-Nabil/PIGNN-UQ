"""
Addendum C4: two-date augmentation for temporal robustness (region 1, §4
spatial CV on train ∪ val; no test node is used).

Graph is duplicated: copy 0 carries t0 imagery, copy 1 carries t1 imagery
(T1_SWAP columns; bands without a t1 acquisition keep t0 values, stated),
same labels, no edges between copies. The augmented model trains on both
copies of the fold-train nodes (early stopping on the t0 copy). Baseline =
the same model trained on t0 only. For each, OOF predictions are made with
t0 inputs and with t1 inputs; reported: OOF R2 (t0 inputs), OOF R2 (t1
inputs), and Spearman rho between the two prediction sets.
Writes results/rev_c4_twodate.csv.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import FEATURE_SETS, load_rev, fit_predict  # noqa: E402
from scripts.run_rev_e1_cv import folds, SEEDS, INNER_FRAC  # noqa: E402
from scripts.run_rev_e6_uq_val import parse  # noqa: E402
from scripts.run_revision_test_eval import T1_SWAP, selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    sel, _ = selections()
    df, edges = load_rev()
    n = len(df)
    df1 = df.copy()
    for a, b in T1_SWAP.items():
        df1[a] = df[b]
    both = pd.concat([df, df1], ignore_index=True)
    e2 = np.hstack([edges, edges + n])
    pool = np.where(df.split.isin(["train", "val"]).values)[0]
    fl = folds(df, pool)
    y = df.soc_tha_mean.values
    rows = []
    for fam in ("gnn", "phys_gnn"):
        _, fs, params = parse(sel[fam])
        cols = FEATURE_SETS[fs]
        for seed in SEEDS:
            o = {k: np.full(n, np.nan) for k in ("base_t0", "base_t1", "aug_t0", "aug_t1")}
            for k, (fit, ev) in enumerate(fl):
                rng = np.random.default_rng(seed * 100 + k)
                g = df.phase1_superblock.values[fit]
                ug = np.unique(g)
                sg = rng.choice(ug, size=max(1, int(round(INNER_FRAC * len(ug)))), replace=False)
                stop, fit2 = fit[np.isin(g, sg)], fit[~np.isin(g, sg)]
                pb = fit_predict(df, edges, fam, cols, fit2, [ev], seed, params, stop_idx=stop)
                pb1 = fit_predict(df, edges, fam, cols, fit2, [ev], seed, params, stop_idx=stop, pred_df=df1)
                pa = fit_predict(both, e2, fam, cols, np.concatenate([fit2, fit2 + n]), [ev, ev + n], seed,
                                 params, stop_idx=stop)
                o["base_t0"][ev], o["base_t1"][ev] = pb[ev], pb1[ev]
                o["aug_t0"][ev], o["aug_t1"][ev] = pa[ev], pa[ev + n]
            yy = y[pool]
            r2 = {k: 1 - np.mean((v[pool] - yy) ** 2) / np.var(yy) for k, v in o.items()}
            rows.append({"family": fam, "seed": seed, **{f"oof_r2_{k}": v for k, v in r2.items()},
                         "rho_t0_t1_base": spearmanr(o["base_t0"][pool], o["base_t1"][pool]).statistic,
                         "rho_t0_t1_aug": spearmanr(o["aug_t0"][pool], o["aug_t1"][pool]).statistic})
            print(fam, seed, json.dumps({k: round(float(v), 3) for k, v in rows[-1].items() if k not in ("family",)}),
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_c4_twodate.csv", index=False)
    print(out.groupby("family").mean(numeric_only=True).round(3).T)


if __name__ == "__main__":
    main()
