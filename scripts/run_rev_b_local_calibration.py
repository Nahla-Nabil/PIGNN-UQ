"""
Addendum B (paper/revision_preregistration.md): cross-region use with a
small local labeled sample. VALIDATION ONLY: target-region test nodes are
never used (they are dropped before anything runs).

Writes results/rev_b_local_calibration.csv.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import FEATURE_SETS, load_rev, fit_predict, metrics  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e6_uq_val import parse  # noqa: E402
from scripts.run_revision_test_eval import combine, POSITIONAL, selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
FAMS = ["xgb", "rf", "gnn", "phys_gnn"]
SEEDS = [1, 2, 3]
ALPHA = 0.10


def load_no_test(rk):
    df, edges = load_rev(REGIONS[rk])
    keep = np.where(df.split != "test")[0]  # test nodes removed entirely
    pos = np.full(len(df), -1)
    pos[keep] = np.arange(len(keep))
    s, d = pos[edges[0]], pos[edges[1]]
    m = (s >= 0) & (d >= 0)
    return df.iloc[keep].reset_index(drop=True), np.vstack([s[m], d[m]])


def split_val(df):
    g = np.unique(df.loc[df.split == "val", "phase1_superblock"])
    local_g = set(np.random.default_rng(7).permutation(g)[: len(g) // 2])
    is_val = (df.split == "val").values
    local = np.where(is_val & df.phase1_superblock.isin(local_g).values)[0]
    check = np.where(is_val & ~df.phase1_superblock.isin(local_g).values)[0]
    return local, check


def conformal(y_cal, p_cal, p_eval):
    s = np.abs(y_cal - p_cal)
    n = len(s)
    q = np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0))
    return p_eval - q, p_eval + q


def interval_stats(y, lo, hi):
    w = hi - lo
    wink = w + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {"cov90": float(np.mean((y >= lo) & (y <= hi))), "width90": float(np.mean(w)),
            "winkler90": float(np.mean(wink))}


def main():
    sel, _ = selections()
    data = {rk: load_no_test(rk) for rk in ("r1", "r2")}
    rows = []
    for a, b in (("r1", "r2"), ("r2", "r1")):
        df, edges = combine(*data[a], *data[b])
        y = df[TARGET_COL].values
        src_tr = np.where((df.region == "A") & (df.split == "train"))[0]
        src_val = np.where((df.region == "A") & (df.split == "val"))[0]
        tgt = df.region.values == "B"
        loc_rel, chk_rel = split_val(df[tgt].reset_index(drop=True))
        tgt_idx = np.where(tgt)[0]
        local, check = tgt_idx[loc_rel], tgt_idx[chk_rel]
        for fam in FAMS:
            _, fs, params = parse(sel[fam])
            cols = [c for c in FEATURE_SETS[fs] if c not in POSITIONAL]
            for seed in SEEDS:
                # V0 zero-shot (source train, source val for GNN stopping)
                p0 = fit_predict(df, edges, fam, cols, src_tr, [src_val, local, check], seed, params, stop_idx=src_val)
                # V1 linear recalibration on the local sample
                bcoef, acoef = np.polyfit(p0[local], y[local], 1)
                p1 = acoef + bcoef * p0
                # V2 pooled: source train + local (GNN stops on source val)
                p2 = fit_predict(df, edges, fam, cols, np.concatenate([src_tr, local]), [local, check], seed,
                                 params, stop_idx=src_val)
                # V3 local-only (GNN stops on source val; tabular needs none)
                p3 = fit_predict(df, edges, fam, cols, local, [local, check], seed, params, stop_idx=src_val)
                for name, p in (("V0_zero_shot", p0), ("V1_recalibrated", p1), ("V2_pooled", p2),
                                ("V3_local_only", p3)):
                    r = {"direction": f"{a}_to_{b}", "family": fam, "seed": seed, "variant": name,
                         "n_local": len(local), "n_check": len(check),
                         **{f"check_{k}": v for k, v in metrics(y[check], p[check], y[local].mean()).items()}}
                    # local conformal (calibrated on the target local sample)
                    r.update({f"localcal_{k}": v for k, v in interval_stats(y[check], *conformal(
                        y[local], p[local], p[check])).items()})
                    if name == "V0_zero_shot":  # source-calibrated intervals for comparison
                        r.update({f"srccal_{k}": v for k, v in interval_stats(y[check], *conformal(
                            y[src_val], p0[src_val], p0[check])).items()})
                    rows.append(r)
            print(f"{a}->{b} {fam} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_b_local_calibration.csv", index=False)
    summ = out.groupby(["direction", "family", "variant"]).mean(numeric_only=True).reset_index()
    summ.to_csv(RES / "rev_b_local_calibration_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ[["direction", "family", "variant", "check_r2", "check_mae", "localcal_cov90", "localcal_width90",
                "srccal_cov90", "srccal_width90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
