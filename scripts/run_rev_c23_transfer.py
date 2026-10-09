"""
Addendum C2 (proper local split) + C3 (region-wise feature
standardization) for cross-region use. VALIDATION ONLY: test nodes are
dropped before anything runs; scoring is on the Addendum B `check` nodes.

local (Addendum B) is split by fine-grid groups 50/50 (seed 11) into
local_fit / local_cal, so conformal calibration residuals are always
out-of-sample.
Writes results/rev_c23_transfer.csv and _summary.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import FEATURE_SETS, fit_predict, metrics  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e6_uq_val import parse  # noqa: E402
from scripts.run_revision_test_eval import combine, POSITIONAL, selections  # noqa: E402
from scripts.run_rev_b_local_calibration import load_no_test, split_val, conformal, interval_stats  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
FAMS = ["xgb", "rf", "gnn", "phys_gnn"]
SEEDS = [1, 2, 3]


def group_half(df, idx, seed):
    g = df.phase1_superblock.values[idx]
    ug = np.unique(g)
    a = set(np.random.default_rng(seed).permutation(ug)[: len(ug) // 2])
    m = np.isin(g, list(a))
    return idx[m], idx[~m]


def region_z(df, cols):
    """C3: z-score each feature with the region's own (non-test) statistics."""
    d = df.copy()
    for c in cols:
        v = d[c].astype(float)
        d[c] = (v - v.mean()) / (v.std() + 1e-12)
    return d


def main():
    sel, _ = selections()
    raw = {rk: load_no_test(rk) for rk in ("r1", "r2")}
    all_cols = sorted({c for f in FAMS for c in FEATURE_SETS[parse(sel[f])[1]]} - POSITIONAL)
    rows = []
    for arm in ("raw", "region_z"):
        data = raw if arm == "raw" else {rk: (region_z(d, all_cols), e) for rk, (d, e) in raw.items()}
        for a, b in (("r1", "r2"), ("r2", "r1")):
            df, edges = combine(*data[a], *data[b])
            y = df[TARGET_COL].values
            src_tr = np.where((df.region == "A") & (df.split == "train"))[0]
            src_val = np.where((df.region == "A") & (df.split == "val"))[0]
            tgt_idx = np.where(df.region.values == "B")[0]
            loc_rel, chk_rel = split_val(df.iloc[tgt_idx].reset_index(drop=True))
            local, check = tgt_idx[loc_rel], tgt_idx[chk_rel]
            lfit, lcal = group_half(df, local, 11)
            for fam in FAMS:
                _, fs, params = parse(sel[fam])
                cols = [c for c in FEATURE_SETS[fs] if c not in POSITIONAL]
                for seed in SEEDS:
                    groups = [lfit, lcal, check]
                    p0 = fit_predict(df, edges, fam, cols, src_tr, groups, seed, params, stop_idx=src_val)
                    bc, ac = np.polyfit(p0[lfit], y[lfit], 1)
                    p1 = ac + bc * p0
                    p2 = fit_predict(df, edges, fam, cols, np.concatenate([src_tr, lfit]), groups, seed, params,
                                     stop_idx=src_val)
                    p3 = fit_predict(df, edges, fam, cols, lfit, groups, seed, params, stop_idx=src_val)
                    for name, p in (("V0_zero_shot", p0), ("V1b_recalibrated", p1), ("V2b_pooled", p2),
                                    ("V3b_local_only", p3)):
                        rows.append({"arm": arm, "direction": f"{a}_to_{b}", "family": fam, "seed": seed,
                                     "variant": name, "n_local_fit": len(lfit), "n_local_cal": len(lcal),
                                     "n_check": len(check),
                                     **{f"check_{k}": v for k, v in metrics(y[check], p[check], y[lfit].mean()).items()},
                                     **{f"localcal_{k}": v for k, v in interval_stats(
                                         y[check], *conformal(y[lcal], p[lcal], p[check])).items()}})
                print(arm, a, "->", b, fam, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_c23_transfer.csv", index=False)
    s = out.groupby(["arm", "direction", "family", "variant"]).mean(numeric_only=True).reset_index()
    s.to_csv(RES / "rev_c23_transfer_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[["arm", "direction", "family", "variant", "check_r2", "check_mae", "localcal_cov90",
             "localcal_width90", "localcal_winkler90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
