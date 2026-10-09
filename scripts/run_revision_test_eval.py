"""
Revision final test evaluation: THE ONLY SCRIPT THAT READS TEST LABELS
(paper/revision_preregistration.md §0, §4-§7). Run once, after
rev_e1_selection.json and rev_e6_uq_selection.json are final.

1. Point models: every selected family, fit on train (GNN early stopping on
   val, as in the submitted paper), test predicted on the test-induced
   subgraph; 3 seeds for stochastic models (mean ± SD). Also: 2-hop
   buffered test subset, prior-generalization diagnostic, and agreement
   with the independent Radabaugh et al. (2023) map (E7).
2. UQ: every method fit on train, calibrated on all of val, evaluated on
   test, global and Mondrian; adaptivity gate (§5).
3. Issuance (§6): rules A and B for the selected UQ method, global
   (post-hoc, flagged) and Mondrian (pre-registered in the submission).
4. E8: selected models re-predicted with second-date (t1) imagery.

Writes results/rev_test_*.csv/json. Refuses to run twice unless --rerun-reason is given.
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import FEATURE_SETS, load_rev, fit_predict, metrics, mass_balance_proxy  # noqa: E402
from src.models.rev_uq import calibrate, interval_metrics, LEVELS, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL, DATA_PROCESSED  # noqa: E402
from src.pricing.issuance import issuance_table  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases, parse  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = [1, 2, 3]
STAMP = RESULTS / "rev_test_RUN_STAMP.json"
T1_SWAP = {"ndvi": "t1_ndvi", "b04": "t1_b04", "b08": "t1_b08", "b11": "t1_b11", "b03": "t1_b03",
           "b05": "t1_b05", "b8a": "t1_b8a", "ndre": "t1_ndre", "ndmi": "t1_ndmi", "mndwi": "t1_mndwi"}


def two_hop_interior(edges, test_mask):
    n = len(test_mask)
    bad = np.zeros(n, bool)
    s, d = edges
    bad[d[~test_mask[s]]] = True  # 1-hop neighbour outside test
    bad2 = bad.copy()
    bad2[d[bad[s]]] = True  # 2-hop
    return test_mask & ~bad2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args()
    if STAMP.exists() and not args.rerun_reason:
        sys.exit(f"{STAMP} exists: test was already evaluated. Pass --rerun-reason to document a rerun.")
    log = json.loads(STAMP.read_text()) if STAMP.exists() else {"runs": []}
    log["runs"].append({"time": datetime.now().isoformat(), "reason": args.rerun_reason or "first run"})
    STAMP.write_text(json.dumps(log, indent=2))

    df, edges = load_rev()
    y = df[TARGET_COL].values
    sel = json.loads((RESULTS / "rev_e1_selection.json").read_text())
    uq_sel = json.loads((RESULTS / "rev_e6_uq_selection.json").read_text())["selected_uq"]
    train = np.where(df.split == "train")[0]
    val = np.where(df.split == "val")[0]
    test = np.where(df.split == "test")[0]
    interior = np.where(two_hop_interior(edges, (df.split == "test").values))[0]
    proxy = mass_balance_proxy(df, train)
    rad = df["radabaugh_bgc_mgha"].values
    ymean_tr = y[train].mean()

    # ---- 1. point models ----
    rows, preds = [], {}
    for fam, key in sel.items():
        _, fs, params = parse(key)
        cols = ["lon", "lat"] if fam == "rf_coords" else FEATURE_SETS[fs]
        seeds = [1] if fam == "linear" else SEEDS
        for seed in seeds:
            p = fit_predict(df, edges, fam, cols, train, [test], seed, params, stop_idx=val)
            preds[(fam, seed)] = p
            m = metrics(y[test], p[test], ymean_tr)
            mi = metrics(y[interior], p[interior], ymean_tr)
            ok = np.isfinite(rad[test])
            rows.append({"family": fam, "key": key, "seed": seed, **{f"test_{k}": v for k, v in m.items()},
                         **{f"interior2hop_{k}": v for k, v in mi.items()},
                         "n_interior": len(interior),
                         "prior_gen_corr": np.corrcoef(p[test], proxy[test])[0, 1],
                         "corr_with_radabaugh": np.corrcoef(p[test][ok], rad[test][ok])[0, 1]})
    pt = pd.DataFrame(rows)
    pt.to_csv(RESULTS / "rev_test_point_by_seed.csv", index=False)
    summ = pt.drop(columns=["key"]).groupby("family").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(RESULTS / "rev_test_point_summary.csv")
    ok = np.isfinite(rad[test])
    label_ref = {"corr_label_vs_radabaugh_test": float(np.corrcoef(y[test][ok], rad[test][ok])[0, 1]),
                 "label_mean_test": float(y[test].mean()), "radabaugh_mean_test": float(np.nanmean(rad[test])),
                 "trainmean_r2_on_test": float(1 - np.mean((y[test] - ymean_tr) ** 2) / np.var(y[test]))}

    # ---- 2. UQ ----
    bases = build_bases(df, edges, sel, train, [val, test])
    sb = df["superblock"].values
    uq_rows, intervals = [], {}
    for name, (kind, base) in bases.items():
        for level in LEVELS:
            for mond in (False, True):
                lo, hi = calibrate(kind, base, y, val, test, level, sb, mondrian=mond)
                intervals[(name, level, mond)] = (lo, hi)
                uq_rows.append({"method": name, "calibration": "mondrian" if mond else "global", "level": level,
                                **interval_metrics(y[test], lo, hi, level)})
    uq = pd.DataFrame(uq_rows)
    uq.to_csv(RESULTS / "rev_test_uq.csv", index=False)
    g = uq[(uq.method == uq_sel) & (uq.level == 0.90) & (uq.calibration == "global")].iloc[0]
    gate = {"selected_uq": uq_sel, "rho": g.rho_width_abserr, "ci": [g.rho_ci_lo, g.rho_ci_hi],
            "gate_passed": bool(g.rho_ci_lo > 0)}

    # ---- 3. issuance ----
    kind, base = bases[uq_sel]
    if kind == "cqr":
        yhat = base[test, QUANTILES.index(0.5)]
    elif kind == "ncp":
        yhat = base[0][test]
    else:
        yhat = np.full(len(test), base)
    iss = []
    for mond in (False, True):
        lo, hi = intervals[(uq_sel, 0.90, mond)]
        iss.append({"method": uq_sel, "calibration": "mondrian (pre-registered)" if mond else "global (post-hoc)",
                    **issuance_table(yhat, lo, hi)})
    # same rules if the label product were Radabaugh's map instead (product-choice sensitivity)
    iss_df = pd.DataFrame(iss)
    iss_df.to_csv(RESULTS / "rev_test_issuance.csv", index=False)
    product_choice = {"sum_label_sanderman_test": float(y[test].sum()),
                      "sum_radabaugh_test": float(np.nansum(rad[test])),
                      "ratio_radabaugh_over_sanderman": float(np.nansum(rad[test]) / y[test][ok].sum())}

    # ---- 4. E8 temporal stability ----
    e8 = []
    df_t1 = df.copy()
    for a, b in T1_SWAP.items():
        df_t1[a] = df[b]
    for fam in ("gnn", "rf", "xgb"):
        _, fs, params = parse(sel[fam])
        if fs == "F3_spec_dem_temporal":
            continue
        cols = FEATURE_SETS[fs]
        p0 = preds[(fam, 1)]
        # same training data (t0 features); only the test-time inputs are swapped to t1
        p1 = fit_predict(df, edges, fam, cols, train, [test], 1, params, stop_idx=val, pred_df=df_t1)
        e8.append({"family": fam, "feature_set": fs, "spearman_t0_t1": spearmanr(p0[test], p1[test]).statistic,
                   "mean_abs_change": float(np.mean(np.abs(p0[test] - p1[test]))),
                   "frac_t1_cloud_imputed": float(df.loc[test, "t1_ndvi"].isna().mean())})
    e8_df = pd.DataFrame(e8)
    e8_df.to_csv(RESULTS / "rev_test_e8_temporal.csv", index=False)

    out = {"label_reference": label_ref, "uq_gate": gate, "product_choice": product_choice}
    (RESULTS / "rev_test_extras.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(summ[[c for c in summ.columns if c.split("_")[0] in ("test", "interior2hop", "prior", "corr")
                and ("r2" in c or "mae" in c or "spearman" in c or "prior" in c or "radabaugh" in c)]].round(3).to_string())
    print(uq.round(3).to_string(index=False))
    print(iss_df.round(3).to_string(index=False))
    print(e8_df.round(3).to_string(index=False))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
