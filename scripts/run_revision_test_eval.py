"""
Revision final evaluation: THE ONLY SCRIPT THAT READS TEST LABELS
(paper/revision_preregistration.md §0, §4-§7 and Addendum A1-A4).

For each region (r1 = submitted AOI, r2 = Everglades coast):
 1. Point models: every selected family (§4; phys_gnn from W2/A1), fit on
    train (GNN early stopping on val), test predicted on the test-induced
    subgraph; 3 seeds for stochastic models. Metrics: R2, MAE, RMSE,
    Spearman, skill vs train mean; the same on the 2-hop interior test
    subset; prior-gen corr (P0 and the selected prior); corr with the
    independent Radabaugh map.
 2. UQ: every §5/A2 method fit on train, calibrated on val, evaluated on
    test, global and Mondrian; adaptivity gate.
 3. Issuance (§6, D4): rules A and B for the selected UQ method, global
    (post-hoc) and Mondrian (pre-registered in the submission); revenue
    sensitivity (A4).
 4. Non-permanence (A4): Spearman(width / deduction, 2010-2020 mangrove loss).
 5. E8: second-date imagery stability.
Cross-region transfer (A3b): train on r1 -> test r2, and r2 -> r1.

--dry-run: test nodes are excluded entirely and val is split by fine-grid
groups into a stand-in val/test, so the whole pipeline is exercised without
reading test labels. Outputs are prefixed dryrun_ and no stamp is written.
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
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import FEATURE_SETS, load_rev, fit_predict, metrics, mass_balance_proxy  # noqa: E402
from src.models.rev_uq import calibrate, interval_metrics, LEVELS, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from src.pricing.issuance import issuance_table, revenue_table, deduction_vm0033, node_uncertainty_pct  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases, parse  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = [1, 2, 3]
STAMP = RESULTS / "rev_test_RUN_STAMP.json"
T1_SWAP = {"ndvi": "t1_ndvi", "b04": "t1_b04", "b08": "t1_b08", "b11": "t1_b11", "b03": "t1_b03",
           "b05": "t1_b05", "b8a": "t1_b8a", "ndre": "t1_ndre", "ndmi": "t1_ndmi", "mndwi": "t1_mndwi"}


def selections():
    sel = json.loads((RESULTS / "rev_e1_selection.json").read_text())
    sel["phys_gnn"] = json.loads((RESULTS / "rev_w2_selection.json").read_text())["selected_phys"]
    uq = json.loads((RESULTS / "rev_e6_uq_selection.json").read_text())["selected_uq"]
    return sel, uq


def prepare(region, dry):
    df, edges = load_rev(region)
    if dry:
        df = df.copy()
        val_groups = np.unique(df.loc[df.split == "val", "phase1_superblock"])
        stand_in = set(np.random.default_rng(7).permutation(val_groups)[: len(val_groups) // 2])
        new = df.split.copy()
        new[df.split == "test"] = "excluded"
        new[(df.split == "val") & df.phase1_superblock.isin(stand_in)] = "test"
        df["split"] = new
    return df, edges


def two_hop_interior(edges, test_mask):
    s, d = edges
    bad = np.zeros(len(test_mask), bool)
    bad[d[~test_mask[s]]] = True
    bad2 = bad.copy()
    bad2[d[bad[s]]] = True
    return test_mask & ~bad2


POSITIONAL = {"dist_to_tide_km"}  # D8: region-specific by construction (each region's own gauge)


def point_rows(df, edges, sel, fit, stop, test, interior, tag, seeds=SEEDS, drop_positional=False):
    y = df[TARGET_COL].values
    rad = df["radabaugh_bgc_mgha"].values if "radabaugh_bgc_mgha" in df else np.full(len(df), np.nan)
    _, _, pp = parse(sel["phys_gnn"])
    proxies = {"P0": mass_balance_proxy(df, fit, "P0"), "Psel": mass_balance_proxy(df, fit, pp.get("proxy", "P0"))}
    rows, preds = [], {}
    for fam, key in sel.items():
        _, fs, params = parse(key)
        if drop_positional and fam == "rf_coords":
            continue
        cols = ["lon", "lat"] if fam == "rf_coords" else FEATURE_SETS[fs]
        if drop_positional:
            cols = [c for c in cols if c not in POSITIONAL]
        for seed in ([1] if fam == "linear" else seeds):
            p = fit_predict(df, edges, fam, cols, fit, [test], seed, params, stop_idx=stop)
            preds[(fam, seed)] = p
            ok = np.isfinite(rad[test])
            r = {"eval": tag, "family": fam, "key": key, "seed": seed,
                 **{f"test_{k}": v for k, v in metrics(y[test], p[test], y[fit].mean()).items()},
                 "prior_gen_corr_P0": np.corrcoef(p[test], proxies["P0"][test])[0, 1],
                 "prior_gen_corr_selected": np.corrcoef(p[test], proxies["Psel"][test])[0, 1],
                 "corr_with_radabaugh": np.corrcoef(p[test][ok], rad[test][ok])[0, 1] if ok.sum() > 10 else np.nan}
            if interior is not None and len(interior) > 10:
                r.update({f"interior2hop_{k}": v for k, v in metrics(y[interior], p[interior], y[fit].mean()).items()})
                r["n_interior"] = len(interior)
            rows.append(r)
    return rows, preds


def combine(da, ea, db, eb):
    da, db = da.copy(), db.copy()
    da["region"], db["region"] = "A", "B"
    cols = sorted(set(da.columns) & set(db.columns))
    df = pd.concat([da[cols], db[cols]], ignore_index=True)
    return df, np.hstack([ea, eb + len(da)])


def run_region(rk, region, dry, sel, uq_sel, out):
    df, edges = prepare(region, dry)
    y = df[TARGET_COL].values
    train = np.where(df.split == "train")[0]
    val = np.where(df.split == "val")[0]
    test = np.where(df.split == "test")[0]
    interior = np.where(two_hop_interior(edges, (df.split == "test").values))[0]
    rows, preds = point_rows(df, edges, sel, train, val, test, interior, f"{rk}_in_region")
    out["point"] += rows
    out["extras"][rk] = {
        "n": {"train": len(train), "val": len(val), "test": len(test), "interior": len(interior)},
        "trainmean_r2_on_test": float(1 - np.mean((y[test] - y[train].mean()) ** 2) / np.var(y[test])),
        "label_sd_test": float(np.std(y[test])),
    }
    if "radabaugh_bgc_mgha" in df:
        ok = np.isfinite(df.radabaugh_bgc_mgha.values[test])
        out["extras"][rk]["corr_label_vs_radabaugh_test"] = float(
            np.corrcoef(y[test][ok], df.radabaugh_bgc_mgha.values[test][ok])[0, 1])

    # UQ
    bases = build_bases(df, edges, sel, train, [val, test])
    sb = df["superblock"].values
    intervals = {}
    for name, (kind, base) in bases.items():
        for level in LEVELS:
            for mond in (False, True):
                lo, hi = calibrate(kind, base, y, val, test, level, sb, mondrian=mond)
                intervals[(name, level, mond)] = (lo, hi)
                out["uq"].append({"region": rk, "method": name, "calibration": "mondrian" if mond else "global",
                                  "level": level, **interval_metrics(y[test], lo, hi, level)})
    g = [r for r in out["uq"] if r["region"] == rk and r["method"] == uq_sel and r["level"] == 0.90
         and r["calibration"] == "global"][0]
    out["extras"][rk]["uq_gate"] = {"selected_uq": uq_sel, "rho": g["rho_width_abserr"],
                                    "ci": [g["rho_ci_lo"], g["rho_ci_hi"]], "gate_passed": bool(g["rho_ci_lo"] > 0)}

    # issuance + revenue + non-permanence
    kind, base = bases[uq_sel]
    yhat = (base[test, QUANTILES.index(0.5)] if kind == "cqr" else base[0][test] if kind == "ncp"
            else np.full(len(test), base))
    loss = df["mangrove_loss_frac_2010_2020"].values[test] if "mangrove_loss_frac_2010_2020" in df else None
    for mond in (False, True):
        lo, hi = intervals[(uq_sel, 0.90, mond)]
        cal = "mondrian (pre-registered)" if mond else "global (post-hoc)"
        it = issuance_table(yhat, lo, hi)
        out["issuance"].append({"region": rk, "method": uq_sel, "calibration": cal, **it})
        for r in revenue_table({"flat_buffer": it["total_flat"], "ruleA_lowerbound": it["total_ruleA_lowerbound"],
                                "ruleB_vm0033_node": it["total_ruleB_vm0033_node"],
                                "ruleB_vm0033_project": it["total_ruleB_vm0033_project"]}):
            out["revenue"].append({"region": rk, "calibration": cal, **r})
        if loss is not None:
            dB = deduction_vm0033(node_uncertainty_pct(yhat, lo, hi))
            out["nonperm"].append({"region": rk, "calibration": cal, "n": len(test),
                                   "frac_nodes_with_loss": float((loss > 0).mean()),
                                   "rho_width_loss": spearmanr(hi - lo, loss).statistic,
                                   "rho_deductionB_loss": spearmanr(dB, loss).statistic})

    # E8 temporal stability
    if {"t1_ndvi", "t1_b04"} <= set(df.columns) and df["t1_ndvi"].notna().any():
        df_t1 = df.copy()
        for a, b in T1_SWAP.items():
            if b in df:
                df_t1[a] = df[b]
        for fam in ("gnn", "rf"):
            _, fs, params = parse(sel[fam])
            p1 = fit_predict(df, edges, fam, FEATURE_SETS[fs], train, [test], 1, params, stop_idx=val, pred_df=df_t1)
            p0 = preds[(fam, 1)]
            out["e8"].append({"region": rk, "family": fam, "feature_set": fs,
                              "spearman_t0_t1": spearmanr(p0[test], p1[test]).statistic,
                              "mean_abs_change": float(np.mean(np.abs(p0[test] - p1[test]))),
                              "frac_t1_cloud_imputed": float(df.loc[test, "t1_ndvi"].isna().mean())})
    return df, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--regions", default="r1,r2")
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args()
    prefix = "dryrun_" if args.dry_run else ""
    if not args.dry_run:
        if STAMP.exists() and not args.rerun_reason:
            sys.exit(f"{STAMP} exists: test was already evaluated. Pass --rerun-reason to document a rerun.")
        log = json.loads(STAMP.read_text()) if STAMP.exists() else {"runs": []}
        log["runs"].append({"time": datetime.now().isoformat(), "reason": args.rerun_reason or "first run"})
        STAMP.write_text(json.dumps(log, indent=2))

    sel, uq_sel = selections()
    out = {"point": [], "uq": [], "issuance": [], "revenue": [], "nonperm": [], "e8": [], "extras": {}}
    regs = args.regions.split(",")
    data = {rk: run_region(rk, REGIONS[rk], args.dry_run, sel, uq_sel, out) for rk in regs}

    if len(regs) == 2:  # cross-region transfer (A3b)
        for a, b in ((regs[0], regs[1]), (regs[1], regs[0])):
            df, edges = combine(*data[a], *data[b])
            fit = np.where((df.region == "A") & (df.split == "train"))[0]
            stop = np.where((df.region == "A") & (df.split == "val"))[0]
            test = np.where((df.region == "B") & (df.split == "test"))[0]
            rows, _ = point_rows(df, edges, sel, fit, stop, test, None, f"transfer_{a}_to_{b}")
            out["point"] += rows
            rows, _ = point_rows(df, edges, sel, fit, stop, test, None, f"transfer_nopos_{a}_to_{b}",
                                 drop_positional=True)
            out["point"] += rows

    pt = pd.DataFrame(out["point"])
    pt.to_csv(RESULTS / f"{prefix}rev_test_point_by_seed.csv", index=False)
    num = pt.drop(columns=["key"]).groupby(["eval", "family"]).agg(["mean", "std"])
    num.columns = [f"{a}_{b}" for a, b in num.columns]
    num.to_csv(RESULTS / f"{prefix}rev_test_point_summary.csv")
    for k in ("uq", "issuance", "revenue", "nonperm", "e8"):
        pd.DataFrame(out[k]).to_csv(RESULTS / f"{prefix}rev_test_{k}.csv", index=False)
    (RESULTS / f"{prefix}rev_test_extras.json").write_text(json.dumps(out["extras"], indent=2, default=float))

    pd.set_option("display.width", 250)
    show = [c for c in num.columns if c in ("test_r2_mean", "test_r2_std", "test_mae_mean", "test_spearman_mean",
                                            "interior2hop_r2_mean", "corr_with_radabaugh_mean",
                                            "prior_gen_corr_selected_mean")]
    print(num[show].round(3).to_string())
    print(pd.DataFrame(out["uq"]).query("level == 0.9").round(3).to_string(index=False))
    for k in ("issuance", "nonperm", "e8"):
        print(pd.DataFrame(out[k]).round(3).to_string(index=False))
    print(json.dumps(out["extras"], indent=2, default=float))


if __name__ == "__main__":
    main()
