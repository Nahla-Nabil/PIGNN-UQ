"""
Addendum D: field soil-core validation against the Smithsonian Coastal
Carbon Network (CCN) library (paper/revision_preregistration.md, D-a..D-d, D9).

Outputs:
  results/rev_d_field_cores.csv       per-location field carbon + matched node values
  results/rev_d_field_comparison.csv  Spearman/Pearson per region x source
  results/rev_d_field_intervals.json  D-d interval coverage / field-calibrated deduction
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import FEATURE_SETS, load_rev, fit_predict  # noqa: E402
from src.models.rev_uq import calibrate, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases, parse  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CCN = ROOT / "data" / "raw" / "validation_refs" / "ccn"
RES = ROOT / "results"
OM2OC = (0.00174, 0.3196, 2.330)  # D9, OC% from OM%
MAX_MATCH_M = 300.0
MODELS = ["xgb", "rf", "phys_gnn"]


def field_carbon():
    c = pd.read_csv(CCN / "CCN_cores.csv", low_memory=False)
    d = pd.read_csv(CCN / "CCN_depthseries.csv", low_memory=False)
    rows = []
    for rk, reg in REGIONS.items():
        b = reg.bbox
        sel = c[c.longitude.between(b[0], b[2]) & c.latitude.between(b[1], b[3]) & (c.habitat == "mangrove")]
        dd = d.merge(sel[["study_id", "core_id"]], on=["study_id", "core_id"])
        oc = dd.fraction_carbon * 100
        oc = oc.where(oc.notna(), np.polyval(OM2OC, dd.fraction_organic_matter * 100))
        dd = dd.assign(dens=dd.dry_bulk_density * oc / 100)  # g C / cm3
        for (sid, cid), g in dd.dropna(subset=["dens", "depth_min", "depth_max"]).groupby(["study_id", "core_id"]):
            g = g.sort_values("depth_min")
            top, bot, den = g.depth_min.values, g.depth_max.values, g.dens.values
            t30 = np.clip(np.minimum(bot, 30) - top, 0, None)
            sampled30 = t30.sum()
            deepest = bot.max()
            t100 = np.clip(np.minimum(bot, 100) - top, 0, None)
            mean_den = np.sum(den * t100) / max(t100.sum(), 1e-9)
            stock100 = (np.sum(den * t100) + mean_den * max(0, 100 - min(deepest, 100))) * 100  # g/cm2 -> t/ha
            core = sel[(sel.study_id == sid) & (sel.core_id == cid)].iloc[0]
            rows.append({"region": rk, "study_id": sid, "core_id": cid, "lat": core.latitude, "lon": core.longitude,
                         "dens_0_30": np.sum(den * t30) / sampled30 if sampled30 >= 15 else np.nan,
                         "stock_0_100_tha": stock100, "deepest_cm": deepest,
                         "stock_reliable": deepest >= 50})
    f = pd.DataFrame(rows)
    # D-a: average cores at identical coordinates into one location
    f["loc"] = f.lat.round(5).astype(str) + "," + f.lon.round(5).astype(str)
    agg = f.groupby(["region", "loc"]).agg(study_id=("study_id", "first"), n_cores=("core_id", "size"),
                                           lat=("lat", "mean"), lon=("lon", "mean"),
                                           dens_0_30=("dens_0_30", "mean"), stock_0_100_tha=("stock_0_100_tha", "mean"),
                                           deepest_cm=("deepest_cm", "max"),
                                           stock_reliable=("stock_reliable", "max")).reset_index()
    return f, agg


def oof_all_nodes(df, edges, sel, fam, seed=1):
    """5-fold spatial CV over ALL nodes (D-c): every node predicted by a model
    that never saw its fine-grid group. No selection, no metric on Sanderman labels."""
    _, fs, params = parse(sel[fam])
    cols = FEATURE_SETS[fs]
    g = df.phase1_superblock.values
    idx = np.arange(len(df))
    pred = np.full(len(df), np.nan)
    for k, (fit, ev) in enumerate(GroupKFold(5, shuffle=True, random_state=20261007).split(idx, groups=g)):
        stop = None
        if fam in ("gnn", "phys_gnn"):
            rng = np.random.default_rng(seed * 100 + k)
            ug = np.unique(g[fit])
            sg = rng.choice(ug, size=max(1, int(round(0.15 * len(ug)))), replace=False)
            stop, fit = fit[np.isin(g[fit], sg)], fit[~np.isin(g[fit], sg)]
        p = fit_predict(df, edges, fam, cols, fit, [ev], seed, params, stop_idx=stop)
        pred[ev] = p[ev]
    return pred


def main():
    sel, _ = selections()
    cores, locs = field_carbon()
    out, comp = [], []
    intervals = {}
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        L = locs[locs.region == rk].copy()
        dy = (df.lat.values[None, :] - L.lat.values[:, None]) * 111000.0
        dx = (df.lon.values[None, :] - L.lon.values[:, None]) * 111000.0 * np.cos(np.radians(L.lat.values[:, None]))
        dist = np.sqrt(dx ** 2 + dy ** 2)
        L["node"] = dist.argmin(1)
        L["match_m"] = dist.min(1)
        L = L[L.match_m <= MAX_MATCH_M].copy()
        nodes = L.node.values
        L["split"] = df.split.values[nodes]
        L["sanderman"] = df[TARGET_COL].values[nodes]
        L["radabaugh"] = df["radabaugh_bgc_mgha"].values[nodes]
        for fam in MODELS:
            L[f"pred_{fam}"] = oof_all_nodes(df, edges, sel, fam)[nodes]
            print(rk, fam, "oof done", flush=True)
        # D-d (1): CQR-GNN 90% intervals (Sanderman-calibrated), cores on val/test nodes only
        train, val, test = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
        kind, base = build_bases(df, edges, sel, train, [val, test])["CQR-GNN"]
        y = df[TARGET_COL].values.astype(float)
        vt = np.concatenate([val, test])
        lo, hi = calibrate(kind, base, y, val, vt, 0.90, df.superblock.values)
        lo_all, hi_all = np.full(len(df), np.nan), np.full(len(df), np.nan)
        lo_all[vt], hi_all[vt] = lo, hi
        L["cqr_lo90"], L["cqr_hi90"] = lo_all[nodes], hi_all[nodes]
        L["cqr_yhat"] = np.where(np.isfinite(lo_all[nodes]), base[nodes, QUANTILES.index(0.5)], np.nan)
        out.append(L)

        for target, mask in (("dens_0_30", L.dens_0_30.notna()), ("stock_0_100_tha", L.stock_reliable.astype(bool))):
            for src in ["sanderman", "radabaugh"] + [f"pred_{m}" for m in MODELS]:
                m = mask & L[src].notna() & L[target].notna()
                if src == "radabaugh":
                    m &= L.study_id != "Radabaugh_et_al_2023"  # D-c (ii)
                if m.sum() >= 5:
                    comp.append({"region": rk, "field_quantity": target, "source": src, "n_locations": int(m.sum()),
                                 "spearman": spearmanr(L.loc[m, target], L.loc[m, src]).statistic,
                                 "pearson": pearsonr(L.loc[m, target], L.loc[m, src])[0]})

    allL = pd.concat(out, ignore_index=True)
    allL.to_csv(RES / "rev_d_field_cores.csv", index=False)
    cores.to_csv(RES / "rev_d_field_cores_raw.csv", index=False)
    comp = pd.DataFrame(comp)
    comp.to_csv(RES / "rev_d_field_comparison.csv", index=False)

    # D-d (1) coverage of reliable field stocks by Sanderman-calibrated CQR intervals
    r = allL[allL.stock_reliable.astype(bool) & allL.cqr_lo90.notna()]
    res = {"n_locations_total": int(len(allL)), "n_reliable_stock": int(allL.stock_reliable.sum()),
           "cqr_sanderman_cal_coverage_of_field": {
               "n": int(len(r)),
               "coverage": float(np.mean((r.stock_0_100_tha >= r.cqr_lo90) & (r.stock_0_100_tha <= r.cqr_hi90)))
               if len(r) else None}}
    # D-d (2) field-calibrated relative half-width k, leave-one-study-out
    rel = allL[allL.stock_reliable.astype(bool)].copy()
    for fam in MODELS:
        e = (rel.stock_0_100_tha - rel[f"pred_{fam}"]).abs() / rel[f"pred_{fam}"]
        studies = rel.study_id.unique()
        cov, ks = [], []
        for s in studies:
            tr, te = e[rel.study_id != s], e[rel.study_id == s]
            if len(tr) < 5:
                continue
            n = len(tr)
            k = np.quantile(tr, min(np.ceil((n + 1) * 0.9) / n, 1.0))
            ks.append(k)
            cov += list(te <= k)
        n = len(e)
        k_all = float(np.quantile(e, min(np.ceil((n + 1) * 0.9) / n, 1.0))) if n else None
        res[f"field_cal_{fam}"] = {"n": int(n), "loso_coverage": float(np.mean(cov)) if cov else None,
                                   "k_all": k_all, "k_loso_median": float(np.median(ks)) if ks else None,
                                   "vm0033_deduction_field_cal": max(0.0, k_all - 0.20) if k_all is not None else None}
    (RES / "rev_d_field_intervals.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 200)
    print(allL.groupby("region").agg(n=("loc", "size"), match_m=("match_m", "median"),
                                     reliable=("stock_reliable", "sum")))
    print(comp.round(3).to_string(index=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
