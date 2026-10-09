"""
Addendum F3 (site-level field validation) and F4 (one model for both regions).
No test metric is computed and nothing is selected.
Writes results/rev_f3_field_sites.csv, rev_f3_field_sites_summary.csv,
rev_f4_pooled.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import load_rev, FEATURE_SETS  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.rev_f_common import predict, winner_name, inner_stop, RES  # noqa: E402
from scripts.run_revision_test_eval import combine  # noqa: E402
from scripts.run_rev_c23_transfer import region_z  # noqa: E402

MODELS = ["xgb", "phys_gnn"]


def oof_all(df, edges, name, seed=1, groups=None):
    g = df.phase1_superblock.values if groups is None else groups
    idx = np.arange(len(df))
    pred = np.full(len(df), np.nan)
    for fit, ev in GroupKFold(5, shuffle=True, random_state=20261007).split(idx, groups=g):
        f2, stop = inner_stop(df, fit, seed)
        pred[ev] = predict(df, edges, name, f2, [ev], seed, stop_idx=stop)[ev]
    return pred


def f3(win):
    cores = pd.read_csv(RES / "rev_d_field_cores.csv")  # D-a/D-b matched locations (node per location)
    raw = pd.read_csv(RES / "rev_d_field_cores_raw.csv")
    ccn = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "data/raw/validation_refs/ccn/CCN_cores.csv",
                      low_memory=False)[["study_id", "core_id", "site_id"]]
    raw = raw.merge(ccn, on=["study_id", "core_id"], how="left")
    raw["loc"] = raw.lat.round(5).astype(str) + "," + raw.lon.round(5).astype(str)
    site_of = raw.groupby(["region", "loc"]).site_id.first()
    cores["site_id"] = [site_of.get((r, l)) for r, l in zip(cores.region, cores["loc"])]
    cores = cores[cores.deepest_cm >= 15]
    names = sorted(set(MODELS + [win]))
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        m = cores.region == rk
        for n in names:
            cores.loc[m, f"pred_{n}"] = oof_all(df, edges, n)[cores.loc[m, "node"].values.astype(int)]
        cores.loc[m, "const"] = df.loc[df.split == "train", TARGET_COL].mean()
        print("F3", rk, "done", flush=True)
    srcs = ["sanderman", "radabaugh", "const"] + [f"pred_{n}" for n in names]
    site = cores.groupby(["region", "study_id", "site_id"]).agg(
        field=("stock_0_100_tha", "mean"), n_locations=("loc", "size"),
        **{s: (s, "mean") for s in srcs}).reset_index()
    site.to_csv(RES / "rev_f3_field_sites.csv", index=False)
    rows = []
    for scope, d in (("both", site), ("r1", site[site.region == "r1"]), ("r2", site[site.region == "r2"])):
        for s in srcs:
            ok = d[s].notna()
            rows.append({"scope": scope, "source": s, "n_sites": int(ok.sum()),
                         "spearman": spearmanr(d.field[ok], d[s][ok]).statistic if ok.sum() > 3 and s != "const" else np.nan,
                         "median_abs_pct_err": float((abs(d[s][ok] - d.field[ok]) / d.field[ok]).median() * 100)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_f3_field_sites_summary.csv", index=False)
    print(out.round(3).to_string(index=False))


def f4(win):
    data = {}
    for rk in ("r1", "r2"):
        df, e = load_rev(REGIONS[rk])
        cols = sorted(set(FEATURE_SETS["F3_spec_dem_temporal"]))
        keep = np.where(df.split != "test")[0]  # test never used
        pos = np.full(len(df), -1)
        pos[keep] = np.arange(len(keep))
        s, d = pos[e[0]], pos[e[1]]
        m = (s >= 0) & (d >= 0)
        data[rk] = (region_z(df.iloc[keep].reset_index(drop=True), cols), np.vstack([s[m], d[m]]))
    df, edges = combine(*data["r1"], *data["r2"])
    groups = (df.region + "_" + df.phase1_superblock.astype(str)).values
    y = df[TARGET_COL].values
    rows = []
    for n in sorted(set(["xgb", win])):
        for seed in (1, 2, 3):
            p = oof_all(df, edges, n, seed, groups=groups)
            for reg, lab in (("A", "r1"), ("B", "r2")):
                mm = df.region.values == reg
                rows.append({"model": n, "seed": seed, "region": lab,
                             "pooled_oof_r2": 1 - np.mean((p[mm] - y[mm]) ** 2) / np.var(y[mm])})
            print("F4", n, seed, flush=True)
    out = pd.DataFrame(rows).groupby(["model", "region"]).pooled_oof_r2.agg(["mean", "std"]).reset_index()
    out.to_csv(RES / "rev_f4_pooled.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    w = winner_name()
    f3(w)
    f4(w)
