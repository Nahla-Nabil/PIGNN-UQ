"""
Addendum C1: fix-only ablation (paper/revision_preregistration.md).

Same features (F0), models (linear, GNN hidden 16), spatial-CV folds and
seeds as §4; the ONLY difference between arms is the S2 feature source:
  (a) submitted     - mis-georeferenced columns from graph_nodes.csv
  (b) corrected     - graph_nodes_rev_features.csv
  (c) re-shifted    - corrected rasters shifted 9.18 km (the bug's offset)
                      before aggregation, i.e. the bug reproduced from
                      clean inputs; must match (a) closely.
No test node is used (pool = train ∪ val).
Writes results/rev_c1_fix_ablation.csv and rev_c1_bug_reproduction.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import R1  # noqa: E402
from src.data_pipeline.build_rev_features import _ref, _mosaic, _block, nd, STAC_URL  # noqa: E402
from src.models.rev_common import load_rev, F0  # noqa: E402
from scripts.run_rev_e1_cv import folds, run_config, SEEDS  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
SHIFT_M = 9180.29  # clip top (2909220.29) - tile top (2900040.0), metres
S2_COLS = ["ndvi", "b04", "b08", "b11"]


def reshifted_features(rows, cols):
    """Corrected T17RMJ+RMK mosaic on the SOC grid, shifted so each node
    receives the value from SHIFT_M further south (what the bug did)."""
    ref = _ref(R1)
    tr = ref[0]
    px_m = abs(tr.e) * 111_000.0  # SOC grid pixel height in metres (deg -> m)
    k = int(round(SHIFT_M / px_m))
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    s2 = cat.get_collection("sentinel-2-l2a")
    items = [s2.get_item(i) for i in R1.s2_items["t0"]]
    out = {}
    for band in ("B04", "B08", "B11"):
        arr = _mosaic(items, band, ref, bbox=(R1.bbox[0], R1.bbox[1] - 0.12, R1.bbox[2], R1.bbox[3]))
        sh = np.full_like(arr, np.nan)
        sh[:-k] = arr[k:]  # row r (north-up) takes the value k rows further south
        out[band.lower()] = _block(sh)[rows, cols]
    out["ndvi"] = nd(out["b08"], out["b04"])
    return out, k


def main():
    df_c, edges = load_rev()
    sub = pd.read_csv(R1.nodes_csv)
    pool = np.where(df_c["split"].isin(["train", "val"]).values)[0]
    fl = folds(df_c, pool)

    df_a = df_c.copy()
    for c in S2_COLS:
        df_a[c] = sub[c].values  # submitted (mis-georeferenced) values
    shifted, k = reshifted_features(df_c.row.values, df_c.col.values)
    df_s = df_c.copy()
    for c in S2_COLS:
        df_s[c] = shifted[c]
    ok = np.isfinite(df_s.b04.values)
    repro = {"shift_rows_on_soc_grid": k,
             "corr_reshifted_vs_submitted": {c: float(np.corrcoef(df_s[c].values[ok], sub[c].values[ok])[0, 1])
                                             for c in S2_COLS},
             "mean_ndvi": {"submitted": float(sub.ndvi.mean()), "corrected": float(df_c.ndvi.mean()),
                           "reshifted": float(np.nanmean(df_s.ndvi))},
             "frac_ndvi_below_0.3": {"submitted": float((sub.ndvi < 0.3).mean()),
                                     "corrected": float((df_c.ndvi < 0.3).mean()),
                                     "reshifted": float(np.nanmean(df_s.ndvi < 0.3))},
             "n_nodes_reshift_valid": int(ok.sum())}
    (RES / "rev_c1_bug_reproduction.json").write_text(json.dumps(repro, indent=2))
    print(json.dumps(repro, indent=2))
    df_s[S2_COLS] = df_s[S2_COLS].fillna(df_s[S2_COLS].median())

    rows = []
    for arm, d in (("a_submitted", df_a), ("b_corrected", df_c), ("c_reshifted", df_s)):
        for fam, params, seeds in (("linear", {}, [1]), ("gnn", {"hidden": 16}, SEEDS)):
            r, oof = [], {}
            run_config(d, edges, pool, fl, fam, "F0_base", params, seeds, r, oof)
            y = d.soc_tha_mean.values[pool]
            for s in seeds:
                p = oof[(r[0]["key"], s)]
                rows.append({"arm": arm, "family": fam, "seed": s,
                             "oof_r2": 1 - np.mean((p - y) ** 2) / np.var(y), "oof_mae": np.mean(np.abs(p - y))})
            for rr in r:
                rows.append({"arm": arm, "family": fam, "seed": rr["seed"], "fold": rr["fold"], "fold_r2": rr["r2"]})
            print(arm, fam, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_c1_fix_ablation.csv", index=False)
    s = out.dropna(subset=["oof_r2"]).groupby(["arm", "family"])[["oof_r2", "oof_mae"]].agg(["mean", "std"])
    print(s.round(3))
    f = out.dropna(subset=["fold_r2"]).pivot_table(index=["family", "seed", "fold"], columns="arm", values="fold_r2")
    f["diff_b_minus_a"] = f["b_corrected"] - f["a_submitted"]
    print("folds where corrected > submitted:", int((f.diff_b_minus_a > 0).sum()), "/", len(f),
          "; mean paired diff", round(f.diff_b_minus_a.mean(), 3))


if __name__ == "__main__":
    main()
