"""
Revision E1/E2/E4: spatial block CV model + feature selection
(paper/revision_preregistration.md §2-§4). Test nodes are never loaded into
any fit or metric here: the pool is train ∪ val only.

Stage 1: linear, RF (4 settings), XGB (2), coords-only RF, GNN (hidden 16/64),
         each x 4 feature sets, 3 seeds x 5 grouped folds.
Stage 2: rf_lag at the selected RF setting (x 4 feature sets); phys_gnn at
         the selected GNN (hidden, feature set), lambda in {0.1, 0.3, 1.0}.

Writes results/rev_e1_cv_folds.csv (every fold x seed x config),
results/rev_e1_cv_summary.csv (per-config pooled OOF metrics) and
results/rev_e1_cv_oof.csv (pooled OOF predictions per config/seed).
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import (FEATURE_SETS, FS_ORDER, load_rev, fit_predict, metrics,  # noqa: E402
                                   mass_balance_proxy)
from src.models.gnn_common import TARGET_COL  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CV_SEED, N_FOLDS, SEEDS, INNER_FRAC = 20261007, 5, [1, 2, 3], 0.15
RF_GRID = [{"max_features": mf, "min_samples_leaf": ml} for mf in (0.33, 1.0) for ml in (5, 20)]
XGB_GRID = [{"max_depth": d} for d in (3, 6)]
GNN_GRID = [{"hidden": h} for h in (16, 64)]
LAMBDAS = [0.1, 0.3, 1.0]
COMPLEXITY = {"linear": 0, "rf_coords": 0, "xgb": 1, "rf": 1, "rf_lag": 2, "gnn": 2, "phys_gnn": 2}


def folds(df, pool):
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    return [(pool[a], pool[b]) for a, b in gkf.split(pool, groups=df["phase1_superblock"].values[pool])]


def run_config(df, edges, pool, fold_list, family, fs, params, seeds, rows, oof_store):
    cols = ["lon", "lat"] if family == "rf_coords" else FEATURE_SETS[fs]
    y = df[TARGET_COL].values
    key = f"{family}|{fs}|{json.dumps(params, sort_keys=True)}"
    for seed in seeds:
        oof = np.full(len(df), np.nan)
        for k, (fit, ev) in enumerate(fold_list):
            stop = None
            if family in ("gnn", "phys_gnn"):
                rng = np.random.default_rng(seed * 100 + k)
                g = df["phase1_superblock"].values[fit]
                ug = np.unique(g)
                stop_g = rng.choice(ug, size=max(1, int(round(INNER_FRAC * len(ug)))), replace=False)
                stop, fit = fit[np.isin(g, stop_g)], fit[~np.isin(g, stop_g)]
            p = fit_predict(df, edges, family, cols, fit, [ev], seed, params, stop_idx=stop)
            oof[ev] = p[ev]
            m = metrics(y[ev], p[ev], y[fit].mean())
            rows.append({"key": key, "family": family, "feature_set": fs, "params": json.dumps(params),
                         "seed": seed, "fold": k, **m})
        oof_store[(key, seed)] = oof[pool]


def summarize(df, pool, fold_list, rows, oof_store, proxy):
    y = df[TARGET_COL].values[pool]
    fr = pd.DataFrame(rows)
    out = []
    for key, g in fr.groupby("key"):
        seeds = sorted(g.seed.unique())
        pooled = [metrics(y, oof_store[(key, s)], y.mean()) for s in seeds]
        pg = [np.corrcoef(oof_store[(key, s)], proxy[pool])[0, 1] for s in seeds]
        out.append({"key": key, "family": g.family.iloc[0], "feature_set": g.feature_set.iloc[0],
                    "params": g.params.iloc[0], "n_seeds": len(seeds),
                    "oof_r2": np.mean([m["r2"] for m in pooled]),
                    "oof_r2_seed_sd": np.std([m["r2"] for m in pooled]),
                    "fold_r2_mean": g.r2.mean(), "fold_r2_se": g.r2.std(ddof=1) / np.sqrt(len(g)),
                    "fold_r2_min": g.r2.min(), "fold_r2_max": g.r2.max(),
                    "oof_mae": np.mean([m["mae"] for m in pooled]),
                    "oof_rmse": np.mean([m["rmse"] for m in pooled]),
                    "oof_spearman": np.mean([m["spearman"] for m in pooled]),
                    "oof_skill_vs_mean": np.mean([m["skill_vs_mean"] for m in pooled]),
                    "oof_prior_gen_corr": np.mean(pg)})
    return pd.DataFrame(out)


def select(summary, family):
    """§4 rule: best mean OOF R2; among configs within 1 SE (fold x seed),
    the simplest (feature-set order, then smaller model)."""
    s = summary[summary.family == family].copy()
    best = s.loc[s.oof_r2.idxmax()]
    cand = s[s.oof_r2 >= best.oof_r2 - best.fold_r2_se].copy()
    cand["fs_rank"] = cand.feature_set.map({f: i for i, f in enumerate(FS_ORDER)})
    cand["size"] = cand.params.map(lambda p: sum(v for v in json.loads(p).values() if isinstance(v, (int, float))
                                                and not isinstance(v, bool)) if family != "rf" else
                                   (json.loads(p)["max_features"], -json.loads(p)["min_samples_leaf"]))
    return cand.sort_values(["fs_rank", "size"]).iloc[0]


def main():
    t0 = time.time()
    df, edges = load_rev()
    pool = np.where(df["split"].isin(["train", "val"]).values)[0]
    fold_list = folds(df, pool)
    print("fold sizes:", [len(e) for _, e in fold_list])
    proxy = mass_balance_proxy(df, pool)
    rows, oof = [], {}

    jobs = [("linear", fs, {}, [1]) for fs in FEATURE_SETS]
    jobs += [("rf_coords", "coords", {"max_features": 1.0, "min_samples_leaf": 5}, SEEDS)]
    jobs += [("xgb", fs, p, SEEDS) for fs in FEATURE_SETS for p in XGB_GRID]
    jobs += [("rf", fs, p, SEEDS) for fs in FEATURE_SETS for p in RF_GRID]
    jobs += [("gnn", fs, p, SEEDS) for fs in FEATURE_SETS for p in GNN_GRID]
    for fam, fs, p, seeds in jobs:
        run_config(df, edges, pool, fold_list, fam, fs, p, seeds, rows, oof)
        print(f"[{time.time() - t0:6.0f}s] {fam:9s} {fs:22s} {p}", flush=True)

    summary = summarize(df, pool, fold_list, rows, oof, proxy)
    best_rf, best_gnn = select(summary, "rf"), select(summary, "gnn")
    print("selected RF:", best_rf.key, " selected GNN:", best_gnn.key)

    for fs in FEATURE_SETS:
        run_config(df, edges, pool, fold_list, "rf_lag", fs, json.loads(best_rf.params), SEEDS, rows, oof)
        print(f"[{time.time() - t0:6.0f}s] rf_lag {fs}", flush=True)
    for lam in LAMBDAS:
        p = {**json.loads(best_gnn.params), "lambda": lam}
        run_config(df, edges, pool, fold_list, "phys_gnn", best_gnn.feature_set, p, SEEDS, rows, oof)
        print(f"[{time.time() - t0:6.0f}s] phys_gnn lambda={lam}", flush=True)

    summary = summarize(df, pool, fold_list, rows, oof, proxy)
    pd.DataFrame(rows).to_csv(RESULTS / "rev_e1_cv_folds.csv", index=False)
    summary.to_csv(RESULTS / "rev_e1_cv_summary.csv", index=False)
    pd.DataFrame({f"{k}|seed{s}": v for (k, s), v in oof.items()}, index=pool).assign(
        y=df[TARGET_COL].values[pool]).to_csv(RESULTS / "rev_e1_cv_oof.csv")

    # selections (§4)
    sel = {}
    for fam in ("linear", "rf", "xgb", "rf_lag", "gnn"):
        sel[fam] = select(summary, fam).key
    ph = summary[summary.family == "phys_gnn"]
    ok = ph[ph.oof_r2 >= ph.oof_r2.max() - 0.05]
    sel["phys_gnn"] = ok.loc[ok.oof_prior_gen_corr.idxmax()].key
    sel["rf_coords"] = summary[summary.family == "rf_coords"].key.iloc[0]
    (RESULTS / "rev_e1_selection.json").write_text(json.dumps(sel, indent=2))
    pd.set_option("display.width", 250)
    print(summary.sort_values("oof_r2", ascending=False)[
        ["key", "oof_r2", "fold_r2_se", "fold_r2_min", "fold_r2_max", "oof_mae", "oof_spearman",
         "oof_skill_vs_mean", "oof_prior_gen_corr"]].to_string(index=False))
    print(json.dumps(sel, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
