"""
Revision W2: physics prior v2 (paper/revision_preregistration.md Addendum A1).
Same pool, folds, seeds and leakage-safe regime as run_rev_e1_cv.py; GNN
fixed at the §4 selection (F1_spectral, hidden 16). Test is never touched.

Grid: {P0, P1, P2} x mse x lambda {0.1, 0.3, 1.0}  +  P1 x mono x lambda {0.1, 1.0}
plus the no-physics GNN reference (re-run here so every row shares code).
Reported per config: OOF R2 (+ fold SE), MAE, Spearman, prior-gen corr
against ITS OWN proxy, and corr with the independent Radabaugh map.

Writes results/rev_w2_physics_cv_summary.csv, rev_w2_physics_cv_folds.csv,
rev_w2_selection.json.
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models.rev_common import load_rev, mass_balance_proxy  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e1_cv import folds, run_config, SEEDS  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
FS, HIDDEN = "F1_spectral", 16
PROXY_ORDER = {"P1": 0, "P2": 1, "P0": 2}  # A1 tie-break: prefer physically grounded priors


def main():
    t0 = time.time()
    df, edges = load_rev()
    pool = np.where(df["split"].isin(["train", "val"]).values)[0]
    fold_list = folds(df, pool)
    y = df[TARGET_COL].values[pool]
    rad = df["radabaugh_bgc_mgha"].values[pool]
    ok = np.isfinite(rad)

    configs = [("gnn", {"hidden": HIDDEN})]
    configs += [("phys_gnn", {"hidden": HIDDEN, "lambda": lam, "proxy": p, "constraint": "mse"})
                for p in ("P0", "P1", "P2") for lam in (0.1, 0.3, 1.0)]
    configs += [("phys_gnn", {"hidden": HIDDEN, "lambda": lam, "proxy": "P1", "constraint": "mono"})
                for lam in (0.1, 1.0)]

    rows, oof, out = [], {}, []
    for fam, params in configs:
        run_config(df, edges, pool, fold_list, fam, FS, params, SEEDS, rows, oof)
        key = rows[-1]["key"]
        g = pd.DataFrame([r for r in rows if r["key"] == key])
        kind = params.get("proxy", "P0")
        prox = mass_balance_proxy(df, pool, kind)[pool]
        r2s = [1 - np.mean((oof[(key, s)] - y) ** 2) / np.var(y) for s in SEEDS]
        out.append({"key": key, "family": fam, "proxy": kind, "constraint": params.get("constraint", "-"),
                    "lambda": params.get("lambda", 0.0),
                    "oof_r2": np.mean(r2s), "fold_r2_se": g.r2.std(ddof=1) / np.sqrt(len(g)),
                    "oof_mae": np.mean([np.mean(np.abs(oof[(key, s)] - y)) for s in SEEDS]),
                    "oof_spearman": g.spearman.mean(),
                    "prior_gen_corr_own_proxy": np.mean([np.corrcoef(oof[(key, s)], prox)[0, 1] for s in SEEDS]),
                    "corr_radabaugh": np.mean([np.corrcoef(oof[(key, s)][ok], rad[ok])[0, 1] for s in SEEDS])})
        print(f"[{time.time() - t0:5.0f}s] {key}  oof_r2={out[-1]['oof_r2']:.3f}", flush=True)

    summ = pd.DataFrame(out)
    summ.to_csv(RESULTS / "rev_w2_physics_cv_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "rev_w2_physics_cv_folds.csv", index=False)

    ph = summ[summ.family == "phys_gnn"].copy()
    best = ph.loc[ph.oof_r2.idxmax()]
    # A1 rule: highest OOF R2; the 1-SE tie-break only applies when the best
    # config uses the non-grounded control P0 (prefer a grounded prior then).
    # (D6: the first run's code applied the proxy order unconditionally.)
    if best.proxy == "P0":
        cand = ph[(ph.oof_r2 >= best.oof_r2 - best.fold_r2_se) & (ph.proxy != "P0")]
        chosen = cand.loc[cand.oof_r2.idxmax()] if len(cand) else best
    else:
        chosen = best
    base = summ[summ.family == "gnn"].iloc[0]
    sel = {"selected_phys": chosen.key, "phys_oof_r2": chosen.oof_r2,
           "no_physics_oof_r2": base.oof_r2, "no_physics_se": base.fold_r2_se,
           "physics_within_1se_of_no_physics": bool(chosen.oof_r2 >= base.oof_r2 - base.fold_r2_se),
           "physics_better_than_no_physics": bool(chosen.oof_r2 > base.oof_r2)}
    (RESULTS / "rev_w2_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(json.dumps(sel, indent=2, default=float))


if __name__ == "__main__":
    main()
