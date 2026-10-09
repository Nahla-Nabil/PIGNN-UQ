"""Revision Fig. 1 (paper/figures/rev_fig1_results.pdf), three panels.

(a) R2 by model: region-1 spatial-CV out-of-fold (grey; whiskers = fold
    min-max where available) and test in R1 and R2 (mean ± SD over 3 seeds).
    The ensemble's test values come from the declared second test read
    (Addendum F5); all other test values from the single first read.
(b) Generalization, ensemble: own-region CV, one model for both regions
    (pooled CV, Addendum F4), and zero-shot transfer without positional
    features (XGBoost, first test read).
(c) 90% interval quality on test, global calibration: Winkler score per UQ
    method and region; dashed = trivial interval; coverage above each bar.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAMS = ["linear", "rf_coords", "rf", "xgb", "gnn", "phys_gnn", "ens"]
LABELS = {"linear": "Linear", "rf_coords": "RF (lon/lat)", "rf": "RF", "xgb": "XGBoost", "gnn": "GNN",
          "phys_gnn": "Phys-GNN", "ens": "Phys-GNN\n+XGBoost"}
UQ = ["CQR-GNN", "CQR-PhysGNN", "CQR-XGB", "NCP-RF", "Ens-Q"]
C = {"cv": "#999999", "r1": "#0072B2", "r2": "#E69F00", "pool": "#56B4E9", "zero": "#D55E00"}


def main():
    pts = pd.read_csv(RES / "rev_test_point_by_seed.csv")
    t2 = pd.read_csv(RES / "rev_test2_point_by_seed.csv")
    uq = pd.read_csv(RES / "rev_test_uq.csv")
    cv = pd.read_csv(RES / "rev_e1_cv_summary.csv")
    folds = pd.read_csv(RES / "rev_e1_cv_folds.csv")
    w2 = pd.read_csv(RES / "rev_w2_physics_cv_summary.csv")
    f1 = pd.read_csv(RES / "rev_f1_cv_summary.csv")
    f4 = pd.read_csv(RES / "rev_f4_pooled.csv")
    sel = json.loads((RES / "rev_e1_selection.json").read_text())
    phys_key = json.loads((RES / "rev_w2_selection.json").read_text())["selected_phys"]

    def cv_r1(f):
        if f == "ens":
            return f1[(f1.region == "r1") & (f1.cand == "ens")].oof_r2.iloc[0], np.nan, np.nan
        if f == "phys_gnn":
            return w2.loc[w2.key == phys_key, "oof_r2"].iloc[0], np.nan, np.nan
        k = sel[f]
        g = folds[folds.key == k].r2
        return cv.loc[cv.key == k, "oof_r2"].iloc[0], g.min(), g.max()

    def test(f, rk):
        if f == "ens":
            g = t2[t2.region == rk].test_r2
        else:
            g = pts[(pts["eval"] == f"{rk}_in_region") & (pts.family == f)].test_r2
        return g.mean(), (g.std() if len(g) > 1 else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1.45, 0.8, 1.0]})
    x = np.arange(len(FAMS))
    w = 0.27
    ax = axes[0]
    vals = [cv_r1(f) for f in FAMS]
    m = np.array([v[0] for v in vals])
    lo = np.array([v[0] - v[1] if np.isfinite(v[1]) else 0 for v in vals])
    hi = np.array([v[2] - v[0] if np.isfinite(v[2]) else 0 for v in vals])
    ax.bar(x - w, m, w, color=C["cv"], label="R1 CV", yerr=[lo, hi], error_kw={"lw": 0.6, "capsize": 1.5})
    for k, rk in enumerate(("r1", "r2")):
        tv = [test(f, rk) for f in FAMS]
        ax.bar(x + k * w, [a for a, _ in tv], w, color=C[rk], label=f"{rk.upper()} test",
               yerr=[b for _, b in tv], error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.85, 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAMS], rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel("$R^2$", fontsize=7)
    ax.set_title("(a) Accuracy by model", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right", ncol=3)
    ax.tick_params(labelsize=6)

    ax = axes[1]
    xr = np.arange(2)
    w = 0.26
    own = [f1[(f1.region == r) & (f1.cand == "ens")].oof_r2.iloc[0] for r in ("r1", "r2")]
    pooled = [f4[(f4.model == "ens") & (f4.region == r)]["mean"].iloc[0] for r in ("r1", "r2")]
    zero = [pts[(pts["eval"] == ev) & (pts.family == "xgb")].test_r2.mean()
            for ev in ("transfer_nopos_r2_to_r1", "transfer_nopos_r1_to_r2")]
    ax.bar(xr - w, own, w, color=C["r1"], label="own region")
    ax.bar(xr, pooled, w, color=C["pool"], label="one model, both")
    ax.bar(xr + w, zero, w, color=C["zero"], label="zero-shot")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xr)
    ax.set_xticklabels(["R1", "R2"], fontsize=6.5)
    ax.set_title("(b) Generalization", fontsize=7)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left")
    ax.set_ylim(-0.25, 0.75)
    ax.tick_params(labelsize=6)

    ax = axes[2]
    g = uq[(uq.level == 0.9) & (uq.calibration == "global")]
    xc = np.arange(len(UQ))
    w = 0.38
    for k, (r, col) in enumerate((("r1", C["r1"]), ("r2", C["r2"]))):
        sub = g[g.region == r].set_index("method")
        vals = sub.loc[UQ, "winkler"].values
        ax.bar(xc + (k - 0.5) * w, vals, w, color=col, label=r.upper())
        for xi, v, cov in zip(xc + (k - 0.5) * w, vals, sub.loc[UQ, "coverage"].values):
            ax.text(xi, v + 4, f"{100 * cov:.0f}", ha="center", fontsize=4.5, rotation=90)
        ax.axhline(sub.loc["Trivial", "winkler"], color=col, ls="--", lw=0.8)
    ax.set_ylim(0.8 * g[g.method.isin(UQ)].winkler.min(), 1.12 * g.winkler.max())
    ax.set_xticks(xc)
    ax.set_xticklabels(UQ, rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel("90% Winkler score (t C/ha)", fontsize=6.3)
    ax.set_title("(c) Interval quality (test)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2, title="dashed = trivial", title_fontsize=5)
    ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.3, w_pad=0.5)
    out = ROOT / "paper" / "figures" / "rev_fig1_results.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
