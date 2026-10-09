"""Revision Fig. 1 (paper/figures/rev_fig1_results.pdf): three panels.

(a) R2 by model family: region-1 spatial-CV out-of-fold (selection data,
    bars = mean over seeds, whiskers = fold min-max) next to the single
    test evaluation in region 1 and region 2 (mean ± SD over 3 seeds).
(b) Cross-region transfer R2 without positional features (D8), both
    directions, against in-region test R2.
(c) 90% interval quality on test, global calibration: Winkler score (lower
    is better) per UQ method and region; the Trivial reference is drawn as
    a dashed line per region; coverage printed above each bar.

Usage: python scripts/make_rev_fig1.py [--prefix dryrun_]
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAMS = ["linear", "rf_coords", "rf", "rf_lag", "xgb", "gnn", "phys_gnn"]
LABELS = {"linear": "Linear", "rf_coords": "RF (lon/lat)", "rf": "RF", "rf_lag": "RF+lag", "xgb": "XGBoost",
          "gnn": "GNN", "phys_gnn": "Phys-GNN"}
UQ = ["CQR-GNN", "CQR-PhysGNN", "CQR-XGB", "NCP-RF", "Ens-Q"]
C = {"cv": "#999999", "r1": "#0072B2", "r2": "#E69F00", "t12": "#56B4E9", "t21": "#D55E00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    pre = ap.parse_args().prefix
    pts = pd.read_csv(RES / f"{pre}rev_test_point_by_seed.csv")
    uq = pd.read_csv(RES / f"{pre}rev_test_uq.csv")
    cv = pd.read_csv(RES / "rev_e1_cv_summary.csv")
    w2 = pd.read_csv(RES / "rev_w2_physics_cv_summary.csv")
    sel = json.loads((RES / "rev_e1_selection.json").read_text())
    sel["phys_gnn"] = json.loads((RES / "rev_w2_selection.json").read_text())["selected_phys"]
    cvall = pd.concat([cv[["key", "oof_r2", "fold_r2_min", "fold_r2_max"]],
                       w2[["key", "oof_r2"]].assign(fold_r2_min=np.nan, fold_r2_max=np.nan)])
    folds = pd.concat([pd.read_csv(RES / "rev_e1_cv_folds.csv"), pd.read_csv(RES / "rev_w2_physics_cv_folds.csv")])

    def test_stats(tag):
        g = pts[pts["eval"] == tag].groupby("family")["test_r2"]
        return g.mean(), g.std().fillna(0)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1.35, 1.1, 1.0]})
    x = np.arange(len(FAMS))

    # (a)
    ax = axes[0]
    w = 0.27
    cvm = [cvall.loc[cvall.key == sel[f], "oof_r2"].iloc[0] for f in FAMS]
    lo = [folds.loc[folds.key == sel[f], "r2"].min() for f in FAMS]
    hi = [folds.loc[folds.key == sel[f], "r2"].max() for f in FAMS]
    ax.bar(x - w, cvm, w, color=C["cv"], label="R1 CV (OOF)",
           yerr=[np.subtract(cvm, lo), np.subtract(hi, cvm)], error_kw={"lw": 0.6, "capsize": 1.5})
    for k, (tag, col, lab) in enumerate((("r1_in_region", C["r1"], "R1 test"), ("r2_in_region", C["r2"], "R2 test"))):
        m, s = test_stats(tag)
        ax.bar(x + k * w, m.reindex(FAMS), w, color=col, label=lab, yerr=s.reindex(FAMS),
               error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAMS], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("$R^2$", fontsize=7)
    ax.set_title("(a) In-region accuracy", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6)

    # (b)
    ax = axes[1]
    fams_b = [f for f in FAMS if f != "rf_coords"]
    xb = np.arange(len(fams_b))
    w = 0.2
    series = [("r1_in_region", C["r1"], "R1 in-region"), ("transfer_nopos_r2_to_r1", C["t21"], "R2→R1"),
              ("r2_in_region", C["r2"], "R2 in-region"), ("transfer_nopos_r1_to_r2", C["t12"], "R1→R2")]
    for k, (tag, col, lab) in enumerate(series):
        m, s = test_stats(tag)
        ax.bar(xb + (k - 1.5) * w, m.reindex(fams_b).clip(lower=-1.0), w, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-1.0, None)
    ax.set_xticks(xb)
    ax.set_xticklabels([LABELS[f] for f in fams_b], rotation=45, ha="right", fontsize=6)
    ax.set_title("(b) Cross-region transfer (test)", fontsize=7)
    ax.legend(fontsize=5.2, frameon=False, ncol=2, loc="lower left")
    ax.tick_params(labelsize=6)
    ax.text(0.99, 0.02, "bars clipped at $-1$", transform=ax.transAxes, ha="right", fontsize=5)

    # (c)
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
    ymin = 0.8 * g[g.method.isin(UQ)].winkler.min()
    ax.set_ylim(ymin, 1.12 * g.winkler.max())
    ax.set_xticks(xc)
    ax.set_xticklabels(UQ, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Winkler score, 90% (t C/ha)", fontsize=6.5)
    ax.set_title("(c) Interval quality (test)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2, title="dashed = trivial interval", title_fontsize=5)
    ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.3, w_pad=0.6)
    out = ROOT / "paper" / "figures" / f"{pre}rev_fig1_results.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
