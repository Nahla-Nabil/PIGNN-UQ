"""Revision Fig. 2 (paper/figures/rev_fig2_deduction_map.pdf): per-node
VM0033 Eq. 92 uncertainty deduction (rule B) over the test nodes of both
regions, from the selected UQ method with global calibration.

Rebuilds exactly the intervals of run_revision_test_eval.py (same bases,
seed and val calibration). It needs NO test labels: the interval bounds
and the point prediction depend only on features and train/val labels.
Also writes results/rev_test_node_intervals_<region>.csv (label-free).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import load_rev  # noqa: E402
from src.models.rev_uq import calibrate, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from src.pricing.issuance import deduction_vm0033, node_uncertainty_pct  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "rev_fig2_deduction_map.pdf"


def node_intervals(rk, sel, uq_sel):
    df, edges = load_rev(REGIONS[rk])
    y = df[TARGET_COL].values  # only train/val entries are used below
    train, val, test = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
    kind, base = build_bases(df, edges, sel, train, [val, test])[uq_sel]
    y_masked = y.copy()
    y_masked[test] = np.nan  # guard: calibration must never see test labels
    lo, hi = calibrate(kind, base, y_masked, val, test, 0.90, df["superblock"].values, mondrian=False)
    yhat = base[test, QUANTILES.index(0.5)] if kind == "cqr" else base[0][test]
    ded = deduction_vm0033(node_uncertainty_pct(yhat, lo, hi))
    out = pd.DataFrame({"row": df.row.values[test], "col": df.col.values[test], "lon": df.lon.values[test],
                        "lat": df.lat.values[test], "yhat": yhat, "lo90": lo, "hi90": hi, "deduction_vm0033": ded})
    out.to_csv(RES / f"rev_test_node_intervals_{rk}.csv", index=False)
    return df, out


def main():
    sel = json.loads((RES / "rev_e1_selection.json").read_text())
    sel["phys_gnn"] = json.loads((RES / "rev_w2_selection.json").read_text())["selected_phys"]
    uq_sel = json.loads((RES / "rev_e6_uq_selection.json").read_text())["selected_uq"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    vmax = 0
    data = {}
    for rk in ("r1", "r2"):
        data[rk] = node_intervals(rk, sel, uq_sel)
        vmax = max(vmax, np.percentile(data[rk][1].deduction_vm0033, 98))
    for ax, rk, title in zip(axes, ("r1", "r2"), ("R1 Ten Thousand Is.", "R2 Everglades coast")):
        df, t = data[rk]
        ax.scatter(df.lon, df.lat, s=0.15, color="#4D4D4D", alpha=0.10, linewidths=0, rasterized=True)
        sc = ax.scatter(t.lon, t.lat, c=100 * t.deduction_vm0033, s=0.6, marker="s", cmap="cividis",
                        vmin=0, vmax=100 * vmax, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=6.5)
        ax.tick_params(labelsize=5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("VM0033 deduction (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    for rk in data:
        d = data[rk][1].deduction_vm0033
        print(rk, f"deduction median {100 * np.median(d):.1f}%, p90 {100 * np.percentile(d, 90):.1f}%, zero {np.mean(d == 0):.2f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
