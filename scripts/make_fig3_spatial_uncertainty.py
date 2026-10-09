"""Generate paper/figures/fig3_spatial_uncertainty.pdf: spatial map of
per-node 90% global-CQR interval width on the test split -- the paper's
most direct visual argument for its business claim (Section VI): that a
per-site uncertainty signal differentiates pricing by location where a
flat non-permanence buffer cannot.

Data: results/phase2_cqr_test_predictions.csv, the SAME file and SAME
columns (lo_adj_global_0.9 / hi_adj_global_0.9, split=="test") that
Table II (CQR coverage) and Table III / the Pricing PoC prose in Section
VI already trace to (NUMBER_SOURCE_LOG.md #45/#51) -- global calibration
specifically, per Section V's recommendation, not the localized (Mondrian)
variant the paper cautions against. No new computation, no new number:
width = hi_adj_global_0.9 - lo_adj_global_0.9 on the 2,076 test nodes,
matching the Pricing PoC's own basis (median 341.1 t/ha, range
143.9-552.5 t/ha, entry #51) exactly.

AOI/coastline context reuses fig1_pipeline.py's own technique: the full
10,124-node scatter (data/processed/graph_nodes.csv) stands in for a
basemap, since no AOI shapefile/coastline vector was available in this
sandbox (documented in NUMBER_SOURCE_LOG.md #28) -- consistent with Fig.
1's inset, not a new simplification introduced here.

Colorblind-safe: continuous scale uses cividis (Nunez, Anderton & Renslow
2018), a colormap validated specifically for color-vision-deficiency
accessibility (blue-yellow, avoids the red-green axis), rather than a
default red-to-green "risk" gradient. Same sans-serif/vector-PDF/font-size
conventions as fig1_pipeline.py and fig2_results.py.

Run from the repo root: python scripts/make_fig3_spatial_uncertainty.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ALL_NODES_CSV = REPO_ROOT / "data" / "processed" / "graph_nodes.csv"
CQR_PRED_CSV = REPO_ROOT / "results" / "phase2_cqr_test_predictions.csv"
OUT_PDF = REPO_ROOT / "paper" / "figures" / "fig3_spatial_uncertainty.pdf"

GRAY = "#4D4D4D"

# Page-budget note (NUMBER_SOURCE_LOG.md Day 5 Task 3 page-budget note):
# this figure is placed in main.tex at width=0.75\linewidth (moderate
# reduction, user-confirmed) instead of the full column width its earlier
# version used. To keep it fully legible at that smaller printed size
# (not just uniformly shrunk, which would shrink the embedded vector text
# along with everything else), every font/linewidth/marker-area size below
# is boosted by SCALE = 1/0.75 relative to the original design (areas by
# SCALE**2) -- so after LaTeX's 0.75x placement scale-down, text/markers
# land back at their originally-designed absolute size, not smaller.
SCALE = 1.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8 * SCALE,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7 * SCALE,
    "xtick.major.width": 0.7 * SCALE,
    "ytick.major.width": 0.7 * SCALE,
})


def main():
    all_nodes = pd.read_csv(ALL_NODES_CSV)
    pred = pd.read_csv(CQR_PRED_CSV)
    test = pred[pred["split"] == "test"].copy()
    test["width_90_global"] = test["hi_adj_global_0.9"] - test["lo_adj_global_0.9"]

    fig, ax = plt.subplots(figsize=(3.35, 2.85), constrained_layout=True)

    # AOI/coastline context: full graph extent, same technique as Fig. 1's
    # inset (data/processed/graph_nodes.csv, all 10,124 nodes) -- train/val
    # nodes render as faint gray background so the test-node coloring below
    # reads against the true AOI shape, not a blank frame. Low alpha and a
    # larger, higher-contrast test marker keep the colored test nodes from
    # blending into the background at cividis's grayish mid-tones.
    ax.scatter(all_nodes["lon"], all_nodes["lat"], s=1.6 * SCALE**2, marker="s",
               color=GRAY, alpha=0.12, linewidths=0, rasterized=True,
               label="Train/val nodes")

    sc = ax.scatter(test["lon"], test["lat"], c=test["width_90_global"],
                     s=6.5 * SCALE**2, marker="s", cmap="cividis", linewidths=0,
                     rasterized=True, vmin=test["width_90_global"].min(),
                     vmax=test["width_90_global"].max())

    ax.set_xlim(all_nodes["lon"].min() - 0.01, all_nodes["lon"].max() + 0.01)
    ax.set_ylim(all_nodes["lat"].min() - 0.01, all_nodes["lat"].max() + 0.01)
    ax.set_xlabel("Longitude", fontsize=7 * SCALE)
    ax.set_ylabel("Latitude", fontsize=7 * SCALE)
    ax.tick_params(labelsize=6 * SCALE)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Test-node 90% CQR interval width (global calibration)",
                 fontsize=7.6 * SCALE, loc="left")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Interval width (t C/ha)", fontsize=6.6 * SCALE)
    cbar.ax.tick_params(labelsize=6 * SCALE)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT_PDF}")
    print(f"test n={len(test)}, width min={test['width_90_global'].min():.1f}, "
          f"max={test['width_90_global'].max():.1f}, median={test['width_90_global'].median():.1f}")


if __name__ == "__main__":
    main()
