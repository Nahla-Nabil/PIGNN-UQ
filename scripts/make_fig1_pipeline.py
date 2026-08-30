"""Generate paper/figures/fig1_pipeline.pdf: method/pipeline schematic.

Boxes/arrows only, no decorative art. The inset AOI map is real data: a
scatter of the actual 10,124 graph node (lon, lat) coordinates, colored by
train/val/test split, read from data/processed/graph_nodes.csv -- the same
file build_graph.py produced and NUMBER_SOURCE_LOG.md entries #6-9 already
cite. Node/edge counts and data-source names in the flow boxes come from
NUMBER_SOURCE_LOG.md entry #7 (build_graph.py stdout) and Section III of
main.tex, not from any new computation.

Run from the repo root: python scripts/make_fig1_pipeline.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODES_CSV = REPO_ROOT / "data" / "processed" / "graph_nodes.csv"
OUT_PDF = REPO_ROOT / "paper" / "figures" / "fig1_pipeline.pdf"

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#4D4D4D"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,  # embed as real (Type 42) fonts, not Type 3 bitmaps
    "axes.linewidth": 0.7,
})

FLOW_BOXES = [
    dict(
        xy=(0.03, 0.36), w=0.16, h=0.42, color=SKY,
        lines=["Data sources", "Sentinel-2 L2A", "GMW v3 extent",
               "SOC (Sanderman et al.)", "NOAA tide gauge"],
    ),
    dict(
        xy=(0.225, 0.36), w=0.155, h=0.42, color=GREEN,
        lines=["Spatial graph", "10,124 nodes", "64,366 directed edges",
               "pilot AOI: SW Florida"],
    ),
    dict(
        xy=(0.415, 0.36), w=0.17, h=0.42, color=VERMILLION,
        lines=["GraphSAGE encoder", "+ physics-informed loss",
               "accretion (NDVI) /", "export (tide-gauge dist.) proxy"],
    ),
    dict(
        xy=(0.62, 0.36), w=0.155, h=0.42, color=BLUE,
        lines=["Split-conformal CQR", "global + localized", "(Mondrian) compared,", "calibrated on val"],
    ),
    dict(
        xy=(0.81, 0.36), w=0.16, h=0.42, color=ORANGE,
        lines=["Pricing PoC", "interval width", u"→ price", "multiplier"],
    ),
]


def draw_flow(ax):
    for box in FLOW_BOXES:
        x, y = box["xy"]
        w, h = box["w"], box["h"]
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.9, edgecolor="black",
            facecolor=box["color"], alpha=0.85,
            mutation_aspect=1,
        )
        ax.add_patch(patch)
        lines = box["lines"]
        title, rest = lines[0], lines[1:]
        ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top",
                 fontsize=7.6, fontweight="bold", color="black")
        rest_text = "\n".join(rest)
        ax.text(x + w / 2, y + h - 0.14, rest_text, ha="center", va="top",
                 fontsize=6.6, color="black", linespacing=1.5)

    # Arrows between consecutive boxes.
    for a, b in zip(FLOW_BOXES[:-1], FLOW_BOXES[1:]):
        x0 = a["xy"][0] + a["w"]
        x1 = b["xy"][0]
        yc = a["xy"][1] + a["h"] / 2
        arrow = FancyArrowPatch(
            (x0 + 0.004, yc), (x1 - 0.004, yc),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=GRAY,
        )
        ax.add_patch(arrow)


def draw_inset_map(ax):
    df = pd.read_csv(NODES_CSV)
    # Single color: this inset shows the AOI/graph extent only (the
    # train/val/test split itself is not this figure's subject).
    ax.scatter(df["lon"], df["lat"], s=0.7, marker="s",
                color=GRAY, linewidths=0, rasterized=True)
    ax.set_xlim(df["lon"].min() - 0.01, df["lon"].max() + 0.01)
    ax.set_ylim(df["lat"].min() - 0.01, df["lat"].max() + 0.01)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.set_aspect("equal")
    ax.set_title("Ten Thousand Is. /\nRookery Bay, FL", fontsize=6.2, pad=2)


def main():
    fig = plt.figure(figsize=(7.16, 2.15))
    ax_flow = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    draw_flow(ax_flow)

    # Small inset map anchored under the "Spatiotemporal graph" box,
    # connected by a short leader line to make clear it depicts that box's
    # graph nodes.
    ax_map = fig.add_axes([0.265, 0.02, 0.075, 0.225])
    draw_inset_map(ax_map)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
