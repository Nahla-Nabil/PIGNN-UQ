"""Framework figure (paper/figures/rev_fig0_framework.pdf): data -> graph ->
models -> conformal intervals -> VM0033 deduction -> credited stock, with the
independent checks (field cores, registry records, mangrove loss) alongside.
Single-column IEEE width, TrueType fonts."""
import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "rev_fig0_framework.pdf"
BLUE, ORANGE, GREEN, GREY, DARK = "#D6E6F5", "#FBE3C4", "#D9EFD9", "#ECECEC", "#333333"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=color, ec=DARK, lw=0.6))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=6.3, weight="bold", color=DARK)
    ax.text(x + w / 2, y + h - 0.125, body, ha="center", va="top", fontsize=5.2, color=DARK, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6, lw=0.7,
                                 color=DARK, linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(3.45, 1.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # row 1: inputs -> graph -> models
    box(ax, 0.01, 0.56, 0.30, 0.42, "Inputs",
        "Sentinel-2 (Feb, Aug 2026)\n3DEP lidar + NOAA tides\nSOC label (Sanderman)\nGMW mangrove mask", BLUE)
    box(ax, 0.355, 0.56, 0.29, 0.42, "Spatial graph",
        "150 m nodes, 8 neighbors\nR1: 10,124  R2: 22,326\nsplit-induced subgraphs\n(no cross-split edge)", BLUE)
    box(ax, 0.69, 0.56, 0.30, 0.42, "Models",
        "Phys-GNN (hydroperiod\nprior, Eq. 1) + XGBoost\n= ensemble\nselected by spatial CV", BLUE)
    arrow(ax, 0.31, 0.77, 0.355, 0.77)
    arrow(ax, 0.645, 0.77, 0.69, 0.77)

    # row 2: intervals -> deduction -> credits
    box(ax, 0.69, 0.08, 0.30, 0.40, "90% intervals",
        "conformalized quantile\nregression, global or\nMondrian calibration", ORANGE)
    box(ax, 0.355, 0.08, 0.29, 0.40, "VM0033 deduction",
        "$U_i=(u_i-l_i)/2\\hat y_i$\ndeduct $\\max(0,U_i-0.20)$\n(Eq. 2)", ORANGE)
    box(ax, 0.01, 0.08, 0.30, 0.40, "Credited stock",
        "$\\times\\,(1-b)$: registry\nnon-permanence buffer,\napplied unchanged", ORANGE)
    arrow(ax, 0.84, 0.56, 0.84, 0.48)
    arrow(ax, 0.69, 0.28, 0.645, 0.28)
    arrow(ax, 0.355, 0.28, 0.31, 0.28)

    ax.text(0.5, 0.025, "Independent checks: 87 field cores (CCN), registry buffers (VROD), 2010-2020 mangrove loss",
            ha="center", va="center", fontsize=5.0, style="italic", color=DARK,
            bbox=dict(boxstyle="round,pad=0.25", fc=GREEN, ec=DARK, lw=0.5))
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
