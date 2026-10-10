"""Framework figure (paper/figures/rev_fig0_framework.pdf): data -> graph ->
models -> conformal intervals -> VM0033 deduction -> credited stock, with the
independent checks (field cores, registry records, mangrove loss) alongside.
Single-column IEEE width, TrueType fonts; gradient fills (blue = learning,
pink = crediting, yellow-green = independent checks)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "rev_fig0_framework.pdf"
BLUE = ("#EAF3FC", "#A9CBEF")
PINK = ("#FDEEF4", "#F2B3CB")
CHECK = ("#FBF0C2", "#C9E8C9")
DARK = "#2B2B2B"


def gradient(ax, patch, colors, vertical=True):
    """Fill a patch with a two-color linear gradient (top -> bottom or left -> right)."""
    ax.add_patch(patch)
    x0, y0, w, h = patch.get_bbox().bounds if hasattr(patch, "get_bbox") else (0, 0, 1, 1)
    ramp = np.linspace(0, 1, 64)
    img = ramp[:, None] if vertical else ramp[None, :]
    im = ax.imshow(img, cmap=LinearSegmentedColormap.from_list("g", colors), extent=(x0, x0 + w, y0, y0 + h),
                   origin="upper" if vertical else "lower", aspect="auto", zorder=patch.get_zorder() - 0.1,
                   interpolation="bicubic")
    im.set_clip_path(patch)


def box(ax, x, y, w, h, title, body, colors):
    gradient(ax, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc="none", ec=DARK, lw=0.6, zorder=2), colors)
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=7.6, weight="bold", color=DARK, zorder=3)
    ax.text(x + w / 2, y + h - 0.13, body, ha="center", va="top", fontsize=6.7, color=DARK, linespacing=1.2, zorder=3)


def arrow(ax, x0, y0, x1, y1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6, lw=0.7,
                                 color=DARK, linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(3.45, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # row 1: inputs -> graph -> models
    box(ax, 0.01, 0.56, 0.30, 0.42, "Inputs",
        "Sentinel-2, 2 dates\nlidar + NOAA tides\nSOC label (Maxwell)", BLUE)
    box(ax, 0.355, 0.56, 0.29, 0.42, "Spatial graph",
        "150 m nodes\n10,124 and 22,326\nno cross-split edge", BLUE)
    box(ax, 0.69, 0.56, 0.30, 0.42, "Models",
        "Phys-GNN (Eq. 1)\nand XGBoost,\nensemble by CV", BLUE)
    arrow(ax, 0.31, 0.77, 0.355, 0.77)
    arrow(ax, 0.645, 0.77, 0.69, 0.77)

    # row 2: intervals -> deduction -> credits
    box(ax, 0.69, 0.08, 0.30, 0.40, "90% intervals",
        "conformalized\nquantile regression", PINK)
    box(ax, 0.355, 0.08, 0.29, 0.40, "VM0033 rule",
        "$U_i=(u_i-l_i)/2\\hat y_i$\ndeduction\n$\\max(0,\\,U_i-0.2)$", PINK)
    box(ax, 0.01, 0.08, 0.30, 0.40, "Credited stock",
        "$\\times\\,(1-b)$, registry\nbuffer unchanged", PINK)
    arrow(ax, 0.84, 0.56, 0.84, 0.48)
    arrow(ax, 0.69, 0.28, 0.645, 0.28)
    arrow(ax, 0.355, 0.28, 0.31, 0.28)

    gradient(ax, FancyBboxPatch((0.008, 0.0), 0.984, 0.062, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc="none", ec=DARK, lw=0.5, zorder=2), CHECK, vertical=False)
    ax.text(0.5, 0.031, "Evaluation only: 87 field cores, registry buffers, mangrove loss",
            ha="center", va="center", fontsize=6.4, style="italic", color=DARK, zorder=3)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
