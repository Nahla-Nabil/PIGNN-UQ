"""Framework figure (paper/figures/rev_fig0_framework.pdf): data -> graph ->
models -> conformal intervals -> VM0033 deduction -> credited stock, with the
independent checks (field cores, registry records, mangrove loss) alongside.
Single-column IEEE width, TrueType fonts. Numbered steps read left to right,
then right to left; blue = learning, pink = crediting, green = evaluation only."""
import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "rev_fig0_framework.pdf"
W_IN, H_IN = 3.45, 1.95
BLUE = {"fill": ("#F3F8FE", "#D7E7F8"), "edge": "#3B6FB6", "title": "#1F3F8F"}
PINK = {"fill": ("#FEF4F8", "#F8D9E5"), "edge": "#E8749A", "title": "#A23B62"}
GREEN = {"fill": ("#EEF8F1", "#EEF8F1"), "edge": "#5BAA7A", "title": "#2F6E48"}
TEXT = "#2B2F38"
NAVY = "#1F3F8F"


def gradient(ax, patch, colors):
    """Fill a patch with a soft top-to-bottom gradient."""
    ax.add_patch(patch)
    x0, y0, w, h = patch.get_bbox().bounds
    im = ax.imshow(np.linspace(0, 1, 64)[:, None], cmap=LinearSegmentedColormap.from_list("g", colors),
                   extent=(x0, x0 + w, y0, y0 + h), origin="upper", aspect="auto",
                   zorder=patch.get_zorder() - 0.1, interpolation="bicubic")
    im.set_clip_path(patch)


def step(ax, n, x, y, w, h, title, body, style):
    gradient(ax, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                                fc="none", ec=style["edge"], lw=1.0, zorder=2), style["fill"])
    # numbered badge on the top-left corner
    r = 0.030
    ax.add_patch(Ellipse((x + 0.028, y + h - 0.045), 2 * r, 2 * r * W_IN / H_IN, fc=style["edge"], ec="white",
                         lw=0.8, zorder=4))
    ax.text(x + 0.028, y + h - 0.045, str(n), ha="center", va="center", fontsize=6.3, weight="bold",
            color="white", zorder=5)
    ax.text(x + 0.066, y + h - 0.045, title, ha="left", va="center", fontsize=7.3, weight="bold",
            color=style["title"], zorder=3)
    ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=6.6, color=TEXT,
            linespacing=1.22, zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=1.0,
                                 color=NAVY, shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.295, 0.36
    xs = (0.012, 0.3525, 0.693)
    y1, y2 = 0.615, 0.195

    step(ax, 1, xs[0], y1, w, h, "Inputs", "Sentinel-2, 2 dates\nlidar + tide gauges\nsoil carbon label", BLUE)
    step(ax, 2, xs[1], y1, w, h, "Spatial graph", "150 m nodes\n10,124 and 22,326\nno cross-split edge", BLUE)
    step(ax, 3, xs[2], y1, w, h, "Models", "Phys-GNN (Eq. 1)\nand XGBoost mean:\npoint estimates", BLUE)
    step(ax, 4, xs[2], y2, w, h, "90% intervals", "conformalized\nquantile GNN,\nvalidation-calibrated", PINK)
    step(ax, 5, xs[1], y2, w, h, "VM0033 rule", "$U_i=(u_i-l_i)/2\\hat y_i$\ndeduction\n$\\max(0,\\,U_i-0.2)$", PINK)
    step(ax, 6, xs[0], y2, w, h, "Credited stock", "$\\times\\,(1-b)$\nregistry buffer\nunchanged", PINK)

    ym1, ym2 = y1 + h / 2, y2 + h / 2
    arrow(ax, xs[0] + w + 0.004, ym1, xs[1] - 0.004, ym1)
    arrow(ax, xs[1] + w + 0.004, ym1, xs[2] - 0.004, ym1)
    arrow(ax, xs[2] + w / 2, y1 - 0.006, xs[2] + w / 2, y2 + h + 0.006)
    arrow(ax, xs[2] - 0.004, ym2, xs[1] + w + 0.004, ym2)
    arrow(ax, xs[1] - 0.004, ym2, xs[0] + w + 0.004, ym2)

    gradient(ax, FancyBboxPatch((0.012, 0.018), 0.976, 0.105, boxstyle="round,pad=0.004,rounding_size=0.03",
                                fc="none", ec=GREEN["edge"], lw=0.9, ls=(0, (3, 2)), zorder=2), GREEN["fill"])
    ax.text(0.5, 0.0705, "Evaluation only: 87 field cores · registry buffers · mangrove loss",
            ha="center", va="center", fontsize=6.5, style="italic", color=GREEN["title"], zorder=3)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
