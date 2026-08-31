"""PREVIEW ONLY -- not wired into paper/main.tex. Generates a candidate
Fig. 4: a visual computational-graph diagram of Eq. 1 (the physics loss,
Section III-D), so the physical constraint's structure is shown, not
just stated as an inline/display equation.

Two tiers, top to bottom:
  (1) How the steady-state mass-balance proxy is built: NDVI (accretion
      proxy) and distance-to-tide-gauge (export proxy) combined by an
      equal-weight, min-max-normalized average -- no fitted coefficients
      (Section III-D, NUMBER_SOURCE_LOG.md).
  (2) How that proxy enters the loss:
      L_phys = lambda * MSE(y_hat, proxy) + w_nonneg * mean(ReLU(-y_hat_raw)^2)
      -- the two terms computed in parallel lanes and summed.

Every box label is text already in main.tex's Eq. 1/Section III-D; no
new claim or number is introduced by this figure.

Run from the repo root: python scripts/make_fig4_physics_loss_schematic.py
Output: paper/figures/fig4_physics_loss_schematic_PREVIEW.pdf (and a
        PNG render for quick viewing) -- deliberately named _PREVIEW so
        it is not mistaken for a figure main.tex already includes.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PDF = REPO_ROOT / "paper" / "figures" / "fig4_physics_loss_schematic_PREVIEW.pdf"
OUT_PNG = REPO_ROOT / "paper" / "figures" / "fig4_physics_loss_schematic_PREVIEW.png"

# Okabe-Ito colorblind-safe palette (shared with the paper's other figures).
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#4D4D4D"
LIGHTGRAY = "#DDDDDD"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
})


def box(ax, x, y, w, h, text, color, fontsize=8.2, fontweight="normal", text_color="black"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9, edgecolor="black", facecolor=color, alpha=0.9,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=text_color,
             linespacing=1.35)
    return (x, y, w, h)


def op_circle(ax, x, y, r, text, color=GRAY):
    c = Circle((x, y), r, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, text, ha="center", va="center", fontsize=9,
             fontweight="bold", color=color, zorder=6)


def arrow(ax, p0, p1, color=GRAY, lw=1.1, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                         linewidth=lw, color=color, shrinkA=2, shrinkB=2, zorder=3)
    ax.add_patch(a)


def right_mid(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left_mid(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top_mid(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom_mid(b):
    x, y, w, h = b
    return (x + w / 2, y)


def main():
    fig, ax = plt.subplots(figsize=(7.16, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Tier labels ----
    ax.text(0.01, 0.975, "How the physical prior is built", fontsize=9.5,
             fontweight="bold", ha="left", va="top", color=GRAY)
    ax.text(0.01, 0.60, r"How the proxy enters the loss  ($\mathcal{L}_{\mathrm{phys}}$, Eq. 1)",
             fontsize=9.5, fontweight="bold", ha="left", va="top", color=GRAY)

    # ---- Tier 1: proxy construction ----
    b_ndvi = box(ax, 0.02, 0.83, 0.20, 0.09, "NDVI\n(accretion proxy)", SKY)
    b_dist = box(ax, 0.02, 0.70, 0.20, 0.09, "Dist. to tide gauge\n(export proxy)", SKY)
    b_norm = box(ax, 0.30, 0.765, 0.20, 0.10,
                 "Equal-weight,\nmin-max normalized\n(train data only)", GREEN, fontsize=7.6)
    b_proxy = box(ax, 0.58, 0.765, 0.20, 0.10,
                  "Steady-state\nmass-balance proxy", ORANGE, fontweight="bold")

    arrow(ax, right_mid(b_ndvi), (0.30, 0.855))
    arrow(ax, right_mid(b_dist), (0.30, 0.775))
    arrow(ax, right_mid(b_norm), left_mid(b_proxy))

    # No fitted coefficients note.
    ax.text(0.40, 0.735, "no fitted coefficients", fontsize=6.6,
             style="italic", ha="center", color=GRAY)

    # Connector from proxy down into tier 2.
    arrow(ax, bottom_mid(b_proxy), (0.68, 0.50), color=GRAY)

    # ---- Tier 2: loss computation ----
    # Left lane: data-fit term.
    b_yhat = box(ax, 0.02, 0.44, 0.14, 0.085, r"$\hat y$" + "\n(model pred.)", SKY, fontsize=8.5)
    b_proxy2 = box(ax, 0.02, 0.32, 0.14, 0.085, "proxy\n(from above)", ORANGE, fontsize=8.5)
    b_mse = box(ax, 0.24, 0.375, 0.18, 0.085, r"MSE$(\hat y,\,\mathrm{proxy})$", GREEN, fontsize=8.2)
    arrow(ax, right_mid(b_yhat), (0.24, 0.46))
    arrow(ax, right_mid(b_proxy2), (0.24, 0.40))

    op_circle(ax, 0.475, 0.4175, 0.022, r"$\times\lambda$", color=VERMILLION)
    arrow(ax, right_mid(b_mse), (0.453, 0.4175))

    # Right lane: non-negativity penalty.
    b_yraw = box(ax, 0.02, 0.18, 0.14, 0.085, r"$\hat y_{\mathrm{raw}}$" + "\n(pre-clip pred.)", SKY, fontsize=8.5)
    b_relu = box(ax, 0.24, 0.18, 0.18, 0.085, r"mean$\!\left(\mathrm{ReLU}(-\hat y_{\mathrm{raw}})^2\right)$",
                 GREEN, fontsize=7.4)
    arrow(ax, right_mid(b_yraw), left_mid(b_relu))

    op_circle(ax, 0.475, 0.2225, 0.022, r"$\times w$", color=VERMILLION)
    arrow(ax, right_mid(b_relu), (0.453, 0.2225))

    # Sum node.
    op_circle(ax, 0.565, 0.32, 0.026, "+", color=GRAY)
    arrow(ax, (0.497, 0.4175), (0.549, 0.334), color=VERMILLION)
    arrow(ax, (0.497, 0.2225), (0.549, 0.306), color=VERMILLION)

    b_total = box(ax, 0.63, 0.275, 0.20, 0.09, r"$\mathcal{L}_{\mathrm{phys}}$",
                  VERMILLION, fontsize=13, fontweight="bold", text_color="white")
    arrow(ax, (0.591, 0.32), left_mid(b_total), color=VERMILLION, lw=1.4)

    ax.text(0.63, 0.235,
            r"added to the standard MSE data loss, $\mathcal{L}_{\mathrm{data}}$" + "\n"
            r"(not shown here -- see Section III-D)",
            fontsize=6.8, ha="left", va="top", color=GRAY, linespacing=1.3)

    fig.tight_layout(pad=0.3)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PNG, format="png", dpi=220, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
