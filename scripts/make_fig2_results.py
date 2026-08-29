"""Generate paper/figures/fig2_results.pdf: two-panel results figure.

(a) Grouped bar chart of Test R^2 and prior-generalization correlation
    (results/day2_model_comparison.csv columns `test_r2` and
    `physical_consistency_corr_test`) for the three compared models.
    NOTE ON NAMING: the source CSV column is `physical_consistency_corr_test`,
    but NUMBER_SOURCE_LOG.md entry #19 records that this label was corrected
    on 2026-08-29 -- the proxy correlated against is the physics loss's own
    training target, so this is prior-generalization, not independent
    physical validation. This script uses the corrected term
    ("prior-generalization corr.") in the figure, per that entry's
    instruction that "every prose reference must use the corrected framing,
    not the column name."
(b) Empirical vs. nominal CQR coverage (results/cqr_coverage.csv), with a
    dashed y=x reference line at perfect calibration.

Values are plotted at full CSV precision (no manual rounding beyond what
matplotlib's default tick formatting does).

Run from the repo root: python scripts/make_fig2_results.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPARISON_CSV = REPO_ROOT / "results" / "day2_model_comparison.csv"
COVERAGE_CSV = REPO_ROOT / "results" / "cqr_coverage.csv"
OUT_PDF = REPO_ROOT / "paper" / "figures" / "fig2_results.pdf"

# Okabe-Ito colorblind-safe palette (shared with fig1_pipeline.py).
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#4D4D4D"

MODEL_ORDER = ["linear_regression", "baseline_gnn_no_physics", "physics_gnn"]
MODEL_LABELS = {
    "linear_regression": "Linear regression",
    "baseline_gnn_no_physics": "Baseline GNN",
    "physics_gnn": "Physics-informed GNN",
}
MODEL_COLORS = {
    "linear_regression": SKY,
    "baseline_gnn_no_physics": GREEN,
    "physics_gnn": VERMILLION,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


def panel_a(ax):
    df = pd.read_csv(COMPARISON_CSV).set_index("model")
    metrics = ["test_r2", "physical_consistency_corr_test"]
    metric_labels = ["Test R²", "Prior-gen.\ncorr."]

    n_models = len(MODEL_ORDER)
    bar_w = 0.8 / n_models
    x = np.arange(len(metrics))

    for i, model in enumerate(MODEL_ORDER):
        vals = [df.loc[model, m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.92,
               color=MODEL_COLORS[model], edgecolor="black", linewidth=0.5,
               label=MODEL_LABELS[model])

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Value")
    ax.set_title("(a) Model comparison", fontsize=8, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left",
               handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    ax.set_ylim(-0.5, 0.85)


def panel_b(ax):
    df = pd.read_csv(COVERAGE_CSV)
    nominal_pct = df["nominal_level"] * 100
    empirical_pct = df["empirical_coverage_test"] * 100

    lims = (78, 100)
    ax.plot(lims, lims, linestyle="--", color=GRAY, linewidth=1.0,
            label="Perfect calibration", zorder=1)

    for _, row in df.iterrows():
        nom = row["nominal_level"] * 100
        emp = row["empirical_coverage_test"] * 100
        ax.plot([nom, nom], [emp, nom], color=VERMILLION, linewidth=1.0,
                 zorder=2)

    ax.scatter(nominal_pct, empirical_pct, s=32, color=VERMILLION,
               edgecolor="black", linewidth=0.5, zorder=3,
               label="Empirical (test)")

    for nom, emp in zip(nominal_pct, empirical_pct):
        ax.annotate(f"{emp:.1f}%", (nom, emp), textcoords="offset points",
                    xytext=(6, -9), fontsize=6.4, color="black")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(b) CQR coverage calibration", fontsize=8, loc="left")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right",
               handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 2.9),
                                       constrained_layout=True)
    panel_a(ax_a)
    panel_b(ax_b)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
