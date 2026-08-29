"""Generate paper/figures/fig2_results.pdf: two-panel results figure.

(a) Grouped bar chart of VAL R^2 and prior-generalization correlation for
    the three compared models, with mean +/- std ERROR BARS from the
    Phase 1A multi-seed runs (results/phase1a_multiseed_val.csv, n=5 seeds
    per model; linear regression's bar has no error bar -- it is
    deterministic, std=0 by construction, not simulated). This is a
    robustness-check figure: it shows whether Day 2's single-seed finding
    holds up, not a single point estimate presented as if noise-free.
(b) Empirical vs. nominal CQR coverage on the Phase 2 TEST split (the one
    real, final evaluation), GLOBAL vs. LOCALIZED (Mondrian) calibration
    side by side, with a dashed y=x perfect-calibration reference line.
    Per NUMBER_SOURCE_LOG.md #45, localized calibration does NOT improve on
    global here -- it undercovers more. That is shown plainly, not
    obscured: both methods' points are plotted, neither is hidden or
    rescaled to look better.

Run from the repo root: python scripts/make_fig2_results.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MULTISEED_CSV = REPO_ROOT / "results" / "phase1a_multiseed_val.csv"
COVERAGE_CSV = REPO_ROOT / "results" / "phase2_cqr_test_coverage.csv"
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
    df = pd.read_csv(MULTISEED_CSV)
    metrics = ["val_r2", "val_prior_gen_corr"]
    metric_labels = ["Val $R^2$\n(5-seed mean)", "Val prior-gen.\ncorr. (5-seed mean)"]

    n_models = len(MODEL_ORDER)
    bar_w = 0.8 / n_models
    x = np.arange(len(metrics))

    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model]
        means = [sub[m].mean() for m in metrics]
        # Linear regression is deterministic (closed-form OLS): std=0 by
        # construction, not simulated variance across "seeds" -- plotted
        # with no error bar rather than a fabricated one.
        stds = [sub[m].std(ddof=0) if model != "linear_regression" else 0.0 for m in metrics]
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, means, width=bar_w * 0.92,
               yerr=stds, capsize=2.2, error_kw=dict(linewidth=0.8, ecolor="black"),
               color=MODEL_COLORS[model], edgecolor="black", linewidth=0.5,
               label=MODEL_LABELS[model])

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Value")
    ax.set_title("(a) Model comparison, val (5 seeds)", fontsize=8, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.0, frameon=False, loc="upper left",
               handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    ax.set_ylim(-0.25, 0.95)


def panel_b(ax):
    df = pd.read_csv(COVERAGE_CSV)
    lims = (78, 100)
    ax.plot(lims, lims, linestyle="--", color=GRAY, linewidth=1.0,
            label="Perfect calibration", zorder=1)

    method_style = {
        "global": dict(color=BLUE, marker="o", label="Global (before)"),
        "localized_mondrian": dict(color=VERMILLION, marker="^", label="Localized/Mondrian (after)"),
    }
    for method, style in method_style.items():
        sub = df[df["calibration_method"] == method].sort_values("nominal_level")
        nom = sub["nominal_level"].values * 100
        emp = sub["empirical_coverage_test"].values * 100
        for n, e in zip(nom, emp):
            ax.plot([n, n], [e, n], color=style["color"], linewidth=0.8, zorder=2, alpha=0.6)
        ax.scatter(nom, emp, s=30, color=style["color"], marker=style["marker"],
                   edgecolor="black", linewidth=0.5, zorder=3, label=style["label"])
        for n, e in zip(nom, emp):
            ax.annotate(f"{e:.1f}%", (n, e), textcoords="offset points",
                        xytext=(6, -2 if style["marker"] == "o" else -12), fontsize=5.8, color=style["color"])

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(b) CQR coverage, test (global vs. localized)", fontsize=8, loc="left")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right",
               handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 2.05),
                                       constrained_layout=True)
    panel_a(ax_a)
    panel_b(ax_b)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
