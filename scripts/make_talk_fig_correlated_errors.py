"""Talk figure from real data: out-of-fold ensemble errors (spatial CV, train+val only, no test
labels) of one 2.5 km pseudo-project in R1, shown as observed and with signs randomized as the
independence assumption implies. Also reports, across all R1 blocks with >= 30 nodes, the ratio of
the actual project error to the error implied by independent node errors.

Output: paper/figures/talk_correlated_errors.png, results/talk_correlated_errors.json
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK = 15                      # a block whose ratio is close to the median
NAVY, PINK, GREEN, INK = "#1F3F8F", "#E8749A", "#5BAA7A", "#2B2F38"
CMAP = LinearSegmentedColormap.from_list("err", ["#3B6FB6", "#F4F6FA", "#E8749A"])

oof = pd.read_csv(ROOT / "results/rev_f1_cv_oof_r1.csv", index_col=0)
nodes = pd.read_csv(ROOT / "data/processed/graph_nodes.csv")
oof["r"] = oof[["ens|seed1", "ens|seed2", "ens|seed3"]].mean(axis=1) - oof["y"]
d = nodes.loc[oof.index, ["lon", "lat", "phase1_superblock"]].join(oof[["r", "y"]])

g = d.groupby("phase1_superblock")
t = pd.DataFrame({"n": g.size(),
                  "actual": g.apply(lambda s: abs(s.r.sum()) / s.y.sum()),
                  "independent": g.apply(lambda s: np.sqrt((s.r ** 2).sum()) / s.y.sum())})
t = t[t.n >= 30]
t["ratio"] = t.actual / t.independent
summary = {"blocks": int(len(t)), "median_ratio": float(t.ratio.median()),
           "share_ratio_above_1": float((t.ratio > 1).mean()), "example_block": BLOCK,
           "example": t.loc[BLOCK].to_dict()}

b = d[d.phase1_superblock == BLOCK].copy()
rng = np.random.default_rng(20261027)
b["r_ind"] = b.r.abs() * rng.choice([-1, 1], size=len(b))
lim = np.percentile(b.r.abs(), 95)
err_ind = abs(b.r_ind.sum()) / b.y.sum() * 100
err_act = abs(b.r.sum()) / b.y.sum() * 100
rss = t.loc[BLOCK, "independent"] * 100

fig = plt.figure(figsize=(12, 5.4))
top = max(err_act, err_ind, rss) * 1.2
panels = (("r_ind", "Same errors, signs independent", err_ind, True, 0.02),
          ("r", "Observed errors (real data)", err_act, False, 0.52))
for col, title, val, good, x0 in panels:
    ax = fig.add_axes([x0, 0.2, 0.33, 0.66])
    ax.scatter(b.lon, b.lat, c=b[col], cmap=CMAP, vmin=-lim, vmax=lim, s=30, marker="s", linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(1.8); sp.set_linestyle((0, (4, 2)))
    ax.set_title(title, fontsize=14, weight="bold", color=NAVY, pad=8)
    bx = fig.add_axes([x0 + 0.35, 0.2, 0.11, 0.66])
    bx.bar([0], [val], width=0.55, color=GREEN if good else PINK)
    bx.axhline(rss, color=GREEN, ls="--", lw=1.8, xmin=0.05, xmax=0.95)
    bx.text(-0.62, rss + top * 0.015, "independent\nlevel", fontsize=8.5, color="#2F6E48", va="bottom", ha="left")
    bx.text(0, val + top * 0.01, f"{val:.1f}%", ha="center", va="bottom", fontsize=13, weight="bold", color=INK)
    bx.set_ylim(0, top)
    bx.set_xlim(-0.65, 0.65)
    bx.axis("off")
    bx.set_title("project error", fontsize=10.5, color=INK, pad=8)
fig.text(0.5, 0.07, f"Real out-of-fold errors of the ensemble in one 2.5 km pseudo-project of R1 ({int(t.loc[BLOCK, 'n'])} nodes; blue: under-prediction, pink: over-prediction).",
         ha="center", fontsize=10.5, color="#5A6475")
fig.text(0.5, 0.025, f"Across {len(t)} such projects, the actual project error was a median {t.ratio.median():.1f} times the independent-error level.",
         ha="center", fontsize=11, weight="bold", color="#A23B62")
fig.savefig(ROOT / "paper/figures/talk_correlated_errors.png", dpi=220, transparent=True)
(ROOT / "results/talk_correlated_errors.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
