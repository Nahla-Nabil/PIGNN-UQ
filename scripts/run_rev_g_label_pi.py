"""
Addendum G (exploratory, post hoc): the label product's own 95% prediction
intervals (Maxwell et al. 2023, Zenodo 7729492, lower/upper layers) checked
against the CCN field cores and converted to the VM0033 deduction (Eq. 2).
No test labels and no model fitting are involved.

Output: results/rev_g_label_pi.json
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NODES = {"r1": ROOT / "data" / "processed" / "graph_nodes.csv",
         "r2": ROOT / "data" / "processed" / "region2" / "graph_nodes.csv"}
Z90_OVER_Z95 = 1.6449 / 1.9600  # rescale a 95% half-width to 90% under a normal approximation


def main():
    f = pd.read_csv(RES / "rev_d_field_cores.csv")
    nodes = {r: pd.read_csv(p) for r, p in NODES.items()}
    for col, src in (("lo", "soc_tha_lower_std"), ("hi", "soc_tha_upper_std"), ("mean", "soc_tha_mean")):
        f[col] = [nodes[r][src].iloc[int(n)] for r, n in zip(f.region, f.node)]
    inside = (f.stock_0_100_tha >= f.lo) & (f.stock_0_100_tha <= f.hi)
    out = {"field_coverage_all": {"n": int(len(f)), "coverage": float(inside.mean())},
           "field_coverage_reliable": {"n": int(f.stock_reliable.sum()),
                                       "coverage": float(inside[f.stock_reliable].mean())}}
    for r, d in nodes.items():
        u95 = (d.soc_tha_upper_std - d.soc_tha_lower_std) / (2 * d.soc_tha_mean)
        u90 = u95 * Z90_OVER_Z95
        out[r] = {"median_rel_halfwidth_95": float(np.median(u95)),
                  "median_rel_halfwidth_90_normal": float(np.median(u90)),
                  "median_vm0033_deduction": float(np.median(np.clip(u90 - 0.20, 0, 1)))}
    (RES / "rev_g_label_pi.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
