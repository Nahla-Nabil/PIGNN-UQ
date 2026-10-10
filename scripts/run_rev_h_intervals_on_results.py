"""
Addendum H (exploratory, post hoc): sampling uncertainty for numbers already reported.
No model is refit and no test label is read again.

  H1  exact (Clopper-Pearson) 95% intervals for project-level coverage (validation data)
  H2  paired site bootstrap for field median-error differences (25 CCN sites)

Output: results/rev_h_intervals.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import beta

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
B = 10000


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    out = {"H1_project_coverage": [], "H2_field_sites": {}}
    agg = pd.read_csv(RES / "rev_e_project_aggregation.csv")
    for _, r in agg.iterrows():
        n = int(r.n_projects)
        k = int(round(r.coverage_of_realized_error * n))
        lo, hi = clopper_pearson(k, n)
        out["H1_project_coverage"].append({"region": r.region, "aggregation": r.aggregation, "k": k, "n": n,
                                           "coverage": k / n, "ci95": [lo, hi]})

    f = pd.read_csv(RES / "rev_f3_field_sites.csv")
    err = {c: (f[c] - f.field).abs() / f.field * 100 for c in ("pred_ens", "pred_phys_gnn", "pred_xgb", "sanderman", "radabaugh", "const")}
    rng = np.random.default_rng(20261010)
    idx = rng.integers(0, len(f), size=(B, len(f)))
    for other in ("sanderman", "radabaugh", "const", "pred_xgb", "pred_phys_gnn"):
        d = np.median(err["pred_ens"].values[idx], axis=1) - np.median(err[other].values[idx], axis=1)
        point = float(np.median(err["pred_ens"]) - np.median(err[other]))
        wins = float(np.mean(err["pred_ens"] < err[other]))
        out["H2_field_sites"][f"ens_minus_{other}"] = {
            "n_sites": int(len(f)), "median_diff_pct_points": point,
            "boot_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_sites_ensemble_closer": wins}
    (RES / "rev_h_intervals.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
