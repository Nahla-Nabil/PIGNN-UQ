"""
Addendum E: which VM0033 project-level aggregation is valid? VALIDATION ONLY
(test nodes are never predicted or read). Writes results/rev_e_project_aggregation.csv.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import load_rev  # noqa: E402
from src.models.rev_uq import calibrate, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
ALLOW = 0.20


def project_stats(df, idx, yhat, lo, hi, y):
    t = pd.DataFrame({"p": df.phase1_superblock.values[idx], "yhat": yhat, "half": (hi - lo) / 2, "y": y})
    g = t.groupby("p")
    out = pd.DataFrame({
        "n": g.size(),
        "e": g.apply(lambda s: abs((s.yhat - s.y).sum()) / s.y.sum()),
        "U_ind": g.apply(lambda s: np.sqrt(((s.half / s.yhat * s.yhat) ** 2).sum()) / s.yhat.sum()),
        "U_corr": g.apply(lambda s: s.half.sum() / s.yhat.sum()),
    })
    return out


def main():
    sel, _ = selections()
    rows = []
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        y = df[TARGET_COL].values.astype(float)
        train = np.where(df.split == "train")[0]
        val = np.where(df.split == "val")[0]
        perm = np.random.default_rng(7).permutation(val)
        cal, chk = perm[: len(perm) // 2], perm[len(perm) // 2:]
        kind, base = build_bases(df, edges, sel, train, [val])["CQR-GNN"]
        med = base[:, QUANTILES.index(0.5)]
        res = {}
        for name, idx in (("cal", cal), ("chk", chk)):
            lo, hi = calibrate(kind, base, y, cal, idx, 0.90, df.superblock.values)
            res[name] = project_stats(df, idx, med[idx], lo, hi, y[idx])
        pc, pk = res["cal"], res["chk"]
        n = len(pc)
        U_block = float(np.quantile(pc.e, min(np.ceil((n + 1) * 0.9) / n, 1.0)))
        pk = pk[pk.n >= 5]
        for meth, U in (("independent (VM0033 Eq.87/88)", pk.U_ind), ("fully correlated", pk.U_corr),
                        ("empirical project conformal", pd.Series(U_block, index=pk.index))):
            rows.append({"region": rk, "aggregation": meth, "n_projects": len(pk),
                         "median_nodes_per_project": float(pk.n.median()),
                         "coverage_of_realized_error": float(np.mean(pk.e <= U)),
                         "median_U": float(np.median(U)), "median_realized_error": float(pk.e.median()),
                         "implied_deduction_median": float(np.median(np.clip(U - ALLOW, 0, None)))})
        print(rk, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_e_project_aggregation.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
