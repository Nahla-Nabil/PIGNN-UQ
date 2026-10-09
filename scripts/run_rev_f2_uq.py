"""
Addendum F2: intervals for the F1 winner, VALIDATION ONLY (val-internal
50/50 split, seed 7). Candidates: selected CQR-GNN, and NCP-winner = winner
prediction ± Q·s(x), s(x) = CQR-GNN raw 90% half-width. Trivial reported as
reference. Rule: lowest mean 90% Winkler over the two regions among methods
with val-check coverage ≥ 88% in both regions.
Writes results/rev_f2_uq_val.csv and rev_f2_selection.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import load_rev  # noqa: E402
from src.models.rev_uq import calibrate, interval_metrics, QUANTILES  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402
from scripts.rev_f_common import predict, winner_name, inner_stop, RES  # noqa: E402


def winner_ncp_base(df, edges, win, train, groups, cqr_base, seed=1):
    fit, stop = inner_stop(df, train, seed)  # val kept for calibration only
    yhat = predict(df, edges, win, fit, groups, seed, stop_idx=stop)
    qi = {q: i for i, q in enumerate(QUANTILES)}
    s = np.maximum((cqr_base[:, qi[0.95]] - cqr_base[:, qi[0.05]]) / 2, 1e-6)
    return yhat, s


def main():
    sel, _ = selections()
    win = winner_name()
    rows = []
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        y = df[TARGET_COL].values.astype(float)
        train = np.where(df.split == "train")[0]
        val = np.where(df.split == "val")[0]
        perm = np.random.default_rng(7).permutation(val)
        cal, chk = perm[: len(perm) // 2], perm[len(perm) // 2:]
        bases = build_bases(df, edges, sel, train, [val])
        kind, cqr = bases["CQR-GNN"]
        yhat, s = winner_ncp_base(df, edges, win, train, [val], cqr)
        methods = {"CQR-GNN": ("cqr", cqr), f"NCP-{win}": ("ncp", (yhat, s)), "Trivial": bases["Trivial"]}
        for name, (k, b) in methods.items():
            lo, hi = calibrate(k, b, y, cal, chk, 0.90, df.superblock.values)
            rows.append({"region": rk, "method": name, **interval_metrics(y[chk], lo, hi, 0.90)})
        print(rk, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "rev_f2_uq_val.csv", index=False)
    c = res[res.method != "Trivial"].groupby("method").agg(winkler=("winkler", "mean"), mincov=("coverage", "min"))
    ok = c[c.mincov >= 0.88]
    chosen = ok.winkler.idxmin() if len(ok) else "CQR-GNN"
    (RES / "rev_f2_selection.json").write_text(json.dumps({"selected_uq": chosen, "winner": win,
                                                           "table": c.round(4).to_dict(orient="index")}, indent=2))
    pd.set_option("display.width", 200)
    print(res[["region", "method", "coverage", "mean_width", "winkler", "rho_width_abserr"]].round(3).to_string(index=False))
    print("selected:", chosen)


if __name__ == "__main__":
    main()
