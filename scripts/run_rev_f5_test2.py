"""
Addendum F5: DECLARED SECOND TEST READ for ONE pre-specified model (the F1
winner) and its F2-selected interval method, both regions. Refuses to run if
the winner is xgb or phys_gnn (already read in the first test evaluation), or
if it was already run. Logs the read in results/rev_test_RUN_STAMP.json.
Writes results/rev_test2_*.
"""
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_pipeline.region import REGIONS  # noqa: E402
from src.models.rev_common import load_rev, metrics, mass_balance_proxy  # noqa: E402
from src.models.rev_uq import calibrate, interval_metrics  # noqa: E402
from src.models.gnn_common import TARGET_COL  # noqa: E402
from src.pricing.issuance import issuance_table  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases  # noqa: E402
from scripts.run_revision_test_eval import selections, two_hop_interior  # noqa: E402
from scripts.rev_f_common import predict, winner_name, RES  # noqa: E402
from scripts.run_rev_f2_uq import winner_ncp_base  # noqa: E402

STAMP = RES / "rev_test_RUN_STAMP.json"


def main():
    win = winner_name()
    uq = json.loads((RES / "rev_f2_selection.json").read_text())["selected_uq"]
    if win in ("xgb", "phys_gnn"):
        sys.exit(f"winner {win} was already evaluated in the first test read: no second read (Addendum F5).")
    log = json.loads(STAMP.read_text())
    if any("F5" in r["reason"] for r in log["runs"]):
        sys.exit("F5 second read already logged; refusing to run again.")
    log["runs"].append({"time": datetime.now().isoformat(),
                        "reason": f"F5 declared second test read: one pre-specified model ({win}) + {uq} (Addendum F)"})
    STAMP.write_text(json.dumps(log, indent=2))

    sel, _ = selections()
    pts, uqr, iss = [], [], []
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        y = df[TARGET_COL].values.astype(float)
        train, val, test = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
        interior = np.where(two_hop_interior(edges, (df.split == "test").values))[0]
        rad = df["radabaugh_bgc_mgha"].values
        prox = mass_balance_proxy(df, train, "P2")
        for seed in (1, 2, 3):
            p = predict(df, edges, win, train, [test], seed, stop_idx=val)
            ok = np.isfinite(rad[test])
            pts.append({"region": rk, "model": win, "seed": seed,
                        **{f"test_{k}": v for k, v in metrics(y[test], p[test], y[train].mean()).items()},
                        **{f"interior2hop_{k}": v for k, v in metrics(y[interior], p[interior], y[train].mean()).items()},
                        "prior_gen_corr_P2": np.corrcoef(p[test], prox[test])[0, 1],
                        "corr_with_radabaugh": np.corrcoef(p[test][ok], rad[test][ok])[0, 1]})
        bases = build_bases(df, edges, sel, train, [val, test])
        kind, cqr = bases["CQR-GNN"]
        if uq.startswith("NCP-"):
            yhat, s = winner_ncp_base(df, edges, win, train, [val, test], cqr)
            kind, base = "ncp", (yhat, s)
            yh_test = yhat[test]
        else:
            base = cqr
            yh_test = cqr[test, 2]
        for mond in (False, True):
            lo, hi = calibrate(kind, base, y, val, test, 0.90, df.superblock.values, mondrian=mond)
            cal = "mondrian" if mond else "global"
            uqr.append({"region": rk, "method": uq, "calibration": cal, **interval_metrics(y[test], lo, hi, 0.90)})
            iss.append({"region": rk, "method": uq, "calibration": cal, **issuance_table(yh_test, lo, hi)})
        print(rk, "done", flush=True)
    pt = pd.DataFrame(pts)
    pt.to_csv(RES / "rev_test2_point_by_seed.csv", index=False)
    summ = pt.groupby(["region", "model"]).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(RES / "rev_test2_point_summary.csv")
    pd.DataFrame(uqr).to_csv(RES / "rev_test2_uq.csv", index=False)
    pd.DataFrame(iss).to_csv(RES / "rev_test2_issuance.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ[[c for c in summ.columns if c.startswith(("test_r2", "test_mae", "interior2hop_r2"))]].round(3))
    print(pd.DataFrame(uqr)[["region", "calibration", "coverage", "mean_width", "winkler", "rho_width_abserr"]].round(3))
    print(pd.DataFrame(iss)[["region", "calibration", "ruleB_node_vs_flat", "dedB_p50"]].round(3))


if __name__ == "__main__":
    main()
