"""
Issuance under EVERY UQ method (reviewer R1: "CQR intervals ... do not even
use the physics-informed model"). Label-free: credited stock depends only
on each node's point prediction and interval (fit on train, calibrated on
val), so no test label is read. Same bases/seeds/calibration as
run_revision_test_eval.py.
Writes results/rev_e_issuance_all_methods.csv.
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
from src.pricing.issuance import issuance_table  # noqa: E402
from scripts.run_rev_e6_uq_val import build_bases  # noqa: E402
from scripts.run_revision_test_eval import selections  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    sel, _ = selections()
    rows = []
    for rk in ("r1", "r2"):
        df, edges = load_rev(REGIONS[rk])
        y = df[TARGET_COL].values.astype(float)
        train, val, test = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
        y[test] = np.nan  # guard: no test label can enter calibration
        bases = build_bases(df, edges, sel, train, [val, test])
        for name, (kind, base) in bases.items():
            if name == "Trivial":
                continue
            yhat = base[test, QUANTILES.index(0.5)] if kind == "cqr" else base[0][test]
            for mond in (False, True):
                lo, hi = calibrate(kind, base, y, val, test, 0.90, df.superblock.values, mondrian=mond)
                rows.append({"region": rk, "method": name, "calibration": "mondrian" if mond else "global",
                             **issuance_table(yhat, lo, hi)})
        print(rk, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev_e_issuance_all_methods.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["region", "method", "calibration", "median_unc_pct", "ruleA_vs_flat", "ruleB_node_vs_flat",
               "dedB_p50", "dedB_p90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
