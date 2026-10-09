"""
Market-practice check (reviewer R2: pricing "not validated against market
outcomes or non-permanence risks"). Source: Berkeley Carbon Trading
Project, Voluntary Registry Offsets Database v2026-08 (CC BY 4.0),
data/raw/market/VROD_v2026-08.xlsx. Blue carbon = methodology VM0033 or
AR-AM0014 (degraded mangrove), or "mangrove"/"blue carbon"/"seagrass" in
the project name. Realized non-permanence buffer share = buffer deposits /
(issued + buffer deposits); demand = retired / issued.
Writes results/rev_f_market_bluecarbon.csv and _summary.json.
"""
import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

d = pd.read_excel(ROOT / "data/raw/market/VROD_v2026-08.xlsx", "PROJECTS", header=3)
d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
col = lambda k: [c for c in d.columns if k.lower() in c.lower()][0]  # noqa: E731
I, B, R = col("Total Credits Issued"), col("Total Buffer Pool"), col("Total Credits Retired")
meth, name = d["Methodology / Protocol"].astype(str), d["Project Name"].astype(str)
blue = meth.str.contains("VM0033|AR-AM0014", case=False) | name.str.contains("mangrove|blue carbon|seagrass", case=False)
b = d[blue].copy()
for c in (I, B, R):
    b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
s = b[b[I] > 0].copy()
s["buffer_pct"] = s[B] / (s[I] + s[B]) * 100
s["retired_pct"] = s[R] / s[I] * 100
cols = ["Project ID", "Project Name", "Methodology / Protocol", "Country", I, B, R, "buffer_pct", "retired_pct"]
s.sort_values(I, ascending=False)[cols].to_csv(ROOT / "results/rev_f_market_bluecarbon.csv", index=False)
summ = {"n_blue_projects_listed": int(len(b)), "n_with_issuances": int(len(s)),
        "n_vm0033_with_issuances": int(s["Methodology / Protocol"].str.contains("VM0033").sum()),
        "total_issued": float(s[I].sum()), "total_buffer": float(s[B].sum()), "total_retired": float(s[R].sum()),
        "buffer_pct_median": float(s.buffer_pct.median()),
        "buffer_pct_iqr": [float(s.buffer_pct.quantile(.25)), float(s.buffer_pct.quantile(.75))],
        "buffer_pct_credit_weighted": float(s[B].sum() / (s[I].sum() + s[B].sum()) * 100),
        "retired_pct_credit_weighted": float(s[R].sum() / s[I].sum() * 100)}
(ROOT / "results/rev_f_market_summary.json").write_text(json.dumps(summ, indent=2))
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 45)
print(s.sort_values(I, ascending=False)[["Project ID", "Project Name", "Country", I, "buffer_pct", "retired_pct"]]
      .head(25).round(1).to_string(index=False))
print(json.dumps(summ, indent=2))
