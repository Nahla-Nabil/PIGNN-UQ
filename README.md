# Physics-Informed Graph Neural Networks with Uncertainty Quantification for Dynamic Valuation of Blue Carbon Assets

5-page IEEE conference paper project (target track: "Artificial Intelligence
for Business and Industry" — see `PLAN.md` for the venue/deadline note).
Model coastal mangrove ecosystems as spatiotemporal graphs from Sentinel-2
imagery, train a GNN with a physics-informed carbon mass-balance loss, and
use calibrated uncertainty to drive risk-adjusted voluntary carbon credit
pricing.

**Status: scaffolding + data acquisition + planning complete. No model code
written yet — see `PLAN.md` for what's next.**

## Start here

1. [`PLAN.md`](PLAN.md) — week-by-week execution plan, scope cuts, minimum
   viable experiment, risks. **Read the deadline note at the top first.**
2. [`paper/literature_notes.md`](paper/literature_notes.md) — literature
   scan (PINN-GNN prior art, mangrove biomass RS, carbon buffer-pool
   methodologies, conformal/ensemble UQ), with the delta this paper claims.
3. [`paper/uq_decision.md`](paper/uq_decision.md) — why conformalized
   quantile regression is the primary UQ method (MC Dropout / deep
   ensembles as ablations).
4. [`data/README.md`](data/README.md) — exact data provenance, licenses,
   access dates, what's actually downloaded vs. blocked.

## Pilot region

Ten Thousand Islands / Rookery Bay NERR margin, Southwest Florida, USA
(western Everglades mangrove fringe) — chosen for data availability, not
ecological novelty; see `data/README.md` for the full justification.

## Repo structure

```
blue-carbon-pinn-gnn/
├── data/
│   ├── raw/               # real pulled samples: Sentinel-2, GMW, SOC, tides
│   ├── processed/         # (empty — graph construction happens in PLAN.md Week 1)
│   └── README.md          # provenance, licenses, access dates, gaps
├── src/
│   ├── data_pipeline/      # aoi.py + fetch_sentinel2/gmw/soc/tides.py (working, tested)
│   ├── models/             # (empty — PLAN.md Weeks 2-4)
│   ├── pricing/            # (empty — PLAN.md Week 5)
│   └── eval/                # (empty)
├── notebooks/               # exploratory only
├── experiments/
│   ├── configs/             # (empty — populated from PLAN.md Week 2 on)
│   └── logs/
├── paper/
│   ├── literature_notes.md
│   ├── uq_decision.md
│   ├── figures/
│   └── tables/
├── requirements.txt
├── PLAN.md
└── README.md
```

## Environment

- **Python:** 3.12, venv at `C:\Users\nahla\venvs\bcpg` (**not** inside this
  repo — see below).
- **Why the venv lives outside the repo:** this project's folder path is
  long enough (project root alone is 183 characters) that compiled
  extensions (rasterio's GDAL bindings) hit Windows' `MAX_PATH` limit
  and failed to load (`DLL load failed ... filename or extension is too
  long`) when the venv was nested inside it. Fix applied: venv relocated to
  a short path. If you recreate the venv, keep it short and outside this
  folder, or enable Windows long-path support first.
- **Installed and verified in this session:** `rasterio`, `geopandas`,
  `pystac-client`, `planetary-computer`, `requests`, `pandas`, `numpy`,
  `shapely`, `pyproj` — used to actually pull the real data in `data/raw/`.
- **Not yet installed anywhere:** `torch`, `torch-geometric`, `mapie` (heavy/
  CUDA-relevant — install locally where training will run; see
  `requirements.txt` and `PLAN.md` Week 2 / Risk R5).

```bash
# Recreate the venv (adjust path as needed to avoid Windows MAX_PATH issues):
python -m venv C:\Users\<you>\venvs\bcpg
C:\Users\<you>\venvs\bcpg\Scripts\pip install -r requirements.txt
# or, on an externally-managed system:
C:\Users\<you>\venvs\bcpg\Scripts\pip install --break-system-packages -r requirements.txt
```

## Data pipeline (working, real data — not synthetic)

```bash
PY=C:/Users/<you>/venvs/bcpg/Scripts/python.exe

# Sentinel-2: search STAC + pull an AOI-clipped sample (keyless)
$PY src/data_pipeline/fetch_sentinel2.py --mode clip --date-range "2026-01-01/2026-07-01" --max-cloud 15

# Global Mangrove Watch extent, one epoch, mosaicked + clipped to AOI
$PY src/data_pipeline/fetch_gmw.py --epoch 2020

# Mangrove SOC typology, windowed-read from the source COG (no full download)
$PY src/data_pipeline/fetch_soc.py

# NOAA tide gauge, <=31-day windows
$PY src/data_pipeline/fetch_tides.py --begin 20260701 --end 20260731
```

All four were run in this session and produced real files under
`data/raw/` — see `data/README.md` for exact provenance, sizes, and the two
documented gaps (SOC stock values, Simard AGB raster) that need a one-time
manual/local step to unblock.
