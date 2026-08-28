# Data Provenance

Pilot region, sources, licenses, access dates, and known gaps. All "verified"
rows below were tested **live from this sandbox on 2026-08-29** — the exact
commands ran successfully and produced the files listed under
"what's on disk now."

## Pilot region decision

**Ten Thousand Islands / Rookery Bay NERR margin, Southwest Florida, USA**
(western Everglades mangrove fringe).

`AOI_BBOX = (-81.90, 25.90, -81.40, 26.30)` (WGS84 lon/lat), defined once in
[`src/data_pipeline/aoi.py`](../src/data_pipeline/aoi.py) and imported by
every fetch script so the AOI is never redefined inconsistently.

**Why this region, not Sundarbans or a generic Southeast Asian site:** it is
the only candidate where *every* data layer the mass-balance term needs is
actually reachable from this sandbox without an account:

| Requirement | This region | Why other candidates lose |
|---|---|---|
| Tide gauge for the hydrological/export term | NOAA CO-OPS station **8725114**, Naples Bay North (26.1367, -81.7883) — on the AOI's doorstep, free JSON API, no key | NOAA CO-OPS is **US-only**. Sundarbans/SE Asia would need a national hydrographic agency portal (India/Bangladesh/Indonesia) — untested, likely needs registration, possibly not machine-accessible at all |
| Sentinel-2 imagery | Microsoft Planetary Computer STAC, global coverage, keyless | Same for any region — not a differentiator |
| Mangrove extent | Global Mangrove Watch, global coverage, open | Same for any region — not a differentiator |
| Independent field-validated carbon-stock reference | Radabaugh et al. (2023) spatial model, Tampa Bay→Everglades — **covers this exact AOI** | A Sundarbans pilot would need a separately-sourced regional validation paper we have not vetted for open data access |

This also matches the task's scope constraint: **one region, chosen for data
defensibility, not ecological novelty.**

## Sources

### 1. Sentinel-2 L2A imagery — ✅ verified, keyless

- **Source:** Microsoft Planetary Computer STAC API
  (`https://planetarycomputer.microsoft.com/api/stac/v1`), collection
  `sentinel-2-l2a`.
- **Access method:** `pystac-client` searches the catalog; asset URLs point
  to public Azure Blob Storage and are signed with a short-lived, **free,
  keyless** SAS token via the `planetary-computer` package
  (`planetary_computer.sign_inplace`) — confirmed no account/API key is
  needed (tested the signing endpoint directly: returns a token with no
  auth header sent).
- **License:** Copernicus Sentinel data is free and open (ESA/Copernicus
  data policy); Planetary Computer's redistribution terms:
  <https://planetarycomputer.microsoft.com/terms>.
- **Script:** [`src/data_pipeline/fetch_sentinel2.py`](../src/data_pipeline/fetch_sentinel2.py)
- **What's on disk now (2026-08-29):**
  - `data/raw/sentinel2/stac_items_ten_thousand_islands_fl_2026-01-01_2026-07-01.json`
    — metadata for **19 real scenes** over the AOI, Jan–Jul 2026, <15% cloud.
  - `data/raw/sentinel2/sample_clip_S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114.tif`
    — a real, AOI-windowed (not full-scene) 3-band (B04/B08/B11) GeoTIFF,
    3537×5022 px, ~58 MB, from the lowest-cloud scene (0.013% cloud,
    2026-02-06). Pulled via HTTP range requests (rasterio windowed read),
    **not** a full scene download.
- **Gap / local step needed:** none for search+sample. A full multi-date
  time series for training (the real pipeline needs many dates, not one) is
  a longer batch job — reasonable to run from this sandbox too, just budget
  the time; see `PLAN.md` Week 1.

### 2. Global Mangrove Watch (GMW) v3 — extent time series — ✅ verified, open

- **Source:** Bunting et al., "Global Mangrove Watch (1996–2020) Version 3.0
  Dataset," Zenodo, doi:[10.5281/zenodo.6894273](https://zenodo.org/records/6894273).
- **Access method:** Zenodo REST API, direct HTTPS file download (per-epoch
  global raster zips, ~65–70 MB each for the "gtiff" product; confirmed the
  gtiff zips are internally tiled by ~1°×1° cell, e.g. `GMW_N26W082_2020_v3.tif`
  covers our AOI — no need to download a global mosaic).
- **License:** **CC-BY-4.0** (confirmed via Zenodo record metadata,
  `access_right: open`). Attribution required: "©Global Mangrove Watch" /
  "©Aberystwyth University, solo Earth Observation, Wetlands International,
  The Nature Conservancy and JAXA."
- **Script:** [`src/data_pipeline/fetch_gmw.py`](../src/data_pipeline/fetch_gmw.py)
  — downloads one epoch's global zip, mosaics whichever tile(s) intersect
  `AOI_BBOX`, clips, writes the small AOI raster, deletes the global
  intermediate.
- **What's on disk now (2026-08-29):**
  `data/raw/gmw/gmw_v3_2020_ten_thousand_islands_fl_clip.tif` — real 2020
  mangrove-extent mask for the full AOI, 1801×2251 px, ~78 KB. Sanity check:
  7.8% of AOI pixels are mangrove (AOI includes open water and uplands, so
  this is plausible for a coastal fringe).
- **Gap:** for a temporal accretion signal, re-run with `--epoch 2010` (or
  another earlier epoch) and diff against 2020 — not done yet, deferred to
  `PLAN.md` Week 1 (single extra ~66 MB download + clip, cheap).

### 3. Global mangrove soil organic carbon (SOC) — ⚠️ partially verified

- **Source:** Sanderman, Worthington et al. (Woods Hole Research Center),
  "Global mangrove soil carbon data set at 30 m resolution for year 2020
  (0–100 cm)," Zenodo, doi:[10.5281/zenodo.7729492](https://zenodo.org/records/7729492).
  Companion publication: PMC10562832 / ScienceDirect (2023).
- **License:** **CC-BY-4.0** (confirmed via Zenodo record metadata).
- **What actually works from this sandbox:** the companion **typology/
  ecotype COG** (`mangroves_typology_v3_cog.tif`, 1.1 GB *source* size) can
  be **windowed-read** via GDAL `/vsicurl/` + rasterio without downloading
  the full file — confirmed: pulled a 1600×2000 px AOI clip
  (`data/raw/soc/mangrove_typology_ten_thousand_islands_fl_clip.tif`, ~56 KB
  on disk) by transferring only the bytes covering the AOI window. Class
  balance (267,714 "mangrove" px of 3.2M total) is consistent with the GMW
  extent mask — good cross-source sanity check.
- **What does NOT work from this sandbox:** the actual **per-pixel SOC
  stock values** ship only inside `mangroves_tiles_SOC_predictions_2020.zip`,
  a single **4.7 GB** archive with no COG mosaic and no tile-scoped download
  endpoint — too large/slow to pull in this session.
- **Script:** [`src/data_pipeline/fetch_soc.py`](../src/data_pipeline/fetch_soc.py)
  (typology clip — done) — prints the exact local command for the SOC zip.
- **Local step needed (run outside this sandbox, on your machine, once):**
  ```bash
  curl -L -o mangroves_tiles_SOC_predictions_2020.zip \
    https://zenodo.org/api/records/7729492/files/mangroves_tiles_SOC_predictions_2020.zip/content
  # then extract only the tile(s) intersecting AOI_BBOX (see aoi.py) —
  # inspect the archive's internal tile naming first with:
  unzip -l mangroves_tiles_SOC_predictions_2020.zip | less
  ```
  This is a one-time ~4.7 GB download; do it on a machine/connection where
  that's cheap, then copy just the extracted AOI tile(s) back into
  `data/raw/soc/`.

### 4. NOAA tide gauge (hydrological driver for the mass-balance term) — ✅ verified, keyless

- **Source:** NOAA CO-OPS station **8725114**, "Naples Bay, North, FL"
  (26.1367, -81.7883).
- **Access method:** NOAA CO-OPS Data API
  (`https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`), free JSON,
  no key. **Caveat confirmed by testing:** the 6-minute `water_level`
  product caps requests at **31 days per call** — the script windows
  requests accordingly; a multi-year pull needs to loop over 31-day chunks.
- **License:** US federal government data — public domain.
- **Script:** [`src/data_pipeline/fetch_tides.py`](../src/data_pipeline/fetch_tides.py)
- **What's on disk now (2026-08-29):**
  - `data/raw/tides/water_level_8725114_20260701_20260731.csv` — 7,440 real
    6-minute water-level observations, July 2026.
  - `data/raw/tides/hilo_predictions_8725114_20260701_20260731.csv` — 120
    real high/low tide predictions, same window.
- **Gap:** none for the pilot window. For the full training period, loop the
  script over consecutive 31-day windows (cheap, just more calls).

### 5. Simard et al. (2019) global mangrove AGB/canopy height — 🔴 blocked, needs local credential step

- **Source:** ORNL DAAC, doi:10.3334/ORNLDAAC/1665 (see
  `literature_notes.md` §2) — the standard global AGB reference product,
  per-country GeoTIFFs including USA/Florida.
- **What we confirmed:** direct HTTPS access to the ORNL DAAC data
  directory returns **HTTP 401 Unauthorized** — this dataset requires a
  **NASA Earthdata Login** account (free, self-service, but a manual
  registration step this sandbox cannot complete headlessly).
- **Local step needed (one-time, ~5 minutes):**
  1. Register at <https://urs.earthdata.nasa.gov/users/new> (free).
  2. Create a `.netrc` entry or a personal access token per ORNL DAAC's
     download instructions:
     <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1665>
  3. Download the USA tile(s) and clip to `AOI_BBOX` the same way
     `fetch_gmw.py` does (reuse the mask/merge pattern).
- **Fallback already in place:** Radabaugh et al. (2023) (source #2 above,
  literature-only — see gap below) and the Lagomasino/Fatoyinbo Everglades
  LiDAR/TanDEM-X study (see `literature_notes.md` §2) both give independent,
  regionally-specific AGB numbers for sanity-checking model output even
  without the Simard raster.

### 6. Radabaugh et al. (2023) field-plot blue carbon stock model — 🔴 data availability unconfirmed

- Our strongest regional validation reference (see `literature_notes.md`
  §2), but we only confirmed the **publication** (Estuaries and Coasts,
  doi:10.1007/s12237-023-01217-7), not a public download link for the
  underlying 30 m raster carbon-stock layers themselves. FWC/FWRI publishes
  a GIS open-data portal (`geodata.myfwc.com`) that plausibly hosts it, but
  this was not confirmed by a working URL in this session.
- **Local step needed:** browse <https://geodata.myfwc.com/pages/marine> and
  <https://myfwc.com/research/gis/maps-data/> by hand for the specific
  layer, or email the corresponding author for the raster outputs (common
  and normal for a state-agency-affiliated study). Until then, treat the
  paper's **reported summary statistics** (393.9 ± 107.1 Mg C ha⁻¹
  above+belowground) as a scalar sanity check, not a per-pixel validation
  raster.

## Summary: what's real and on disk right now vs. blocked

| Layer | Status | Blocker |
|---|---|---|
| Sentinel-2 imagery | ✅ real sample on disk | none |
| GMW mangrove extent | ✅ real, full AOI, on disk | none |
| SOC typology (proxy) | ✅ real, windowed sample, on disk | none |
| SOC stock values (actual carbon numbers) | 🔶 documented, not pulled | 4.7 GB file, needs local run |
| NOAA tide gauge | ✅ real, on disk | none |
| Simard 2019 AGB (global standard) | 🔴 blocked | needs free NASA Earthdata login |
| Radabaugh 2023 raster (best regional validation) | 🔴 blocked | download link not confirmed, needs manual portal search or author contact |

**Net assessment:** the pipeline can be built and tested end-to-end today on
real (not synthetic) imagery, extent, tide, and typology data. The two
gaps that need a human in the loop before the *quantitative validation*
claims of the paper are solid are (a) the SOC zip and (b) either Simard
2019 or a confirmed Radabaugh raster — both are one-time local steps, not
recurring blockers, and both have a scalar/proxy fallback already
documented above so the pipeline is not stalled waiting on them.
