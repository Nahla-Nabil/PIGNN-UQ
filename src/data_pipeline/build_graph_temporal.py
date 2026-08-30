"""
Journal-extension sprint ("post-submission extension", NOT part of the
submitted paper): builds a SECOND, SEPARATE graph file with genuine
two-date temporal-change features, replacing the submitted paper's
single-date NDVI + static tide-distance steady-state physics proxy with an
actual dC/dt (rate-of-change) signal -- exactly the limitation named in
`paper/main.tex` Section "Limitations": "Single-date imagery constrains
the physics term to a cross-sectional relationship, not an actual dC/dt."

**Writes NEW files only. Never touches `data/processed/graph_nodes.csv` /
`graph_edges.npy` (the files the submitted paper's Table I/II trace to) --
those are read here in ONE place, read-only, purely to look up which
coarse superblocks the submitted paper's test split already used, so the
new test partition can avoid them. No existing file is opened for writing
anywhere in this module.**

Node geometry (reference grid, BLOCK aggregation, AOI bbox, SUPERBLOCK_GRID)
is copied verbatim from `build_graph.py` so this graph's node (row, col)
indices and coarse `superblock` ids line up exactly with the submitted
graph's -- required for the exclusion-set lookup in step 3 below to be
meaningful. `build_graph.py` itself is imported read-only (for its
`_reproject_to_ref` helper and path constants) and not modified.

## Step 1 -- second Sentinel-2 date

Scene 1 (already used by the submitted paper): S2B_MSIL2A_20260206T160339_
R097_T17RMJ_20260206T213114, 2026-02-06 (dry season), cloud cover per the
original STAC search (`fetch_sentinel2.py --max-cloud 15`, `data/README.md`).

Scene 2 (NEW, this extension): S2B_MSIL2A_20260815T155819_R097_T17RMJ_
20260815T195532, 2026-08-15 (wet season), cloud cover 20.07% (scene-wide,
`data/raw/sentinel2/stac_items_ten_thousand_islands_fl_2026-08-15_2026-08-16.json`).
Same MGRS tile (T17RMJ) as scene 1. Delta = 190 days (~6.2 months) --
at the low end of the requested "6-12 months" window, not a shortcut: this
session's STAC access only has scenes up to "today" (2026-08-30, current
session date); no scene closer to 12 months apart exists yet. 190 days
still spans SW Florida's dry-to-wet season transition (real, expected
NDVI/moisture contrast), and cloud cover (20.07%) is the lowest available
scene-wide value in the 2026-06-01/2026-08-30 search window (Florida wet
season is persistently cloudy at the whole-tile level; disclosed, not
hidden -- see `paper/NUMBER_SOURCE_LOG.md` and
`paper/extension_temporal_notes.md`).

## Step 2 -- the dC/dt proxy

The submitted paper's accretion proxy was single-date NDVI (a vegetation
*level*, not a rate). This module computes NDVI at both dates and forms
    d_ndvi = ndvi_t2 - ndvi_t1
    ndvi_rate_per_day = d_ndvi / DELTA_DAYS
as the genuine temporal-change signal, plus the same computation for two
other normalized-difference indices already used in the submitted paper's
Phase 1C feature ablation (NDWI, NDMI) and for raw B11 (SWIR), all "other
available band deltas" per the instruction. `src/models/physics_terms_
temporal.py` (separate module, new) uses `ndvi_rate_per_day` in place of
static `ndvi` as the accretion term of the mass-balance proxy -- same
functional form (equal-weighted, train-only min-max scaling, no fitted
coefficients) as `src/models/physics_terms.py`, only the accretion input
changes from a level to a rate. The export term (distance-to-tide-gauge)
is UNCHANGED -- that proxy's limitation (a positional, not hydrological,
stand-in) is a separate, still-open limitation this extension does not
claim to fix.

## Step 3 -- fresh, non-overlapping test partition

`data/processed/graph_nodes.csv`'s `superblock` column (the coarse 8x8
grid -- the unit the paper's Method section and NUMBER_SOURCE_LOG use
whenever they say "superblock") records, for every node, which of the 20
occupied coarse superblocks it falls in. Reading that column read-only:
the submitted paper's OPERATIVE test split (`split == "test"`, what Table
I/II report) occupies exactly 10 of the 20 occupied coarse superblocks.
Those 10 are excluded wholesale from this extension's test/val candidate
pool -- any node whose coarse superblock is one of the 10 is forced to
`split_ext == "train"` here, never eligible for the new test OR the new
val. (The legacy Day 1-2 split's test/val superblocks are, by construction
of the Phase-0 fix already in `build_graph.py`, entirely disjoint coarse
superblocks from the current operative test's -- verified below by
assertion -- so this exclusion already reaches every coarse superblock
the submitted paper's Table I/II test numbers come from.)

From the remaining pool, the identical spatially-blocked selection logic
(cumulative node-fraction greedy assignment over a finer regrid, same
technique `build_graph.py`'s own Phase-0 fix uses for exactly this
"restricted candidate pool" situation) draws a fresh ~20% test / ~15% val,
with a new seed. Train is everything else, including the excluded (forced-
train) superblocks -- training on them is safe; the fix here is only about
never evaluating on them again, mirroring the project's established
Phase-0 rationale one level up.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME, NOAA_TIDE_STATION_LATLON  # noqa: E402
from src.data_pipeline.build_graph import (  # noqa: E402
    _reproject_to_ref, SOC_MEAN_PATH, SOC_LOWER_PATH, SOC_UPPER_PATH, GMW_PATH,
    S2_PATH, S2_B03_PATH, BLOCK, SUPERBLOCK_GRID, TEST_SUPERBLOCK_FRACTION, VAL_SUPERBLOCK_FRACTION,
)

DATA_RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

# --- Second Sentinel-2 date (this extension's own pull, see module docstring) ---
S2_T2_ID = "S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532"
S2_T2_PATH = DATA_RAW / "sentinel2" / f"sample_clip_{S2_T2_ID}.tif"
S2_T2_B03_PATH = DATA_RAW / "sentinel2" / f"sample_clip_B03_{S2_T2_ID}.tif"
DATE_T1 = "2026-02-06"
DATE_T2 = "2026-08-15"
DELTA_DAYS = 190  # 2026-02-06 -> 2026-08-15, computed and disclosed in NUMBER_SOURCE_LOG.md

EXT_SPLIT_SEED = 20260830  # this extension's own seed (session date) -- distinct from both
# SPLIT_SEED (42, legacy Day 1-2) and PHASE1_SPLIT_SEED (20260829, the submitted paper's
# Phase-0 fix), so it is unambiguous in a diff/audit which seed produced which split.
EXT_SUPERBLOCK_GRID = 20  # same finer-regrid technique/resolution as build_graph.py's
# PHASE1_SUPERBLOCK_GRID, reused here for the same reason: the restricted candidate pool
# (here, all coarse superblocks EXCEPT the submitted paper's 10 operative-test ones) is
# lumpy at the 8x8 coarse resolution, so greedy cumulative-node-fraction selection needs a
# finer regrid to land close to the 20%/15% targets.


def _load_band_stack(path, bands, ref_profile, ref_transform, ref_crs, h, w):
    out = {}
    with rasterio.open(path) as src:
        for dst_name, band_idx in bands:
            arr = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band_idx),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            out[dst_name] = arr
    return out


def build():
    with rasterio.open(SOC_MEAN_PATH) as ref:
        ref_profile = ref.profile.copy()
        soc_mean = ref.read(1).astype(np.float32)
        soc_nodata = ref.nodata
        ref_transform = ref.transform
        ref_crs = ref.crs
        h, w = ref.shape

    soc_valid = (soc_mean != soc_nodata) & (soc_mean > 0)
    soc_lower = _reproject_to_ref(SOC_LOWER_PATH, ref_profile, Resampling.bilinear)
    soc_upper = _reproject_to_ref(SOC_UPPER_PATH, ref_profile, Resampling.bilinear)
    gmw = _reproject_to_ref(GMW_PATH, ref_profile, Resampling.nearest)
    gmw = np.nan_to_num(gmw, nan=0.0)

    # --- date 1 (submitted-paper scene, re-read here identically -- same file, read-only) ---
    t1 = _load_band_stack(S2_PATH, [("b04", 1), ("b08", 2), ("b11", 3)], ref_profile, ref_transform, ref_crs, h, w)
    t1["b03"] = _reproject_to_ref(S2_B03_PATH, ref_profile, Resampling.bilinear)

    # --- date 2 (this extension's new scene) ---
    t2 = _load_band_stack(S2_T2_PATH, [("b04", 1), ("b08", 2), ("b11", 3)], ref_profile, ref_transform, ref_crs, h, w)
    t2["b03"] = _reproject_to_ref(S2_T2_B03_PATH, ref_profile, Resampling.bilinear)

    h_blocks, w_blocks = h // BLOCK, w // BLOCK

    def block_mean(arr, mask=None):
        cropped = arr[: h_blocks * BLOCK, : w_blocks * BLOCK]
        reshaped = cropped.reshape(h_blocks, BLOCK, w_blocks, BLOCK)
        if mask is None:
            return np.nanmean(reshaped, axis=(1, 3))
        m = mask[: h_blocks * BLOCK, : w_blocks * BLOCK].reshape(h_blocks, BLOCK, w_blocks, BLOCK)
        with np.errstate(invalid="ignore"):
            out = np.where(m.sum(axis=(1, 3)) > 0,
                            np.nansum(np.where(m, reshaped, 0), axis=(1, 3)) / np.maximum(m.sum(axis=(1, 3)), 1),
                            np.nan)
        return out

    soc_valid_frac = block_mean(soc_valid.astype(np.float32))
    soc_mean_b = block_mean(np.where(soc_valid, soc_mean, np.nan))
    soc_lower_b = block_mean(np.where(soc_valid, soc_lower, np.nan))
    soc_upper_b = block_mean(np.where(soc_valid, soc_upper, np.nan))
    gmw_frac_b = block_mean(gmw)

    t1_b = {k: block_mean(v) for k, v in t1.items()}
    t2_b = {k: block_mean(v) for k, v in t2.items()}

    def ndvi(b): return (b["b08"] - b["b04"]) / (b["b08"] + b["b04"] + 1e-9)
    def ndwi(b): return (b["b03"] - b["b08"]) / (b["b03"] + b["b08"] + 1e-9)
    def ndmi(b): return (b["b08"] - b["b11"]) / (b["b08"] + b["b11"] + 1e-9)

    ndvi_t1, ndvi_t2 = ndvi(t1_b), ndvi(t2_b)
    ndwi_t1, ndwi_t2 = ndwi(t1_b), ndwi(t2_b)
    ndmi_t1, ndmi_t2 = ndmi(t1_b), ndmi(t2_b)

    d_ndvi = ndvi_t2 - ndvi_t1
    d_ndwi = ndwi_t2 - ndwi_t1
    d_ndmi = ndmi_t2 - ndmi_t1
    d_b11 = t2_b["b11"] - t1_b["b11"]
    ndvi_rate_per_day = d_ndvi / DELTA_DAYS

    keep = (
        (gmw_frac_b > 0.5) & (soc_valid_frac > 0.5) & np.isfinite(soc_mean_b)
        & np.isfinite(ndvi_t1) & np.isfinite(t1_b["b11"]) & np.isfinite(ndwi_t1) & np.isfinite(ndmi_t1)
        & np.isfinite(ndvi_t2) & np.isfinite(t2_b["b11"]) & np.isfinite(ndwi_t2) & np.isfinite(ndmi_t2)
    )
    rows, cols = np.where(keep)
    print(f"grid: {h_blocks}x{w_blocks} blocks, kept nodes (both dates valid): {keep.sum()}")

    block_transform = ref_transform * ref_transform.scale(BLOCK, BLOCK)
    xs, ys = rasterio.transform.xy(block_transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)

    tide_lat, tide_lon = NOAA_TIDE_STATION_LATLON
    dist_to_tide_km = np.sqrt(
        ((ys - tide_lat) * 111.0) ** 2 + ((xs - tide_lon) * 111.0 * np.cos(np.radians(tide_lat))) ** 2
    )

    df = pd.DataFrame({
        "row": rows, "col": cols, "lon": xs, "lat": ys,
        # date-1 features (identical formulas/values to data/processed/graph_nodes.csv)
        "ndvi": ndvi_t1[rows, cols], "b04": t1_b["b04"][rows, cols], "b08": t1_b["b08"][rows, cols],
        "b11": t1_b["b11"][rows, cols], "b03": t1_b["b03"][rows, cols],
        "ndwi": ndwi_t1[rows, cols], "ndmi": ndmi_t1[rows, cols],
        # date-2 features (new)
        "ndvi_t2": ndvi_t2[rows, cols], "b04_t2": t2_b["b04"][rows, cols], "b08_t2": t2_b["b08"][rows, cols],
        "b11_t2": t2_b["b11"][rows, cols], "b03_t2": t2_b["b03"][rows, cols],
        "ndwi_t2": ndwi_t2[rows, cols], "ndmi_t2": ndmi_t2[rows, cols],
        # genuine temporal-change (dC/dt proxy input) features -- new
        "d_ndvi": d_ndvi[rows, cols], "d_ndwi": d_ndwi[rows, cols], "d_ndmi": d_ndmi[rows, cols],
        "d_b11": d_b11[rows, cols], "ndvi_rate_per_day": ndvi_rate_per_day[rows, cols],
        "delta_days": DELTA_DAYS,
        "dist_to_tide_km": dist_to_tide_km,
        "soc_tha_mean": soc_mean_b[rows, cols],
        "soc_tha_lower_std": soc_lower_b[rows, cols],
        "soc_tha_upper_std": soc_upper_b[rows, cols],
    })

    # coarse superblock id -- identical formula to build_graph.py, so ids line up 1:1
    # with data/processed/graph_nodes.csv's `superblock` column for the same (row, col).
    lon_min, lon_max, lat_min, lat_max = AOI_BBOX[0], AOI_BBOX[2], AOI_BBOX[1], AOI_BBOX[3]
    sb_lon = np.clip(((df["lon"] - lon_min) / (lon_max - lon_min) * SUPERBLOCK_GRID).astype(int), 0, SUPERBLOCK_GRID - 1)
    sb_lat = np.clip(((df["lat"] - lat_min) / (lat_max - lat_min) * SUPERBLOCK_GRID).astype(int), 0, SUPERBLOCK_GRID - 1)
    df["superblock"] = sb_lat * SUPERBLOCK_GRID + sb_lon

    # ------------------------------------------------------------------
    # Step 3: fresh, non-overlapping test partition (see module docstring).
    # Read-only lookup of which coarse superblocks the SUBMITTED paper's
    # operative test split occupies -- data/processed/graph_nodes.csv is
    # opened here strictly for this lookup, never written to.
    # ------------------------------------------------------------------
    submitted = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv", usecols=["superblock", "split", "split_legacy_day1_2"])
    excluded_superblocks = set(submitted.loc[submitted["split"] == "test", "superblock"].unique().tolist())
    legacy_test_val_superblocks = set(
        submitted.loc[submitted["split_legacy_day1_2"].isin(["test", "val"]), "superblock"].unique().tolist()
    )
    overlap_check = excluded_superblocks & legacy_test_val_superblocks
    print(f"submitted paper's operative-test superblocks (excluded from new test/val pool): "
          f"{sorted(excluded_superblocks)} (n={len(excluded_superblocks)} of "
          f"{submitted['superblock'].nunique()} occupied)")
    print(f"[sanity] overlap between operative-test superblocks and legacy Day1-2 test/val "
          f"superblocks (expected 0, by construction of the submitted paper's own Phase-0 fix): "
          f"{sorted(overlap_check)}")

    pool_mask = ~df["superblock"].isin(excluded_superblocks)
    fine_sb_lon = np.clip(((df["lon"] - lon_min) / (lon_max - lon_min) * EXT_SUPERBLOCK_GRID).astype(int),
                            0, EXT_SUPERBLOCK_GRID - 1)
    fine_sb_lat = np.clip(((df["lat"] - lat_min) / (lat_max - lat_min) * EXT_SUPERBLOCK_GRID).astype(int),
                            0, EXT_SUPERBLOCK_GRID - 1)
    df["ext_fine_superblock"] = fine_sb_lat * EXT_SUPERBLOCK_GRID + fine_sb_lon

    pool_sb_counts = df.loc[pool_mask, "ext_fine_superblock"].value_counts()
    pool_occupied = pool_sb_counts.index.to_numpy()
    rng = np.random.default_rng(EXT_SPLIT_SEED)
    order = rng.permutation(pool_occupied)
    target_n_test = int(round(len(df) * TEST_SUPERBLOCK_FRACTION))
    target_n_val = int(round(len(df) * VAL_SUPERBLOCK_FRACTION))

    new_test_sb, new_val_sb = set(), set()
    running = 0
    for sb in order:
        if running >= target_n_test:
            break
        new_test_sb.add(int(sb))
        running += int(pool_sb_counts.loc[sb])
    running = 0
    for sb in order:
        if int(sb) in new_test_sb:
            continue
        if running >= target_n_val:
            break
        new_val_sb.add(int(sb))
        running += int(pool_sb_counts.loc[sb])

    def _assign(row):
        if not pool_mask.loc[row.name]:
            return "train"  # forced -- coarse superblock overlaps submitted paper's operative test
        sb = row["ext_fine_superblock"]
        if sb in new_test_sb:
            return "test"
        if sb in new_val_sb:
            return "val"
        return "train"

    df["split_ext"] = df.apply(_assign, axis=1)

    test_mask = df["split_ext"] == "test"
    val_mask = df["split_ext"] == "val"
    assert not test_mask[~pool_mask].any(), "no new-test node may fall in an excluded (submitted-test) superblock"
    assert not val_mask[~pool_mask].any(), "no new-val node may fall in an excluded (submitted-test) superblock"
    assert not (df.loc[test_mask, "superblock"].isin(excluded_superblocks)).any(), \
        "new-test superblocks must be disjoint from the submitted paper's operative-test superblocks"

    # 8-connectivity edges among kept nodes on the (row, col) block grid
    node_id = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    edges = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = node_id.get((r + dr, c + dc))
                if j is not None:
                    edges.append((i, j))
    edges = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "graph_nodes_ext_temporal.csv", index=False)
    np.save(DATA_PROCESSED / "graph_edges_ext_temporal.npy", edges)

    print(f"nodes: {len(df)}, edges (directed): {edges.shape[1]}")
    print(f"[extension split_ext] train: {(df['split_ext']=='train').sum()}, "
          f"val: {(df['split_ext']=='val').sum()}, test: {(df['split_ext']=='test').sum()}")
    print(f"  per-split SOC mean (t/ha): {df.groupby('split_ext')['soc_tha_mean'].mean().to_dict()}")
    print(f"  per-split d_ndvi mean: {df.groupby('split_ext')['d_ndvi'].mean().to_dict()}")
    print(f"SOC label range: min={df.soc_tha_mean.min():.1f}, max={df.soc_tha_mean.max():.1f}, "
          f"mean={df.soc_tha_mean.mean():.1f}, std={df.soc_tha_mean.std():.1f}")
    print(f"d_ndvi range: min={df.d_ndvi.min():.4f}, max={df.d_ndvi.max():.4f}, "
          f"mean={df.d_ndvi.mean():.4f}, std={df.d_ndvi.std():.4f}")
    print(f"wrote -> {DATA_PROCESSED / 'graph_nodes_ext_temporal.csv'}, "
          f"{DATA_PROCESSED / 'graph_edges_ext_temporal.npy'}")
    return df, edges


if __name__ == "__main__":
    build()
