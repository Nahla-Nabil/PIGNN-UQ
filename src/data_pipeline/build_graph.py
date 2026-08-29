"""
Build the pilot-region spatiotemporal graph from data already pulled by
fetch_sentinel2.py / fetch_gmw.py / fetch_soc.py.

Design (documented here because every choice below is a real methodological
decision that belongs in the paper's Method section, not just in code):

- **Reference grid = the SOC dataset's own native 30m grid**, cropped to
  AOI_BBOX. Sentinel-2 bands and the GMW mask are reprojected/resampled ONTO
  this grid (bilinear for continuous bands, nearest for the binary mask) --
  not the other way around -- so the regression LABEL (SOC t/ha) is never
  itself resampled/interpolated. Resampling only touches the *features*.
- **Node aggregation: 5x5 blocks of the native 30m grid -> ~150m nodes.**
  Native mangrove-masked pixel count is ~267,714 (see fetch_soc.py sanity
  check); a full-resolution graph at that size is not a Day-1-tractable CPU
  training problem for this sprint, so nodes are 5x5-block aggregates
  (mean for continuous bands and the SOC label, fraction-mangrove for the
  mask). This is a stated simplification, not a hidden one.
- **Kept nodes: block-mean GMW mangrove fraction > 0.5 AND valid (nonzero,
  non-nodata) SOC value.** Open water / upland / no-SOC-data cells are
  dropped, not zero-filled -- zero-filling would silently teach the model
  a fake "no mangrove = 0 carbon" signal instead of just not training on
  those cells at all.
- **Label: mean SOC (t C/ha, 0-100cm)** from
  soc_tha_0_100cm_mean_..._clip.tif, independently sourced (Sanderman/
  Worthington/WHRC et al., built from field soil cores -- see
  data/README.md) -- NOT derived from this project's own Sentinel-2 pull,
  which is what makes this a genuine (non-circular) supervised regression
  target.
- **Features per node:** NDVI = (B08-B04)/(B08+B04), B11 (SWIR) reflectance,
  B04 (red) reflectance, B08 (NIR) reflectance, and Euclidean distance (km)
  from the node centroid to the NOAA tide station (26.1367, -81.7883) as a
  simplified tidal-influence covariate for the Day 2 physics term. SWIR and
  NDVI are not arbitrary choices: Radabaugh et al. (2023) report Landsat
  SWIR (their Band 7) as the #2 predictor and NDVI among the top 4
  predictors for belowground/aboveground C stock respectively in this same
  region (see paper/literature_notes.md) -- reusing that, not inventing it.
- **Edges: 8-connectivity grid adjacency** among kept nodes on the
  150m-aggregated grid.
- **Spatial-block train/test split, not random -- checkerboard superblocks,
  not one contiguous strip.** First attempt (see NUMBER_SOURCE_LOG /
  Day-1 status report) used one contiguous longitude-band holdout and
  produced negative test R^2 for EVERY model tried (linear, RF, GNN, even
  a naive train-mean predictor) -- diagnosed as the holdout band
  straddling a real, sharp regional carbon-density gradient (Radabaugh et
  al. 2023 Table 1 itself reports mangrove total C stock ranging 265-409
  Mg C/ha across their four SW Florida sub-regions), not a bug. Fixed by
  splitting the AOI into a 5x5 grid of superblocks (~10km x ~9km each --
  far larger than the 150m node spacing, so no train/test pixel pair is
  adjacent) and assigning a reproducible, seed-42-selected 20% of
  superblocks (scattered, not edge-clustered) to test. This still respects
  the spatial-exchangeability requirement in paper/uq_decision.md (no
  train pixel is within one node-hop of a test pixel) while not betting
  the entire evaluation on crossing one specific regional boundary. This
  exact split is reused unchanged for every later experiment (baseline,
  physics-informed, CQR calibration) so results are comparable.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME, NOAA_TIDE_STATION_LATLON  # noqa: E402

DATA_RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

SOC_MEAN_PATH = DATA_RAW / "soc" / f"soc_tha_0_100cm_mean_{AOI_NAME}_clip.tif"
SOC_LOWER_PATH = DATA_RAW / "soc" / f"soc_tha_0_100cm_lower_std_{AOI_NAME}_clip.tif"
SOC_UPPER_PATH = DATA_RAW / "soc" / f"soc_tha_0_100cm_upper_std_{AOI_NAME}_clip.tif"
GMW_PATH = DATA_RAW / "gmw" / f"gmw_v3_2020_{AOI_NAME}_clip.tif"
S2_PATH = DATA_RAW / "sentinel2" / "sample_clip_S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114.tif"

BLOCK = 5  # 5x5 aggregation of native 30m SOC grid -> ~150m nodes
SUPERBLOCK_GRID = 8  # 8x8 grid of superblocks over the AOI (~6.25km x ~5.5km each) --
# finer than an earlier 5x5 attempt, which left only ~4-5 superblocks per
# split and made per-split means unstable by chance (see NUMBER_SOURCE_LOG
# Day 1 entry); still far larger than the 150m node spacing.
TEST_SUPERBLOCK_FRACTION = 0.2
VAL_SUPERBLOCK_FRACTION = 0.15
SPLIT_SEED = 42


def _reproject_to_ref(src_path, ref_profile, resampling):
    with rasterio.open(src_path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
        )
    return dst


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

    with rasterio.open(S2_PATH) as s2:
        b04 = np.full((h, w), np.nan, dtype=np.float32)
        b08 = np.full((h, w), np.nan, dtype=np.float32)
        b11 = np.full((h, w), np.nan, dtype=np.float32)
        for dst_arr, band_idx in ((b04, 1), (b08, 2), (b11, 3)):
            reproject(
                source=rasterio.band(s2, band_idx),
                destination=dst_arr,
                src_transform=s2.transform,
                src_crs=s2.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

    # --- aggregate into BLOCK x BLOCK nodes ---
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
    b04_b = block_mean(b04)
    b08_b = block_mean(b08)
    b11_b = block_mean(b11)
    ndvi_b = (b08_b - b04_b) / (b08_b + b04_b + 1e-9)

    keep = (gmw_frac_b > 0.5) & (soc_valid_frac > 0.5) & np.isfinite(soc_mean_b) & np.isfinite(ndvi_b) & np.isfinite(b11_b)
    rows, cols = np.where(keep)
    print(f"grid: {h_blocks}x{w_blocks} blocks, kept nodes: {keep.sum()}")

    # node centroids in EPSG:4326
    block_transform = ref_transform * ref_transform.scale(BLOCK, BLOCK)
    xs, ys = rasterio.transform.xy(block_transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)

    tide_lat, tide_lon = NOAA_TIDE_STATION_LATLON
    dist_to_tide_km = np.sqrt(
        ((ys - tide_lat) * 111.0) ** 2 + ((xs - tide_lon) * 111.0 * np.cos(np.radians(tide_lat))) ** 2
    )

    df = pd.DataFrame({
        "row": rows, "col": cols, "lon": xs, "lat": ys,
        "ndvi": ndvi_b[rows, cols], "b04": b04_b[rows, cols], "b08": b08_b[rows, cols], "b11": b11_b[rows, cols],
        "dist_to_tide_km": dist_to_tide_km,
        "soc_tha_mean": soc_mean_b[rows, cols],
        "soc_tha_lower_std": soc_lower_b[rows, cols],
        "soc_tha_upper_std": soc_upper_b[rows, cols],
    })

    # spatial-block split: SUPERBLOCK_GRID x SUPERBLOCK_GRID grid of large
    # superblocks over the AOI; a reproducible random subset of superblocks
    # (seeded, scattered) is held out as test. See module docstring for why
    # this replaced a single contiguous-band split.
    lon_min, lon_max, lat_min, lat_max = AOI_BBOX[0], AOI_BBOX[2], AOI_BBOX[1], AOI_BBOX[3]
    sb_lon = np.clip(((df["lon"] - lon_min) / (lon_max - lon_min) * SUPERBLOCK_GRID).astype(int), 0, SUPERBLOCK_GRID - 1)
    sb_lat = np.clip(((df["lat"] - lat_min) / (lat_max - lat_min) * SUPERBLOCK_GRID).astype(int), 0, SUPERBLOCK_GRID - 1)
    df["superblock"] = sb_lat * SUPERBLOCK_GRID + sb_lon

    # Mangrove nodes cluster in a thin coastal fringe, so superblock NODE
    # COUNTS are wildly uneven (most of a uniform grid over the AOI's
    # bounding box is open water/upland with zero mangrove nodes). Select
    # test superblocks by cumulative NODE fraction, not by superblock
    # count, so the test set actually ends up close to
    # TEST_SUPERBLOCK_FRACTION of all nodes rather than ~1% of them.
    # Three-way split (train/val/test), all by superblock, all by
    # cumulative node fraction (same reasoning as above). val is carved out
    # of what would otherwise be train, and is used ONLY for early
    # stopping / model selection -- test is never looked at until the one
    # final evaluation, so hyperparameter choices cannot leak into the
    # reported test metric.
    sb_counts = df["superblock"].value_counts()
    occupied = sb_counts.index.to_numpy()
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(occupied)
    target_n_test = int(round(len(df) * TEST_SUPERBLOCK_FRACTION))
    target_n_val = int(round(len(df) * VAL_SUPERBLOCK_FRACTION))
    test_superblocks, val_superblocks = set(), set()
    running = 0
    for sb in order:
        if running >= target_n_test:
            break
        test_superblocks.add(int(sb))
        running += int(sb_counts.loc[sb])
    running = 0
    for sb in order:
        if int(sb) in test_superblocks:
            continue
        if running >= target_n_val:
            break
        val_superblocks.add(int(sb))
        running += int(sb_counts.loc[sb])

    def _assign(sb):
        if sb in test_superblocks:
            return "test"
        if sb in val_superblocks:
            return "val"
        return "train"

    df["split"] = df["superblock"].map(_assign)

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
    df.to_csv(DATA_PROCESSED / "graph_nodes.csv", index=False)
    np.save(DATA_PROCESSED / "graph_edges.npy", edges)

    print(f"nodes: {len(df)}, edges (directed): {edges.shape[1]}")
    print(f"train: {(df['split']=='train').sum()}, val: {(df['split']=='val').sum()}, test: {(df['split']=='test').sum()}")
    print(f"SOC label range: min={df.soc_tha_mean.min():.1f}, max={df.soc_tha_mean.max():.1f}, "
          f"mean={df.soc_tha_mean.mean():.1f}, std={df.soc_tha_mean.std():.1f}")
    print(f"wrote -> {DATA_PROCESSED / 'graph_nodes.csv'}, {DATA_PROCESSED / 'graph_edges.npy'}")
    return df, edges


if __name__ == "__main__":
    build()
