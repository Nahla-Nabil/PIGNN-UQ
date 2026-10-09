"""
Revision feature build (paper/revision_preregistration.md §2 and §8).

**Georeferencing fix (found 2026-10-07, revision Deviation D1).** The
submitted paper's Sentinel-2 features came from
data/raw/sentinel2/sample_clip_*.tif, written by
fetch_sentinel2.clip_bands(). Its AOI window extended 918 rows (9.18 km)
north of tile T17RMJ's top edge. rasterio's non-boundless read returned
tile rows starting at row 0, but the file was written with the *window's*
transform, so every pixel is labeled 9.18 km north of its true location.
Verified: local clip row 0 == tile row 0 exactly (corr 1.0000, 100% of pixels
identical). The AOI's northern part (lat > ~26.21) is also not covered by
T17RMJ at all; it lies in T17RMK (same datatake). This module rebuilds all
Sentinel-2 features from the correctly georeferenced tiles (T17RMJ + T17RMK
mosaic, same date), without touching data/processed/graph_nodes.csv or its
split column.

Output: data/processed/graph_nodes_rev_features.csv, keyed by (row, col).
Grid/aggregation match build_graph.py: each raster is reprojected
(bilinear; SCL nearest) onto the native 30 m SOC grid, then 5x5
block-meaned onto the node grid. Cloud/shadow pixels (SCL 3, 8, 9, 10) are
masked before the block mean; nodes with no clear pixel get NaN (handled at
load time). Bands are read remotely from Planetary Computer with a window
around the AOI, so nothing large is stored.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.build_graph import BLOCK  # noqa: E402
from src.data_pipeline.region import REGIONS, R1  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# S2 items (both MGRS tiles over region 1; same datatakes for both dates)
# now live in src/data_pipeline/region.py.
BANDS = {
    "t0": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "t1": ["B03", "B04", "B05", "B08", "B8A", "B11"],
}
CLOUD_SCL = [3, 8, 9, 10]  # cloud shadow, cloud medium/high prob., thin cirrus


def _ref(region=R1):
    with rasterio.open(region.soc_mean) as ref:
        return ref.transform, ref.crs, ref.shape


def _warp_remote(href, ref_transform, ref_crs, shape, resampling=Resampling.bilinear, bbox=R1.bbox,
                 zero_is_nodata=True):
    """Windowed boundless read around the bbox, warped onto the SOC grid.
    Boundless + the window's own transform keeps georeferencing correct
    even where the bbox runs past the tile edge (the bug this module fixes).
    zero_is_nodata: True for S2 L2A (nodata 0); False for DEMs (0 m is a
    real elevation; their own nodata value is used instead)."""
    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, bbox[0] - 0.02, bbox[1] - 0.02, bbox[2] + 0.02, bbox[3] + 0.02)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        fill = 0 if zero_is_nodata or src.nodata is None else src.nodata
        arr = src.read(1, window=win, boundless=True, fill_value=fill).astype(np.float32)
        if zero_is_nodata:
            arr[arr == 0] = np.nan  # S2 L2A nodata is 0 (SCL 0 = no data as well)
        elif src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        reproject(source=arr, destination=dst, src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs, resampling=resampling,
                  src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def _mosaic(items, asset, ref, resampling=Resampling.bilinear, bbox=R1.bbox):
    out = None
    for it in items:
        a = _warp_remote(it.assets[asset].href, *ref, resampling, bbox=bbox)
        out = a if out is None else np.where(np.isnan(out), a, out)
    return out


def _block(arr, fn=np.nanmean):
    h, w = arr.shape
    hb, wb = h // BLOCK, w // BLOCK
    r = arr[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(r, axis=(1, 3))


def nd(a, b):
    return (a - b) / (a + b + 1e-9)


def build(region=R1):
    ref = _ref(region)
    nodes = pd.read_csv(region.nodes_csv, usecols=["row", "col", "b04"])
    rows, cols = nodes["row"].values, nodes["col"].values
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    s2 = cat.get_collection("sentinel-2-l2a")

    out = pd.DataFrame({"row": rows, "col": cols})
    for tag, ids in region.s2_items.items():
        items = [s2.get_item(i) for i in ids]
        scl = _mosaic(items, "SCL", ref, Resampling.nearest, bbox=region.bbox)
        cloud = np.isin(scl, CLOUD_SCL)
        out[f"{tag}_cloud_frac"] = _block(cloud.astype(np.float32))[rows, cols]
        for band in BANDS[tag]:
            print(f"{tag} {band} ...", flush=True)
            arr = _mosaic(items, band, ref, bbox=region.bbox)
            arr[cloud] = np.nan
            out[f"{tag}_{band.lower()}"] = _block(arr)[rows, cols]

    dem = np.full(ref[2], np.nan, dtype=np.float32)
    for it in cat.search(collections=["cop-dem-glo-30"], bbox=list(region.bbox)).items():
        print(f"DEM {it.id} ...", flush=True)
        tile = _warp_remote(it.assets["data"].href, *ref, bbox=region.bbox)  # kept as built for region 1
        dem = np.where(np.isnan(dem), tile, dem)
    # GLO-30 is a DSM; over this coast it is 0 m wherever the producer's
    # water-body edit applies, so the 0s are kept (not NaN) and stated.
    out["elevation"] = np.nan_to_num(_block(dem)[rows, cols], nan=0.0)
    out["elevation_std"] = np.nan_to_num(_block(dem, np.nanstd)[rows, cols], nan=0.0)

    # Corrected versions of the submitted paper's base columns (t0 date)
    out["b03"], out["b04"], out["b08"], out["b11"] = out.t0_b03, out.t0_b04, out.t0_b08, out.t0_b11
    out["ndvi"] = nd(out.t0_b08, out.t0_b04)
    out["ndwi"] = nd(out.t0_b03, out.t0_b08)
    out["ndmi"] = nd(out.t0_b08, out.t0_b11)
    for c in ("b02", "b05", "b06", "b07", "b8a", "b12"):
        out[c] = out[f"t0_{c}"]
    out["ndre"] = nd(out.t0_b8a, out.t0_b05)
    out["mndwi"] = nd(out.t0_b03, out.t0_b11)
    # second date (cloud-masked) and change terms
    out["t1_ndvi"] = nd(out.t1_b08, out.t1_b04)
    out["t1_ndre"] = nd(out.t1_b8a, out.t1_b05)
    out["t1_ndmi"] = nd(out.t1_b08, out.t1_b11)
    out["t1_mndwi"] = nd(out.t1_b03, out.t1_b11)
    out["d_ndvi"] = out.t1_ndvi - out.ndvi
    out["d_ndmi"] = out.t1_ndmi - out.ndmi

    print(f"corr(corrected b04, submitted b04) = {np.corrcoef(out.b04.fillna(out.b04.mean()), nodes.b04)[0, 1]:.4f}")
    print(f"NDVI corrected: mean={out.ndvi.mean():.3f}, frac<0.3={np.mean(out.ndvi < 0.3):.3f}")
    print(f"cloud frac mean t0={out.t0_cloud_frac.mean():.3f}, t1={out.t1_cloud_frac.mean():.3f}")
    na = out.isna().sum()
    print(f"NaN counts:\n{na[na > 0]}")
    out.to_csv(region.rev_features_csv, index=False)
    print(f"wrote {region.rev_features_csv} ({len(out)} rows)")


if __name__ == "__main__":
    build(REGIONS[sys.argv[1]] if len(sys.argv) > 1 else R1)
