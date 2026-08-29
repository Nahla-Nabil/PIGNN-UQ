"""
Search and pull Sentinel-2 L2A imagery for the pilot AOI from Microsoft
Planetary Computer's STAC API. No API key/account required: assets are
served from public blob storage behind short-lived SAS tokens that the
`planetary_computer` package signs for free.

Two modes:
  --mode search   : query the STAC catalog for the AOI/date range, save the
                     item list (metadata only, small) to data/raw/sentinel2/.
  --mode clip      : additionally windowed-read (via rasterio, over HTTP
                     range requests -- does NOT download full scenes) a few
                     bands for the AOI bbox from one representative
                     low-cloud scene, and save a small AOI-clipped GeoTIFF.

Docs: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
"""
import argparse
import json
import pathlib
import sys

import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "sentinel2"


def search(date_range: str, max_cloud: float = 20.0, limit: int = 20):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=AOI_BBOX,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    items = list(search.items())
    return items


def save_item_index(items, date_range: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": it.id,
            "datetime": it.properties.get("datetime"),
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "mgrs_tile": it.properties.get("s2:mgrs_tile"),
            "bbox": it.bbox,
            "assets": {k: it.assets[k].href for k in ("visual", "B04", "B08", "B11", "SCL") if k in it.assets},
        }
        for it in items
    ]
    out_path = OUT_DIR / f"stac_items_{AOI_NAME}_{date_range.replace('/', '_')}.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} item records -> {out_path}")
    return out_path


def clip_bands(item, bands=("B04", "B08", "B11"), out_name="sample_clip"):
    """Windowed-read (HTTP range requests, not full download) the given
    bands over AOI_BBOX from one STAC item and save as a stacked GeoTIFF.
    Bands at coarser native resolution (e.g. 20 m B11) are resampled to
    match the first (finest, 10 m) band's grid so they stack cleanly --
    the real pipeline will do this resampling anyway before feeding a GNN."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_name}_{item.id}.tif"

    arrays, profile, target_shape = [], None, None
    for b in bands:
        href = item.assets[b].href
        with rasterio.open(href) as src:
            left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            if target_shape is None:
                data = src.read(1, window=window)
                target_shape = data.shape
                profile = src.profile.copy()
                profile.update(
                    height=target_shape[0],
                    width=target_shape[1],
                    count=len(bands),
                    transform=src.window_transform(window),
                )
            else:
                data = src.read(1, window=window, out_shape=target_shape)
            arrays.append(data)
    import numpy as np

    stacked = np.stack(arrays)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)
        dst.descriptions = bands
    print(f"wrote AOI-clipped {stacked.shape} sample ({bands}) -> {out_path}")
    return out_path


GREEN_BAND_ITEM_ID = "S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114"
# The exact scene already used for B04/B08/B11 (sample_clip_<this id>.tif).
# Phase 1C (NUMBER_SOURCE_LOG.md) needs B03 (Green) for a standard McFeeters
# NDWI = (Green-NIR)/(Green+NIR), which is not derivable from the 3 bands
# already on disk. Rather than a fresh scene search, this re-fetches ONE
# additional band from the SAME already-identified item by id -- disclosed
# explicitly in NUMBER_SOURCE_LOG.md as a scoped, minimal exception to the
# "no new data pulls" instruction, made necessary by a gap that instruction
# didn't anticipate (confirmed with the user before pulling).


def fetch_green_band(item_id: str = GREEN_BAND_ITEM_ID):
    """Fetch B03 (Green, 10m) for the exact scene already used for
    B04/B08/B11, windowed identically to clip_bands() so it reprojects onto
    the same grid in build_graph.py. Writes a single-band GeoTIFF."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    results = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(results.items())
    if not items:
        raise RuntimeError(f"item {item_id} not found via STAC search")
    item = items[0]
    out_path = clip_bands(item, bands=("B03",), out_name="sample_clip_B03")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "clip", "fetch-green"], default="search")
    ap.add_argument("--date-range", default="2026-01-01/2026-07-01")
    ap.add_argument("--max-cloud", type=float, default=20.0)
    args = ap.parse_args()

    if args.mode == "fetch-green":
        fetch_green_band()
    else:
        items = search(args.date_range, args.max_cloud)
        print(f"found {len(items)} scenes with <{args.max_cloud}% cloud cover over AOI, {args.date_range}")
        save_item_index(items, args.date_range)

        if args.mode == "clip" and items:
            best = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
            print(f"clipping bands from lowest-cloud scene: {best.id} ({best.properties.get('eo:cloud_cover')}% cloud)")
            clip_bands(best)
