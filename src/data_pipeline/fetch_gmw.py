"""
Download one epoch of the Global Mangrove Watch v3 raster extent mask
(Bunting et al., Zenodo doi:10.5281/zenodo.6894273, CC-BY-4.0) and clip it
to the pilot AOI.

GMW v3 is distributed as global per-epoch zips (~65-70 MB for the raster
"gtiff" product; no country/tile-scoped download exists at the source), so
we download the full global raster for one epoch, clip locally to the AOI,
and discard the global intermediate to keep the repo small.

For a multi-epoch change signal, re-run with a different --epoch and diff
the resulting AOI-clipped rasters (see data/README.md).
"""
import argparse
import pathlib
import shutil
import sys
import zipfile

import requests
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box, mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME  # noqa: E402

ZENODO_RECORD = "6894273"
RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "gmw"
TMP_DIR = RAW_DIR / "_tmp"


def download_epoch_zip(epoch: str) -> pathlib.Path:
    fname = f"gmw_v3_{epoch}_gtiff.zip"
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{fname}/content"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_DIR / fname
    if out_path.exists():
        print(f"already downloaded: {out_path}")
        return out_path
    print(f"downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"downloaded {out_path.stat().st_size / 1e6:.1f} MB -> {out_path}")
    return out_path


def clip_to_aoi(zip_path: pathlib.Path, epoch: str) -> pathlib.Path:
    with zipfile.ZipFile(zip_path) as zf:
        tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
        assert tif_names, f"no .tif found in {zip_path}"
        zf.extractall(TMP_DIR)

    aoi_geom = [mapping(box(*AOI_BBOX))]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"gmw_v3_{epoch}_{AOI_NAME}_clip.tif"

    # GMW gtiff products are tiled globally (~1deg tiles); the AOI can span
    # more than one tile, so mosaic every intersecting tile before clipping.
    intersecting = []
    for name in tif_names:
        tif_path = TMP_DIR / name
        src = rasterio.open(tif_path)
        src_bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        if box(*src_bounds).intersects(box(*AOI_BBOX)):
            intersecting.append(src)
        else:
            src.close()
    if not intersecting:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        raise RuntimeError("No GMW tile intersected the AOI bbox -- check AOI_BBOX in aoi.py")

    mosaic, mosaic_transform = merge(intersecting)
    profile = intersecting[0].profile.copy()
    for src in intersecting:
        src.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform)

    tmp_mosaic_path = TMP_DIR / "_mosaic.tif"
    with rasterio.open(tmp_mosaic_path, "w", **profile) as dst:
        dst.write(mosaic)

    with rasterio.open(tmp_mosaic_path) as src:
        clipped, transform = mask(src, aoi_geom, crop=True)
        profile = src.profile.copy()
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped)
    print(f"mosaicked {len(intersecting)} tile(s), clipped to AOI -> {out_path}")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="2020", help="one of: 1996 2007 2008 2009 2010 2015 2016 2017 2018 2019 2020 core union")
    ap.add_argument("--keep-global-zip", action="store_true")
    args = ap.parse_args()

    zpath = download_epoch_zip(args.epoch)
    out = clip_to_aoi(zpath, args.epoch)
    if not args.keep_global_zip:
        zpath.unlink(missing_ok=True)
    print(f"done -> {out}")
