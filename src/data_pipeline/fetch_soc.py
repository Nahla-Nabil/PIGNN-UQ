"""
Windowed-read (HTTP range requests via GDAL /vsicurl, NOT a full download)
the AOI extent from the global mangrove typology Cloud-Optimized GeoTIFF
published alongside the global mangrove soil organic carbon (SOC) stocks
dataset (Sanderman / Woods Hole Research Center et al., Zenodo
doi:10.5281/zenodo.7729492, CC-BY-4.0).

IMPORTANT LIMITATION (documented in data/README.md): the actual per-pixel
SOC stock predictions ship only inside a single ~4.7 GB
"mangroves_tiles_SOC_predictions_2020.zip" with no COG mosaic and no
tile-scoped download endpoint -- too large to pull in this sandbox session.
This script fetches the companion typology COG (ecotype/zonation class,
1.1 GB source but windowed-read costs only the AOI's tiles) as an
immediately-usable proxy layer, and documents the exact local command to
extract just the Florida tile(s) from the full SOC zip.
"""
import pathlib
import sys

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME  # noqa: E402

TYPOLOGY_COG_URL = (
    "https://zenodo.org/api/records/7729492/files/mangroves_typology_v3_cog.tif/content"
)
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "soc"


def clip_typology():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mangrove_typology_{AOI_NAME}_clip.tif"

    # /vsicurl/ tells GDAL to treat the URL as a remote file and use HTTP
    # range requests, exploiting the COG's internal tiling -- only the
    # bytes covering the AOI window are transferred.
    vsi_url = "/vsicurl/" + TYPOLOGY_COG_URL
    with rasterio.open(vsi_url) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(window))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    print(f"windowed-read {data.shape} -> {out_path} (no full 1.1GB file downloaded)")
    return out_path


if __name__ == "__main__":
    clip_typology()
    print()
    print("NOTE: full per-pixel SOC stock values require the 4.7GB zip below.")
    print("Run locally (not in this sandbox session) to extract just the Florida tile:")
    print(
        "  curl -L -o mangroves_tiles_SOC_predictions_2020.zip "
        "https://zenodo.org/api/records/7729492/files/mangroves_tiles_SOC_predictions_2020.zip/content"
    )
    print("  # then inspect the archive's tile index / filenames and extract only tiles")
    print("  # intersecting AOI_BBOX from src/data_pipeline/aoi.py, e.g.:")
    print("  unzip -l mangroves_tiles_SOC_predictions_2020.zip | grep -i <florida_tile_id>")
