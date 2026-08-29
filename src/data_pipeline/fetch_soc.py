"""
Global mangrove soil organic carbon (SOC) stocks, 0-100cm depth, 30m
(Sanderman / Worthington / Woods Hole Research Center et al., Zenodo
doi:10.5281/zenodo.7729492, CC-BY-4.0).

Two products, both used here:
  1. Mangrove typology COG (windowed-read only, no full download needed --
     see clip_typology()).
  2. Actual per-pixel SOC stock values (t C / ha, 0-100cm), which ship
     ONLY inside a 4.7GB tiled zip with no COG mosaic -- see
     extract_and_clip_soc_tiles(). This was downloaded in full in this
     session (Day 1 of the ICDABI sprint, 2026-08-29) after the original
     scaffolding session flagged it as too large; it took ~5 minutes over
     this sandbox's connection, so "too large" was a stale assessment, not
     a hard blocker -- re-attempted once real training depended on it.

Tile naming convention (reverse-engineered by inspection, not documented
by the source): tiles are named "<lon>W_<lat>N" where the numeric values
are the tile's EASTERN/NORTHERN edge, i.e. tile "081W_26N" spans
lon -82.0 to -81.0, lat 26.0 to 27.0. AOI_BBOX = (-81.90, 25.90, -81.40,
26.30) requires tiles 081W_26N (covers lat 26.0-26.30 of the AOI) and
081W_25N (covers lat 25.90-26.00 of the AOI).
"""
import pathlib
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME  # noqa: E402

TYPOLOGY_COG_URL = (
    "https://zenodo.org/api/records/7729492/files/mangroves_typology_v3_cog.tif/content"
)
SOC_ZIP_URL = "https://zenodo.org/api/records/7729492/files/mangroves_tiles_SOC_predictions_2020.zip/content"
SOC_ZIP_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "soc" / "_bg_download" / "mangroves_tiles_SOC_predictions_2020.zip"
EXTRACT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "soc" / "_extracted"
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "soc"

AOI_AVAILABLE_TILES = ["081W_26N", "081W_25N"]
SOC_VARIANTS = {
    "mean": "sol_soc.tha_mangroves.typology_m_30m_s0..100cm_2020_global_v1.1.tif",
    "lower_std": "sol_soc.tha_mangroves.typology_l.std_30m_s0..100cm_2020_global_v1.1.tif",
    "upper_std": "sol_soc.tha_mangroves.typology_u.std_30m_s0..100cm_2020_global_v1.1.tif",
}


def clip_typology():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mangrove_typology_{AOI_NAME}_clip.tif"
    vsi_url = "/vsicurl/" + TYPOLOGY_COG_URL
    with rasterio.open(vsi_url) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(window))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    print(f"windowed-read {data.shape} -> {out_path}")
    return out_path


def _extract_variant_tiles(variant: str) -> list[pathlib.Path]:
    with zipfile.ZipFile(SOC_ZIP_PATH) as z:
        members = [f"tt_final/{t}/{SOC_VARIANTS[variant]}" for t in AOI_AVAILABLE_TILES]
        members = [m for m in members if m in z.namelist()]
        z.extractall(EXTRACT_DIR, members=members)
    return [EXTRACT_DIR / m for m in members]


def extract_and_clip_soc_tiles():
    """Mosaic the AOI-covering tile(s) for each SOC variant (mean, lower_std,
    upper_std) and clip to AOI_BBOX. Requires SOC_ZIP_PATH to already exist
    (see module docstring / data/README.md for the download command)."""
    if not SOC_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"{SOC_ZIP_PATH} not found. Download it first (one-time, ~4.7GB):\n"
            f"  curl -L -o \"{SOC_ZIP_PATH}\" \"{SOC_ZIP_URL}\""
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aoi_geom = [mapping(box(*AOI_BBOX))]
    outputs = {}
    for variant in SOC_VARIANTS:
        tile_paths = _extract_variant_tiles(variant)
        srcs = [rasterio.open(p) for p in tile_paths]
        mosaic, mosaic_transform = merge(srcs)
        profile = srcs[0].profile.copy()
        for s in srcs:
            s.close()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform)

        tmp_path = EXTRACT_DIR / f"_mosaic_{variant}.tif"
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(mosaic)

        out_path = OUT_DIR / f"soc_tha_0_100cm_{variant}_{AOI_NAME}_clip.tif"
        with rasterio.open(tmp_path) as src:
            clipped, transform = mask(src, aoi_geom, crop=True, nodata=src.nodata)
            profile = src.profile.copy()
            profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(clipped)
        outputs[variant] = out_path
        print(f"{variant}: mosaicked {len(tile_paths)} tile(s), clipped -> {out_path}")

    return outputs


def sanity_check_against_radabaugh(mean_path: pathlib.Path):
    """AOI-wide plausibility check only (NOT a per-pixel validation) against
    Radabaugh et al. (2023) Table 1, mangrove ecosystem type, Ten Thousand
    Islands region: belowground C stock (0-100cm, comparable depth/quantity
    to this SOC layer) = 383.6 +/- 87.7 Mg C ha-1."""
    with rasterio.open(mean_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        valid = arr[arr != nodata] if nodata is not None else arr
        valid = valid[valid > 0]  # drop non-mangrove/ocean 0s for a mangrove-only comparison
    print(f"AOI SOC (t/ha, 0-100cm) over nonzero pixels: n={valid.size}, "
          f"mean={valid.mean():.1f}, std={valid.std():.1f}, "
          f"median={np.median(valid):.1f}")
    print("Radabaugh et al. (2023) Table 1, Ten Thousand Islands, mangrove, "
          "belowground C stock (0-100cm): 383.6 +/- 87.7 Mg C ha-1 (reference, not identical quantity -- "
          "SOC-only vs. belowground incl. roots; see data/README.md for the caveat).")


if __name__ == "__main__":
    clip_typology()
    outputs = extract_and_clip_soc_tiles()
    print()
    sanity_check_against_radabaugh(outputs["mean"])
