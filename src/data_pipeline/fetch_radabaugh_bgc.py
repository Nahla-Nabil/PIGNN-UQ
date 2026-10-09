"""
Independent reference for revision E7 (paper/revision_preregistration.md §7):
Radabaugh et al. (2023)'s 30 m belowground carbon stock map (Mg C/ha,
0-1 m), calibrated on their own SW Florida field cores and independent of
the Sanderman SOC product we train on. The paper's Data Availability
statement points to https://gis.myfwc.com/SWFLCarbonStocksMap/, whose web
map (ArcGIS item 50b8a2c855b448209aa5aabeb9301193) serves layer
"BG Carbon Stocks Mg/ha" as a tiles-only ImageServer with raw float32
LERC2D tiles. This module fetches the finest level (level 5,
0.000281771697 deg, about 30 m) over the AOI, decodes the tiles, and writes
a GeoTIFF.
"""
import math
import pathlib
import sys

import lerc
import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import AOI_BBOX, AOI_NAME  # noqa: E402

SERVICE = ("https://tiledimageservices2.arcgis.com/z6TmTIyYXEYhuNM0/arcgis/rest/services/"
           "BlueCRaster_BGC_0824221/ImageServer")
LEVEL, RES, TILE = 5, 0.000281771696999999, 256
ORIGIN_X, ORIGIN_Y = -180.0, 90.0
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "validation_refs" / \
    f"radabaugh2023_bgc_mgha_{AOI_NAME}.tif"


def fetch():
    span = TILE * RES
    c0, c1 = math.floor((AOI_BBOX[0] - ORIGIN_X) / span), math.floor((AOI_BBOX[2] - ORIGIN_X) / span)
    r0, r1 = math.floor((ORIGIN_Y - AOI_BBOX[3]) / span), math.floor((ORIGIN_Y - AOI_BBOX[1]) / span)
    mosaic = np.full(((r1 - r0 + 1) * TILE, (c1 - c0 + 1) * TILE), np.nan, dtype=np.float32)
    n_ok = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            resp = requests.get(f"{SERVICE}/tile/{LEVEL}/{r}/{c}", timeout=60)
            if resp.status_code != 200 or not resp.content:
                continue
            res = lerc.decode(resp.content)
            err, arr, mask = res[0], res[1], res[2]
            if err != 0:
                continue
            arr = np.asarray(arr, dtype=np.float32).reshape(TILE, TILE, -1)[..., 0] if arr.ndim == 3 \
                else np.asarray(arr, dtype=np.float32)
            if mask is not None:
                arr = np.where(np.asarray(mask).reshape(TILE, TILE) > 0, arr, np.nan)
            mosaic[(r - r0) * TILE:(r - r0 + 1) * TILE, (c - c0) * TILE:(c - c0 + 1) * TILE] = arr
            n_ok += 1
    transform = from_origin(ORIGIN_X + c0 * span, ORIGIN_Y - r0 * span, RES, RES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT, "w", driver="GTiff", height=mosaic.shape[0], width=mosaic.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan) as dst:
        dst.write(mosaic, 1)
    v = mosaic[np.isfinite(mosaic)]
    print(f"tiles decoded: {n_ok}; valid px {v.size}; mean {v.mean():.1f}, min {v.min():.1f}, max {v.max():.1f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    fetch()


def to_nodes():
    """Average the Radabaugh map onto the node grid (same reprojection onto
    the 30 m SOC grid + 5x5 block mean as every other layer). Written to its
    own file: it is an evaluation reference, never a model feature."""
    from rasterio.warp import Resampling
    import pandas as pd
    from src.data_pipeline.build_rev_features import _ref, _block
    from src.data_pipeline.build_graph import _reproject_to_ref, SOC_MEAN_PATH, DATA_PROCESSED
    with rasterio.open(SOC_MEAN_PATH) as r:
        prof = r.profile
    arr = _reproject_to_ref(OUT, prof, Resampling.bilinear)
    nodes = pd.read_csv(DATA_PROCESSED / "graph_nodes.csv", usecols=["row", "col", "soc_tha_mean", "split"])
    nodes["radabaugh_bgc_mgha"] = _block(arr)[nodes.row.values, nodes.col.values]
    nodes[["row", "col", "radabaugh_bgc_mgha"]].to_csv(DATA_PROCESSED / "graph_nodes_radabaugh_ref.csv", index=False)
    v = nodes.dropna(subset=["radabaugh_bgc_mgha"])
    print(f"nodes with reference: {len(v)}/{len(nodes)}; "
          f"corr(Sanderman label, Radabaugh) = {np.corrcoef(v.soc_tha_mean, v.radabaugh_bgc_mgha)[0, 1]:.3f}; "
          f"means {v.soc_tha_mean.mean():.1f} vs {v.radabaugh_bgc_mgha.mean():.1f}")
    print(v.groupby("split").size().to_dict())
