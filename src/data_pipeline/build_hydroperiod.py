"""
Hydroperiod prior inputs (paper/revision_preregistration.md Addendum A1,
prior P1).

Per node:
  elev_navd88_m  mean USGS 3DEP 1/3 arc-second (~10 m) bare-earth elevation
                 (NAVD88, m), warped onto the 30 m SOC grid and 5x5
                 block-meaned like every other layer
  inund_freq     fraction of hours in 2025 at which the NOAA CO-OPS hourly
                 height (datum NAVD88, m) exceeds elev_navd88_m

One gauge's water level stands in for the whole region; this is a stated
simplification (tidal range varies along the coast).
Output: <region.proc_dir>/graph_nodes_hydroperiod.csv keyed by (row, col).
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import requests
from rasterio.warp import Resampling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.region import REGIONS, R1  # noqa: E402
from src.data_pipeline.build_rev_features import _ref, _warp_remote, _block, STAC_URL  # noqa: E402

COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
YEAR_BEGIN, YEAR_END = "20250101", "20251231"


def hourly_heights_navd(station):
    """One year of hourly heights in NAVD88 (one request per month to stay
    within CO-OPS limits)."""
    frames = []
    for m in range(1, 13):
        b = f"2025{m:02d}01"
        e = f"2025{m:02d}{pd.Period(f'2025-{m:02d}').days_in_month:02d}"
        r = requests.get(COOPS, params={"product": "hourly_height", "station": station, "begin_date": b,
                                        "end_date": e, "datum": "NAVD", "units": "metric", "time_zone": "gmt",
                                        "format": "json", "application": "bcpg_revision"}, timeout=60)
        js = r.json()
        if "data" not in js:
            raise RuntimeError(f"CO-OPS {station} {b}: {js.get('error')}")
        frames.append(pd.DataFrame(js["data"]))
    df = pd.concat(frames)
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna(subset=["v"])


def build(region=R1):
    ref = _ref(region)
    nodes = pd.read_csv(region.nodes_csv, usecols=["row", "col"])
    rows, cols = nodes.row.values, nodes.col.values
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    dem = np.full(ref[2], np.nan, dtype=np.float32)
    items = [it for it in cat.search(collections=["3dep-seamless"], bbox=list(region.bbox)).items()
             if it.properties.get("gsd") == 10]
    for it in items:
        print(f"3DEP {it.id} ...", flush=True)
        tile = _warp_remote(it.assets["data"].href, *ref, Resampling.bilinear, bbox=region.bbox,
                            zero_is_nodata=False)
        dem = np.where(np.isnan(dem), tile, dem)
    elev = _block(dem)[rows, cols]

    wl = hourly_heights_navd(region.tide_station)
    levels = np.sort(wl.v.values)
    # fraction of hours with water level > node elevation
    inund = 1 - np.searchsorted(levels, elev, side="right") / len(levels)
    inund[np.isnan(elev)] = np.nan

    out = pd.DataFrame({"row": rows, "col": cols, "elev_navd88_m": elev, "inund_freq": inund})
    out.to_csv(region.hydro_csv, index=False)
    print(f"tide hours: {len(levels)}, NAVD88 level p5/p50/p95 = "
          f"{np.percentile(levels, 5):.2f}/{np.percentile(levels, 50):.2f}/{np.percentile(levels, 95):.2f} m")
    print(f"elevation: mean {np.nanmean(elev):.2f} m, p5/p95 {np.nanpercentile(elev, 5):.2f}/"
          f"{np.nanpercentile(elev, 95):.2f}, NaN {np.isnan(elev).sum()}")
    print(f"inund_freq: mean {np.nanmean(inund):.3f}, p5/p95 {np.nanpercentile(inund, 5):.3f}/"
          f"{np.nanpercentile(inund, 95):.3f}")
    print(f"wrote {region.hydro_csv}")


if __name__ == "__main__":
    build(REGIONS[sys.argv[1]] if len(sys.argv) > 1 else R1)
