"""
Pull NOAA CO-OPS water-level (tide) observations and high/low predictions for
the pilot AOI's tide station, as a proxy hydrological driver for the
mass-balance physics term (tidal export of sediment/carbon).

Station: 8725114 - Naples Bay, North, FL (edge of Ten Thousand Islands /
Rookery Bay NERR mangrove fringe). No API key required.

Docs: https://api.tidesandcurrents.noaa.gov/api/prod/
"""
import argparse
import pathlib
import sys

import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.aoi import NOAA_TIDE_STATION_ID  # noqa: E402

BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "tides"


def fetch_water_level(begin_date: str, end_date: str, station: str = NOAA_TIDE_STATION_ID) -> pd.DataFrame:
    params = {
        "product": "water_level",
        "application": "blue-carbon-pinn-gnn",
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": "MLLW",
        "station": station,
        "time_zone": "lst",
        "units": "metric",
        "format": "json",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    df = pd.DataFrame(payload["data"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.rename(columns={"t": "timestamp", "v": "water_level_m"})[["timestamp", "water_level_m"]]


def fetch_high_low_predictions(begin_date: str, end_date: str, station: str = NOAA_TIDE_STATION_ID) -> pd.DataFrame:
    params = {
        "product": "predictions",
        "application": "blue-carbon-pinn-gnn",
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": "MLLW",
        "station": station,
        "time_zone": "lst",
        "units": "metric",
        "interval": "hilo",
        "format": "json",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    df = pd.DataFrame(payload["predictions"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.rename(columns={"t": "timestamp", "v": "predicted_level_m", "type": "hi_lo"})


if __name__ == "__main__":
    # NOTE: NOAA's 6-min "water_level" product caps requests at 31 days per
    # call. For a full multi-year time series, loop this script over
    # consecutive <=31-day windows and concatenate.
    ap = argparse.ArgumentParser()
    ap.add_argument("--begin", default="20260701")
    ap.add_argument("--end", default="20260731")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wl = fetch_water_level(args.begin, args.end)
    wl_path = OUT_DIR / f"water_level_{NOAA_TIDE_STATION_ID}_{args.begin}_{args.end}.csv"
    wl.to_csv(wl_path, index=False)
    print(f"wrote {len(wl)} rows -> {wl_path}")

    hilo = fetch_high_low_predictions(args.begin, args.end)
    hilo_path = OUT_DIR / f"hilo_predictions_{NOAA_TIDE_STATION_ID}_{args.begin}_{args.end}.csv"
    hilo.to_csv(hilo_path, index=False)
    print(f"wrote {len(hilo)} rows -> {hilo_path}")
