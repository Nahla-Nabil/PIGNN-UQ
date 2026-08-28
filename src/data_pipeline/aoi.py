"""
Single source of truth for the pilot Area of Interest (AOI).

Pilot region: Ten Thousand Islands / Rookery Bay NERR margin, Southwest
Florida, USA (western edge of Everglades National Park mangrove complex).

Chosen because (see data/README.md for full justification):
  - Sits inside Sentinel-2 MGRS tile 17RMK.
  - Has an on-site NOAA CO-OPS tide gauge (station 8725114, Naples Bay North).
  - Covered by Global Mangrove Watch extent time series (Zenodo 6894273).
  - Covered by the global mangrove soil organic carbon COG (Zenodo 7729492).
  - Independently validated by Radabaugh et al. (2023, Estuaries and Coasts,
    doi:10.1007/s12237-023-01217-7), a field-plot-based blue carbon stock
    model for Tampa Bay -> Everglades, Southwest Florida -- our external
    validation reference.
"""

# Bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max).
# ~40km x 40km box covering Ten Thousand Islands / Rookery Bay NERR mangrove
# fringe, chosen to sit fully inside Sentinel-2 tile T17RMK and within range
# of NOAA station 8725114.
AOI_BBOX = (-81.90, 25.90, -81.40, 26.30)

AOI_NAME = "ten_thousand_islands_fl"

SENTINEL2_MGRS_TILE = "17RMK"

NOAA_TIDE_STATION_ID = "8725114"
NOAA_TIDE_STATION_NAME = "Naples Bay, North, FL"
NOAA_TIDE_STATION_LATLON = (26.1367, -81.7883)

CRS_WGS84 = "EPSG:4326"
