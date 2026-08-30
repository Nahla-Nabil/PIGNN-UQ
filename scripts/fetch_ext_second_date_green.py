"""
Journal-extension sprint, Step 1 (temporal dC/dt experiment).

Fetches B03 (Green, 10m) for the SECOND Sentinel-2 date's scene, the same
minimal-exception pattern `fetch_sentinel2.py:fetch_green_band()` already
used for the original (Day-1-2 / Phase-0) date -- reused here unmodified,
just called with the new scene's item id instead of the hardcoded default,
so B03 is available for an NDWI delta alongside the NDVI delta.

Does not touch any existing file (data/raw/sentinel2/sample_clip_B03_...
for the ORIGINAL scene is untouched; this writes a NEW file for the new
scene id only).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_pipeline.fetch_sentinel2 import fetch_green_band  # noqa: E402

SECOND_DATE_ITEM_ID = "S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532"

if __name__ == "__main__":
    out_path = fetch_green_band(item_id=SECOND_DATE_ITEM_ID)
    print(f"second-date B03 -> {out_path}")
