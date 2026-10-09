"""
Historical mangrove loss per node, an observable non-permanence indicator
(paper/revision_preregistration.md Addendum A4).

loss_frac_2010_2020 = (# 30 m cells mangrove in GMW v3 2010 but not 2020)
                      / (# 30 m cells mangrove in 2010), per 5x5 node block.
Both GMW epochs are reprojected (nearest) onto the region's SOC grid, like
build_graph.py's mask. Output: <proc_dir>/graph_nodes_gmw_change.csv.
It is never a model feature; it is used only for the A4 correlation check.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.region import REGIONS, R1  # noqa: E402
from src.data_pipeline.build_graph import BLOCK, _reproject_to_ref  # noqa: E402


def build(region=R1):
    with rasterio.open(region.soc_mean) as r:
        prof = r.profile
    gm = {}
    for ep in ("2010", "2020"):
        p = region.raw_dir / "gmw" / f"gmw_v3_{ep}_{region.name}_clip.tif"
        gm[ep] = np.nan_to_num(_reproject_to_ref(p, prof, Resampling.nearest), nan=0.0) > 0
    lost = (gm["2010"] & ~gm["2020"]).astype(np.float32)
    had = gm["2010"].astype(np.float32)
    h, w = lost.shape
    hb, wb = h // BLOCK, w // BLOCK

    def bsum(a):
        return a[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum(axis=(1, 3))

    L, H = bsum(lost), bsum(had)
    nodes = pd.read_csv(region.nodes_csv, usecols=["row", "col"])
    r, c = nodes.row.values, nodes.col.values
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(H[r, c] > 0, L[r, c] / H[r, c], 0.0)
    out = pd.DataFrame({"row": r, "col": c, "mangrove_loss_frac_2010_2020": frac})
    out.to_csv(region.proc_dir / "graph_nodes_gmw_change.csv", index=False)
    print(f"{region.key}: nodes with any 2010-2020 loss: {(frac > 0).mean():.3f}; "
          f"mean loss frac {frac.mean():.4f}; p95 {np.percentile(frac, 95):.3f}")


if __name__ == "__main__":
    build(REGIONS[sys.argv[1]] if len(sys.argv) > 1 else R1)
