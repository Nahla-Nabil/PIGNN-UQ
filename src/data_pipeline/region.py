"""
Region configuration for the revision (paper/revision_preregistration.md
Addendum A3). Region 1 is the submitted paper's AOI, with paths unchanged.
Region 2 is the pre-registered second region (Everglades NP coast, bbox
fixed in Addendum A3 before its labels were viewed).
"""
import pathlib
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


@dataclass
class Region:
    key: str
    name: str
    bbox: tuple  # (lon_min, lat_min, lon_max, lat_max)
    raw_dir: pathlib.Path
    proc_dir: pathlib.Path
    tide_station: str
    tide_latlon: tuple
    s2_items: dict = field(default_factory=dict)  # {"t0": [...], "t1": [...]}

    @property
    def soc_mean(self):
        return self.raw_dir / "soc" / f"soc_tha_0_100cm_mean_{self.name}_clip.tif"

    @property
    def soc_lower(self):
        return self.raw_dir / "soc" / f"soc_tha_0_100cm_lower_std_{self.name}_clip.tif"

    @property
    def soc_upper(self):
        return self.raw_dir / "soc" / f"soc_tha_0_100cm_upper_std_{self.name}_clip.tif"

    @property
    def gmw(self):
        return self.raw_dir / "gmw" / f"gmw_v3_2020_{self.name}_clip.tif"

    @property
    def nodes_csv(self):
        return self.proc_dir / "graph_nodes.csv"

    @property
    def edges_npy(self):
        return self.proc_dir / "graph_edges.npy"

    @property
    def rev_features_csv(self):
        return self.proc_dir / "graph_nodes_rev_features.csv"

    @property
    def hydro_csv(self):
        return self.proc_dir / "graph_nodes_hydroperiod.csv"

    @property
    def radabaugh_csv(self):
        return self.proc_dir / "graph_nodes_radabaugh_ref.csv"


R1 = Region(
    key="r1", name="ten_thousand_islands_fl", bbox=(-81.90, 25.90, -81.40, 26.30),
    raw_dir=RAW, proc_dir=PROC, tide_station="8725114", tide_latlon=(26.1367, -81.7883),
    s2_items={
        "t0": ["S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114",
               "S2B_MSIL2A_20260206T160339_R097_T17RMK_20260206T213114"],
        "t1": ["S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532",
               "S2B_MSIL2A_20260815T155819_R097_T17RMK_20260815T195532"],
    },
)

# Region 2: tide station and S2 items are filled by the region-2 build
# (nearest CO-OPS station with 2025 NAVD88 hourly heights; same-datatake
# S2 scenes if they cover the bbox), per Addendum A3.
R2 = Region(
    key="r2", name="everglades_coast_fl", bbox=(-81.20, 25.30, -80.90, 25.60),
    raw_dir=RAW / "region2", proc_dir=PROC / "region2", tide_station="", tide_latlon=(None, None),
)

REGIONS = {"r1": R1, "r2": R2}
