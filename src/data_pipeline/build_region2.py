"""
Second-region build (paper/revision_preregistration.md Addendum A3).

Region 2 = Everglades NP coast, bbox fixed in the addendum before its labels
were viewed. Same construction as region 1 (build_graph.py):
- Sanderman et al. 30 m SOC (mean, lower/upper std) as the reference grid
  and label; tiles are read *remotely* from the Zenodo zip via HTTP range
  requests (no 4.7 GB download);
- GMW v3 2020 mask; node = 5x5 SOC-cell block with GMW fraction > 0.5 and
  valid-SOC fraction > 0.5; 8-connectivity edges;
- superblock (8x8) and fine (20x20, "phase1_superblock") grids over the
  bbox; split 20% test / 15% val / rest train by cumulative node count over
  shuffled fine cells (seed 20261009), the operative procedure region 1's
  split used;
- nearest NOAA CO-OPS water-level station with 2025 NAVD88 hourly heights;
- Sentinel-2: items from the same datatakes as region 1 if they cover the
  bbox, otherwise the nearest date with < 20% cloud, recorded in
  region2_meta.json.

GMW zips (66 MB each) are downloaded once to data/raw/gmw/_zips and also
used for the 2010 epoch (non-permanence check, Addendum A4, both regions).
"""
import io
import json
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box, mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.data_pipeline.region import R1, R2, RAW  # noqa: E402
from src.data_pipeline.build_graph import BLOCK, _reproject_to_ref  # noqa: E402
from rasterio.warp import Resampling  # noqa: E402

SOC_ZIP_URL = "https://zenodo.org/api/records/7729492/files/mangroves_tiles_SOC_predictions_2020.zip/content"
SOC_VARIANTS = {
    "mean": "sol_soc.tha_mangroves.typology_m_30m_s0..100cm_2020_global_v1.1.tif",
    "lower_std": "sol_soc.tha_mangroves.typology_l.std_30m_s0..100cm_2020_global_v1.1.tif",
    "upper_std": "sol_soc.tha_mangroves.typology_u.std_30m_s0..100cm_2020_global_v1.1.tif",
}
GMW_URL = "https://zenodo.org/api/records/6894273/files/gmw_v3_{epoch}_gtiff.zip/content"
GMW_ZIPS = RAW / "gmw" / "_zips"
SPLIT_SEED = 20261009
COOPS_MD = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for remote zips)."""

    def __init__(self, url):
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        d = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300).content
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def _clip_mosaic(paths, bbox, out_path):
    srcs = [rasterio.open(p) for p in paths]
    srcs_in = [s for s in srcs if box(*rasterio.warp.transform_bounds(s.crs, "EPSG:4326", *s.bounds)).intersects(box(*bbox))]
    mos, tr = merge(srcs_in)
    prof = srcs_in[0].profile.copy()
    prof.update(height=mos.shape[1], width=mos.shape[2], transform=tr)
    tmp = out_path.with_suffix(".mosaic.tif")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(mos)
    for s in srcs:
        s.close()
    with rasterio.open(tmp) as src:
        clipped, t2 = mask(src, [mapping(box(*bbox))], crop=True, nodata=src.nodata)
        prof = src.profile.copy()
        prof.update(height=clipped.shape[1], width=clipped.shape[2], transform=t2)
        with rasterio.open(out_path, "w", **prof) as d:
            d.write(clipped)
    tmp.unlink()
    return out_path


def fetch_soc(region):
    out_dir = region.raw_dir / "soc"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(HTTPFile(SOC_ZIP_URL), buffer_size=1 << 22))
    tiles = sorted({n.split("/")[1] for n in z.namelist() if n.startswith("tt_final/")})
    cand = [t for t in tiles if t.endswith(("25N", "24N", "26N")) and t.startswith(("080W", "081W", "082W"))]
    tmp = region.raw_dir / "soc" / "_tiles"
    for variant, fname in SOC_VARIANTS.items():
        paths = []
        for t in cand:
            m = f"tt_final/{t}/{fname}"
            if m in z.namelist():
                p = tmp / t / fname
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(z.read(m))
                paths.append(p)
        out = {"mean": region.soc_mean, "lower_std": region.soc_lower, "upper_std": region.soc_upper}[variant]
        _clip_mosaic(paths, region.bbox, out)
        print(f"SOC {variant}: {len(paths)} candidate tiles -> {out}")


def fetch_gmw(region, epoch):
    GMW_ZIPS.mkdir(parents=True, exist_ok=True)
    zp = GMW_ZIPS / f"gmw_v3_{epoch}_gtiff.zip"
    if not zp.exists():
        with requests.get(GMW_URL.format(epoch=epoch), stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zp, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
    ex = GMW_ZIPS / f"_x_{epoch}"
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif") and ("N25W08" in n or "N26W08" in n or "N27W08" in n)]
        z.extractall(ex, members=names)
    out = region.raw_dir / "gmw" / f"gmw_v3_{epoch}_{region.name}_clip.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    _clip_mosaic([ex / n for n in names], region.bbox, out)
    print(f"GMW {epoch}: {len(names)} tiles -> {out}")
    return out


def nearest_navd_station(lat, lon):
    st = requests.get(COOPS_MD, timeout=60).json()["stations"]
    st = sorted(st, key=lambda s: (s["lat"] - lat) ** 2 + ((s["lng"] - lon) * np.cos(np.radians(lat))) ** 2)
    for s in st[:15]:
        r = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
                         params={"product": "hourly_height", "station": s["id"], "begin_date": "20250101",
                                 "end_date": "20250107", "datum": "NAVD", "units": "metric", "time_zone": "gmt",
                                 "format": "json", "application": "bcpg_revision"}, timeout=60).json()
        if "data" in r and len(r["data"]) > 100:
            return s["id"], s["name"], (s["lat"], s["lng"])
    raise RuntimeError("no NAVD88 hourly station found")


def s2_items(region):
    import planetary_computer as pc
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    out = {}
    for tag, date in (("t0", "2026-02-06"), ("t1", "2026-08-15")):
        its = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(region.bbox), datetime=date).items())
        if not its:
            raise RuntimeError(f"no same-datatake S2 item for {tag} {date}")
        out[tag] = [i.id for i in its]
    return out


def build_graph(region):
    with rasterio.open(region.soc_mean) as ref:
        prof, tr = ref.profile.copy(), ref.transform
        soc = ref.read(1).astype(np.float32)
        nod = ref.nodata
        h, w = ref.shape
    valid = (soc != nod) & (soc > 0)
    lower = _reproject_to_ref(region.soc_lower, prof, Resampling.bilinear)
    upper = _reproject_to_ref(region.soc_upper, prof, Resampling.bilinear)
    gmw = np.nan_to_num(_reproject_to_ref(region.gmw, prof, Resampling.nearest), nan=0.0)
    hb, wb = h // BLOCK, w // BLOCK

    def bm(a):
        r = a[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK)
        with np.errstate(invalid="ignore"):
            return np.nanmean(r, axis=(1, 3))

    vf = bm(valid.astype(np.float32))
    sm, sl, su = bm(np.where(valid, soc, np.nan)), bm(np.where(valid, lower, np.nan)), bm(np.where(valid, upper, np.nan))
    gf = bm(gmw)
    keep = (gf > 0.5) & (vf > 0.5) & np.isfinite(sm)
    rows, cols = np.where(keep)
    xs, ys = rasterio.transform.xy(tr * tr.scale(BLOCK, BLOCK), rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    tlat, tlon = region.tide_latlon
    dist = np.sqrt(((ys - tlat) * 111.0) ** 2 + ((xs - tlon) * 111.0 * np.cos(np.radians(tlat))) ** 2)
    df = pd.DataFrame({"row": rows, "col": cols, "lon": xs, "lat": ys, "dist_to_tide_km": dist,
                       "soc_tha_mean": sm[rows, cols], "soc_tha_lower_std": sl[rows, cols],
                       "soc_tha_upper_std": su[rows, cols]})
    b = region.bbox

    def grid(n):
        gx = np.clip(((df.lon - b[0]) / (b[2] - b[0]) * n).astype(int), 0, n - 1)
        gy = np.clip(((df.lat - b[1]) / (b[3] - b[1]) * n).astype(int), 0, n - 1)
        return gy * n + gx

    df["superblock"], df["phase1_superblock"] = grid(8), grid(20)
    counts = df.phase1_superblock.value_counts()
    order = np.random.default_rng(SPLIT_SEED).permutation(counts.index.to_numpy())
    n_test, n_val = round(0.20 * len(df)), round(0.15 * len(df))
    test, val, run = set(), set(), 0
    for sb in order:
        if run >= n_test:
            break
        test.add(int(sb))
        run += int(counts.loc[sb])
    run = 0
    for sb in order:
        if int(sb) in test:
            continue
        if run >= n_val:
            break
        val.add(int(sb))
        run += int(counts.loc[sb])
    df["split"] = df.phase1_superblock.map(lambda s: "test" if s in test else ("val" if s in val else "train"))
    node_id = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    edges = [(i, node_id[(r + dr, c + dc)]) for i, (r, c) in enumerate(zip(rows, cols))
             for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in node_id]
    edges = np.array(edges, dtype=np.int64).T
    region.proc_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(region.nodes_csv, index=False)
    np.save(region.edges_npy, edges)
    # deliberately no label statistics printed per split (test labels stay unread)
    print(f"region2 nodes {len(df)}, edges {edges.shape[1]}, split sizes {df.split.value_counts().to_dict()}")


def main():
    meta = {}
    sid, sname, sll = nearest_navd_station((R2.bbox[1] + R2.bbox[3]) / 2, (R2.bbox[0] + R2.bbox[2]) / 2)
    R2.tide_station, R2.tide_latlon = sid, sll
    meta["tide_station"] = {"id": sid, "name": sname, "latlon": sll}
    print("tide station:", meta["tide_station"])
    fetch_gmw(R2, "2020")
    fetch_soc(R2)
    build_graph(R2)
    meta["s2_items"] = s2_items(R2)
    print("S2:", meta["s2_items"])
    (R2.proc_dir / "region2_meta.json").write_text(json.dumps(meta, indent=2))
    # 2010 epoch for both regions (Addendum A4 non-permanence check)
    fetch_gmw(R1, "2010")
    fetch_gmw(R2, "2010")


if __name__ == "__main__":
    main()
