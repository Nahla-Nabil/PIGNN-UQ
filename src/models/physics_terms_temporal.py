"""
Journal-extension sprint (post-submission, temporal dC/dt experiment).

The submitted paper's physics proxy (`src/models/physics_terms.py`)
combined a single-date NDVI LEVEL (accretion proxy) with static
distance-to-tide-gauge (export proxy) -- an explicitly disclosed
steady-state simplification, named as a limitation in `paper/main.tex`:
"Single-date imagery constrains the physics term to a cross-sectional
relationship, not an actual dC/dt."

This module computes the SAME functional form of proxy (equal-weighted,
train-only min-max scaling, no fitted coefficients -- so it stays a
physical PRIOR, not a second regression model, exactly like the
original) with ONE change: the accretion input is now `ndvi_rate_per_day`
-- a genuine two-date NDVI CHANGE RATE (see `build_graph_temporal.py`:
(ndvi_t2 - ndvi_t1) / 190 real days between the two Sentinel-2 scenes) --
in place of static single-date `ndvi`. This is the real physical quantity
the mass-balance loss term was always meant to approximate (a RATE, not a
LEVEL), and directly targets the limitation quoted above.

**Scope, disclosed explicitly:** the export term (`dist_to_tide_km`) is
UNCHANGED. It has its own, separate, already-disclosed limitation (a
positional proxy, not a real per-node tidal-energy/inundation-frequency
measurement) that a second Sentinel-2 date does nothing to fix -- this
extension does not claim to address it, and reusing it unchanged keeps
the accretion-term swap a single-variable change relative to the
submitted paper's proxy, not a confounded redesign.

`physics_loss()` itself (the loss function operating on the already-
computed proxy) is UNCHANGED and reused directly from `physics_terms.py`
-- only the proxy CONSTRUCTION differs, not how it enters the loss.
"""
import numpy as np


def compute_mass_balance_proxy_dcdt(df, train_mask):
    """Returns (proxy_raw: np.ndarray over ALL nodes, scaler stats dict).
    Scaling parameters (min/max for ndvi_rate_per_day and dist_to_tide_km)
    are fit on TRAIN rows only (this extension's own `split_ext=="train"`
    rows), then applied to all rows -- no leakage from val/test into the
    proxy's own normalization, identical discipline to the submitted
    paper's `physics_terms.compute_mass_balance_proxy`."""
    accretion_rate = df["ndvi_rate_per_day"].values
    dist = df["dist_to_tide_km"].values

    rate_tr, dist_tr = accretion_rate[train_mask], dist[train_mask]
    rate_min, rate_max = rate_tr.min(), rate_tr.max()
    dist_min, dist_max = dist_tr.min(), dist_tr.max()

    rate_scaled = (accretion_rate - rate_min) / (rate_max - rate_min + 1e-9)
    dist_scaled = (dist - dist_min) / (dist_max - dist_min + 1e-9)

    proxy_raw = 0.5 * rate_scaled + 0.5 * dist_scaled  # equal weight, no fitted coefficients

    proxy_tr = proxy_raw[train_mask]
    proxy_mean, proxy_std = proxy_tr.mean(), proxy_tr.std()

    stats = {
        "ndvi_rate_min": float(rate_min), "ndvi_rate_max": float(rate_max),
        "dist_min": float(dist_min), "dist_max": float(dist_max),
        "proxy_mean": float(proxy_mean), "proxy_std": float(proxy_std),
    }
    return proxy_raw, stats
