"""
The ONE physics constraint term for this project (per PLAN.md's scope cut),
and its exact, stated simplifications -- this module is the single source
of truth for the physics formulation, so the paper's Method section can
cite it directly rather than re-deriving it.

## What the real mass-balance relationship is

Carbon stock at a site changes as:
    dC/dt = Accretion (biomass NPP + allochthonous/autochthonous sediment
            carbon input) - Export (tidal flushing / lateral sediment
            export) - Decomposition

## What we can and cannot compute from data actually on disk

We have ONE Sentinel-2 date (2026-02-06) -- no time series -- so dC/dt
cannot be measured directly, and the flux terms (Accretion, Export) have no
real per-node measurement (no sediment trap / flux tower data for this
AOI). This is a real, stated limitation (see PLAN.md Risk R2/R3 and
DAY1_STATUS.md), not something this module works around silently.

## The simplification actually implemented

Instead of a temporal flux-balance equation, we impose a **steady-state
plausibility prior**: under an assumption that stock has locally
equilibrated with two observable proxies --

  - **Accretion proxy = NDVI** (vegetation vigor/productivity signal from
    the real Sentinel-2 pull). Higher NDVI -> more biomass productivity ->
    more carbon input, plausibly.
  - **Export proxy = distance to the NOAA tide gauge** (26.1367, -81.7883).
    Empirically, distance-to-gauge correlates POSITIVELY with observed SOC
    in this AOI (r=+0.35 on train, see NUMBER_SOURCE_LOG.md) -- i.e.
    farther from the main tidal energy source, less flushing, more
    retained carbon, which is at least directionally consistent with a
    real export mechanism. **Stated caveat, not hidden:** this is a
    single-point Euclidean distance, not a real per-node tidal-energy or
    inundation-frequency layer, so it is at least partly a spatial-position
    proxy rather than a validated hydrological one (see DAY1_STATUS.md).

  physically-expected stock ~ accretion_proxy + export_proxy
  (both min-max scaled to [0,1] on TRAIN data only, equal-weighted --
  no fitted coefficients, so this stays a physical PRIOR, not a second
  regression model).

This gives a soft, differentiable target the model's prediction is
encouraged (not forced) to stay consistent with, PLUS one hard physical
law that needs no proxy at all: **carbon stock cannot be negative.**

  L_physics = w_prior * MSE(y_pred_z, mass_balance_proxy_z)
            + w_nonneg * mean(relu(-y_pred_raw)^2)

Both components are folded into ONE named physics loss term (not two
separate physics constraints), consistent with PLAN.md's "ONE physics
constraint" scope cut -- the non-negativity penalty is a direct corollary
of mass balance (a stock that mass-balances against real, bounded fluxes
cannot go negative), not an independent physical law being smuggled in.
"""
import numpy as np
import torch


def compute_mass_balance_proxy(df, train_mask):
    """Returns (proxy_raw: np.ndarray over ALL nodes, scaler stats dict).
    Scaling parameters (min/max for ndvi and dist_to_tide_km) are fit on
    TRAIN rows only, then applied to all rows -- no leakage from val/test
    into the proxy's own normalization."""
    ndvi = df["ndvi"].values
    dist = df["dist_to_tide_km"].values

    ndvi_tr, dist_tr = ndvi[train_mask], dist[train_mask]
    ndvi_min, ndvi_max = ndvi_tr.min(), ndvi_tr.max()
    dist_min, dist_max = dist_tr.min(), dist_tr.max()

    ndvi_scaled = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-9)
    dist_scaled = (dist - dist_min) / (dist_max - dist_min + 1e-9)

    proxy_raw = 0.5 * ndvi_scaled + 0.5 * dist_scaled  # equal weight, no fitted coefficients

    proxy_tr = proxy_raw[train_mask]
    proxy_mean, proxy_std = proxy_tr.mean(), proxy_tr.std()

    stats = {
        "ndvi_min": float(ndvi_min), "ndvi_max": float(ndvi_max),
        "dist_min": float(dist_min), "dist_max": float(dist_max),
        "proxy_mean": float(proxy_mean), "proxy_std": float(proxy_std),
    }
    return proxy_raw, stats


def physics_loss(y_pred_scaled: torch.Tensor, y_pred_raw: torch.Tensor,
                  proxy_z: torch.Tensor, y_std: float, w_prior: float, w_nonneg: float):
    """y_pred_scaled: model output in the same standardized (z-score) space
    as the data loss and as proxy_z (both ~O(1) scale, so w_prior directly
    trades off against the data MSE term at face value).
    y_pred_raw: the SAME predictions unscaled to raw t C/ha units (needed
    because "non-negative" is a raw-unit physical statement -- z-score 0
    corresponds to the MEAN stock, not zero carbon). The raw-unit penalty
    is divided by y_std^2 so a violation of ~1 std in raw units contributes
    ~O(1) to the loss, comparable to the other terms rather than dwarfing
    them (raw t/ha^2 values would otherwise be 3-4 orders of magnitude
    larger than the z-scored data loss)."""
    prior_term = torch.mean((y_pred_scaled - proxy_z) ** 2)
    nonneg_term = torch.mean(torch.relu(-y_pred_raw) ** 2) / (y_std ** 2)
    return w_prior * prior_term + w_nonneg * nonneg_term, prior_term.item(), nonneg_term.item()
