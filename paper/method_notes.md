# Method Notes: the physics-informed loss term

Full implementation and reasoning: `src/models/physics_terms.py` (single
source of truth — this file is a condensed, paper-ready version of that
module's docstring, not a separate derivation).

## Paste-able paragraph for the Method section

> The full carbon mass-balance relationship for a mangrove site is
> dC/dt = Accretion − Export − Decomposition, where Accretion is biomass
> productivity plus sediment carbon input and Export is tidal flushing and
> lateral sediment loss. Computing this directly requires a time series of
> stock observations and per-site flux measurements, neither of which is
> available for the pilot AOI (a single Sentinel-2 date; no sediment-trap
> or flux-tower data). We therefore impose a simplified steady-state
> plausibility prior in place of the full temporal equation: predicted
> carbon stock is softly encouraged toward an equal-weighted combination of
> an accretion proxy (NDVI, from the available Sentinel-2 imagery) and an
> export proxy (distance from the node to the nearest tide gauge, used as
> a stand-in for tidal energy/flushing intensity in the absence of a
> per-node hydrodynamic layer), both min-max normalized on the training
> split only. A second, unconditional term penalizes physically impossible
> negative carbon predictions. Both terms are combined into a single
> physics loss added to the standard data-fitting loss, with no additional
> free-fit coefficients beyond the two loss weights. We report this as one
> physics constraint, not two, since the non-negativity penalty is a
> direct corollary of mass balance rather than an independent physical law.

## What is included

- Accretion proxy: NDVI (real, from the AOI's pulled Sentinel-2 scene).
- Export proxy: Euclidean distance to the NOAA tide gauge (real station
  location; not a per-node hydrodynamic quantity).
- Non-negativity: a hard physical bound (carbon stock ≥ 0), implemented as
  a soft penalty.
- Steady-state assumption: stock has locally equilibrated with the two
  proxies above (explicitly not a claim that the AOI is literally in
  steady state — a scoping simplification stated as such).

## What is explicitly dropped, and why

- **Temporal dC/dt.** Only one Sentinel-2 date was available this sprint
  (Day 1 checkpoint decision, Path A: no second-date detour). The physics
  term therefore constrains a cross-sectional plausibility relationship,
  not an actual rate of change.
- **Decomposition.** No per-site decomposition rate data exists for this
  AOI; folding an unconstrained third term in would only add another
  unvalidated proxy without a data source to ground it.
- **Real tidal flux/inundation data.** Distance-to-gauge is a single-point
  Euclidean proxy, not a per-node inundation-frequency or tidal-energy
  layer — flagged already in `DAY1_STATUS.md` as functioning partly as a
  spatial-position proxy rather than a validated hydrological signal. This
  is the physics term's most significant simplification and is stated as
  such everywhere it's used, including in the paper's Limitations section.

## Why this is still a legitimate physics-informed loss, not just a second regression

The proxy combination is equal-weighted and min-max normalized with **no
fitted coefficients** — the model cannot learn its way around the
constraint by re-weighting the proxy inputs, unlike a stacked/ensembled
second model would. This mirrors the flood-forecasting PINN-GNN precedent
in `literature_notes.md` §1: a discretized, simplified conservation-law
residual added as an auxiliary loss term, not a full physics simulator.

## Result (see DAY2_STATUS.md for full numbers)

The physics term improved test R² (−0.383 → −0.219) without closing the
gap to a plain linear baseline (+0.031), but produced a large,
independently measurable effect on physical-consistency correlation under
spatial holdout (0.048 → 0.645) — the basis for this paper's reframed
central claim: the constraint improves prediction *reliability/
plausibility* under distribution shift, not necessarily point accuracy,
which is the paper's actual motivation for pairing it with calibrated UQ
rather than presenting it as an accuracy improvement on its own.
