# Revision pre-registration (ICDABI 2026 major revision)

Written 2026-10-07, **before any revision experiment was run**. This file is
committed on branch `revision-icdabi2026` before the first model fit; that
commit's hash and timestamp are the verifiable pre-registration. Any
departure from it is logged in §8 with its reason, so nothing changes
silently.

Reviewer IDs (R1.x / R2.x) refer to `REVISION_PLAN.md` §1.

## 0. What is frozen

- Node set, edges, and the `split` column of `data/processed/graph_nodes.csv`
  (6,465 train / 1,583 val / 2,076 test) are **unchanged**.
- New features go into a *separate* file,
  `data/processed/graph_nodes_rev_features.csv`, keyed by (row, col) and
  joined at load time. `graph_nodes.csv` is never rewritten.
- **The test split is read exactly once**, by
  `scripts/run_revision_test_eval.py`, after every selection below is
  final, for all carried-forward models together. The original submission's
  test numbers stay in the paper as "original submission" rows.

## 1. Leakage-safe graph regime (R1.6, R2.3), fixed in advance, not selected

All revision GNNs train on the **train-induced subgraph** (edges with both
endpoints in train). Validation and test nodes are predicted on their own
**split-induced subgraphs**, so no edge crosses a split boundary at
training or inference (the 2,520 cross-split edges are dropped). The same
regime is used for the tabular spatial-lag features (§2): neighbor means
are computed only over same-split neighbors. As a secondary check, test
metrics are also reported on the subset of test nodes ≥ 2 hops away from
any non-test node (2 hops = the GraphSAGE receptive field).

## 2. Candidate feature sets (R1.1)

Sources: the two Sentinel-2 L2A scenes already used (2026-02-06 = "t0",
2026-08-15 = "t1"; same tile/orbit), fetched for the additional bands
B02, B05, B06, B07, B8A, B12 via Planetary Computer, and Copernicus GLO-30
DEM. Everything is aggregated with the same 5×5 block mean onto the
existing node grid.

| Name | Columns |
|---|---|
| `F0_base` | ndvi, b04, b08, b11, dist_to_tide_km (the submitted paper) |
| `F1_spectral` | F0 + b02, b03, b05, b06, b07, b8a, b12, ndre (B8A/B05), mndwi (B03/B11), ndmi |
| `F2_spec_dem` | F1 + elevation (m), within-node elevation std |
| `F3_spec_dem_temporal` | F2 + t1 values of ndvi, ndre, ndmi, mndwi + Δndvi, Δndmi (t1 − t0) |

## 3. Candidate models (R1.1, R2.4)

| Family | Settings (small, fixed grid) |
|---|---|
| Linear (OLS) | n/a |
| Random Forest | n_estimators=500, max_features ∈ {0.33, 1.0}, min_samples_leaf ∈ {5, 20} |
| XGBoost | n_estimators=600, learning_rate=0.03, max_depth ∈ {3, 6}, subsample=0.8, colsample_bytree=0.8, min_child_weight=10 |
| RF + spatial lag | RF with each feature's 1-hop same-split neighbor mean appended (best RF setting) |
| Coordinates-only RF | lon, lat only, a location-only reference, never "selected" |
| Baseline GNN (GraphSAGE) | hidden ∈ {16, 64}, layers = 2, dropout 0.3, lr 0.01, wd 1e-3, early stopping on an inner holdout |
| Physics GNN | best baseline-GNN architecture, λ ∈ {0.1, 0.3, 1.0}, w_nonneg = 0.1 |

Target: raw t C/ha (no transform), so the selection space stays small.

## 4. Selection procedure (R1.1, also answers "one block is one draw")

- **Selection pool:** train ∪ val nodes (8,048). Test is excluded.
- **Spatial block CV:** 5 folds, grouped by `phase1_superblock` (the 20×20
  grid the val/test split was built on), with group→fold assignment by
  `GroupKFold` on a seeded shuffle (seed 20261007). Each CV fit is
  leakage-safe as in §1 (fold-induced subgraphs). GNN early stopping uses
  an inner holdout: a random 15% of the fold-train's groups (seeded).
- **Seeds:** stochastic models are run with 3 seeds {1, 2, 3} per fold.
- **Primary metric:** pooled out-of-fold R² (mean over seeds).
  Secondary metrics are reported, never used to select: MAE, RMSE, Spearman ρ,
  skill vs. the fold-train-mean predictor (1 − MSE/MSE_mean).
- **Rule:** within each family, choose the (feature set, setting) with the
  highest mean OOF R². If several fall within one standard error (SE over
  seed × fold R²) of the best, pick the **simplest**: fewer features
  (F0 < F1 < F2 < F3), then the smaller model.
- **Physics λ:** chosen on the same CV by the *submitted paper's* rule: the
  best mean OOF prior-generalization correlation among λ whose mean OOF R²
  is within 0.05 of the best λ's.
- **Carried forward to test:** the selected setting of *every* family (so
  the test table is a full comparison, not just a winner), each refit on
  train with val for early stopping, exactly as in the submitted paper.

## 5. Uncertainty quantification (R1.3)

Candidates, all split-conformal and calibrated on val (as submitted):

| Name | Base model |
|---|---|
| `CQR-GNN` | quantile GraphSAGE, data loss only (the submitted method), re-run under §1/§2's selected feature set |
| `CQR-PhysGNN` | quantile GraphSAGE + λ·physics term on the median head (λ from §4) |
| `CQR-XGB` | XGBoost quantile regression (`reg:quantileerror`), selected XGB setting |
| `NCP-RF` | normalized conformal: \|y − ŷ\| / σ̂(x), where ŷ = selected RF and σ̂ = an RF fit on out-of-bag absolute residuals |
| `Trivial` | constant train-mean predictor + split conformal (the reference row, never selected) |

**Calibration variants:** global (primary) and Mondrian by superblock
(≥ 30 points, global fallback), reported as in the submitted paper.

**Selection (val-internal, test untouched):** val is split 50/50
(seed 7, as in Phase 1D) into calibrate/check. Primary metric: mean Winkler
interval score at 90% on val-check, among methods whose val-check coverage
is ≥ 88% at the 90% level. Reported for every method, regardless:
coverage and width at 90/95%; Spearman ρ(width, |y − midpoint|) with a
1,000-sample bootstrap 95% CI; coverage per width tercile.

**Pre-registered gate for pricing (R1.4):** the uncertainty-aware issuance
rule in §6 is presented as *operational* only if the selected method's
ρ(width, |error|) on **test** has a bootstrap CI that excludes 0. Otherwise
the paper presents it as a conditional design, stated as such.

## 6. Pricing redesign (R1.4, R2.5, R1.7), specified before seeing results

- Non-permanence stays the registry's own buffer (Plan Vivo 20%), applied
  unchanged. The model never adjusts it.
- Estimation uncertainty enters only through **how much stock is
  credited**: credited stock = the lower bound of the 90% interval
  (one-sided conservative), compared against crediting the point estimate.
  **No premium above the point estimate.** Registry uncertainty-deduction
  rules (e.g. Verra/CDM conservativeness guidance) are cited only after
  their primary documents have been fetched and quoted.
- Reported: portfolio total credited stock over test nodes (flat vs.
  conservative), the per-node deduction distribution, and the same under
  **both** global (post-hoc deviation, flagged) and Mondrian (pre-registered
  in the submission) calibration.

## 7. Independent checks (R1.2, R2.2) — no selection

- **E7 (field data):** sub-regional field-plot carbon stocks from Radabaugh
  et al. (2023) for sub-region(s) overlapping the AOI vs. the
  model-predicted and label-product means over the same nodes. This is
  descriptive only.
- **E8 (temporal stability):** the final model's predictions and interval
  widths using t1-only imagery vs. t0 (labels static), Spearman ρ across
  dates. Only applicable to F0–F2 models; F3 already uses both dates.
- The prior-generalization correlation is still computed and reported, but
  described as a *constraint-transfer diagnostic*, not validation.

## 8. Deviations log

**D1 (2026-10-07, found while building §2 features, before any model fit):
the submitted paper's Sentinel-2 features were mis-georeferenced by 9.18 km.**
`fetch_sentinel2.clip_bands()` requested an AOI window that extended 918
rows north of tile T17RMJ's top edge. The non-boundless read returned tile
rows starting at row 0, but the file was written with the window's
transform, so every S2 pixel in `data/raw/sentinel2/sample_clip_*.tif` is
labeled 9.18 km north of its true position (verified: clip row 0 equals
tile row 0 for 100% of compared pixels). The northern AOI (lat > ~26.21) is
outside T17RMJ entirely (it lies in T17RMK). As a result, the submitted
features for "mangrove" nodes had mean NDVI 0.003, with 94.8% below 0.3,
i.e. mostly water. The SOC labels, the GMW mask (merge + mask-crop), the
node set, the edges and the splits are **not** affected; nor is
dist_to_tide_km, which is computed from coordinates.

Consequence for this pre-registration: every feature set in §2 (including
`F0_base`) uses the **corrected** features from
`src/data_pipeline/build_rev_features.py` (T17RMJ + T17RMK mosaic of the
same datatake, boundless windowed reads, SCL cloud mask). Corrected
F0 NDVI: mean 0.458, 4.0% below 0.3. Nothing else in §1–§7 changes. The
submitted paper's numbers are reported as "original submission
(mis-georeferenced features)", and the paper states the error explicitly.

**D2 (2026-10-07, during the smoke test on CV fold 0, before the full CV run):
GNN early-stopping patience changed from 30 to 100 (minimum 50 epochs,
maximum 500), for both GNN variants equally.** The inner-holdout loss
curve is noisy for the first ~40 epochs (it rises from 0.80 to 0.90 before
falling to ~0.56 by epoch ~140; trace for λ=0 and λ=0.1, fold 0, seed 1).
With patience 30, runs stopped at an essentially random early epoch: in
the smoke test the λ=0.1 run stopped after ~35 epochs with a collapsed
prediction SD (41 vs 96 t/ha). This is a training-procedure fix, not a
selection choice. It applies uniformly and was decided from loss curves,
not from comparing CV R² between settings.

**D3 (addition, 2026-10-07):** the E7 reference is upgraded from
Radabaugh et al. (2023) Table 1 regional means to their full 30 m
belowground-carbon map (FWC ArcGIS item 50b8a2c855b448209aa5aabeb9301193,
fetched by `src/data_pipeline/fetch_radabaugh_bgc.py`). It is used for
evaluation only, never as a feature or for selection. Descriptive fact
recorded on fetch, before any model test: corr(Sanderman label, Radabaugh)
= −0.100 over 10,115 nodes; −0.017 over 20 superblock means.

**D4 (addition, 2026-10-07):** issuance rule B = VM0033 v2.1 (Verra,
4 Sept 2023) p. 64, Eq. 92, quoted in `src/pricing/issuance.py` (primary PDF in
`paper/refs_primary/`). Rule A (§6) remains the pre-registered primary rule.

**C1 (clarification):** quantile GNNs early-stop on an inner 15% holdout
of train superblock groups, so val is used only for conformal calibration.

---

# Addendum A (2026-10-09): pre-registered BEFORE running W2/W5/W6/W7

Written after the §4 CV results were known (`results/rev_e1_cv_summary.csv`)
and before any of the experiments below were run. Test is still unread.

## A1. Physics prior v2 (reviewer R2: "physics relies on NDVI/distance proxies")
Facts already known from CV, which motivated this: NDVI is uncorrelated with
the label (r = −0.04 on the pool), and the proxy's signal comes from
distance to the gauge.

Candidate priors. Each is min-max scaled on fit nodes, so no coefficient is
fitted:
- `P0_original`: 0.5·NDVI + 0.5·dist (the submitted paper's prior), the control;
- `P1_hydroperiod`: tidal inundation frequency = fraction of hours in one
  year (2025-01-01 to 2025-12-31) of NOAA CO-OPS 8725114 hourly heights
  (datum NAVD88) exceeding the node's mean USGS 3DEP 1/3″ (10 m) bare-earth
  elevation (NAVD88). Physical basis: hydroperiod controls sediment
  delivery and soil-carbon preservation (anoxia). A single-gauge water
  level for the whole AOI is a stated simplification;
- `P2_hydro_veg`: 0.5·P1 + 0.5·(t0 NDRE), i.e. inundation plus a canopy
  productivity term (red-edge, which tracks canopy chlorophyll better than
  NDVI for dense mangrove).

Constraint forms:
- `mse`: λ·MSE(ŷ_z, proxy_z), as before;
- `mono`: λ·mean(ReLU(−∂ŷ/∂P1)) via finite difference. This penalizes
  predictions that *decrease* with inundation frequency, i.e. a sign
  constraint, not a target. It applies to P1 only.

Grid: {P0, P1, P2} × mse × λ ∈ {0.1, 0.3, 1.0}, plus P1 × mono × λ ∈ {0.1, 1.0},
on the selected GNN (F1, hidden 16), same CV folds/seeds as §4.
Selection rule: highest mean OOF R². Ties within 1 SE go to P0 < P1 < P2
order reversed (prefer the physically grounded prior, P1 or P2, if within
1 SE of the best). If no physics config is within 1 SE of the no-physics
GNN, the paper states that the physics prior does not improve accuracy.
Reported for every config, never used to select: OOF prior-gen corr and
OOF corr with the Radabaugh map (independent).

## A2. UQ ensemble candidate (R1: intervals wider than the label SD)
Added to §5: `Ens-Q` = the element-wise mean of CQR-GNN, CQR-XGB quantiles
and NCP-RF's yhat ± z·s (z from normal quantiles), then CQR-conformalized
on val. The same §5 selection rule applies, and Trivial stays the reference.

## A3. Second region (R1: single region)
- Region 2 bbox: (−81.20, 25.30, −80.90, 25.60), Everglades NP coast
  (Lostmans River to Shark River). Fixed now, before its labels are viewed.
- The same graph construction (5×5 blocks of the Sanderman 30 m grid,
  GMW 2020 > 0.5, 8-connectivity) and the same split procedure as region 1
  (`build_graph.py` superblock assignment: 20% test / 15% val by cumulative
  node count, seed 20261009). The same feature builder.
- Tide gauge: the nearest NOAA CO-OPS station with 2025 hourly heights in
  NAVD88.
- **No new selection in region 2.** Region-1 selections are reused.
  Two evaluations, read once, together with region 1's test:
  (a) in-region: train on region-2 train (+ val for GNN early stopping),
  evaluate on region-2 test;
  (b) transfer: train on region-1 train, evaluate on all region-2 test
  nodes, and vice versa (train on region-2 train, evaluate on region-1
  test).

## A4. Non-permanence check (R1/R2: pricing conflates risks, not validated)
- Per node: mangrove loss fraction = GMW 2010 mangrove pixels absent in GMW
  2020, divided by GMW 2010 mangrove pixels in the node (GMW v3).
- Reported on test nodes: Spearman ρ between loss fraction and (i) 90%
  interval width, (ii) rule-B deduction. Hypothesis stated now: |ρ| < 0.1,
  i.e. estimation uncertainty is not a proxy for non-permanence, which is
  why the two factors stay separate. Reported whatever the result.
- Market sensitivity (labeled not-validation): credited tC × 3.667 → tCO2e
  × published blue-carbon price range (sources verified before use).

**D5 (2026-10-09, before running W2):** the A1 `mono` constraint is
implemented as a *neighbour-pair* sign constraint rather than a finite-difference
partial dependence, because inund_freq is not a model input (F1 features).
For ordered graph neighbours (a, b) in the fit subgraph whose inund_freq
difference exceeds the median absolute neighbour difference, the loss adds
λ·mean(ReLU(ŷ_b − ŷ_a)) (z units), i.e. the wetter neighbour should not be
predicted to hold less carbon. The non-negativity term is kept. Elevation
source for P1: USGS 3DEP seamless 1/3″ (10 m) bare earth; GLO-30 (used in F2)
is 0 m over most of the AOI.
