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

**D6 (2026-10-09, after W2 CV, before any test read):** selection-code bug.
A1's rule is "highest mean OOF R²; within 1 SE prefer the physically grounded
priors". The W2 script applied the proxy order P1 < P2 < P0 to *every* config
within 1 SE, so it returned P1/λ=0.1 (OOF R² 0.394) instead of the best config,
P2/λ=0.1 (0.405), which is itself physically grounded. Corrected to P2/λ=0.1;
`rev_w2_selection.json` keeps the buggy-code value under
`selected_phys_by_buggy_code`. Both are reported in the paper's supplementary
table. No-physics GNN: 0.385 (fold SE 0.033), so the physics gain (+0.020) is
within 1 SE, i.e. not significant, and is stated as such.

**D7 (2026-10-09, region-2 build, before any region-2 model was fit):** the
pre-registered same-datatake S2 scenes (R097, 2026-02-06 / 2026-08-15) leave
the eastern part of region 2 (lon > −81.007; 5,169 of 22,326 nodes, 23%)
outside the swath. That is not cloud: t0 cloud fraction is 0 there.
Uncovered pixels are filled from the nearest-date adjacent-orbit scene
with < 20% cloud (relative orbit R054: S2C 2026-02-08 for t0, S2C
2026-08-17 for t1). The original datatake keeps priority wherever it has
data. Cross-sensor (S2B/S2C) differences are a stated, minor caveat.
Region-2 tide gauge: the nearest NAVD88 hourly station is 8723970 Vaca Key
(about 70 km from the bbox), a stated simplification for the P1/P2 prior.

**D8 (2026-10-09, after the dry run on val stand-in data, before any test read):**
an additional cross-region transfer evaluation is added, `transfer_nopos_*`,
with the positional feature `dist_to_tide_km` removed (and the
coordinates-only model skipped). Reason: that feature is computed from each
region's own gauge (region 2: about 70–100 km from Vaca Key vs 0–30 km in
region 1), so it is non-transferable by construction and forces
extrapolation. In the dry run, GNN transfer R² was −6.7 ± 8.6 across seeds.
The original A3b transfer evaluation is kept and reported unchanged. Other
dry-run observations, reported but not acted on: project-level VM0033
aggregation (Eq. 88, independence assumption) yields 1.4% uncertainty and
so no deduction; the GNN is less stable than RF under the t1 image swap.

---

# Addendum B (2026-10-09 21:06): local calibration for cross-region use, VALIDATION ONLY

Written after the single test read, which showed that zero-shot
cross-region transfer fails (all transfer R² < 0). This addendum asks a
different, practical question: **with a small local labeled sample, as
any real crediting project would collect, does a model trained elsewhere
become usable, and do conformal intervals regain valid coverage?** It is
evaluated **only on target-region validation nodes**. Test is not read
again, and the zero-shot test numbers are reported unchanged.

- Directions: R1→R2 and R2→R1. Families: XGBoost, RF, GNN, Phys-GNN
  (selected configs), with no positional features (as in D8).
- Target val is split by fine-grid groups 50/50 (seed 7, the dry-run
  split) into `local` (the "field sample") and `check` (evaluation).
- Variants, all scored on `check`:
  - V0 zero-shot: source-trained model;
  - V1 recalibrated: V0 + linear correction y ≈ a + b·ŷ fitted on `local`;
  - V2 pooled: model retrained on source train ∪ target `local`;
  - V3 local-only: model trained on target `local` alone (reference).
- Intervals: split-conformal on |y − ŷ| for each variant. Two calibration
  sets are compared: source val (zero-shot UQ) and target `local` (local
  UQ). Coverage, width and Winkler at 90% are reported on `check`.
- Reported for every variant; nothing is selected. Pre-stated hypothesis:
  V1/V2 recover positive R², and local conformal calibration restores
  ~90% coverage, whereas source calibration undercovers.

---

# Addendum C (2026-10-09, written before running C1-C4): remaining weaknesses, NO test read

All evaluations use CV on train ∪ val (region 1) or target-region val
`check` nodes (Addendum B split). Test is not read again.

**C1. Fix-only ablation (credibility of D1).** The submitted paper's exact
setup is reproduced: features F0 (ndvi, b04, b08, b11, dist), the linear
and GNN models (hidden 16), and the §4 spatial CV folds/seeds. The *only*
difference between arms is the feature source:
- (a) the submitted, mis-georeferenced S2 columns from `graph_nodes.csv`;
- (b) the corrected columns.
Reported: OOF R² and MAE per arm and the paired per-fold difference. A
further arm (c) adds a deliberate 9.18 km shift to the *corrected* rasters,
reproducing the bug from clean inputs.

**C2. Proper local split (Addendum B fix).** V2/V3 used the same local
sample for training and conformal calibration, so the residuals were
in-sample and the intervals invalid. C2 splits `local` by fine-grid groups
50/50 into `local_fit` and `local_cal` (seed 11):
- V2b pooled (source train ∪ local_fit), calibrated on local_cal;
- V3b local_fit only, calibrated on local_cal;
- V1b recalibration fitted on local_fit, conformal on local_cal.
All are scored on the same `check` nodes as Addendum B.

**C3. Region-wise feature standardization (domain adaptation).** Each
region's features are z-scored with its own non-test statistics before
cross-region training and prediction, a standard first-moment
alignment. Evaluated with zero-shot, V1b and V2b, as in C2.

**C4. Two-date augmentation for temporal robustness.** The GNN and
Phys-GNN are trained on the union of t0 and t1 inputs: each training node
appears twice, with t0 and t1 imagery and the same label; t1 cloud gaps
are imputed as before. Reported: CV OOF R² (region 1, §4 folds) with t0
inputs, and the Spearman ρ between t0- and t1-input predictions on the
OOF nodes, vs the t0-only model (E8 found ρ = 0.42 on test).

Nothing here changes any reported test number. Results are additions,
labelled as such.

---

# Addendum D (2026-10-09 21:25): field soil-core validation, written before any core is matched to a prediction

Source: Coastal Carbon Network Data Library (Smithsonian CCN),
`data/CCN_synthesis` (CCN_cores.csv, CCN_depthseries.csv), downloaded
2026-10-09 into `data/raw/validation_refs/ccn/`. Habitat = mangrove,
inside the region-1 or region-2 bbox: 87 cores (R1 56, R2 31; 11 studies).
Facts known before writing this (no carbon values or matching yet):
max_depth median 25 cm; DBD available in 84 cores, fraction_carbon in 40,
fraction_organic_matter in 77.

**D-a. Field carbon per core.**
- SOC density per interval = DBD × fraction_carbon. Where fraction_carbon
  is missing, it is derived from organic matter using Craft et al. (1991)
  (OC = 0.40·OM + 0.0025·OM², in %), verified against the primary source
  before use.
- Primary field quantity: **mean SOC density over the sampled 0–30 cm**
  (cores with ≥ 15 cm sampled). This is scale-free with respect to
  extrapolation.
- Secondary: 0–100 cm stock (t C/ha) with the mean density below the
  deepest sample held constant to 100 cm (standard constant-density
  extrapolation). Reported only for cores sampled ≥ 50 cm; the rest are
  flagged as extrapolated.
- Several cores at one site with identical coordinates are averaged into
  one location.

**D-b. Matching.** Each location is matched to the nearest node of its
region within 300 m; unmatched cores are reported and dropped.

**D-c. Comparisons (descriptive, no selection).** Spearman ρ (primary) and
Pearson r between field carbon and: (i) the Sanderman label; (ii) the
Radabaugh map, excluding Radabaugh_et_al_2023 cores, which calibrated
that map; (iii) out-of-fold predictions of the selected XGB, RF and
Phys-GNN. Predictions come from a 5-fold spatial CV over *all* nodes of
the region (`phase1_superblock` groups, seed 20261007), so every
core-matched node is predicted by a model that never saw its group. This
is not a test-set evaluation: no model or setting is selected and no
Sanderman test metric is computed. Possible overlap of these cores with
Sanderman's own training data is stated as a caveat.

**D-d. Do intervals cover field truth?** (1) Coverage of field 0–100 cm
stock (≥ 50 cm cores) by the CQR-GNN 90% intervals (Sanderman-calibrated).
(2) Field-calibrated conformal: a scale factor k is chosen so that
ŷ·[1 − k, 1 + k] covers 90% of the field stocks, estimated by
leave-one-study-out (studies as groups) to avoid within-study leakage.
Reported: held-out coverage and the implied VM0033 deduction
(k − 0.20)⁺, against the Sanderman-calibrated deduction.

**D9 (2026-10-09, before matching cores to predictions):** the Craft et al.
(1991) coefficients could not be verified from an accessible primary text
(the article is on JSTOR; the citation itself is verified:
Estuaries 14:175, doi:10.2307/1351691). Instead, the OM→OC conversion
keeps Craft's quadratic *form* and is **fitted on SW-Florida mangrove CCN
intervals with both measurements** (679 intervals, 61 cores, 7 studies):
OC% = 0.00174·OM%² + 0.3196·OM% + 2.330 (R² = 0.908). It is used only where
fraction_carbon is missing.

---

# Addendum E (2026-10-09, before running): which VM0033 aggregation is valid? VALIDATION ONLY

Concern anticipated from R1: VM0033 applies Eq. 92 at project level, and
with Eq. 87/88 (independent errors) our project uncertainty is under 1%,
so the deduction is zero. Per-node application is our extension. Test, on
val only: CQR-GNN (selected) fit on train, calibrated on val-calib, with
the Phase-1D 50/50 val split (seed 7). Each fine-grid cell
(`phase1_superblock`, about 2.5 km) of val-check is a pseudo-project.
Per project p:
- realized aggregate relative error e_p = |Σ(ŷ − y)| / Σy;
- predicted project uncertainty (90% half-width, % of the stock):
  - (i) U_ind: VM0033 Eq. 87/88 with independent node errors;
  - (ii) U_corr: fully correlated, the stock-weighted mean of node half-widths;
  - (iii) U_block: an empirical aggregate conformal half-width, the 90%
    quantile of e over val-calib projects (out-of-sample for val-check
    projects).

Reported: the fraction of val-check projects with e_p ≤ U (nominal 90%) for
each of (i)-(iii), and the implied deduction (U − 20%)⁺. A method is called
valid if its coverage is ≥ 85% (small project count). Both regions.
