# Number Source Log

Same standard as `conformal-coverage-geohazard/abstract_draft.md`'s footer:
every number that could end up in the paper is traced here to an exact
file, row, and column before it is allowed into prose. Updated as the
sprint progresses; entries are never deleted, only marked superseded so
the audit trail of what changed and why stays intact.

---

## Regional validation reference (Radabaugh et al. 2023)

Source PDF: `data/raw/validation_refs/Radabaugh_et_al_2023_SWFL_blue_carbon.pdf`
(primary source, read directly — not taken from a search-engine summary).
Radabaugh, K.R., Moyer, R.P., Chappel, A.R., Breithaupt, J.L., Lagomasino,
D., Dontis, E., Russo, C.E., Rosenheim, B.E., Chambers, L.G., Peneva-Reed,
E.I., Smoak, J.M. (2023). "A Spatial Model Comparing Above- and
Belowground Blue Carbon Stocks in Southwest Florida Mangroves and Salt
Marshes." *Estuaries and Coasts* 46:1536–1556.
doi:[10.1007/s12237-023-01217-7](https://doi.org/10.1007/s12237-023-01217-7).

| # | Number | Exact source | Note |
|---|---|---|---|
| 1 | Whole-SW-Florida mangrove total C stock: 393.9 ± 107.1 Mg C ha⁻¹ | PDF p.1, Abstract, sentence: "mangroves and salt marshes in Southwest Florida store an average of 393.9±107.1 and 286.7±71.9 Mg C ha⁻¹, respectively, in above- and belowground C stocks" | **Ambiguity resolved:** "respectively" refers back to "mangroves and salt marshes" (the sentence's first-listed pair), not to "above-/belowground." Confirmed against p.12 prose: "Total C stocks in mangroves ... 393.9±107.1 Mg C ha⁻¹ ... Table 1)" — Table 1's `All` row, `Mangrove` ecosystem type, `Total C stocks` column reads exactly `393.9±107.1`, settling the ambiguity by matching the table verbatim. |
| 2 | **Ten Thousand Islands region, mangrove, total C stock: 434.6 ± 96.8 Mg C ha⁻¹** — the number this project actually uses (region-matched to our AOI, not the whole-SWFL average above) | PDF p.12, Table 1, row: `Ecosystem type=Mangrove, Region=Ten Thousand Islands`; column `Total C stocks (Mg C ha⁻¹)` = `434.6±96.8` | Same row also gives: Aboveground C stock `50.9±14.4`, Belowground C stock `383.6±87.7`, Total C stocks per region `13.06 Tg`, Pixels in model (n) `333,975` |
| 3 | Ten Thousand Islands, salt marsh, total C stock: 279.5 ± 78.7 Mg C ha⁻¹ | PDF p.12, Table 1, row: `Ecosystem type=Salt marsh and other, Region=Ten Thousand Islands`, column `Total C stocks` | Not currently used (our graph is mangrove-masked only), kept here in case non-mangrove nodes are added later |
| 4 | Belowground C stock model: RMSE 6.99 kg C m⁻³, R²=0.784 (cross-validation) | PDF p.9, "Belowground C Model Development" paragraph | Their own model's reported cross-validation error — useful benchmark for what error magnitude is "normal" for this kind of prediction in this region |

---

## SOC dataset (Sanderman/Worthington/WHRC et al., Zenodo 7729492) — our actual training label

| # | Number | Exact source | Note |
|---|---|---|---|
| 5 | AOI-wide SOC (0-100cm) over nonzero pixels: n=267,714, mean=421.6 t/ha, std=117.0, median=396.0 | Printed by `src/data_pipeline/fetch_soc.py`'s `sanity_check_against_radabaugh()`, run 2026-08-29, reading `data/raw/soc/soc_tha_0_100cm_mean_ten_thousand_islands_fl_clip.tif` | AOI-WIDE plausibility check only (not per-pixel validation) against Radabaugh's belowground figure (#2 above: 383.6±87.7). 421.6 is within ~1 std of 383.6 — same ballpark, not identical metric (SOC-only vs. belowground incl. live/dead roots), corroborating rather than contradicting |
| 6 | Node-level SOC label range after graph construction: min=174.0, max=1048.6, mean=421.0, std=103.4, n=10,124 nodes | Printed by `src/data_pipeline/build_graph.py`, run 2026-08-29; full node table in `data/processed/graph_nodes.csv`, column `soc_tha_mean` | Post-aggregation (5×5 block mean of native 30m SOC grid → ~150m nodes), post-masking to GMW-mangrove ∩ valid-SOC cells only |

---

## Graph / experiment design

| # | Number | Exact source | Note |
|---|---|---|---|
| 7 | Graph size: 10,124 nodes, 64,366 directed edges (8-connectivity) | `src/data_pipeline/build_graph.py` stdout, 2026-08-29; `data/processed/graph_nodes.csv` (10,124 rows) and `data/processed/graph_edges.npy` (shape (2,64366)) | |
| 8 | Final 3-way split (8×8 superblock grid): train=5,464, val=2,424, test=2,236 nodes, over 20 occupied superblocks (10 train / 3 val / 7 test) | `data/processed/graph_nodes.csv`, `value_counts()` of the `split` column; superblock counts via `groupby('split')['superblock'].nunique()`, both reproduced 2026-08-29 | Per-split label means: train=404.6, val=474.2, test=403.3 (t/ha) — computed the same way, see next row |
| 9 | Per-split SOC label mean (t/ha): train=404.59, val=474.22, test=403.32 | `data/processed/graph_nodes.csv`, `groupby('split')['soc_tha_mean'].mean()`, 2026-08-29 | Train and test means are close (404.6 vs 403.3); val is higher by chance (only 3 superblocks) — val used only for early stopping, never for the reported test metric |

---

## Baseline GNN (no physics term) — Day 1 result

Full precision values (used in the table below) are in
`results/baseline_gnn_metrics.csv`; the table here rounds to 3 sig figs for
readability, exactly matching that file's `rmse`/`mae`/`r2` columns.

| # | Number | Exact source | Note |
|---|---|---|---|
| 10 | Train: RMSE=79.4, MAE=60.9, R²=0.246, n=5,464 | `results/baseline_gnn_metrics.csv`, row `split=train` | seed=42, best_epoch=66 (early-stopped on val, `results/baseline_gnn_metrics.csv` `best_epoch` column) |
| 11 | Val: RMSE=107.1, MAE=83.6, R²=0.137, n=2,424 | `results/baseline_gnn_metrics.csv`, row `split=val` | |
| 12 | **Test (spatial block, held out entirely during training/selection): RMSE=114.7, MAE=89.7, R²=−0.383, n=2,236** | `results/baseline_gnn_metrics.csv`, row `split=test_spatial_block` | **Headline Day-1 result: negative test R².** Model is worse than predicting the test set's own mean. See entries #13-15 for the diagnostic ruling out a GNN-specific bug. |

---

## Split/model diagnostic (isolating "is this a GNN bug or a data/feature limitation?")

Full values in `results/split_diagnostic_comparison.csv`.

| # | Number | Exact source | Note |
|---|---|---|---|
| 13 | Linear regression: train R²=0.040, test R²=0.031 | `results/split_diagnostic_comparison.csv`, rows `model=linear_regression` | The only model of the four compared that beats naive on test |
| 14 | Random forest (n_estimators=300, max_depth=8, seed=42): train R²=0.428, test R²=−0.495 | `results/split_diagnostic_comparison.csv`, rows `model=random_forest` | Worst test performance of the four — overfits hardest |
| 15 | Naive (predict train mean for every test node): test R²=−0.0002 | `results/split_diagnostic_comparison.csv`, row `model=naive_train_mean` | Reference point: "no skill" is ~0.000 by construction on a representative split (train/test means are close here, 404.6 vs 403.3, so naive lands near zero rather than sharply negative as it did on the earlier, rejected contiguous-band split) |

**Reading #10-15 together:** the baseline GNN (R²=−0.383) underperforms
linear regression (R²=+0.031) on identical features and an identical
spatially-blocked test set, and underperforms the naive predictor too.
This is evidence of a real generalization problem specific to how much
capacity (GNN, RF) is being fit to this feature set and node count — not a
pipeline bug (confirmed: sane prediction magnitudes, proper early
stopping, val/test never mixed) and not purely a "no signal in the
data" problem (linear regression does find a small positive signal).
Recorded honestly as the Day 1 checkpoint result in `DAY1_STATUS.md`.

---

---

## Day 2 (2026-08-29 continued): three-way model comparison + CQR

Per-model regression metrics, full precision in
`results/{linear_baseline,baseline_gnn,physics_gnn}_metrics.csv`; the
combined table (metrics + plausibility together) is
`results/day2_model_comparison.csv`. All three models share the identical
graph, features, seed=42, and train/val/test split (`data/processed/graph_nodes.csv`)
— only the model/loss differs.

| # | Number | Exact source | Note |
|---|---|---|---|
| 16 | Linear regression: test R²=0.031, RMSE=96.0, MAE=72.9, n=2,236 | `results/linear_baseline_metrics.csv`, row `split=test_spatial_block` | Formalized version of Day 1's ad hoc diagnostic (#13) — same split, same numbers, now a first-class saved result |
| 17 | Baseline GNN (no physics): test R²=−0.383, RMSE=114.7, MAE=89.7 | `results/baseline_gnn_metrics.csv`, row `split=test_spatial_block` | Same as Day 1 entry #12 — unchanged, re-confirmed on this exact graph |
| 18 | **Physics-informed GNN: test R²=−0.219, RMSE=107.7, MAE=85.1, n=2,236** | `results/physics_gnn_metrics.csv`, row `split=test_spatial_block`; `w_prior=0.3`, `w_nonneg=0.1`, `best_epoch=57` | **Physics term improves test R² over the no-physics baseline by +0.165 (−0.383 → −0.219), and improves val R² by +0.064 (0.137 → 0.201, `split=val` rows in the same two files) — but does NOT close the gap to linear regression's +0.031.** This is the Day 2 hard-checkpoint trigger — see DAY2_STATUS.md. |
| 19 | Prior-generalization correlation (test, vs. the same mass-balance proxy the physics loss trained toward): linear=−0.060, baseline GNN=+0.048, **physics GNN=+0.645** | `results/day2_model_comparison.csv`, column `physical_consistency_corr_test`, all three model rows | **Corrected wording (2026-08-29, Day 3):** an earlier pass labeled this "physical-consistency correlation" and treated it as evidence of physical reliability. It is not independent validation — the proxy correlated against is the exact training target of the physics loss, so this measures whether the imposed prior generalized to spatially held-out nodes (real and non-trivial, since nothing else in this AOI generalized well) rather than whether the prior is physically true. Column name in `results/` files is left as `physical_consistency_corr_test` (already generated, not worth regenerating for a rename); every prose reference must use the corrected framing, not the column name. |
| 20 | Edge-incoherence rate (test): linear=0.00031, baseline GNN=0.0, physics GNN=0.0 | `results/day2_model_comparison.csv`, column `edge_incoherence_rate_test`; threshold=102.6 t/ha computed by `train_edge_jump_threshold()` (95th pct of real train-side neighbor differences, `src/eval/plausibility.py`) | Does NOT differentiate the two GNN variants (both 0.0) — this metric mainly separates "has graph smoothing" (both GNNs) from "doesn't" (linear), not baseline-vs-physics specifically. Reported as-is, not spun as a physics-term win. |
| 21 | Negative-prediction rate (test): 0.0 for all three models | `results/day2_model_comparison.csv`, column `neg_rate_test` | No model violates non-negativity on this data even without the constraint — the non-negativity term (part of the physics loss) had nothing to visibly fix here; reported honestly as a non-differentiating metric, not omitted |
| 22 | CQR empirical coverage: 85.7% at 90% nominal, 90.2% at 95% nominal | `results/cqr_coverage.csv`, `empirical_coverage_test` column, rows `nominal_level=0.90` and `0.95` | Both undershoot nominal by ~4.3-4.8pp — calibrated on val (n=2,424), evaluated once on test (n=2,236, same split as all other Day 2 results). Consistent with the spatial-exchangeability caveat already flagged in `uq_decision.md` before any experiment was run |
| 23 | CQR mean interval width: 261.9 t/ha (90%), 310.0 t/ha (95%) | `results/cqr_coverage.csv`, column `mean_interval_width_tha` | For reference, SOC label std is 103.4 t/ha (`NUMBER_SOURCE_LOG.md` #6) — intervals are ~2.5-3x the label's std, i.e. genuinely wide, not falsely confident, consistent with the weak point-prediction skill found in #16-18 |
| 24 | Conformal correction Q: −3.99 (90%), −5.31 (95%) | `results/cqr_coverage.csv`, column `conformal_correction_Q` | Negative correction means the base pinball-loss quantile regressor was already slightly conservative (wide) on the calibration set — CQR tightened rather than widened here. Legitimate CQR behavior (Romano et al. 2019 designed the method to correct in either direction), not a bug. Test undercoverage despite this tightening reinforces that calibration-set (val) and evaluation-set (test) residual distributions genuinely differ. |

**Hard-checkpoint verdict (per the user's Day 2 instructions):** physics-GNN
test R² (−0.219, #18) is still below linear regression's test R² (+0.031,
#16). **The reframe is required and has been applied** — see
`DAY2_STATUS.md` for the explicit statement. Entry #19 (prior-
generalization correlation) is the number that substantiates the reframed
claim: the constraint generalizes to held-out nodes, not that it has been
physically validated.

---

---

## Day 3 (2026-08-29 continued): pricing proof-of-concept

Full precision in `results/pricing_poc_examples.csv`, generated by
`src/pricing/pricing_poc.py`. Illustrative mechanism, not a validated
pricing system — see that script's docstring for the full disclosure
(ALPHA=1.0 and MAX_ADJ=0.5 are stated, arbitrary illustrative choices, not
fit to any market data). FLAT_BUFFER_PCT=20 is Plan Vivo Standard V5.0's
stated flat non-permanence buffer (`literature_notes.md` §3), not an
average — **not arbitrary**, unlike ALPHA/MAX_ADJ above.

**Corrected 2026-08-29 (same day, later pass): FLAT_BUFFER_PCT was 17.5
(the midpoint of Plan Vivo v4.0's "15-20%" range), changed to 20 (Plan
Vivo V5.0's single stated value) once V5.0 was confirmed as the current,
correctly-cited version (`literature_notes.md` §3 merge).** Averaging a
superseded version's range when the actual cited source states one fixed
number was the error — fixed by using that number directly.
`results/pricing_poc_examples.csv` regenerated; `price_multiplier` column
(our own mechanism, entry #26) is unchanged, since it does not depend on
`FLAT_BUFFER_PCT` — only the comparator columns (`flat_buffer_multiplier`,
`multiplier_vs_flat_buffer_pp`) changed.

| # | Number | Exact source | Note |
|---|---|---|---|
| 25 | Test-set 90%-CQR interval width: median=257.6 t/ha, range 114.8-425.4 t/ha | `results/pricing_poc_examples.csv` printed summary / computed from `results/cqr_predictions.csv` `hi_adj_0.9 - lo_adj_0.9` on `split=test` rows | Reference distribution the 5 examples are percentile ranks of (min/p25/median/p75/max), not hand-picked nodes. Unaffected by the FLAT_BUFFER_PCT correction. |
| 26 | 5 example nodes' price multipliers: 1.500 (min width), 1.072 (p25), 1.000 (median), 0.899 (p75), 0.500 (max width) | `results/pricing_poc_examples.csv`, column `price_multiplier` | Min/max are clipped at the stated ±50% bound (`MAX_ADJ=0.5`); p25/median/p75 are unclipped, smoothly graduated values. Unaffected by the FLAT_BUFFER_PCT correction (verified identical before/after). |
| 27 | **Flat-buffer comparator multiplier: 0.800** (i.e. a flat 20% discount, Plan Vivo V5.0) for every node, vs. our per-node range of 0.500-1.500 | `results/pricing_poc_examples.csv`, columns `flat_buffer_multiplier` and `price_multiplier` | **Corrected from 0.825.** The illustrative point: a 100-percentage-point spread (50 to 150) across 5 real test nodes vs. one fixed number for all of them under a flat buffer — the spread itself is unchanged by the correction, only the fixed comparator value is. |

---

---

## Day 3 (2026-08-29 continued): paper figures

Both figures are generated programmatically (matplotlib, vector PDF output,
`pdf.fonttype=42` so text stays embedded/editable rather than rasterized) by
scripts under `scripts/`, and inserted into `paper/main.tex` in
Section III (Fig. 1) and Section IV (Fig. 2). Neither figure introduces any
number not already logged above; this entry records which script reads
which already-logged source file, per the same standard as every other
number in this log.

| # | Figure | Exact source | Note |
|---|---|---|---|
| 28 | **Fig. 1 (`paper/figures/fig1_pipeline.pdf`), pipeline schematic.** Flow-box text (10,124 nodes, 64,366 directed edges, four data source names) is static prose pulled from entry #7 / `main.tex` Section III, not recomputed. The inset AOI map is real, plotted data: a scatter of all 10,124 node `(lon, lat)` pairs. | Generated by `scripts/make_fig1_pipeline.py`, reading `data/processed/graph_nodes.csv` columns `lon`/`lat` for the inset map. Node/edge counts trace to entry #7 (`build_graph.py` stdout, same CSV). | No shapefile or GMW raster was available as a ready AOI boundary, so the inset uses the actual graph node locations instead — real data, not a fabricated basemap. `data/raw/gmw/gmw_v3_2020_ten_thousand_islands_fl_clip.tif` (the GMW extent raster) exists in the repo but was not used here since the node scatter already gives the true AOI shape at equal or better fidelity for a thumbnail-sized inset. |
| 29 | **Fig. 2 (`paper/figures/fig2_results.pdf`), two-panel results figure.** Panel (a): `test_r2` and `physical_consistency_corr_test` for all three models, at full CSV precision (no additional rounding). Panel (b): `nominal_level` and `empirical_coverage_test` for both nominal levels, plus a reference `y=x` line and vertical connectors sized from the same two columns. | Generated by `scripts/make_fig2_results.py`, reading `results/day2_model_comparison.csv` (panel a) and `results/cqr_coverage.csv` (panel b) directly — same two files as entries #16-19 and #22 respectively. | Panel (a)'s y-axis label uses "prior-gen. corr.", the corrected framing from entry #19, not "physical-consistency correlation" (the CSV column's literal name) — same correction, applied consistently in the figure as in prose. Panel (b) deliberately does not clip or rescale the y-axis to hide the gap between the empirical points and the dashed diagonal; the undercoverage (entry #22) is a real, reported finding. |

---

---

---

## Phase 0 (2026-08-29, journal-extension sprint): test-set integrity fix

**Problem, stated plainly:** Day 1-2's 2,236-node test split (`split_legacy_day1_2`
below) was directly inspected to make model-selection/narrative decisions
-- the baseline-vs-physics "hard-checkpoint" comparison in `DAY2_STATUS.md`
explicitly compares test $R^2$ across three model families to decide what
the paper claims, and Day 2's val numbers were separately used for a
similar cross-model narrative point. Reusing either set as "test" for any
further decision would let information from an already-inspected
evaluation leak into a number presented as untouched. Fixed by carving a
genuinely fresh val/test pair from ONLY the legacy-train superblocks (never
individually inspected -- only ever touched via gradient steps), folding
legacy val+test entirely into the new train pool. Full design rationale is
in `src/data_pipeline/build_graph.py`'s Phase-0 comment block (search
`PHASE1_SPLIT_SEED`); this is a genuine three-way spatial split (not a
random one), same superblock-geometry logic as Day 1-2, different candidate
pool and RNG seed.

| # | Number | Exact source | Note |
|---|---|---|---|
| 30 | **New (operative) split: train=6,465, val=1,583, test=2,076** (all 10,124 nodes; NDWI/NDMI columns added, node count unchanged from Day 1-2's 10,124) | `src/data_pipeline/build_graph.py` stdout, re-run 2026-08-29; `data/processed/graph_nodes.csv` column `split` | Per-split SOC mean: train=427.9, val=421.2, test=399.4 t/ha (AOI mean 421.0) -- reasonably balanced, no split near-degenerate |
| 31 | Legacy Day 1-2 split, kept for audit only: train=5,464, val=2,424, test=2,236 | `data/processed/graph_nodes.csv` column `split_legacy_day1_2` | **Verified disjoint from the new split by assertion in `build_graph.py`** (`new-test nodes that were legacy val/test (must be 0)`: confirmed 0) -- every new-test/new-val node comes exclusively from legacy-train |
| 32 | NDWI/NDMI added as extra columns, all 10,124 nodes retained (no NaNs) | `data/processed/graph_nodes.csv` columns `ndwi`, `ndmi`, `b03`; `fetch_sentinel2.py:fetch_green_band()` stdout | B03 (Green) pulled for the EXACT same already-identified scene (`S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114`) used for B04/B08/B11 -- a small, disclosed exception to "no new data pulls," confirmed with the user before pulling (see chat record) because standard McFeeters NDWI needs Green, which was never downloaded. NDWI=(B03-B08)/(B03+B08) (McFeeters), NDMI=(B08-B11)/(B08+B11) (Gao) -- kept as two DIFFERENT formulas specifically so the two Phase 1C ablations are non-degenerate |

---

## Phase 1 (2026-08-29, journal-extension sprint): pre-registered robustness/selection experiments, VAL-ONLY

All numbers below come from `scripts/run_phase1{a,b,c,d}_*.py`, on the Phase 0 split, decided on **val only** -- test was not read by any of these scripts (verified: none of them import or touch a `test` mask except for early-stopping-irrelevant model internals). Every row run is logged, including non-selected configurations, per the instruction not to drop "obviously bad" results.

### 1A: multi-seed stability (5 seeds x 3 models)

| # | Number | Exact source | Note |
|---|---|---|---|
| 33 | Linear regression (deterministic, no seed dependence): val R²=−0.030, RMSE=91.4, prior-gen corr=0.486 (std=0 by construction) | `results/phase1a_multiseed_summary.csv`, row `linear_regression` | OLS has a closed-form solution -- reported once, not simulated across 5 seeds |
| 34 | Baseline GNN (no physics): val R² mean=+0.005, std=0.019; RMSE mean=89.8, std=0.9; prior-gen corr mean=0.544, std=0.056 | `results/phase1a_multiseed_summary.csv`, row `baseline_gnn_no_physics`; full 5 rows in `results/phase1a_multiseed_val.csv` | |
| 35 | **Physics GNN (Day-2 weights, w_prior=0.3): val R² mean=−0.040, std=0.026; RMSE mean=91.8, std=1.1; prior-gen corr mean=0.632, std=0.137** | `results/phase1a_multiseed_summary.csv`, row `physics_gnn` | **Key finding: under multi-seed averaging on the Phase-0 split, the physics GNN's mean val R² is now BELOW the baseline GNN's (−0.040 vs +0.005) -- reversing part of Day 2's single-seed narrative. The prior-gen-corr gap (0.632 vs 0.544/0.486) is real but far smaller than Day 2's single-seed gap (0.645 vs 0.048/−0.060).** This is exactly what a multi-seed robustness check is for -- reported plainly, not smoothed over. |

### 1B: pre-registered lambda sweep (7 values x 5 seeds = 35 runs)

Decision rule (fixed before running, see `scripts/run_phase1b_lambda_sweep.py` docstring): select the lambda with the best mean val prior-gen corr among those within 0.05 of the best mean val R².

| # | Number | Exact source | Note |
|---|---|---|---|
| 36 | All 35 runs (7 lambda x 5 seeds), val R²/RMSE/prior-gen corr, none dropped | `results/phase1b_lambda_sweep.csv` | lambda=0.01 best mean val R² (+0.0019); lambda=2.0 worst (−0.069) |
| 37 | Per-lambda means: see table below | `results/phase1b_lambda_summary.csv` | λ=0.01: R²=+0.0019, corr=0.557 · λ=0.05: R²=−0.0005, corr=0.612 · λ=0.1: R²=−0.0067, corr=0.670 · λ=0.3: R²=−0.0395, corr=0.632 · λ=0.5: R²=−0.0529, corr=0.629 · λ=1.0: R²=−0.0681, corr=0.622 · λ=2.0: R²=−0.0694, corr=0.587 |
| 38 | **Candidates within tolerance of best (R²≥−0.0481): {0.01, 0.05, 0.1, 0.3}. SELECTED λ=0.1 (mean val prior-gen corr=0.6697, highest among candidates)** | `scripts/run_phase1b_lambda_sweep.py` stdout, mechanical application of the pre-registered rule | **Supersedes Day 2's ad hoc λ=0.3.** λ=0.3 (0.3 was Day 2's arbitrary choice) is NOT selected under the pre-registered rule -- λ=0.1 has both better mean val R² and better mean val prior-gen corr than 0.3 |

### 1C: feature ablation (base+NDWI vs base+NDMI, at selected λ=0.1, 5 seeds each = 10 runs)

Decision rule (fixed before running, symmetric with 1B, see script docstring): among {base, base+NDWI, base+NDMI}, select the config with best mean val prior-gen corr among those within 0.05 val R² of the best.

| # | Number | Exact source | Note |
|---|---|---|---|
| 39 | base (reused from 1B's λ=0.1 rows): R²=−0.0067, corr=0.6697 · base+NDMI: R²=−0.0103, corr=0.5727 · base+NDWI: R²=−0.0204, corr=0.5091 | `results/phase1c_feature_summary.csv`; full 10 new rows in `results/phase1c_feature_ablation.csv` | All 3 candidates within the 0.05 R² tolerance band |
| 40 | **SELECTED feature set: base (neither NDWI nor NDMI improved mean val prior-gen corr over base)** | `scripts/run_phase1c_feature_ablation.py` stdout | Honest null result -- reported as such, not suppressed |

### 1D: localized (Mondrian) vs. global vs. density-weighted CQR, val-internal comparison

Val (n=1,583) split 50/50 (seed=7, disclosed in script) into `val_calib`/`val_check` so the global-vs-localized comparison never touches test. Quantile model: same disclosed scope as Day 2 (pinball loss, data term only, no physics loss), trained on the Phase 1C-selected base feature set.

| # | Number | Exact source | Note |
|---|---|---|---|
| 41 | 90% nominal: global cov=88.0%, width=307.6 · localized(Mondrian) cov=92.2%, width=305.3 · density-weighted cov=81.1%, width=220.2. 95% nominal: global cov=95.6%, width=367.8 · localized cov=95.6%, width=358.2 · density-weighted cov=84.7%, width=241.5 | `results/phase1d_cqr_calibration_comparison.csv` | Localized used 6 val-superblocks with ≥30 calib points (`MIN_LOCAL_CAL_N`), 0 test-check nodes needed global fallback (all val_check nodes fell inside a calibrated block) |
| 42 | **Density-weighted variant (Gaussian kernel, bandwidth=0.125km = median calib-node nearest-neighbor distance): under-covers both levels, notably narrower intervals** | Same file, `calibration_method=density_weighted` rows | **Honest negative finding, not hidden:** at this data-driven but very small bandwidth (≈ one node-hop), each query point's effective calibration sample is small, producing noisy, systematically-narrow, under-covering intervals. Not tuned further (time-boxed, per the instruction's own "if time allows" framing) -- reported as a first-pass sensitivity check that did not outperform, not as a validated alternative |
| 43 | **CQR method carried forward to Phase 2: localized (Mondrian).** This was the method Phase 1D was instructed to implement (not a competing candidate selected by a mechanical rule the way 1B/1C were) -- global remains the "before" comparator, density-weighted is a reported-but-not-adopted secondary check | `paper/method_notes.md` (Phase 1D section) | Decided BEFORE Phase 2 touched test -- see entry #45 for what happened when this was actually evaluated on test |

---

## Phase 2 (2026-08-29, journal-extension sprint): the one and only test-set evaluation

Configuration evaluated (fixed by Phase 1's mechanical rules, no test-informed tuning): λ=0.1, w_nonneg=0.1 (fixed, never swept), feature set=base, CQR=localized Mondrian, seed=42 (project's declared standard seed). `scripts/run_phase2_test_eval.py`, run once.

| # | Number | Exact source | Note |
|---|---|---|---|
| 44 | **Three-way comparison, TEST: linear regression R²=−0.450, RMSE=105.4 · baseline GNN R²=−0.383, RMSE=103.0 · physics GNN (λ=0.1) R²=−0.430, RMSE=104.7. Prior-gen corr (test): linear=0.686, baseline GNN=0.682, physics GNN=0.755** | `results/phase2_test_comparison.csv` | **Different headline from Day 2: under the corrected split + pre-registered λ, the BASELINE (no-physics) GNN has the best test R² of the three, not linear regression, and not the physics GNN.** The physics GNN still has the highest prior-gen corr, but by a much smaller margin (+0.07-0.09) than Day 2's single-seed 0.645-vs-0.05 gap -- consistent with the multi-seed Phase 1A finding (#35). Reported as measured, no reframing to match the old narrative. |
| 45 | **CQR test coverage: global 90%→91.1% (width 335.6), 95%→96.4% (width 382.6). Localized Mondrian 90%→83.4% (width 317.5), 95%→92.4% (width 358.8)** | `results/phase2_cqr_test_coverage.csv` | **Localized (Mondrian) -- the method pre-registered in Phase 1D based on val-internal performance -- UNDERCOVERS on the real test set, and undercovers MORE than global does.** This is reported as the primary/selected result (it was fixed before test was touched) with this limitation stated plainly, not swapped post-hoc for the better-looking global number. Likely cause: only 6 val superblocks qualify for local calibration (≥30 points each); per-block quantile estimates trade robustness for spatial specificity that this AOI's calibration volume can't fully support -- stated as a limitation, not hidden |
| 46 | 1 test node (of 2,076) produced an inverted/degenerate Mondrian interval (hi<lo) at both nominal levels -- floored to 0 width for reporting, excluded from pricing PoC's representative examples | `results/phase2_cqr_test_coverage.csv` column `n_degenerate_interval_test`; `src/pricing/pricing_poc.py` | A block-local correction can occasionally over-tighten past zero width; disclosed, not silently patched. Does not affect the coverage calculation itself (an inverted interval already correctly counts as a coverage miss) |
| 47 | Pricing PoC (Phase 2 test intervals, localized Mondrian, 90%): median width=328.9 t/ha, range 50.7-471.5 (after excluding the 1 degenerate node); 5 example multipliers 1.500/1.238/1.000/0.784/0.566 vs. flat 0.800 | `results/pricing_poc_examples.csv` | Same illustrative mechanism as Day 3 (`ALPHA=1.0`, `MAX_ADJ=0.5`, `FLAT_BUFFER_PCT=20`, all unchanged), re-run on Phase 2's test intervals instead of Day 2's |

---

## Phase 3 (2026-08-29, journal-extension sprint): figures regenerated for the new numbers

| # | Figure | Exact source | Note |
|---|---|---|---|
| 48 | **Fig. 1 pipeline schematic, updated.** Node/edge counts (10,124 / 64,366) and AOI inset unchanged (Phase 0 did not change node count, only which nodes are train/val/test); CQR box text updated to name the actual method used. | `scripts/make_fig1_pipeline.py`, regenerated 2026-08-29 -> `paper/figures/fig1_pipeline.pdf` | Box now reads "Localized CQR ... block-local (Mondrian), calibrated on val", replacing the old generic "CQR calibration, split-conformal" text, matching entry #43/#45 |
| 49 | **Fig. 2 results panel, rebuilt (new data sources, not just re-styled).** Panel (a): val $R^2$ and val prior-gen. corr., 5-seed mean +/- std bars, model comparison. Panel (b): empirical vs. nominal CQR coverage on TEST, global vs. localized (Mondrian), both plotted plainly (localized undercovers more -- not hidden or rescaled). | `scripts/make_fig2_results.py`, regenerated 2026-08-29 -> `paper/figures/fig2_results.pdf`, reading `results/phase1a_multiseed_val.csv` (panel a) and `results/phase2_cqr_test_coverage.csv` (panel b) | Panel (a) is now a genuine robustness figure (error bars from Phase 1A's 5 seeds), not single-seed point estimates as Day 3's version was; linear regression's bar has no error bar (deterministic, std=0 by construction, not simulated) |

---

## Phase 6 (2026-08-29, journal-extension sprint): final source-log audit

Every number in the compiled `main.tex`/`main.pdf` (abstract, all 3 tables,
every inline figure in Method/Results/Discussion/Pricing/Limitations) was
re-checked programmatically against the exact source CSV, not re-typed
from memory or from this log's own rounded entries:

- `results/phase2_test_comparison.csv` vs. Table I and the Results/
  Discussion prose (train/test $R^2$, RMSE, edge incoherence, prior-gen
  corr., edge-jump threshold) -- exact match to the precision shown.
- `results/phase2_cqr_test_coverage.csv` vs. Table II and Results/
  Discussion prose (coverage, width, both methods, both levels) --
  exact match.
- `results/phase1b_lambda_summary.csv` vs. the Method section's stated
  selected $\lambda$, its mean val $R^2$/prior-gen corr., and the
  mechanical tolerance-band arithmetic (best $R^2$ 0.0019, band
  $\geq{-0.0481}$, candidates \{0.01, 0.05, 0.1, 0.3\}, argmax corr.\ at
  0.1) -- recomputed from full-precision values, not the rounded log
  entries, and confirmed.
- `results/phase1a_multiseed_summary.csv` vs. Section IV-A prose and
  Fig.~2(a)'s underlying data -- exact match, including linear
  regression's std=0 (deterministic, not simulated).
- `results/phase1d_cqr_calibration_comparison.csv` vs. Section IV-B --
  exact match, all three calibration methods, both nominal levels.
- `results/phase1c_feature_summary.csv` vs. the Method section's feature-
  ablation selection statement -- exact match; base wins, as stated.
- `results/pricing_poc_examples.csv` vs. Table III and the Pricing PoC
  prose (widths, multipliers, the 93.4-percentage-point spread, the
  0.566/1.500 endpoints) -- exact match, including the post-fix (1
  degenerate node excluded, entry #46) distribution.
- Abstract cross-checked sentence-by-sentence against the body sections
  that first state each number -- no abstract-only number found.

**Result: every number in the manuscript traces to a specific results CSV,
row/column, or a direct, reproducible arithmetic derivation from one
already logged (percentage-point spreads, tolerance-band arithmetic).
Nothing added, changed, or fabricated to close a gap.** LaTeX compile
(`pdflatex` + `bibtex`, 3-pass, from a clean `rm *.aux *.bbl *.blg *.log`):
**5 pages, 0 warnings/errors** on the final pass (`paper/main.log`) --
the last two sub-pixel hbox notices were resolved by an explicit
`\penalty0` break inside one long bibliography URL and a modest
`\emergencystretch`/compact-spacing preamble (documented inline in
`main.tex`), not by hiding real overflow; the rendered page layout is
otherwise the IEEEtran conference default.

---

---

## Day 4 continued (2026-08-29): final reframe -- global-calibration pricing PoC + 2 new citations

Triggered by the user's explicit final-reframe instruction: since the
Discussion (Section~V) now recommends global CQR as the defensible
near-term deployment choice (localized/Mondrian undercovered more on
test, entry #45), the Pricing PoC was rebuilt on global intervals so the
paper doesn't recommend one method while illustrating deployment on the
other. `src/pricing/pricing_poc.py` updated to read
`lo_adj_global_0.9`/`hi_adj_global_0.9` instead of the Mondrian columns;
re-run once, same mechanism (`ALPHA=1.0`, `MAX_ADJ=0.5`,
`FLAT_BUFFER_PCT=20`, all unchanged).

| # | Number | Exact source | Note |
|---|---|---|---|
| 50 | **Superseded: entry #47's Mondrian-based pricing numbers** (median width 328.9, range 50.7-471.5, multipliers 1.500/1.238/1.000/0.784/0.566) | -- | Not wrong, not a bug -- superseded because the paper's own Discussion now recommends against deploying on localized/Mondrian intervals, so illustrating pricing on them would be internally inconsistent. Kept in the log (entry #47 above, unedited) for audit, not restated as current. |
| 51 | **Global-calibration pricing PoC (current): test-set 90%-CQR interval width median=341.1 t/ha, range 143.9-552.5 t/ha (n=2,076, no degenerate/inverted intervals under global, unlike Mondrian's 1)** | `results/pricing_poc_examples.csv`, regenerated 2026-08-29; verified `n_degenerate_interval_test`=0 for the `global` rows in `results/phase2_cqr_test_coverage.csv` before switching | |
| 52 | **5 example nodes' price multipliers (global): 1.500 (min width, clipped), 1.125 (p25), 1.000 (median), 0.917 (p75), 0.500 (max width, clipped)** vs. flat-buffer 0.800 | `results/pricing_poc_examples.csv`, column `price_multiplier` | Min and max both land exactly on the stated $\pm$50\% clip bound this time (coincidental, not adjusted to make a rounder number) -- full 100-percentage-point spread (0.500 to 1.500) |
| 53 | Mao, Martin \& Reich (2024), *JASA* 119(546):904-914, "Valid Model-Free Spatial Prediction," doi:10.1080/01621459.2022.2147531 | Verified directly against the arXiv abstract page (arXiv:2006.15640, preprint of this published version) before citing -- author list, volume/issue/pages/DOI all confirmed, not taken from a search snippet alone | Cited in Discussion (Section~V) as the theoretical grounding for the localized-CQR undercoverage result: their infill-asymptotic validity condition requires calibration-point density per local region that this AOI's 1,583-node validation set, split across spatial blocks, likely didn't meet |
| 54 | Hjort, Hermansen, Pensar \& Williams (2023), arXiv:2312.06531, "Uncertainty Quantification in Automated Valuation Models with Spatially Weighted Conformal Prediction" | Verified directly against the arXiv abstract page before citing | Cited in Discussion for the precise, verified finding actually in their paper (not overstated): their own Mondrian-style variant showed "little to no improvement" over global in one tested scenario, and they caution that an apparent localization gain can reflect a non-conformity score's failure to capture spatial trends rather than localization succeeding -- situates our negative result within a known, domain-general pattern in asset-pricing conformal calibration, not an isolated failure. Full verification notes: `paper/literature_notes.md` §4. |
| 55 | Willard, Jia, Xu, Steinbach \& Kumar (2022, formally ACM Comput.\ Surv.\ 55(4), Article 66, 2023), "Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems," doi:10.1145/3514228 | Verified directly against the ACM DL/arXiv listing (author list, DOI, volume/issue) before citing, not taken from a search snippet alone | Cited in Related Work §1 as the general physics-guided-ML motivating survey, not as environmental/blue-carbon-specific prior art; note the online-availability year (2022) differs from the formal journal-issue year (2023) -- cited as 2023 to match the journal's own bibliographic record, disclosed in `literature_notes.md` §1 |
| 56 | Roy, Pitumpe Arachchige, Watt et al.\ (2024), *Science of The Total Environment* 938:173270, "Remote sensing-based mangrove blue carbon assessment in the Asia-Pacific: A systematic review," doi:10.1016/j.scitotenv.2024.173270 | Verified directly (DOI, journal, volume/article number, full 24-author list) against a mirror carrying the publisher's own citation record before citing | Cited in Related Work §2 to independently supplement (not replace) Radabaugh et al.\ as evidence the belowground-carbon/valuation gap is field-recognized (their own systematic review of 105 studies states this). **Explicitly not used to broaden this paper's own gap statement** -- the Introduction's narrow-gap sentence (single-region calibration-strategy validity, not the broader multi-region/multi-pool MRV agenda Roy et al.\ separately identifies) was added specifically to keep this distinction stated, not implied. |

**Final reframe audit (2026-08-29, after entries #50-56):** every changed
number re-checked once more against its source CSV after the pricing
switch (Table III / abstract range recomputed from
`results/pricing_poc_examples.csv` directly, not derived by hand); both
new citations (#53-56, 4 total incl.\ #55-56) verified against a primary
source (arXiv abstract page, ACM DL listing, or a mirror carrying the
publisher's own citation record) before being added to `references.bib`
-- none taken from a search-engine snippet alone. LaTeX compile
(`pdflatex` + `bibtex`, 3-pass, from a clean `rm *.aux *.bbl *.blg *.log`):
**5 pages, 0 warnings/errors** on the final pass. Author block updated to
the real affiliation (Nahla Nabil Skaik, Arab Open University, Bahrain),
provided directly by the user -- the `% TODO(author)` placeholder open
since Day 3 is now resolved.

---

---

---

## Post-submission journal-extension sprint (2026-08-30): real dC/dt experiment

**Everything in this section is exploratory work for the eventual journal
extension only. It does NOT touch `paper/main.tex`, does not change
Table I/II or any headline submitted number, and does not read from or
write to `data/processed/graph_nodes.csv` / `graph_edges.npy` (the
submitted paper's graph -- read exactly once, read-only, to look up which
coarse superblocks its test split occupies; see entry #60). All new code
lives in files with an `ext`/`temporal` name; nothing existing was
modified except an additive `.gitignore` whitelist entry for the two new
processed-graph files. Full write-up:
`paper/extension_temporal_notes.md`.**

### Step 1: second Sentinel-2 date

| # | Number | Exact source | Note |
|---|---|---|---|
| 57 | **Scene 1 (unchanged, already used by the submitted paper): `S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114`, 2026-02-06, tile T17RMJ.** Scene 2 (NEW): `S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532`, 2026-08-15, tile T17RMJ, cloud cover 20.07% (scene-wide). **Separation: 190 days (~6.2 months).** | `data/raw/sentinel2/stac_items_ten_thousand_islands_fl_2026-08-15_2026-08-16.json` (STAC search+clip output, run 2026-08-30 via Microsoft Planetary Computer, keyless); clipped bands at `data/raw/sentinel2/sample_clip_S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532.tif` (B04/B08/B11) and `sample_clip_B03_S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532.tif` (B03, via `scripts/fetch_ext_second_date_green.py`, reusing `fetch_sentinel2.py:fetch_green_band()` unchanged with the new scene's item id) | **Disclosed, not a shortcut:** 190 days is at the low end of the requested 6-12 month window because this session's live STAC access only has scenes up to "today" (2026-08-30); no scene closer to 12 months apart exists yet to pull. A search of 2026-07-15/2026-08-30 at <20% cloud returned 0 scenes (Florida wet season); widening to <60% over 2026-06-01/2026-08-30 returned 22, and the same-tile (T17RMJ), lowest-cloud candidate closest to "today" (maximizing separation) was 2026-08-15 at 20.07%. Still spans SW Florida's real dry-to-wet season transition -- confirmed by a real, non-degenerate NDVI-delta distribution (entry #59). |

### Step 2: dC/dt proxy graph build

| # | Number | Exact source | Note |
|---|---|---|---|
| 58 | **All 10,124 nodes from the submitted paper's graph retained with both dates' bands valid (0 dropped for date-2 cloud/nodata); 64,366 directed edges (identical count -- same node set, same 8-connectivity adjacency, unchanged geometry).** `d_ndvi` (= `ndvi_t2 - ndvi_t1`) distribution: min=-0.4034, max=0.2733, mean=+0.0248, std=0.0662. | `src/data_pipeline/build_graph_temporal.py` stdout, run 2026-08-30; `data/processed/graph_nodes_ext_temporal.csv` (new file, not the submitted graph) | Positive mean d_ndvi (dry-season Feb -> wet-season Aug green-up) is the expected direction ecologically and confirms the two-date signal is real, not degenerate. `ndvi_rate_per_day = d_ndvi / 190`. Also computed (per the instruction's "other available band deltas"): `d_ndwi`, `d_ndmi`, `d_b11` -- all written as columns, only `ndvi_rate_per_day` is used in the physics proxy (see #59 below for why). |
| 59 | **dC/dt proxy formula (accretion term only changed): `physics_gnn_dcdt` proxy = 0.5·scaled(`ndvi_rate_per_day`) + 0.5·scaled(`dist_to_tide_km`)**, same equal-weighted / train-only-min-max / no-fitted-coefficients form as the submitted paper's proxy | `src/models/physics_terms_temporal.py`, `compute_mass_balance_proxy_dcdt()` | Export term (`dist_to_tide_km`) deliberately UNCHANGED -- its own, separate, already-disclosed positional-proxy limitation is not addressed by this extension (see that module's docstring). Directly targets the submitted paper's stated Limitations-section sentence: "Single-date imagery constrains the physics term to a cross-sectional relationship, not an actual $dC/dt$." |

### Step 3: fresh, non-overlapping test partition

| # | Number | Exact source | Note |
|---|---|---|---|
| 60 | **Excluded (never eligible for new test/val) coarse superblocks: `{4, 5, 7, 12, 17, 18, 26, 33, 41, 57}` -- exactly the 10 of 20 occupied coarse superblocks the SUBMITTED paper's operative test split (`data/processed/graph_nodes.csv`, `split=="test"`) occupies.** New split (`split_ext`, `data/processed/graph_nodes_ext_temporal.csv`): train=6,429, val=1,541, test=2,154 (of 10,124). | `src/data_pipeline/build_graph_temporal.py` stdout, run 2026-08-30 | Sanity check (printed, passed): zero overlap between these 10 excluded superblocks and the legacy Day 1-2 test/val superblocks -- confirms, independent of this extension, that the submitted paper's own Phase-0 fix left its operative-test superblocks entirely disjoint from every other split ever computed on this AOI. Two `assert`s in the build script additionally enforce, mechanically, that no `split_ext=="test"` or `"val"` node falls in an excluded superblock. Per-split SOC mean (t/ha): train=409.0, val=436.5, test=445.5 (AOI mean 421.0, same as always -- node set unchanged). New split seed: `EXT_SPLIT_SEED=20260830`, distinct from both `SPLIT_SEED=42` (legacy) and `PHASE1_SPLIT_SEED=20260829` (submitted paper's Phase-0 fix). |

### Step 4: pre-registered robustness/selection experiments, VAL-ONLY

Mirrors the submitted paper's Phase 1A/1B exactly (same 5 seeds {1..5}, same
lambda grid {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0}, same mechanical selection
rule: candidates = lambda with mean val R² within 0.05 of the best; select
the candidate with the highest mean val prior-gen corr), run TWICE on the
identical new split -- once against the new dC/dt proxy, once against the
submitted paper's original steady-state proxy re-targeted at the new split
(a CONTROL run, so the dC/dt-vs-steady-state comparison in Phase 2 is
single-variable: same split/seed/architecture, only the proxy differs).

| # | Number | Exact source | Note |
|---|---|---|---|
| 61 | Extension Phase 1A (5 seeds, w_prior=0.3/w_nonneg=0.1 fixed starting config): linear val R²=0.132 (deterministic); baseline GNN val R² mean=0.226, std=0.008, prior-gen corr mean=0.799; **physics GNN (dC/dt) val R² mean=0.293, std=0.012, prior-gen corr mean=0.916** | `results/ext_phase1a_multiseed_val.csv`, `results/ext_phase1a_multiseed_summary.csv` | On this new split, unlike the submitted paper's Phase 1A finding (#35, physics GNN's mean val R² fell BELOW baseline's), the physics-dC/dt GNN's mean val R² is clearly above both baseline GNN and linear regression, motivating the lambda re-sweep below rather than treating 0.3 as final. |
| 62 | **dC/dt-proxy lambda sweep (35 runs, all reported): per-lambda mean val R² rises monotonically 0.01->1.0 (0.229 -> 0.319) then dips at 2.0 (0.312); prior-gen corr rises monotonically throughout (0.809 -> 0.959 -> 0.960). Best mean val R²=0.3193 (lambda=1.0); tolerance band >= 0.2693; candidates={0.3, 0.5, 1.0, 2.0}. SELECTED lambda=2.0 (mean val R²=0.3122, mean val prior-gen corr=0.9595, highest among candidates).** | `results/ext_phase1b_lambda_sweep.csv`, `results/ext_phase1b_lambda_summary.csv`, `scripts/run_ext_phase1b_lambda_sweep.py` stdout | Supersedes nothing from the submitted paper (different track entirely) -- selection rule identical in form to the submitted paper's Phase 1B (entry #38), applied mechanically to new numbers. |
| 63 | **Steady-state-proxy CONTROL sweep (same new split, submitted paper's original proxy formula, 35 runs): mean val R² peaks at lambda=0.3 (0.269), DECLINES for lambda>=0.5 (0.264, 0.247, 0.210) -- unlike the dC/dt sweep's monotonic rise. Prior-gen corr still rises monotonically (0.496 -> 0.955). Best mean val R²=0.2695 (lambda=0.3); tolerance band >= 0.2195; candidates={0.01, 0.05, 0.1, 0.3, 0.5, 1.0}. SELECTED lambda=1.0 (mean val R²=0.2468, mean val prior-gen corr=0.9223, highest among candidates).** | `results/ext_phase1b_steadystate_control_sweep.csv`, `results/ext_phase1b_steadystate_control_summary.csv`, `scripts/run_ext_phase1b_steadystate_control_sweep.py` stdout | At every lambda from 0.01 to 1.0, the dC/dt sweep's mean val R² (#62) is higher than the steady-state control's at the same lambda on the SAME split -- an early (val-only, pre-registered-selection-stage) signal in the same direction as the eventual test result (#64). |

### Step 5: the one and only new-test-partition evaluation

| # | Number | Exact source | Note |
|---|---|---|---|
| 64 | **Extension Phase 2, four-way comparison, TEST (n=2,154, evaluated once, never read by any script above): linear regression R²=0.104, RMSE=102.5, prior-gen corr=0.665 · baseline GNN (no physics) R²=0.291, RMSE=91.2, prior-gen corr=0.787 · physics GNN, steady-state-proxy CONTROL (lambda=1.0) R²=0.214, RMSE=96.0, prior-gen corr=0.933 · **physics GNN, real dC/dt (lambda=2.0) R²=0.320, RMSE=89.3, prior-gen corr=0.952**.** | `results/ext_phase2_test_comparison.csv`, `scripts/run_ext_phase2_test_eval.py` stdout | **Headline single-variable result (same split/seed/architecture/features for both physics arms, only the proxy formula differs): the real-dC/dt physics GNN beats the steady-state-proxy control on TEST by +0.107 R² (0.320 vs 0.214) and +0.019 prior-gen corr (0.952 vs 0.933). It also has the best test R² of all four models on this split (beating baseline GNN by +0.029, linear regression by +0.216) and the highest prior-gen corr of all four.** Full interpretation, including the honest caveats (both physics variants have NEGATIVE train R² despite positive, competitive test R² -- a known pattern in this project's physics-loss variants, not a new anomaly; the steady-state control's val-ranking above baseline GNN did not hold on test, an honest val/test mismatch reported as such) in `paper/extension_temporal_notes.md`. |

### Step 4b (user-requested, after reviewing the single-seed Phase 2 table): multi-seed TEST characterization

**Deliberate, disclosed departure from this project's "evaluate test
exactly once" norm.** The user asked, before drawing any conclusion from
entry #64, whether its ~0.2-0.3 test-$R^2$ level (vs.\ the submitted
paper's negative test $R^2$s) was driven specifically by the dC/dt proxy
or largely by the new split's superblocks being easier/more homogeneous
-- a question a single seed cannot answer. `scripts/run_ext_phase2b_
multiseed_test.py` re-evaluates the SAME four already-fixed
configurations (nothing selected or tuned by this script -- every
lambda/feature-set choice was already fixed by Step 4's val-only
pre-registered sweeps) across the same 5 seeds {1..5} used throughout
this project's val-only stability checks, on TEST. This is variance
characterization of an already-decided comparison, not a second selection
round -- no downstream choice depends on any number this script produced.

| # | Number | Exact source | Note |
|---|---|---|---|
| 65 | **Four-way TEST comparison, 5-seed mean $\pm$ std (linear deterministic, std=0 by construction): linear R²=0.1044 · baseline GNN (no physics) R²=0.2886$\pm$0.0078 · physics GNN, steady-state control ($\lambda$=1.0) R²=0.2294$\pm$0.0167 · physics GNN, real dC/dt ($\lambda$=2.0) R²=0.2797$\pm$0.0274.** Prior-gen corr (test, 5-seed mean$\pm$std): linear=0.6654 · baseline=0.7975$\pm$0.0305 · steady-state control=0.9242$\pm$0.0141 · **dC/dt=0.9585$\pm$0.0107**. | `results/ext_phase2b_multiseed_test.csv` (all 20 rows), `results/ext_phase2b_multiseed_test_summary.csv` | **Revises entry #64's single-seed(42) reading.** At seed=42 alone, dC/dt's test R² (0.320) exceeded the no-physics baseline's (0.291); under 5-seed averaging, the RANKING FLIPS -- baseline GNN's mean test R² (0.289) is now marginally ABOVE dC/dt's (0.280), well within dC/dt's own std (0.027, more than 3x baseline's std of 0.008). Per-seed, baseline beats dC/dt on 3 of 5 seeds. **Conclusion: dC/dt does NOT reliably beat the no-physics baseline on raw test R² once seed variance is accounted for -- that specific comparison is a wash, not a win.** |
| 66 | **Per-seed dC/dt-vs-steady-state-control test-R² difference (the genuine single-variable, same-split/seed/architecture comparison): +0.0575, +0.0687, +0.0063, +0.0350, +0.0840 (seeds 1-5 respectively) -- POSITIVE at all 5/5 seeds, mean +0.0503.** | Computed directly from `results/ext_phase2b_multiseed_test.csv`'s per-seed `test_r2` columns, paired by seed | Unlike the baseline comparison above, this one-variable-differs comparison DOES hold up under multi-seed averaging, consistently in the dC/dt-favoring direction at every seed -- though the margin varies considerably (0.006 to 0.084), and is smaller on average than the single-seed(42) estimate in entry #64 (+0.107). Prior-gen corr shows the same pattern even more strongly: dC/dt > steady-state control at all 5/5 seeds, non-overlapping mean$\pm$std ranges (0.9585$\pm$0.0107 vs.\ 0.9242$\pm$0.0141). |
| 67 | **Split-effect magnitude (diagnostic, not a new experiment): the new split's own no-physics baseline GNN test R² (5-seed mean 0.2886) minus the SUBMITTED paper's no-physics baseline GNN test R² (single seed=42, entry #44: $-0.383$) = +0.672.** | `results/ext_phase2b_multiseed_test_summary.csv` row `baseline_gnn_no_physics` vs.\ `results/phase2_test_comparison.csv` row `baseline_gnn_no_physics` (entry #44, READ-ONLY, not re-run -- the submitted paper's test partition was not touched again to produce this number) | **This is the answer to the user's question.** A model with ZERO physics term shows almost the entire magnitude of the "jump" (+0.67 R²) between the submitted paper's test split and this extension's new one. The proxy-specific effect isolated in entry #66 (+0.05 mean R²) is real, consistent, and an order of magnitude smaller than the split effect. **The bulk of the improvement this extension appeared to show is attributable to the new split's superblocks, not to the dC/dt proxy.** Caveat on the comparison basis: the submitted-side figure is single-seed (Phase 2 never ran multi-seed on test, by design -- see that script's docstring); the new-side figure is a 5-seed mean. The submitted paper's own Phase 1A found baseline-GNN val-R² seed-to-seed std to be small (0.019, entry #34), so a 0.67 gap is not plausibly seed noise, but the comparison is not perfectly matched and is reported as a diagnostic, not a formal test. |

**Selected-$\lambda$ boundary caveat (both proxy tracks):** $\lambda$=2.0
(dC/dt, entry #62) and, in the control sweep, $\lambda$=1.0 came from a
grid whose largest value was 2.0 -- i.e.\ the dC/dt track's selected value
sits at the UPPER EDGE of the tested range $\{0.01,...,2.0\}$. The true
optimum may lie above 2.0; this sweep cannot rule that out. Per this
project's own pre-registration discipline, extending the grid upward now,
after seeing that 2.0 was selected, would not be a legitimate mechanical
application of the existing rule -- it would require a NEW pre-registered
grid, stated before running, not a post-hoc extension of this one. Not
done here. See `paper/extension_temporal_notes.md` for the full caveat.

---

---

## Superseded/rejected numbers (kept for the audit trail, not for the paper)

| # | Number | Why rejected |
|---|---|---|
| S1 | First graph split (5 equal-width longitude bands, contiguous middle band as test): baseline GNN test R²=−0.814, RF test R²=−0.517, linear test R²=−0.412, naive test R²=−0.595 | Diagnosed as an unusually harsh, single-contiguous-block split straddling a real regional carbon-density gradient (see #2's Ten Thousand Islands vs. e.g. Radabaugh's Tampa Bay figure of 265.1±43.2, PDF Table 1) — replaced by the checkerboard-superblock design in entries #7-9. Not used in the paper except possibly as a methods-section illustration of why the split design mattered. |
| S2 | First checkerboard attempt (5×5 superblock grid, superblock-count-based test selection): test=116 nodes (1.1% of graph) | Bug: selected test superblocks by COUNT of superblocks, not by node count, and mangrove nodes are extremely unevenly distributed across a uniform grid (thin coastal fringe). Fixed by selecting superblocks by cumulative node fraction instead (current `build_graph.py`). |
| S3 | Second graph split (5×5 superblock grid, node-count-balanced): baseline GNN (64-hidden, no early stopping, 300 fixed epochs) test R²=−0.334, with test loss rising monotonically from epoch 20 | Diagnosed as overfitting (train R²=0.528 vs. this test R²) compounded by a methodological error: the training script was watching test loss during training (informal early stopping against the test set). Fixed by (a) adding a proper val split (#7-9), (b) reducing model capacity 64→16 hidden units and dropout 0.1→0.3, (c) increasing weight decay 1e-4→1e-3, (d) moving to the finer 8×8 superblock grid for split stability. |
| S4 | Day 1-2's train/val/test split (5,464/2,424/2,236 nodes, `split_legacy_day1_2` column, entries #7-9 above) and every result computed on it (entries #10-29): all Day 1-2/Day 3 headline numbers (physics GNN beats baseline GNN on test R², λ=0.3, global-only CQR, 85.7%/90.2% coverage) | **Not wrong, not a bug -- superseded for test-integrity reasons (Phase 0, entries #30-32).** Day 1-2's test split was directly inspected to make model-selection/narrative decisions (the "hard-checkpoint" comparison in `DAY2_STATUS.md`), so reusing it as "test" for the journal-extension experiments would leak already-inspected information into a number presented as untouched. A fresh spatial test block was carved instead (entries #30-32); Day 1-2's numbers are kept here for audit/comparison but the paper's main results table now reports Phase 2's numbers (entries #44-47), not these. |
