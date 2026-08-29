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
| 19 | Physical-consistency correlation (test, vs. mass-balance proxy): linear=−0.060, baseline GNN=+0.048, **physics GNN=+0.645** | `results/day2_model_comparison.csv`, column `physical_consistency_corr_test`, all three model rows | The physics term's clearest, largest effect: an order-of-magnitude jump in how consistent predictions stay with the physical prior under spatial holdout, vs. either non-physics-informed model |
| 20 | Edge-incoherence rate (test): linear=0.00031, baseline GNN=0.0, physics GNN=0.0 | `results/day2_model_comparison.csv`, column `edge_incoherence_rate_test`; threshold=102.6 t/ha computed by `train_edge_jump_threshold()` (95th pct of real train-side neighbor differences, `src/eval/plausibility.py`) | Does NOT differentiate the two GNN variants (both 0.0) — this metric mainly separates "has graph smoothing" (both GNNs) from "doesn't" (linear), not baseline-vs-physics specifically. Reported as-is, not spun as a physics-term win. |
| 21 | Negative-prediction rate (test): 0.0 for all three models | `results/day2_model_comparison.csv`, column `neg_rate_test` | No model violates non-negativity on this data even without the constraint — the non-negativity term (part of the physics loss) had nothing to visibly fix here; reported honestly as a non-differentiating metric, not omitted |
| 22 | CQR empirical coverage: 85.7% at 90% nominal, 90.2% at 95% nominal | `results/cqr_coverage.csv`, `empirical_coverage_test` column, rows `nominal_level=0.90` and `0.95` | Both undershoot nominal by ~4.3-4.8pp — calibrated on val (n=2,424), evaluated once on test (n=2,236, same split as all other Day 2 results). Consistent with the spatial-exchangeability caveat already flagged in `uq_decision.md` before any experiment was run |
| 23 | CQR mean interval width: 261.9 t/ha (90%), 310.0 t/ha (95%) | `results/cqr_coverage.csv`, column `mean_interval_width_tha` | For reference, SOC label std is 103.4 t/ha (`NUMBER_SOURCE_LOG.md` #6) — intervals are ~2.5-3x the label's std, i.e. genuinely wide, not falsely confident, consistent with the weak point-prediction skill found in #16-18 |
| 24 | Conformal correction Q: −3.99 (90%), −5.31 (95%) | `results/cqr_coverage.csv`, column `conformal_correction_Q` | Negative correction means the base pinball-loss quantile regressor was already slightly conservative (wide) on the calibration set — CQR tightened rather than widened here. Legitimate CQR behavior (Romano et al. 2019 designed the method to correct in either direction), not a bug. Test undercoverage despite this tightening reinforces that calibration-set (val) and evaluation-set (test) residual distributions genuinely differ. |

**Hard-checkpoint verdict (per the user's Day 2 instructions):** physics-GNN
test R² (−0.219, #18) is still below linear regression's test R² (+0.031,
#16). **The reframe is required and has been applied** — see
`DAY2_STATUS.md` for the explicit statement. Entry #19 (physical
consistency correlation) is the number that substantiates the reframed
claim.

---

## Superseded/rejected numbers (kept for the audit trail, not for the paper)

| # | Number | Why rejected |
|---|---|---|
| S1 | First graph split (5 equal-width longitude bands, contiguous middle band as test): baseline GNN test R²=−0.814, RF test R²=−0.517, linear test R²=−0.412, naive test R²=−0.595 | Diagnosed as an unusually harsh, single-contiguous-block split straddling a real regional carbon-density gradient (see #2's Ten Thousand Islands vs. e.g. Radabaugh's Tampa Bay figure of 265.1±43.2, PDF Table 1) — replaced by the checkerboard-superblock design in entries #7-9. Not used in the paper except possibly as a methods-section illustration of why the split design mattered. |
| S2 | First checkerboard attempt (5×5 superblock grid, superblock-count-based test selection): test=116 nodes (1.1% of graph) | Bug: selected test superblocks by COUNT of superblocks, not by node count, and mangrove nodes are extremely unevenly distributed across a uniform grid (thin coastal fringe). Fixed by selecting superblocks by cumulative node fraction instead (current `build_graph.py`). |
| S3 | Second graph split (5×5 superblock grid, node-count-balanced): baseline GNN (64-hidden, no early stopping, 300 fixed epochs) test R²=−0.334, with test loss rising monotonically from epoch 20 | Diagnosed as overfitting (train R²=0.528 vs. this test R²) compounded by a methodological error: the training script was watching test loss during training (informal early stopping against the test set). Fixed by (a) adding a proper val split (#7-9), (b) reducing model capacity 64→16 hidden units and dropout 0.1→0.3, (c) increasing weight decay 1e-4→1e-3, (d) moving to the finer 8×8 superblock grid for split stability. |
