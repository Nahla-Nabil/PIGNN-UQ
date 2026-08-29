# Day 4 Status — 2026-08-29 (journal-extension sprint)

Full source trail for every number below: `paper/NUMBER_SOURCE_LOG.md`
entries #30-49 (new this sprint). This file is the phase-by-phase
narrative; the log is the audit trail.

## Phase 0 — test-set integrity fix

**Problem, verified before touching anything else:** Day 1-2's test split
(2,236 nodes) was directly inspected to decide the paper's central claim
(`DAY2_STATUS.md`'s "hard-checkpoint" compares test R² across three model
families) — a real, if informal, use of test data for a decision, on top
of its intended single-touch role.

**Fix:** `src/data_pipeline/build_graph.py` now carves a genuinely fresh
val/test pair using *only* superblocks that were legacy-train (never
individually inspected — only touched via gradient steps). Legacy val +
test are folded entirely into the new train pool. A finer 20×20 regrid
(`PHASE1_SUPERBLOCK_GRID`) was needed for this second-stage carve — the
original 8×8 grid's 10 legacy-train superblocks were too few/lumpy to hit
target proportions by greedy selection (verified empirically before
settling on 20×20). New operative split: **6,465 train / 1,583 validation
/ 2,076 test** (label means 427.9/421.2/399.4 t/ha). Disjointness from
every legacy-inspected node verified by assertion in the build script.
Legacy split kept as `split_legacy_day1_2` for audit only.

**Data-layer side quest:** Phase 1C's NDWI ablation needed the Green band
(B03), never downloaded. Confirmed with you before proceeding: pulled B03
for the exact same already-identified scene (not a new search) via a
small addition to `fetch_sentinel2.py`. NDWI (McFeeters) and NDMI (Gao)
are now genuinely distinct formulas, not the same number under two names.

**Logged:** `paper/NUMBER_SOURCE_LOG.md` #30-32; narrative in
`paper/method_notes.md`'s new opening section.

## Phase 1 — pre-registered robustness/selection, val-only

**1A (multi-seed, 5 seeds × 3 models).** Re-ran the three-way comparison at
Day 2's exact weights (λ=0.3). **Key finding:** the physics-GNN's mean val
R² (−0.040) sat *below* the baseline GNN's (+0.005) — Day 2's single-seed
R² advantage for the physics term was not stable. The prior-generalization
correlation advantage held directionally but was much smaller than Day 2's
single-seed gap (0.632 vs. 0.544/0.486, not 0.645 vs. 0.048/−0.060).

**1B (λ sweep, 7 values × 5 seeds = 35 runs, all logged).** Pre-registered
rule (best mean val prior-gen corr within 0.05 R² of the best mean val R²)
selected **λ=0.1** — not Day 2's ad hoc 0.3, which the mechanical rule does
not select.

**1C (feature ablation, NDWI/NDMI × 5 seeds = 10 runs, at λ=0.1).**
Pre-registered rule (same tolerance-band logic, declared before running)
selected **base features** — neither NDWI nor NDMI improved mean val
prior-gen correlation within tolerance. Honest null result, reported as
such.

**1D (localized/Mondrian CQR, val-internal comparison, pre-test).**
Implemented block-local (Mondrian) split-conformal calibration with a
≥30-point fallback to global, plus a density-weighted variant (Gaussian
kernel, data-driven bandwidth) as the "if time allows" extra. On a
50/50 val-internal calibration/check split (never touching test):
localized ≈ global (slightly better), density-weighted clearly
under-covered — an honest negative finding for that variant, kept in the
log, not hidden. **Localized (Mondrian) was carried forward to Phase 2**
per its pre-registered role as the method under test, not because a
mechanical rule "won" it.

All 90 individual training runs, every configuration, are in
`results/phase1{a,b,c,d}_*.csv` and logged in `NUMBER_SOURCE_LOG.md` #33-43.

## Phase 2 — the one test-set evaluation

Final configuration (λ=0.1, base features, localized-Mondrian CQR,
seed=42) evaluated **exactly once** on the 2,076-node test split.

**Headline result, different from Day 2's:** the no-physics baseline GNN
has the best test R² of the three (−0.383 vs. −0.430 physics-GNN vs.
−0.450 linear). The physics-informed model's prior-generalization
correlation is still highest (0.755 vs. 0.682/0.686) but by a much
smaller margin than Day 2 reported. **Localized CQR undercovered *more*
than global on test** (83.4%/92.4% vs. 90%/95% nominal, vs. global's
91.1%/96.4%) — the opposite of what the val-internal comparison suggested.
Reported as the pre-registered result, not swapped for the better-looking
global number after the fact; one degenerate (inverted) interval (1/2,076
nodes) was floored to zero width and excluded from the pricing examples,
disclosed in the log.

Pricing PoC re-run on these test intervals: multiplier range 0.566–1.500
vs. one fixed 0.800 flat-buffer value.

Logged: `NUMBER_SOURCE_LOG.md` #44-47.

## Phase 3 — figures

Fig. 1: pipeline box text updated to name the actual (localized/Mondrian)
CQR method; node/edge counts and AOI inset unchanged (Phase 0 didn't
change node count). Fig. 2: **rebuilt from different source data**, not
just restyled — panel (a) now shows 5-seed mean±std error bars (Phase 1A)
instead of Day 3's single-seed point estimates; panel (b) shows global vs.
localized coverage on the real test evaluation, plotted plainly (localized
undercovers more — not hidden or rescaled to look better). Logged:
`NUMBER_SOURCE_LOG.md` #48-49.

## Phase 4 — sharpened research gap

Replaced the generic "no one has combined PINN and GNN for blue carbon"
framing with a specific one: flat buffer-pool methodologies (VM0033, Plan
Vivo V5.0) don't condition the discount on prediction confidence; existing
mangrove/SOC carbon-stock models (Radabaugh, Sanderman, Simard) report
point estimates with no calibrated uncertainty; conformal methods
(GeoConformal, González-Abad & Baño-Medina, Singh et al. 2024) haven't
been paired with a physics-informed predictor or a buffer-pool pricing
mechanism. Built only from already-verified citations — nothing new added.
Now in both `paper/literature_notes.md` (dedicated Phase 4 section) and
`main.tex`'s Introduction.

## Phase 5 — business framing

Abstract now opens with the business problem (flat buffers price a
well- and poorly-characterized site identically) before the technical
method. Introduction leads with the same framing. Discussion's closing
paragraph and the Pricing PoC section now explicitly contrast the
mechanism against the flat-buffer status quo (concretely: what a min-width
vs. max-width node would pay under each regime) — stated as a plausible
implication, not a validated outcome, with the Limitations section's
"pricing mechanism is illustrative" item unchanged in substance.

## Phase 6 — writing audit

- Results section now opens with an explicit "measurements only, ...
  interpretation deferred to Section V" sentence; interpretive language
  (why the physics term's advantage didn't replicate, why localized CQR
  underperformed) moved into a new, dedicated Discussion section.
- Every Method/Results claim maps to a defined method and vice versa
  (re-checked while merging Table I+II and trimming for length).
- Method/Results passed a tense-consistency pass to simple past tense for
  narrated actions and observations (figure/table-reference verbs, e.g.
  "Fig. 2 plots...", kept in the standard present-tense convention used
  even in past-tense-narrated papers).
- Abstract: Background (business problem) → Method → Results (real
  numbers) → Conclusion, zero citations, physics-informed result stated
  with "prior-generalization" wording throughout, never "physical
  validation."
- Limitations: each item stated, then contextualized (why it doesn't
  invalidate the core contribution), not a bare list.
- **Every number in the final manuscript re-checked programmatically
  against its source CSV** (not against this log's own rounded entries) —
  exact match, logged in `NUMBER_SOURCE_LOG.md`'s new Phase 6 section.

## Final deliverable

`paper/main.pdf`: **5 pages, 0 LaTeX warnings/errors** on a clean
`rm *.aux *.bbl *.blg *.log` → `pdflatex` → `bibtex` → `pdflatex` ×2
rebuild. Getting from the initial 6-7-page draft to 5 pages took real
content cuts (not font/margin tricks) plus two small, disclosed,
standard camera-ready conventions: compact float/caption spacing and a
`\penalty0` break inside one long bibliography URL (both noted inline in
`main.tex`'s preamble/`references.bib`).

## What's still open (not attempted this sprint, flagged not hidden)

- Density-weighted CQR's bandwidth was a first-pass, data-driven-but-naive
  choice (median nearest-neighbor distance) that under-covered; a properly
  tuned bandwidth might change that comparison — not attempted, time-boxed
  per the instruction's own "if time allows" framing.
- Deep ensembles / MC dropout remain future-work ablations, as in Day 1-3.
- No second region or second Sentinel-2 date — both already-flagged,
  unchanged limitations, not newly discovered this sprint.

## Addendum (same day): final reframe

Requested after the above was reported — supersedes prior framing
instructions, not a new phase. Full trace: `NUMBER_SOURCE_LOG.md` #50-56.

- **Reframed the paper's two contributions explicitly**: (1) a methodology
  lesson (single-seed/ad hoc evaluation overstated the physics-informed
  advantage; pre-registered multi-seed/test-separated evaluation found a
  smaller, real one — 0.755 vs. 0.682/0.686), and (2) a genuine cautionary
  result (localized/Mondrian CQR looked competitive on validation-internal
  checks but underperformed global on the untouched test set — 83.4%/92.4%
  vs. global's 91.1%/96.4%). Added to Abstract, Introduction (explicit
  "both a naive single-run estimate and a...calibration refinement can
  each mislead" framing), Discussion, and Limitations.
- **Added a Discussion paragraph explaining *why* localized CQR
  underperformed**, citing Mao, Martin & Reich (2024, *JASA*) — spatial
  conformal validity needs an infill-asymptotic calibration-point density
  this AOI's validation set likely lacked — and Hjort et al. (2023,
  arXiv) — an asset-pricing (real-estate AVM) precedent where their own
  Mondrian variant also showed "little to no improvement" over global.
  Both citations verified directly against a primary source before being
  added, not trusted from a search snippet.
- **Made the paper internally consistent with its own recommendation**:
  since Discussion now says global calibration is the defensible
  near-term deployment choice, the Pricing PoC (Section VI) was rebuilt on
  global intervals instead of Mondrian — new numbers throughout Table III
  and the abstract (median width 341.1 t/ha, multiplier range now the
  full clipped 0.500–1.500).
- **Added two more verified citations** on your explicit request: Willard
  et al. (2022/2023, *ACM Computing Surveys*) as the general
  physics-guided-ML motivating survey (Related Work §1), and Roy et al.
  (2024, *Science of The Total Environment*, a 105-study systematic
  review) supplementing Radabaugh et al. on the belowground-carbon/
  valuation gap (Related Work §2). Per your instruction, did **not**
  broaden the paper's own gap statement to match that review's larger
  multi-region MRV agenda — added one narrow-scope sentence to the
  Introduction instead, explicitly distinguishing this paper's tested
  gap (single-region calibration-strategy validity) from that broader one.
- **Filled the affiliation placeholder** with the real institution you
  provided (Nahla Nabil Skaik, Arab Open University, Bahrain) — the
  `% TODO(author)` open since Day 3 is now resolved.
- Recompiled clean: **5 pages, 0 warnings/errors**, verified from a
  `rm *.aux *.bbl *.blg *.log` rebuild.
