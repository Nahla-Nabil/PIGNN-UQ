# Extension: A Genuine dC/dt Physics Proxy from Two Sentinel-2 Dates

**Status: post-submission exploratory work for a future journal extension.
Not merged into `main.tex`. Does not change Table I/II or any number in
the submitted conference paper. Does not touch the submitted paper's
train/val/test partition (`data/processed/graph_nodes.csv`,
`split` column) -- that file is read once, read-only, in Step 3 below.**

Written 2026-08-30, branch `extension-temporal-dcdt`. Every number below
traces to `paper/NUMBER_SOURCE_LOG.md` entries #57-67 and the exact
results CSV/script named there -- same audit standard as the rest of this
project, continuing its numbering rather than starting a new log.

---

## 1. Motivation

The submitted paper's Limitations section states plainly:

> "Single-date imagery constrains the physics term to a cross-sectional
> relationship, not an actual $dC/dt$ -- not invalidating the
> prior-generalization result, defined relative to that same proxy."

The submitted mass-balance loss term was always meant to approximate
$dC/dt = \text{Accretion} - \text{Export} - \text{Decomposition}$, but with
one Sentinel-2 date on disk, "Accretion" could only be a single-date NDVI
*level* (vegetation vigor at one instant), not a *rate of change*. This
extension pulls a second, real Sentinel-2 date and replaces that one
proxy input with a genuine two-date NDVI change rate -- the actual
physical quantity the term was always supposed to approximate.

## 2. Step 1: second Sentinel-2 date

Via the same Microsoft Planetary Computer STAC access the submitted paper
used (`src/data_pipeline/fetch_sentinel2.py`, keyless, unmodified):

| | Scene 1 (submitted paper) | Scene 2 (this extension, NEW) |
|---|---|---|
| Scene ID | `S2B_MSIL2A_20260206T160339_R097_T17RMJ_20260206T213114` | `S2B_MSIL2A_20260815T155819_R097_T17RMJ_20260815T195532` |
| Date | 2026-02-06 (dry season) | 2026-08-15 (wet season) |
| Tile | T17RMJ | T17RMJ (same tile) |
| Cloud cover | per original STAC search, `--max-cloud 15` | 20.07% (scene-wide) |

**Separation: 190 days (~6.2 months).** This is at the low end of the
requested 6-12 month window, disclosed plainly: this session's live STAC
access only has scenes up to the current session date (2026-08-30), so no
scene closer to a full year apart exists yet to pull. A search restricted
to <20% cloud over 2026-07-15/2026-08-30 returned zero scenes (Florida's
wet season is persistently cloudy region-wide); widening to <60% cloud
over 2026-06-01/2026-08-30 returned 22 candidates, and the scene above --
same tile as the original, lowest cloud cover, closest to "today" (so
maximizing the achievable separation) -- was selected. 190 days still
spans SW Florida's real dry-to-wet season transition, and the resulting
NDVI-delta distribution (below) confirms a real, non-degenerate signal,
not noise.

B03 (Green) was also pulled for scene 2 (`scripts/fetch_ext_second_date_
green.py`, reusing `fetch_green_band()` unchanged), so all four
normalized-difference/SWIR quantities the submitted paper's Phase 1C
feature ablation used (NDVI, NDWI, NDMI, B11) are available at both dates.

## 3. Step 2: the dC/dt proxy

`src/data_pipeline/build_graph_temporal.py` builds a **separate** graph
file (`data/processed/graph_nodes_ext_temporal.csv` /
`graph_edges_ext_temporal.npy` -- never overwriting the submitted paper's
files) with identical node geometry (same 5x5-block aggregation, same
GMW-mangrove/valid-SOC masking, same reference grid) as `build_graph.py`,
plus:

- Date-2 bands and indices (`ndvi_t2`, `ndwi_t2`, `ndmi_t2`, `b11_t2`, ...).
- Genuine change features: `d_ndvi = ndvi_t2 - ndvi_t1`, `d_ndwi`, `d_ndmi`,
  `d_b11`, and `ndvi_rate_per_day = d_ndvi / 190`.

**All 10,124 nodes from the submitted graph are retained** with both
dates' bands valid -- zero nodes dropped for date-2 cloud/nodata within
the mangrove-masked AOI. Edge count is unchanged (64,366 directed,
identical adjacency -- same node set). `d_ndvi` ranges from -0.403 to
+0.273, mean **+0.025**, std 0.066 -- a positive mean shift (dry-season
February to wet-season August green-up) is the ecologically expected
direction, confirming this is a real signal, not an artifact.

The new proxy (`src/models/physics_terms_temporal.py`):

$$\text{proxy}_{\text{dC/dt}} = 0.5 \cdot \text{scaled}(\text{ndvi\_rate\_per\_day}) + 0.5 \cdot \text{scaled}(\text{dist\_to\_tide\_km})$$

Same functional form as the submitted paper's proxy (equal-weighted,
train-only min-max scaling, no fitted coefficients -- stays a physical
*prior*, not a second regression model). **Only the accretion input
changed** (a rate instead of a level); the export input
(`dist_to_tide_km`) is untouched. That term's own, separate limitation --
it is a positional proxy, not a real per-node tidal-energy/inundation
layer -- is **not** addressed by this extension and remains open. This
keeps the accretion-term swap a genuine single-variable change relative
to the submitted paper's proxy.

## 4. Step 3: fresh, non-overlapping test partition

The submitted paper's operative test split (`data/processed/graph_nodes.
csv`, `split=="test"`) occupies exactly 10 of the 20 occupied coarse
superblocks (the 8x8-grid unit used throughout this project) in this AOI:
`{4, 5, 7, 12, 17, 18, 26, 33, 41, 57}`.

`build_graph_temporal.py` reads that column **read-only** (the only place
in this entire extension that touches the submitted paper's graph file)
and excludes all 10 of those coarse superblocks wholesale from the new
test/val candidate pool -- any node in one of them is forced to
`split_ext=="train"`, never eligible for the new test *or* the new val.
From the remaining pool, the same cumulative-node-fraction greedy
selection technique already established in `build_graph.py`'s own
Phase-0 fix (just a fresh seed, `EXT_SPLIT_SEED=20260830`, and a finer
20x20 regrid restricted to the pool) draws a fresh ~20% test / ~15% val.

Result: **train=6,429, val=1,541, test=2,154** (of 10,124). Two
mechanical assertions in the build script confirm zero new-test/new-val
node falls in an excluded superblock. A sanity check also confirmed the
excluded superblocks have zero overlap with the legacy Day 1-2 test/val
superblocks -- independent evidence that the submitted paper's own
Phase-0 fix already left its test region cleanly separated from
everything else ever computed on this AOI.

## 5. Step 4: same rigor protocol, no shortcuts

Mirrors the submitted paper's Phase 1A (multi-seed) -> Phase 1B (lambda
sweep) -> Phase 2 (one test evaluation) sequence exactly: same 5 seeds
{1..5}, same lambda grid {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0}, same
mechanical selection rule (candidates = lambda within 0.05 mean-val-R² of
the best; select the candidate with the highest mean val prior-
generalization correlation), stated before running, applied mechanically
after.

Because the loss target's distribution changed meaningfully (a rate,
`ndvi_rate_per_day`, in place of a level, `ndvi`), the lambda sweep was
re-run rather than reusing the submitted paper's selected $\lambda=0.1$.
**To isolate the proxy-formula effect from the split effect**, a second,
identical sweep was also run with the submitted paper's *original*
steady-state proxy formula, unmodified, retargeted at the same new split
-- a same-split, same-seed, same-architecture **control**. Without this
control, "does dC/dt outperform steady-state" would be confounded by the
split also having changed.

| | dC/dt proxy | Steady-state proxy (control, same new split) |
|---|---|---|
| Best mean val $R^2$ | 0.319 ($\lambda=1.0$) | 0.269 ($\lambda=0.3$) |
| Trend across $\lambda$ | rises monotonically 0.01$\to$1.0, small dip at 2.0 | rises to 0.3, then *declines* for $\lambda\geq0.5$ |
| Selected $\lambda$ (mechanical rule) | **2.0** | **1.0** |
| Mean val $R^2$ at selected $\lambda$ | 0.312 | 0.247 |
| Mean val prior-gen.\ corr.\ at selected $\lambda$ | 0.960 | 0.922 |

At every shared $\lambda$ value, the dC/dt sweep's mean val $R^2$ exceeds
the steady-state control's -- an early, val-only signal in the same
direction the test result (below) confirms.

## 6. Step 5: the test-set result

**First evaluated once** (seed=42, the project's standard single-seed
convention), on the new held-out test partition (n=2,154), never read by
any script before that one:

| Model | Test $R^2$ | Test RMSE (t/ha) | Test prior-gen.\ corr. |
|---|---|---|---|
| Linear regression | 0.104 | 102.5 | 0.665 |
| Baseline GNN (no physics) | 0.291 | 91.2 | 0.787 |
| Physics GNN, steady-state proxy (control, $\lambda=1.0$) | 0.214 | 96.0 | 0.933 |
| Physics GNN, real dC/dt ($\lambda=2.0$) | 0.320 | 89.3 | 0.952 |

At this single seed, dC/dt looked like a clean win over every other
model. **Before drawing that conclusion, the next section re-runs all
four configurations across the same 5 seeds used everywhere else in this
project, on test -- and the picture changes.**

## 6b. Multi-seed test characterization (does dC/dt really win, or is the new split just easier?)

A single seed cannot distinguish two very different explanations for why
this extension's test $R^2$s (roughly 0.1-0.3) look so much better than
the submitted paper's (all negative, entry #44): **(a)** the real dC/dt
proxy specifically improves the physics-informed model, or **(b)** the
new split's superblocks are simply easier/more homogeneous, in which case
even a model with *zero* physics term would show a similar jump.
`scripts/run_ext_phase2b_multiseed_test.py` re-evaluates the same four
already-fixed configurations (nothing tuned or selected here -- every
$\lambda$/feature choice was already fixed by Section 5's val-only
sweeps) across the same 5 seeds $\{1,...,5\}$ used throughout this
project's stability checks, on TEST.

**This is a deliberate, disclosed departure from this project's
"evaluate test exactly once" norm**, done specifically because the user
asked for it after seeing the single-seed table above -- not a second
round of test-informed tuning (see `paper/NUMBER_SOURCE_LOG.md`, Step 4b,
for the full disclosure of why this is still defensible: no downstream
choice depends on any number this produced).

| Model | Test $R^2$ (mean $\pm$ std, 5 seeds) | Test RMSE (t/ha) | Test prior-gen.\ corr. |
|---|---|---|---|
| Linear regression | 0.1044 (deterministic) | 102.46 | 0.6654 (deterministic) |
| Baseline GNN (no physics) | **0.2886 $\pm$ 0.0078** | 91.32 $\pm$ 0.50 | 0.7975 $\pm$ 0.0305 |
| Physics GNN, steady-state control ($\lambda=1.0$) | 0.2294 $\pm$ 0.0167 | 95.04 $\pm$ 1.03 | 0.9242 $\pm$ 0.0141 |
| Physics GNN, real dC/dt ($\lambda=2.0$) | 0.2797 $\pm$ 0.0274 | 91.87 $\pm$ 1.75 | **0.9585 $\pm$ 0.0107** |

### What actually holds up under multi-seed averaging

**1. dC/dt vs.\ the no-physics baseline on raw test $R^2$: does NOT hold
up.** At seed 42 alone, dC/dt (0.320) beat the baseline (0.291). Across 5
seeds, the ranking flips: the baseline's mean (0.2886) is now marginally
*above* dC/dt's mean (0.2797) -- well within dC/dt's own standard
deviation (0.027, more than 3x the baseline's 0.008). Per-seed, the
baseline beats dC/dt on 3 of 5 seeds. **This specific comparison is a
wash, not a win, once seed variance is accounted for.**

**2. dC/dt vs.\ the steady-state-proxy control (the genuine
single-variable comparison -- same split, same seed, same architecture,
only the proxy formula differs): DOES hold up.** Per-seed differences
(dC/dt $R^2$ minus control $R^2$): $+0.0575, +0.0687, +0.0063, +0.0350,
+0.0840$ -- **positive at all 5 of 5 seeds**, mean $+0.0503$. Smaller than
the single-seed(42) estimate ($+0.107$) but consistent in direction every
time. Prior-generalization correlation shows the same pattern more
strongly still: dC/dt beats the control at all 5/5 seeds, with
non-overlapping mean$\pm$std ranges ($0.9585\pm0.0107$ vs.\
$0.9242\pm0.0141$).

**3. The split, not the proxy, explains most of the headline-looking
improvement.** The new split's own no-physics baseline GNN (5-seed mean
test $R^2=0.2886$) minus the submitted paper's no-physics baseline GNN
(single seed=42, test $R^2=-0.383$, entry #44 -- read-only, not re-run)
$= +0.672$. A model with **zero physics term** accounts for almost the
entire "jump" from negative to positive test $R^2$ between the submitted
paper's test split and this extension's new one. The proxy-specific
effect isolated in point 2 above ($+0.05$ mean $R^2$) is real and
consistent, but an order of magnitude smaller than the split effect.
(Caveat on this comparison: the submitted-side number is single-seed --
Phase 2 in the submitted paper never ran multi-seed on test, by design --
while the new-side number is a 5-seed mean; the submitted paper's own
Phase 1A found baseline-GNN val-$R^2$ seed-to-seed std to be small,
0.019, entry #34, so a 0.67-point gap is not plausibly seed noise, but
the two sides are not a perfectly matched comparison and this is reported
as a diagnostic, not a formal statistical test. Nothing about the
submitted paper's own test partition was re-touched to produce this
number.)

### Does the real dC/dt-based physics term outperform the original steady-state-proxy version? (revised, honest answer)

**Mixed, and smaller than it first looked.** On the one genuinely
single-variable comparison available -- dC/dt vs.\ the steady-state
control, same split/seed/architecture -- yes, consistently across all 5
seeds, on both test $R^2$ ($+0.050$ mean) and prior-generalization
correlation ($+0.034$ mean, non-overlapping ranges). That comparison is
real and worth reporting. But **the dC/dt physics GNN does not reliably
beat a model with no physics term at all** on raw test $R^2$ once seed
variance is accounted for -- and the large majority of why this
extension's numbers look so much better than the submitted paper's is the
new split, not the new proxy. Consistent with the submitted paper's
established framing (Section III-F/V, restated here so this document
reads correctly on its own): **prior-generalization correlation measures
whether a model's predictions stay consistent with the physics loss's own
training target under spatial holdout -- transfer of an imposed prior,
not independent physical validation.** A genuine two-date NDVI *rate* is
a strictly less simplified accretion proxy than a single-date NDVI
*level*, but it remains a proxy, not a measurement, and this extension's
own baseline-GNN result shows that "improves on the submitted paper's
test $R^2$" and "the dC/dt proxy specifically helps" are two different
claims that must not be conflated.

### Honest caveats (reported, not smoothed over)

- **Both physics-GNN variants have negative train $R^2$** at seed=42
  (steady-state control: $-0.103$; dC/dt: $-0.150$) despite positive,
  competitive test $R^2$. This is a known pattern already present in the
  submitted paper's own physics-GNN variants (the physics loss pulls
  predictions away from a pure data fit on train), not a new anomaly
  introduced by this extension.
- **The steady-state control's val-stage ranking did not hold on test**
  at seed=42 (val: 0.242 vs.\ baseline's 0.230; test: 0.214 vs.\
  baseline's 0.291) -- reported plainly, exactly as the submitted paper's
  own val/test mismatches were (e.g.\ Phase 1D's localized-CQR result),
  not reframed to look more favorable.
- **$\lambda=2.0$ (the dC/dt track's selected value) sits at the UPPER
  EDGE of the pre-registered sweep grid $\{0.01, 0.05, 0.1, 0.3, 0.5,
  1.0, 2.0\}$.** The true optimum may lie above 2.0 -- this sweep cannot
  rule that out, since larger $\lambda$ was never tested. Per this
  project's own pre-registration discipline, extending the grid upward
  now, having already seen that 2.0 was selected, would not be a
  legitimate mechanical application of the existing rule; it would
  require a genuinely NEW pre-registered grid, stated before running, not
  a post-hoc extension of this one. **Not done here, and should not be
  done without a fresh pre-registration if this work continues.**
- **This is one fresh split, one new date pair, one AOI.** The
  single-variable proxy effect (point 2 above) is real within this
  experiment, but it is not yet a claim that a real dC/dt proxy is a
  validated replacement for the steady-state one in general -- that would
  need replication across more date pairs and, ideally, a third, truly
  independent test partition.
- **The export-proxy limitation is unchanged and still open** (see
  Section 3 above) -- this extension improves the accretion half of the
  mass-balance proxy only.

## 7. What this would take to merge into the paper

Not done here, per the task's explicit scope: no edits to `main.tex`, no
new Table/Figure in the submitted structure, no change to the submitted
paper's headline numbers. **Also not done, pending review of Section 6b:
this document's central claim was revised once already after multi-seed
testing tempered a single-seed result -- any merge should wait until the
$\lambda$-boundary caveat is resolved (a fresh pre-registered sweep
extending past 2.0) and, ideally, a second date pair replicates the
single-variable dC/dt-vs-steady-state effect, before treating it as
established rather than a first, honestly-reported signal.** If a
journal-extension revision is pursued later, the natural home for this
material is a new subsection in Method (the dC/dt proxy formula) plus a
new row/column in the Results table -- reporting the modest, single-
variable finding (dC/dt beats the steady-state control by a small but
consistent margin; it does not reliably beat a no-physics baseline) --
and an update to the Limitations paragraph quoted in Section 1 above (the
single-date-imagery caveat would need to be revised, not removed -- 190
days is still one date *pair*, not a time series).
