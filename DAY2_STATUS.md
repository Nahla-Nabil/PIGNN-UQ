# Day 2 Status — 2026-08-29 (continued)

Path A confirmed by user: no Sentinel-2 temporal detour, proceed with
physics term on the existing baseline. All numbers below trace to
`paper/NUMBER_SOURCE_LOG.md` (Day 2 section) and `results/`.

## The three-way table

| Model | Train R² | Test R² | Test RMSE | Test MAE | Neg. rate | Edge incoherence | Physical consistency corr. |
|---|---|---|---|---|---|---|---|
| Linear regression | 0.040 | **+0.031** | 96.0 | 72.9 | 0.0 | 0.00031 | −0.060 |
| Baseline GNN (no physics) | 0.246 | −0.383 | 114.7 | 89.7 | 0.0 | 0.0 | +0.048 |
| **Physics-informed GNN** | 0.160 | **−0.219** | 107.7 | 85.1 | 0.0 | 0.0 | **+0.645** |

(`results/day2_model_comparison.csv` — n=2,236 test nodes for every row,
identical spatially-blocked split, seed=42 throughout.)

## What the physics term actually did

**On R² alone: helped, but not enough.** Adding the physics loss moved
test R² from −0.383 (no physics) to −0.219 — a real, +0.165 improvement,
and val R² improved similarly (0.137 → 0.201). But it does not close the
gap to linear regression (+0.031). By the hard-checkpoint rule set for
today, **this triggers the required reframe below** — not softened, not
hidden.

**On physical-consistency correlation: a large, clear effect.** The
physics-informed model's test-set predictions correlate with the
mass-balance proxy at **r=0.645** — versus 0.048 for the no-physics GNN
and −0.060 for linear regression. This isn't a marginal difference; it's
roughly an order of magnitude larger than either non-physics-informed
model, and it holds on the held-out spatial block, not just on data the
physics term was trained on. This is the real, measurable effect the
physics constraint has, even though it's a different kind of effect than
"beats the R² leaderboard."

**On the other two plausibility metrics: no differentiation, reported
honestly.** Negative-prediction rate is 0.0 for all three models (nothing
to fix here — this AOI's label range never got close to zero, so the
non-negativity term never had to do real work). Edge-incoherence rate is
0.0 for both GNN variants and near-zero for linear regression — this
mainly reflects that graph message-passing smooths predictions across
edges *by architectural default*, not something the physics term adds on
top of the plain GNN. Neither of these two metrics is claimed as evidence
for the physics term specifically; only the physical-consistency
correlation is.

## The required reframe (per the Day 2 hard-checkpoint instruction)

**Triggered: physics-GNN test R² (−0.219) is below linear regression's
test R² (+0.031).** Per instruction, this is not softened in the writeup.

**Old framing (was the plan through end of Day 1):** "the physics
constraint improves point-prediction accuracy."

**New framing (what the evidence actually supports, and what the paper
will now claim):** *the physics constraint improves prediction
reliability/plausibility under spatial distribution shift — measured by
consistency with a physically-motivated prior, not by R² — which is
exactly the situation calibrated uncertainty quantification, not point
accuracy, is designed to make defensible.* Same logic as the landslide
paper's finding that a pooled metric can mask what's actually happening:
here, R² alone hides the fact that the physics-informed model is doing
something real and measurably different (staying consistent with a
physical prior under exactly the kind of regional heterogeneity this AOI
has shown throughout both days) that a bare accuracy number doesn't
capture.

This reframe is now load-bearing for the paper's Abstract/Intro/Discussion
and needs to be stated explicitly there too, not just here — flagging for
Day 3 writing.

## CQR calibration (same split, val as calibration set)

| Nominal | Empirical coverage (test) | Mean interval width | Conformal correction Q |
|---|---|---|---|
| 90% | 85.7% | 261.9 t/ha | −3.99 |
| 95% | 90.2% | 310.0 t/ha | −5.31 |

(`results/cqr_coverage.csv`.) Both levels undershoot nominal by ~4.3-4.8
percentage points — reported as-is, not adjusted or excused. This is
consistent with (not a new surprise on top of) the spatial-exchangeability
caveat `uq_decision.md` flagged before any model was trained: calibrating
on one spatial sample (val) and evaluating on another (test) is exactly
the setting where conformal prediction's exchangeability assumption is
most likely to be stressed, and the R² results across all three models
this sprint have consistently shown train/val/test aren't fully
exchangeable in this AOI. Interval widths (262-310 t/ha, vs. a label std
of 103.4) are wide relative to the label's own spread — i.e. the model is
not falsely confident; the miscalibration is an undercoverage gap on top
of already-wide intervals, not narrow-and-wrong intervals.

**Disclosed scope choice:** the quantile regressor is trained with pinball
loss on data alone, without the physics loss term (documented in
`src/models/cqr_uq.py`'s docstring as a stated simplification, not a
silent one). Combining physics-informed training with the UQ head is left
as future work, consistent with `uq_decision.md`'s existing "ensembles/
MC-Dropout as ablations, time-permitting" framing — this is the same
category of extra, not a walked-back promise.

## Does Day 3's plan still hold?

**Yes, proceeding to Day 3 writing as planned**, with two required
additions to the outline (not schedule changes, just content that must
land where PLAN.md already allocated space):

1. The reframed central claim (above) needs to be explicit in the
   Abstract and Discussion, not just implied.
2. The three-way table and the physical-consistency correlation
   (`results/day2_model_comparison.csv`) are now Table 2 / the paper's
   central empirical evidence — the CQR coverage table remains Table 3
   as planned.

No further scope cuts triggered today. The pricing proof-of-concept
(Day 3, Task 1 per the sprint prompt) can proceed using the physics-GNN's
predictions + CQR intervals exactly as originally scoped.
