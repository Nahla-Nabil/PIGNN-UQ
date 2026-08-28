# UQ Method Decision

Decision date: 2026-08-29. See [`literature_notes.md`](literature_notes.md#4-conformal-prediction--deep-ensembles-for-environmental--geospatial-uq)
§4 for the underlying citations.

## The three candidates

| | MC Dropout | Deep Ensembles | Conformal Prediction (CQR) |
|---|---|---|---|
| **Mechanism** | Keep dropout active at inference, sample T forward passes | Train M independently-initialized models, use spread of predictions | Wrap any point/quantile predictor; calibrate interval width on a held-out set so empirical coverage matches nominal |
| **Coverage guarantee** | None (heuristic variational approximation) | None (empirical, no finite-sample guarantee) | **Finite-sample, distribution-free marginal coverage**, under exchangeability (Romano et al. 2019) |
| **Training cost** | 1x (single model) | Mx (M full training runs, typically 5–10) | 1x point/quantile model + a cheap calibration pass |
| **Inference cost** | Tx forward passes (T~20–100) | Mx forward passes | 1x forward pass (+ negligible calibration lookup) |
| **Calibration quality in practice** | Known to be poor, esp. under distribution/covariate shift (Gal & Ghahramani 2016; widely reported since) | Best empirically-calibrated of the two heuristic methods; "gold standard" among non-conformal approaches | Calibration is *guaranteed by construction* on the calibration set, conditional only on the exchangeability assumption holding |
| **Assumption that can break it** | Dropout rate / architecture choice determines uncertainty quality somewhat arbitrarily | None strong, but still no formal guarantee | **Exchangeability** — violated by spatial autocorrelation unless calibration is spatially blocked |
| **Interpretability for a pricing rule** | Ad hoc (variance across T samples) | Ad hoc (variance across M models) | Direct: prediction interval width *is* a defensible, auditable per-credit uncertainty number |

## Why this matters for THIS paper specifically

The end use of the UQ output is not a diagnostic plot — it is a **discount
factor applied to money** (buffer discount / premium on a carbon credit
price). That downstream use raises the bar on defensibility:

- A reviewer (or, more importantly, a carbon-market stakeholder) will ask
  "why should I trust that this specific number should shave 12% off a
  credit's price?" A variance-across-dropout-samples answer is much weaker
  than "this interval has guaranteed ≥90% empirical coverage on held-out
  data."
- Existing buffer-pool tools (VM0033 / Plan Vivo, see
  `literature_notes.md` §3) already operate on **explicit, auditable
  percentages**. A coverage-guaranteed interval is the most direct
  drop-in replacement for that kind of rule; an ensemble-variance heuristic
  is a harder sell as "better" than a fixed audited percentage.
- The user has prior work applying conformal prediction to landslide
  susceptibility (geospatial regression with the same exchangeability
  concerns) — directly transferable experience and code patterns, and a
  natural self-citation for the paper's related work.

## Recommendation: **Conformalized Quantile Regression (CQR) as the primary method**

- Train a quantile-regression head on the physics-informed GNN (predict
  e.g. the 5th/50th/95th percentile of carbon stock per node directly,
  via pinball loss) instead of / alongside the point estimate.
- Calibrate with **split conformal, using a spatially-blocked calibration
  set** (hold out entire contiguous sub-regions of the AOI, not randomly
  scattered nodes) — required per the spatial-exchangeability caveat in
  `literature_notes.md` §4, not optional.
- Report empirical coverage on the held-out spatial block as the headline
  UQ validity check (this becomes one of the paper's figures — see
  `PLAN.md`).
- Feed interval width (or a normalized version of it) directly into the
  pricing module as the risk signal.

## What to do with MC Dropout and Deep Ensembles

Not discarded — positioned as **ablation / robustness check, time
permitting**:

- **MC Dropout**: near-zero extra engineering cost (same trained model, just
  toggle dropout at inference). Worth including as a one-line "cheap
  baseline that under-covers" comparison point in the results table even if
  nothing else is added — it directly substantiates the calibration claim
  above with our own numbers instead of only citing it.
- **Deep Ensembles**: only if time and compute allow training 5 full model
  replicates (see `PLAN.md` risk section — this is the first thing cut
  under schedule pressure). If included, it's the "expensive but well-
  calibrated" comparison point that shows CQR gets similar/better coverage
  at a fraction of the training cost.

## Explicit tradeoff for the paper to preempt (reviewers will ask)

**"Why not deep ensembles, which are known to be well-calibrated in
practice?"** Answer to state directly in the paper: (1) CQR gives a formal
finite-sample guarantee that ensembles do not, which matters when the
output prices a financial instrument; (2) CQR costs 1 training run vs. M,
which matters under this project's compute/time budget; (3) if the
ablation is completed, we show empirically that CQR's coverage is
competitive with the ensemble's at a fraction of the cost — turning a
potential weakness ("why the cheaper method") into a result.

**"Doesn't conformal prediction assume exchangeability that spatial data
violates?"** Answer to state directly in the paper: yes — this is exactly
why calibration uses spatially-blocked splits rather than random splits,
and we report on that basis rather than claiming an unconditional
guarantee we can't back.
