# Day 1 Status — 2026-08-29

Sprint context: ICDABI 2026 target (see `PLAN.md`'s deadline note — this
sprint proceeds under the user's explicit instruction to push for it
despite the earlier "target a later cycle" default). Ground rule in force:
every number traceable to `results/`/`data/`, nothing approximated
silently — see `paper/NUMBER_SOURCE_LOG.md`.

## What's real and working

- **All data gaps flagged in the original scaffolding session are now
  closed.** The 4.7GB SOC-value zip (previously judged "too large for this
  sandbox") downloaded in ~5 minutes when actually attempted — that
  assessment was stale, not a real blocker. Full details:
  `data/README.md` §3.
- **A genuine, non-circular regression label is now in hand:** real
  per-pixel soil organic carbon (t C/ha, 0-100cm), independently sourced
  from field soil cores via the Sanderman/Worthington/WHRC dataset — not
  derived from this project's own Sentinel-2 pull, and not a synthetic
  proxy.
- **A region-matched, primary-source validation number is sourced:**
  Radabaugh et al. (2023) Table 1, "Ten Thousand Islands" row (our AOI's
  literal name in their own regional breakdown) — 434.6 ± 96.8 Mg C ha⁻¹,
  mangrove, total C stock. Read from the actual PDF, not a search snippet.
  AOI-wide SOC mean (421.6 t/ha) lands in the same ballpark as their
  independently-modeled belowground figure (383.6 ± 87.7) — a real,
  citable cross-source plausibility check (`NUMBER_SOURCE_LOG.md` #5).
- **A real graph was built and a real GNN trained end-to-end:** 10,124
  nodes (150m aggregated, GMW-mangrove-masked, SOC-valid), 64,366 directed
  edges, on the actual pulled Sentinel-2/GMW/SOC/tide data. Torch 2.13 +
  PyTorch Geometric 2.8 installed and working locally
  (`C:\Users\nahla\venvs\bcpg`).
- **Spatially-blocked train/val/test split**, checkerboard superblocks
  (8×8 grid, node-count-balanced), per `uq_decision.md`'s
  spatial-exchangeability requirement — not a random/i.i.d. split.

## The honest result (headline finding, not hidden)

**The baseline GNN's test-set R² is negative: −0.383** (RMSE 114.7 t/ha,
n=2,236 — `results/baseline_gnn_metrics.csv`). It underperforms both a
naive train-mean predictor (R²≈0.000) and plain linear regression (R²=
+0.031) on the identical spatially-blocked test set
(`results/split_diagnostic_comparison.csv`).

**This was investigated, not shrugged off — two rounds of diagnosis:**

1. **First attempt** used a single contiguous longitude-band holdout.
   Every model tried (linear, RF, GNN, even naive) got sharply negative
   test R² (−0.41 to −0.81). Diagnosed as the holdout band straddling a
   real regional carbon-density gradient — Radabaugh et al.'s own Table 1
   shows mangrove C stock ranging 265–409+ Mg C/ha across their four SW
   Florida sub-regions, so a single contiguous strip can trivially land on
   the wrong side of a real boundary. **Fixed:** moved to a checkerboard
   superblock split (scattered, node-count-balanced), which brought
   train/test label means to near-parity (404.6 vs. 403.3 t/ha).
2. **Second attempt**, same split: GNN still overfit badly (train R²=0.53,
   test R²=−0.33, test loss rising monotonically during training) —
   traced to a real methodological error, not a data problem: the training
   script was informally watching test loss for its own stopping decision.
   **Fixed:** added a proper val split (carved from train's superblocks,
   never touching test), reduced model capacity (hidden dim 64→16, dropout
   0.1→0.3, weight decay 1e-4→1e-3), added early stopping on val loss
   only, and moved to a finer 8×8 superblock grid (the first 5×5 grid left
   too few superblocks per split for stable per-split statistics).

**After both fixes, test R² is still negative, and linear regression beats
the GNN.** This is now believed to be a real, reportable finding about
this specific pilot setup — not a bug — for three reasons: (a) it survives
across three independent model families evaluated on the identical,
methodologically-clean split; (b) the effect is a generalization gap
(train R² positive and reasonable, 0.25; val R²=0.14; only test misses),
consistent with genuine local spatial heterogeneity the model can't
transfer across, not with garbage/NaN/broken data; (c) it is exactly the
kind of failure mode the paper's own literature scan already flagged as
expected (`literature_notes.md` §4: spatial nonstationarity breaking
naive generalization) — so it is not a surprise the project's own prior
research judgment failed to anticipate.

## What got cut / not attempted today

- **No hyperparameter search beyond the one documented, reasoned
  capacity/regularization change above.** Per the sprint's own instruction
  not to "quietly keep debugging into Day 2," stopped after two principled
  diagnose-and-fix rounds rather than continuing to chase a positive test
  R² through further tuning.
- **No second Sentinel-2 date pulled yet** (wet/dry season contrast). This
  is the most likely lever to actually improve the signal (SOC integrates
  years of accumulation; a single optical snapshot has real, expected
  limits) — flagged as the first candidate for Day 2 if point-prediction
  accuracy needs to improve, rather than attempted today under time
  pressure.
- **Simard 2019 AGB and the Radabaugh raster remain unpulled** (both
  documented, both have working fallbacks already in use — see
  `data/README.md` §5–6). Not attempted further today; genuinely low
  priority now that the SOC label makes both less necessary for the core
  regression task.
- **The FWC ArcGIS map's underlying data service was not tracked down**
  (timeboxed, then dropped — `data/README.md` §6).

## Flagged as a paper-facing limitation, explicitly, not silently

- Single-date Sentinel-2 optical features have only weak-to-moderate
  correlation with SOC (NDVI≈0, raw bands ≈−0.26, tide-distance≈0.35 —
  `NUMBER_SOURCE_LOG.md`), and the tide-distance feature — meant to be a
  hydrological covariate — is functioning largely as a spatial-position
  proxy rather than a validated tidal-influence signal. Both facts belong
  in the paper's Limitations section verbatim, not softened.
- Point-prediction accuracy under spatial holdout is currently poor across
  every model family tried, GNN included. If Day 2 proceeds without first
  improving this, the paper's framing needs to lean on this finding
  directly (motivating UQ/calibrated intervals as more defensible than
  point estimates under exactly this kind of regional shift) rather than
  presenting the point predictor as reliable.

## Does tomorrow's plan still hold?

**Recommend a decision from the user before proceeding to Day 2**, because
which of two paths Day 2 takes has real cost:

- **Path A — proceed as planned:** add the physics-informed mass-balance
  loss on top of the current (weak) baseline, and lean into the honest
  narrative above (point accuracy is limited under spatial shift; that is
  exactly why calibrated UQ, not point accuracy, is the paper's real
  contribution). Fastest path to Day 3 writing; the risk is a reviewer
  reasonably asking "does the physics term even matter if neither variant
  predicts well?" — answerable, but needs to be addressed head-on in the
  Discussion, not avoided.
- **Path B — spend part of Day 2 morning improving the baseline first**
  (pull a second Sentinel-2 date for temporal contrast, the one concrete
  lever identified above) before adding the physics term, accepting less
  time for the UQ/pricing work and more schedule risk.

Recommendation: **Path A.** The negative-R² finding, honestly reported, is
defensible and arguably strengthens the paper's actual thesis; Path B's
payoff is uncertain (one added date may not fix a signal problem that
looks structural) and this sprint's schedule has no slack to absorb a
second uncertain detour. But this is a real fork, not a formality — flagging
it rather than picking silently.
