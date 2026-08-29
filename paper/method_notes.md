# Method Notes: test-set protocol (Phase 0, journal-extension sprint)

**Day 1-2's split was structurally three-way (train/val/test), but was
used, in practice, like a two-way split for decision-making purposes:**
val was correctly reserved for early stopping, but the test split's
metrics were then directly inspected to make model-selection/narrative
decisions -- most explicitly the "hard-checkpoint" comparison in
`DAY2_STATUS.md`, which compares test $R^2$ across linear regression,
baseline GNN, and physics GNN to decide what the paper's central claim
would be. That is a real, if informal, use of test data to make a
decision, on top of its intended role as a single untouched confirmation.
Val's numbers were also cited once in a similar cross-model narrative
point ("val $R^2$ improved similarly"). Continuing to treat either set as
"held out" for the robustness/selection work below would bias the numbers
reported as final.

**Fix, applied before any Phase 1 experiment:** a fresh test block (and a
fresh val block) was carved using ONLY the legacy-train superblocks --
nodes that were only ever touched via gradient steps during training, never
individually inspected to make a decision. Legacy val and legacy test were
folded entirely into the new train pool (training on previously-seen data
is always safe; evaluating on it again is what leaks). Full implementation:
`src/data_pipeline/build_graph.py` (search `PHASE1_SPLIT_SEED`); full
numbers: `paper/NUMBER_SOURCE_LOG.md` entries #30-32.

**The operative protocol from Phase 0 onward, stated plainly for the
paper's Method section:** every experiment in Phase 1 (multi-seed
stability, the pre-registered lambda sweep, the feature ablation, CQR
calibration-method comparison) is decided on **val only**, using rules
fixed before results were seen. The new test split (2,076 nodes) is read
by exactly one script (`scripts/run_phase2_test_eval.py`), exactly once,
after every Phase-1 selection was already locked in. This two-stage
protocol (val-based selection, single held-out test confirmation) is a
methodological strength of the journal-extension version, not merely a
fix -- it is what makes the "we tested X and selected Y by a pre-stated
rule" sentence in the paper defensible to a reviewer, and it is stated as
a strength in the Method section, not only as a correction of Day 1-2.

**What changed as a result, reported plainly, not reframed to match the
old narrative:** under the corrected split, the pre-registered lambda
sweep selected $\lambda=0.1$ (not Day 2's ad hoc 0.3), the physics-informed
GNN's test $R^2$ advantage over the no-physics baseline GNN did NOT
replicate (the baseline GNN has the best test $R^2$ of the three models
under this split/config), and the physics-informed model's
prior-generalization correlation advantage, while still the largest of the
three, is far smaller (test: 0.755 vs. 0.682/0.686) than Day 2's
single-seed finding (0.645 vs. 0.048/-0.060) suggested. See
`paper/NUMBER_SOURCE_LOG.md` entries #33-47 for the full trace. This is
exactly what a proper val/test protocol and a multi-seed robustness check
are supposed to surface, and it is reported as such.

---

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

## Result — Day 2 pilot number (single seed 42, legacy split, λ=0.3 chosen ad hoc)

The physics term improved test R² (−0.383 → −0.219) without closing the
gap to a plain linear baseline (+0.031). It also raised the correlation
between test-node predictions and the mass-balance proxy itself (0.048 →
0.645, both non-physics models near zero). **This correlation is a
prior-generalization check, not independent physical validation:** the
proxy being correlated against is the exact quantity the physics loss
trained the model toward, so a high value shows the constraint was learned
and carried forward to spatially held-out nodes rather than only fit on
training nodes — real and non-trivial, since overfitting to train and
failing to generalize the constraint was a live possibility given
everything else in this AOI generalizes poorly (Day 1-2 findings) — but it
is not evidence that the constraint reflects true carbon dynamics, since
this AOI's single-date, flux-tower-free data cannot supply an independent
measurement to validate against.

## Result — journal-extension number (5-seed mean, corrected split, λ=0.1 pre-registered)

**Superseded by the Phase 0-2 protocol above, not merely repeated with
new numbers.** Under the corrected test-set protocol, a pre-registered
5-value-x-5-seed lambda sweep (`paper/NUMBER_SOURCE_LOG.md` #36-38) selects
$\lambda=0.1$, not Day 2's ad hoc 0.3. At that value, on the single final
test evaluation (#44): baseline GNN test R²=−0.383 is now the BEST of the
three models (physics GNN=−0.430, linear=−0.450) — the physics term's
point-accuracy edge over the no-physics baseline, reported in Day 2, does
**not** replicate under this split/seed-averaged protocol. The
prior-generalization correlation result is directionally the same but far
smaller in magnitude: physics GNN=0.755 vs. baseline GNN=0.682 and
linear=0.686 (a 0.07-0.09 gap, not Day 2's 0.6-point gap). The
multi-seed check (#33-35) shows why: averaged over 5 seeds, the physics
GNN's mean val R² (−0.040) is already below the baseline GNN's (+0.005),
and its prior-gen-corr advantage (0.632 vs. 0.544) is real but modest —
Day 2's dramatic single-seed gap was not a stable phenomenon. **This is
the number and the framing the paper now uses; the Day 2 number above is
kept for audit, not restated as the headline result.** The
prior-generalization framing itself (not independent physical validation)
is unchanged and still applies with equal force to the smaller gap.
