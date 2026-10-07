# Revision Plan: ICDABI 2026 (DATA'26), paper #1571347280

Written 2026-10-07. Status: **accepted with major revisions**. Every
reviewer comment below has an ID (R1.x / R2.x), and each ID has to map to
a concrete change in the paper and a row in the response letter.

Working branch: `revision-icdabi2026`, created from `day5-edas-final-polish`.
The binding conventions still apply: every number goes into
`paper/NUMBER_SOURCE_LOG.md` before it goes into `main.tex`, all selection
happens on validation only, and test is touched at most once more (Phase 3).

---

## 0. Pre-flight findings (checked 2026-10-07, validation data only)

These shape the plan's priorities. They come from a quick check on
validation data, not from test, so the protocol is not affected.

| Finding | Number | Why it matters |
|---|---|---|
| A trivial interval (constant train-mean predictor + split conformal, calibrated on val) has 90% width **274.4 t/ha** and 95% width **338.8 t/ha** | val | Our global CQR is **307.6 / 367.8** on val and **335.6 / 382.6** on test, so on average it is *wider than doing nothing*. R1's "intervals several times the label SD" comment points at this. **This is the most dangerous weakness and is priority #1.** |
| The label product's own per-node uncertainty: median half-band of (upper_std − lower_std)/2 is **≈403 t/ha** across all nodes and **≈413** on val | `graph_nodes.csv` | The label (Sanderman SOC) is itself a model output, and its own per-pixel uncertainty is about 4–5× the between-node SD (87.6 on test). This is a legitimate, citable context for both the low R² and the width critique. It is context, not an excuse. |
| Test label mean 399.4 vs train 427.9 | `graph_nodes.csv` | The train-mean predictor already scores R² ≈ −0.105 on test, so part of the negative R² is mean shift between blocks. Report MAE and a skill score against the train-mean predictor, not only R². |
| A second Sentinel-2 date (2026-08-15) and the `extension-temporal-dcdt` branch already exist | `data/raw/sentinel2/`, `paper/extension_temporal_notes.md` | Gives a cheap answer to R1's "single date" comment (prediction/width stability across dates). |
| The 2-layer GraphSAGE receptive field is 2 hops, but the leakage analysis only excluded 1-hop boundary nodes | `main.tex` Limitations | The leakage fix needs a 2-hop buffer or inductive training (R1.6 / R2.3). |

---

## 1. Reviewer comment → action matrix

| ID | Comment (condensed) | Action | Phase | Tier |
|---|---|---|---|---|
| R1.1 / R2.1 | Negative test R² for all models (worse than the mean) | E1 stronger baselines, E2 a small capacity/feature grid, E4 spatial block CV for a distribution instead of one block; report MAE, skill vs train-mean, Spearman ρ; honest reframe if it stays negative | 2, 5 | A |
| R1.2 / R2.2 | "Prior-generalization" is circular / no independent physical validation | Demote it from a headline result to a *constraint-transfer diagnostic*; E7 independent check against Radabaugh field-plot stocks; drop it from the abstract's claims | 2, 5 | A |
| R1.3 | CQR does not use the physics model; widths several times the label SD | E5 physics-informed quantile model; E6 trivial-interval baseline + adaptivity metrics (width–error ρ, conditional coverage); context from the label-product uncertainty | 2, 5 | A (E6), B (E5) |
| R1.4 / R2.5 | Pricing is ad hoc; arbitrary α; conflates estimation uncertainty with non-permanence | Phase 4 redesign: two separate factors, anchored to registry uncertainty-deduction practice, conservative (no premium above 1.0), sensitivity analysis | 4 | A |
| R1.5 | Single region, single date | E8 second-date stability (cheap); second region is journal scope, stated explicitly | 2, 5 | B (E8), C (region) |
| R1.6 / R2.3 | Split leakage | E3 inductive training on the train-induced subgraph + 2-hop buffered evaluation | 2 | A |
| R1.7 | Abandoned the pre-registered Mondrian method after seeing test results | Add a "Deviations from pre-registration" table; report pricing under **both** Mondrian (pre-registered) and global (post-hoc, flagged) | 4, 5 | A |
| R1.8 | Each component is standard; a protocol is ordinary good practice, not a contribution | Reframe the contribution (Phase 5) around uncertainty-aware issuance for blue carbon + empirical evidence on when spatial UQ is informative; the protocol becomes the *method of evidence*, not the contribution | 5 | A |
| R2.4 | No Random Forest / XGBoost baselines | E1 | 2 | A |
| R3 | Positive; no requested changes | Thank the reviewer; keep the methodology detail they praised | 6 | — |

Tiers: **A** = must do for this revision. **B** = do if the deadline allows.
**C** = journal extension, stated in Limitations/Future work.

---

## Phase 0: Admin and deadlines (Nahla, on EDAS, **today**)

- [ ] **Copyright form**: the email says "within 5 days". Do this first.
- [ ] Confirm the **revised-paper deadline** on EDAS / the ICDABI site
      (PLAN.md recorded camera-ready as Oct 10, 2026; the major-revision
      deadline may differ). Tier B is cut entirely if the window is short.
- [ ] Confirm the **page limit**, and whether an extra paid page is allowed.
      The paper is at 5 pages now; the new tables need about 0.5–0.75 pages.
- [ ] Register at least one author (needed for inclusion in the program).
- [ ] Affiliation/email consistency: EDAS lists `6230202BH@std.aou.org.bh`
      and "Arab Open University - Bahrain", but `main.tex` uses the Gmail
      address. Pick one and make both match.
- [ ] Check whether the revision upload needs a rebuttal (EDAS has a
      "Rebuttal" tab and a "Revised Manuscript (with highlighted changes)"
      slot). Plan for both.

## Phase 1: Pre-registration amendment (before running *any* new experiment)

1. Create `paper/revision_preregistration.md` stating, for every
   experiment E1–E8: candidate set, selection metric, tie/tolerance rule,
   seeds, and what is reported regardless of outcome.
2. Fixed rules (to be written there verbatim):
   - All selection happens on **validation only**, using the same 0.05
     R²-tolerance rule style the paper already uses.
   - Test is read **once**, by one script (`scripts/run_revision_test_eval.py`),
     covering every carried-forward model at once, and the outcome is logged
     whatever it is.
   - The original Table I/II numbers stay in the paper as "original
     submission" for transparency, next to the revision numbers.
3. **Commit that file before any run.** The commit hash and timestamp serve
   as the verifiable pre-registration (cited in the paper / response letter).

## Phase 2: Experiments (validation only)

Python env: `C:\Users\nahla\venvs\bcpg` (torch_geometric and sklearn are
present; install `xgboost` there, not in the repo).

**E1. Tabular baselines (R2.4, R1.1).** RF and XGBoost on the same node
features, plus:
(a) the same models + **spatial-lag features** (neighbor-mean of each feature
over the same 8-connectivity), a fair non-graph spatial baseline;
(b) a **coordinates-only RF**, which tests whether the graph adds anything
beyond location.
Use a small pre-registered hyperparameter grid with 5 seeds.
Output: `results/rev_e1_tabular_val.csv`.

**E2. GNN capacity/feature grid (R1.1).** Hidden dimension {16, 64}, layers
{2, 3}, target {raw, log}, features {current, + all available S2 bands from
the existing raw tif}. Check the band count first; no new downloads.
Keep the grid ≤ 16 configs × 5 seeds. Apply the same selection rule to the
baseline and physics GNN.

**E3. Leakage fix (R1.6, R2.3).** Train on the **train-induced subgraph**
(no edges touching val/test during training); evaluate inductively on the
full graph. Also report a **2-hop buffered** test subset (2 hops = the
GraphSAGE receptive field). This replaces the current 1-hop-only
sensitivity analysis. Becomes the default training regime if it isn't
worse on val by more than the tolerance; otherwise both are reported.

**E4. Spatial block cross-validation (R1.1, R1.5 partial).** Grouped K-fold
(K=5) over superblocks within the train+val pool (test untouched). Report
mean ± SD of R², MAE, and Spearman ρ across folds for every model, which
answers "one block is one draw". Folds are fixed by seed and committed
beforehand.

**E5. Physics-informed quantile model (R1.3).** Pinball loss + λ·physics
term (λ from the existing rule), conformalized the same way. Compare against
the data-only CQR on val: coverage, width, and the E6 metrics. This removes
the "two separate results" seam the Conclusion currently admits to.

**E6. Is the uncertainty informative? (R1.3, the priority finding).**
For every UQ method (global CQR, Mondrian, E5, plus CQR on an XGBoost/
quantile-RF base from E1):
- the **trivial-interval baseline** (constant predictor + conformal) as a
  reference row;
- **adaptivity**: Spearman ρ(width, |residual|), and coverage per width
  tercile and per superblock (conditional coverage);
- **interval score** (Winkler) for a single proper-scoring comparison.

Decision rule (pre-registered): the pricing PoC is justified only if the
carried-forward method has ρ(width, |error|) > 0 with a bootstrap CI
excluding 0 on val. If it fails, the paper says so and the pricing section
becomes explicitly conditional.

**E7. Independent validation (R1.2, R2.2).** Extract the Radabaugh et al.
2023 field-plot carbon stocks (sub-regional means ± SD, in
`data/raw/validation_refs/`) for the sub-region(s) overlapping the AOI.
Compare the model-predicted aggregate (and the Sanderman label aggregate)
against them. This is coarse, but it is the only check in the paper that
uses neither the proxy nor the training label, so it directly answers
"no independent physical validation". Report the overlap honestly, even if
it is only one or two sub-regions.

**E8. Second-date stability (R1.5).** Run the trained models with the
2026-08-15 Sentinel-2 features (labels are static): Spearman ρ of
predictions and of interval widths between dates. A pricing signal that
reorders under a seasonal image change would be a problem worth reporting.
Optional: one sentence pointing to the dC/dt extension as future work.
Do **not** bring its separate split into this paper.

## Phase 3: Single test evaluation

One script, one run, every carried-forward model. Log to
`results/rev_test_*.csv`, then add NUMBER_SOURCE_LOG entries. No
re-runs after reading the output; if a bug is found, document the fix and
the re-run explicitly (same as the Day 4 incident policy).

## Phase 4: Pricing redesign (R1.4, R2.5, R1.7)

1. **Separate the two risks explicitly:**
   `credits issued = Q_conservative(stock) × (1 − buffer_nonpermanence)`.
   - The buffer stays the registry's (Plan Vivo 20% / VM0033 risk tool),
     untouched by our model.
   - Estimation uncertainty enters only through *how much stock is
     credited*. Use the lower CQR bound, or a registry-style **uncertainty
     deduction**.
2. **Anchor to existing practice instead of an arbitrary α.** Verify and
   cite the actual registry/CDM rules on uncertainty deductions and
   conservativeness (e.g. Verra's uncertainty-deduction threshold rules,
   CDM conservativeness-factor guidance). Fetch the primary documents and
   quote the exact rule. Nothing gets cited from memory.
3. **No premium above 1.0.** The current symmetric 0.5–1.5 multiplier is
   ad hoc and invites gaming (an issuer wants narrow intervals). A
   conservative-only discount is defensible to registries.
4. **Sensitivity analysis** for any remaining parameter (a small table or one
   line per value) instead of a single illustrative α.
5. **Report both Mondrian (pre-registered) and global (post-hoc).** Global
   is flagged as a deviation, with its reason. This answers R1.7 directly.
6. Report **portfolio-level** effects (total credited tC under flat vs.
   uncertainty-aware issuance over the test set), not just five nodes. That
   is more meaningful for "business and industry".

## Phase 5: Manuscript rewrite

- **Contribution reframe (R1.8):** "Can calibrated spatial uncertainty
  support conservative, registry-compatible blue-carbon issuance, and when
  is it informative?" The protocol becomes the method of evidence.
  Title change only if EDAS/chairs allow it. Otherwise keep the title and
  shift the abstract/intro.
- **Abstract/Intro:** remove "the contribution is the protocol itself";
  state the new results with numbers from NUMBER_SOURCE_LOG.
- **Related Work:** add 3–5 citations for RF/XGBoost mangrove carbon
  mapping, conformal interval-adaptivity/conditional coverage, and registry
  uncertainty deductions. Verify every one (DOI/URL). Keep ≥ 1/3 of
  references from the last 10 years. No Wikipedia.
- **Methods:** the leakage-safe training regime (E3), spatial CV (E4), the
  physics-quantile model (E5), and a new "Deviations from pre-registration"
  paragraph/table.
- **Results:** Table I extended (RF, XGBoost, spatial-lag, coordinates-only;
  R², MAE, skill vs train-mean, Spearman ρ, CV mean ± SD). Table II extended
  (trivial baseline, Winkler score, width–error ρ). Short E7/E8 results.
- **Discussion:** the label-product uncertainty context (≈400 t/ha per-node
  half-band vs 87.6 between-node SD). Prior-generalization becomes one
  sentence-level diagnostic.
- **Limitations:** update the leakage numbers (now fixed, not just
  disclosed), keep single region, and state what pricing still lacks
  (market validation).
- **Figures:** Fig. 2b adds the trivial baseline; Fig. 3 stays. Fig. 4
  (the physics-loss schematic PREVIEW) only if the page budget allows.
- **IEEE / chair checklist (from the acceptance email):**
  - [ ] IEEE 2-column, **no page numbers**, all fonts embedded (`pdffonts`)
  - [ ] every reference cited; every table/figure cited in text *before* it appears
  - [ ] table captions above and figure captions below, on the same page
  - [ ] self-citations < 3; no Wikipedia; ≥ 1/3 of references from 2016+
  - [ ] spell check + grammar check
  - [ ] similarity < 15% (it was 1% last time; re-check after the rewrite)
  - [ ] title/author/affiliation identical to EDAS

## Phase 6: Response letter and submission package

- `paper/response_to_reviewers.md` (→ PDF): a point-by-point table,
  with one row per ID in §1. Each row gives the comment, what was done,
  where (Section/Table/Eq.), and the new numbers. State plainly where a
  critique is accepted and not fixed (e.g. single region). Thank R3 briefly.
- **Highlighted-changes PDF:** `latexdiff` original `main.tex` (tag
  `submitted-v1`) vs revised. Upload to the "Revised Manuscript (with
  highlighted changes)" slot.
- Clean revised PDF → "Revised Paper" slot.
- Final checks: recompile with 0 warnings, run `pdffonts`, verify the page
  count, run IEEE PDF eXpress if the conference requires it.

---

## Honesty rules for the outcome (decided now, before results)

- If R² is still negative after E1–E4, the paper says so. The
  framing becomes "SOC-product emulation at 150 m from Sentinel-2 is weakly
  learnable here", backed by MAE/skill/ρ and the label-uncertainty context.
  Nothing gets hidden or relabeled.
- If a tabular model (RF/XGBoost) beats both GNNs, report it as a result
  and discuss why (small graph, smooth label). The paper's value then rests
  on the UQ/issuance side.
- If E6 shows the intervals are not adaptive, the pricing section becomes
  a conditional design ("requires ρ > 0, which this AOI's model does not
  yet meet"), not a demonstration.

## Rough effort (to be cut against the real deadline from Phase 0)

| Phase | Effort |
|---|---|
| 0 Admin | Nahla, ~1 h |
| 1 Pre-registration | 1 session |
| 2 Tier A (E1, E2, E3, E6, E7) | 2–3 sessions |
| 2 Tier B (E4, E5, E8) | 1–2 sessions |
| 3 Test eval | ½ session |
| 4 Pricing | 1 session (incl. verifying the registry rules) |
| 5 Rewrite + 6 Response | 2 sessions |
