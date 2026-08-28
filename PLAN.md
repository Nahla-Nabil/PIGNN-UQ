# Execution Plan

Written 2026-08-29.

## Deadline reality check (read this first)

The real ICDABI 2026 (7th edition) dates, confirmed directly from
`icdabi.uob.edu.bh` on 2026-08-29:

- Paper submission: **September 1, 2026 — 3 days from today.**
- Acceptance notice: October 1, 2026. Camera-ready: October 10, 2026.
- Conference: October 28–29, 2026, online.
- Track confirmed to exist: **"Artificial Intelligence for Business and
  Industry"** (matches this project's intended track).
- No extension is currently posted for 2026 (ICDABI 2025's deadline *was*
  extended by ~5 weeks last cycle — not something to plan around, but worth
  re-checking the site closer to the date in case it recurs).

**Three days is not enough runway for a genuine trained model + validated
result**, and the user has confirmed: this plan targets a **later
submission cycle** (ICDABI's typical annual cadence is an Aug/Sept
deadline for an Oct/Nov conference — i.e. effectively ICDABI 2027 — or any
other 5-page IEEE-format venue with a comparable window, once one is
locked in). The week numbers below are relative to project kickoff, not
tied to a specific calendar date — slot in the real deadline once chosen
and count backward from Week 8.

## What's already done (today, 2026-08-29 — "Week 0")

Completed in this session, not scheduled work:
- Repo scaffold, git init, venv (see README.md for the Windows path-length
  note on venv location).
- Literature scan across all 4 required areas (`paper/literature_notes.md`).
- UQ method decision: **CQR primary**, MC Dropout / deep ensembles as
  ablation (`paper/uq_decision.md`).
- Pilot region locked: Ten Thousand Islands / Rookery Bay margin, SW
  Florida (`data/README.md`).
- Real data pulled and verified: Sentinel-2 sample (19 scenes indexed, one
  AOI-clipped multi-band raster), full-AOI GMW 2020 extent mask, windowed
  SOC-typology sample, 30 days of real NOAA tide data. Two gaps
  (SOC stock zip, Simard AGB) documented with exact unblock steps.

---

## Scope cuts (explicit, so nothing sprawls)

Confirmed, not up for revisiting mid-project without a deliberate
re-scoping decision:

1. **ONE region:** Ten Thousand Islands / Rookery Bay margin, SW Florida.
   No second region, no cross-region generalization claim.
2. **ONE physics constraint:** a discrete carbon mass-balance residual per
   node per time step —
   `ΔC_predicted ≈ Accretion_proxy − Export_proxy`
   where `Accretion_proxy` comes from a Sentinel-2 vegetation-index /
   biomass-proxy time series (NDVI or a published index→AGB regression) and
   `Export_proxy` comes from the NOAA tide series (e.g. tidal range /
   inundation frequency at each node, as a proxy for sediment/carbon
   export — see Risk R3 below on why this is a proxy, not full
   hydrodynamics). No second physics term (no separate decomposition model,
   no explicit sea-level-rise submergence term) unless Week 6 has slack.
3. **ONE UQ method as primary:** conformalized quantile regression (CQR),
   spatially-blocked calibration. MC Dropout is a cheap same-model
   comparison; deep ensembles are cut first under time pressure (see
   `uq_decision.md`).
4. **ONE architecture family:** a single message-passing GNN (e.g. GraphSAGE
   or GAT — pick one in Week 2, do not architecture-search). A non-graph
   RF/XGBoost baseline is kept only as the literature-standard comparison
   point, not as a second model to tune.

## Minimum viable experiment (the smallest result that still supports the paper's core claim)

The paper's core claim has two parts — both need a result, but each has a
minimal form:

- **Claim A: "the physics constraint improves prediction reliability."**
  Minimum viable version: train the GNN with vs. without the mass-balance
  loss term on the *same* train/val/test split; report point-prediction
  error (RMSE/MAE against the best available regional carbon-stock
  reference — Radabaugh et al. 2023 numbers or the Simard raster, whichever
  unblocks first) **and** physical-residual violation (mean |mass-balance
  residual|) for both variants. This does not require deep ensembles, does
  not require multi-year Sentinel-2 coverage — one wet-season and one
  dry-season composite is enough to give the accretion/export proxy a
  contrast to fit against.
- **Claim B: "UQ enables defensible pricing."** Minimum viable version:
  CQR intervals from the physics-informed model, calibrated on a spatially
  held-out block, with **measured empirical coverage reported honestly**
  (even if it undershoots nominal — report it, don't hide it). Feed
  interval width through a simple monotonic discount function and show,
  for a handful of representative nodes (stable interior / eroding fringe /
  open-water-adjacent), how the resulting discount differs from Verra/Plan
  Vivo's flat 15–20% buffer. This does not require the pricing function to
  be economically sophisticated — a simple linear or step function from
  interval width to discount is sufficient and should be stated as such.

If Week 3–4 run over budget, **Claim A alone with an honest, clearly-labeled
"future work" note on Claim B** is the fallback minimum for the paper —
not zero results.

---

## Week-by-week timeline

| Week | Focus | Concrete deliverables |
|---|---|---|
| **1** | Close remaining data gaps + graph construction | Run the NASA Earthdata registration + pull Simard AGB (or confirm/obtain Radabaugh raster); pull full SOC stock zip locally, extract AOI tile; pull a second GMW epoch for a change signal; assemble a multi-date Sentinel-2 composite (wet + dry season minimum) via `fetch_sentinel2.py`. Define graph nodes (grid cells or mangrove-patch polygons over the AOI), edges (adjacency), and node feature table joining all layers. **Output:** `data/processed/graph_v1.*` + a notebook showing node count, feature completeness, label coverage. |
| **2** | Baseline models | Install heavy ML stack locally (torch, torch-geometric — not attempted in the sandbox, see `requirements.txt` note). Implement + train: (a) RF/XGBoost tabular baseline, (b) plain GNN, no physics term. Fix train/val/test split as **spatially blocked**, decided once and reused for every later experiment (required for valid CQR calibration later — see `uq_decision.md`). **Output:** baseline RMSE/MAE table, `experiments/configs/baseline_gnn.yaml`. |
| **3** | Physics-informed loss | Implement the mass-balance residual term; train PINN-GNN variant on the identical split. Ablate the loss weight (2–3 values, not a full sweep). **Output:** Claim A comparison table + residual-violation plot — this is Figure/Table #1 (see mapping below). |
| **4** | UQ module | Add quantile heads to the chosen GNN (physics-informed variant), implement split-conformal calibration on the spatially held-out block, report empirical coverage. Add MC Dropout as the cheap comparison. **Output:** coverage table + interval-width map — Figure/Table #2. |
| **5** | Pricing module | Implement the UQ→discount/premium function (`src/pricing/`), apply to representative nodes, build the comparison against the VM0033/Plan Vivo flat buffer baseline from `literature_notes.md` §3. **Output:** pricing comparison figure/table — Figure/Table #3. |
| **6** | Consolidation + slack | Run whatever ablations Weeks 1–5 left slack for (deep ensemble comparison; second physics-term variant; second GMW epoch change-signal experiment) **only if on schedule** — this week is the designated cut point if behind. Freeze all numbers. |
| **7** | Writing | Full 5-page IEEE draft: Intro, Related Work (from `literature_notes.md`), Method (architecture + physics loss + UQ), Experiments/Results, Pricing/Discussion, Conclusion. Figures finalized from Weeks 3–5 outputs. |
| **8** | Buffer | Internal re-read against IEEE 5-page limit, citation formatting check, co-author/advisor review pass if applicable, resubmit-ready PDF, submission logistics (IEEE PDF eXpress if ICDABI requires it — confirm on the CFP page closer to the date). This week exists *because* Week 7 draft quality depends on Weeks 1–6 landing on schedule — do not compress it to claw back time elsewhere. |

**Total: 8 weeks from kickoff.** Front-loaded exactly as requested: data +
both models done by end of Week 3, UQ + pricing (the paper's actual novel
contribution combination) done by end of Week 5, three full weeks left for
writing/buffer.

---

## Open risks and mitigations

| Risk | Why it matters | Mitigation / fallback |
|---|---|---|
| **R1: No field-validated ground truth is fully unblocked in time.** Simard/ORNL needs a manual login step; the Radabaugh raster's download link is unconfirmed. | Claim A needs *some* external number to validate against, or it's a self-consistency check only. | Two independent fallbacks already identified (`data/README.md` §5–6): the Lagomasino/Fatoyinbo Everglades LiDAR AGB numbers, and Radabaugh et al.'s *published scalar* (393.9 ± 107.1 Mg C ha⁻¹) even without the raster. Worst case: validate against the SOC/typology layers we already have on disk (weaker, but still real, not synthetic). |
| **R2: The mass-balance equation is too simplified to be defensible.** Using NDVI as an accretion proxy and tidal range as an export proxy is a large simplification of real sediment/carbon dynamics. | A reviewer in the "AI for Business and Industry" track is unlikely to be a coastal biogeochemist, but *someone* on the panel may push back on the physics. | State the simplification explicitly and up front in the Method section as a scoping decision, not a hidden assumption (matches the paper's stated "ONE physics constraint" scope cut). Frame the contribution as "a *tractable* physics-informed constraint that measurably improves reliability," not "a high-fidelity biogeochemical model." Cite the flood-forecasting PINN-GNN precedent (`literature_notes.md` §1) — they make an analogous simplification (discretized mass conservation, not full Navier-Stokes) and it is accepted practice in that literature. |
| **R3: NOAA tide data is a proxy for export, not a direct sediment/carbon flux measurement.** | Same family of concern as R2 — the physics term's second half is indirect. | Documented explicitly in the scope-cuts section above; no fix planned, just honest framing. If Week 6 has slack, a sensitivity check (does the result hold under a different export-proxy formulation?) would strengthen this rather than trying to source real flux data, which doesn't appear to exist for this AOI. |
| **R4: CQR coverage undershoots nominal on the spatially-blocked test set.** | Conformal guarantees hold under exchangeability; spatial autocorrelation is exactly the violation that can break this in practice (`literature_notes.md` §4). | This is a *reportable result*, not a failure to hide — Claim B's minimum viable version already asks for honest coverage reporting. If it undershoots badly, discuss it as a limitation and a concrete direction for the "future work" ablation (spatially-aware conformal variants, cited in `literature_notes.md`). |
| **R5: Heavy ML stack (torch, torch-geometric) is untested in this sandbox.** | Only lightweight geospatial packages were installed and verified here; GPU-accelerated training was never attempted. | `requirements.txt` flags this explicitly. Week 2 should install and smoke-test the stack on whatever machine will actually run training *first*, before writing any model code against it — don't assume it installs cleanly. |
| **R6: Windows MAX_PATH issue recurs.** | Already hit once this session (rasterio DLL load failure from the very long project folder path) — worked around by placing the venv outside the repo at `C:\Users\nahla\venvs\bcpg`. | Documented in README.md. Keep using the external venv location; don't move it back inside the deeply-nested project folder. If it recurs elsewhere (e.g. a deeply nested experiment output path), the same fix applies: shorten the path, don't fight Windows' loader. |
| **R7: No confirmed target venue/date yet.** | This plan is deadline-agnostic by design (see top of this document) — that's a feature for avoiding a repeat of the Sept-1 scramble, but it means there's no forcing function yet. | Lock in a real venue + date before Week 1 starts, and recheck that venue's CFP page directly (not just search results) the way this session did for ICDABI — the gap between what search snippets say and what the live page says was substantial here. |

---

## Experiment → claim / figure / table mapping

Nothing below should be built if it doesn't map to one of these rows — this
is the check against scope creep in a 5-page paper.

| Experiment (Week) | Claim it supports | Paper artifact |
|---|---|---|
| Baseline GNN vs. RF/XGBoost (Week 2) | Establishes the graph representation is at least competitive with the field-standard tabular approach (`literature_notes.md` §2) before adding physics | Table 1 (baseline comparison) — small, sets up Table 2 |
| Plain GNN vs. physics-informed GNN, same split (Week 3) | **Claim A**: physics constraint improves prediction reliability | Table 2 (RMSE/MAE + residual violation) + Figure 1 (map of physics-residual before/after) |
| CQR calibration + coverage report (Week 4) | **Claim B, part 1**: UQ is honestly calibrated (or honestly reported if it isn't) | Figure 2 (coverage plot) + Table 3 (interval width by node type: interior/fringe/open-water-adjacent) |
| Pricing function applied to representative nodes (Week 5) | **Claim B, part 2**: UQ enables a defensible, differentiated pricing signal vs. the industry's flat buffer | Figure 3 (discount/premium vs. flat-buffer comparison) — the paper's business-relevance payoff figure for the "AI for Business and Industry" track |
| MC Dropout comparison (Week 4, cheap) | Substantiates the UQ method choice in `uq_decision.md` with the project's own numbers, not just citations | One row added to Table 3, not a separate table |
| Deep ensemble comparison (Week 6, only if on schedule) | Strengthens the same claim further | Cut first if behind schedule — explicitly the first thing to drop |
| Second GMW epoch / change signal (Week 6, only if on schedule) | Would strengthen the accretion-proxy formulation | Cut second if behind schedule |

**If only 3 rows survive schedule pressure, keep the first three** (baseline
comparison, physics ablation, CQR coverage) — those three alone still
support both halves of the core claim; everything below the line in the
table is genuinely optional.
