# Day 3 Status — 2026-08-29 (continued)

## Required fix, applied first

`physical_consistency_corr_test` was reworded everywhere it appeared
(`DAY2_STATUS.md`, `paper/method_notes.md`, `paper/NUMBER_SOURCE_LOG.md`
entry #19) to the precise framing: the 0.645 correlation shows the
physics-informed model's predictions **generalize the imposed prior to
held-out nodes**, not that the prior has been **independently validated**
against physical measurements. The paper (`main.tex`, Section IV) uses
this exact qualified language where the number first appears, and repeats
the distinction, briefly, in Limitations — not only in Limitations.

**Optional independent check: considered, not attempted.** Every
already-downloaded layer not used as a training feature or physics-loss
target is degenerate after mangrove-only node masking (the SOC typology
raster is binary and every kept node is class 1, confirmed by direct
inspection before deciding this). Forcing a correlation against a
constant would not have produced real information, so this was skipped
rather than manufactured — consistent with the instruction not to let it
become scope creep.

## Task 1: pricing proof-of-concept

`src/pricing/pricing_poc.py`, illustrative throughout (code comments,
results file, paper text). On the test split: 90%-CQR interval width
median 257.6 t/ha (range 114.8–425.4). Five representative nodes (min/p25/
median/p75/max of that distribution) get price multipliers 1.500 / 1.072
/ 1.000 / 0.899 / 0.500, against one fixed 0.825 multiplier under a flat
17.5% buffer (midpoint of the cited 15–20% Verra/Plan Vivo range). All
numbers logged in `paper/NUMBER_SOURCE_LOG.md` #25–27 before use in prose.

## Task 2: the paper

`paper/main.tex` + `paper/references.bib`, compiled with pdflatex + bibtex
(MiKTeX, available locally) to `paper/main.pdf` — **4 pages**, IEEEtran
conference format, zero LaTeX warnings on the final pass. Structure
exactly as specified: Intro → Related Work (4 subsections matching
`literature_notes.md`) → Method → Experiments (3 tables) → Pricing PoC (1
table) → Limitations → Conclusion. Abstract is 235 words, Background→
Purpose→Method→Findings→Implications structure, no citations, every
number sourced.

Two citations were verified directly against primary sources rather than
trusted from memory or search summaries before going into
`references.bib`: the Scientific Reports 2024 conformal-prediction paper's
author list (a search-engine summary gave two contradictory author lists;
resolved by reading the PMC mirror directly: Singh, Moncrieff, Venter,
Cawse-Nicholson, Slingsby, Robinson, vol. 14, p. 16166), and the Simard et
al. 2019 dataset citation (confirmed against the ORNL DAAC/Earthdata
catalog page directly — matched what was already in
`literature_notes.md`, no change needed).

**One open item, not a content gap:** the author block has a placeholder
affiliation ("Independent Researcher") marked with a `% TODO(author)`
comment in `main.tex` — I don't have your actual institutional affiliation
on file. Fill this in before submission if one applies.

## Final source-log audit

Read every number in the compiled PDF against
`paper/NUMBER_SOURCE_LOG.md` line by line (abstract, all four tables,
every inline figure in Method/Results/Limitations). Specific checks worth
noting:
- Table I/II/III/IV values matched their source CSVs to the precision
  shown, including derived arithmetic (test$-$val $R^2$ deltas of $+0.165$
  and $+0.064$, recomputed from full-precision values in
  `results/*_metrics.csv` rather than from the already-rounded log
  entries, and confirmed to 3 decimals).
- The undercoverage figures ("4.3–4.8 percentage points," "2.5–3$\times$
  the label's std") were recomputed from `results/cqr_coverage.csv`
  directly ($90-85.7=4.3$, $95-90.2=4.8$; $261.9/103.4=2.53$,
  $310.0/103.4=3.00$), not asserted.
- Radabaugh sub-regional figures quoted in Method ($265.1\pm43.2$ Tampa
  Bay/Sarasota, $409\pm104.4$ Everglades) were checked against the actual
  Table 1 image read earlier this sprint, not retyped from memory.
- Nothing in the manuscript required a number that wasn't already logged;
  no claim was cut or softened to make this audit pass.

**Result: every number in the manuscript traces to a specific file,
row/column, or a direct arithmetic derivation from one already logged.
Nothing added, changed, or fabricated to close a gap.**

## Still open before submission

1. Author affiliation placeholder (above).
2. This sprint's "Day 1/2/3" was three days of project-management framing
   within one continuous working session, not three elapsed calendar
   days — the paper is ready now. Actual submission timing against the
   real ICDABI 2026 deadline (Sept 1, per the site check at the start of
   this project) is a logistics call for you, not something I'm
   re-raising as a decision to relitigate.
3. Everything in `main.tex`'s own Limitations section is real and
   substantive, not boilerplate: single-date imagery, no flux/
   decomposition data, distance-to-gauge as a positional proxy, the
   prior-generalization (not validation) framing, CQR trained without the
   physics loss, CQR undercoverage, and single-seed/single-region scope.
   Nothing here is new information — it's the same limitations tracked in
   `DAY1_STATUS.md`/`DAY2_STATUS.md` throughout, now stated in the paper
   itself.
