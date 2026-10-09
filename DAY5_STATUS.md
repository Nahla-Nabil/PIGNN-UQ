# Day 5 Status — 2026-08-31 (EDAS final-polish + self-audit)

This file was not kept phase-by-phase during Day 5 the way `DAY1-4_STATUS.md`
were (the day's work is instead fully reconstructable from `git log` on
`day5-edas-final-polish` and from `paper/NUMBER_SOURCE_LOG.md` entries
#57-93, which cover the page-budget work, Fig.~1 cut, Fig.~2/Fig.~3
legibility bugs found and fixed, the Must-Fix response to the 5-reviewer
synthesis, and the full figure/citation/reference self-audit). This entry
is a narrow, targeted note for one specific request, not a retroactive
backfill of the whole day.

## Two-sentence rewording pass (text-only, no NUMBER_SOURCE_LOG.md entry)

Two sentences in `paper/main.tex` were reworded on explicit, narrowly-scoped
request: rephrase so neither reuses a near-verbatim textbook/field-standard
phrasing, while stating the exact same fact and keeping the same citation(s)
in the same place. No number, table, other citation, or the abstract was
touched — confirmed via `git diff --stat` (1 file, 10 insertions/9
deletions, both hunks exactly these two sentences) and a full recompile.

1. **Related Work, CQR general-definition sentence.** "CQR [17] gives a
   finite-sample, distribution-free coverage guarantee under
   exchangeability, validated for earth observation [6]..." →
   "CQR [17] turns a point predictor into calibrated intervals with a
   coverage guarantee that holds without distributional assumptions, so
   long as calibration and test data are exchangeable, validated for
   earth observation [6]..." Citations [17] (`romano2019cqr`) and [6]
   (`eo_conformal2024`) unchanged in position; only the wording between
   them changed.

2. **Discussion, infill-asymptotic sentence.** "This matches theory:
   spatial conformal validity requires an *infill asymptotic* regime with
   sufficient calibration-point density per local region [20], plausibly
   what our split validation set lacked..." → "This matches theory: [20]
   shows that local conformal validity needs calibration points dense
   enough within each local region, an *infill asymptotic* requirement
   this AOI's validation split plausibly did not satisfy..." Reworded to
   lead with citation [20] (`mao2024validmodelfree`) explaining *why*,
   rather than restating the general theory first — the "only 6 spatial
   superblocks" observation itself was left exactly where it already was,
   one sentence earlier, unchanged, so the two sentences don't end up
   repeating the same specific fact twice.

**Verification:** both sentences read back in full paragraph context to
confirm natural flow (no grammatical seam). Full clean recompile
(`.aux`/`.bbl`/`.blg` deleted first): **5 pages, 0 warnings/errors,
21/21 references present**, identical to the pre-edit state.

**EDAS re-upload:** this is a precautionary wording pass, not a fix for a
reviewer-facing defect, a citation error, or a page-budget problem — the
previously-submitted PDF (if already uploaded to EDAS) is not wrong or
broken. Re-uploading this version is optional, at the author's discretion,
not required.
