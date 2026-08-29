"""
Pricing proof-of-concept: CQR interval width -> a risk-adjusted price
multiplier, illustrated on 5 representative held-out (test-split) nodes.

**Labeled everywhere as illustrative/proof-of-concept, not a validated
pricing system** (per PLAN.md's minimum-viable-experiment scope and the
Day 3 sprint instruction). This demonstrates the MECHANISM -- that a
per-node uncertainty signal can drive a differentiated price adjustment,
unlike a flat buffer -- not a calibrated, market-ready price.

## The mechanism (fully disclosed, no hidden fitting)

1. Use the physics-informed GNN's predictions and the 90%-nominal CQR
   interval (results/cqr_predictions.csv, columns lo_adj_0.9/hi_adj_0.9) on
   the TEST split only (the same held-out spatial block used throughout
   this project -- these are genuine out-of-sample intervals, not
   calibration-set or training-set intervals).
2. relative_width(node) = interval_width(node) / median(interval_width)
   over ALL test nodes -- a real, computed reference point, not an assumed
   constant.
3. price_adjustment_pct = -ALPHA * (relative_width - 1), clipped to
   [-MAX_ADJ, +MAX_ADJ]. ALPHA=1.0 (a 1:1 illustrative sensitivity: a node
   with double the median interval width gets a 100%-of-ALPHA discount,
   i.e. moves the full clipped range) is an arbitrary, stated illustrative
   choice -- NOT fit, optimized, or validated against any real market
   outcome. This is declared explicitly in every place this number is used.
4. price_multiplier = 1 + price_adjustment_pct.

## The comparator

Plan Vivo Standard V5.0 (2022) -- the current version, verified against
the primary source (paper/literature_notes.md §3) -- states a FLAT,
non-permanence-risk buffer of 20%, applied uniformly to every credit in a
project regardless of node-level uncertainty. FLAT_BUFFER_PCT=20 is that
stated value directly, not an average. An earlier version of this script
used 17.5, the midpoint of the older Plan Vivo v4.0's "15-20%" range;
that was corrected once V5.0's single fixed number was confirmed as the
current standard -- averaging a superseded version's range when the
actual cited source gives one fixed number would itself be a defensible
target for reviewer criticism, so it was fixed rather than left as a
rounding convenience. FLAT_BUFFER_PCT is used here as a single reference
number against which every example node's multiplier is compared, to
illustrate that our UQ-driven signal differentiates where a flat buffer
cannot -- not to claim it is the exact number every methodology or
project uses (VM0033's own scored range can differ by project).

Writes results/pricing_poc_examples.csv.
"""
import pathlib

import numpy as np
import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"

ALPHA = 1.0
MAX_ADJ = 0.5  # clip price_adjustment_pct to +/-50%
FLAT_BUFFER_PCT = 20  # Plan Vivo Standard V5.0's stated flat non-permanence buffer (not an averaged range)
N_EXAMPLES = 5


def compute_price_multiplier(interval_width: np.ndarray) -> np.ndarray:
    rel_width = interval_width / np.median(interval_width)
    adj = np.clip(-ALPHA * (rel_width - 1), -MAX_ADJ, MAX_ADJ)
    return 1.0 + adj, adj


def main():
    # Phase 2b (post test-set-integrity fix, revised after the localized-
    # vs-global reframe): sourced from the ONE test-set evaluation
    # (phase2_cqr_test_predictions.csv), using the GLOBAL calibration
    # columns. Changed from localized (Mondrian) -- Phase 1D's
    # pre-registered carry-forward method -- because Phase 2's single test
    # evaluation showed Mondrian undercovers MORE than global on the real
    # test set (NUMBER_SOURCE_LOG.md #45), reversing the validation-internal
    # comparison. The paper's Discussion states this plainly: global is the
    # currently defensible choice for any near-term pricing deployment, so
    # the illustrative pricing mechanism below is now built on global
    # intervals to keep the paper internally consistent -- it would
    # undercut the paper's own recommendation to illustrate deployment on
    # the method the paper itself says not to deploy. Mondrian columns
    # remain in phase2_cqr_test_predictions.csv for audit/comparison.
    cqr = pd.read_csv(RESULTS_DIR / "phase2_cqr_test_predictions.csv")
    test = cqr[cqr["split"] == "test"].copy()
    test["interval_width_90"] = test["hi_adj_global_0.9"] - test["lo_adj_global_0.9"]
    test = test.rename(columns={"lo_adj_global_0.9": "lo_adj_0.9", "hi_adj_global_0.9": "hi_adj_0.9"})

    # Global calibration produced no inverted/degenerate intervals on this
    # test set (n_degenerate_interval_test=0 for the global rows in
    # phase2_cqr_test_coverage.csv) -- unlike the Mondrian variant, whose
    # 1-node degenerate case motivated this same guard originally. Kept as
    # a defensive check, not because it currently triggers.
    n_degenerate = int((test["interval_width_90"] < 0).sum())
    if n_degenerate:
        print(f"excluding {n_degenerate} node(s) with an inverted (degenerate) interval "
              f"from the representative pricing examples (coverage failure, not a narrow interval)")
    test = test[test["interval_width_90"] >= 0].copy()

    multiplier, adj_pct = compute_price_multiplier(test["interval_width_90"].values)
    test["price_multiplier"] = multiplier
    test["price_adjustment_pct"] = adj_pct * 100
    test["flat_buffer_multiplier"] = 1 - FLAT_BUFFER_PCT / 100
    test["multiplier_vs_flat_buffer_pp"] = (test["price_multiplier"] - test["flat_buffer_multiplier"]) * 100

    # 5 representative examples spanning the interval-width distribution:
    # min, 25th pct, median, 75th pct, max (real percentile ranks on the
    # actual test-set distribution, not hand-picked nodes).
    sorted_test = test.sort_values("interval_width_90").reset_index(drop=True)
    n = len(sorted_test)
    idx = [0, int(0.25 * (n - 1)), int(0.50 * (n - 1)), int(0.75 * (n - 1)), n - 1]
    examples = sorted_test.iloc[idx][[
        "lon", "lat", "y_true_soc_tha", "lo_adj_0.9", "hi_adj_0.9", "interval_width_90",
        "price_multiplier", "price_adjustment_pct", "flat_buffer_multiplier", "multiplier_vs_flat_buffer_pp",
    ]].copy()
    examples.insert(0, "example_rank", ["min", "p25", "median", "p75", "max"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    examples.to_csv(RESULTS_DIR / "pricing_poc_examples.csv", index=False)

    print(f"test-set median 90% interval width: {np.median(test['interval_width_90']):.1f} t/ha")
    print(f"test-set interval width range: {test['interval_width_90'].min():.1f} - {test['interval_width_90'].max():.1f} t/ha")
    print()
    print(examples.to_string(index=False))
    print(f"\nwrote -> {RESULTS_DIR / 'pricing_poc_examples.csv'}")
    return examples


if __name__ == "__main__":
    main()
