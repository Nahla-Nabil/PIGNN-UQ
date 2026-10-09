"""
Uncertainty-aware credit issuance (revision, paper/revision_preregistration.md §6).

Replaces the submitted paper's symmetric 0.5-1.5 price multiplier (arbitrary
alpha, premium above the point estimate) with two conservative rules. Both
keep estimation uncertainty and non-permanence risk as SEPARATE factors:

    credited_i = stock_i * (1 - deduction_i) * (1 - buffer)

where buffer is the registry's own non-permanence buffer (Plan Vivo V5.0
20%, unchanged) and deduction_i depends only on estimation uncertainty.

Rule A (pre-registered, §6): one-sided conservative. Credit the lower bound
    of the 90% interval: deduction_i = 1 - lo_i / yhat_i.
Rule B (registry-anchored): VM0033 v2.1 (Verra, 4 Sept 2023), p. 64, Eq. 92:
    "The allowable uncertainty is 20% or 30% of NER_t at a 90% or 95%
    confidence level, respectively. Where this precision level is met, no
    deduction must result for uncertainty. Where this precision level is
    exceeded, a deduction equal to the amount that the uncertainty exceeds
    the allowable level must be applied."
    adjusted_NER_t = NER_t x (100% - NER_ERROR + allowable_uncert)
    Uncertainty is "expressed as 90% confidence interval as a percentage of
    the mean" (Eqs. 87-89 definitions). We use the per-node 90% conformal
    interval half-width / yhat as that percentage. The methodology applies
    the rule at project/stratum level; applying it per node is our
    extension. Project-level aggregation (VM0033 Eq. 88, area-weighted
    root-sum-of-squares) is also reported, with the caveat that it assumes
    independent node errors, which spatial autocorrelation violates.

Neither rule ever credits more than the point estimate (no premium).
"""
import numpy as np

PLAN_VIVO_BUFFER = 0.20
VM0033_ALLOWABLE_90 = 0.20


def deduction_lower_bound(yhat, lo):
    return np.clip(1 - np.maximum(lo, 0) / yhat, 0, 1)


def node_uncertainty_pct(yhat, lo, hi):
    return (hi - lo) / 2 / yhat


def deduction_vm0033(unc_pct, allowable=VM0033_ALLOWABLE_90):
    return np.clip(unc_pct - allowable, 0, 1)


def project_uncertainty_vm0033(unc_pct, stock, area=None):
    """VM0033 Eq. 87/88-style combination: sqrt(sum((U_i * E_i)^2)) / sum(E_i)
    with E_i = stock_i * area_i (equal node areas here). Assumes independent
    errors; spatially correlated errors make this optimistic."""
    area = np.ones_like(stock) if area is None else area
    e = stock * area
    return np.sqrt(np.sum((unc_pct * e) ** 2)) / np.sum(e)


def issuance_table(yhat, lo, hi, buffer=PLAN_VIVO_BUFFER):
    """Per-node and portfolio summaries (stock units, t C/ha summed over
    equal-area nodes, so totals are proportional to tonnes)."""
    unc = node_uncertainty_pct(yhat, lo, hi)
    dA = deduction_lower_bound(yhat, lo)
    dB = deduction_vm0033(unc)
    flat = yhat * (1 - buffer)
    A = yhat * (1 - dA) * (1 - buffer)
    B = yhat * (1 - dB) * (1 - buffer)
    proj_unc = project_uncertainty_vm0033(unc, yhat)
    B_proj = yhat * (1 - deduction_vm0033(proj_unc)) * (1 - buffer)
    return {
        "n_nodes": len(yhat),
        "median_unc_pct": float(np.median(unc)),
        "frac_nodes_within_allowable": float(np.mean(unc <= VM0033_ALLOWABLE_90)),
        "total_flat": float(flat.sum()),
        "total_ruleA_lowerbound": float(A.sum()),
        "total_ruleB_vm0033_node": float(B.sum()),
        "total_ruleB_vm0033_project": float(B_proj.sum()),
        "project_unc_pct_independent": float(proj_unc),
        "ruleA_vs_flat": float(A.sum() / flat.sum()),
        "ruleB_node_vs_flat": float(B.sum() / flat.sum()),
        "ruleB_project_vs_flat": float(B_proj.sum() / flat.sum()),
        "dedB_p10": float(np.percentile(dB, 10)), "dedB_p50": float(np.percentile(dB, 50)),
        "dedB_p90": float(np.percentile(dB, 90)),
    }


# Market sensitivity (Addendum A4): NOT market validation. All three prices
# come from Farahmand, Hilmi & Duarte (2025), "The rise and flows of blue
# carbon credits advance global climate and biodiversity goals", npj Ocean
# Sustainability 4:39, doi:10.1038/s44183-025-00141-6:
#   26.96 USD = "weighted average price ... per BCC" (2022-2023);
#   21.9  USD = 2024 Australian Safeguard Mechanism carbon price (cited there);
#   54.3  USD = "estimated value of carbon in the buying countries" (2022-2023).
C_TO_CO2E = 44.0 / 12.0
PRICES_USD_PER_TCO2E = {"safeguard_AU_2024": 21.9, "bcc_weighted_avg_2022_23": 26.96, "buyer_country_value": 54.3}
NODE_AREA_HA = 150.0 * 150.0 / 1e4  # 5x5 blocks of ~30 m cells (2.25 ha); approximate, stated


def revenue_table(totals_tc_per_ha: dict):
    """totals_tc_per_ha: {rule: sum over nodes of credited t C/ha}. Converts
    to tCO2e over the node area and to USD at each reference price."""
    rows = []
    for rule, tot in totals_tc_per_ha.items():
        tco2e = tot * NODE_AREA_HA * C_TO_CO2E
        rows.append({"rule": rule, "tCO2e": tco2e,
                     **{f"usd_{k}": tco2e * p for k, p in PRICES_USD_PER_TCO2E.items()}})
    return rows
