"""
Physical-plausibility metrics, applied identically to every model's
predictions (linear regression, baseline GNN, physics-informed GNN) so the
three-way comparison table is apples-to-apples -- this is the metric the
physics-informed model is specifically expected to win on even if it does
not win on R² (see DAY2_STATUS.md for whether that's what happened).

Three metrics, all computed on the TEST split only (never train/val):

1. **neg_rate**: fraction of predictions that are physically impossible
   (negative carbon stock). Zero free parameters, unambiguous.

2. **edge_incoherence_rate**: fraction of graph edges connecting two TEST
   nodes where |y_pred[i]-y_pred[j]| exceeds a threshold. The threshold is
   NOT arbitrary -- it is the 95th percentile of |y_true[i]-y_true[j]|
   computed over TRAIN-side edges (i.e. "how big a neighbor-to-neighbor
   jump actually occurs in real, observed data"), so a model whose
   predicted jumps are more extreme than what real neighboring pixels ever
   actually differ by is flagged, calibrated from data rather than guessed.

3. **physical_consistency_corr**: Pearson correlation between a model's
   test-set predictions and the mass-balance proxy (physics_terms.py) on
   the SAME test nodes. Measures whether a model's predictions stay
   consistent with the physical prior under spatial holdout -- this is the
   metric most directly targeted by adding the physics loss term.
"""
import numpy as np


def negative_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < 0))


def train_edge_jump_threshold(df, edges: np.ndarray, percentile: float = 95.0) -> float:
    """95th percentile of |y_true[i]-y_true[j]| over edges where BOTH
    endpoints are train nodes -- the reference distribution of 'how much
    do real neighboring pixels actually differ' in this AOI."""
    y = df["soc_tha_mean"].values
    is_train = (df["split"] == "train").values
    src, dst = edges[0], edges[1]
    both_train = is_train[src] & is_train[dst]
    diffs = np.abs(y[src[both_train]] - y[dst[both_train]])
    return float(np.percentile(diffs, percentile))


def edge_incoherence_rate(y_pred: np.ndarray, df, edges: np.ndarray, threshold: float) -> float:
    """Fraction of edges with BOTH endpoints in the test split where the
    model's predicted jump exceeds `threshold`."""
    is_test = (df["split"] == "test").values
    src, dst = edges[0], edges[1]
    both_test = is_test[src] & is_test[dst]
    if both_test.sum() == 0:
        return float("nan")
    diffs = np.abs(y_pred[src[both_test]] - y_pred[dst[both_test]])
    return float(np.mean(diffs > threshold))


def physical_consistency_corr(y_pred: np.ndarray, proxy: np.ndarray, mask: np.ndarray) -> float:
    yp, pr = y_pred[mask], proxy[mask]
    if np.std(yp) == 0 or np.std(pr) == 0:
        return float("nan")
    return float(np.corrcoef(yp, pr)[0, 1])


def full_plausibility_report(y_pred: np.ndarray, df, edges: np.ndarray, proxy: np.ndarray,
                              edge_threshold: float) -> dict:
    test_mask = (df["split"] == "test").values
    return {
        "neg_rate_test": negative_rate(y_pred[test_mask]),
        "edge_incoherence_rate_test": edge_incoherence_rate(y_pred, df, edges, edge_threshold),
        "edge_jump_threshold_tha": edge_threshold,
        "physical_consistency_corr_test": physical_consistency_corr(y_pred, proxy, test_mask),
    }
