"""
Conformalized Quantile Regression (CQR), per paper/uq_decision.md's
recommendation of CQR as the primary UQ method.

Scope decision (stated, not hidden): the quantile regressor here is
trained with pinball loss on the DATA term only, without the physics loss
from physics_terms.py. Combining physics-informed training with
quantile-head UQ in the same run is a reasonable extension but is left as
future work (uq_decision.md already lists ensembles/MC-Dropout ablations
as time-permitting extras; this is the same category) -- the paper's
minimum viable experiment (PLAN.md) only requires CQR calibration on top
of the better-performing model's predictions, not a full physics+UQ joint
model.

Split-conformal calibration set: **val** (already carved out, disjoint
from train and test). This reuses val, which was already used for the
point models' early stopping -- disclosed explicitly here and in
DAY2_STATUS.md, not a silent choice. **test is untouched until the single,
final coverage computation.**

Procedure (Romano, Patterson & Candes 2019):
  1. Train q_low/q_high heads (pinball loss) on train.
  2. On val (calibration): conformity score E_i = max(q_low(x_i) - y_i,
     y_i - q_high(x_i)).
  3. Q = the ceil((n_cal+1)(1-alpha))/n_cal quantile of {E_i}.
  4. Adjusted interval on test: [q_low(x) - Q, q_high(x) + Q].
  5. Report empirical coverage = P(y in adjusted interval) on test, vs.
     nominal (1-alpha), for alpha in {0.10, 0.05}.

Writes results/cqr_coverage.csv and results/cqr_predictions.csv.
"""
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.gnn_common import SEED, FEATURE_COLS, load_graph, make_pyg_data, set_seed  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results"
MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-3
QUANTILES = [0.025, 0.05, 0.95, 0.975]  # covers both 90% (0.05/0.95) and 95% (0.025/0.975) intervals
ALPHAS = {0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}


class QuantileSAGE(nn.Module):
    """Same backbone shape as SAGERegressor (gnn_common.py) -- 2-layer
    GraphSAGE, hidden_dim=16, dropout=0.3 -- but a 4-output head (one per
    quantile in QUANTILES) instead of a single point estimate."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, dropout: float = 0.3, n_quantiles: int = 4):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_quantiles)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.dropout(h, p=self.dropout, train=self.training)
        h = torch.relu(self.conv2(h, edge_index))
        return self.head(h)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """preds: (N, Q), target: (N, 1). Standard pinball/quantile loss,
    summed over quantiles, mean over nodes."""
    target = target.expand(-1, preds.shape[1])
    errors = target - preds
    q = torch.tensor(quantiles, dtype=preds.dtype).unsqueeze(0)
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


def train():
    set_seed(SEED)
    df, edges = load_graph()
    data, scaler = make_pyg_data(df, edges)
    y_mean, y_std = scaler["y_mean"], scaler["y_std"]

    model = QuantileSAGE(in_dim=len(FEATURE_COLS), n_quantiles=len(QUANTILES))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = pinball_loss(out[data.train_mask], data.y[data.train_mask], QUANTILES)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_loss = pinball_loss(out_eval[data.val_mask], data.y[data.val_mask], QUANTILES).item()

        if val_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, no_improve = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == MAX_EPOCHS:
            print(f"epoch {epoch:4d}  train_pinball(std)={loss.item():.4f}  val_pinball(std)={val_loss:.4f}  best_val={best_val:.4f}@{best_epoch}")
        if no_improve >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(data.x, data.edge_index).numpy()
    preds_raw = preds_scaled * y_std + y_mean  # (N, 4), columns = QUANTILES order

    y_true_raw = data.y_raw.numpy().flatten()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    q_idx = {q: i for i, q in enumerate(QUANTILES)}
    coverage_rows = []
    all_pred_cols = {}
    for nominal, (q_lo, q_hi) in ALPHAS.items():
        alpha = 1 - nominal
        lo_raw, hi_raw = preds_raw[:, q_idx[q_lo]], preds_raw[:, q_idx[q_hi]]

        # calibrate on val
        cal_lo, cal_hi, cal_y = lo_raw[val_mask], hi_raw[val_mask], y_true_raw[val_mask]
        scores = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
        n_cal = len(scores)
        level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        Q = np.quantile(scores, level)

        lo_adj, hi_adj = lo_raw - Q, hi_raw + Q

        test_lo, test_hi, test_y = lo_adj[test_mask], hi_adj[test_mask], y_true_raw[test_mask]
        empirical_coverage = float(np.mean((test_y >= test_lo) & (test_y <= test_hi)))
        mean_width = float(np.mean(test_hi[test_lo <= test_hi] - test_lo[test_lo <= test_hi])) if np.any(test_lo <= test_hi) else float("nan")

        coverage_rows.append({
            "nominal_level": nominal, "alpha": alpha, "n_calibration": n_cal,
            "conformal_correction_Q": Q, "n_test": int(test_mask.sum()),
            "empirical_coverage_test": empirical_coverage,
            "mean_interval_width_tha": mean_width,
        })
        all_pred_cols[f"q{q_lo}_raw"] = lo_raw
        all_pred_cols[f"q{q_hi}_raw"] = hi_raw
        all_pred_cols[f"lo_adj_{nominal}"] = lo_adj
        all_pred_cols[f"hi_adj_{nominal}"] = hi_adj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(RESULTS_DIR / "cqr_coverage.csv", index=False)

    pred_df = df.copy()
    pred_df["y_true_soc_tha"] = y_true_raw
    for k, v in all_pred_cols.items():
        pred_df[k] = v
    pred_df.to_csv(RESULTS_DIR / "cqr_predictions.csv", index=False)

    torch.save(model.state_dict(), RESULTS_DIR / "cqr_quantile_weights.pt")

    print()
    print("=== CQR coverage (calibrated on val, evaluated once on test) ===")
    print(cov_df.to_string(index=False))
    print(f"wrote -> {RESULTS_DIR / 'cqr_coverage.csv'}")
    return cov_df


if __name__ == "__main__":
    train()
