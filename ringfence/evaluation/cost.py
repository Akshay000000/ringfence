"""Rupee cost model and threshold selection.

The objective is not F1. A merchant does not care about the harmonic mean of
two ratios; they care about how much money the system saves net of the money it
destroys by blocking good customers.

    cost(t) = fraud_let_through(t) * (1 - recovery_rate)
            + blocked_good_gmv(t) * margin * churn_factor
            + reviewed(t) * review_cost

Three properties make this honest rather than decorative:

  1. False positives are charged at more than the lost margin on the order,
     because a wrongly blocked customer does not come back. churn_factor is the
     LTV multiple, and it is the assumption most likely to be argued with --
     which is why the sensitivity table publishes the answer across a range.
  2. Review is not free. A model that routes 30% of traffic to a human queue
     has not solved the problem, it has moved it, and the cost curve says so.
  3. The comparison baseline is do-nothing (accept everything), not a
     hypothetical perfect system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _components(
    y: np.ndarray,
    scores: np.ndarray,
    amounts_inr: np.ndarray,
    block_t: float,
    review_lo: float,
    costs: dict,
) -> dict:
    blocked = scores >= block_t
    reviewed = (scores >= review_lo) & (~blocked)

    caught_fraud = blocked & (y == 1)
    missed_fraud = (~blocked) & (y == 1)
    blocked_good = blocked & (y == 0)

    recovery = float(costs["fraud_recovery_rate"])
    fraud_loss = amounts_inr[missed_fraud].sum() * (1 - recovery)
    chargeback = missed_fraud.sum() * float(costs["chargeback_fee_inr"])
    fp_loss = (
        amounts_inr[blocked_good].sum()
        * float(costs["gross_margin"])
        * float(costs["false_block_churn_factor"])
    )
    review_loss = reviewed.sum() * float(costs["manual_review_cost_inr"])

    return {
        "block_threshold": block_t,
        "blocked": int(blocked.sum()),
        "reviewed": int(reviewed.sum()),
        "caught_fraud": int(caught_fraud.sum()),
        "missed_fraud": int(missed_fraud.sum()),
        "blocked_good": int(blocked_good.sum()),
        "fraud_loss_inr": float(fraud_loss),
        "chargeback_fee_inr": float(chargeback),
        "false_block_loss_inr": float(fp_loss),
        "review_cost_inr": float(review_loss),
        "total_cost_inr": float(fraud_loss + chargeback + fp_loss + review_loss),
    }


def do_nothing_cost(y: np.ndarray, amounts_inr: np.ndarray, costs: dict) -> float:
    recovery = float(costs["fraud_recovery_rate"])
    fraud = y == 1
    return float(
        amounts_inr[fraud].sum() * (1 - recovery)
        + fraud.sum() * float(costs["chargeback_fee_inr"])
    )


def cost_curve(
    y: np.ndarray,
    scores: np.ndarray,
    amounts_inr: np.ndarray,
    costs: dict,
    n_points: int = 200,
) -> pd.DataFrame:
    review_lo = float(costs["review_band"][0])
    grid = np.unique(np.round(np.linspace(0.01, 0.999, n_points), 4))
    rows = [_components(y, scores, amounts_inr, t, review_lo, costs) for t in grid]
    curve = pd.DataFrame(rows)
    baseline = do_nothing_cost(y, amounts_inr, costs)
    curve["do_nothing_cost_inr"] = baseline
    curve["net_saving_inr"] = baseline - curve["total_cost_inr"]
    return curve


def optimal_operating_point(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["total_cost_inr"].idxmin()]


def sensitivity(
    y: np.ndarray,
    scores: np.ndarray,
    amounts_inr: np.ndarray,
    costs: dict,
    churn_factors: tuple[float, ...] = (1.0, 1.8, 2.6, 4.0, 6.0),
) -> pd.DataFrame:
    """How much does the conclusion depend on the most arguable assumption?

    If the optimal threshold and the net saving move wildly with churn_factor,
    the headline number is an artefact of a guess and should be presented as a
    range, not a point.
    """
    rows = []
    for factor in churn_factors:
        variant = dict(costs)
        variant["false_block_churn_factor"] = factor
        curve = cost_curve(y, scores, amounts_inr, variant, n_points=90)
        best = optimal_operating_point(curve)
        rows.append(
            {
                "churn_factor": factor,
                "optimal_threshold": round(float(best["block_threshold"]), 4),
                "blocked": int(best["blocked"]),
                "caught_fraud": int(best["caught_fraud"]),
                "net_saving_inr": round(float(best["net_saving_inr"])),
                "saving_vs_do_nothing_%": round(
                    100 * float(best["net_saving_inr"]) / float(best["do_nothing_cost_inr"]), 1
                ),
            }
        )
    return pd.DataFrame(rows)
