"""The rupee cost model.

This is the layer that decides the operating threshold, so an error here moves
every headline number in the project. It is also the layer whose assumptions a
reviewer is most likely to disagree with, which is why the sensitivity sweep
exists and is tested.
"""
from __future__ import annotations

import numpy as np

from ringfence.evaluation import cost

COSTS = {
    "chargeback_fee_inr": 1500,
    "fraud_recovery_rate": 0.12,
    "gross_margin": 0.28,
    "false_block_churn_factor": 2.6,
    "manual_review_cost_inr": 45,
    "review_band": [0.35, 0.80],
}


def _case():
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.99, 0.10, 0.98, 0.05, 0.50, 0.02])
    amounts = np.array([5000.0, 8000.0, 2000.0, 1000.0, 3000.0, 500.0])
    return y, scores, amounts


def test_doing_nothing_costs_the_unrecovered_fraud_plus_a_fee_per_case():
    y, _, amounts = _case()
    expected = amounts[y == 1].sum() * (1 - 0.12) + 2 * 1500
    assert cost.do_nothing_cost(y, amounts, COSTS) == expected


def test_blocking_a_good_customer_costs_more_than_the_lost_margin():
    """A wrongly blocked customer does not come back. If the churn multiple were
    ignored, the model would happily block its way to a better-looking score."""
    y, scores, amounts = _case()
    curve = cost.cost_curve(y, scores, amounts, COSTS, n_points=40)
    blocked = curve[curve["blocked_good"] > 0].iloc[0]
    per_block = blocked["false_block_loss_inr"] / blocked["blocked_good"]
    assert per_block > 0.28 * amounts.min()


def test_review_is_not_free():
    y, scores, amounts = _case()
    priced = cost.cost_curve(y, scores, amounts, COSTS, n_points=40)
    free = dict(COSTS, manual_review_cost_inr=0)
    unpriced = cost.cost_curve(y, scores, amounts, free, n_points=40)
    reviewed = priced[priced["reviewed"] > 0]
    if len(reviewed):
        t = reviewed.iloc[0]["block_threshold"]
        a = priced.loc[priced["block_threshold"] == t, "total_cost_inr"].iloc[0]
        b = unpriced.loc[unpriced["block_threshold"] == t, "total_cost_inr"].iloc[0]
        assert a > b, "routing work to a human has to show up as a cost"


def test_the_optimal_point_is_the_cheapest_point_on_the_curve():
    y, scores, amounts = _case()
    curve = cost.cost_curve(y, scores, amounts, COSTS, n_points=60)
    best = cost.optimal_operating_point(curve)
    assert best["total_cost_inr"] == curve["total_cost_inr"].min()


def test_saving_is_measured_against_doing_nothing_not_against_perfection():
    y, scores, amounts = _case()
    curve = cost.cost_curve(y, scores, amounts, COSTS, n_points=40)
    baseline = cost.do_nothing_cost(y, amounts, COSTS)
    assert np.allclose(curve["net_saving_inr"], baseline - curve["total_cost_inr"])


def test_a_harsher_churn_assumption_never_makes_the_saving_look_better():
    """The sensitivity sweep exists so a reviewer can substitute their own
    assumption. It is only honest if the direction is monotone."""
    y, scores, amounts = _case()
    table = cost.sensitivity(y, scores, amounts, COSTS, churn_factors=(1.0, 2.6, 6.0))
    savings = table.sort_values("churn_factor")["net_saving_inr"].to_numpy()
    assert (np.diff(savings) <= 0).all()
