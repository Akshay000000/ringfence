"""Why did this payment score the way it did?

Attribution by **group occlusion**, not SHAP.

The question an analyst asks is not "what did feature `cl_signup_synchrony`
contribute" -- it is "was this flagged because of velocity, or because of who
this account is connected to?". So features are bucketed into semantic reason
groups, and each group is attributed by replacing it wholesale with what a
normal customer looks like and re-scoring:

    contribution(group) = p(payment) - p(payment with `group` set to normal)

A positive number means that group pushed the score up. This is a genuine
counterfactual on the actual model -- "if this account's connections had looked
ordinary, the score would have been 0.31 instead of 0.99" -- which is both
easier to defend in a dispute and easier to test than an additive approximation.

It is also honest about its limits: occluding groups one at a time cannot
untangle interactions between them, so the contributions do not sum to the
score and are never presented as if they do. They are ranked, not totalled.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..features.graph_features import CLUSTER_FEATURE_COLS
from ..model.dataset import xy

# Groups are ordered by how an analyst would read them: what this payment did,
# then who this account is, then what it is connected to.
REASON_GROUPS: dict[str, dict] = {
    "transaction_velocity": {
        "label": "Transaction velocity",
        "cols": [
            "vel_cust_1h", "vel_cust_24h", "vel_cust_7d", "vel_dev_1h",
            "vel_dev_24h", "vel_dev_7d", "vel_card_1h", "vel_card_24h",
            "vel_ip_1h", "vel_ip_24h", "seconds_since_prev_customer",
            "seconds_since_prev_device",
        ],
        "evidence": [
            ("vel_dev_1h", "{n:.0f} attempt(s) from this device in the past hour"),
            ("vel_dev_24h", "{n:.0f} attempt(s) from this device in the past day"),
            ("vel_cust_1h", "{n:.0f} attempt(s) from this account in the past hour"),
            ("vel_cust_24h", "{n:.0f} attempt(s) from this account in the past day"),
        ],
    },
    "card_fanout": {
        "label": "Cards per device",
        "cols": [
            "prior_distinct_cards_device", "new_cards_on_device_24h",
            "new_cards_on_device_7d", "prior_distinct_cards_customer",
        ],
        "evidence": [
            ("prior_distinct_cards_device", "{n:.0f} distinct card(s) already seen on this device"),
            ("new_cards_on_device_24h", "{n:.0f} new card(s) appeared on this device in the past day"),
            ("prior_distinct_cards_customer", "{n:.0f} distinct card(s) used by this account"),
        ],
    },
    "decline_pattern": {
        "label": "Decline pattern",
        "cols": ["prior_customer_failure_rate", "prior_customer_failures"],
        "evidence": [
            ("prior_customer_failure_rate", "{n:.0%} of this account's prior attempts were declined"),
            ("prior_customer_failures", "{n:.0f} prior decline(s) on this account"),
        ],
    },
    "account_tenure": {
        "label": "Account tenure",
        "cols": ["account_age_days", "prior_customer_payments"],
        "evidence": [
            ("account_age_days", "account is {n:.0f} day(s) old"),
            ("prior_customer_payments", "only {n:.0f} prior payment(s) on this account"),
        ],
    },
    "order_value": {
        "label": "Order value",
        "cols": ["amount", "log_amount", "is_micro_ticket", "amount_z_vs_customer"],
        "evidence": [
            ("amount_z_vs_customer", "{n:.1f}x this account's typical order value"),
            ("amount", "order value Rs {rupees:,.0f}"),
        ],
    },
    "refund_history": {
        "label": "Refund history",
        "cols": ["prior_customer_refunds", "prior_customer_refund_rate"],
        "evidence": [
            ("prior_customer_refund_rate", "{n:.0%} of this account's prior orders were refunded"),
            ("prior_customer_refunds", "{n:.0f} prior refund(s) on this account"),
        ],
    },
    "device_sharing": {
        "label": "Device sharing",
        "cols": ["prior_distinct_customers_device", "prior_distinct_devices_customer"],
        "evidence": [
            ("prior_distinct_customers_device", "{n:.0f} distinct account(s) have used this device"),
            ("prior_distinct_devices_customer", "{n:.0f} distinct device(s) used by this account"),
        ],
    },
    "address_concentration": {
        "label": "Address concentration",
        "cols": [
            "prior_distinct_customers_address", "vel_addr_24h", "vel_addr_7d",
            "cl_customers_per_address", "cl_distinct_addresses",
        ],
        "evidence": [
            ("prior_distinct_customers_address", "{n:.0f} distinct account(s) have shipped to this address"),
            ("cl_customers_per_address", "{n:.0f} account(s) per address across the linked cluster"),
            ("vel_addr_24h", "{n:.0f} order(s) to this address in the past day"),
        ],
    },
    "ring_cohesion": {
        "label": "Linked-account cluster",
        "cols": [
            "g_degree", "g_weighted_degree", "g_max_edge_weight",
            "g_mean_edge_weight", "size", "edges", "density",
            "mean_edge_weight", "max_edge_weight", "clsig_cohesion",
            "clsig_scale", "g_in_cluster", "g_match", "g_snapshot_lag",
            "cl_customers", "cl_payments",
        ],
        "evidence": [
            ("cl_customers", "sits in a cluster of {n:.0f} linked account(s)"),
            ("g_degree", "shares an identifier with {n:.0f} other account(s)"),
            ("mean_edge_weight", "linked by high-confidence identifiers (combined edge weight {n:.2f})"),
        ],
    },
    "cohort_synchrony": {
        "label": "Account cohort synchrony",
        "cols": [
            "cl_signup_std", "cl_signup_span", "cl_first_seen_std",
            "cl_first_seen_span", "cl_signup_synchrony",
            "cl_first_seen_synchrony", "cl_signup_span_per_customer",
            "cl_new_account_share", "clsig_cohort_synchrony",
            "clsig_new_accounts", "clsig_fresh", "cl_account_age_mean",
            "cl_account_age_min",
        ],
        "evidence": [
            ("cl_new_account_share", "{n:.0%} of the linked accounts are under 30 days old"),
            ("cl_signup_span", "all linked accounts opened within {n:.0f} day(s) of each other"),
            ("cl_account_age_mean", "linked accounts average {n:.0f} day(s) old"),
        ],
    },
    "cluster_card_fanout": {
        "label": "Cards per device in cluster",
        "cols": ["cl_cards_per_device", "cl_distinct_cards", "cl_distinct_devices",
                 "clsig_card_fanout"],
        "evidence": [
            ("cl_cards_per_device", "{n:.0f} card(s) per device across the linked cluster"),
            ("cl_distinct_cards", "{n:.0f} distinct card(s) used across the cluster"),
        ],
    },
    "cluster_declines": {
        "label": "Cluster decline rate",
        "cols": ["cl_failure_rate", "clsig_decline"],
        "evidence": [
            ("cl_failure_rate", "cluster decline rate {n:.0%}"),
        ],
    },
    "cluster_tempo": {
        "label": "Cluster tempo",
        "cols": ["cl_burstiness", "cl_payments_per_active_day", "cl_active_days",
                 "cl_day_span", "cl_last_seen_lag", "clsig_burst"],
        "evidence": [
            ("cl_day_span", "the whole cluster transacted inside {n:.0f} day(s)"),
            ("cl_payments_per_active_day", "{n:.1f} payments per active day across the cluster"),
            ("cl_burstiness", "cluster activity is compressed (burstiness {n:.2f})"),
        ],
    },
    "cluster_value": {
        "label": "Cluster order values",
        "cols": ["cl_amount_mean", "cl_amount_median", "cl_amount_cv"],
        "evidence": [
            ("cl_amount_mean", "cluster averages Rs {rupees:,.0f} per order"),
            ("cl_amount_cv", "unusually uniform order values across the cluster (cv {n:.2f})"),
        ],
    },
    "cluster_issuer_spread": {
        "label": "Card issuer spread",
        "cols": ["cl_distinct_issuers", "cl_issuers_per_customer", "cl_distinct_ips",
                 "cl_ring_risk", "clsig_address_concentration"],
        "evidence": [
            ("cl_distinct_issuers", "{n:.0f} different card issuer(s) behind one cluster"),
            ("cl_ring_risk", "cluster ring-risk prior {n:.2f}"),
        ],
    },
    "context": {
        "label": "Payment context",
        "cols": ["hour_of_day", "day_of_week", "method", "card_network",
                 "card_type", "card_issuer", "city"],
        "evidence": [("method", "paid by {n}")],
    },
}

# Every model feature must belong to exactly one group. Enforced by a test:
# an ungrouped feature is silently invisible in every explanation, which is the
# kind of gap you only notice when a reviewer asks why a score has no reason.
GRAPH_GROUPS = {
    "ring_cohesion", "cohort_synchrony", "cluster_card_fanout",
    "cluster_declines", "cluster_tempo", "cluster_value",
    "cluster_issuer_spread",
}


def build_reference(train: pd.DataFrame, use_graph: bool) -> pd.DataFrame:
    """What 'normal' looks like, and how much normal varies.

    `center` is the median honest payment -- what occlusion replaces a group
    with. Using the honest median rather than zero matters: zero is not neutral
    for a feature like `cl_signup_synchrony`, where zero means "maximally spread
    out" and is itself an exculpatory signal.

    `scale` is the honest median absolute deviation, used to rank which piece of
    evidence to quote. Without it, ranking by raw deviation always picks whatever
    feature happens to be measured in the largest units -- a count of 72 beats a
    rate of 0.9 every time, regardless of which is actually more abnormal.
    """
    honest = train[~train["is_fraud"]]
    X, _ = xy(honest, use_graph)
    centers, scales = {}, {}
    for col in X.columns:
        series = X[col]
        if str(series.dtype) == "category" or series.dtype == object:
            mode = series.mode()
            centers[col] = mode.iloc[0] if len(mode) else series.iloc[0]
            scales[col] = np.nan
        else:
            values = series.to_numpy(dtype="float64")
            center = float(np.nanmedian(values))
            mad = float(np.nanmedian(np.abs(values - center)))
            centers[col] = center
            # MAD collapses to 0 for near-constant features; fall back to IQR,
            # then to a unit floor, so the ratio never explodes.
            if not np.isfinite(mad) or mad == 0:
                q75, q25 = np.nanpercentile(values, [75, 25])
                mad = float(q75 - q25)
            scales[col] = mad if np.isfinite(mad) and mad > 0 else 1.0
    return pd.DataFrame({"center": pd.Series(centers), "scale": pd.Series(scales)})


def reference_centers(reference) -> pd.Series:
    """Accept either the new DataFrame form or a bare Series of centers."""
    if isinstance(reference, pd.DataFrame):
        return reference["center"]
    return reference


def attribute(arm, rows: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Group-occlusion contributions for a batch of payments.

    Vectorised over rows: one predict per group, not one per (row, group).
    """
    from ..model.train import arm_matrix

    centers = reference_centers(reference)
    X = arm_matrix(arm, rows)
    base = arm.model.predict_proba(X)[:, 1]

    records = []
    for name, spec in REASON_GROUPS.items():
        cols = [c for c in spec["cols"] if c in X.columns]
        if not cols:
            continue
        occluded = X.copy()
        for col in cols:
            value = centers.get(col)
            if value is None:
                continue
            if str(occluded[col].dtype) == "category":
                if value not in occluded[col].cat.categories:
                    occluded[col] = occluded[col].cat.add_categories([value])
            occluded[col] = value
        scored = arm.model.predict_proba(occluded)[:, 1]
        records.append(
            pd.DataFrame(
                {
                    "payment_id": rows["payment_id"].to_numpy(),
                    "group": name,
                    "label": spec["label"],
                    "contribution": base - scored,
                    "score": base,
                    "is_graph_group": name in GRAPH_GROUPS,
                }
            )
        )

    if not records:
        return pd.DataFrame(columns=["payment_id", "group", "label", "contribution", "score"])
    return pd.concat(records, ignore_index=True)


def _pluralise(text: str) -> str:
    """Resolve 'account(s)' using the first number in the sentence.

    The number and the pluralised noun are not always adjacent -- "2 distinct
    account(s)" has a word between them -- so this keys off the leading quantity
    rather than trying to match them as a pair.
    """
    lead = re.search(r"-?[\d,]+(?:\.\d+)?", text)
    singular = False
    if lead:
        try:
            singular = abs(float(lead.group(0).replace(",", ""))) == 1.0
        except ValueError:
            singular = False
    return re.sub(r"\(s\)", "" if singular else "s", text)


def _fmt(value, template: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        rendered = template.format(n=value, rupees=(float(value) / 100 if "rupees" in template else 0.0))
    except (ValueError, TypeError):
        return None
    return _pluralise(rendered)


def _robust_z(value, col: str, reference) -> float:
    """How many honest-population MADs from normal this value sits."""
    try:
        if isinstance(reference, pd.DataFrame) and col in reference.index:
            center = float(reference.at[col, "center"])
            scale = float(reference.at[col, "scale"])
            if np.isfinite(scale) and scale > 0:
                return abs(float(value) - center) / scale
        elif isinstance(reference, pd.Series) and col in reference.index:
            center = float(reference[col])
            return abs(float(value) - center) / (abs(center) + 1.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


# A value must be at least this abnormal before it is worth quoting to an analyst.
EVIDENCE_Z_THRESHOLD = 3.0


def _pick_evidence(row: pd.Series, spec: dict, reference=None) -> str | None:
    """Quote the first *genuinely abnormal* piece of evidence, in author order.

    Ranking purely by deviation does not work: a feature with a tiny spread in
    the honest population -- `cl_payments_per_active_day`, whose MAD is close to
    zero -- produces an enormous z-score for any deviation at all and wins every
    explanation, so every alert ended up narrated the same way.

    So deviation is a *gate*, not a ranking. The evidence list for each group is
    ordered by how a human would want to hear it, and the first line clearing the
    abnormality threshold is the one quoted. If nothing clears it, the most
    deviant line is used, so a reason is never left blank.
    """
    candidates = []
    for col, template in spec.get("evidence", []):
        if col not in row.index:
            continue
        value = row[col]
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        rendered = _fmt(value, template)
        if rendered is None:
            continue
        candidates.append((rendered, _robust_z(value, col, reference)))

    if not candidates:
        return None
    for rendered, z in candidates:
        if z >= EVIDENCE_Z_THRESHOLD:
            return rendered
    best, best_z = max(candidates, key=lambda pair: pair[1])
    # Quoting an unremarkable value as the reason for a high score is worse than
    # quoting nothing: "0 attempts from this device in the past hour" reads as
    # evidence of innocence. Below this floor, fall back to the group name.
    return best if best_z >= 0.5 else None


def _has_cluster(row: pd.Series) -> bool:
    value = row.get("g_in_cluster")
    try:
        return bool(float(value) > 0)
    except (TypeError, ValueError):
        return False


def narrate(
    row: pd.Series,
    attributions: pd.DataFrame,
    top_k: int = 3,
    min_contribution: float = 0.01,
    reference: pd.Series | None = None,
) -> dict:
    """Turn contributions into a sentence an analyst can act on or contest."""
    ranked = attributions.sort_values("contribution", ascending=False)
    ranked = ranked[ranked["contribution"] >= min_contribution]

    caveat = None
    if not _has_cluster(row):
        # No cluster resolved. Graph groups still produce a large "contribution"
        # here, but it is an artefact, not evidence: occlusion replaces the
        # missing cluster features with honest-population medians, which makes
        # the payment look more normal and drops the score. Reporting that as a
        # top reason tells an analyst the payment is suspicious *because* nothing
        # is known about it, which is exactly backwards.
        #
        # So graph groups are dropped from the reasons and the gap is stated
        # plainly instead. The score still stands; it simply rests on the
        # transaction features alone.
        ranked = ranked[~ranked["group"].isin(GRAPH_GROUPS)]
        caveat = (
            "No linked-account evidence was available at scoring time — this "
            "score rests on transaction features alone."
        )

    ranked = ranked.head(top_k)

    reasons = []
    seen_details: set[str] = set()
    for _, entry in ranked.iterrows():
        spec = REASON_GROUPS[entry["group"]]
        detail = _pick_evidence(row, spec, reference)
        text = detail or spec["label"].lower()
        if text in seen_details:
            continue
        seen_details.add(text)
        reasons.append(
            {
                "group": entry["group"],
                "label": spec["label"],
                "contribution": round(float(entry["contribution"]), 4),
                "detail": text,
                # True when no evidence line cleared the abnormality floor and
                # `detail` is just the group name restated. Surfaces suppress the
                # detail line rather than printing "Payment context - payment
                # context".
                "generic": detail is None,
                "from_graph": bool(entry["is_graph_group"]),
            }
        )

    score = float(attributions["score"].iloc[0]) if len(attributions) else float("nan")
    if not reasons:
        summary = f"Scored {score:.3f} with no single dominant driver."
    else:
        specific = [r["detail"] for r in reasons if not r["generic"]]
        parts = "; ".join(specific[:3]) if specific else ", ".join(
            r["label"].lower() for r in reasons[:3]
        )
        summary = f"Scored {score:.3f}. Driven by {parts}."
    graph_share = sum(r["contribution"] for r in reasons if r["from_graph"])
    total = sum(r["contribution"] for r in reasons) or 1.0
    return {
        "payment_id": row.get("payment_id"),
        "score": round(score, 4),
        "summary": summary,
        "reasons": reasons,
        "caveat": caveat,
        "graph_driven": bool(graph_share / total > 0.5),
    }
