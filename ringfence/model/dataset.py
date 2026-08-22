"""Assembles the model matrix and enforces the split/leakage rules in one place.

Every guarantee the project claims about honesty is implemented here, so there
is exactly one file to audit:

  * training rows are restricted to labels that had matured by as_of_day;
  * the graph feature block can be switched off wholesale for the baseline arm;
  * label-bearing and identity columns are dropped by an explicit allowlist, not
    by hoping nothing leaked in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, splits_from_config
from ..features.graph_features import GRAPH_FEATURE_COLS
from ..features.tabular import CATEGORICAL, FEATURE_COLS, NUMERIC

# Columns that must never reach the model under any arm.
FORBIDDEN = {
    "is_fraud", "ring_id", "ring_type", "refunded", "disputed", "refund_day",
    "dispute_day", "label_available_day", "label_matured", "split",
    "benign_cluster", "customer_id", "payment_id", "order_id", "email",
    "email_root", "contact", "device_fingerprint", "ip",
    "shipping_address_hash", "card_fingerprint", "card_last4", "cluster",
    "as_of_day", "dispute_lag_days", "entity", "currency", "status",
    "error_code", "error_reason", "signup_day", "created_at", "day",
}

EXTRA_GRAPH_COLS = ["g_in_cluster", "g_snapshot_lag", "g_match"]


def feature_columns(use_graph: bool) -> tuple[list[str], list[str]]:
    numeric = list(NUMERIC)
    if use_graph:
        numeric += list(GRAPH_FEATURE_COLS) + list(EXTRA_GRAPH_COLS)
    return numeric, list(CATEGORICAL)


def assemble(
    payments: pd.DataFrame,
    tabular: pd.DataFrame,
    graph: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        "payment_id", "customer_id", "day", "amount", "status", "is_fraud",
        "ring_id", "ring_type", "split", "label_available_day", "label_matured",
        "benign_cluster", "account_age_days",
    ]
    base = payments[[c for c in keep if c in payments.columns]].copy()
    merged = base.merge(tabular, on="payment_id", how="left", suffixes=("", "_tab"))
    graph_cols = ["payment_id", "cluster"] + [
        c for c in graph.columns if c in set(GRAPH_FEATURE_COLS) | set(EXTRA_GRAPH_COLS)
    ]
    merged = merged.merge(graph[graph_cols], on="payment_id", how="left")
    if "amount_tab" in merged.columns:
        merged = merged.drop(columns=["amount_tab"])
    return merged


def split_frames(matrix: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    splits = splits_from_config(cfg)
    out = {}
    for name in splits:
        sub = matrix[matrix["split"] == name].copy()
        if name in ("train", "val"):
            # A label that had not matured by as_of_day did not exist when the
            # model was fitted. Using it would be time travel.
            before = len(sub)
            sub = sub[sub["label_matured"].fillna(False)]
            sub.attrs["dropped_immature"] = before - len(sub)
        out[name] = sub
    return out


def xy(frame: pd.DataFrame, use_graph: bool) -> tuple[pd.DataFrame, np.ndarray]:
    numeric, categorical = feature_columns(use_graph)
    cols = [c for c in numeric + categorical if c in frame.columns]
    assert not (set(cols) & FORBIDDEN), f"leak: {set(cols) & FORBIDDEN}"
    X = frame[cols].copy()
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    y = frame["is_fraud"].to_numpy().astype(int)
    return X, y
