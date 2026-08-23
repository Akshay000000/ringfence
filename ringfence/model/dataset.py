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


def feature_columns(use_graph: bool, frame: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Model columns for an arm.

    `src_*` columns are transaction-level features the source dataset already
    ships -- on IEEE-CIS, Vesta's C-counters and D-timedeltas. They go to BOTH
    arms, which matters: they are exactly the kind of feature a real fraud team
    already has, and withholding them from the baseline would inflate the graph
    lift by comparing against a strawman. Excluded from the graph block, so the
    ablation still isolates the one thing under test.
    """
    numeric = list(NUMERIC)
    if frame is not None:
        numeric += sorted(c for c in frame.columns if c.startswith("src_"))
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
    keep += sorted(c for c in payments.columns if c.startswith("src_"))
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


# sklearn's HistGradientBoosting refuses a categorical with more than 255
# levels. IEEE-CIS's billing region (addr1) has 330.
MAX_CATEGORIES = 254
OTHER = "__other__"
NONE = "__none__"


def build_category_vocab(frame: pd.DataFrame, use_graph: bool) -> dict[str, list[str]]:
    """Fix each categorical's level set, from TRAINING data only.

    The obvious fix for an over-cardinality column is "keep the commonest levels
    and fold the rest into __other__". But computing that inside the transform
    means the vocabulary is derived from whichever split is being transformed,
    so the test set helps decide its own encoding. Mild, and still leakage. The
    vocabulary is fitted once on train and carried on the trained arm.
    """
    _, categorical = feature_columns(use_graph, frame)
    vocab: dict[str, list[str]] = {}
    for col in categorical:
        if col not in frame.columns:
            continue
        counts = frame[col].astype("string").fillna(NONE).value_counts()
        levels = [str(v) for v in counts.index[:MAX_CATEGORIES]]
        if len(counts) > MAX_CATEGORIES:
            levels.append(OTHER)
        vocab[col] = levels
    return vocab


def xy(
    frame: pd.DataFrame,
    use_graph: bool,
    categories: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    numeric, categorical = feature_columns(use_graph, frame)
    cols = [c for c in numeric + categorical if c in frame.columns]
    assert not (set(cols) & FORBIDDEN), f"leak: {set(cols) & FORBIDDEN}"
    X = frame[cols].copy()
    for c in categorical:
        if c not in X.columns:
            continue
        values = X[c].astype("string").fillna(NONE)
        if categories and c in categories:
            levels = categories[c]
            if OTHER in levels:
                values = values.where(values.isin(set(levels)), OTHER)
            X[c] = pd.Categorical(values, categories=levels)
        else:
            X[c] = values.astype("category")
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    y = frame["is_fraud"].to_numpy().astype(int)
    return X, y
