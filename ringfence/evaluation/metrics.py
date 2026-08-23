"""Metrics that a fraud reviewer would actually ask for.

PR-AUC and ROC-AUC are reported because they are expected, but they are not the
headline. At a 1-2% base rate, ROC-AUC is nearly uninformative -- it flatters
every model. The numbers that matter are recall at a precision a human queue can
absorb, the per-archetype breakdown, and the split between rings the model had
already seen and rings that are entirely new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PRECISION_TARGETS = (0.95, 0.90, 0.80, 0.60)
RECALL_TARGETS = (0.50, 0.70, 0.90)


def headline(y: np.ndarray, scores: np.ndarray) -> dict:
    if y.sum() == 0:
        return {"pr_auc": np.nan, "roc_auc": np.nan, "base_rate": 0.0, "n": len(y), "positives": 0}
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def recall_at_precision(y: np.ndarray, scores: np.ndarray, target: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # precision_recall_curve returns one more point than thresholds.
    ok = precision[:-1] >= target
    if not ok.any():
        return {"target_precision": target, "recall": 0.0, "threshold": 1.0, "achieved_precision": float(precision[:-1].max()) if len(precision) > 1 else 0.0}
    idx = int(np.argmax(recall[:-1] * ok))
    return {
        "target_precision": target,
        "recall": float(recall[idx]),
        "threshold": float(thresholds[idx]),
        "achieved_precision": float(precision[idx]),
    }


def precision_at_recall(y: np.ndarray, scores: np.ndarray, target: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    ok = recall[:-1] >= target
    if not ok.any():
        return {"target_recall": target, "precision": 0.0, "threshold": 0.0}
    idx = int(np.argmax(precision[:-1] * ok))
    return {
        "target_recall": target,
        "precision": float(precision[idx]),
        "threshold": float(thresholds[idx]),
    }


def operating_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = [recall_at_precision(y, scores, t) for t in PRECISION_TARGETS]
    return pd.DataFrame(rows)


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "alert_rate": (tp + fp) / len(y) if len(y) else 0.0,
    }


def per_archetype(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    df = frame.copy()
    df["_score"] = scores
    df["_flag"] = scores >= threshold
    rows = []
    for archetype in ("card_testing", "refund_abuse", "bust_out"):
        sub = df[df["ring_type"] == archetype]
        if sub.empty:
            continue
        rows.append(
            {
                "archetype": archetype,
                "payments": len(sub),
                "recall": float(sub["_flag"].mean()),
                "median_score": float(sub["_score"].median()),
                "amount_inr": round(sub.loc[sub["status"] == "captured", "amount"].sum() / 100),
                "caught_inr": round(
                    sub.loc[sub["_flag"] & (sub["status"] == "captured"), "amount"].sum() / 100
                ),
            }
        )
    return pd.DataFrame(rows)


def has_ring_labels(frame: pd.DataFrame) -> bool:
    """Real datasets carry fraud labels but not ring membership."""
    if "ring_id" not in frame.columns:
        return False
    fraud = frame.loc[frame["is_fraud"], "ring_id"].astype(str)
    return bool(fraud.str.len().gt(0).any())


def novel_vs_seen_rings(
    test: pd.DataFrame, train: pd.DataFrame, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Split test recall by whether the ring was already active during training.

    This is the number that predicts production behaviour. A model that only
    recognises rings it has already met is a lookup table.
    """
    if not has_ring_labels(test):
        return pd.DataFrame(
            [{"cohort": "unavailable — dataset has no ring labels", "rings": 0,
              "payments": int(test["is_fraud"].sum()), "recall": np.nan}]
        )
    seen = set(train.loc[train["is_fraud"], "ring_id"].unique())
    df = test.copy()
    df["_flag"] = scores >= threshold
    fraud = df[df["is_fraud"]]
    rows = []
    for label, mask in (
        ("seen_in_training", fraud["ring_id"].isin(seen)),
        ("novel_ring", ~fraud["ring_id"].isin(seen)),
    ):
        sub = fraud[mask]
        rows.append(
            {
                "cohort": label,
                "rings": int(sub["ring_id"].nunique()),
                "payments": len(sub),
                "recall": float(sub["_flag"].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def false_positive_composition(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Where do the false positives land? If they concentrate on the benign
    shared-identifier structures, the graph is over-reaching and we say so."""
    df = frame.copy()
    df["_flag"] = scores >= threshold
    fps = df[df["_flag"] & (~df["is_fraud"])]
    if fps.empty:
        return pd.DataFrame(columns=["bucket", "false_positives", "share"])

    def bucket(value: str) -> str:
        if not value:
            return "no_shared_identifier"
        kinds = {part.rsplit("_", 1)[0] for part in str(value).split(";") if part}
        return "+".join(sorted(kinds)) if kinds else "no_shared_identifier"

    labels = fps["benign_cluster"].fillna("").map(bucket)
    counts = labels.value_counts()
    return pd.DataFrame(
        {
            "bucket": counts.index,
            "false_positives": counts.to_numpy(),
            "share": (counts / counts.sum()).round(4).to_numpy(),
        }
    )


def missed_fraud_profile(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """The honest exception list: what the system does not catch, and its cost.

    Every submission shows what it catches. This is the table that says what it
    misses, in rupees, broken out by attack type -- which is the table a risk
    team would actually act on.
    """
    df = frame.copy()
    df["_score"] = scores
    df["_flag"] = scores >= threshold
    missed = df[(~df["_flag"]) & df["is_fraud"]]
    if missed.empty:
        return pd.DataFrame(columns=["archetype", "missed_payments", "missed_inr", "median_score"])
    grouped = missed.groupby("ring_type")
    out = pd.DataFrame(
        {
            "missed_payments": grouped.size(),
            "missed_inr": (
                missed[missed["status"] == "captured"].groupby("ring_type")["amount"].sum() / 100
            ).round(),
            "median_score": grouped["_score"].median().round(4),
            "median_amount_inr": (grouped["amount"].median() / 100).round(),
        }
    ).fillna(0)
    return out.reset_index().rename(columns={"ring_type": "archetype"})
