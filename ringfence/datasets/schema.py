"""The canonical payment schema every dataset adapter must produce.

RingFence's graph, featurisers, model and evaluation code were written against
the synthetic generator. Rather than fork them for a second dataset, an adapter
converts the source data *into this schema* and the whole pipeline runs
unchanged. If that works, the architecture is dataset-agnostic — which is a
stronger claim than any single benchmark number.

Columns fall into three tiers:

  REQUIRED   the pipeline cannot run without them
  LINK       identifier columns the identity graph may join on; a dataset that
             lacks one supplies nulls and the graph simply never links on it
  DERIVED    behavioural context the tabular featuriser reads; a dataset that
             cannot supply one fills the neutral value and that feature goes
             inert rather than lying
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["payment_id", "customer_id", "day", "created_at", "amount", "is_fraud"]

LINK = [
    "card_fingerprint", "device_fingerprint", "shipping_address_hash",
    "contact", "email_root", "ip",
]

DERIVED = [
    "status", "method", "card_network", "card_type", "card_issuer", "city",
    "signup_day", "account_age_days", "refunded", "disputed", "refund_day",
    "dispute_day",
]

# Present only where ground-truth ring membership is known (i.e. the synthetic
# corpus). Real data has fraud labels but no ring labels, so per-archetype and
# novel-ring reporting are simply unavailable there — and are reported as
# unavailable rather than approximated.
GROUND_TRUTH = ["ring_id", "ring_type", "benign_cluster"]

NEUTRAL = {
    "status": "captured",
    "method": "unknown",
    "card_network": None,
    "card_type": None,
    "card_issuer": None,
    "city": None,
    "refunded": False,
    "disputed": False,
    "refund_day": -1,
    "dispute_day": -1,
    "ring_id": "",
    "ring_type": "none",
    "benign_cluster": "",
}


def finalise(df: pd.DataFrame, cfg, name: str) -> pd.DataFrame:
    """Fill neutrals, derive split and maturity, and assert the contract holds."""
    df = df.copy()

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: adapter did not produce required columns {missing}")

    for col in LINK:
        if col not in df.columns:
            df[col] = None
    for col, default in NEUTRAL.items():
        if col not in df.columns:
            df[col] = default

    if "signup_day" not in df.columns:
        df["signup_day"] = df["day"]
    if "account_age_days" not in df.columns:
        df["account_age_days"] = (df["day"] - df["signup_day"]).clip(lower=0)

    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["day"] = df["day"].astype(int)
    df["amount"] = df["amount"].astype(float)

    sim = cfg["simulation"]
    split = np.full(len(df), "unassigned", dtype=object)
    day = df["day"].to_numpy()
    for part in ("train", "val", "test"):
        lo, hi = sim[f"{part}_days"]
        split[(day >= lo) & (day <= hi)] = part
    df["split"] = split

    # Real labels arrive with the dataset; there is no maturity model to apply,
    # so everything counts as matured and the trainer's filter is a no-op.
    if "label_available_day" not in df.columns:
        df["label_available_day"] = df["day"]
    if "label_matured" not in df.columns:
        df["label_matured"] = True

    return df
