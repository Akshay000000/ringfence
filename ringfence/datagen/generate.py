"""Orchestrates the synthetic corpus: population -> confounders -> rings ->
label maturation -> temporal split.

The label-maturation step is the one most fraud demos skip. A chargeback on a
day-80 payment does not exist until day ~125. Training on it is time travel.
This module stamps every payment with the day its outcome actually became
observable, and the trainer is forbidden from using rows that had not matured
by the end of its window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, ensure_dirs, splits_from_config
from ..io import write_table
from . import entities as ent
from . import population as pop_mod
from . import rings as rings_mod

REFUND_LAG_RANGE = (2, 26)


def _assign_maturation(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    mat = cfg.get("label_maturity", {}) or {}
    RETURN_WINDOW_DAYS = int(mat.get("return_window_days", 30))
    DISPUTE_WINDOW_DAYS = int(mat.get("dispute_window_days", 45))
    n = len(df)
    day = df["day"].to_numpy()

    refund_lag = rng.integers(*REFUND_LAG_RANGE, size=n)
    if "dispute_lag_days" in df.columns:
        dispute_lag = df["dispute_lag_days"].fillna(0).to_numpy().astype(int)
        dispute_lag = np.where(dispute_lag > 0, dispute_lag, rng.integers(20, 70, size=n))
    else:
        dispute_lag = rng.integers(20, 70, size=n)

    refunded = df["refunded"].fillna(False).to_numpy().astype(bool)
    disputed = df["disputed"].fillna(False).to_numpy().astype(bool)
    failed = (df["status"].to_numpy() == "failed")

    # A failed authorisation is its own outcome and is known immediately.
    matured = np.full(n, 0, dtype=int)
    matured = np.where(failed, day, day + RETURN_WINDOW_DAYS)
    matured = np.where(refunded, day + refund_lag, matured)
    matured = np.where(disputed, day + dispute_lag, matured)
    # A clean captured payment is only provably clean once its dispute window shuts.
    clean = (~failed) & (~refunded) & (~disputed)
    matured = np.where(clean, day + DISPUTE_WINDOW_DAYS, matured)

    df = df.copy()
    df["refund_day"] = np.where(refunded, day + refund_lag, -1)
    df["dispute_day"] = np.where(disputed, day + dispute_lag, -1)
    df["label_available_day"] = matured
    return df


def _assign_split(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    splits = splits_from_config(cfg)
    day = df["day"].to_numpy()
    split = np.full(len(df), "unassigned", dtype=object)
    for name, sp in splits.items():
        split[(day >= sp.start_day) & (day <= sp.end_day)] = name
    df = df.copy()
    df["split"] = split
    return df


def _add_benign_cluster(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    lookup = dict(zip(pop["customer_id"], pop["benign_cluster"]))
    df = df.copy()
    df["benign_cluster"] = df["customer_id"].map(lookup).fillna("")
    return df


def generate_corpus(cfg: Config, save: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(cfg["seed"]))

    pop = pop_mod.build_population(cfg, rng)
    pop = pop_mod.apply_confounders(cfg, pop, rng)

    honest = pop_mod.generate_honest_payments(cfg, pop, rng)
    honest = honest.drop(columns=["customer_idx"])

    parts = [
        honest,
        rings_mod.gen_card_testing(cfg, rng),
        rings_mod.gen_refund_abuse(cfg, rng),
        rings_mod.gen_bust_out(cfg, rng),
    ]
    payments = pd.concat(parts, ignore_index=True, sort=False)

    payments = payments.sort_values("created_at", kind="stable").reset_index(drop=True)
    payments.insert(0, "payment_id", ent.mint_ids(rng, "pay_", len(payments)))
    payments["order_id"] = ent.mint_ids(rng, "order_", len(payments))
    payments["currency"] = cfg["simulation"]["currency"]
    payments["entity"] = "payment"

    for col in ("refunded", "disputed"):
        payments[col] = payments[col].fillna(False).astype(bool)
    payments["is_fraud"] = payments["is_fraud"].fillna(False).astype(bool)
    payments["ring_id"] = payments["ring_id"].fillna("").astype(str)
    payments["ring_type"] = payments["ring_type"].fillna("none").astype(str)
    payments["signup_day"] = payments["signup_day"].astype(int)
    payments["account_age_days"] = (payments["day"] - payments["signup_day"]).clip(lower=0)

    payments = _assign_maturation(payments, cfg, rng)
    payments = _assign_split(payments, cfg)
    as_of = int(cfg["simulation"].get("as_of_day", cfg["simulation"]["days"]))
    payments["label_matured"] = payments["label_available_day"] <= as_of
    payments = _add_benign_cluster(payments, pop)

    refunds = payments.loc[payments["refunded"], ["payment_id", "amount", "refund_day"]].copy()
    refunds.insert(0, "refund_id", ent.mint_ids(rng, "rfnd_", len(refunds)))
    refunds["entity"] = "refund"
    refunds["created_at"] = refunds["refund_day"] * 86400

    disputes = payments.loc[payments["disputed"], ["payment_id", "amount", "dispute_day"]].copy()
    disputes.insert(0, "dispute_id", ent.mint_ids(rng, "disp_", len(disputes)))
    disputes["entity"] = "dispute"
    disputes["phase"] = rng.choice(
        ["fraud", "chargeback", "pre_arbitration"], size=len(disputes), p=[0.34, 0.56, 0.10]
    )
    disputes["status"] = rng.choice(["open", "under_review", "lost", "won"],
                                    size=len(disputes), p=[0.18, 0.22, 0.49, 0.11])
    disputes["created_at"] = disputes["dispute_day"] * 86400

    out = {"payments": payments, "refunds": refunds, "disputes": disputes, "population": pop}

    if save:
        ensure_dirs()
        for name, frame in out.items():
            write_table(frame, name)
    return out


def summarise(payments: pd.DataFrame) -> pd.DataFrame:
    # Real datasets carry fraud labels but no ring labels. Report that as "n/a"
    # rather than silently printing a meaningless 1.
    has_rings = (
        "ring_id" in payments.columns
        and payments.loc[payments["is_fraud"], "ring_id"].astype(str).str.len().gt(0).any()
    )
    rows = []
    for split in ("train", "val", "test"):
        sub = payments[payments["split"] == split]
        if sub.empty:
            continue
        rows.append(
            {
                "split": split,
                "payments": len(sub),
                "fraud": int(sub["is_fraud"].sum()),
                "fraud_rate_%": round(100 * sub["is_fraud"].mean(), 3),
                "rings": sub.loc[sub["is_fraud"], "ring_id"].nunique() if has_rings else "n/a",
                "gmv": round(sub.loc[sub["status"] == "captured", "amount"].sum() / 100),
                "fraud_value": round(
                    sub.loc[sub["is_fraud"] & (sub["status"] == "captured"), "amount"].sum() / 100
                ),
            }
        )
    return pd.DataFrame(rows)
