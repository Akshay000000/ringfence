"""Honest customer population, benign confounding structures, and honest traffic.

The confounders are the point of this module. Real commerce is full of
legitimate identifier sharing, and a ring detector that has never been shown a
family sharing a credit card will flag every family that shares a credit card.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import entities as ent


def _activity_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """One-and-done / occasional / loyal mixture."""
    u = rng.random(n)
    weights = np.where(
        u < 0.52, rng.gamma(1.2, 0.20, size=n),          # one-and-done
        np.where(u < 0.88, rng.gamma(2.0, 0.55, size=n),  # occasional
                 rng.gamma(3.0, 1.60, size=n)),           # loyal
    )
    return np.clip(weights, 0.02, None)


def build_population(cfg, rng: np.random.Generator) -> pd.DataFrame:
    n = int(cfg["simulation"]["n_customers"])
    days = int(cfg["simulation"]["days"])

    email, email_root = ent.make_emails(rng, n)
    cards = ent.make_cards(rng, n)

    # Signup dates: a legacy base plus continuous acquisition through the whole
    # window, so "new account" is a normal state for an honest customer.
    honest_cfg = cfg["honest"]
    legacy_share = float(honest_cfg.get("legacy_customer_share", 0.42))
    lg_lo, lg_hi = honest_cfg.get("legacy_signup_range", [-400, -60])
    rc_lo, rc_hi = honest_cfg.get("recent_signup_range", [-60, days - 1])
    is_legacy = rng.random(n) < legacy_share
    signup_day = np.where(
        is_legacy,
        rng.integers(lg_lo, lg_hi, size=n),
        rng.integers(rc_lo, max(rc_hi, rc_lo + 1), size=n),
    )

    pop = pd.DataFrame(
        {
            "customer_id": ent.mint_ids(rng, "cust_", n),
            "email": email,
            "email_root": email_root,
            "contact": ent.make_contacts(rng, n),
            "device_fingerprint": ent.make_devices(rng, n),
            "ip": ent.make_ips(rng, n),
            "shipping_address_hash": ent.make_addresses(rng, n),
            "signup_day": signup_day,
            "city": rng.choice(ent.CITIES, size=n),
            # Activity propensity: a mixture, not a single Pareto. Most accounts
            # buy once or twice; a minority are loyal. A pure Pareto gives honest
            # accounts hundreds of prior payments and hands the model a
            # giveaway feature.
            "activity_weight": _activity_weights(rng, n),
            "is_ring": False,
            "ring_id": "",
            "ring_type": "none",
            **cards,
        }
    )
    return pop


def _apply_sharing(
    pop: pd.DataFrame,
    rng: np.random.Generator,
    column: str,
    n_groups: int,
    group_size_range: tuple[int, int],
    tag: str,
) -> pd.DataFrame:
    """Force `n_groups` clusters of honest customers to share one identifier value.

    These are the false-positive traps. Each is annotated so the eval can report
    exactly how many alerts landed on benign structure.
    """
    if n_groups <= 0 or len(pop) == 0:
        return pop

    lo, hi = group_size_range
    sizes = rng.integers(lo, hi + 1, size=n_groups)
    total = int(sizes.sum())
    if total >= len(pop):
        return pop

    picks = rng.choice(len(pop), size=total, replace=False)
    col_values = pop[column].to_numpy(copy=True)
    confounder = pop["benign_cluster"].to_numpy(copy=True)

    cursor = 0
    for gi, size in enumerate(sizes):
        members = picks[cursor : cursor + size]
        cursor += size
        anchor = col_values[members[0]]
        col_values[members] = anchor
        for m in members:
            existing = confounder[m]
            confounder[m] = f"{existing};{tag}_{gi}" if existing else f"{tag}_{gi}"

    pop[column] = col_values
    pop["benign_cluster"] = confounder
    return pop


def apply_confounders(cfg, pop: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    conf = cfg["confounders"]
    pop = pop.copy()
    pop["benign_cluster"] = ""

    # A family sharing one card: strong link type, small cluster, aged accounts.
    pop = _apply_sharing(pop, rng, "card_fingerprint", int(conf["shared_family_cards"]), (2, 4), "family_card")
    # Carrier / hostel / office NAT: weak link type, enormous cluster.
    pop = _apply_sharing(pop, rng, "ip", int(conf["shared_nat_ips"]), (40, 400), "nat_ip")
    # Shared or refurbished handset: strong link type, mid cluster.
    pop = _apply_sharing(pop, rng, "device_fingerprint", int(conf["shared_kiosk_devices"]), (5, 25), "kiosk_device")
    # Apartment complex / office delivery point.
    pop = _apply_sharing(pop, rng, "shipping_address_hash", int(conf["shared_building_addresses"]), (6, 30), "building_addr")
    return pop


def generate_honest_payments(cfg, pop: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    days = int(cfg["simulation"]["days"])
    honest = cfg["honest"]
    per_day = int(cfg["simulation"]["payments_per_day_mean"])

    total = int(rng.poisson(per_day * days))
    weights = pop["activity_weight"].to_numpy()
    weights = weights / weights.sum()
    cust_idx = rng.choice(len(pop), size=total, p=weights)

    signup = pop["signup_day"].to_numpy()[cust_idx]
    lower = np.maximum(signup, 0)
    span = np.maximum(days - lower, 1)
    day = lower + (rng.random(total) * span).astype(int)
    day = np.clip(day, 0, days - 1)

    # Intra-day timing: Indian retail double hump, lunch and late evening.
    hour = np.where(
        rng.random(total) < 0.42,
        rng.normal(13.5, 1.9, size=total),
        rng.normal(21.0, 2.1, size=total),
    )
    hour = np.clip(hour, 0, 23.999)
    ts = day * 86400 + (hour * 3600).astype(int) + rng.integers(0, 3600, size=total)

    amount = np.round(
        rng.lognormal(honest["amount_lognorm_mu"], honest["amount_lognorm_sigma"], size=total)
    ).astype(int) * 100  # paise
    amount = np.clip(amount, 4900, 90_000_00)

    # Overlay the small-ticket segment so that low amount is suggestive, not
    # diagnostic. Card testing has to be caught on structure, not on price.
    micro = rng.random(total) < float(honest.get("micro_ticket_share", 0.0))
    if micro.any():
        lo, hi = honest.get("micro_amount_range_paise", [200, 4900])
        amount[micro] = rng.integers(lo, hi + 1, size=int(micro.sum()))

    methods = list(honest["method_mix"].keys())
    probs = np.array([honest["method_mix"][m] for m in methods], dtype=float)
    probs = probs / probs.sum()
    method = rng.choice(methods, size=total, p=probs)

    failed = rng.random(total) < float(honest["base_failure_rate"])
    fail_idx = rng.choice(len(ent.FAILURE_REASONS), size=total, p=ent.HONEST_FAILURE_P)
    error_code = np.where(failed, [ent.FAILURE_REASONS[i][0] for i in fail_idx], None)
    error_reason = np.where(failed, [ent.FAILURE_REASONS[i][1] for i in fail_idx], None)

    df = pd.DataFrame(
        {
            "customer_idx": cust_idx,
            "day": day,
            "created_at": ts,
            "amount": amount,
            "method": method,
            "status": np.where(failed, "failed", "captured"),
            "error_code": error_code,
            "error_reason": error_reason,
            "is_fraud": False,
            "ring_id": "",
            "ring_type": "none",
        }
    )

    # Identifier drift: people change phones and networks. This injects the
    # graph noise that stops the detector from assuming one account = one device.
    df["device_fingerprint"] = pop["device_fingerprint"].to_numpy()[cust_idx]
    df["ip"] = pop["ip"].to_numpy()[cust_idx]
    fresh_dev = rng.random(total) < 0.05
    fresh_ip = rng.random(total) < 0.18
    if fresh_dev.any():
        df.loc[fresh_dev, "device_fingerprint"] = ent.make_devices(rng, int(fresh_dev.sum()))
    if fresh_ip.any():
        df.loc[fresh_ip, "ip"] = ent.make_ips(rng, int(fresh_ip.sum()))

    for col in ("customer_id", "email", "email_root", "contact", "shipping_address_hash",
                "card_fingerprint", "card_last4", "card_network", "card_type",
                "card_issuer", "signup_day", "city"):
        df[col] = pop[col].to_numpy()[cust_idx]

    # Non-card methods carry no card fingerprint; leaving one in would hand the
    # graph a link that does not exist in reality.
    non_card = df["method"].to_numpy() != "card"
    for col in ("card_fingerprint", "card_last4", "card_network", "card_type", "card_issuer"):
        df.loc[non_card, col] = None

    df["refunded"] = (df["status"] == "captured") & (rng.random(total) < float(honest["refund_rate"]))
    df["disputed"] = (df["status"] == "captured") & (rng.random(total) < float(honest["dispute_rate"]))
    return df
