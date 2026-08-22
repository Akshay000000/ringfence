"""Three abuse-ring archetypes, each with a different link topology.

They are deliberately different shapes, because a detector that only works on
one shape is a detector that has learned the simulator rather than the problem:

  card_testing  strong device link, huge card fan-out, minutes-long burst
  refund_abuse  shared address + email-root aliasing, weeks-long, refund exit
  bust_out      *no* card link at all, only a drop address, chargeback exit at T+45d

bust_out is the adversarial case on purpose: the crew uses a distinct stolen
card and a distinct handset per mule account, so every transaction-level signal
looks clean and only the delivery address betrays them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import entities as ent

PAYMENT_COLUMNS = [
    "customer_id", "email", "email_root", "contact", "device_fingerprint", "ip",
    "shipping_address_hash", "card_fingerprint", "card_last4", "card_network",
    "card_type", "card_issuer", "signup_day", "city", "day", "created_at",
    "amount", "method", "status", "error_code", "error_reason", "refunded",
    "disputed", "is_fraud", "ring_id", "ring_type",
]


def _blank_frame(n: int) -> dict:
    return {c: [None] * n for c in PAYMENT_COLUMNS}


def _ri(rng, rng_range) -> int:
    lo, hi = rng_range
    return int(rng.integers(lo, hi + 1))


def _signup_days(cfg, rng: np.random.Generator, start_day: int, n: int, fresh_max: int) -> np.ndarray:
    """Ring account signup days.

    A configurable share of the accounts are aged rather than freshly minted --
    bought, farmed, or taken over. This is the single most important realism
    knob in the generator: if every ring account is three days old, account age
    alone solves the problem and nothing else in the system is being measured.
    """
    rings_cfg = cfg["rings"]
    aged_share = float(rings_cfg.get("aged_account_share", 0.0))
    lo, hi = rings_cfg.get("aged_signup_range", [60, 400])
    fresh = start_day - rng.integers(0, max(fresh_max, 1) + 1, size=n)
    aged = start_day - rng.integers(lo, hi + 1, size=n)
    return np.where(rng.random(n) < aged_share, aged, fresh)


def gen_card_testing(cfg, rng: np.random.Generator) -> pd.DataFrame:
    spec = cfg["rings"]["card_testing"]
    days = int(cfg["simulation"]["days"])
    rows = []

    for r in range(int(spec["count"])):
        ring_id = f"ring_ct_{r:03d}"
        n_dev = _ri(rng, spec["devices_per_ring"])
        n_cards = _ri(rng, spec["cards_per_ring"])
        devices = ent.make_devices(rng, n_dev)
        ips = ent.make_ips(rng, max(1, n_dev))
        n_accounts = int(rng.integers(1, 6))
        emails, roots = ent.make_emails(rng, n_accounts)
        accounts = ent.mint_ids(rng, "cust_", n_accounts)
        contacts = ent.make_contacts(rng, n_accounts)
        cards = ent.make_cards(rng, n_cards)
        addr = ent.make_addresses(rng, 1)[0]
        city = rng.choice(ent.CITIES)

        span = _ri(rng, spec["active_day_span"])
        start_day = int(rng.integers(0, max(days - span, 1)))
        burst_min = _ri(rng, spec["burst_minutes"])

        # Probes fire in a tight burst; that velocity is the whole signal.
        offsets = np.sort(rng.random(n_cards)) * burst_min * 60
        day_of = start_day + (rng.random(n_cards) * span).astype(int)
        day_of = np.clip(day_of, 0, days - 1)
        base_sec = rng.integers(0, 86400 - int(burst_min * 60) - 1)
        ts = day_of * 86400 + base_sec + offsets.astype(int)

        acct_pick = rng.integers(0, n_accounts, size=n_cards)
        dev_pick = rng.integers(0, n_dev, size=n_cards)
        lo, hi = spec["amount_range"]
        amount = rng.integers(lo, hi + 1, size=n_cards)

        failed = rng.random(n_cards) < float(spec["failure_rate"])
        fidx = rng.choice(len(ent.FAILURE_REASONS), size=n_cards, p=ent.PROBE_FAILURE_P)

        frame = pd.DataFrame(
            {
                "customer_id": accounts[acct_pick],
                "email": emails[acct_pick],
                "email_root": roots[acct_pick],
                "contact": contacts[acct_pick],
                "device_fingerprint": devices[dev_pick],
                "ip": np.asarray(ips, dtype=object)[dev_pick % len(ips)],
                "shipping_address_hash": addr,
                "card_fingerprint": cards["card_fingerprint"],
                "card_last4": cards["card_last4"],
                "card_network": cards["card_network"],
                "card_type": cards["card_type"],
                "card_issuer": cards["card_issuer"],
                "signup_day": _signup_days(cfg, rng, start_day, n_cards, 3),
                "city": city,
                "day": day_of,
                "created_at": ts,
                "amount": amount,
                "method": "card",
                "status": np.where(failed, "failed", "captured"),
                "error_code": [ent.FAILURE_REASONS[i][0] if f else None for i, f in zip(fidx, failed)],
                "error_reason": [ent.FAILURE_REASONS[i][1] if f else None for i, f in zip(fidx, failed)],
                "refunded": False,
                "disputed": ~failed & (rng.random(n_cards) < 0.35),
                "is_fraud": True,
                "ring_id": ring_id,
                "ring_type": "card_testing",
            }
        )
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(_blank_frame(0))


def gen_refund_abuse(cfg, rng: np.random.Generator) -> pd.DataFrame:
    spec = cfg["rings"]["refund_abuse"]
    honest = cfg["honest"]
    days = int(cfg["simulation"]["days"])
    rows = []

    for r in range(int(spec["count"])):
        ring_id = f"ring_ra_{r:03d}"
        n_acct = _ri(rng, spec["accounts_per_ring"])
        n_addr = _ri(rng, spec["shared_addresses"])
        n_dev = _ri(rng, spec["shared_devices"])

        addrs = ent.make_addresses(rng, n_addr)
        devices = ent.make_devices(rng, n_dev)
        ips = ent.make_ips(rng, max(1, n_dev))
        accounts = ent.mint_ids(rng, "cust_", n_acct)
        contacts = ent.make_contacts(rng, n_acct)
        city = rng.choice(ent.CITIES)

        # Identity multiplication on the cheap: most accounts are plus-aliases of
        # one or two real inboxes, so email_root collapses them back together.
        n_roots = max(1, n_acct // 6)
        _, base_roots = ent.make_emails(rng, n_roots)
        root_pick = rng.integers(0, n_roots, size=n_acct)
        roots = base_roots[root_pick]
        local_parts = np.char.partition(roots.astype(str), "@")
        aliases = ent.mint_ids(rng, "+", n_acct, n=4)
        emails = np.char.add(
            np.char.add(np.char.add(local_parts[:, 0], np.char.lower(aliases)), "@"),
            local_parts[:, 2],
        )

        span = _ri(rng, spec["active_day_span"])
        start_day = int(rng.integers(0, max(days - span, 1)))

        per_acct = rng.integers(spec["orders_per_account"][0], spec["orders_per_account"][1] + 1, size=n_acct)
        acct_pick = np.repeat(np.arange(n_acct), per_acct)
        n = int(acct_pick.size)
        if n == 0:
            continue

        day_of = np.clip(start_day + (rng.random(n) * span).astype(int), 0, days - 1)
        ts = day_of * 86400 + rng.integers(0, 86400, size=n)

        amount = np.round(
            rng.lognormal(honest["amount_lognorm_mu"], honest["amount_lognorm_sigma"] * 0.7, size=n)
            * float(spec["amount_multiplier"])
        ).astype(int) * 100
        amount = np.clip(amount, 9900, 60_000_00)

        methods = list(honest["method_mix"].keys())
        probs = np.array([honest["method_mix"][m] for m in methods], dtype=float)
        probs /= probs.sum()
        method = rng.choice(methods, size=n, p=probs)
        cards = ent.make_cards(rng, n)

        frame = pd.DataFrame(
            {
                "customer_id": accounts[acct_pick],
                "email": emails[acct_pick],
                "email_root": roots[acct_pick],
                "contact": contacts[acct_pick],
                "device_fingerprint": devices[rng.integers(0, n_dev, size=n)],
                "ip": np.asarray(ips, dtype=object)[rng.integers(0, len(ips), size=n)],
                "shipping_address_hash": addrs[rng.integers(0, n_addr, size=n)],
                "signup_day": _signup_days(cfg, rng, start_day, n, 20),
                "city": city,
                "day": day_of,
                "created_at": ts,
                "amount": amount,
                "method": method,
                "status": "captured",
                "error_code": None,
                "error_reason": None,
                "refunded": rng.random(n) < float(spec["refund_rate"]),
                "disputed": False,
                "is_fraud": True,
                "ring_id": ring_id,
                "ring_type": "refund_abuse",
                **{k: v for k, v in cards.items()},
            }
        )
        non_card = frame["method"].to_numpy() != "card"
        for col in ("card_fingerprint", "card_last4", "card_network", "card_type", "card_issuer"):
            frame.loc[non_card, col] = None
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(_blank_frame(0))


def gen_bust_out(cfg, rng: np.random.Generator) -> pd.DataFrame:
    spec = cfg["rings"]["bust_out"]
    honest = cfg["honest"]
    days = int(cfg["simulation"]["days"])
    rows = []

    for r in range(int(spec["count"])):
        ring_id = f"ring_bo_{r:03d}"
        n_acct = _ri(rng, spec["accounts_per_ring"])
        n_addr = _ri(rng, spec["shared_drop_addresses"])
        addrs = ent.make_addresses(rng, n_addr)
        accounts = ent.mint_ids(rng, "cust_", n_acct)
        emails, roots = ent.make_emails(rng, n_acct)
        contacts = ent.make_contacts(rng, n_acct)
        city = rng.choice(ent.CITIES)

        span = _ri(rng, spec["active_day_span"])
        start_day = int(rng.integers(0, max(days - span, 1)))

        n = n_acct * int(rng.integers(1, 3))
        acct_pick = rng.integers(0, n_acct, size=n)
        day_of = np.clip(start_day + (rng.random(n) * span).astype(int), 0, days - 1)
        ts = day_of * 86400 + rng.integers(0, 86400, size=n)

        amount = np.round(
            rng.lognormal(honest["amount_lognorm_mu"], honest["amount_lognorm_sigma"] * 0.95, size=n)
            * float(spec["amount_multiplier"])
        ).astype(int) * 100
        amount = np.clip(amount, 49900, 250_000_00)

        # One distinct stolen card and one distinct handset per mule account:
        # there is deliberately no card or device link to find.
        cards = ent.make_cards(rng, n)
        devices = ent.make_devices(rng, n)
        ips = ent.make_ips(rng, n)

        disputed = rng.random(n) < float(spec["dispute_rate"])
        lag_lo, lag_hi = spec["dispute_lag_days"]
        dispute_lag = rng.integers(lag_lo, lag_hi + 1, size=n)

        frame = pd.DataFrame(
            {
                "customer_id": accounts[acct_pick],
                "email": emails[acct_pick],
                "email_root": roots[acct_pick],
                "contact": contacts[acct_pick],
                "device_fingerprint": devices,
                "ip": ips,
                "shipping_address_hash": addrs[rng.integers(0, n_addr, size=n)],
                "signup_day": _signup_days(cfg, rng, start_day, n, int(spec["account_age_days_max"])),
                "city": city,
                "day": day_of,
                "created_at": ts,
                "amount": amount,
                "method": "card",
                "status": "captured",
                "error_code": None,
                "error_reason": None,
                "refunded": False,
                "disputed": disputed,
                "dispute_lag_days": dispute_lag,
                "is_fraud": True,
                "ring_id": ring_id,
                "ring_type": "bust_out",
                **{k: v for k, v in cards.items()},
            }
        )
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(_blank_frame(0))
