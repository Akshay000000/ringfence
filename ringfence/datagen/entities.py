"""Identifier pools and Razorpay-shaped record minting.

Nothing in this module produces a usable payment credential. Card
"fingerprints" are opaque random tokens; last4 digits are random and are not
derived from any real BIN range. The generator exists solely to create a
labelled evaluation corpus.
"""
from __future__ import annotations

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

CARD_NETWORKS = ("Visa", "MasterCard", "RuPay", "American Express")
CARD_NETWORK_P = (0.42, 0.34, 0.21, 0.03)
CARD_TYPES = ("debit", "credit")
CARD_TYPE_P = (0.63, 0.37)
ISSUERS = ("HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "IDFB", "PUNB", "YESB", "BARB", "INDB")

# Razorpay payment.error_code / error_reason vocabulary for failed payments.
FAILURE_REASONS = (
    ("BAD_REQUEST_ERROR", "payment_failed_due_to_insufficient_funds"),
    ("BAD_REQUEST_ERROR", "incorrect_card_details"),
    ("BAD_REQUEST_ERROR", "payment_authentication_failed"),
    ("GATEWAY_ERROR", "payment_failed_at_bank"),
    ("BAD_REQUEST_ERROR", "card_declined_by_issuer"),
    ("BAD_REQUEST_ERROR", "payment_timed_out"),
)
# Failure mix differs sharply between honest traffic and card testing: probing
# a stolen card list produces declines and CVV failures, not timeouts.
HONEST_FAILURE_P = (0.31, 0.13, 0.14, 0.24, 0.10, 0.08)
PROBE_FAILURE_P = (0.09, 0.34, 0.19, 0.06, 0.31, 0.01)

EMAIL_DOMAINS = ("gmail.com", "outlook.com", "yahoo.in", "protonmail.com", "hotmail.com", "rediffmail.com")
EMAIL_DOMAIN_P = (0.64, 0.13, 0.11, 0.04, 0.05, 0.03)

CITIES = (
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Indore", "Bhopal", "Surat",
)


def rand_token(rng: np.random.Generator, n: int = 14) -> str:
    idx = rng.integers(0, len(ALPHABET), size=n)
    return "".join(ALPHABET[i] for i in idx)


def mint_ids(rng: np.random.Generator, prefix: str, count: int, n: int = 14) -> np.ndarray:
    """Vectorised minting of Razorpay-style prefixed identifiers."""
    idx = rng.integers(0, len(ALPHABET), size=(count, n))
    chars = np.array(list(ALPHABET))
    bodies = chars[idx].view(f"<U{n}").reshape(count)
    return np.char.add(prefix, bodies)


def make_emails(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (display_email, email_root).

    email_root normalises gmail dot- and plus-aliasing, which is the single most
    common cheap identity-multiplication trick in refund abuse. Linking on the
    raw string would miss it entirely; linking on the root catches it.
    """
    locals_ = mint_ids(rng, "", count, n=10)
    locals_ = np.char.lower(locals_)
    domains = rng.choice(EMAIL_DOMAINS, size=count, p=EMAIL_DOMAIN_P)
    roots = np.char.add(np.char.add(locals_, "@"), domains)

    display = roots.copy()
    # ~9% of accounts use an alias form of a root they already own.
    alias_mask = rng.random(count) < 0.09
    n_alias = int(alias_mask.sum())
    if n_alias:
        suffixes = mint_ids(rng, "+", n_alias, n=4)
        suffixes = np.char.lower(suffixes)
        aliased = np.char.add(np.char.add(locals_[alias_mask], suffixes), "@")
        display[alias_mask] = np.char.add(aliased, domains[alias_mask])
    return display, roots


def make_contacts(rng: np.random.Generator, count: int) -> np.ndarray:
    numbers = rng.integers(6000000000, 9999999999, size=count)
    return np.char.add("+91", numbers.astype(str))


def make_cards(rng: np.random.Generator, count: int) -> dict[str, np.ndarray]:
    return {
        "card_fingerprint": mint_ids(rng, "cfp_", count),
        # Zero-padded so it looks like the field it stands in for. Random
        # digits with no BIN structure -- this is a display artefact, never a
        # link key, and nothing in the pipeline joins on it.
        "card_last4": np.char.zfill(rng.integers(0, 10000, size=count).astype(str), 4),
        "card_network": rng.choice(CARD_NETWORKS, size=count, p=CARD_NETWORK_P),
        "card_type": rng.choice(CARD_TYPES, size=count, p=CARD_TYPE_P),
        "card_issuer": rng.choice(ISSUERS, size=count),
    }


def make_ips(rng: np.random.Generator, count: int) -> np.ndarray:
    octets = rng.integers(1, 255, size=(count, 4))
    return np.array(
        [f"{a}.{b}.{c}.{d}" for a, b, c, d in octets], dtype=object
    )


def make_addresses(rng: np.random.Generator, count: int) -> np.ndarray:
    return mint_ids(rng, "addr_", count, n=12)


def make_devices(rng: np.random.Generator, count: int) -> np.ndarray:
    return mint_ids(rng, "dev_", count, n=16)
