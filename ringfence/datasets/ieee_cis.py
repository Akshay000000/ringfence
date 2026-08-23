"""IEEE-CIS Fraud Detection (Vesta) adapter — the real-data validation.

Source: https://www.kaggle.com/competitions/ieee-fraud-detection
590,540 real card-not-present transactions over ~182 days, 3.5% fraud, with a
genuine identity surface: card hashes, billing region, email domains, device
strings and browser/screen fingerprints.

Why this dataset and not one of the tidier public fraud sets: almost every other
option (the ULB credit-card set, for instance) ships PCA components with the
entity columns stripped out, so there is nothing to build a graph *from*. Here
the linking columns survive — and notably, the strongest published solutions to
this competition all turned on reconstructing a client identity from
`card1 + addr1 + (day - D1)`. That is entity resolution by another name, which
makes this the fairest available real test of RingFence's central claim.

## The client identity

There is no customer ID in the data. The standard reconstruction:

    D1  = days elapsed since the first transaction on this card
    D1n = day - D1                      -> the card's first-seen day, constant
                                           for the life of that card
    uid = card1 · addr1 · D1n

Two transactions sharing all three are, with high probability, the same client.
This is a *heuristic*, not a label, and it matters that it is used only to form
`customer_id` — the identity graph then links those clients by the identifier
columns exactly as it does on synthetic data. The graph is not handed the answer.

## What this dataset cannot supply

Stated plainly, because these features go inert rather than wrong:

  * no authorisation outcome — every row is a completed transaction, so
    decline-rate features are constant and carry no signal;
  * no refunds or disputes as events — `isFraud` is the given label, so the
    label-maturity model does not apply and is switched off;
  * no phone or IP;
  * **no ring labels at all**. Per-archetype recall and novel-ring reporting are
    unavailable here and are reported as unavailable, never approximated.

What remains is the claim actually under test: do graph features beat a strong
tabular baseline on real fraud, under a temporal split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

# The 25 transaction columns RingFence needs, out of 394. Everything else is
# Vesta's own engineered V-block, which would hand the model pre-computed
# aggregations and muddy the ablation -- the question is whether *our* graph
# features add signal, not whether Vesta's do.
TRANSACTION_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C5", "C13", "C14",
    "D1", "D2", "D10", "D15",
]

IDENTITY_COLS = [
    "TransactionID", "DeviceType", "DeviceInfo",
    "id_30", "id_31", "id_33",
]

PASSTHROUGH_NUMERIC = ["C1", "C2", "C5", "C13", "C14", "D2", "D10", "D15", "dist1"]


# The Kaggle download is a 683 MB CSV. A slimmed, gzipped export of just the
# columns below is ~13 MB and is what actually travels between machines, so both
# forms are accepted.
_VARIANTS = ("{stem}.csv", "{stem}_slim.csv", "{stem}.csv.gz", "{stem}_slim.csv.gz")


def _first_existing(raw: Path, stem: str) -> Path | None:
    for pattern in _VARIANTS:
        candidate = raw / pattern.format(stem=stem)
        if candidate.exists():
            return candidate
    # Kaggle sometimes unpacks each CSV into a folder of the same name.
    nested = raw / f"{stem}.csv" / f"{stem}.csv"
    return nested if nested.exists() else None


def _read(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read only the columns that exist -- the slimmed export may drop some."""
    header = pd.read_csv(path, nrows=0)
    present = [c for c in usecols if c in header.columns]
    return pd.read_csv(path, usecols=present, low_memory=False)


def _client_uid(df: pd.DataFrame) -> pd.Series:
    """card1 · addr1 · (day - D1) -- the reconstructed client identity."""
    day = df["day"].to_numpy()
    d1 = pd.to_numeric(df.get("D1"), errors="coerce").to_numpy()
    first_seen = np.where(np.isfinite(d1), day - d1, np.nan)

    card1 = pd.to_numeric(df["card1"], errors="coerce").astype("Int64").astype("string").fillna("na")
    addr1 = pd.to_numeric(df["addr1"], errors="coerce").astype("Int64").astype("string").fillna("na")
    seen = pd.Series(first_seen).map(lambda v: "na" if not np.isfinite(v) else str(int(v)))
    uid = ("uid_" + card1 + "_" + addr1 + "_" + pd.Series(seen.to_numpy(), index=df.index, dtype="string")).to_numpy()

    # Rows with no card1 cannot be attributed to a client; give each its own
    # identity rather than collapsing them all into one giant fake customer.
    orphan = df["card1"].isna().to_numpy()
    uid = np.where(orphan, "uid_solo_" + df["TransactionID"].astype(str), uid)
    return pd.Series(uid, index=df.index)


def _device_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Device string + OS + browser + screen resolution, when identity is present.

    Only ~24% of transactions carry identity data, so most rows have no device
    to link on. That is a property of the data, not a bug: the graph links on
    what exists and the coverage figure is reported honestly.
    """
    parts = []
    for col in ("DeviceInfo", "id_30", "id_31", "id_33"):
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
    if not parts:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    joined = parts[0]
    for extra in parts[1:]:
        joined = joined + "|" + extra
    joined = joined.str.strip("|")
    # An all-empty join means no identity record at all.
    return joined.where(joined.str.replace("|", "", regex=False).str.len() > 0, pd.NA)


def load(cfg, raw_dir: str | Path | None = None) -> pd.DataFrame:
    raw = Path(raw_dir or cfg.get_path("dataset.raw_dir", "data/raw/ieee"))
    if not raw.is_absolute():
        from ..config import REPO_ROOT

        raw = REPO_ROOT / raw

    tx_path = _first_existing(raw, "train_transaction")
    if tx_path is None:
        raise FileNotFoundError(
            f"IEEE-CIS transactions not found under {raw}. Download "
            "train_transaction.csv and train_identity.csv from "
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
        )
    tx = _read(tx_path, TRANSACTION_COLS)

    id_path = _first_existing(raw, "train_identity")
    if id_path is not None:
        ident = _read(id_path, IDENTITY_COLS)
        tx = tx.merge(ident, on="TransactionID", how="left")

    tx["day"] = (tx["TransactionDT"] // 86400).astype(int)
    tx["day"] = tx["day"] - tx["day"].min()

    out = pd.DataFrame(
        {
            "payment_id": "pay_" + tx["TransactionID"].astype(str),
            "customer_id": _client_uid(tx),
            "day": tx["day"],
            "created_at": tx["TransactionDT"].astype("int64"),
            # Stored in minor units to match the synthetic corpus, so the rupee
            # cost model reads the same field without a special case.
            "amount": (tx["TransactionAmt"].astype(float) * 100).round(),
            "is_fraud": tx["isFraud"].astype(int).astype(bool),
        }
    )

    def as_key(series: pd.Series) -> pd.Series:
        """Integer-ish column -> stable string key, with NA as a literal 'na'.

        DataFrame.astype("Int64").astype(str) leaves float NaNs in place on a
        multi-column frame, which then blows up inside a string join. Doing it
        one column at a time and filling explicitly is both correct and legible.
        """
        return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string").fillna("na")

    card_cols = [c for c in ("card1", "card2", "card3", "card5") if c in tx.columns]
    fingerprint = as_key(tx[card_cols[0]])
    for col in card_cols[1:]:
        fingerprint = fingerprint + "_" + as_key(tx[col])
    out["card_fingerprint"] = ("card_" + fingerprint).where(tx["card1"].notna())

    out["device_fingerprint"] = _device_fingerprint(tx)

    addr = as_key(tx["addr1"]) + "_" + as_key(tx["addr2"])
    out["shipping_address_hash"] = ("addr_" + addr).where(tx["addr1"].notna())

    # Email *domain*, not an address -- so this is a far weaker link than the
    # synthetic corpus's alias-normalised inbox. gmail.com joins a third of the
    # dataset; the IDF weighting and the per-type hub cap exist precisely to
    # stop that from becoming an edge.
    out["email_root"] = tx["P_emaildomain"].astype("string")
    out["contact"] = None   # not present in this dataset
    out["ip"] = None        # not present in this dataset

    out["method"] = tx["ProductCD"].astype("string").fillna("unknown")
    out["card_network"] = tx.get("card4", pd.Series(index=tx.index, dtype="string")).astype("string")
    out["card_type"] = tx.get("card6", pd.Series(index=tx.index, dtype="string")).astype("string")
    out["card_issuer"] = as_key(tx["card5"]) if "card5" in tx.columns else None
    out["city"] = as_key(tx["addr1"])
    out["status"] = "captured"  # no authorisation outcome in this dataset

    d1 = pd.to_numeric(tx.get("D1"), errors="coerce")
    out["account_age_days"] = d1.fillna(0).clip(lower=0).astype(int)
    out["signup_day"] = out["day"] - out["account_age_days"]

    for col in PASSTHROUGH_NUMERIC:
        if col in tx.columns:
            out[f"src_{col}"] = pd.to_numeric(tx[col], errors="coerce")

    out = out.sort_values("created_at", kind="stable").reset_index(drop=True)
    return schema.finalise(out, cfg, "ieee_cis")


def coverage(payments: pd.DataFrame) -> pd.DataFrame:
    """How much of each link type actually exists. Published, not assumed."""
    rows = []
    for col in schema.LINK:
        series = payments[col]
        present = series.notna() & (series.astype("string").str.len() > 0)
        rows.append(
            {
                "link_type": col,
                "rows_present": int(present.sum()),
                "coverage_%": round(100 * float(present.mean()), 2),
                "distinct_values": int(series[present].nunique()) if present.any() else 0,
            }
        )
    return pd.DataFrame(rows)
