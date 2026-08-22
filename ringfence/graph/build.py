"""Identity graph construction with IDF-weighted edges.

The whole difficulty of ring detection is that shared identifiers are common
and mostly innocent. This module turns "shared" into a graded quantity.

An identifier's evidential strength is:

    w(v) = prior[type] * log(N / df(v)) / log(N)

where df(v) is the number of distinct customers that touched value v inside the
window and N is the customer count in that window. A card fingerprint seen on 2
accounts approaches prior weight. An IP seen on 400 accounts approaches zero.
Hub identifiers above `max_identifier_degree` are dropped outright rather than
down-weighted, because a carrier NAT block carries no information at any weight
and keeping it turns the customer projection into a dense blob.

Customer-customer edge weight is the sum of the weights of the identifiers the
pair shares, so agreeing on three weak things can still add up to one strong
thing -- which is exactly how a real analyst reasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Snapshot:
    """An as-of-day view of the identity graph."""

    as_of_day: int
    customers: np.ndarray                      # customer_id, index-aligned
    adjacency: sp.csr_matrix                   # weighted customer x customer
    identifier_weight: dict[str, float] = field(default_factory=dict)
    edge_provenance: dict[tuple[int, int], list[str]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.customers)


def _link_specs(link_types: dict) -> dict[str, dict]:
    """Accept both the rich {prior, max_df} form and a bare float prior."""
    specs = {}
    for name, spec in link_types.items():
        if isinstance(spec, dict):
            specs[name] = {"prior": float(spec.get("prior", 1.0)),
                           "max_df": int(spec.get("max_df", 10**9))}
        else:
            specs[name] = {"prior": float(spec), "max_df": 10**9}
    return specs


def _identifier_frame(window: pd.DataFrame, link_types: dict) -> pd.DataFrame:
    """Long-form (customer, link_type, value) with nulls dropped."""
    parts = []
    for col in link_types:
        if col not in window.columns:
            continue
        sub = window.loc[:, ["customer_id", col]].dropna()
        sub = sub[sub[col].astype(str).str.len() > 0]
        if sub.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "customer_id": sub["customer_id"].to_numpy(),
                    "link_type": col,
                    # Namespace the value so an IP can never collide with an address.
                    "value": col + "::" + sub[col].astype(str).to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["customer_id", "link_type", "value"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def precompute_pairs(payments: pd.DataFrame, cfg) -> pd.DataFrame:
    """Melt every payment into its (customer, link_type, value) rows once.

    The snapshot loop runs 150 times; deriving this table inside it burned ~20%
    of total runtime re-doing identical work. Computed once, each snapshot is
    then a day-range slice.
    """
    specs = _link_specs(dict(cfg["graph"]["link_types"]))
    parts = []
    for col in specs:
        if col not in payments.columns:
            continue
        sub = payments.loc[:, ["customer_id", "day", col]].dropna()
        sub = sub[sub[col].astype(str).str.len() > 0]
        if sub.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "customer_id": sub["customer_id"].to_numpy(),
                    "day": sub["day"].to_numpy(),
                    "link_type": col,
                    "value": col + "::" + sub[col].astype(str).to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["customer_id", "day", "link_type", "value"])
    return pd.concat(parts, ignore_index=True)


def build_snapshot(window: pd.DataFrame, cfg, as_of_day: int, pairs: pd.DataFrame | None = None) -> Snapshot:
    gcfg = cfg["graph"]
    specs = _link_specs(dict(gcfg["link_types"]))
    link_types = {k: v["prior"] for k, v in specs.items()}
    max_degree = int(gcfg["max_identifier_degree"])
    min_weight = float(gcfg["idf_min_weight"])

    if pairs is None:
        pairs = _identifier_frame(window, link_types)
    else:
        pairs = pairs.loc[:, ["customer_id", "link_type", "value"]].drop_duplicates()
    customers = np.sort(window["customer_id"].dropna().unique())
    if pairs.empty or len(customers) == 0:
        return Snapshot(as_of_day, customers, sp.csr_matrix((len(customers), len(customers))))

    cust_index = pd.Series(np.arange(len(customers)), index=customers)
    pairs = pairs[pairs["customer_id"].isin(cust_index.index)]

    # df(v): distinct customers per identifier value.
    df_counts = pairs.groupby("value")["customer_id"].nunique()

    # Hub pruning, per link type. The cap for an IP is not the cap for a card:
    # 20 accounts on one IP is a coffee shop, 20 accounts on one card is a ring.
    value_type_all = pairs.drop_duplicates("value").set_index("value")["link_type"]
    per_type_cap = value_type_all.map(lambda t: specs[t]["max_df"]).astype(float)
    df_aligned = df_counts.reindex(per_type_cap.index)
    keep_values = df_aligned[
        (df_aligned >= 2)
        & (df_aligned <= per_type_cap)
        & (df_aligned <= max_degree)
    ].index
    pairs = pairs[pairs["value"].isin(keep_values)]
    if pairs.empty:
        return Snapshot(as_of_day, customers, sp.csr_matrix((len(customers), len(customers))))

    n_customers = float(len(customers))
    df_kept = df_counts.loc[keep_values].astype(float)
    idf = np.log(n_customers / df_kept) / np.log(n_customers)
    idf = idf.clip(lower=0.0)

    value_type = pairs.drop_duplicates("value").set_index("value")["link_type"]
    prior = value_type.map(link_types).astype(float)
    weight = (idf * prior).rename("weight")

    weight = weight[weight >= min_weight]
    pairs = pairs[pairs["value"].isin(weight.index)]
    if pairs.empty:
        return Snapshot(as_of_day, customers, sp.csr_matrix((len(customers), len(customers))))

    value_index = pd.Series(np.arange(len(weight)), index=weight.index)
    rows = cust_index.loc[pairs["customer_id"]].to_numpy()
    cols = value_index.loc[pairs["value"]].to_numpy()
    data = np.ones(len(pairs), dtype=np.float32)

    incidence = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(customers), len(weight)), dtype=np.float32
    )
    weight_diag = sp.diags(weight.to_numpy().astype(np.float32))

    # A = C W C^T -> customer-customer weight = sum of shared identifier weights.
    adjacency = (incidence @ weight_diag @ incidence.T).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    adjacency.data[adjacency.data < min_weight] = 0.0
    adjacency.eliminate_zeros()

    return Snapshot(
        as_of_day=as_of_day,
        customers=customers,
        adjacency=adjacency,
        identifier_weight=weight.to_dict(),
    )
