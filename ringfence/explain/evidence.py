"""The evidence packet behind an alert.

A score is not actionable on its own. What an analyst -- or a merchant
disputing a block -- needs is the specific claim: *these* accounts, linked by
*this* identifier, behaving *this* way, as of *this* date.

So every alert can produce the subgraph it came from: the cluster members, the
identifiers that tie them together with the weight each contributed, and how
this particular payment attached to the cluster (through its own history, or
inherited through an identifier it touched).

Identifier values are masked. The packet is meant to be pasted into a ticket or
shown to a merchant, and a full device fingerprint or address hash is not
something that should travel in either.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..graph.build import build_snapshot, precompute_pairs
from ..graph.rings import cluster_behaviour, cluster_cohesion, detect_clusters, score_clusters

LINK_LABELS = {
    "card_fingerprint": "card",
    "device_fingerprint": "device",
    "shipping_address_hash": "shipping address",
    "contact": "phone",
    "email_root": "email (alias-normalised)",
    "ip": "IP address",
}


def mask(value: str, keep: int = 3) -> str:
    """Show enough to correlate across tickets, not enough to identify."""
    text = str(value)
    if "::" in text:
        text = text.split("::", 1)[1]
    if len(text) <= keep * 2 + 1:
        return text[:keep] + "…"
    return f"{text[:keep]}…{text[-keep:]}"


@dataclass
class Evidence:
    payment_id: str
    cluster: str
    as_of_day: int
    members: pd.DataFrame
    shared_identifiers: pd.DataFrame
    edges: pd.DataFrame
    attachment: str

    def to_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "cluster": self.cluster,
            "as_of_day": int(self.as_of_day),
            "attachment": self.attachment,
            "member_count": int(len(self.members)),
            "members": self.members.to_dict(orient="records"),
            "shared_identifiers": self.shared_identifiers.to_dict(orient="records"),
            "edges": self.edges.to_dict(orient="records"),
        }

    def render(self, max_members: int = 8, max_links: int = 6) -> str:
        lines = [
            f"Evidence for {self.payment_id}",
            f"  cluster {self.cluster} as of day {self.as_of_day} "
            f"({len(self.members)} linked accounts)",
            f"  attached: {self.attachment}",
        ]
        if not self.shared_identifiers.empty:
            lines.append("  linked by:")
            for _, link in self.shared_identifiers.head(max_links).iterrows():
                lines.append(
                    f"    {link['link_label']:<26} {link['masked']:<14} "
                    f"{int(link['accounts'])} accounts   weight {link['weight']:.2f}"
                )
        if not self.members.empty:
            lines.append("  accounts:")
            for _, member in self.members.head(max_members).iterrows():
                lines.append(
                    f"    {mask(member['customer_id'], 5):<14} "
                    f"{int(member['payments']):>3} payments  "
                    f"Rs {member['amount_inr']:>10,.0f}  "
                    f"age {int(member['account_age_days']):>4}d  "
                    f"first seen day {int(member['first_day'])}"
                )
            if len(self.members) > max_members:
                lines.append(f"    … and {len(self.members) - max_members} more")
        return "\n".join(lines)


class EvidenceBuilder:
    """Rebuilds the graph snapshot an alert was scored against.

    Snapshots are cached per day: an analyst working a queue looks at many
    alerts from the same few days, and rebuilding costs ~2s each.
    """

    def __init__(self, payments: pd.DataFrame, cfg):
        self.cfg = cfg
        self.payments = payments.sort_values("day", kind="stable")
        self._pairs = precompute_pairs(self.payments, cfg)
        self._pair_day = self._pairs["day"].to_numpy()
        self._day = self.payments["day"].to_numpy()
        self._window_days = int(cfg["graph"]["window_days"])
        self._seed = int(cfg["seed"])
        self._cache: dict[int, tuple] = {}

    def snapshot_for(self, as_of_day: int):
        if as_of_day in self._cache:
            return self._cache[as_of_day]
        lo = max(0, as_of_day - self._window_days)
        wmask = (self._day >= lo) & (self._day < as_of_day)
        window = self.payments[wmask]
        pairs = self._pairs[(self._pair_day >= lo) & (self._pair_day < as_of_day)]
        snapshot = build_snapshot(window, self.cfg, as_of_day, pairs=pairs)
        clusters = detect_clusters(snapshot, self.cfg, seed=self._seed)
        scored = score_clusters(
            cluster_cohesion(snapshot, clusters), cluster_behaviour(window, clusters)
        )
        result = (snapshot, clusters, window, pairs, scored)
        if len(self._cache) > 12:
            self._cache.pop(next(iter(self._cache)))
        self._cache[as_of_day] = result
        return result

    def evidence_for(self, payment: pd.Series, cluster: str | None = None) -> Evidence | None:
        as_of_day = int(payment["day"])
        snapshot, clusters, window, pairs, _scored = self.snapshot_for(as_of_day)

        label = cluster if cluster else clusters.get(payment["customer_id"], "")
        attachment = "own prior activity in this cluster"
        if not label:
            # The payment's account is new; find the cluster via its identifiers.
            for col in LINK_LABELS:
                value = payment.get(col)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                key = f"{col}::{value}"
                touching = pairs.loc[pairs["value"] == key, "customer_id"]
                labels = [clusters.get(c, "") for c in touching]
                labels = [x for x in labels if x]
                if labels:
                    label = pd.Series(labels).mode().iloc[0]
                    attachment = (
                        f"new account; matched on shared {LINK_LABELS[col]} "
                        f"{mask(str(value))}"
                    )
                    break
        if not label:
            return None

        member_ids = clusters[clusters == label].index.to_numpy()
        member_rows = window[window["customer_id"].isin(member_ids)]

        members = (
            member_rows.groupby("customer_id")
            .agg(
                payments=("payment_id", "size"),
                amount_inr=("amount", lambda s: s.sum() / 100),
                account_age_days=("account_age_days", "max"),
                first_day=("day", "min"),
                declines=("status", lambda s: int((s == "failed").sum())),
            )
            .reset_index()
            .sort_values("amount_inr", ascending=False)
        )

        member_pairs = pairs[pairs["customer_id"].isin(member_ids)]
        counts = member_pairs.groupby(["link_type", "value"])["customer_id"].nunique()
        counts = counts[counts >= 2].sort_values(ascending=False)
        links = counts.reset_index().rename(columns={"customer_id": "accounts"})
        if not links.empty:
            weights = snapshot.identifier_weight
            links["weight"] = links["value"].map(weights).fillna(0.0)
            links["link_label"] = links["link_type"].map(LINK_LABELS).fillna(links["link_type"])
            links["masked"] = links["value"].map(mask)
            links = links.sort_values(["weight", "accounts"], ascending=False)
            links = links[["link_label", "masked", "accounts", "weight"]]
        else:
            links = pd.DataFrame(columns=["link_label", "masked", "accounts", "weight"])

        index = pd.Series(np.arange(snapshot.n), index=snapshot.customers)
        positions = index.reindex(member_ids).dropna().astype(int).to_numpy()
        edges = pd.DataFrame(columns=["a", "b", "weight"])
        if len(positions) >= 2:
            sub = snapshot.adjacency[positions][:, positions].tocoo()
            keep = sub.row < sub.col
            if keep.any():
                edges = pd.DataFrame(
                    {
                        "a": [mask(member_ids[i], 5) for i in sub.row[keep]],
                        "b": [mask(member_ids[j], 5) for j in sub.col[keep]],
                        "weight": np.round(sub.data[keep], 3),
                    }
                ).sort_values("weight", ascending=False)

        return Evidence(
            payment_id=str(payment["payment_id"]),
            cluster=str(label),
            as_of_day=as_of_day,
            members=members,
            shared_identifiers=links,
            edges=edges,
            attachment=attachment,
        )
