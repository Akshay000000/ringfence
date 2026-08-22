"""Community detection and ring risk scoring over an identity snapshot.

Two-stage, for speed and for sanity:

1. Weighted connected components (scipy) split the graph into candidate blobs.
   Most are pairs and triples -- a couple sharing a card. Cheap to compute and
   it throws away 95% of the work before the expensive step.
2. Louvain runs only inside components large enough to plausibly hide a ring.
   A 300-account component behind one apartment block needs splitting into its
   real sub-structures; a 2-account component does not.

The resulting cluster is *not* a verdict. It is a feature. Scoring it produces
a set of numbers -- cohesion, tenure, velocity, card fan-out -- that the model
downstream is free to ignore if they turn out not to predict anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .build import Snapshot

# Components smaller than this are kept whole. Louvain on a 6-node component
# costs more than the structure it recovers, and 90% of components are couples
# sharing a card.
LOUVAIN_MIN_COMPONENT = 14
# Components at or above this edge density are kept whole rather than split.
DENSE_COMPONENT = 0.40


def _component_density(adj: sp.csr_matrix, nodes: np.ndarray) -> float:
    k = len(nodes)
    if k < 2:
        return 0.0
    sub = adj[nodes][:, nodes]
    edges = sub.nnz / 2.0
    return float(edges / (k * (k - 1) / 2.0))


def _louvain_split(adj: sp.csr_matrix, nodes: np.ndarray, resolution: float, seed: int) -> list[np.ndarray]:
    import networkx as nx

    sub = adj[nodes][:, nodes].tocoo()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    mask = sub.row < sub.col
    graph.add_weighted_edges_from(
        zip(sub.row[mask].tolist(), sub.col[mask].tolist(), sub.data[mask].tolist())
    )
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )
    return [nodes[np.fromiter(c, dtype=int)] for c in communities if len(c) > 0]


def detect_clusters(snapshot: Snapshot, cfg, seed: int = 0) -> pd.Series:
    """Return a Series mapping customer_id -> cluster label ('' when isolated)."""
    adj = snapshot.adjacency
    n = snapshot.n
    if n == 0 or adj.nnz == 0:
        return pd.Series("", index=snapshot.customers, dtype=object)

    n_comp, comp_labels = connected_components(adj, directed=False)
    resolution = float(cfg["graph"]["louvain_resolution"])

    labels = np.full(n, -1, dtype=int)
    next_label = 0
    order = np.argsort(comp_labels, kind="stable")
    boundaries = np.searchsorted(comp_labels[order], np.arange(n_comp + 1))

    for ci in range(n_comp):
        members = order[boundaries[ci] : boundaries[ci + 1]]
        if len(members) < 2:
            continue
        if len(members) < LOUVAIN_MIN_COMPONENT or _component_density(adj, members) >= DENSE_COMPONENT:
            # A dense component is already one thing. Splitting a near-clique
            # produces arbitrary halves and destroys exactly the structure we
            # are looking for.
            labels[members] = next_label
            next_label += 1
            continue
        for part in _louvain_split(adj, members, resolution, seed):
            if len(part) >= 2:
                labels[part] = next_label
                next_label += 1

    out = np.where(labels >= 0, np.char.add("cl_", labels.astype(str)), "")
    return pd.Series(out, index=snapshot.customers, dtype=object)


def cluster_cohesion(snapshot: Snapshot, clusters: pd.Series) -> pd.DataFrame:
    """Structural statistics per cluster: size, density, mean/max edge weight."""
    adj = snapshot.adjacency.tocoo()
    idx = pd.Series(np.arange(snapshot.n), index=snapshot.customers)
    label_by_pos = clusters.reindex(snapshot.customers).to_numpy()

    mask = (adj.row < adj.col)
    r, c, w = adj.row[mask], adj.col[mask], adj.data[mask]
    same = label_by_pos[r] == label_by_pos[c]
    internal = pd.DataFrame(
        {"cluster": label_by_pos[r][same], "weight": w[same]}
    )
    internal = internal[internal["cluster"] != ""]

    sizes = pd.Series(label_by_pos).value_counts()
    sizes = sizes[sizes.index != ""]

    if internal.empty:
        return pd.DataFrame(
            columns=["cluster", "size", "edges", "density", "mean_edge_weight", "max_edge_weight"]
        )

    agg = internal.groupby("cluster")["weight"].agg(
        edges="size", mean_edge_weight="mean", max_edge_weight="max", total_weight="sum"
    )
    agg["size"] = sizes.reindex(agg.index).fillna(0).astype(int)
    possible = agg["size"] * (agg["size"] - 1) / 2
    agg["density"] = (agg["edges"] / possible.replace(0, np.nan)).fillna(0.0)
    _ = idx  # kept for symmetry with explain-layer lookups
    return agg.reset_index().rename(columns={"index": "cluster"})


def cluster_behaviour(window: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Behavioural statistics per cluster, computed only from in-window payments.

    Nothing here touches is_fraud, ring_id, refunded-in-future, or any other
    label-bearing column. Refund and dispute rates use only events that had
    already landed inside the window.
    """
    df = window.copy()
    df["cluster"] = df["customer_id"].map(clusters).fillna("")
    df = df[df["cluster"] != ""]
    if df.empty:
        return pd.DataFrame(columns=["cluster"])

    captured = df["status"].to_numpy() == "captured"
    df["_captured"] = captured
    df["_failed"] = ~captured
    day_hi = df["day"].max()

    # First in-window appearance of each account, used for cohort synchrony.
    first_seen = df.groupby(["cluster", "customer_id"]).agg(
        first_day=("day", "min"), signup=("signup_day", "first")
    ).reset_index()
    cohort = first_seen.groupby("cluster").agg(
        cl_signup_std=("signup", "std"),
        cl_signup_span=("signup", lambda s: float(s.max() - s.min())),
        cl_first_seen_std=("first_day", "std"),
        cl_first_seen_span=("first_day", lambda s: float(s.max() - s.min())),
    )

    grouped = df.groupby("cluster")
    out = pd.DataFrame(
        {
            "cl_payments": grouped.size(),
            "cl_customers": grouped["customer_id"].nunique(),
            "cl_distinct_cards": grouped["card_fingerprint"].nunique(),
            "cl_distinct_devices": grouped["device_fingerprint"].nunique(),
            "cl_distinct_addresses": grouped["shipping_address_hash"].nunique(),
            "cl_distinct_ips": grouped["ip"].nunique(),
            "cl_failure_rate": grouped["_failed"].mean(),
            "cl_amount_mean": grouped["amount"].mean(),
            "cl_amount_median": grouped["amount"].median(),
            "cl_amount_cv": grouped["amount"].std() / grouped["amount"].mean().replace(0, np.nan),
            "cl_account_age_mean": grouped["account_age_days"].mean(),
            "cl_account_age_min": grouped["account_age_days"].min(),
            "cl_active_days": grouped["day"].nunique(),
            "cl_day_span": grouped["day"].max() - grouped["day"].min() + 1,
            "cl_last_seen_lag": day_hi - grouped["day"].max(),
            "cl_distinct_issuers": grouped["card_issuer"].nunique(),
            "cl_new_account_share": grouped["account_age_days"].apply(lambda s: float((s < 30).mean())),
        }
    )
    out = out.join(cohort, how="left")

    out["cl_cards_per_device"] = out["cl_distinct_cards"] / out["cl_distinct_devices"].replace(0, np.nan)
    out["cl_customers_per_address"] = out["cl_customers"] / out["cl_distinct_addresses"].replace(0, np.nan)
    out["cl_payments_per_active_day"] = out["cl_payments"] / out["cl_active_days"].replace(0, np.nan)
    out["cl_burstiness"] = out["cl_active_days"] / out["cl_day_span"].replace(0, np.nan)

    # Cohort synchrony. This is how a human analyst separates a drop address
    # from an apartment block: a building's residents opened their accounts over
    # years, a mule cohort opened theirs the same week. Without it the two
    # structures are indistinguishable in the graph, and the graph actively
    # *hurt* bust-out recall (0.800 -> 0.633, measured; see FINDINGS.md).
    out["cl_signup_synchrony"] = 1.0 / (1.0 + out["cl_signup_std"].fillna(400.0))
    out["cl_first_seen_synchrony"] = 1.0 / (1.0 + out["cl_first_seen_std"].fillna(60.0))
    out["cl_signup_span_per_customer"] = (
        out["cl_signup_span"] / out["cl_customers"].replace(0, np.nan)
    )
    out["cl_issuers_per_customer"] = (
        out["cl_distinct_issuers"] / out["cl_customers"].replace(0, np.nan)
    )
    return out.reset_index().rename(columns={"index": "cluster"})


def score_clusters(cohesion: pd.DataFrame, behaviour: pd.DataFrame) -> pd.DataFrame:
    """An interpretable 0-1 ring-risk prior, used as one feature and as the
    ordering for the analyst queue. Deliberately hand-specified rather than
    learned, so it stays explainable and cannot silently absorb label leakage.
    """
    if cohesion.empty or behaviour.empty:
        return pd.DataFrame(columns=["cluster", "cl_ring_risk"])

    merged = cohesion.merge(behaviour, on="cluster", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["cluster", "cl_ring_risk"])

    def unit(series: pd.Series, lo: float, hi: float) -> pd.Series:
        return ((series - lo) / (hi - lo)).clip(0, 1).fillna(0)

    signals = pd.DataFrame(
        {
            # Tight, strongly-linked cluster of non-trivial size.
            "cohesion": unit(merged["mean_edge_weight"], 0.05, 0.6) * unit(merged["density"], 0.2, 1.0),
            "scale": unit(np.log1p(merged["cl_customers"]), np.log1p(3), np.log1p(40)),
            # Card fan-out per device is the card-testing tell.
            "card_fanout": unit(merged["cl_cards_per_device"], 3, 60),
            # Declines cluster hard when a stolen list is being probed.
            "decline": unit(merged["cl_failure_rate"], 0.25, 0.9),
            # Many fresh accounts is the bust-out tell.
            "fresh": unit(30 - merged["cl_account_age_mean"], 5, 30),
            # Many accounts behind one delivery point is the refund-abuse tell.
            "address_concentration": unit(merged["cl_customers_per_address"], 4, 30),
            # Compressed activity: a real cohort is diffuse, a crew is not.
            "burst": unit(merged["cl_payments_per_active_day"], 3, 40),
            # Accounts opened in the same window, weighted by how many of them
            # there are. Ten accounts on one address is a building; ten accounts
            # on one address that all opened last Tuesday is a drop.
            "cohort_synchrony": (
                unit(merged["cl_signup_synchrony"], 0.002, 0.05)
                * unit(np.log1p(merged["cl_customers"]), np.log1p(4), np.log1p(25))
            ),
            "new_accounts": unit(merged["cl_new_account_share"], 0.25, 0.9),
        }
    )
    weights = pd.Series(
        {
            "cohesion": 0.17, "scale": 0.07, "card_fanout": 0.15, "decline": 0.13,
            "fresh": 0.09, "address_concentration": 0.10, "burst": 0.06,
            "cohort_synchrony": 0.14, "new_accounts": 0.09,
        }
    )
    merged["cl_ring_risk"] = (signals * weights).sum(axis=1).clip(0, 1)
    for name in signals.columns:
        merged[f"clsig_{name}"] = signals[name]
    return merged
