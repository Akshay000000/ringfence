"""Is there any collusion structure in this dataset to find?

F8 reported that the graph adds nothing on IEEE-CIS. F9 offered an explanation,
that the fraud there is single-actor rather than collusive, and withdrew it when
a subgroup test found no supporting trend. That left the null unexplained, which
is honest but weak.

This asks the question underneath both: does fraud *concentrate* inside the
clusters the graph finds, more than it would if the same accounts were shuffled
into clusters of identical sizes?

The test is a permutation null. Take the real clusters, record how concentrated
fraud is inside them, then repeatedly reassign accounts at random into clusters
of exactly the same size distribution and record the same statistic. If the real
value sits inside the shuffled distribution, the clusters carry no information
about fraud, and no model could have extracted any. If it sits far outside, the
structure is there and the failure is ours.

Either answer is worth having. The first explains the null. The second says the
method is leaving signal on the table, which is a sharper and more uncomfortable
finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _concentration(labels: np.ndarray, fraud: np.ndarray) -> float:
    """Share of fraud sitting in clusters that are more than half fraudulent.

    Chosen over a variance or entropy statistic because it means something in
    plain words: how much of the fraud lands in a group you could actually act
    on as a group.
    """
    frame = pd.DataFrame({"c": labels, "f": fraud})
    per = frame.groupby("c")["f"].agg(["sum", "size"])
    per = per[per["size"] >= 2]
    if per.empty or fraud.sum() == 0:
        return 0.0
    hot = per[per["sum"] / per["size"] > 0.5]
    return float(hot["sum"].sum() / fraud.sum())


def permutation_test(
    frame: pd.DataFrame,
    cluster_col: str = "cluster",
    n_permutations: int = 400,
    seed: int = 20260905,
) -> dict:
    """Compare real cluster fraud-concentration against same-shaped random ones."""
    sub = frame[frame[cluster_col].fillna("").astype(str) != ""]
    if sub.empty:
        return {"clustered_rows": 0, "verdict": "no clusters to test"}

    labels = sub[cluster_col].to_numpy()
    fraud = sub["is_fraud"].to_numpy().astype(int)
    observed = _concentration(labels, fraud)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        # Shuffle the labels, not the fraud flags: cluster sizes stay exactly as
        # the graph produced them, only membership becomes random.
        null[i] = _concentration(rng.permutation(labels), fraud)

    mean, sd = float(null.mean()), float(null.std(ddof=1))
    z = (observed - mean) / sd if sd > 0 else float("nan")
    p = float((null >= observed).sum() + 1) / (n_permutations + 1)

    return {
        "clustered_rows": int(len(sub)),
        "clusters": int(pd.Series(labels).nunique()),
        "fraud_in_clustered_rows": int(fraud.sum()),
        "observed_concentration": round(observed, 4),
        "null_mean": round(mean, 4),
        "null_sd": round(sd, 4),
        "z_score": round(z, 2) if z == z else None,
        "p_value": round(p, 4),
        "permutations": n_permutations,
        "verdict": _verdict(z, observed, mean),
    }


def _verdict(z: float, observed: float, mean: float) -> str:
    if z != z:
        return "indeterminate"
    if z < 2:
        return ("no detectable collusion structure: fraud is no more concentrated "
                "in these clusters than in random groups of the same sizes")
    direction = "more" if observed > mean else "less"
    return (f"structure present: fraud is {direction} concentrated than chance "
            f"({z:.1f} sd), so the clusters do carry information")


def run(frames: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        result = permutation_test(frame, **kwargs)
        result["dataset"] = name
        rows.append(result)
    cols = ["dataset", "clustered_rows", "clusters", "fraud_in_clustered_rows",
            "observed_concentration", "null_mean", "null_sd", "z_score",
            "p_value", "permutations", "verdict"]
    return pd.DataFrame(rows)[[c for c in cols if c in rows[0]]]
