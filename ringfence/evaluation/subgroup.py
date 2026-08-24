"""Does the graph help where the fraud is actually relational?

F8 reported a null on IEEE-CIS: no measurable lift overall. The explanation
offered there — that the graph pays off on collusion and this dataset is mostly
single-actor fraud — is not a shrug, it is a **falsifiable prediction**. If it is
right, the graph's advantage should grow with how much linked-account structure a
payment actually sits in. If the advantage is flat across that axis, the
explanation is wrong and should be withdrawn.

The slice variable is the resolved cluster size, which is a property of the data
available at scoring time, not a label. Both arms are scored on identical rows in
every bucket, and every comparison is repeated across seeds, because the whole
reason F8 exists is that a single-seed run here was off by three standard
deviations.

Read this as a subgroup analysis: it is evidence about *where* a method works,
declared in advance from a stated hypothesis. It is not a headline metric, and a
bucket's absolute PR-AUC is not comparable to another bucket's — the base rates
differ. Only the within-bucket gap between arms means anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import headline
from ..model.dataset import build_category_vocab, xy
from ..model.train import _make_model

BUCKET_EDGES = [-1, 0, 4, 19, 99, 10**9]
BUCKET_LABELS = ["no cluster", "2-4 accounts", "5-19 accounts", "20-99 accounts", "100+ accounts"]
DEFAULT_SEEDS = (11, 23, 42, 77, 101)


def bucket_by_cluster_size(frame: pd.DataFrame) -> pd.Series:
    sizes = pd.to_numeric(frame.get("cl_customers"), errors="coerce").fillna(0)
    return pd.cut(sizes, BUCKET_EDGES, labels=BUCKET_LABELS)


def run(
    cfg: Config,
    splits: dict[str, pd.DataFrame],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    train = splits["train"]
    test = splits["test"].reset_index(drop=True)
    y = test["is_fraud"].to_numpy().astype(int)
    buckets = bucket_by_cluster_size(test)
    weight = float(cfg["model"]["class_weight_positive"])

    scores: dict[str, list[np.ndarray]] = {"baseline": [], "graph": []}
    for arm, use_graph in (("baseline", False), ("graph", True)):
        categories = build_category_vocab(train, use_graph)
        X_train, y_train = xy(train, use_graph, categories)
        X_test, _ = xy(test, use_graph, categories)
        X_test = X_test[X_train.columns]
        for seed in seeds:
            model = _make_model(cfg, seed)
            model.fit(X_train, y_train, sample_weight=np.where(y_train == 1, weight, 1.0))
            scores[arm].append(model.predict_proba(X_test)[:, 1])

    rows = []
    for label in BUCKET_LABELS:
        mask = (buckets == label).to_numpy()
        if mask.sum() == 0 or y[mask].sum() < 20:
            continue
        per_arm = {}
        for arm in ("baseline", "graph"):
            vals = [headline(y[mask], s[mask])["pr_auc"] for s in scores[arm]]
            per_arm[arm] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        (bm, bs), (gm, gs) = per_arm["baseline"], per_arm["graph"]
        pooled = float(np.sqrt((bs**2 + gs**2) / 2))
        gap = gm - bm
        rows.append(
            {
                "cluster_size_bucket": label,
                "rows": int(mask.sum()),
                "fraud": int(y[mask].sum()),
                "base_rate_%": round(100 * float(y[mask].mean()), 2),
                "baseline_pr_auc": round(bm, 4),
                "graph_pr_auc": round(gm, 4),
                "difference": round(gap, 4),
                "relative_%": round(100 * gap / bm, 1) if bm else None,
                "pooled_sd": round(pooled, 4),
                "effect_in_sd": round(abs(gap) / pooled, 2) if pooled else None,
                "verdict": (
                    "no measurable difference"
                    if not pooled or abs(gap) / pooled < 2
                    else ("graph better" if gap > 0 else "baseline better")
                ),
            }
        )
    return pd.DataFrame(rows)
