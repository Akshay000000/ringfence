"""Is the ablation difference real, or is it seed noise?

A single-seed comparison told me the graph arm beat the baseline by +2.9% on
IEEE-CIS. Refitting the same two arms across five seeds showed the run-to-run
standard deviation was +/- 0.004 PR-AUC -- three times the size of the
"improvement". The lift was noise, and reporting it would have been a fabricated
result arrived at honestly.

So any ablation claim has to clear its own noise floor. This module refits both
arms across several seeds and reports the gap in units of pooled standard
deviation. Below ~2 sd, the honest answer is "no measurable difference", and
that is what gets written down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import headline, recall_at_precision
from ..model.dataset import build_category_vocab, xy
from ..model.train import _make_model

DEFAULT_SEEDS = (11, 23, 42, 77, 101)


def _fit_score(cfg: Config, train: pd.DataFrame, test: pd.DataFrame,
               use_graph: bool, seed: int, drop: set[str], cache: dict):
    key = (use_graph, tuple(sorted(drop)))
    if key not in cache:
        categories = build_category_vocab(train, use_graph)
        X_train, y_train = xy(train, use_graph, categories)
        X_test, _ = xy(test, use_graph, categories)
        keep = [c for c in X_train.columns if c not in drop]
        cache[key] = (X_train[keep], y_train, X_test[keep])
    X_train, y_train, X_test = cache[key]

    weight = float(cfg["model"]["class_weight_positive"])
    model = _make_model(cfg, seed)
    model.fit(X_train, y_train, sample_weight=np.where(y_train == 1, weight, 1.0))
    return model.predict_proba(X_test)[:, 1]


def run(
    cfg: Config,
    splits: dict[str, pd.DataFrame],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    conditions: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Refit both arms per seed and summarise the gap against the noise floor.

    `conditions` maps a label to a set of feature columns to withhold from BOTH
    arms, which is how a redundancy question gets asked: withhold the source
    dataset's own entity-aggregation features and see whether the graph then
    earns its place.
    """
    train = splits["train"]
    test = splits["test"].reset_index(drop=True)
    y = test["is_fraud"].to_numpy().astype(int)
    conditions = conditions or {"all_features": set()}

    cache: dict = {}
    rows = []
    for label, drop in conditions.items():
        scores = {}
        for arm, use_graph in (("baseline", False), ("graph", True)):
            per_seed = []
            for seed in seeds:
                s = _fit_score(cfg, train, test, use_graph, seed, drop, cache)
                per_seed.append(
                    {
                        "pr_auc": headline(y, s)["pr_auc"],
                        "recall_at_p80": recall_at_precision(y, s, 0.80)["recall"],
                    }
                )
            scores[arm] = pd.DataFrame(per_seed)

        base, graph = scores["baseline"], scores["graph"]
        for metric in ("pr_auc", "recall_at_p80"):
            b, g = base[metric], graph[metric]
            pooled = float(np.sqrt((b.std(ddof=1) ** 2 + g.std(ddof=1) ** 2) / 2)) or float("nan")
            gap = float(g.mean() - b.mean())
            rows.append(
                {
                    "condition": label,
                    "metric": metric,
                    "seeds": len(seeds),
                    "baseline_mean": round(float(b.mean()), 4),
                    "baseline_sd": round(float(b.std(ddof=1)), 4),
                    "graph_mean": round(float(g.mean()), 4),
                    "graph_sd": round(float(g.std(ddof=1)), 4),
                    "difference": round(gap, 4),
                    "relative_%": round(100 * gap / float(b.mean()), 1) if b.mean() else None,
                    "pooled_sd": round(pooled, 4),
                    "effect_in_sd": round(abs(gap) / pooled, 2) if pooled == pooled else None,
                    "verdict": _verdict(gap, pooled),
                }
            )
    return pd.DataFrame(rows)


def _verdict(gap: float, pooled: float) -> str:
    if pooled != pooled or pooled == 0:
        return "indeterminate"
    effect = abs(gap) / pooled
    if effect < 2:
        return "no measurable difference"
    return ("graph better" if gap > 0 else "baseline better") + f" ({effect:.1f} sd)"
