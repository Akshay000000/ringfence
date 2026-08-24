"""Does the result survive labels that are wrong?

Both arms train on ground-truth `is_fraud`. Production labels are not like that:
a chargeback can be filed on a legitimate purchase, a genuine abuser is never
disputed, and a manual review queue mislabels under time pressure. A method whose
advantage evaporates at 10% label noise is not deployable, however good its
clean-label number looks.

So: corrupt a fraction of the **training** labels, leave the test labels intact,
and watch what happens to the gap. Two corruption modes, because they are not
equally hard:

  symmetric   flip labels in both directions at rate p. The easy case.
  fn_only     flip only positives to negatives, fraud that was never caught and
              therefore trains as legitimate. This is what actually happens in a
              fraud system, and it is worse, because it teaches the model that
              real attacks are fine.

Test labels are never corrupted. The question is whether a model trained on dirty
labels still finds real fraud, not whether it reproduces the dirt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import headline, recall_at_precision
from ..model.dataset import build_category_vocab, xy
from ..model.train import _make_model

DEFAULT_RATES = (0.0, 0.05, 0.10, 0.20)
DEFAULT_SEEDS = (11, 42, 101)


def corrupt(labels: np.ndarray, rate: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    y = labels.copy()
    if rate <= 0:
        return y
    if mode == "fn_only":
        positives = np.flatnonzero(y == 1)
        n = int(round(len(positives) * rate))
        if n:
            y[rng.choice(positives, size=n, replace=False)] = 0
        return y
    flip = rng.random(len(y)) < rate
    y[flip] = 1 - y[flip]
    return y


def run(
    cfg: Config,
    splits: dict[str, pd.DataFrame],
    rates: tuple[float, ...] = DEFAULT_RATES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    modes: tuple[str, ...] = ("symmetric", "fn_only"),
) -> pd.DataFrame:
    train, test = splits["train"], splits["test"].reset_index(drop=True)
    y_test = test["is_fraud"].to_numpy().astype(int)
    weight = float(cfg["model"]["class_weight_positive"])

    prepared = {}
    for arm, use_graph in (("baseline", False), ("graph", True)):
        categories = build_category_vocab(train, use_graph)
        X_train, y_train = xy(train, use_graph, categories)
        X_test, _ = xy(test, use_graph, categories)
        prepared[arm] = (X_train, y_train, X_test[X_train.columns])

    rows = []
    for mode in modes:
        for rate in rates:
            per_arm = {}
            for arm in ("baseline", "graph"):
                X_train, y_clean, X_test = prepared[arm]
                pr, rec = [], []
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    y_dirty = corrupt(y_clean, rate, mode, rng)
                    model = _make_model(cfg, seed)
                    model.fit(X_train, y_dirty,
                              sample_weight=np.where(y_dirty == 1, weight, 1.0))
                    s = model.predict_proba(X_test)[:, 1]
                    pr.append(headline(y_test, s)["pr_auc"])
                    rec.append(recall_at_precision(y_test, s, 0.90)["recall"])
                per_arm[arm] = (float(np.mean(pr)), float(np.std(pr, ddof=1)),
                                float(np.mean(rec)))
            (bp, bs, br), (gp, gs, gr) = per_arm["baseline"], per_arm["graph"]
            pooled = float(np.sqrt((bs**2 + gs**2) / 2))
            rows.append(
                {
                    "mode": mode,
                    "noise_rate": rate,
                    "baseline_pr_auc": round(bp, 4),
                    "graph_pr_auc": round(gp, 4),
                    "pr_auc_gap": round(gp - bp, 4),
                    "pooled_sd": round(pooled, 4),
                    "effect_in_sd": round(abs(gp - bp) / pooled, 1) if pooled else None,
                    "baseline_recall_at_p90": round(br, 4),
                    "graph_recall_at_p90": round(gr, 4),
                    "recall_gap": round(gr - br, 4),
                }
            )
            print(f"  {mode:<10} noise={rate:<5} baseline={bp:.4f} graph={gp:.4f} "
                  f"gap={gp - bp:+.4f}", flush=True)
    return pd.DataFrame(rows)
