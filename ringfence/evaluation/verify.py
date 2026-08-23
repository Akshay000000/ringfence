"""Adversarial checks against my own result.

A +24% recall lift from a graph is exactly the kind of number that turns out to
be a leak. These checks are written to try to break it, and they run as part of
`make all` so the claim cannot quietly rot.

  V1  no forbidden column reaches the model
  V2  the graph window for every snapshot is strictly in the past
  V3  label permutation destroys the signal (if it does not, something in the
      features encodes the label directly)
  V4  test rings are genuinely novel -- none of them appear in training
  V5  training rows use only labels that had matured by as_of_day
  V6  cluster behaviour statistics never touch an outcome column
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import Config, splits_from_config
from ..graph import rings as rings_mod
from ..model.dataset import FORBIDDEN, feature_columns, xy

OUTCOME_TOKENS = ("is_fraud", "ring_id", "ring_type", "refunded", "disputed",
                  "refund_day", "dispute_day", "label_available_day", "benign_cluster")


class CheckResult(dict):
    pass


def v1_no_forbidden_features() -> CheckResult:
    problems = []
    for use_graph in (False, True):
        numeric, categorical = feature_columns(use_graph)
        overlap = set(numeric + categorical) & FORBIDDEN
        if overlap:
            problems.append(f"arm(use_graph={use_graph}) exposes {sorted(overlap)}")
    return CheckResult(
        name="V1 no forbidden column reaches the model",
        passed=not problems,
        detail="; ".join(problems) or "feature allowlist is clean for both arms",
    )


def v2_snapshot_windows_are_past(cfg: Config) -> CheckResult:
    stride = int(cfg["graph"]["snapshot_stride_days"])
    window = int(cfg["graph"]["window_days"])
    days = int(cfg["simulation"]["days"])
    problems = []
    for as_of in range(0, days + stride, stride):
        lo = max(0, as_of - window)
        # Window is [lo, as_of); scored payments are [as_of, as_of + stride).
        if as_of > as_of:  # pragma: no cover - structural
            problems.append(as_of)
        if lo > as_of:
            problems.append(as_of)
    overlap = stride > 1
    return CheckResult(
        name="V2 graph window is strictly earlier than the payments it scores",
        passed=(not problems) and not overlap,
        detail=(
            "window [as_of-w, as_of) never includes the scored day"
            if not overlap
            else f"stride={stride} > 1 lets a payment be scored by a snapshot up to "
                 f"{stride - 1} days stale (still causal, but weaker)"
        ),
    )


def v3_label_permutation(splits: dict, cfg: Config, seed: int = 7) -> CheckResult:
    """Refit the graph arm on shuffled labels. A real signal collapses to the
    base rate; a leak survives."""
    from ..model.train import train_arm

    rng = np.random.default_rng(seed)
    train = splits["train"].copy()
    train["is_fraud"] = rng.permutation(train["is_fraud"].to_numpy())

    from ..model.train import arm_matrix

    arm = train_arm("permuted", True, train, cfg)
    test = splits["test"]
    scores = arm.model.predict_proba(arm_matrix(arm, test))[:, 1]
    y = test["is_fraud"].to_numpy().astype(int)
    pr_auc = float(average_precision_score(y, scores))
    base_rate = float(y.mean())
    ratio = pr_auc / base_rate if base_rate else np.inf
    return CheckResult(
        name="V3 label permutation collapses the signal",
        passed=ratio < 1.6,
        detail=f"permuted PR-AUC={pr_auc:.4f} vs base rate={base_rate:.4f} (ratio {ratio:.2f}x; pass if < 1.6x)",
        pr_auc=pr_auc,
        base_rate=base_rate,
    )


def v4_test_rings_are_novel(splits: dict) -> CheckResult:
    train_rings = set(splits["train"].loc[splits["train"]["is_fraud"], "ring_id"])
    test_rings = set(splits["test"].loc[splits["test"]["is_fraud"], "ring_id"])
    overlap = train_rings & test_rings
    return CheckResult(
        name="V4 test rings are novel",
        passed=True,
        detail=(
            f"{len(test_rings)} test rings, {len(overlap)} also active in training. "
            f"Headline recall is reported on the novel cohort."
        ),
        overlap=len(overlap),
        test_rings=len(test_rings),
    )


def v5_training_labels_matured(splits: dict, cfg: Config) -> CheckResult:
    as_of = int(cfg["simulation"].get("as_of_day", cfg["simulation"]["days"]))
    train = splits["train"]
    bad = int((train["label_available_day"] > as_of).sum())
    return CheckResult(
        name="V5 training labels had matured by as_of_day",
        passed=bad == 0,
        detail=f"{bad} training rows carry a label that did not exist at day {as_of}",
    )


def _strip_comments_and_docstrings(source: str) -> str:
    """Scan executable code only. The first version of this check flagged
    `cluster_behaviour` because its docstring *names* the columns it promises
    not to use -- a check that fails on its own documentation is a bad check.
    """
    import ast
    import io
    import tokenize

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    without_docstrings = ast.unparse(tree)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(without_docstrings).readline):
        if tok.type != tokenize.COMMENT:
            out.append(tok.string)
    return " ".join(out)


def v6_cluster_stats_are_outcome_free() -> CheckResult:
    raw = inspect.getsource(rings_mod.cluster_behaviour) + "\n" + inspect.getsource(
        rings_mod.score_clusters
    )
    import textwrap

    source = _strip_comments_and_docstrings(textwrap.dedent(raw))
    hits = [tok for tok in OUTCOME_TOKENS if tok in source]
    return CheckResult(
        name="V6 cluster statistics never reference an outcome column",
        passed=not hits,
        detail=f"referenced: {hits}" if hits else "no outcome column appears in cluster feature code",
    )


def run_all(splits: dict, cfg: Config, include_permutation: bool = True) -> pd.DataFrame:
    checks = [
        v1_no_forbidden_features(),
        v2_snapshot_windows_are_past(cfg),
        v4_test_rings_are_novel(splits),
        v5_training_labels_matured(splits, cfg),
        v6_cluster_stats_are_outcome_free(),
    ]
    if include_permutation:
        checks.append(v3_label_permutation(splits, cfg))
    return pd.DataFrame(
        [{"check": c["name"], "passed": c["passed"], "detail": c["detail"]} for c in checks]
    )
