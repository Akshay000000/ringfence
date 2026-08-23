"""Trains the two ablation arms and persists them.

Arm A  tabular only          -- the strong production-rule-engine baseline
Arm B  tabular + graph       -- the claim under test

Same algorithm, same hyperparameters, same rows, same seed. The only thing that
changes between arms is the presence of the graph feature block, which is the
only way the comparison means anything.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import Config, reports_dir
from .dataset import build_category_vocab, xy

ARMS = {"baseline": False, "graph": True}


@dataclass
class TrainedArm:
    name: str
    use_graph: bool
    model: HistGradientBoostingClassifier
    columns: list[str]
    n_train: int
    n_train_positive: int
    categories: dict[str, list[str]] | None = None


def _make_model(cfg: Config, seed: int) -> HistGradientBoostingClassifier:
    m = cfg["model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(m["learning_rate"]),
        max_iter=int(m["max_iter"]),
        max_leaf_nodes=int(m["max_leaf_nodes"]),
        l2_regularization=float(m["l2_regularization"]),
        early_stopping=bool(m["early_stopping"]),
        n_iter_no_change=25,
        validation_fraction=0.12,
        categorical_features="from_dtype",
        random_state=seed,
    )


def train_arm(
    name: str,
    use_graph: bool,
    train_df: pd.DataFrame,
    cfg: Config,
) -> TrainedArm:
    categories = build_category_vocab(train_df, use_graph)
    X, y = xy(train_df, use_graph, categories)
    seed = int(cfg["seed"]) % (2**31)
    model = _make_model(cfg, seed)

    pos_weight = float(cfg["model"]["class_weight_positive"])
    sample_weight = np.where(y == 1, pos_weight, 1.0)

    model.fit(X, y, sample_weight=sample_weight)
    return TrainedArm(
        name=name,
        use_graph=use_graph,
        model=model,
        columns=list(X.columns),
        n_train=len(X),
        n_train_positive=int(y.sum()),
        categories=categories,
    )


def arm_matrix(arm: TrainedArm, frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in the arm's own column order and category vocabulary."""
    X, _ = xy(frame, arm.use_graph, getattr(arm, "categories", None))
    return X[arm.columns]


def predict(arm: TrainedArm, frame: pd.DataFrame) -> np.ndarray:
    return arm.model.predict_proba(arm_matrix(arm, frame))[:, 1]


def train_all(splits: dict[str, pd.DataFrame], cfg: Config) -> dict[str, TrainedArm]:
    arms = {}
    for name, use_graph in ARMS.items():
        arm = train_arm(name, use_graph, splits["train"], cfg)
        arms[name] = arm
        print(
            f"  arm={name:<9} features={len(arm.columns):>3} "
            f"rows={arm.n_train:>7} positives={arm.n_train_positive:>6}",
            flush=True,
        )
    return arms


def save_arms(arms: dict[str, TrainedArm]) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    with open(reports_dir() / "models.pkl", "wb") as handle:
        pickle.dump(arms, handle)


def load_arms() -> dict[str, TrainedArm]:
    with open(reports_dir() / "models.pkl", "rb") as handle:
        return pickle.load(handle)
