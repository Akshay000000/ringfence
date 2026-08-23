"""Config loading. One YAML controls every knob in the system."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Each dataset gets its own workspace. Without this the synthetic and IEEE-CIS
# runs write payments.csv.gz and results.json to the same paths and silently
# overwrite each other -- which is how a "real data" number can end up being
# reported from synthetic artefacts, or vice versa.
_WORKSPACE = "synthetic"


def use_workspace(name: str) -> None:
    global _WORKSPACE
    _WORKSPACE = name or "synthetic"


def workspace() -> str:
    return _WORKSPACE


def data_dir() -> Path:
    return REPO_ROOT / "data" / _WORKSPACE


def reports_dir() -> Path:
    return REPO_ROOT / "reports" / _WORKSPACE


# Back-compat module attributes for anything still importing the constants.
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


class Config(dict):
    """Dict with attribute access and dotted-path lookup."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = Config(copy.deepcopy(raw))
    use_workspace(cfg.get_path("dataset.name") or cfg.get_path("dataset.kind") or "synthetic")
    return cfg


@dataclass(frozen=True)
class Split:
    """Inclusive day-index window."""

    name: str
    start_day: int
    end_day: int

    def contains(self, day: int) -> bool:
        return self.start_day <= day <= self.end_day


def splits_from_config(cfg: Config) -> dict[str, Split]:
    sim = cfg["simulation"]
    return {
        name: Split(name, *sim[f"{name}_days"])
        for name in ("train", "val", "test")
    }


def ensure_dirs() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    reports_dir().mkdir(parents=True, exist_ok=True)
