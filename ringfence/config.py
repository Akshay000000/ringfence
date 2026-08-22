"""Config loading. One YAML controls every knob in the system."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
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
    return Config(copy.deepcopy(raw))


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
