"""Explanation-layer guarantees."""
from __future__ import annotations

import pandas as pd

from ringfence.explain.reasons import GRAPH_GROUPS, REASON_GROUPS, _pluralise
from ringfence.model.dataset import feature_columns


def test_every_feature_belongs_to_exactly_one_reason_group():
    numeric, categorical = feature_columns(True)
    features = set(numeric + categorical)
    seen: dict[str, str] = {}
    for group, spec in REASON_GROUPS.items():
        for col in spec["cols"]:
            assert col not in seen, f"{col} is in both {seen[col]} and {group}"
            seen[col] = group
    ungrouped = features - set(seen)
    assert not ungrouped, f"features with no explanation group: {sorted(ungrouped)}"


def test_no_group_references_an_unknown_feature():
    numeric, categorical = feature_columns(True)
    features = set(numeric + categorical)
    for group, spec in REASON_GROUPS.items():
        unknown = set(spec["cols"]) - features
        assert not unknown, f"{group} references non-features: {sorted(unknown)}"


def test_evidence_columns_are_inside_their_own_group():
    for group, spec in REASON_GROUPS.items():
        cols = set(spec["cols"])
        for col, _template in spec.get("evidence", []):
            assert col in cols, f"{group} quotes {col}, which it does not own"


def test_graph_groups_are_real_groups():
    assert GRAPH_GROUPS <= set(REASON_GROUPS)


def test_pluralisation_uses_the_leading_quantity():
    assert _pluralise("1 attempt(s) from this device") == "1 attempt from this device"
    assert _pluralise("4 attempt(s) from this device") == "4 attempts from this device"
    # number and noun separated by an adjective
    assert _pluralise("2 distinct account(s) shipped here") == "2 distinct accounts shipped here"
    assert _pluralise("1 distinct account(s) shipped here") == "1 distinct account shipped here"
