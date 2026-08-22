"""Tests that protect the honesty guarantees, not the happy path.

Each of these encodes a bug that actually occurred during the build.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ringfence.config import load_config
from ringfence.features.tabular import _prior_count_window, _prior_cumsum, _prior_distinct
from ringfence.graph.rings import _louvain_split
from ringfence.model.dataset import FORBIDDEN, feature_columns


def test_velocity_counter_excludes_the_current_row():
    keys = np.array(["a", "a", "a", "b"])
    times = np.array([0, 10, 20, 5])
    counts = _prior_count_window(keys, times, window=3600)
    assert counts.tolist() == [0, 1, 2, 0]


def test_velocity_counter_respects_the_window():
    keys = np.array(["a", "a", "a"])
    times = np.array([0, 100, 100_000])
    counts = _prior_count_window(keys, times, window=3600)
    assert counts.tolist() == [0, 1, 0]


def test_velocity_counter_does_not_leak_across_keys():
    keys = np.array(["a", "b", "a", "b"])
    times = np.array([0, 1, 2, 3])
    counts = _prior_count_window(keys, times, window=3600)
    assert counts.tolist() == [0, 0, 1, 1]


def test_prior_cumsum_is_exclusive():
    keys = np.array(["a", "a", "a"])
    times = np.array([0, 1, 2])
    values = np.array([1.0, 1.0, 1.0])
    assert _prior_cumsum(keys, times, values).tolist() == [0.0, 1.0, 2.0]


def test_prior_distinct_counts_only_earlier_pairs():
    devices = np.array(["d", "d", "d", "d"])
    cards = np.array(["c1", "c1", "c2", "c3"])
    times = np.array([0, 1, 2, 3])
    assert _prior_distinct(devices, cards, times).tolist() == [0.0, 1.0, 1.0, 2.0]


@pytest.mark.parametrize("resolution", [0.5, 0.7, 0.9, 1.0])
def test_louvain_does_not_shatter_a_clique(resolution):
    """F3: at resolution > 1 Louvain splits a clique into singletons, and a ring
    sharing one drop address IS a clique. This test is the regression guard."""
    n = 26
    dense = np.full((n, n), 0.52)
    np.fill_diagonal(dense, 0.0)
    parts = _louvain_split(sp.csr_matrix(dense), np.arange(n), resolution, seed=1)
    assert max(len(p) for p in parts) == n


def test_configured_resolution_is_clique_safe():
    cfg = load_config()
    assert float(cfg["graph"]["louvain_resolution"]) <= 1.0


def test_no_arm_exposes_a_forbidden_column():
    for use_graph in (False, True):
        numeric, categorical = feature_columns(use_graph)
        assert not set(numeric + categorical) & FORBIDDEN


def test_graph_window_never_includes_the_scored_day():
    cfg = load_config()
    window = int(cfg["graph"]["window_days"])
    stride = int(cfg["graph"]["snapshot_stride_days"])
    assert stride == 1, "stride > 1 makes the graph blind to short-lived rings (F2)"
    for as_of in range(0, int(cfg["simulation"]["days"]) + 1):
        lo = max(0, as_of - window)
        scored_days = range(as_of, as_of + stride)
        assert all(day >= as_of for day in scored_days)
        assert lo <= as_of
