"""The dataset contract, and the IEEE-CIS adapter that has to satisfy it.

The whole two-dataset claim rests on one idea: an adapter converts a source into
a canonical schema, and the pipeline runs unchanged. If the contract quietly
drifts, that claim stops being true and nothing else notices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ringfence.config import load_config
from ringfence.datasets import ieee_cis, schema


def _minimal_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "payment_id": [f"pay_{i}" for i in range(n)],
            "customer_id": [f"cust_{i % 3}" for i in range(n)],
            "day": list(range(n)),
            "created_at": [i * 86400 for i in range(n)],
            "amount": [10000] * n,
            "is_fraud": [False] * n,
        }
    )


def _cfg():
    cfg = load_config()
    cfg["simulation"]["train_days"] = [0, 2]
    cfg["simulation"]["val_days"] = [3, 3]
    cfg["simulation"]["test_days"] = [4, 5]
    return cfg


def test_finalise_fills_every_link_column_even_when_absent():
    out = schema.finalise(_minimal_frame(), _cfg(), "test")
    for col in schema.LINK:
        assert col in out.columns, f"{col} missing; the graph would crash on it"


def test_finalise_fills_neutral_values_rather_than_leaving_gaps():
    out = schema.finalise(_minimal_frame(), _cfg(), "test")
    assert (out["status"] == "captured").all()
    assert (out["refund_day"] == -1).all()
    assert not out["refunded"].any()


def test_finalise_rejects_a_frame_missing_a_required_column():
    frame = _minimal_frame().drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        schema.finalise(frame, _cfg(), "test")


def test_finalise_assigns_the_temporal_split_from_config():
    out = schema.finalise(_minimal_frame(), _cfg(), "test")
    assert out.loc[out["day"] == 0, "split"].iloc[0] == "train"
    assert out.loc[out["day"] == 3, "split"].iloc[0] == "val"
    assert out.loc[out["day"] == 5, "split"].iloc[0] == "test"


def test_real_data_has_no_ring_labels_and_says_so():
    """IEEE-CIS labels fraud, not ring membership. Reporting a per-archetype
    breakdown there would be inventing a number."""
    from ringfence.evaluation.metrics import has_ring_labels

    out = schema.finalise(_minimal_frame(), _cfg(), "test")
    out.loc[0, "is_fraud"] = True
    assert not has_ring_labels(out)


def test_client_identity_is_stable_for_the_same_card_address_and_first_seen():
    """The reconstruction is card1 + addr1 + (day - D1). Two payments from one
    client must land on one id, or the graph is built on sand."""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "card1": [1000.0, 1000.0, 2000.0],
            "addr1": [204.0, 204.0, 204.0],
            "D1": [10.0, 12.0, 10.0],
            "day": [30, 32, 30],
        }
    )
    uid = ieee_cis._client_uid(frame)
    assert uid.iloc[0] == uid.iloc[1], "same card, same address, same first-seen day"
    assert uid.iloc[2] != uid.iloc[0], "different card must not collapse together"


def test_rows_without_a_card_get_their_own_identity():
    """A missing card1 must not collapse every anonymous payment into one giant
    fake customer, which would hand the graph an enormous false cluster."""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "card1": [np.nan, np.nan],
            "addr1": [204.0, 204.0],
            "D1": [10.0, 10.0],
            "day": [30, 30],
        }
    )
    uid = ieee_cis._client_uid(frame)
    assert uid.iloc[0] != uid.iloc[1]


def test_device_fingerprint_is_absent_when_there_is_no_identity_record():
    frame = pd.DataFrame({"DeviceInfo": [None], "id_30": [None], "id_31": [None], "id_33": [None]})
    assert ieee_cis._device_fingerprint(frame).isna().all()


def test_device_fingerprint_combines_the_parts_it_does_have():
    frame = pd.DataFrame({"DeviceInfo": ["Windows"], "id_31": ["chrome 63.0"]})
    value = ieee_cis._device_fingerprint(frame).iloc[0]
    assert "Windows" in value and "chrome 63.0" in value


def test_coverage_reports_missing_link_types_as_zero():
    out = schema.finalise(_minimal_frame(), _cfg(), "test")
    cover = ieee_cis.coverage(out).set_index("link_type")
    assert cover.loc["ip", "coverage_%"] == 0.0
