"""The service, and the defense-only guarantee it is supposed to enforce.

The track disqualifies anything offense-capable. That claim is made in the
README, in ARCHITECTURE and on the site, so it needs a test rather than a
promise: if someone later adds a POST route that cancels a payment, the build
should fail, not the submission.
"""
from __future__ import annotations

import inspect

from starlette.routing import Route

from ringfence.api import service


def _routes():
    return [r for r in service.routes if isinstance(r, Route)]


def test_every_route_is_read_only():
    offending = []
    for route in _routes():
        methods = set(route.methods or {"GET"}) - {"GET", "HEAD", "OPTIONS"}
        if methods:
            offending.append((route.path, sorted(methods)))
    assert not offending, f"defense-only violated by {offending}"


def test_no_route_name_suggests_a_write_action():
    """A read-only verb list is necessary but not sufficient: a GET that blocks a
    payment is still a write. Names are the cheap second signal."""
    banned = ("block", "refund", "cancel", "chargeback", "update", "delete",
              "approve", "reject", "capture", "void")
    for route in _routes():
        lowered = route.path.lower()
        assert not any(word in lowered for word in banned), route.path


def test_the_service_module_never_imports_a_payment_client():
    """There is no path from this service to a payment API, so it cannot act on
    one even by mistake."""
    source = inspect.getsource(service)
    for forbidden in ("razorpay", "requests.post", "httpx.post", "urllib.request"):
        assert forbidden not in source.lower(), f"{forbidden} reachable from the service"


def test_expected_routes_exist():
    paths = {r.path for r in _routes()}
    assert {"/", "/api/health", "/api/summary", "/api/alerts", "/api/rings"} <= paths


def test_alert_filters_are_all_handled():
    """Every filter offered by the console must be implemented server-side too,
    or the live service and the static build disagree about what a filter means."""
    source = inspect.getsource(service.alerts)
    console = (service.CONSOLE).read_text(encoding="utf-8")
    offered = set()
    for chunk in console.split('data-f="')[1:]:
        value = chunk.split('"')[0]
        if value:
            offered.add(value)
    for name in offered:
        assert name in source, f"console offers filter {name!r} the service does not implement"


def test_row_payload_never_leaks_an_identifier():
    """The console shows masked evidence. A raw device fingerprint or address
    hash must not travel in the alert list itself."""
    import pandas as pd

    row = pd.Series(
        {
            "payment_id": "pay_x", "score": 0.9, "baseline_score": 0.1, "amount": 1000,
            "exposure_inr": 9.0, "day": 5, "is_fraud": True, "ring_type": "bust_out",
            "g_in_cluster": 1, "account_age_days": 3,
            "device_fingerprint": "dev_SECRET", "shipping_address_hash": "addr_SECRET",
            "customer_id": "cust_SECRET",
        }
    )
    payload = service._row_payload(row)
    assert "SECRET" not in str(payload)
