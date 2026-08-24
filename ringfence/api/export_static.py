"""Bake the console's API responses into a static bundle.

The analyst console is backed by a service that loads 458k payments, scores
them, and rebuilds graph snapshots on demand. None of that can run on a static
host, and a demo that requires a reviewer to `pip install` first is a demo most
reviewers will not see.

So the same responses are precomputed once and written as a JS blob the page
reads instead of calling the API. The console itself is unchanged: every read
already goes through `api()`, which resolves from the bundle when one is present
and hits the network when it is not.

What ships: the alert queue, each alert's reasons, its what-if, and its evidence
subgraph. Identifier values are already masked by the evidence layer, and the
whole corpus is synthetic, so nothing sensitive travels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT
from .service import _row_payload, state


def build(limit: int | None = None) -> dict:
    s = state()
    queue = s.queue if limit is None else s.queue.head(limit)

    alerts, details = [], {}
    for _, row in queue.iterrows():
        payment_id = str(row["payment_id"])
        payload = _row_payload(row)
        payload["ring_type"] = str(row["ring_type"])
        alerts.append(payload)

        story = s.narration(payment_id)
        detail = dict(payload)
        detail["summary"] = story.get("summary")
        detail["reasons"] = story.get("reasons", [])
        detail["caveat"] = story.get("caveat")
        detail["graph_driven"] = story.get("graph_driven")

        evidence = None
        try:
            raw = row.get("cluster")
            known = raw if isinstance(raw, str) and raw else None
            built = s.evidence.evidence_for(s.payments.loc[payment_id], cluster=known)
            if built is not None:
                evidence = built.to_dict()
        except Exception:
            evidence = None
        detail["evidence"] = evidence

        with_graph = float(row["score"])
        without_graph = float(row["baseline_score"])
        detail["whatif"] = {
            "payment_id": payment_id,
            "with_graph": round(with_graph, 4),
            "without_graph": round(without_graph, 4),
            "delta": round(with_graph - without_graph, 4),
        }
        details[payment_id] = detail

    summary = json.loads(
        json.dumps(
            {
                "test_set": {
                    "payments": int(len(s.test)),
                    "fraud": int(s.test["is_fraud"].sum()),
                    "base_rate": round(float(s.test["is_fraud"].mean()), 5),
                    "rings": int(s.test.loc[s.test["is_fraud"], "ring_id"].nunique()),
                },
                "baseline": _arm_block(s, "baseline"),
                "graph": _arm_block(s, "graph"),
                "ablation": s.ablation,
                "per_archetype": s.archetypes,
                "verification": s.verification,
                "false_positive_composition": s.fp_composition,
                "missed": s.missed,
            },
            default=float,
        )
    )
    return {"summary": summary, "alerts": alerts, "details": details}


def _arm_block(s, name: str) -> dict:
    payload = s.results.get("arms", {}).get(name, {})
    head = payload.get("headline", {})
    conf = payload.get("confusion_at_cost_optimal", {})
    cost = payload.get("cost_optimal", {})
    return {
        "pr_auc": head.get("pr_auc"),
        "roc_auc": head.get("roc_auc"),
        "precision": conf.get("precision"),
        "recall": conf.get("recall"),
        "alert_rate": conf.get("alert_rate"),
        "blocked_good": cost.get("blocked_good"),
        "net_saving_inr": cost.get("net_saving_inr"),
        "do_nothing_inr": cost.get("do_nothing_cost_inr"),
    }


SITE_DIR = REPO_ROOT / "site"


def build_console_page() -> Path:
    """Emit the static twin of the console.

    The page itself is unchanged; it gains a script tag that defines
    `window.__RINGFENCE_DATA__` before the app runs, which is the switch that
    makes every `api()` call resolve locally instead of over the network.

    Injection is anchored on the first inline `<script>`, not on a snippet of
    the script's own body, because anchoring on body text silently stopped
    matching the moment the file was edited and produced a page that fell
    through to a network call that does not exist.
    """
    source = (REPO_ROOT / "ringfence" / "api" / "console.html").read_text(encoding="utf-8")
    if "__RINGFENCE_DATA__" not in source:
        raise RuntimeError("console.html has no static-data switch; refusing to build a broken page")

    marker = "<script>"
    index = source.find(marker)
    if index == -1:
        raise RuntimeError("console.html has no inline <script> to anchor on")
    page = (
        source[:index]
        + '<script src="data/console-data.js"></script>\n'
        + source[index:]
    )
    page = page.replace(
        '<span class="defense">Read-only, no block/refund endpoint exists</span>',
        '<a href="index.html" class="defense" style="margin-left:auto;text-decoration:none">'
        "&larr; overview</a>\n"
        '    <span class="defense" style="margin-left:0">Read-only, no block or refund endpoint</span>',
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "console.html"
    out.write_text(page, encoding="utf-8")
    return out


def write(out_dir: Path | None = None, limit: int | None = None) -> Path:
    out_dir = Path(out_dir or REPO_ROOT / "site" / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build(limit=limit)
    path = out_dir / "console-data.js"
    payload = json.dumps(bundle, separators=(",", ":"), default=str)
    path.write_text(f"window.__RINGFENCE_DATA__ = {payload};\n", encoding="utf-8")
    build_console_page()
    return path
