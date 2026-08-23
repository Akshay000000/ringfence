"""RingFence analyst console — HTTP service.

Built on Starlette rather than FastAPI. FastAPI runs on Starlette anyway, and
for six routes its request-model machinery buys nothing that hand-written
validation does not; skipping it keeps the dependency list short enough that
`pip install -r requirements.txt` works on a bare machine.

The service is **read-only by design**. It exposes scores, reasons and
evidence. There is no endpoint that blocks a payment, issues a refund, or
mutates any account, because the track disqualifies anything offense-capable
and "the API had a write path but we didn't call it" is not a defence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from ..config import REPORTS_DIR, load_config
from ..explain.evidence import EvidenceBuilder
from ..explain.reasons import attribute, build_reference, narrate
from ..io import read_table
from ..model.dataset import split_frames
from ..model.train import load_arms, predict

CONSOLE = Path(__file__).parent / "console.html"
PRECOMPUTED_ALERTS = 400


class State:
    """Everything loaded once at startup.

    The alert queue is precomputed because an analyst console that takes four
    seconds to paint its first screen does not get used. Evidence subgraphs are
    built lazily and cached per day -- a queue is worked in date order, so the
    cache hit rate is high.
    """

    def __init__(self) -> None:
        self.cfg = load_config()
        self.arms = load_arms()
        self.arm = self.arms["graph"]
        self.baseline = self.arms["baseline"]

        matrix = pd.read_pickle(REPORTS_DIR / "matrix.pkl")
        splits = split_frames(matrix, self.cfg)
        self.train = splits["train"]
        test = splits["test"].reset_index(drop=True)
        test["score"] = predict(self.arm, test)
        test["baseline_score"] = predict(self.baseline, test)
        self.test = test

        self.reference = build_reference(self.train, self.arm.use_graph)

        payments = read_table("payments")
        self.payments = payments.set_index("payment_id", drop=False)
        self.evidence = EvidenceBuilder(payments, self.cfg)

        # Rank by expected loss, not by score.
        #
        # Sorting on raw score produced a queue of 392 card-testing probes out of
        # 400, every one tied at 0.9998 and worth about five rupees. That is a
        # correct ranking and a useless one: a risk analyst triages by money at
        # risk, so a Rs 50,000 bust-out order at 0.97 must outrank a Rs 5 probe
        # at 0.9998.
        test["exposure_inr"] = test["score"] * test["amount"] / 100.0
        ranked = test.sort_values("exposure_inr", ascending=False)

        # Stratify so every filter in the console has something in it. Pure
        # top-N by any single measure starves the rarer attack types and the
        # false positives, which are the two things a reviewer most wants to see.
        pool = [ranked.head(PRECOMPUTED_ALERTS // 2)]
        for archetype in ("card_testing", "refund_abuse", "bust_out"):
            pool.append(ranked[ranked["ring_type"] == archetype].head(60))
        pool.append(ranked[~ranked["is_fraud"]].head(60))
        pool.append(ranked[ranked["g_in_cluster"].fillna(0).astype(float) > 0].head(80))
        self.queue = (
            pd.concat(pool)
            .drop_duplicates(subset="payment_id")
            .sort_values("exposure_inr", ascending=False)
            .reset_index(drop=True)
        )
        self._attributions = attribute(self.arm, self.queue, self.reference)
        self._narrations: dict[str, dict] = {}

        self.results = {}
        results_path = REPORTS_DIR / "results.json"
        if results_path.exists():
            self.results = json.loads(results_path.read_text(encoding="utf-8"))
        self.ablation = _read_csv("ablation.csv")
        self.archetypes = _read_csv("per_archetype_at_p90.csv")
        self.verification = _read_csv("verification.csv")
        self.fp_composition = _read_csv("graph_fp_composition.csv")
        self.missed = _read_csv("graph_missed.csv")

    def narration(self, payment_id: str) -> dict:
        if payment_id in self._narrations:
            return self._narrations[payment_id]
        rows = self.queue[self.queue["payment_id"] == payment_id]
        if rows.empty:
            rows = self.test[self.test["payment_id"] == payment_id]
            if rows.empty:
                return {}
            attributions = attribute(self.arm, rows, self.reference)
        else:
            attributions = self._attributions[self._attributions["payment_id"] == payment_id]
        story = narrate(rows.iloc[0], attributions, top_k=5, reference=self.reference)
        self._narrations[payment_id] = story
        return story


def _read_csv(name: str) -> list[dict]:
    path = REPORTS_DIR / name
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def _clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _row_payload(row: pd.Series) -> dict:
    return {
        "payment_id": str(row["payment_id"]),
        "score": round(float(row["score"]), 4),
        "baseline_score": round(float(row["baseline_score"]), 4),
        "amount_inr": round(float(row["amount"]) / 100, 2),
        "exposure_inr": round(float(row.get("exposure_inr", 0.0)), 2),
        "day": int(row["day"]),
        "is_fraud": bool(row["is_fraud"]),
        "ring_type": str(row["ring_type"]),
        "verdict": str(row["ring_type"]) if bool(row["is_fraud"]) else "honest",
        "in_cluster": bool(_clean(row.get("g_in_cluster")) or 0),
        "account_age_days": _clean(row.get("account_age_days")),
    }


STATE: State | None = None


def state() -> State:
    global STATE
    if STATE is None:
        STATE = State()
    return STATE


async def console(request):
    return FileResponse(CONSOLE, media_type="text/html")


async def health(request):
    return JSONResponse({"status": "ok", "alerts_loaded": len(state().queue)})


async def summary(request):
    s = state()
    arms = s.results.get("arms", {})

    def arm_block(name: str) -> dict:
        payload = arms.get(name, {})
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

    return JSONResponse(
        {
            "test_set": {
                "payments": int(len(s.test)),
                "fraud": int(s.test["is_fraud"].sum()),
                "base_rate": round(float(s.test["is_fraud"].mean()), 5),
                "rings": int(s.test.loc[s.test["is_fraud"], "ring_id"].nunique()),
            },
            "baseline": arm_block("baseline"),
            "graph": arm_block("graph"),
            "ablation": s.ablation,
            "per_archetype": s.archetypes,
            "verification": s.verification,
            "false_positive_composition": s.fp_composition,
            "missed": s.missed,
        }
    )


async def alerts(request):
    s = state()
    params = request.query_params
    limit = min(int(params.get("limit", 50)), 400)
    verdict = params.get("verdict")

    frame = s.queue
    if verdict == "fraud":
        frame = frame[frame["is_fraud"]]
    elif verdict == "honest":
        frame = frame[~frame["is_fraud"]]
    elif verdict == "graph_backed":
        # Alerts where a cluster actually resolved. Worth its own filter: ranking
        # by money at risk surfaces high-value *first* orders, and a mule's first
        # order has no linked-account history yet, so the flagship view is
        # thinner on graph evidence than the aggregate numbers suggest.
        frame = frame[frame["g_in_cluster"].fillna(0).astype(float) > 0]
    elif verdict in ("card_testing", "refund_abuse", "bust_out"):
        frame = frame[frame["ring_type"] == verdict]

    rows = [_row_payload(row) for _, row in frame.head(limit).iterrows()]
    return JSONResponse({"count": len(rows), "alerts": rows})


async def alert_detail(request):
    s = state()
    payment_id = request.path_params["payment_id"]
    rows = s.test[s.test["payment_id"] == payment_id]
    if rows.empty:
        return JSONResponse({"error": "unknown payment_id"}, status_code=404)
    row = rows.iloc[0]

    story = s.narration(payment_id)
    payload = _row_payload(row)
    payload["summary"] = story.get("summary")
    payload["reasons"] = story.get("reasons", [])
    payload["graph_driven"] = story.get("graph_driven")
    payload["caveat"] = story.get("caveat")

    evidence = None
    try:
        raw_cluster = row.get("cluster")
        known = raw_cluster if isinstance(raw_cluster, str) and raw_cluster else None
        built = s.evidence.evidence_for(s.payments.loc[payment_id], cluster=known)
        if built is not None:
            evidence = built.to_dict()
    except Exception as exc:  # pragma: no cover - evidence is best-effort
        evidence = {"error": str(exc)}
    payload["evidence"] = evidence
    return JSONResponse(payload)


async def whatif(request):
    """Score with the graph evidence removed.

    This is the project's whole thesis made clickable: take the alert, replace
    every graph-derived feature with what a normal customer looks like, and show
    what the score would have been without it.
    """
    s = state()
    payment_id = request.path_params["payment_id"]
    rows = s.test[s.test["payment_id"] == payment_id]
    if rows.empty:
        return JSONResponse({"error": "unknown payment_id"}, status_code=404)

    with_graph = float(predict(s.arm, rows)[0])
    without_graph = float(predict(s.baseline, rows)[0])
    return JSONResponse(
        {
            "payment_id": payment_id,
            "with_graph": round(with_graph, 4),
            "without_graph": round(without_graph, 4),
            "delta": round(with_graph - without_graph, 4),
        }
    )


async def rings(request):
    """Clusters ranked by how much captured value sits inside them."""
    s = state()
    limit = min(int(request.query_params.get("limit", 25)), 100)
    frame = s.test[s.test["cluster"].notna() & (s.test["cluster"] != "")]
    if frame.empty:
        return JSONResponse({"count": 0, "rings": []})

    grouped = (
        frame.groupby("cluster")
        .agg(
            payments=("payment_id", "size"),
            accounts=("customer_id", "nunique"),
            amount_inr=("amount", lambda s_: float(s_.sum()) / 100),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            flagged_fraud=("is_fraud", "sum"),
            ring_risk=("cl_ring_risk", "max"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        )
        .reset_index()
    )
    grouped["exposure_inr"] = grouped["amount_inr"] * grouped["mean_score"]
    grouped = grouped.sort_values("exposure_inr", ascending=False).head(limit)
    records = json.loads(grouped.round(4).to_json(orient="records"))
    return JSONResponse({"count": len(records), "rings": records})


routes = [
    Route("/", console),
    Route("/api/health", health),
    Route("/api/summary", summary),
    Route("/api/alerts", alerts),
    Route("/api/alerts/{payment_id}", alert_detail),
    Route("/api/alerts/{payment_id}/whatif", whatif),
    Route("/api/rings", rings),
]

app = Starlette(routes=routes)
