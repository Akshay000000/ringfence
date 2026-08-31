"""Turn an evidence packet into something a merchant can read.

The rest of this project deliberately contains no language model. Detection over
structured data at scale is not a job for one: it would be slower, costlier and
non-deterministic, and using one there would be decoration.

This is the one place the job is genuinely linguistic. An analyst has a decision
and a pile of structured evidence, and needs a short written explanation to send
to a merchant or attach to a chargeback representment. That is a writing task.

The danger is obvious. A model that invents a detail in a document sent to a
merchant, or filed with a card scheme, is worse than no document at all. So the
design is not "ask a model to explain the alert". It is:

  1. assemble a closed set of facts from the evidence packet;
  2. ask the model to rephrase *only those facts*, with no new ones;
  3. verify the output against the fact set and reject it if it drifts;
  4. fall back to a deterministic template whenever the check fails, or when no
     model is configured at all.

The verifier is the actual engineering here. Every number in the draft must
appear in the facts, and every entity it names must have been supplied. If the
model adds a figure, invents an amount, or hallucinates a merchant name, the
draft is thrown away and the template is used instead. The system is never worse
than the template, and the model can only make it read better.

Nothing here is required to run RingFence. With no API key configured the
template path runs, which is also what CI exercises.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Words a draft may use freely: they carry no factual claim.
_SAFE_NUMERIC_WORDS = {"one", "two", "three", "a", "an", "the", "first", "second"}


# The analyst vocabulary is not the merchant vocabulary.
#
# The reason strings are written for somebody looking at a graph. "Combined edge
# weight 1.34" and "cluster ring-risk prior 0.82" are meaningful there and mean
# nothing in a note to a shop owner whose payout is on hold, where they read as
# jargon deployed to sound authoritative.
#
# Two rules. Anything purely internal is dropped rather than paraphrased, since a
# reason a merchant cannot check is not a reason worth giving. Everything else
# has "cluster", which is our word, replaced with "linked accounts", which is
# theirs.
_DROP_FROM_MERCHANT_NOTES = (
    "edge weight",
    "ring-risk prior",
)

# Link labels carry the same problem. "email (alias-normalised)" tells an analyst
# that plus-addressing and dots were stripped before comparing, which is the
# right label on a graph edge and noise in a sentence.
_LINK_LABELS = {
    "email (alias-normalised)": "email address",
    "device": "device",
    "ip address": "internet connection",
    "shipping address": "delivery address",
}

_PLAIN = (
    ("the whole cluster transacted inside", "the linked accounts all transacted inside"),
    ("across the linked cluster", "across the linked accounts"),
    ("across the cluster", "across the linked accounts"),
    ("cluster decline rate", "declined payments among the linked accounts:"),
    ("cluster averages", "the linked accounts average"),
    ("an identifier shared with", "a detail on this order is shared with"),
    ("a cluster of", "a group of"),
    ("cluster", "linked accounts"),
)


def to_merchant_language(detail: str) -> str | None:
    """Rewrite one reason for a merchant, or drop it if it cannot be rewritten."""
    text = str(detail or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _DROP_FROM_MERCHANT_NOTES):
        return None
    # A rate of zero is a signal to the model and nonsense to a reader. "Declined
    # payments among the linked accounts: 0%" was going out as a reason to hold a
    # payment, which invites exactly the reply it deserves.
    if re.search(r"\b0(\.0+)?%", text):
        return None
    for internal, plain in _PLAIN:
        text = text.replace(internal, plain)
    # The reason strings say "3 card per device" because the ratio is what the
    # feature means. In a sentence a person reads, it is a typo.
    text = re.sub(r"\b(\d+) (card|account|payment) per\b",
                  lambda m: f"{m.group(1)} {m.group(2)}{'' if m.group(1) == '1' else 's'} per",
                  text)
    return text


@dataclass
class Facts:
    """The closed set of claims a draft is allowed to make."""

    payment_id: str
    amount_inr: float
    decision: str
    score: float
    reasons: list[str] = field(default_factory=list)
    linked_accounts: int | None = None
    shared_identifiers: list[str] = field(default_factory=list)
    cluster_day_span: int | None = None
    account_age_days: int | None = None

    def numbers(self) -> set[str]:
        """Every numeric token a draft is permitted to contain."""
        out: set[str] = set()

        def add(value):
            if value is None:
                return
            if isinstance(value, float):
                out.add(f"{value:.0f}")
                out.add(f"{value:.2f}")
                out.add(f"{value:,.0f}")
                out.add(f"{value:.3f}")
            else:
                out.add(str(value))

        add(self.amount_inr)
        add(self.score)
        add(self.linked_accounts)
        add(self.cluster_day_span)
        add(self.account_age_days)
        for text in list(self.reasons) + list(self.shared_identifiers):
            out.update(re.findall(r"\d[\d,]*\.?\d*", str(text)))
        return {n.replace(",", "") for n in out if n}


def _plural(label: str) -> str:
    """Enough pluralisation for four link labels. Not a general solution."""
    return label + ("es" if label.endswith(("s", "x", "ch", "sh")) else "s")


def facts_from_alert(alert: dict) -> Facts:
    """Read a console alert payload into the closed fact set."""
    evidence = alert.get("evidence") or {}

    # Collapse by link type. The raw evidence lists every shared identifier, so a
    # cluster held together by three handsets produced three near-identical
    # bullets ("device shared by 4 accounts", "device shared by 5 accounts") in a
    # note meant for a merchant. One line per type, carrying the widest sharing,
    # says the same thing and reads like a sentence.
    #
    # Identifiers the graph pruned as hubs are dropped here, not merely ranked
    # below. Letting gmail.com back in as a written claim would assert as
    # evidence the exact thing the graph threw away for being meaningless.
    widest: dict[str, int] = {}
    counts: dict[str, int] = {}
    for link in evidence.get("shared_identifiers") or []:
        if link.get("pruned"):
            continue
        raw_label = str(link.get("link_label") or "shared detail")
        label = _LINK_LABELS.get(raw_label.lower(), raw_label)
        accounts = int(link.get("accounts", 0) or 0)
        widest[label] = max(widest.get(label, 0), accounts)
        counts[label] = counts.get(label, 0) + 1

    links = []
    for label, accounts in sorted(widest.items(), key=lambda kv: -kv[1])[:4]:
        if counts[label] > 1:
            links.append(
                f"{counts[label]} different {_plural(label)}, one of them shared "
                f"by {accounts} accounts"
            )
        else:
            links.append(f"{label} shared by {accounts} accounts")
    return Facts(
        payment_id=str(alert.get("payment_id")),
        amount_inr=float(alert.get("amount_inr") or 0.0),
        decision="hold for review" if alert.get("score", 0) >= 0.5 else "allow",
        score=round(float(alert.get("score") or 0.0), 3),
        # Generic reasons are dropped. When no evidence line cleared the
        # abnormality floor, `detail` is the group name restated, and "payment
        # context" tells a merchant nothing while looking like it should.
        reasons=[
            plain
            for plain in (
                to_merchant_language(r.get("detail", ""))
                for r in (alert.get("reasons") or [])
                if not r.get("generic")
            )
            if plain
        ][:3],
        linked_accounts=evidence.get("member_count"),
        shared_identifiers=links,
        account_age_days=alert.get("account_age_days"),
    )


def template_draft(facts: Facts) -> str:
    """The deterministic fallback. Correct by construction, if a little stiff."""
    bullets = [f"  - {reason}" for reason in facts.reasons if reason]
    if facts.linked_accounts:
        bullets.append(
            f"  - this account is linked to {facts.linked_accounts} others through "
            f"shared details"
        )
    bullets += [f"  - {link}" for link in facts.shared_identifiers]

    lines = [
        f"Payment {facts.payment_id} for Rs {facts.amount_inr:,.0f} has been placed "
        f"on {facts.decision}.",
        "",
    ]
    # A "Why:" heading with nothing under it is worse than no heading. It happens
    # when every reason was generic and no identifier resolved, which is a real
    # state the model reaches and not one to paper over.
    if bullets:
        lines.append("Why:")
        lines += bullets
    else:
        lines.append(
            "The reason is a combination of signals rather than any single one, "
            "so an analyst will look at it directly."
        )
    # The closing line has to match what the alert actually rests on. Saying
    # "based on connections to other accounts" under a payment held purely on
    # velocity is a small lie, and a merchant who checks will find it.
    basis = (
        "the account's connections to other accounts"
        if facts.linked_accounts or facts.shared_identifiers
        else "this account's recent activity"
    )
    lines += [
        "",
        f"This is an automated risk assessment based on {basis}. If you believe "
        "it is wrong, reply to this message and an analyst will review it.",
    ]
    return "\n".join(lines)


def build_prompt(facts: Facts) -> str:
    """A prompt that supplies the facts and forbids anything else."""
    bullets = "\n".join(f"- {r}" for r in facts.reasons if r)
    links = "\n".join(f"- {l}" for l in facts.shared_identifiers)
    return (
        "Write a short, plain, non-accusatory note to a merchant explaining why a "
        "payment was held for review.\n\n"
        "Rules:\n"
        "- Use ONLY the facts below. Do not add any number, name, date or claim "
        "that is not listed.\n"
        "- Do not speculate about intent or accuse anyone of fraud.\n"
        "- Three short sentences maximum, then the reasons as bullets.\n"
        "- Plain English, no jargon.\n\n"
        f"Facts:\n"
        f"- Payment reference: {facts.payment_id}\n"
        f"- Amount: Rs {facts.amount_inr:,.0f}\n"
        f"- Decision: {facts.decision}\n"
        f"- Linked accounts: {facts.linked_accounts}\n"
        f"{bullets}\n{links}\n"
    )


def verify(draft: str, facts: Facts) -> tuple[bool, list[str]]:
    """Reject a draft that states anything the facts do not support.

    Deliberately blunt. It checks numbers, because a wrong number in a document
    sent to a merchant or a card scheme is the failure that actually matters, and
    it checks that the payment reference survived intact.
    """
    problems: list[str] = []
    allowed = facts.numbers()

    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", draft):
        clean = token.replace(",", "").rstrip(".")
        if clean in allowed:
            continue
        # A number that is part of the payment reference is fine.
        if clean and clean in facts.payment_id:
            continue
        problems.append(f"unsupported number: {token}")

    if facts.payment_id not in draft:
        problems.append("payment reference missing or altered")

    for banned in ("fraudster", "criminal", "stolen", "guilty", "scam"):
        if banned in draft.lower():
            problems.append(f"accusatory language: {banned}")

    return (not problems), problems


def _call_model(prompt: str) -> str | None:
    """Call an OpenAI-compatible endpoint if one is configured, else give up.

    Configuration is by environment only, so the repo carries no key and the
    default path for anyone cloning it is the template.
    """
    key = os.environ.get("RINGFENCE_LLM_KEY")
    if not key:
        return None
    base = os.environ.get("RINGFENCE_LLM_BASE", "https://api.openai.com/v1")
    model = os.environ.get("RINGFENCE_LLM_MODEL", "gpt-4o-mini")
    try:  # pragma: no cover - requires network and a key
        import json
        import urllib.request

        body = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def draft_for(alert: dict) -> dict:
    """Produce a merchant-facing note, and say honestly how it was produced."""
    facts = facts_from_alert(alert)
    fallback = template_draft(facts)

    generated = _call_model(build_prompt(facts))
    if generated is None:
        return {"text": fallback, "source": "template",
                "note": "no language model configured; deterministic template used"}

    ok, problems = verify(generated, facts)
    if not ok:
        return {"text": fallback, "source": "template",
                "note": "model draft rejected by the fact check",
                "rejected_because": problems}
    return {"text": generated, "source": "model", "note": "model draft passed the fact check"}
