"""The fact check is the feature, so this is where the tests point.

A language model that rephrases an evidence packet is worth very little. A
language model that *cannot* state anything the evidence packet does not
contain is worth something, and the only way to claim the second is to show the
verifier rejecting drafts that a human reviewer might have waved through.
"""
from __future__ import annotations

from ringfence.explain.narrative import (
    Facts,
    build_prompt,
    draft_for,
    facts_from_alert,
    template_draft,
    verify,
)

ALERT = {
    "payment_id": "pay_A1B2C3D4",
    "amount_inr": 3342.0,
    "score": 0.981,
    "account_age_days": 6,
    "reasons": [
        {"detail": "the whole cluster transacted inside 4 days"},
        {"detail": "device shared by 12 accounts"},
    ],
    "evidence": {
        "member_count": 12,
        "shared_identifiers": [
            {"link_label": "device", "accounts": 12, "pruned": False},
            {"link_label": "email domain", "accounts": 8100, "pruned": True},
        ],
    },
}


def facts() -> Facts:
    return facts_from_alert(ALERT)


def test_pruned_identifiers_never_reach_the_draft():
    """A hub dropped by the graph must not come back as a written claim.

    The email-domain link above was pruned as infrastructure precisely because
    8,100 accounts share it. Letting it into a merchant note would be asserting
    as evidence the one thing the model was told to ignore.
    """
    f = facts()
    assert any("device" in link for link in f.shared_identifiers)
    assert not any("email domain" in link for link in f.shared_identifiers)
    assert "8100" not in f.numbers()


def test_template_passes_its_own_check():
    f = facts()
    ok, problems = verify(template_draft(f), f)
    assert ok, problems


def test_invented_number_is_rejected():
    f = facts()
    draft = (
        "Payment pay_A1B2C3D4 for Rs 3,342 is on hold. It is linked to 47 other "
        "accounts and Rs 91,000 of activity."
    )
    ok, problems = verify(draft, f)
    assert not ok
    assert any("47" in p for p in problems)
    assert any("91000" in p or "91,000" in p for p in problems)


def test_altered_payment_reference_is_rejected():
    """The reference is what a merchant looks the case up by; it cannot drift."""
    f = facts()
    ok, problems = verify("Payment pay_WRONG99 for Rs 3,342 is on hold.", f)
    assert not ok
    assert any("reference" in p for p in problems)


def test_accusatory_language_is_rejected():
    """The system holds payments for review. It does not accuse people."""
    f = facts()
    draft = template_draft(f) + "\nThis account belongs to a fraudster."
    ok, problems = verify(draft, f)
    assert not ok
    assert any("accusatory" in p for p in problems)


def test_supported_numbers_survive_reformatting():
    """3342, 3,342 and 3342.00 are the same claim, and all three are allowed."""
    f = facts()
    for written in ("3342", "3,342", "3342.00"):
        ok, problems = verify(f"Payment pay_A1B2C3D4 for Rs {written} is on hold.", f)
        assert ok, (written, problems)


def test_prompt_carries_the_facts_and_forbids_the_rest():
    prompt = build_prompt(facts())
    assert "pay_A1B2C3D4" in prompt
    assert "Do not add any number" in prompt
    assert "Do not speculate" in prompt


def test_no_model_configured_falls_back_to_the_template(monkeypatch):
    """The default path for anyone cloning the repo, and what CI exercises."""
    monkeypatch.delenv("RINGFENCE_LLM_KEY", raising=False)
    result = draft_for(ALERT)
    assert result["source"] == "template"
    assert result["text"] == template_draft(facts())


def test_a_hallucinating_model_never_reaches_the_output(monkeypatch):
    """The one test that justifies shipping a model at all."""
    import ringfence.explain.narrative as narrative

    monkeypatch.setattr(
        narrative,
        "_call_model",
        lambda prompt: "Payment pay_A1B2C3D4 was blocked. We recovered Rs 4,500.",
    )
    result = narrative.draft_for(ALERT)
    assert result["source"] == "template"
    assert "4,500" not in result["text"]
    assert result["rejected_because"]


def test_a_clean_model_draft_is_used(monkeypatch):
    import ringfence.explain.narrative as narrative

    clean = "Payment pay_A1B2C3D4 for Rs 3,342 is on hold. It shares a device with 12 other accounts."
    monkeypatch.setattr(narrative, "_call_model", lambda prompt: clean)
    result = narrative.draft_for(ALERT)
    assert result["source"] == "model"
    assert result["text"] == clean


def test_generic_reasons_never_reach_a_merchant():
    """"Payment context" is a group name restated, not a reason a merchant can act on."""
    alert = dict(ALERT, reasons=[
        {"detail": "payment context", "generic": True},
        {"detail": "3 orders to this address in the past day", "generic": False},
    ])
    f = facts_from_alert(alert)
    assert f.reasons == ["3 orders to this address in the past day"]
    assert "payment context" not in template_draft(f)


def test_a_note_with_no_usable_reason_still_reads_as_a_sentence():
    """An empty "Why:" heading is worse than no heading, and this state is real."""
    alert = {
        "payment_id": "pay_Z9",
        "amount_inr": 500.0,
        "score": 0.7,
        "reasons": [{"detail": "payment context", "generic": True}],
        "evidence": {},
    }
    f = facts_from_alert(alert)
    draft = template_draft(f)
    assert "Why:" not in draft
    assert "combination of signals" in draft
    ok, problems = verify(draft, f)
    assert ok, problems


def test_the_closing_line_matches_what_the_alert_rests_on():
    """A payment held on velocity alone is not held on its connections."""
    linked = template_draft(facts())
    assert "connections to other accounts" in linked

    alone = template_draft(facts_from_alert({
        "payment_id": "pay_Z9", "amount_inr": 500.0, "score": 0.7,
        "reasons": [{"detail": "5 attempts from this device in the past day"}],
        "evidence": {},
    }))
    assert "connections to other accounts" not in alone
    assert "recent activity" in alone


def test_repeated_link_types_collapse_into_one_line():
    """Three handsets produced three near-identical bullets in a merchant note."""
    f = facts_from_alert(dict(ALERT, evidence={
        "member_count": 12,
        "shared_identifiers": [
            {"link_label": "device", "accounts": 4, "pruned": False},
            {"link_label": "device", "accounts": 7, "pruned": False},
            {"link_label": "device", "accounts": 5, "pruned": False},
        ],
    }))
    assert len(f.shared_identifiers) == 1
    assert "3 different devices" in f.shared_identifiers[0]
    assert "7 accounts" in f.shared_identifiers[0]


def test_internal_vocabulary_is_dropped_not_paraphrased():
    """"Combined edge weight 1.34" is a reason an analyst can check and a merchant cannot."""
    from ringfence.explain.narrative import to_merchant_language

    assert to_merchant_language("high-confidence shared identifiers (combined edge weight 1.34)") is None
    assert to_merchant_language("cluster ring-risk prior 0.82") is None


def test_our_words_become_their_words():
    from ringfence.explain.narrative import to_merchant_language

    assert to_merchant_language("the whole cluster transacted inside 4 days") == (
        "the linked accounts all transacted inside 4 days")
    assert "cluster" not in to_merchant_language("3 card per device across the linked cluster")
    assert to_merchant_language("3 card per device across the linked cluster").startswith("3 cards")


def test_a_rate_of_zero_is_not_a_reason():
    """"Declined payments among the linked accounts: 0%" invites the reply it deserves."""
    from ringfence.explain.narrative import to_merchant_language

    assert to_merchant_language("cluster decline rate 0%") is None
    assert to_merchant_language("cluster decline rate 18%") == (
        "declined payments among the linked accounts: 18%")
