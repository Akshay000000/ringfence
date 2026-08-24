# RingFence: Architecture

**Track 02 · AI Risk Manager · Razorpay AI Buildathon 2026**

> Abuse rings are not detectable one transaction at a time. RingFence builds an
> identity graph over payment traffic, finds the collusion structure inside it,
> and feeds that structure back into a per-transaction risk model, then proves
> the graph earned its place with a held-out ablation.

---

## 1. The problem, stated precisely

A merchant on Razorpay loses money to three things that look different in a
dashboard but are the same thing underneath:

| Loss class | What it looks like per-transaction | What it looks like in aggregate |
|---|---|---|
| Card testing | A few small failed payments | One device, 200 cards, 40 minutes |
| Promo / refund abuse | A normal first order, later refunded | 60 "new customers", one shipping address |
| Bust-out / triangulation | A large captured payment | 30 fresh accounts, shared drop address, chargebacks at T+45d |

Every one of these is **invisible to a transaction-level classifier** because the
evidence lives in the *relationships between accounts*, not in the account.
A single card-testing transaction is a ₹5 failed card payment. There is nothing
in that row to learn from. There is everything to learn from the fact that the
same device fingerprint tried 200 of them.

**RingFence's thesis:** relational features extracted from an identity graph
produce a measurable recall lift at fixed precision over the strongest tabular
baseline. The entire project is built to test that claim honestly and to be
wrong in public if it isn't.

## 2. The trap this project is designed to avoid

The naive version of this idea is: *link accounts that share an identifier, flag
the clusters.* That system is worthless in production, because normal commerce
is full of legitimate identifier sharing:

- A family shares one credit card across four accounts.
- A hostel, office, or mobile carrier NAT puts 400 unrelated customers behind one IP.
- A shared kiosk or a refurbished phone carries one device fingerprint.
- A housing society has one delivery address for many flats.

So the synthetic data **deliberately contains all four of these benign
structures**, at rates that make a naive connected-components detector produce
a false-positive flood. Two mechanisms handle them:

1. **IDF-weighted edges.** An identifier's evidential strength is inversely
   proportional to how many entities touch it. A shared card fingerprint across
   3 accounts is loud. A shared IP across 400 is nearly silent. Edge weight
   `w = log(N / df(identifier))`, thresholded. This is TF-IDF applied to
   identity resolution, and it is the single highest-leverage design decision
   in the system.
2. **The model decides, not the graph.** Ring membership is a *feature*, never a
   verdict. A dense ring of a family sharing a card has low velocity, low refund
   rate, and aged accounts; a bust-out ring does not. The gradient-boosted model
   learns that separation from labelled outcomes.

## 3. System design

```
                 ┌──────────────────────────────────────────┐
   Razorpay-     │  payments · refunds · disputes · orders   │
   shaped  ────► │  (synthetic, ground-truth labelled)       │
   event stream  └────────────────┬─────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │                                        │
      ┌───────▼────────┐                      ┌────────▼─────────┐
      │ Tabular        │                      │ Identity graph   │
      │ featureiser    │                      │ builder          │
      │ (velocity,     │                      │  nodes: customer │
      │  amount, MCC,  │                      │  device, card,   │
      │  method, hour) │                      │  email, phone,   │
      └───────┬────────┘                      │  ip, address     │
              │                               │  IDF-weighted    │
              │                               └────────┬─────────┘
              │                                        │
              │                               ┌────────▼─────────┐
              │                               │ Ring detection   │
              │                               │ Louvain + k-core │
              │                               │ → ring risk score│
              │                               └────────┬─────────┘
              │                                        │
              │                               ┌────────▼─────────┐
              │                               │ Graph features   │
              │                               │ (as-of time t,   │
              │                               │  strictly causal)│
              │                               └────────┬─────────┘
              └────────────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Risk model          │
                    │  A: tabular only     │  ◄── the ablation
                    │  B: tabular + graph  │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼──────────────────┐
              │ Cost-weighted decision layer      │
              │  allow / review / block           │
              │  threshold minimises expected ₹   │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │ Evidence layer                    │
              │  subgraph + SHAP → reason string  │
              └───────────────────────────────────┘
```

## 4. Leakage control: the part that decides whether the numbers are real

This is where most hackathon fraud projects quietly cheat. Three rules:

**R1: temporal split, not random split.** Train on days 0-89, validate on
90-109, test on 110-149. A random split leaks the future: the same ring appears on
both sides and the model memorises ring IDs rather than ring *shape*.

**R2: graph features are computed as-of the transaction's day, from prior days
only.** The graph is snapshotted daily. A transaction on day 80 sees the graph
built from days 0-79. It never sees its own edge, and never sees a ring-mate's
later chargeback. Implemented by construction, not by hope.

**R3: ring-disjoint reporting.** Test-set rings that also appear in the training
window are reported separately from rings that are entirely novel. The headline
number is the **novel-ring** number, because that is what production looks like
on the day a new fraud crew arrives.

## 5. Cost model: why F1 is the wrong objective

Precision and recall are not the merchant's objective function. Rupees are.

```
expected_cost(threshold) =
      FN(t) · (avg_fraud_amount + chargeback_fee)      # fraud we let through
    + FP(t) · (avg_order_value · margin · churn_factor) # good customers we blocked
    + REVIEW(t) · manual_review_cost                    # analyst minutes
```

A false positive is not free and is not symmetric with a false negative: a
blocked good customer costs the margin on that order *plus* a probability-
weighted lifetime value hit. RingFence selects its operating threshold by
minimising this curve, reports the ₹ saved against a do-nothing baseline, and
publishes the whole curve so a reviewer can substitute their own cost
assumptions and see the answer move.

## 6. Defense-only guarantees

The track disqualifies anything offense-capable. RingFence is structurally
incapable of offense, and this is enforced, not promised:

- The system consumes transaction records and emits **risk scores and evidence**.
  It has no write path to any payment, refund, or account API.
- The synthetic generator produces **labelled abuse patterns for evaluation
  only**; it does not produce working card numbers, real BINs, live credentials,
  or bypass techniques. Card "fingerprints" are opaque random tokens.
- No component recommends how to evade detection, and the repo contains no
  attack tooling.
- The dataset is entirely synthetic. No real cardholder data ever enters the
  system.

## 7. Deliverables against the track bar

| Track requirement | How RingFence satisfies it |
|---|---|
| "A working detector, verifier or auto-responder" | Detector: FastAPI scoring service + analyst console |
| "One class of loss" | Collusive abuse rings (card testing, refund abuse, bust-out) |
| "Measured precision and recall on a held-out test set" | Temporal held-out split, PR curves, per-archetype breakdown |
| "Honest metrics including false-positive cost" | Explicit ₹ cost model, expected-cost curve, published FP burden |
| "Strictly defense-only" | Section 6, enforced by architecture |

## 8. Repository layout

```
ringfence/
  datagen/     synthetic Razorpay-shaped traffic + injected rings + benign confounders
  graph/       identity graph construction, IDF weighting, community detection
  features/    tabular featuriser and strictly-causal graph featuriser
  model/       baseline and graph-augmented training, ablation runner
  evaluation/  metrics, PR curves, cost curves, honest exception reporting
  explain/     evidence subgraph + SHAP → analyst-readable reason
  api/         FastAPI scoring service
configs/       one YAML controls every knob; nothing is hardcoded in a notebook
reports/       generated artefacts, all regenerable with `make all`
```

## 9. Reproducibility contract

Every number in the README and the pitch deck is produced by `make all` from a
fixed seed on a clean checkout. No notebook-only results, no hand-copied
figures, no cherry-picked example. If a number cannot be regenerated, it does
not go in the pitch.
