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
   ┌──────────────────────┐        ┌──────────────────────┐
   │ synthetic generator  │        │ IEEE-CIS adapter     │
   │ ring-labelled        │        │ real Vesta payments  │
   └──────────┬───────────┘        └──────────┬───────────┘
              └────────────┬──────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ canonical payment schema│   one contract, any source
              └────────────┬────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
 ┌────────────────┐                 ┌──────────────────┐
 │ Tabular        │                 │ Identity graph   │
 │ featuriser     │                 │  nodes: customer │
 │ velocity,      │                 │  card, device,   │
 │ entity counts, │                 │  address, phone, │
 │ amount, tenure │                 │  email, IP       │
 └───────┬────────┘                 │  IDF-weighted    │
         │                          └────────┬─────────┘
         │                                   ▼
         │                          ┌──────────────────┐
         │                          │ Ring detection   │
         │                          │ components then  │
         │                          │ clique-safe      │
         │                          │ Louvain          │
         │                          └────────┬─────────┘
         │                                   ▼
         │                          ┌──────────────────┐
         │                          │ Graph features   │
         │                          │ as-of day d,     │
         │                          │ strictly causal  │
         │                          └────────┬─────────┘
         └─────────────────┬─────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Risk model              │
              │  A: tabular only        │ ◄── the ablation
              │  B: tabular + graph     │
              └────────────┬────────────┘
                           ▼
        ┌──────────────────┴───────────────────┐
        ▼                                      ▼
 ┌──────────────────┐              ┌──────────────────────┐
 │ Cost-weighted    │              │ Evidence layer       │
 │ decision         │              │ group occlusion +    │
 │ allow/review/    │              │ subgraph, masked     │
 │ block, minimises │              └──────────┬───────────┘
 │ expected rupees  │                         ▼
 └────────┬─────────┘              ┌──────────────────────┐
          └───────────────────────►│ Read-only console    │
                                   │ queue ranked by      │
                                   │ money at risk        │
                                   └──────────────────────┘
```

## 4. The dataset abstraction

The graph, featurisers, model and evaluation were written against the synthetic
generator. Rather than fork them to run a second dataset, an adapter converts the
source into one canonical payment schema and the whole pipeline runs unchanged.

`ringfence/datasets/schema.py` defines the contract in three tiers:

| tier | meaning | behaviour when a dataset lacks it |
|---|---|---|
| REQUIRED | payment id, customer, day, timestamp, amount, label | the adapter fails loudly |
| LINK | card, device, address, phone, email root, IP | the column is null and the graph never links on it |
| DERIVED | status, method, tenure, refund and dispute events | filled with a neutral value, so the feature goes inert rather than lying |
| GROUND_TRUTH | ring id, ring type, benign cluster | absent on real data, and reported as unavailable rather than approximated |

That last row is the one that matters. IEEE-CIS has fraud labels but no ring
labels, so per-archetype recall and novel-ring reporting genuinely cannot be
computed there. The evaluation prints "unavailable, dataset has no ring labels"
instead of quietly producing a meaningless number.

Each dataset also owns its workspace: `data/<name>/` and `reports/<name>/`.
Before that split existed the two runs overwrote each other's `payments` and
`results.json`, which is exactly how a "real data" figure ends up sourced from
synthetic artefacts.

## 5. Leakage control: the part that decides whether the numbers are real

This is where most hackathon fraud projects quietly cheat. Three rules:

**R1: temporal split, not random split.** Synthetic trains on days 0-89,
validates on 90-109, tests on 110-149. IEEE-CIS uses 0-119 / 120-149 / 150-181.
A random split leaks the future: the same ring appears on both sides and the
model memorises ring IDs rather than ring *shape*.

**R2: graph features are computed as-of the transaction's day, from prior days
only.** The graph is snapshotted daily. A transaction on day 80 sees the graph
built from days 0-79. It never sees its own edge, and never sees a ring-mate's
later chargeback. Implemented by construction, not by hope.

**R3: ring-disjoint reporting.** Test-set rings that also appear in the training
window are reported separately from rings that are entirely novel. The headline
number is the **novel-ring** number, because that is what production looks like
on the day a new fraud crew arrives.

**R4: the category vocabulary is fitted on training data only.** An
over-cardinality categorical has to fold its rare levels into "other". Computing
which levels survive from whichever split is being transformed lets the test set
help decide its own encoding. Mild, and still leakage.

## 6. Statistical discipline

Three modules exist because a number was believed once and turned out to be
noise.

**`evaluation/seed_study.py`.** A single-seed run showed the graph ahead by
+2.9% on IEEE-CIS. Refitting across five seeds showed the run-to-run standard
deviation was three times the size of that effect. Every ablation claim now
reports its gap in pooled standard deviations, and anything under 2 sd is
written down as "no measurable difference".

**`evaluation/label_noise.py`.** Both arms train on ground-truth labels, which
production never has. Corrupting a fraction of the *training* labels while
leaving test labels clean answers whether the advantage survives. Two modes:
symmetric flipping, and the realistic one, fraud that was never caught and
therefore trains as legitimate.

**`evaluation/subgroup.py`.** When the real-data run returned a null, the
explanation offered for it made a falsifiable prediction: the advantage should
grow with how much linked-account structure a payment sits in. This module tests
that. It found no trend, and the explanation was withdrawn.

## 7. Cost model: why F1 is the wrong objective


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

## 8. Explainability

A score is not actionable and "the model said so" is not something to put in
front of a merchant whose payment was blocked. Two layers, in
`ringfence/explain/`.

**Reasons, by group occlusion rather than SHAP.** Features are bucketed into
semantic groups (velocity, card fan-out, address concentration, cohort
synchrony, cluster tempo) and each group is attributed by replacing it wholesale
with what a normal customer looks like and re-scoring. That produces a real
counterfactual on the real model: "if this account's connections had looked
ordinary, the score would have been 0.31 instead of 0.99". It reads as an
analyst sentence rather than 88 feature names, and it drops a heavy dependency.
It is also honest about its limit: occluding groups one at a time cannot
untangle interactions, so contributions are ranked and never totalled.

Where no cluster resolved, the graph groups are dropped from the reasons and a
caveat is stated instead. Occlusion still produces a large contribution for them
on those rows, because replacing missing cluster features with honest medians
makes the payment look more normal, but reporting that as a top reason would
tell an analyst the payment is suspicious *because* nothing is known about it.

**Evidence subgraphs.** Each alert can rebuild the snapshot it was scored
against and return the cluster: members, the identifiers linking them with the
weight each contributed, and how this payment attached (its own history, or
inherited through an identifier it touched). Identifier values are masked,
because the packet is meant to travel into a ticket. Identifiers pruned as hubs
are listed as pruned and are never drawn as edges, since they contributed none.

## 9. Serving

`ringfence/api/` is a Starlette service plus a single-file console. FastAPI runs
on Starlette anyway and for six routes its request-model machinery buys nothing.

Two decisions worth recording, both because the first version was wrong:

**The queue ranks by money at risk, not score.** Sorting on raw score produced
392 card-testing probes out of 400, every one tied at 0.9998 and worth about
five rupees. Correct ranking, useless product. `score x amount` puts a large
bust-out order above a small probe, which is how a risk analyst triages.

**Alerts are marked when linked-account evidence exists.** Ranking by money
surfaces high-value *first* orders, and a mule's first order has no
linked-account history yet, so the flagship view is thinner on graph evidence
than the aggregate numbers suggest. Shown, not hidden.

## 10. Defense-only guarantees

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

## 11. Deliverables against the track bar

| Track requirement | How RingFence satisfies it |
|---|---|
| "A working detector, verifier or auto-responder" | Detector: Starlette scoring service + analyst console |
| "One class of loss" | Collusive abuse rings (card testing, refund abuse, bust-out) |
| "Measured precision and recall on a held-out test set" | Temporal held-out split, PR curves, per-archetype breakdown, every gap checked against seed noise |
| "Honest metrics including false-positive cost" | Explicit rupee cost model where a false block costs margin x LTV churn, expected-cost curve, published sensitivity, and a reported null on real data |
| "Strictly defense-only" | Section 10, enforced by architecture |

## 12. Repository layout

```
ringfence/
  datasets/     canonical payment schema + the IEEE-CIS adapter
  datagen/      synthetic Razorpay-shaped traffic, rings, benign confounders
  graph/        identity graph, IDF weighting, clique-safe communities
  features/     tabular featuriser and strictly-causal graph featuriser
  model/        baseline and graph-augmented arms, explicit leakage allowlist
  evaluation/   metrics, cost curves, verification, seed study, label noise,
                subgroup analysis
  explain/      group-occlusion reasons + masked evidence subgraphs
  api/          read-only Starlette service and single-file analyst console
configs/        default.yaml (synthetic), ieee_cis.yaml (real); nothing hardcoded
data/<name>/    per-dataset working files, gitignored
reports/<name>/ per-dataset artefacts, all regenerable
tests/          regression guards, each encoding a bug that actually occurred
```

## 13. Reproducibility contract

Every number in the README and the pitch deck is produced by `make all` from a
fixed seed on a clean checkout. No notebook-only results, no hand-copied
figures, no cherry-picked example. If a number cannot be regenerated, it does
not go in the pitch.

The real-data run is reproduced the same way, once the Kaggle files are in
`data/raw/ieee/`:

```bash
python -m ringfence.cli --config configs/ieee_cis.yaml all
python -m ringfence.cli --config configs/ieee_cis.yaml seedstudy
```

The raw Kaggle download is not redistributable and is gitignored. The repo
carries the adapter, not the data.
