# RingFence

**Graph-based abuse-ring detection for merchant payments.**
Razorpay AI Buildathon 2026 · Track 02 — AI Risk Manager

Abuse rings are not detectable one transaction at a time. RingFence builds an
identity graph over payment traffic, finds the collusion structure inside it,
and feeds that structure back into a per-transaction risk model — then proves
the graph earned its place with a held-out, leakage-audited ablation.

---

## Validated on two datasets, with opposite outcomes

| | synthetic corpus | IEEE-CIS (Vesta) |
|---|---|---|
| what it is | 458k payments, ring-labelled, benign confounders planted | 590k **real** card-not-present transactions, 3.5% fraud |
| what it can measure | ring detection against ground-truth ring membership | whether the mechanism transfers to real money |
| graph vs baseline | **+24.5%** recall at precision 0.95 | **no measurable difference** (0.4 pooled sd, 5 seeds) |

Both are in the repo, run by the same pipeline, and both are reported.

I then offered an explanation for the null — that IEEE-CIS fraud is single-actor
rather than collusive — and **tested it**. If that were right, the graph's
advantage should grow with how much linked-account structure a payment sits in.
Bucketed by cluster size across five seeds, there is no such trend, and the only
significant result is the graph doing *worse* on small clusters. The explanation
was withdrawn.

> The graph produces a large, verified improvement on a benchmark whose ring
> structure I constructed. It produces no measurable improvement on IEEE-CIS, and
> a targeted search for the subgroup where it should have helped found nothing.
> Why it does not transfer is **unresolved**.

That is the honest state of it. What would settle the question is a real dataset
with confirmed *collusion* labels rather than transaction-level fraud flags;
IEEE-CIS was the closest public dataset with an identity surface and it does not
have them. Until then the positive result reads as **mechanism demonstrated under
constructed conditions**, not as evidence of production accuracy.

See **[FINDINGS.md](FINDINGS.md) § F8 and § F9**.

---

## What it saves the merchant (synthetic corpus)

Held-out **temporal** test split: 187,149 payments, 1,649 fraudulent (0.88%),
42 abuse rings — **none of which appear in the training window.** ₹41.4L of fraud
exposure sitting in it.

| | do nothing | tabular baseline | **+ identity graph** |
|---|---|---|---|
| total cost — fraud, chargebacks, false blocks, review | ₹41.4L | ₹15.2L | **₹6.2L** |
| fraud caught | 0% | 87.9% | **92.2%** |
| good customers wrongly blocked | 0 | 564 | **49** |
| alerts sent to a human | 0 | 1.08% of traffic | **0.84%** |
| **net saving vs doing nothing** | — | ₹26.2L | **₹35.2L** |

The graph arm recovers **85% of the exposure** and blocks **11× fewer real
customers** while catching more fraud and queueing fewer alerts. The saving holds
between 82% and 87% across every churn assumption tested — the full sensitivity
table ships in `reports/synthetic/graph_sensitivity.csv`, so a reviewer can
substitute their own cost model and watch the answer move.

A false positive is not free and is not the mirror of a false negative: a blocked
good customer costs the margin on that order *plus* a probability-weighted
lifetime-value hit. That asymmetry is what picks the operating threshold here,
not F1.

### The detection numbers underneath

Same algorithm, same rows, same seed. One feature block apart.

| precision target | baseline recall | **+ graph** | lift |
|---|---|---|---|
| 0.95 | 0.747 | **0.929** | **+24.5%** |
| 0.90 | 0.754 | **0.940** | **+24.7%** |
| 0.80 | 0.824 | **0.950** | +15.2% |
| 0.60 | 0.913 | **0.965** | +5.6% |

PR-AUC 0.9075 → **0.9712**.

### Where the lift actually comes from

Both arms pinned to precision 0.90:

| archetype | baseline | + graph | lift |
|---|---|---|---|
| refund abuse | 0.428 | **0.922** | +0.494 |
| bust-out / triangulation | 0.543 | **0.853** | +0.310 |
| card testing | 0.973 | 0.986 | +0.012 |

**The graph adds essentially nothing to card testing** — velocity counters
already catch it at 97%, and a card-testing burst resolves in ~2 days, faster
than any graph snapshot can form. The graph earns its place on the slow,
distributed attacks where no single transaction looks wrong. That is a narrower
claim than "graphs improve fraud detection", and it is the one the data supports.

---

## The idea in one diagram

```
   payment arrives
         │
         ├──────────────► tabular featuriser ──────┐
         │                velocity counters,       │
         │                per-entity history       │
         │                                         ▼
         │                                   ┌───────────┐
         └──► identity graph, as of yesterday│ risk model│──► allow
              ├ IDF-weighted edges           └───────────┘    review
              ├ clique-safe communities            │          block
              ├ cohort synchrony                   │
              └ resolve by IDENTIFIER, not         ▼
                just by customer              evidence:
                                              subgraph + reasons
```

Three design decisions carry the result.

**1. IDF-weighted edges.** Shared identifiers are common and mostly innocent.
An identifier's evidential strength is `prior[type] × log(N/df(v)) / log(N)`:
a card fingerprint on 3 accounts is loud, an IP on 400 accounts is silent.
Hub identifiers past a per-type cap are dropped outright, not down-weighted —
a carrier NAT block carries no information at any weight, and keeping it injects
an 80,000-edge clique that merges the whole customer base into one blob.

**2. Cohort synchrony.** A drop address and an apartment block are structurally
identical in a graph — both are ~20 accounts on one address. What separates them
is *when the accounts were created*. A building's residents signed up over
years; a mule cohort signed up the same week. Without this feature the graph
actively **hurt** bust-out recall.

**3. Resolve by identifier, not by customer.** A mule account's first payment
has no history, so a customer-keyed lookup returns nothing for exactly the
transactions that matter most. Resolving through the drop address the payment
ships to — an address already inside a scored cluster — lifted fraud-row
coverage from ~40% to ~60% and flipped the ablation from −4.6% to +23.9%.

---

## Honesty, and how it is enforced

Most of the engineering in this repo is spent making the number believable
rather than making it big.

**Temporal split, not random.** Train days 0–89, validate 90–109, test 110–149.
A random split leaks the future and lets the model memorise ring IDs instead of
ring shape.

**Graph features are strictly causal by construction.** A payment on day *d* is
described only by a graph built from `[d-45, d)`. It never contributes an edge
to the graph that describes it, and it can never see a ring-mate's later
chargeback.

**Label maturity is modelled.** A chargeback on a day-80 payment does not exist
until day ~125. Every payment carries the day its outcome became observable, and
training rows are restricted to labels that had matured by `as_of_day`. Most
fraud demos quietly skip this.

**Novel-ring reporting.** The headline recall is measured on rings the model has
never seen. A model that only recognises rings it has already met is a lookup
table.

**The false-positive cost is a rupee figure, not a ratio.** F1 is not the
merchant's objective. A blocked good customer costs the margin on the order
*plus* a probability-weighted lifetime-value hit, review time is charged per
alert, and the comparison baseline is do-nothing. The full cost curve and a
sensitivity table across the most arguable assumption ship with the results.

**Every ablation claim must clear its own noise floor.** A single-seed run on
IEEE-CIS showed the graph ahead by +2.9%; refitting across five seeds showed the
run-to-run standard deviation was three times that. Both apparent lifts were
noise. `python -m ringfence.cli seedstudy` refits both arms across seeds and
reports the gap in pooled standard deviations, and anything under 2 sd is written
down as "no measurable difference".

**The advantage survives labels being wrong.** With 20% of fraud going
unlabelled in training — the realistic failure of any review queue — the gap
narrows from +0.062 to +0.037 PR-AUC and stays 20 standard deviations wide
(`python -m ringfence.cli labelnoise`, FINDINGS § F10).

**Six verification checks** run in the pipeline, which refuses to report numbers
if any fails:

| check | result |
|---|---|
| V1 no forbidden column reaches the model | pass |
| V2 graph window strictly earlier than the payments it scores | pass |
| V3 **label permutation collapses the signal** | pass — permuted PR-AUC 0.0075 vs base rate 0.0088 (0.86×) |
| V4 test rings are novel | pass — 42 rings, 0 seen in training |
| V5 training labels had matured by `as_of_day` | pass |
| V6 cluster statistics never reference an outcome column | pass |

V3 is the one that matters: if the +24% lift were leakage, a model fitted on
shuffled labels would still find it. It scores *below* the base rate.

**[FINDINGS.md](FINDINGS.md) documents the four times this build produced a
number that looked fine and was wrong**, including a PR-AUC of 0.99 that meant
the benchmark was broken, and a Louvain resolution setting that was shattering
exactly the cliques it existed to find.

---

## Every alert is contestable

A score is not actionable on its own, and "the model said so" is not something
you can put in front of a merchant whose payment you just blocked. Each alert
carries a reason and the evidence behind it.

**Reasons** come from group occlusion, not SHAP. Features are bucketed into
semantic groups — velocity, card fan-out, address concentration, cohort
synchrony — and each is attributed by replacing it wholesale with what a normal
customer looks like and re-scoring. That yields a genuine counterfactual on the
actual model, and it removes a heavy dependency. It is also honest about its
limit: occluding groups one at a time cannot untangle interactions, so
contributions are ranked, never totalled.

**Evidence** is the subgraph the alert came from — the linked accounts, the
identifiers tying them together with the weight each contributed, and how this
payment attached to the cluster. Identifier values are masked, because the
packet is meant to travel into a ticket.

A refund-abuse ring, as the system explains it:

```
Scored 1.000. Driven by the whole cluster transacted inside 4 days.

  cluster cl_239 as of day 134 (12 linked accounts)
  attached: own prior activity in this cluster
  linked by:
    device                     dev…bWc        12 accounts   weight 0.66
    email (alias-normalised)   m1w…com         3 accounts   weight 0.58
    shipping address           add…Zw2        12 accounts   weight 0.58
    IP address                 97.…239        12 accounts   weight 0.27
```

Twelve "different customers", one handset, one delivery address, and three of
the email addresses collapsing to the same inbox once plus-aliasing is
normalised.

`python -m ringfence.cli explain` writes these for the top alerts **and for the
highest-scoring false positives**, which is the more useful half of the report.
The worst one is a two-day-old account sharing a card with a 343-day-old
account — a family, not a ring. Exactly the confounder the generator plants, and
the system falls for it.

---

## The analyst console

```bash
python -m ringfence.cli serve      # http://127.0.0.1:8000
```

Built on Starlette rather than FastAPI — FastAPI runs on Starlette anyway, and
for six routes its request-model machinery buys nothing, so skipping it keeps
`pip install -r requirements.txt` short.

The service is **read-only by design**. It returns scores, reasons and evidence.
There is no endpoint that blocks a payment, issues a refund, or mutates an
account, because "the API had a write path but we didn't call it" is not a
defence against the track's defense-only rule.

Three decisions in the console are worth calling out, because each came from the
first version being wrong:

**The queue ranks by money at risk, not by score.** Sorting on raw score gave a
queue of 392 card-testing probes out of 400, every one tied at 0.9998 and worth
about five rupees each. Correct ranking, useless product. Ranking on
`score × amount` puts a ₹29,000 bust-out order above a ₹5 probe, which is how a
risk analyst actually triages.

**Alerts are marked ◆ when linked-account evidence exists.** Ranking by money at
risk surfaces high-value *first* orders — and a mule's first order has no
linked-account history yet, so the flagship view is thinner on graph evidence
than the aggregate numbers suggest. That is worth showing, not hiding: 159 of
the 303 queued alerts are graph-backed, and only 9 of those 159 are false
positives.

**Every alert has a what-if.** One panel shows the score with the graph and
without it. The console opens on the alert where those two numbers differ most —
typically a refund-abuse ring the tabular baseline scored as essentially clean.

Where no cluster resolved, the console says so in a caveat rather than inventing
a reason. Occlusion still produces a large "contribution" for the graph block on
those payments, but it is an artefact — replacing missing cluster features with
honest medians makes the payment look more normal — and reporting it as a top
reason would tell an analyst the payment is suspicious *because* nothing is
known about it, which is exactly backwards.

---

## What it does not catch

Published because a risk team would act on this table, not on the headline.

| archetype | missed payments | missed value |
|---|---|---|
| bust-out | 78 | ₹2.44L |
| refund abuse | 29 | ₹0.42L |
| card testing | 21 | ₹21 |

Bust-out is the remaining hole. It is the archetype deliberately built to be
adversarial to this method — a distinct stolen card and a distinct handset per
mule account, with the delivery address as the only link.

The 49 false positives at the operating point break down as 27 behind a carrier
NAT, 12 sharing no identifier at all, and 10 inside family-shared-card clusters.
Those are precisely the benign structures the generator plants to trip a naive
ring detector — and they account for fewer than fifty alerts across 187,149
payments. No false-positive flood.

---

## Defense-only

The track disqualifies anything offense-capable. RingFence is structurally
incapable of offense:

- it consumes transaction records and emits **risk scores and evidence**; it has
  no write path to any payment, refund, or account API;
- the generator produces labelled abuse patterns **for evaluation only** — no
  working card numbers, no real BIN ranges, no credentials, no bypass
  techniques. Card "fingerprints" are opaque random tokens;
- no component recommends how to evade detection, and the repo contains no
  attack tooling;
- all data is synthetic. No real cardholder data enters the system.

---

## Against the track brief

Track 02 asks for *"a working detector, verifier or auto-responder for one class
of loss, with measured precision and recall on a held-out test set"*, and sets
the bar at *"honest metrics including false-positive cost. Strictly defense-only:
anything offense capable is disqualified."*

| requirement | where it is met |
|---|---|
| a working detector | scoring service + analyst console — `python -m ringfence.cli serve` |
| **one** class of loss | collusive abuse rings: card testing, refund abuse, bust-out |
| measured precision and recall | temporal held-out split, PR curves, per-archetype breakdown |
| on a held-out test set | train days 0–89, val 90–109, test 110–149; test rings never seen in training |
| honest metrics | negative transfer result reported (§ F8), my own explanation for it refuted (§ F9), every claim checked against seed noise |
| **including false-positive cost** | rupee cost model where a false block costs margin × LTV churn, plus review time; sensitivity published across the arguable assumption |
| strictly defense-only | read-only service, no write path to any payment or account, synthetic generator produces no usable credentials — see below |

Two things the brief does not require but a reviewer will ask for anyway: the
result survives 20% of fraud going unlabelled in training (§ F10), and the whole
pipeline regenerates from a fixed seed with six leakage checks that block
publication on failure.

---

## Running it

```bash
pip install -r requirements.txt

make all            # synthetic: data → features → train → evaluate → explain → verify
```

To reproduce the real-data run, put `train_transaction.csv` and
`train_identity.csv` from
[IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) under
`data/raw/ieee/`, then:

```bash
python -m ringfence.cli --config configs/ieee_cis.yaml all
python -m ringfence.cli --config configs/ieee_cis.yaml seedstudy
```

Each dataset writes to its own `data/<name>/` and `reports/<name>/`, so the two
runs cannot overwrite each other's artefacts.

Or stage by stage:

```bash
python -m ringfence.cli data       # synthetic corpus (~20s)
python -m ringfence.cli features   # tabular + causal graph features (~5min)
python -m ringfence.cli train      # both ablation arms (~20s)
python -m ringfence.cli evaluate   # every number in this README
python -m ringfence.cli explain    # analyst-readable alerts + evidence packets
python -m ringfence.cli verify     # leakage and honesty checks
python -m ringfence.cli seedstudy  # is the ablation gap bigger than seed noise?
python -m ringfence.cli serve      # analyst console on :8000
```

Every figure above is regenerated by `make all` from a fixed seed on a clean
checkout. Nothing is hand-copied from a notebook. If a number cannot be
regenerated, it does not go in the pitch.

### Layout

```
ringfence/
  datagen/     synthetic Razorpay-shaped traffic, injected rings, benign confounders
  graph/       identity graph, IDF weighting, clique-safe community detection
  features/    tabular featuriser and strictly-causal graph featuriser
  model/       ablation arms, leakage allowlist
  evaluation/  metrics, rupee cost model, verification checks
  datasets/    canonical payment schema + the IEEE-CIS adapter
  explain/     group-occlusion attribution and evidence subgraphs
  api/         read-only Starlette service + single-file analyst console
configs/       one YAML controls every knob
reports/       generated artefacts — all regenerable
```

`configs/default.yaml` is the single source of truth. Every threshold, cost
assumption, and ring parameter lives there with a comment explaining why it has
the value it has.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the system design and
**[FINDINGS.md](FINDINGS.md)** for the engineering log.
