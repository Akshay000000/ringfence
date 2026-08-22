# Findings — what broke, and what it cost to find out

This is the engineering log. It is in the repo on purpose. Four times during
this build the system produced a number that looked fine and was wrong, and
the reasoning that caught each one is more transferable than the final metric.

---

## F1. A PR-AUC of 0.99 meant the benchmark was broken, not that the model was good

**Symptom.** First end-to-end run: PR-AUC 0.9906 on the held-out test split.

**Why that is a red flag.** Card-not-present fraud detection at a ~1% base rate
does not produce 0.99 PR-AUC. Published production systems live around
0.5–0.8. A number far above the field either means the task is trivial or the
evaluation is broken.

**Diagnosis.** No single feature scored above 0.53 alone, so it was not one leaky
column — it was a combination. The culprit was population realism:

| | honest | fraud |
|---|---|---|
| account age, 25th pct | 116 days | — |
| account age, max | 629 days | 48 days |

Every ring account was freshly minted; almost every honest account was aged.
`account_age_days < 50` captured ~100% of fraud and ~10% of honest traffic, and
two more features finished the job.

**Fix.** Two changes to the generator, both toward realism, not toward
difficulty:
- honest customers now include a large continuously-acquired cohort
  (`legacy_customer_share: 0.42`), so "new account" is a normal honest state;
- a configurable share of ring accounts are aged rather than fresh
  (`aged_account_share`), because real crews buy, farm, and take over accounts.

**After:** honest 25th percentile age 35 days, fraud median 23 days — genuine
overlap. PR-AUC fell to 0.907, which is a believable number.

**Transferable point:** when a benchmark is easy, the instinct to celebrate and
the instinct to investigate point in opposite directions. The second one is
correct.

---

## F2. The graph made the model *worse*, and the reason was missingness

**Symptom.** Adding 38 graph features moved recall at precision 0.95 from 0.747
to 0.665 — an 11% relative *loss*.

**Diagnosis.** Graph snapshots ran on a 5-day stride, and the window for
snapshot *d* was `[d-60, d)`. Ring lifetimes:

| archetype | median lifetime |
|---|---|
| card_testing | 2 days |
| bust_out | 6 days |
| refund_abuse | 16 days |

A card-testing ring is born and dead inside one stride. Coverage measured on
the test split:

| | rows with graph features |
|---|---|
| honest | 83.2% |
| card_testing | 3.2% |
| bust_out | 33.3% |
| refund_abuse | 38.2% |

The graph block was not a fraud signal. It was a *tenure* signal — present for
established honest accounts, absent for fraud — and the model learned
"has graph features → safe."

**Fix.** Stride 1. Necessary but not sufficient; see F4.

---

## F3. Louvain at resolution > 1 shatters cliques, and a ring is a clique

**Symptom.** After the stride fix, bust-out rings were still barely clustered.
The pattern was too clean to be noise: rings with 4, 7, 8, 9, 10 accounts
clustered perfectly; rings with 14, 19, 20, 26 accounts came back with *zero*
clustered members.

**The tell.** The cutoff sat exactly at `LOUVAIN_MIN_COMPONENT = 14` — the
threshold above which the code stopped keeping components whole and handed them
to Louvain. That is not a data property, it is a code path.

**Diagnosis.** Reproduced on a synthetic 26-node clique:

```
resolution 1.15  ->  26 communities of size 1
resolution 1.00  ->  1 community of size 26
resolution 0.90  ->  1 community of size 26
```

At resolution > 1 the modularity null-model penalty exceeds the within-community
edge reward, so singletons win. A ring sharing one drop address *is* a clique.
The config was destroying precisely the structure it was there to find.

**Fix.** `louvain_resolution: 0.92`, plus a guard that keeps any component with
edge density ≥ 0.40 whole rather than splitting a near-clique at all.

**Result.** bust-out clustering 45.9% → 100.0%; honest clustering unchanged at
28.6%, so this bought recall without spending precision.

---

## F4. The real blocker was that a new account is not yet a node

**Symptom.** Even with stride 1 and clique-safe clustering, the graph arm still
trailed the baseline (PR-AUC 0.894 vs 0.908).

**The experiment that settled it.** Rather than tuning further, restrict the
test set to rows where a cluster actually resolved and re-score both arms:

| | all test rows | rows with a resolved cluster |
|---|---|---|
| baseline PR-AUC | 0.9075 | 0.9122 |
| graph PR-AUC | 0.8943 | **0.9979** |

Where the graph had data it was overwhelming. It had data on 23.9% of rows.
The problem was never signal quality; it was coverage.

**Root cause.** Features were keyed on `customer_id`. A mule account's first
payment has no prior history, so the lookup returned nothing — for exactly the
transactions that matter most. Fraud-row coverage was 36–50%.

**Fix — identifier-level resolution.** Build a `value → cluster` map alongside
the customer map. When an unseen account appears, resolve it through what it
*touches*: the drop address, the device, the card. Link types are tried
strongest first, so a card match beats an IP match. This is how a human analyst
works — *"I have never seen this account, but I have seen this address."*

**Result.** Coverage 23.9% → ~60%, and the ablation flipped from −4.6% to
**+23.9%** relative recall at precision 0.95.

---

## F5. What the graph does *not* do

Reported because it is true, not because it helps the pitch.

At matched precision 0.90:

| archetype | baseline | graph | lift |
|---|---|---|---|
| refund_abuse | 0.428 | 0.922 | **+0.494** |
| bust_out | 0.543 | 0.835 | **+0.293** |
| card_testing | 0.973 | 0.981 | +0.008 |

**The graph adds essentially nothing to card testing.** Velocity counters
already catch it at 97%, because card testing is a single-device burst that a
per-transaction counter sees perfectly well. Card testing resolves in ~2 days,
faster than any graph snapshot can form.

The graph earns its place on the slow, distributed attacks — refund abuse over
weeks, bust-out over days — where no single transaction looks wrong and the
evidence exists only in the relationships. That is a narrower claim than "graphs
improve fraud detection," and it is the one the data supports.

---

## F6. Residual limitations

Things a reviewer should push on, listed before they have to ask.

1. **Synthetic data.** Ring topologies are ones I designed, so the model is
   partly being graded on a world I built. Mitigations: benign confounders that
   deliberately mimic ring structure; three archetypes with deliberately
   different topologies; a bust-out archetype built to be adversarial to the
   method (distinct card and device per mule, address as the only link). The
   honest position is that this measures *method viability*, not production
   accuracy.
2. **Oracle labels.** Training uses ground-truth `is_fraud`. Production labels
   are noisy and arrive late. Label maturity is modelled (F5/V5 below); label
   *noise* is not yet.
3. **Test-window maturity.** Only 14% of test-split labels had matured by
   `as_of_day`. Test metrics are what a reviewer would read ~45 days after the
   window closes — measurable in simulation today, not in production today.
4. **Stride-1 cost.** The snapshot roll-forward is ~5 minutes over 458k
   payments on 2 cores. A production system would maintain the graph
   incrementally rather than rebuilding it 150 times.
5. **No intra-day graph.** A payment is scored against the graph as of the
   start of its day. A true streaming system would include earlier events from
   the same day; that would raise coverage further and is the obvious next step.

---

## Verification

Six checks run as part of `make all` and the pipeline refuses to report numbers
if any fails.

| check | result |
|---|---|
| V1 no forbidden column reaches the model | pass |
| V2 graph window strictly earlier than the payments it scores | pass |
| V3 label permutation collapses the signal | pass — permuted PR-AUC 0.0075 vs base rate 0.0088 (0.86×) |
| V4 test rings are novel | pass — 42 test rings, 0 seen in training |
| V5 training labels had matured by as_of_day | pass — 0 violations |
| V6 cluster statistics never reference an outcome column | pass |

V3 is the one that matters. If the +24% lift were leakage, a model fitted on
permuted labels would still find it. It scores *below* the base rate.

*(V6 initially failed — on `cluster_behaviour`'s own docstring, which names the
columns it promises not to use. The check now parses the AST and strips
docstrings and comments before scanning. A check that fails on its own
documentation is a bad check.)*
