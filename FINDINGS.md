# Findings: what broke, and what it cost to find out

This is the engineering log. It is in the repo on purpose. Four times during
this build the system produced a number that looked fine and was wrong, and
the reasoning that caught each one is more transferable than the final metric.

---

## F1. A PR-AUC of 0.99 meant the benchmark was broken, not that the model was good

**Symptom.** First end-to-end run: PR-AUC 0.9906 on the held-out test split.

**Why that is a red flag.** Card-not-present fraud detection at a ~1% base rate
does not produce 0.99 PR-AUC. Published production systems live around
0.5 to 0.8. A number far above the field either means the task is trivial or the
evaluation is broken.

**Diagnosis.** No single feature scored above 0.53 alone, so it was not one leaky
column. It was a combination. The culprit was population realism:

| | honest | fraud |
|---|---|---|
| account age, 25th pct | 116 days | n/a |
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

**After:** honest 25th percentile age 35 days, fraud median 23 days. Genuine
overlap. PR-AUC fell to 0.907, which is a believable number.

**Transferable point:** when a benchmark is easy, the instinct to celebrate and
the instinct to investigate point in opposite directions. The second one is
correct.

---

## F2. The graph made the model *worse*, and the reason was missingness

**Symptom.** Adding 38 graph features moved recall at precision 0.95 from 0.747
to 0.665, an 11% relative *loss*.

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

The graph block was not a fraud signal. It was a *tenure* signal, present for
established honest accounts and absent for fraud, and the model learned
"has graph features → safe."

**Fix.** Stride 1. Necessary but not sufficient; see F4.

---

## F3. Louvain at resolution > 1 shatters cliques, and a ring is a clique

**Symptom.** After the stride fix, bust-out rings were still barely clustered.
The pattern was too clean to be noise: rings with 4, 7, 8, 9, 10 accounts
clustered perfectly; rings with 14, 19, 20, 26 accounts came back with *zero*
clustered members.

**The tell.** The cutoff sat exactly at `LOUVAIN_MIN_COMPONENT = 14`, the
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
payment has no prior history, so the lookup returned nothing, for exactly the
transactions that matter most. Fraud-row coverage was 36% to 50%.

**Fix: identifier-level resolution.** Build a `value → cluster` map alongside
the customer map. When an unseen account appears, resolve it through what it
*touches*: the drop address, the device, the card. Link types are tried
strongest first, so a card match beats an IP match. This is how a human analyst
works: *"I have never seen this account, but I have seen this address."*

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

The graph earns its place on the slow, distributed attacks (refund abuse over
weeks, bust-out over days) where no single transaction looks wrong and the
evidence exists only in the relationships. That is a narrower claim than "graphs
improve fraud detection," and it is the one the data supports.

---

## F6. A pruned identifier was still resolving payments into clusters

**Symptom.** On IEEE-CIS, **100% of transactions** resolved into a cluster, fraud
and honest alike, and the median cluster held 153 accounts. Those are not rings,
they are blobs, and a feature that is true of every row describes nothing.

**The misleading part.** Snapshot-level clustering looked healthy. Sweeping the
hub caps on a single snapshot gave a median cluster of 3 accounts and only ~42%
of accounts clustered at all. The clusters were fine; something downstream was
pouring them together.

**Diagnosis.** The identifier→cluster map used by the new-account resolution path
(F4) was built from *all* identifiers, including the hub identifiers that pruning
had already thrown away. So `gmail.com`, a value shared by ~100,000 accounts,
correctly discarded as infrastructure and contributing no edge to the graph,
still mapped to whatever cluster it happened to touch. Every payment carrying a
gmail address inherited it.

The bug is the exact mirror of one already fixed in the console, where a pruned
hub was drawn as an edge it never contributed. Same mistake, two layers apart.

**Fix.** The map is built only from identifiers present in
`snapshot.identifier_weight`, the set that survived pruning.

**Effect.** Inherited resolution fell from 82% to 35% on IEEE-CIS, and from ~900
to ~165 payments per day on synthetic.

**And the synthetic headline went up, not down.** +23.9% → **+24.5%** at
precision 0.95; bust-out recall 0.835 → 0.853. The leak had not been propping the
result up. It had been feeding the graph junk edges through NAT hubs and weak
domains. This is the reassuring direction to find a bug in, and it was worth
re-running both datasets to check rather than assuming.

---

## F7. On real data the graph adds nothing measurable, and that is the finding

RingFence was validated a second time on **IEEE-CIS** (Vesta), 590,540 real
card-not-present transactions, 3.5% fraud, temporal split, same pipeline and same
code, with only the dataset adapter differing.

**Result: no measurable lift.** Five seeds per arm:

| condition | baseline PR-AUC | + graph | difference |
|---|---|---|---|
| all features | 0.4546 ± 0.0043 | 0.4561 ± 0.0041 | +0.3%, **0.4 pooled sd** |
| without Vesta's entity counters | 0.2493 ± 0.0029 | 0.2523 ± 0.0223 | +1.2%, **0.2 pooled sd** |

**How close this came to being reported as a win.** A single-seed run showed the
graph ahead by **+2.9%**, and withholding Vesta's C-counters pushed it to
**+6.3%**, a tidy story about the graph reproducing hand-engineered entity
aggregation. Refitting across five seeds showed the run-to-run standard deviation
was ±0.004 PR-AUC, three times the size of the "+2.9%". Both numbers were noise.
`ringfence/evaluation/seed_study.py` now exists so no ablation claim in this repo
can be made without clearing its own noise floor.

**Why the graph does not help here.** Three reasons, and none of them is "graphs
don't work":

1. **The identifiers are proxies, not identities.** `card1` is a card *group*:
   ~14,800 values across 590k transactions, ~40 transactions each, not a card.
   `addr1` is a billing *region*: 437 distinct values, median 2 accounts but a
   90th percentile of 696. Email is a *domain*, not an inbox; 59 values, median
   154 accounts each. A device fingerprint exists on only 24% of rows. There is
   no phone and no IP. Hub pruning correctly discards most of this, and what
   survives is card and device.
2. **The fraud is not collusive.** IEEE-CIS labels individual card-not-present
   fraud. It has no ring labels because it largely has no rings. RingFence
   detects *collusion structure*; asking it to improve single-actor fraud
   detection is asking the wrong question of it.
3. **There is barely any ring structure to find.** Quantified in F10: only 1.31%
   of test fraud sits in a majority-fraud cluster, which is below the resolution
   of the measurement.
4. **The entity aggregation was already done.** Vesta's `C1` to `C14` are counts of
   addresses and phones associated with a card, a hand-engineered version of
   what the graph computes. The baseline already has them, and the honest
   ablation gave them to both arms.

Those three points are true *descriptions* of the dataset. Whether they
**explain** the null is a separate question, and F8 tests it.

---

## F8. I tested my own explanation for the null, and it did not survive

F7 offered a reassuring story: the graph adds nothing on IEEE-CIS because that
dataset's fraud is single-actor rather than collusive. That is a comfortable
thing to believe about your own method, which is exactly why it needed testing
rather than asserting.

It makes a falsifiable prediction. If the graph pays off on relational fraud,
its advantage should **grow with how much linked-account structure a payment
actually sits in**. So: bucket the test set by resolved cluster size (a property
of the data available at scoring time, not a label) and score both arms on identical
rows in every bucket, and repeat across five seeds.

| cluster size | rows | fraud rate | baseline | + graph | difference | effect |
|---|---|---|---|---|---|---|
| no cluster | 56,542 | 2.84% | 0.3686 | 0.3707 | +0.6% | 0.5 sd |
| 2-4 accounts | 5,384 | 2.06% | 0.2968 | 0.2544 | **−14.3%** | **3.0 sd** |
| 5-19 accounts | 7,709 | 1.78% | 0.2627 | 0.2831 | +7.8% | 1.4 sd |
| 20-99 accounts | 6,615 | 3.57% | 0.5184 | 0.5184 | 0.0% | 0.0 sd |
| 100+ accounts | 16,177 | 6.93% | 0.5977 | 0.6029 | +0.9% | 0.6 sd |

**There is no trend.** The advantage does not grow with cluster size. Four of the
five buckets show no measurable difference, and the only statistically
significant result in the table is the graph being **worse** on small clusters,
where a two-to-four account cluster gives the model a handful of noisy relational
features and it does worse than having none.

**So the explanation is withdrawn.** The structural facts in F7 remain true.
`card1` really is a card group, Vesta's C-counters really are pre-computed entity
But I cannot claim they *explain* the null, because the mechanism
they imply does not show up when measured. The honest position is narrower:

> The graph produces a large, verified improvement on a benchmark whose ring
> structure I constructed. It produces no measurable improvement on IEEE-CIS, and
> a targeted search for the subgroup where it should have helped found nothing.
> Why it does not transfer is **unresolved**.

That is less satisfying than F7's version and it is what the evidence supports.
A method that only works where its author built the world is a method with a real
open question attached, and the open question belongs in the write-up rather than
in a footnote.

**What would settle it** is a real dataset with genuine collusion labels:
confirmed abuse rings rather than transaction-level fraud flags. IEEE-CIS was the
closest public dataset with an identity surface, and it has fraud labels but no
ring labels. Until such a benchmark is run, the positive result should be read as
*mechanism demonstrated under constructed conditions*, not as evidence of
production accuracy.

---

## F9. The advantage survives realistic label noise

Both arms train on ground-truth `is_fraud`, which production never has. So:
corrupt a fraction of the **training** labels, leave the test labels clean, and
see whether the gap survives. Two modes, because they are not equally realistic.

**Missed fraud (`fn_only`)**: flip positives to negatives: attacks that were
never caught and therefore train as legitimate. This is what actually happens in
a fraud system, and it is the nastier failure conceptually, because it teaches
the model that real attacks are fine.

| noise | baseline PR-AUC | + graph | gap | effect | recall@P90 gap |
|---|---|---|---|---|---|
| 0% | 0.8987 | 0.9607 | +0.062 | 21.8 sd | +0.185 |
| 5% | 0.9379 | 0.9785 | +0.041 | 12.8 sd | +0.087 |
| 10% | 0.9451 | 0.9829 | +0.038 | 6.8 sd | +0.072 |
| 20% | 0.9419 | 0.9785 | +0.037 | 20.2 sd | +0.065 |

The gap narrows but never closes, and stays many standard deviations wide at
every level. **A fifth of the fraud going unlabelled does not take the graph's
advantage away.**

(The absolute numbers *rising* with a little noise is real and slightly awkward:
dropping some positives appears to offset an over-aggressive positive class
weight in the clean-label fit. It suggests `class_weight_positive: 6.0` is tuned
a notch too high, which is worth revisiting, noted rather than quietly smoothed
over.)

**Symmetric noise is a different story, and the reason is arithmetic.** Flipping
labels in *both* directions at rate p, against a 2% base rate, means p×98% of
negatives become false positives. At p=0.05 that is ~4.9% spurious positives
against 2% real ones, so the positive class is now **71% noise**. That is not a
noisy label, it is the absence of a label. Both arms collapse (baseline 0.899 →
0.744) and the gap becomes unstable and insignificant (0.7, 0.6, 1.8 sd across
rates).

So the honest reading is: symmetric noise at these rates is not a realistic
stress test for a rare-event problem, and the result is reported because it was
run, not because it says the method is fragile. The realistic mode is
`fn_only`, and there the answer is clean.

---

## F10. The structure is real on real data, and too small to measure

F7 found no lift on IEEE-CIS. F8 offered an explanation and then withdrew it,
which left the null sitting there unexplained. Both had skipped the question
underneath: **does fraud concentrate inside the clusters the graph finds, more
than it would if account membership were random?**

That is answerable without any model. Take the clusters exactly as the graph
produced them, shuffle the fraud labels across accounts 400 times, and compare
the real concentration of fraud in majority-fraud clusters against the null
distribution. Cluster sizes are held fixed, so the test asks only whether
membership carries information, not whether big clusters exist.

| dataset | concentration | by chance | distance | p |
|---|---|---|---|---|
| synthetic | 31.1% | 0.0% | 1314 sd | < 0.0025 |
| IEEE-CIS | 2.6% | 0.4% | 9.2 sd | 0.0025 |

**Both are real.** At 9.2 sd the IEEE-CIS clusters are not an artefact: there
genuinely are accounts that share identifiers and share fraud. F8's withdrawn
explanation was wrong to imply the real data has no rings at all.

**And it does not matter.** Only 42 of the 3,213 fraudulent test payments sit in
a majority-fraud cluster. Even in the impossible best case, where the graph
catches every one of those and the baseline catches none, the ceiling on the
improvement is **1.31% of all fraud**. Against the synthetic corpus, the same
number is 300 of 1,649, or 18.2%.

The noise floor measured in F7 is +/- 0.0043 PR-AUC. The observed gap was
+0.0015.

> The largest effect the data could contain is smaller than the smallest effect
> the measurement could resolve.

So the null is explained, and it is not a failure of the method. There was
almost nothing there to find. It also sharpens the scope of the claim: the
precondition is not "collusion exists" but "enough collusion to matter", and
that is now a number rather than an intuition.

Reproduce: `python -m ringfence.cli --config configs/ieee_cis.yaml structure`.

---

## F11. A language model, in exactly one place, on a leash

Everything up to here is deliberately model-free, and the reasoning holds:
detection over structured data at scale is not a language task, and putting a
model there would be decoration that costs latency and determinism.

The last step is different. Once a payment is held, somebody writes to the
merchant. That is a writing task, and doing it with string concatenation
produces the stiff, faintly threatening prose that makes support queues long.

The problem is that this is also the single worst place in the system to
hallucinate. A note sent to a merchant, or attached to a chargeback
representment, that states a wrong amount or invents a linked account is worse
than no note at all.

So the design is not "ask a model to explain the alert". It is:

1. assemble a **closed set of facts** from the evidence packet the alert already
   produced, with hub identifiers that the graph pruned excluded, because letting
   `gmail.com` back in as written evidence would assert the exact thing the graph
   threw away;
2. ask the model to rephrase **only those facts**;
3. **verify** the draft against the fact set: every number in it must appear in
   the facts, the payment reference must survive intact, and a short list of
   accusatory words is banned, because the system holds payments for review and
   does not accuse people;
4. **fall back** to a deterministic template whenever the check fails, or when no
   model is configured at all.

The verifier is the engineering, not the prompt. A prompt is a request; a
verifier is a guarantee. The template is the floor the system can never drop
below, and the model is only ever allowed to make it read better.

Three smaller things fell out of building it, and all three came from reading
the drafts rather than the code.

The **closing sentence** originally read "based on the account's connections to
other accounts" on every note. That is false on a payment held purely on
velocity, so it is now conditional on whether any link actually resolved.

The **analyst vocabulary leaked into the merchant note**. The first drafts told
a shop owner that their payout was held because of "high-confidence shared
identifiers (combined edge weight 1.34)" and "cluster decline rate 0%". Both are
correct and both are jargon deployed at somebody who cannot check it. Purely
internal reasons are now dropped rather than paraphrased, because a reason a
merchant cannot check is not a reason worth giving, and "cluster", which is our
word, becomes "linked accounts", which is theirs.

And **generic reasons had to go**. When no evidence line clears the abnormality
floor, the reason detail is the group name restated, so notes were going out
saying "Why: payment context". That says nothing while looking like it says
something, which is the specific failure this whole feature is supposed to avoid.

The repo ships with no key, so the default path for anyone cloning it, and the
path CI exercises, is the template.

Seventeen tests cover it, and the one that justifies shipping a model at all is the
one that feeds the verifier a draft claiming "we recovered Rs 4,500" and asserts
that the number never reaches the output.

Reproduce: `python -m ringfence.cli narrative`.

---

## F12. Green locally, red in CI, and the tests were never the problem

Three consecutive CI runs failed at the `Tests` step against a suite that passed
on every machine it was run on. The tests were fine. The command was not.

`pytest tests` and `python -m pytest tests` are not the same command. The module
form puts the working directory on `sys.path`; the console script does not. Every
test here imports `ringfence` from the checkout rather than from an installed
package, so the console script died at collection with `ModuleNotFoundError`
before running a single test, while `make test`, which uses the module form,
stayed green.

The failure mode is worth naming because it is invisible from the inside: the
suite is not broken, the code is not broken, and reproducing it locally requires
running the command exactly the way CI runs it, which is the one thing nobody
does. It is also silent in the sense that mattered here, since the run went red
on a commit whose changes had nothing to do with it.

Fixed twice over, deliberately. `pyproject.toml` sets `pythonpath = ["."]` so a
bare `pytest` works for anyone who clones this, and CI now calls the module form
so it cannot drift away from the Makefile again.

---

## F13. Residual limitations

Things a reviewer should push on, listed before they have to ask.

1. **Synthetic data carries the positive result.** Ring topologies are ones I
   designed, so the +24.5% is measured on a world I built. Mitigations: benign
   confounders that deliberately mimic ring structure; three archetypes with
   deliberately different topologies; a bust-out archetype built to be
   adversarial to the method. The real-data run (F7) is the counterweight, and
   it returns a null, which is why the claim is scoped to collusion-with-raw-
   identifiers rather than stated generally.
2. **Oracle labels.** Training uses ground-truth `is_fraud`. Label maturity is
   modelled (V5) and label *noise* is now tested (F9): the advantage survives
   20% of fraud going unlabelled. What is still untested is *systematic*
   mislabelling correlated with the features themselves, which is the failure
   mode a real review queue actually has.
3. **Test-window maturity.** Only 14% of test-split labels had matured by
   `as_of_day`. Test metrics are what a reviewer would read ~45 days after the
   window closes, measurable in simulation today, not in production today.
4. **Stride-1 cost.** The snapshot roll-forward is ~5 minutes over 458k
   payments on 2 cores. A production system would maintain the graph
   incrementally rather than rebuilding it 150 times.
5. **No intra-day graph.** A payment is scored against the graph as of the
   start of its day. A true streaming system would include earlier events from
   the same day; that would raise coverage further and is the obvious next step.
6. **The note verifier checks tokens, not meaning.** It catches invented
   numbers, an altered payment reference and accusatory vocabulary, which are
   the failures that actually reach a card scheme. It cannot catch a draft that
   rearranges true facts into a misleading emphasis, because that requires
   understanding the claim rather than checking it. The template is the floor
   under that gap, not a fix for it, and a production deployment should put a
   human on the first few hundred drafts before trusting the model path.

---

## Verification

Six checks run as part of `make all` and the pipeline refuses to report numbers
if any fails.

| check | result |
|---|---|
| V1 no forbidden column reaches the model | pass |
| V2 graph window strictly earlier than the payments it scores | pass |
| V3 label permutation collapses the signal | pass, permuted PR-AUC 0.0081 vs base rate 0.0088 (0.92×) |
| V4 test rings are novel | pass, 42 test rings, 0 seen in training |
| V5 training labels had matured by as_of_day | pass, 0 violations |
| V6 cluster statistics never reference an outcome column | pass |

V3 is the one that matters. If the +24% lift were leakage, a model fitted on
permuted labels would still find it. It scores *below* the base rate.

*(V6 initially failed on `cluster_behaviour`'s own docstring, which names the
columns it promises not to use. The check now parses the AST and strips
docstrings and comments before scanning. A check that fails on its own
documentation is a bad check.)*
