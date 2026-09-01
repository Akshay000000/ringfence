<h1>RingFence</h1>

**Finding fraud rings in payment data by looking at what accounts share.**
Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager

[**Live demo**](https://ringfence-razor.vercel.app/) · [**Analyst console**](https://ringfence-razor.vercel.app/console.html) · [What broke along the way](FINDINGS.md) · [Architecture](ARCHITECTURE.md)

---

## The idea in one minute

**The problem.** Fraudsters work in groups. One person opens twenty accounts, or
one device tests two hundred stolen cards. Each individual payment looks
completely normal, so a system that judges payments one at a time misses all of
it.

**What this does.** It draws a map of which accounts share a phone, a card, a
device or a delivery address, then looks for suspicious clusters in that map and
feeds what it finds into a fraud model.

**Does it work?** On our test data, yes: it catches **92% of fraud** while
wrongly blocking **11x fewer** real customers than a strong conventional system.

**What's the catch?** On a real public dataset it gave **no improvement at all**.
That is reported here too, along with why. It is the most useful result in the
project.

---

## Results

Measured on 187,149 payments the model had never seen, from a later time period
than it was trained on. All 42 fraud rings in that test were new to it.

| | Conventional system | RingFence | |
|---|---|---|---|
| Fraud caught | 87.9% | **92.2%** | |
| Alerts that are real fraud | 72.0% | **96.9%** | fewer wasted investigations |
| Good customers wrongly blocked | 564 | **49** | 11x fewer |
| Total cost to the merchant | ₹15.2L | **₹6.2L** | from ₹41.4L if you do nothing |

"Total cost" is fraud that gets through, plus chargeback fees, plus the profit
and future value lost when a real customer is blocked, plus analyst time. The
system is tuned to minimise that, not to win a leaderboard metric.

---

## The honest part

We tested the same system on two datasets. It only worked on one.

| | Our own data | Real public data |
|---|---|---|
| | 458k payments, fraud rings we planted | 590k real card payments (IEEE-CIS, Vesta) |
| Result | **+24.5%** more fraud caught | **No measurable change** |

**We nearly reported a win that wasn't there.** One training run showed the map
ahead by 2.9%. We retrained it five times with different random starts and found
the run-to-run wobble was three times bigger than that "improvement". It was
noise. There is now a command that makes this check mandatory before any result
can be claimed.

**Then we tested our own excuse, and it was wrong.** Our first explanation was
that the public dataset contains lone fraudsters rather than organised rings. We
tested it, found no support, and withdrew it.

**So we asked the question underneath.** Does fraud actually cluster in that
data, more than chance? We shuffled account membership 400 times, keeping the
cluster sizes identical, and compared.

| | Fraud clustered together | By chance | Verdict |
|---|---|---|---|
| Our data | 31.1% | 0.0% | Overwhelming structure |
| Real data | 2.6% | 0.4% | Real structure, 9.2 sd, but tiny |

There *are* rings in the real data. Just barely any. Only **42 of 3,213**
fraudulent payments sit in a mostly-fraudulent group, so even a perfect ring
detector could reach at most **1.3% of the fraud**. Our measurement noise is
larger than that.

> The null is explained. There is real collusion in the public dataset, roughly
> twelve times weaker than in ours, and far below what any experiment at this
> size could detect. Reporting "no difference" was correct, because there was
> almost nothing there to find.

The condition for this method is not "collusion exists" but **"enough collusion
to matter"**.

Full write-up in [FINDINGS.md](FINDINGS.md).

---

## Try it

```bash
pip install -r requirements.txt
make all          # builds the data, trains, evaluates, and checks itself
make serve        # analyst console at localhost:8000
```

`make all` takes about fifteen minutes and regenerates every number in this
README from scratch at a fixed random seed. Nothing here is hand-copied.

To reproduce the real-data run, put `train_transaction.csv` and
`train_identity.csv` from [IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
into `data/raw/ieee/`, then `make ieee`.

---

## How it works

Three ideas carry the result. Each one came from something breaking first.

**1. Not all sharing is suspicious.** Families share a credit card. An apartment
block shares a delivery address. Four hundred people on one mobile network share
an IP. So each shared detail is weighted by how rare it is: a card used by three
accounts is strong evidence, an IP used by four hundred is thrown away entirely.

**2. Look at when the accounts were opened.** Twenty accounts at one address
could be an apartment block or a fraud ring, and on a map they look identical.
The giveaway is timing: residents signed up over years, a ring signed up the same
week. Without this the system was actively worse at catching the most expensive
fraud type.

**3. Look up the address, not the account.** A fraudster's brand new account has
no history, so searching for the account returns nothing exactly when it matters
most. Looking up what the payment *touches* instead, such as the delivery address
it ships to, took the system from slightly worse than the baseline to
substantially better. This one change is the difference between the idea working
and not.

---

## Why you can trust the numbers

Fraud detection is easy to fake accidentally. Five things guard against it.

**The test set is from the future.** Training uses days 0 to 89, testing uses
days 110 to 149. A random split would let the model memorise specific fraud rings
instead of learning what rings look like.

**The map can only see the past.** A payment on day 100 is described by a map
built from days 55 to 99. It never contributes to the map that judges it.

**Labels have to have existed yet.** A chargeback on a day-80 payment does not
arrive until day 125. Training on it is time travel, so every payment records
when its outcome actually became knowable.

**Six automated checks** run in the pipeline, which refuses to print results if
any fail. The important one shuffles the labels randomly and confirms the model
then finds nothing: if our result were an accident of data leakage, the shuffled
version would still score well. It scores below chance.

**No claim is made without clearing its own noise floor.** Every comparison is
retrained across several random seeds and reported in standard deviations. Under
two, we write "no measurable difference".

<details>
<summary><b>Precise numbers, for reviewers</b></summary>

Held-out temporal test split: 187,149 payments, 1,649 fraudulent (0.88%), 42
rings, none seen in training.

| | Baseline | + graph |
|---|---|---|
| PR-AUC | 0.9075 | **0.9712** |
| ROC-AUC | 0.9978 | 0.9991 |
| Recall @ precision 0.95 | 0.7465 | **0.9290** (+24.5%) |
| Recall @ precision 0.90 | 0.7538 | **0.9400** (+24.7%) |
| Precision at cost-optimal threshold | 0.720 | **0.969** |
| Recall at cost-optimal threshold | 0.879 | **0.922** |
| Good customers blocked | 564 | **49** |
| Net saving vs doing nothing | ₹26.2L | **₹35.2L** |

Per attack type, both arms pinned to precision 0.90:

| | Baseline | + graph |
|---|---|---|
| Refund abuse | 0.428 | **0.922** |
| Bust-out | 0.543 | **0.853** |
| Card testing | 0.973 | 0.986 |

Real-data run, five seeds per arm on IEEE-CIS:

| Condition | Baseline | + graph | Gap |
|---|---|---|---|
| All features | 0.4546 ± 0.0043 | 0.4561 ± 0.0041 | +0.3%, **0.4 pooled sd** |
| Without Vesta's entity counters | 0.2493 ± 0.0029 | 0.2523 ± 0.0223 | +1.2%, **0.2 pooled sd** |

Label-noise robustness on the synthetic corpus: with 20% of fraud left unlabelled
in training, the realistic failure of any review queue, the graph arm's advantage
holds at +0.037 PR-AUC, which is 20 pooled standard deviations.

Savings hold between 82% and 87% of the do-nothing loss across every churn
assumption tested; the full sensitivity table is in
`reports/synthetic/graph_sensitivity.csv`.

</details>

---

## What broke along the way

Six times a result looked right and wasn't. All six are documented in
[FINDINGS.md](FINDINGS.md) with the reasoning that caught them, not just the fix.

| | What happened |
|---|---|
| A 99% score | Not a great model, a broken test. Every fraud account we generated was brand new and almost every honest one was old, so account age gave the answer away. |
| The map made things worse | We rebuilt it every 5 days, but card-testing rings live 2 days. The map never saw them, so "has map data" came to mean "probably honest". |
| Clustering split every ring apart | One parameter set slightly too high made the algorithm break tightly connected groups into individuals, which is exactly what a ring is. |
| Everyone was in a cluster | Every payment from a gmail address inherited a "cluster" of 100,000 strangers. Fixing it made the results better, not worse. |
| A 2.9% gain that was noise | Caught by retraining five times. It nearly went into the results. |
| Our explanation didn't survive | We made a prediction, tested it, found no support, and withdrew it. |

---

## What's inside

```
ringfence/
  datasets/     one schema both datasets convert into, plus the IEEE-CIS adapter
  datagen/      the synthetic payment corpus and the fraud rings planted in it
  graph/        building the identity map and finding clusters in it
  features/     signals from each payment, and signals from the map
  model/        the two systems being compared, and the anti-cheating allowlist
  evaluation/   scoring, cost model, verification, noise and robustness studies
  explain/      why a payment was flagged, the evidence behind it, and the note sent out
  api/          read-only service and the analyst console
```

**No neural network. One language model, in one place, on a leash.** A graph
neural network was considered and rejected: it would memorise 4,500 examples and
quietly cheat by looking at future connections. Classical algorithms build the
map, and a well-understood model makes the call.

The single place a language model earns its keep is the last one. Once a payment
is held, somebody has to write to the merchant, and that is a writing task, not a
detection task. So RingFence drafts that note with a model, and then refuses to
trust it: every number in the draft is checked back against the evidence packet,
the payment reference must survive intact, accusatory words are banned, and a
draft that fails any of those is thrown away in favour of a deterministic
template. The template is the floor. The model can only make it read better,
never make it say more. With no API key configured, which is how the repo ships
and how CI runs it, the template path is the only path.

**It cannot attack anything.** The service only reads. There is no way to block a
payment, issue a refund or change an account through it, and the synthetic data
contains no usable card numbers. That is enforced by how it is built, not by a
promise.

**56 tests**, run by CI on every push along with a 40-second end-to-end build of
the whole pipeline. Many of them guard a specific bug that actually happened;
others hold the project to its own claims, including one that fails the build if
any route stops being read-only.

---

## The demo site

`site/` is a static build: the showcase page plus the analyst console running
against a baked copy of its own API responses, so the whole demo works on a CDN
with no server behind it. The console reads that bundle when it is present and
falls back to the live service when it is not, so one page serves both.

The console tells a first-time visitor what to do: it opens on a strong case so
the screen is not blank, and a one-time pointer above the queue says that each
row is a payment and what clicking one will show. It goes away for good the
moment anyone actually clicks something.

Each case in the console ends with the note that would go to the merchant,
drafted from the evidence on screen and then checked back against it. The
fact check is visible in the panel: it says whether you are reading a model
draft or the template that gets used when a draft fails.

Deploy to Vercel or Render with publish directory `site` and no build command.
See [`site/README.md`](site/README.md).

---

## Against the track brief

Track 02 asks for *"a working detector, verifier or auto-responder for one class
of loss, with measured precision and recall on a held-out test set"*, and sets
the bar at *"honest metrics including false-positive cost. Strictly defense-only:
anything offense capable is disqualified."*

| requirement | where it is met |
|---|---|
| a working detector | scoring service + analyst console, `python -m ringfence.cli serve` |
| **one** class of loss | collusive abuse rings: card testing, refund abuse, bust-out |
| measured precision and recall | temporal held-out split, PR curves, per-archetype breakdown |
| on a held-out test set | train days 0-89, val 90-109, test 110-149; test rings never seen in training |
| honest metrics | negative transfer result reported (§ F7), my own explanation for it refuted (§ F8), every claim checked against seed noise |
| **including false-positive cost** | rupee cost model where a false block costs margin × LTV churn, plus review time; sensitivity published across the arguable assumption |
| strictly defense-only | read-only service, no write path to any payment or account, synthetic generator produces no usable credentials (see below) |

Two things the brief does not require but a reviewer will ask for anyway: the
result survives 20% of fraud going unlabelled in training (§ F9), and the whole
pipeline regenerates from a fixed seed with six leakage checks that block
publication on failure.

---

## All the commands

```bash
python -m ringfence.cli data       # build the synthetic corpus        ~20s
python -m ringfence.cli features   # payment signals + map signals     ~5min
python -m ringfence.cli train      # both systems, one feature apart   ~20s
python -m ringfence.cli evaluate   # every number in this README
python -m ringfence.cli explain    # readable alerts + evidence packs
python -m ringfence.cli verify     # the six anti-cheating checks
python -m ringfence.cli seedstudy  # is a gap bigger than random noise?
python -m ringfence.cli labelnoise # does it survive imperfect labels?
python -m ringfence.cli structure  # is there any ring structure here at all?
python -m ringfence.cli narrative  # draft merchant notes, then fact-check them
python -m ringfence.cli site       # rebuild the static demo
python -m ringfence.cli serve      # analyst console on :8000
```

Add `--config configs/ieee_cis.yaml` to any of them to run against the real
dataset instead. Each dataset writes to its own `data/<name>/` and
`reports/<name>/`, so the two runs cannot overwrite each other.

`configs/default.yaml` is the single source of truth. Every threshold, cost
assumption and ring parameter lives there with a comment explaining why it has
the value it has.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the system design and
**[FINDINGS.md](FINDINGS.md)** for the engineering log.
