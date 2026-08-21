# Results — including the ones that say "no"

Most world-model write-ups lead with a win. The two findings below are the
ones that actually changed how we build with this library, and both are
negative or complicating results. We think that's the more useful thing to
publish.

## 1. Aggregate accuracy on a transition-prediction task is rigged by class imbalance

We trained `MLPWorldModel` to predict `(state, action) → next_state` and
compared it against the simplest possible baseline: a two-line
self-updating lookup table keyed on `(state_desc, action)`.

Naive aggregate accuracy said the trained model was competitive. It wasn't
a fair comparison — bucket the same rows by whether that exact
`(state_desc, action)` pair had been seen during training:

| bucket | share of rows | lookup accuracy | trained-model accuracy |
|---|---|---|---|
| seen before | 75.2% | ~0.973 | ~0.973 |
| unseen | 24.8% | 0.402 | — |
| **aggregate** | 100% | **0.972** | **0.936** |

Three-quarters of the evaluation set was already at ceiling for a
zero-parameter lookup table, so the aggregate number is dominated by a
bucket where neither method has to generalize at all. It can't show a real
win or a real regression — an aggregate metric here answers a different
question than the one you're asking.

Once we corrected the loop that fits the model (see below) and re-ran with
the bucket split in place, the trained model **still lost to the lookup
table on 98.9% of rows** (0.9357 vs 0.9720). It won only on the 1.1% of
rows carrying a genuinely novel action — which is exactly the case a
lookup table structurally cannot handle, and the only case where a model
is worth the extra machinery.

**Takeaway:** if you're evaluating a learned transition model, bucket by
seen/unseen `(state, action)` before you trust an aggregate number, and
always compare against the dumbest possible baseline first. A cache that
already covers 75% of your evaluation set at ceiling will make almost
anything look "competitive."

### The bug that made the first version of this comparison meaningless

The training loop that produced the first "trained model" here declared
"3 epochs" in its own metadata while a bug in the batch sampler meant each
"epoch" drew one random batch of 64 rows — 192 rows actually touched, out
of a 20,000-row dataset that was reported as fully consumed. The reported
loss curve looked fine (it was the loss on those 192 rows), and the
regression gate that compares a new checkpoint's loss to the previous
checkpoint's loss could only ever catch "worse than an untrained model" —
it structurally could not tell a genuinely fitted model from a
randomly-initialized one scoring against 192 samples. Fixing the loop
(3 epochs of 192 rows → a real 10,296 gradient steps over the full set) is
what produced the 0.9357 number above; the number before the fix was not
measuring what it claimed to.

**Takeaway:** a green test suite over your training pipeline — shapes,
checkpoint fields, loop plumbing — proves the plumbing works. It cannot
prove the model was actually fit, because a plumbing test can't
distinguish a fitted model from an unfit one initialized the same way.
Every trainer needs at least one positive assertion that it beats a named
baseline, on held-out data, with the bucket split above applied.

## 2. An ensemble's aggregate "champion" model was the worst individual predictor

Comparing several candidate world-model configurations by pooled/aggregate
accuracy across an evaluation set picked a "champion" that, decomposed
per-scenario, was in fact the *worst* individual predictor of state change
among the candidates. The aggregate metric rewarded a model that did well
on the easy majority of cases (again, largely the "seen before" bucket
from finding #1) and masked that it was actively worse than its
competitors on the harder minority — the cases the metric exists to
measure in the first place.

**Takeaway:** the same class of bug as #1, from a different angle — a
single pooled number over an imbalanced evaluation set will reliably pick
the wrong winner if "wrong" means "worse where it matters." We now
score every world-model candidate on a stratified breakdown before
trusting any ranking between them.

## 3. ARC-AGI-3 solving agent — where it actually stands

Running this world model as the planning core of an ARC-AGI-3 solving
agent gets a measured solve rate of **2.12%**, with the agent building
reusable memory inside exactly one game so far. This number survived three
separate attempts to argue it was an artifact of measurement (a stricter
scoring pass, a check for silent no-ops in the harness, a check that the
eval could distinguish "the model is wrong" from "the endpoint is dead") —
each one adversarially trying to find a reason the number was better or
worse than it looked, and each one leaving the number where it started.
We're publishing it at that value rather than waiting for a better one,
because a solve rate nobody adversarially checked is worth less than one
that survived someone trying to break it.

---

If you use this library and get a result — especially a negative one —
we'd rather hear about it than not.
