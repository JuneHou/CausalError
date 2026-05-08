# Conservative-vs-Exploratory Behavior Analysis

## Goal

Different LLM families show systematically different behavior on TRAIL: some
flag many errors per trace (exploratory), others flag few (conservative). We
want a single visualization that separates **how much** a model flags from
**how accurate** the flagged errors are, without re-running experiments and
without per-prediction confidence scores.

## Two recommended plots (per model, one point per trace)

### Plot 1 — `log(pred/gold)` vs F1

- **x-axis: `log((pred + 1) / (gold + 1))`** — count-bias of the predictions
  against ground truth. Smoothed by `+1` to handle traces with `gold = 0`
  (no ground-truth errors) or `pred = 0`.
  - `x < 0` → fewer predictions than gold → **conservative**.
  - `x > 0` → more predictions than gold → **exploratory**.
  - `x ≈ 0` → count-calibrated.
- **y-axis: F1 per trace.**

**What this decomposes.** Pairing `log(pred/gold)` on x with F1 on y separates
behavior from quality:

- The **x-axis answers "how much does the model flag?"** — purely a behavioral
  question about output volume, independent of whether the flags are right.
- The **y-axis answers "of what's involved, how much is right?"** — the
  correctness/quality dimension.

A model at `x ≈ 0` with **low F1** reveals exactly the failure mode that
count-calibration alone hides: the model predicts roughly the right *number*
of errors, but flags the wrong spans. The plot makes this case visible rather
than hiding it. Conversely, a model far from `x = 0` with high F1 is
"miscalibrated in count but still finds the right things."

### Plot 2 — Precision vs Recall scatter with iso-F1 contours

A more direct geometric view of the same decomposition:

- **Diagonal `P = R`** = unbiased operating point.
- **Distance from the diagonal** = conservativeness bias (above the diagonal:
  exploratory, high recall / low precision; below: conservative, high
  precision / low recall).
- **Distance from the origin / iso-F1 contours** = overall quality.

Algebraically the two plots are equivalent — `pred / gold = R / P`, so
`log(pred/gold) = log(R/P)` — but Plot 2 keeps precision and recall as
independent axes and lets the reader read off both quality and bias
geometrically.

## Aggregate vs per-trace

For each model we plot:

- **Per-trace points** showing the distribution of behavior across traces.
- **A single aggregate marker** at `(P_micro, R_micro)` computed from
  `Σ TP / Σ pred` and `Σ TP / Σ gold` over all traces. The aggregate is more
  stable than the per-trace average and is the number that should appear in
  any cross-model comparison.

## Caveat on terminology — "confidence"

This plot captures **behavioral bias**, not **confidence** in the
probabilistic-calibration sense (predicted probability matching empirical
frequency). Without per-prediction logprobs / confidence scores, calibration
in the strict sense is not measurable from these outputs. If "confidence" is
read loosely as "willingness to commit to flagging," the x-axis covers it; if
it is read in the calibration sense, a separate experiment with logprobs is
required.

## Related work

The framing — **comparing predicted-positive counts to gold-positive counts as
a diagnostic for over- vs under-flagging** — has precedent across several
adjacent literatures, even though no single one names it identically.

### NER and span extraction (MUC, CoNLL)
Spurious-vs-missing entity counts are a standard diagnostic going back to the
MUC and CoNLL evaluation campaigns. "Predicted span count vs gold span count"
appears as a routine check in entity recognition and relation extraction
papers, typically reported alongside precision/recall as a sanity signal that
the system is not systematically over- or under-generating mentions.

### LLM-as-judge — verbosity and over-flagging bias
Work on calibrating LLM judges (e.g., MT-Bench, Zheng et al., 2023, and
follow-ups on judge-bias) has documented that LLM judges systematically
favor longer or more elaborately reasoned answers, and that some judges
over-flag minor issues while others under-flag clear ones. The
"over-flagging tendency" of a judge is essentially the same quantity as the
`pred/gold` ratio used here.

### Hallucination in long-form generation
FactScore (Min et al., 2023) and HaluEval (Li et al., 2023) implicitly track
over-generation by comparing the number of generated atomic claims against
the number of supportable claims. The decomposition is the same
"count + correctness" pair: how many claims did the model produce, and what
fraction were verifiable.

### Process reward models (PRM800K, Lightman et al.)
The per-step error-detection literature, most directly Lightman et al.
(2023) and the PRM800K dataset, explicitly discusses asymmetry between
over-flagging and under-flagging at the step level. Process reward models
that over-flag are penalized differently from those that under-flag, and
this asymmetry shows up in their precision/recall trade-off curves.

### Agentic-eval surveys and TRAIL
Recent agent-debugging benchmarks observe that some model families
systematically over-predict errors, and that this is a property of the model
family rather than of the trace difficulty. The original TRAIL paper is the
most direct precedent — its discussion of model behavior across families is
worth re-checking before publication, since any explicit count-bias analysis
there would be the closest prior reference and should be cited directly.

## Implementation

`baselines/outputs/_plot_conservativeness.py` produces both plots side-by-side
for one (model, dataset, method) triple. TP is computed as
`|set(pred_locations) ∩ set(gt_locations)|` to match
`benchmarking/eval/calculate_scores.py`. Default invocation runs on
Mistral-Small-3.1-24B / TRAIL-GAIA(dedup) / Baseline; pass `--pred`, `--gt`,
`--label`, `--out` to retarget.
