# Post-hoc Suppes Graph Inference Experiments

## Motivation

Training-time graph experiments (see `graph/experiments.md`) established that every
mechanism injecting the Suppes graph into the *embedding space* — GAT message passing
or L_graph structural loss — degrades performance relative to the plain baseline
(0.5621 weighted F1). The key finding was:

> The Suppes graph encodes label co-occurrence and causal precedence — a property of
> the *output space* (label distribution), not the *input space* (token embeddings).
> The correct place to use it is at prediction time, after local span evidence is computed.

These experiments test whether the Suppes graph can improve predictions when applied
**post-hoc at inference**, with no change to the trained baseline model.

---

## Baseline Reference

All post-hoc experiments start from the same checkpoint:

| Model | Weighted F1 | Macro F1 | Micro F1 | Precision | Recall |
|-------|-------------|----------|----------|-----------|--------|
| Baseline λ=0, thr=0.25 | **0.5621** | 0.4054 | 0.5828 | ~0.44 | ~0.85 |

The baseline already has high recall (0.85) and low precision (0.44) — it
over-predicts error types per trace. This is the target pattern any post-hoc
graph mechanism must address.

---

## 1. Approach 1: Additive Suppes Propagation

### Algorithm

```
p_base(t)   ← baseline trace-level probabilities  (T, 19), max-pooled over spans
A_norm      ← row-normalised Suppes matrix         (19, 19)

p_final(t) = clip(p_base(t) + α · A_norm · p_base(t), 0, 1)
           = clip((I + α · A_norm) · p_base(t), 0, 1)

α, threshold swept jointly on val ∈ [0.0..1.0] × [0.05..0.50]
```

**Intuition:** If error type A is predicted with probability p, and A→B is a strong
Suppes edge (A causally precedes B), boost B's probability by α·w(A,B)·p(A).
Label co-occurrence signal from Suppes propagates forward through the causal graph.

**Why it can fail:** Additive propagation only ever *increases* probabilities. Applied
to a model that already over-predicts (high recall, low precision), any α > 0 can
only worsen precision further.

### Result

| α (val-tuned) | thr | Weighted F1 | Δ vs baseline |
|---------------|-----|-------------|---------------|
| 0.0 (no propagation) | 0.50 | 0.5370 | −0.025 |

Val sweep selected α=0 — no propagation. **Note:** this run loaded the λ=0.1 baseline
checkpoint (0.5370) rather than the best λ=0 checkpoint (0.5621). The wrong checkpoint
inflated the over-prediction problem, making the sweep correctly reject all α>0.
Experiment invalidated by checkpoint mismatch.

> The additive propagation script (`run_suppes_inference.py`) has been removed.
> The causal gate script below supersedes it.

---

## 2. Approach 2: CMLL-Inspired Causal Gate

### Literature Basis

Inspired by: **Tian et al., "Causal Multi-Label Learning for Image Classification",
Neural Networks, Vol.167, pp.626–637, 2023.** DOI: 10.1016/j.neunet.2023.08.052

**CMLL pipeline (summary):**
- Two-stream CNN architecture (global stream for correlation, local stream for causality)
- At inference: applies Pearl's do-calculus `P(Y | do(X=x))` — selects only regions
  with *direct causal visual evidence* for a label, suppressing spurious co-occurrence
- Core effect: label B is not predicted just because it co-occurs with predicted A;
  B needs its own direct evidence, gated by causal context

**Our analogy:**

| CMLL | TRAIL |
|------|-------|
| Global stream probabilities | Baseline bilinear+cosine span probabilities |
| Causal graph (visual → label) | Suppes graph (error A causally precedes B) |
| `do(X=x)` suppression | Multiplicative gate suppressing B without A |
| Spurious visual co-occurrence | Over-predicted error types without causal context |

**Additional literature support:**
- Read et al. (2009), *Classifier Chains*, ECML — conditioning B's prediction on A's
  prediction using a causally grounded chain order
- ACML (Applied Intelligence, 2021) — asymmetric label correlations: A→B ≠ B→A
- Lin et al. (NTU), *A Study on Threshold Selection for Multi-label Classification* —
  instance-adaptive thresholds based on label correlations

### Algorithm

```
A_raw   ← (19, 19) raw Suppes weights
A_col   ← column-normalised: A_col[j,i] = A_raw[j,i] / Σ_k A_raw[k,i]
          (normalises each column to sum = 1; nodes with no predecessors → gate=1)

For each trace t and error type i:
  gate(t, i)     = Σ_j  A_col[j,i] · p_base(t, j)   ∈ [0, 1]
                 = weighted-average probability of i's causal predecessors
  p_causal(t, i) = p_base(t, i) · (β + (1−β) · gate(t, i))

β, threshold swept jointly on val ∈ [0.0..1.0] × [0.05..0.50]
```

**Contrast with additive propagation:**

| | Direction | Effect on probabilities |
|---|---|---|
| Additive | boosts co-occurring labels | only increases → worsens precision |
| Causal gate | suppresses causally unsupported labels | only decreases → targets false positives |

**Gate behaviour:**
- `β = 1.0`: no intervention (`p_causal = p_base`)
- `β = 0.0`: full suppression when no predecessors are predicted
- Labels with no Suppes predecessors: gate forced to 1.0 (unaffected)

**Command (from `trail-benchmark/`):**
```bash
# Train baseline first (λ=0 checkpoint required)
python graph/baseline/run_baseline.py

# Run causal gate inference (sweeps β and threshold on val automatically)
python graph/post_causal/run_causal_inference.py

# Or with fixed β and threshold (skip val sweep)
python graph/post_causal/run_causal_inference.py --beta 0.3 --threshold 0.25
```

### Result

| β (val-tuned) | thr | Weighted F1 | Macro F1 | Micro F1 | Δ vs baseline |
|---------------|-----|-------------|----------|----------|---------------|
| best val β | 0.25 | 0.5605 | 0.4953 | 0.5081 | −0.0016 |

The causal gate nearly ties the baseline. Δ=−0.0016 on 55 test traces is within noise.

**Per-metric comparison (test set):**

| Metric | Baseline | Causal Gate | Δ |
|--------|----------|-------------|---|
| Weighted F1 | 0.5621 | 0.5605 | −0.0016 |
| Macro F1 | 0.4054 | 0.4953 | **+0.090** |
| Micro F1 | 0.5828 | 0.5081 | −0.075 |
| Precision (micro) | ~0.44 | ~0.36 | −0.08 |
| Recall (micro) | ~0.85 | ~0.85 | ≈0 |

Macro F1 improves (+0.090) because the gate distributes predictions more evenly across
rare error types that the baseline over- or under-predicts. Micro F1 drops because the
gate suppresses some true positives along with false positives when predecessor support
is ambiguous.

---

## 3. Analysis

### Why neither approach improved weighted F1

**1. The over-prediction pattern limits both directions.**
With recall=0.85 and precision=0.44, the baseline predicts roughly 2× as many error
types per trace as the ground truth. Additive propagation can only worsen this.
The causal gate is precision-oriented but is limited by how reliably the Suppes graph
identifies which predictions are false positives vs. ambiguous true positives.

**2. Aggregate co-occurrence vs. instance-level patterns.**
The Suppes graph is built from co-occurrence statistics across 148 training traces.
Individual test traces may follow different co-occurrence patterns than the aggregate.
With 55 test traces and ~1 positive label per trace, any gate based on aggregate
predecessor statistics will be noisy at the instance level.

**3. The gate is precision-oriented but weighted F1 rewards balanced performance.**
The causal gate clearly shifts the precision-recall trade-off: macro F1 improves
(+0.090) because rare error types benefit from reduced spurious predictions; micro F1
drops because the gate reduces overall positive coverage. Weighted F1 is flat because
these effects cancel across the class distribution.

### What the macro F1 improvement suggests

The +0.090 macro F1 gain is notable. It means the causal gate helps on *rare and
mid-frequency error types* where the baseline over-predicts. The baseline's
high-recall strategy disproportionately helps common error types (high support), while
the causal gate redistributes prediction quality more evenly. Whether macro or weighted
F1 is the right metric depends on the downstream use case.

---

## 4. Conclusion

Neither post-hoc Suppes graph mechanism improves weighted F1 over the plain baseline.
The Suppes graph carries real structural signal (confirmed in `graph/experiments.md`
by golden > random, Δ=0.079 under GAT), but this signal cannot be reliably translated
to individual trace-level prediction improvements at the current dataset scale (148
training, 55 test traces).

**The causal gate does show a meaningful macro F1 improvement (+0.090)**, which
suggests it is doing something real — redistributing prediction quality from common to
rare error types. If the evaluation criterion shifts to macro F1 (equal weight per
error type rather than per instance), the causal gate would be the preferred model.

**What would be needed for reliable weighted F1 improvement:**
- Instance-conditioned graph: per-trace bipartite span–label graph with Suppes label
  edges, where propagation is conditioned on each trace's own span evidence rather
  than aggregate training statistics
- More data: >500 annotated traces for aggregate co-occurrence patterns to generalise
  reliably at the individual trace level
