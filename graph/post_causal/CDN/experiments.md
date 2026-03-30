# CDN Experiments — Conditional Dependency Network

## Motivation

The causal gate (see `post_causal/experiments.md`) showed that post-hoc use of the Suppes graph
can shift prediction quality toward rare error types (+0.090 macro F1) but does not improve
weighted F1. The gate is a one-shot multiplicative suppression with a single tunable scalar β.

The CDN approach replaces the fixed gate with **learned conditional classifiers** — one per label —
that can capture asymmetric, non-linear dependencies between error types. It is post-hoc in the
same sense: the baseline model is frozen, and the CDN only refines its trace-level outputs.

Reference: Heckerman et al. (2000). *Dependency Networks for Inference, Collaborative Filtering,
and Data Visualization.* JMLR, 1, 49–75.
<https://dl.acm.org/doi/abs/10.5555/2283516.2283613>

---

## Algorithm

### Phase A — Training (graph-restricted CDN)

For each error label `i` (0..18):

```
neighbors[i]  = parents(i) ∪ children(i) in the Suppes graph  [or parents-only]

X_i = concat(
    baseline_logits_train,               # (N_train, 19)  max-pooled pre-sigmoid scores
    gold_labels_train[:, neighbors[i]]   # (N_train, d_i)  teacher-forced neighbor labels
)
y_i = gold_labels_train[:, i]

model_i = LogisticRegression(class_weight='balanced', C=1.0).fit(X_i, y_i)
```

Key design choices:
- **Logits not probabilities**: baseline logits (pre-sigmoid) are used as features; the logistic
  regression applies its own linear transformation so logits carry more signal.
- **Teacher forcing**: neighbor labels are gold at training time (standard CDN protocol).
- **`class_weight='balanced'`**: handles severe class imbalance (~few positives per error type).
- **Zero-positive labels**: labels with no positive training examples get a `None` classifier
  (always predict 0). Prevents overfitting on degenerate cases.

### Phase B — Inference (deterministic mean-field)

At test time, true neighbor labels are unknown. Iterative updates are used:

```
Initialize: y^(0) = (sigmoid(baseline_logits) >= 0.5)

For each sweep s = 1 .. n_sweeps:
    For each label i:
        p_i = model_i.predict_proba(concat(logits, y[:, neighbors[i]]))[1]
        y_i ← p_i    (soft update; threshold at end of all sweeps)

Final: threshold marginals at val-tuned threshold
```

A stochastic Gibbs mode is also available (`--inference gibbs`): each label is sampled
from `Bernoulli(p_i)` and post-burn-in samples are averaged for marginal probabilities.

### Hyperparameter sweep (val)

Grid-searched jointly on val split:
- `n_sweeps` ∈ {1, 3, 5, 10, 20}
- `threshold` ∈ [0.05, 0.50] (step 0.05)

---

## Results

### Experiment 1: Default CDN (deterministic, neighborhood=both)

**Val sweep best**: n_sweeps=5, threshold=0.35

**Test set:**

| Metric | CDN | Baseline (thr=0.50) | Δ |
|--------|-----|---------------------|---|
| Weighted F1 | 0.5078 | 0.5203 | −0.0125 |
| Macro F1 | 0.3502 | 0.3815 | −0.0313 |
| Micro F1 | 0.5065 | 0.4918 | +0.0147 |
| Precision (weighted) | 0.4089 | 0.4012 | +0.0077 |
| Recall (weighted) | 0.7091 | 0.8182 | −0.1091 |

### Experiment 2: Parents-only neighborhood

**Val sweep best**: n_sweeps=5, threshold=0.35

**Test set:** Identical to Experiment 1 (see analysis below).

---

## Analysis: Why `neighborhood=both` and `neighborhood=parents` Produce Identical Output

### Finding

Running with `--neighborhood both` and `--neighborhood parents` produced bit-for-bit
identical test predictions, metrics, and val-tuned hyperparameters.

### Investigation

Three properties were verified:

**1. Feature dimensions differ** — for exactly 1 label:

| Label | parents features | both features |
|-------|-----------------|---------------|
| Instruction Non-compliance (6) | 19 + 8 = 27 | 19 + 9 = 28 |
| All other 18 labels | same | same |

**2. Neighbor lists differ** — for exactly 1 label:
- `Instruction Non-compliance` has 8 parents and 9 in `parents ∪ children`
- The extra neighbor under `both`: **Environment Setup Errors**
  (Instruction Non-compliance → Environment Setup Errors is an edge, but not the reverse)
- All other 18 labels satisfy `children[i] ⊆ parents[i]`

**3. Inference correctly uses the selected neighborhood**:
- Feature dimensions passed at inference match each model's expectation (verified)
- The extra feature does change label 6's marginal probabilities (`max |Δp| = 0.017`)
- But no prediction flips at threshold=0.35 across 13 test traces

### Root cause: The Suppes graph is nearly fully bidirectional

The Suppes graph has 155 directed edges over 19 nodes (~43% density). For 18/19 labels,
every child is already a parent — i.e., if A→B is an edge, B→A is also an edge. This is
expected: Suppes probability-raising is approximately symmetric for commonly co-occurring
error pairs. Only one asymmetric edge remains after CAPRI pruning:

```
Instruction Non-compliance → Environment Setup Errors  (but NOT the reverse)
```

Because the graph is nearly symmetric, `both = parents` for 18/19 labels. The one extra
feature for label 6 changes probabilities by at most 0.017, insufficient to flip any
binary prediction on the 13-trace test set.

**Practical implication**: The `parents` / `both` / `children` distinction is meaningful
only if the graph is asymmetric. In this Suppes graph it is almost entirely symmetric,
making the neighborhood choice irrelevant in practice.

---

## Why CDN Underperforms the Baseline

### 1. Threshold shift dominates

CDN val sweep selects threshold=0.35 (vs. baseline's 0.50). The lower threshold
increases recall but hurts precision. The CDN classifiers are not precise enough to
recover from the resulting false positive increase.

### 2. Very small training set

With ~118 training traces and 19 classifiers (each with 19–34 features), each classifier
sees at most a handful of positive examples per label. Logistic regression with
`class_weight='balanced'` compensates but cannot overcome structural data sparsity.

### 3. The Suppes graph is output-space structure injected at the wrong stage

The graph encodes label co-occurrence statistics across traces. At training time,
teacher-forced gold neighbor labels give the CDN real signal. At inference, the
neighbors are estimated from the same baseline logits being refined — a circular
dependency that can reinforce baseline errors rather than correct them.

### 4. The baseline is already strong

The baseline (weighted F1=0.5621, recall=0.85) sets a high bar. The CDN's deterministic
inference at threshold=0.35 achieves recall=0.71 — lower than the baseline — while not
gaining enough precision to compensate.

---

## Comparison: All Post-hoc Methods (test set)

| Method | Weighted F1 | Macro F1 | Micro F1 | Precision | Recall |
|--------|-------------|----------|----------|-----------|--------|
| **Baseline** (thr=0.25) | **0.5621** | 0.4054 | 0.5828 | ~0.44 | **0.85** |
| Causal Gate | 0.5605 | **0.4953** | 0.5081 | ~0.36 | 0.85 |
| CDN deterministic (thr=0.35) | 0.5078 | 0.3502 | **0.5065** | 0.41 | 0.71 |

The CDN does not improve over either the baseline or the causal gate on weighted F1.
The causal gate remains the best post-hoc method for macro F1 (+0.090 over baseline).

---

## What Would Be Needed for CDN to Help

1. **Asymmetric graph**: A directed graph where parents ≠ children for most labels would
   let the CDN exploit directionality. The current Suppes graph is nearly symmetric,
   collapsing `both` into `parents` for 18/19 labels.

2. **More data**: CDN classifiers with 19–34 features need more than ~118 training traces.
   At 500+ traces, neighbor label features would provide reliable conditional signal.

3. **Better initialization at inference**: Instead of thresholding baseline probabilities
   at 0.5, an adaptive per-label threshold (calibrated on val) would give cleaner
   starting states for the Gibbs chain.

4. **Non-linear classifiers**: Logistic regression assumes linear decision boundaries in
   the (logits, neighbor labels) space. A gradient-boosted tree or small MLP conditioned
   on neighbors might capture interaction effects missed by logistic regression.
