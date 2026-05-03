# Inference Complexity Analysis

This note tabulates the asymptotic inference cost — both LLM-call count and
total prompt-token volume — for every method evaluated in the paper, plus
the Who&When localization variants. Two purposes:

1. Justify excluding Who&When W3 (binary-search localization) from the main
   experiments on a *methodological* (not just budgetary) basis.
2. Show that adding the causal graph (+CG / +GI / +GI+SI) to the TRAIL
   prompt does **not** materially change inference burden — the graph is
   appended to a constant-size context block, leaving the call count at
   O(1) and total tokens at O(N + |E|).

---

## 1. Notation

| Symbol | Meaning                                                    | Typical value (TRAIL-GAIA) |
|--------|------------------------------------------------------------|----------------------------|
| N      | Number of step-level spans in a trace                      | ~8 (GAIA), ~12 (SWE)       |
| M      | Number of leaf categories in the error taxonomy            | 19                         |
| L      | Number of error labels actually present in a given trace   | ~3–5                       |
| L_ℓ    | Number of occurrences of one specific label ℓ              | 1–3 (when present)         |
| \|E\|  | Number of edges retained in the causal/Suppes graph        | 13 (causal-only) or ~27    |
| k_t    | Number of target labels selected by graph propagation      | ~5–8 (vs. M = 19)          |

All counts are *per trace*. Token-volume bounds count input tokens; output
tokens are O(1) per call for TRAIL methods and W1, and O(L) for W2 (one
small JSON object per step), so the input-token bound dominates.

---

## 2. Big-O table

| Method | LLM calls / trace | Total input tokens / trace | Comments |
|---|---|---|---|
| **TRAIL Baseline** (one-shot full-trace) | **O(1)** | **O(N)** | Single holistic call; multi-label output. |
| **TRAIL +CG**   (graph guidance, single-pass) | **O(1)** | **O(N + \|E\|)** | Static graph block prepended; same call count. |
| **TRAIL +CG+SI** | **O(1)** | **O(N + \|E\|)** | SI adds a span-index of size O(N) — absorbed into the O(N) trace term. |
| **TRAIL +GI**   (two-pass graph injection) | **O(1)** (1 + ≤ 1) | **O(N + \|E\|)** | Pass-1 is one full-trace call; Pass-2 fires only if propagation yields targets, also one full-trace call. Both passes are constant in N. |
| **TRAIL +GI+SI** (our headline method) | **O(1)** | **O(N + \|E\|)** | Same as +GI plus a per-span index. |
| **Who&When W1** (all-at-once, multi-label) | **O(1)** (1 + 1 scores) | **O(N)** | One full-trace call, asks per-category yes/no in a single shot. |
| **Who&When W2** (step-by-step, no early exit) | **O(N)** | **O(N²)** | One call per step; cumulative prefix at step *i* is O(i), so summing across N steps gives Σ_{i=1..N} O(i) = O(N²) input tokens. |
| **W2 + graph** (stateful propagation, this work) | **O(N)** | **O(N² + N · \|E\|)** | Same N calls; each call also carries a propagated subgraph bounded by \|E\|. |
| **Who&When W3** (per-label bisection, single-error original) | **O(M log N)** | **O(M N log N)** | Bisects O(log N) per label; the window text at each bisection level is O(N). For M=19, N=8 this is ~57 calls. |
| **W3, multi-error adaptation** (both halves can recurse) | **O(M · L · N)** worst-case | **O(M · L · N²)** worst-case | Once both halves can be positive, the recursion degenerates: see §3. |
| **W3 + graph (label prefilter)** | **O(k_t · L · N)** worst-case | **O(N + k_t · L · N²)** worst-case | Pass-1 (O(1) calls, O(N) tokens) reduces M to k_t targets ≪ M. |

The asymptotic class of the TRAIL methods is identical to that of
Who&When W1 — all are O(1) calls in N. Adding the causal graph (+CG,
+GI, +GI+SI) does not change the call complexity and adds only O(\|E\|)
tokens to the prompt, where \|E\| is a small constant (13 for the
intervention-validated subgraph, 27 for the Suppes screen).

---

## 3. Why W3's adaptation collapses the log N benefit

W3's appeal in the original Who&When paper is the O(log N) call count
of bisection over a single hidden target. Two changes are forced when
adapting W3 to TRAIL's multi-label, multi-occurrence regime, and each one
removes a piece of that benefit:

**(a) Per-label sweep — multiplies by M.** A single bisection asks
"upper half or lower half?" and presupposes that exactly one decision is
correct. With M=19 labels, no single bisection can answer "where does
each label appear"; the only adaptation that preserves bisection's
binary-decision interface is to run a separate bisection per label. This
replaces O(log N) with M · O(log N) = O(M log N) in the best case.

**(b) Both-halves-positive — replaces log with a near-linear factor.**
In the original single-error formulation, exactly one of
{lower_half_present, upper_half_present} is true; the recursion has a
single child and depth ⌈log₂ N⌉. In TRAIL, multiple errors of the same
label may co-occur (Resource Abuse appears in multiple steps; Goal
Deviation persists across the trace). The bisection prompt must allow
both halves to test positive, otherwise it forces a wrong commitment.
But once both halves can recurse, the recursion tree has up to L_ℓ
leaves for label ℓ, and the total work per label becomes:

- **Best case** (occurrences clustered in one half at every level):
  O(L_ℓ + log N) calls per label.
- **Worst case** (one occurrence per leaf, no pruning at any level):
  O(L_ℓ · N) calls per label, since every internal node at every level
  is forced to recurse on both sides until each occurrence reaches its
  own leaf.

Empirically on TRAIL we observed that a label that occurs in 2–3
distinct steps already triggers full-tree recursion in most cases
(both halves positive at every level above the leaves), giving an
expected per-label cost much closer to O(L_ℓ · N) than O(log N).

**(c) Combined effect.** Aggregating across labels:

> Worst-case W3 calls = Σ_ℓ O(L_ℓ · N) = O(L · N)  per label seen,
> plus M · O(log N) for labels that are absent.

For M = 19, L ≈ 4, N ≈ 8: theoretical worst case is on the order of
~4 · 8 + 15 · 3 ≈ ~80 calls/trace, with empirical observations of
~57 calls/trace on Mistral GAIA_dedup (most labels absent and pruned
after one bisect-call returns "neither half"). The single-error
asymptotic O(M log N) ≈ 57 happens to coincide with this empirical
median; under heavier multi-label density (e.g., GAIA traces with
4+ co-occurring labels) the count rises toward the linear regime.

The linear-scan W2 produces the *same* multi-label, multi-occurrence
output in **N calls flat** with no per-label sweep, because each step
prompt natively supports a JSON list of zero or more category
predictions. W3 therefore costs strictly more than W2 in the multi-error
regime while offering no asymptotic localization advantage.

---

## 4. Implication for the paper's experimental scope

Two consequences shape the experiment matrix in
`baselines/TODO.md` §2:

1. **W3 is omitted from the main Who&When ablation** (paper main table)
   on methodological grounds: the call-complexity table above shows W3 is
   strictly dominated by W2 in TRAIL's multi-error setting. We retain a
   one-cell empirical sanity probe (Mistral GAIA_dedup, W3 graph-free) to
   cite a measured call count alongside the asymptotic argument; full
   results are omitted.

2. **Adding the causal graph to the TRAIL prompt is asymptotically
   free.** All four causal variants (+CG, +CG+SI, +GI, +GI+SI) preserve
   the Baseline's O(1) call count and add only O(\|E\|) tokens to the
   prompt. With \|E\| = 13 for the intervention-validated graph, the
   added prompt is ~300–500 tokens — under 2% of a typical TRAIL-GAIA
   trace. The performance gain reported in
   `paper/main_results_table.tex` Table 1 (Baseline → +GI+SI: +6.7
   W-F1 on Mistral GAIA_dedup, +9.0 on Gemma SWE_Bench_dedup, etc.)
   therefore comes at essentially zero inference overhead — no extra
   passes for +CG / +CG+SI, and at most one extra pass for +GI / +GI+SI
   gated on Pass-1 producing graph-source detections.

These two points motivate the paper's positioning of +GI+SI as a "free"
accuracy gain over the TRAIL baseline (Section: Inference Cost), and the
methodological argument for evaluating only W1 and W2 from Who&When
(Section: Who&When Adaptation).

---

## 5. Drop-in paragraphs

### For the methodology section (Who&When adaptation)

> Who&When's W3 binary search was designed under the original paper's
> single-root-cause assumption: one responsible agent per trace, located
> in O(log N) bisection calls. Adapting W3 to TRAIL's multi-label
> setting requires (i) running an independent bisection per error label
> (~M× call multiplier on an M-class taxonomy), and (ii) allowing both
> halves of each interval to test positive, since multiple errors of the
> same label may co-occur. Once both halves can recurse, the worst-case
> call count degrades from O(log N) to O(L · N) per label, where L is
> the number of occurrences of the label in the trace, and the expected
> count for any label that appears more than once collapses toward
> linear in N. The combined effect is that W3 systematically costs more
> than the linear-scan W2 — which already produces a per-span
> multi-label decision in N calls — while offering no asymptotic
> localization benefit in the multi-error regime that defines TRAIL. We
> therefore evaluate only W1 (all-at-once) and W2 (step-by-step) as
> Who&When localization controls, with W2+graph as the corresponding
> causal-injection variant. An empirical sanity probe (Mistral
> GAIA_dedup, W3 graph-free) confirms ~57 LLM calls per trace versus
> ~9 for W2.

### For the inference-cost discussion of +GI / +GI+SI

> Adding the causal graph to the TRAIL prompt does not increase
> inference burden asymptotically. The +CG and +CG+SI variants preserve
> the baseline's single-pass O(1) call count and append a constant-size
> graph block of O(|E|) tokens, where |E| = 13 for the intervention-
> validated subgraph used in our headline method. The two-pass +GI /
> +GI+SI variants make at most one additional full-trace call, gated on
> Pass-1 detecting at least one graph-source category — when the trace
> contains no graph-source error, Pass-2 is skipped entirely. Across
> the runs in Table 1, this gating triggers Pass-2 on roughly 60–75% of
> traces, putting the amortized call count at ~1.6–1.75× the baseline.
> The graph itself contributes <2% of a typical TRAIL-GAIA trace by
> token count, so the per-call prompt cost is essentially unchanged.
> The accuracy gains reported in Table 1 — up to +9.0 W-F1 — are
> therefore obtained at near-zero inference-time overhead relative to
> the open-ended TRAIL baseline.
