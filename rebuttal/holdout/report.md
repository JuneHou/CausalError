# Held-out vs Full-corpus Graph: Edge Diff Report

**Generated:** 2026-05-27
**Context:** Rebuttal R1 (data-leakage). Documents how the Suppes graph changes when refit on the 80% training side of the stratified held-out split, vs the full-corpus graph used in Table 1 of the main paper.

The held-out Suppes graph is built from scratch on the 80% training onsets (`data/onsets_train/{trail,mast}.jsonl`). It is **not** a subset or mask of the full-corpus graph: every per-edge statistic (joint count, precedence, pr_delta) is recomputed on the smaller sample, so the edge set, weights, and corr-union threshold passes all shift independently.

The detector consults the **corr-union at $\tau=0.35$**, i.e. edges with $\sqrt{\text{precedence} \cdot \text{pr\_delta}} \geq 0.35$. The "passes $\tau$" counts below are what actually flows into the +EDGE Stage-2 prior on the held-out 20%.

---

## TRAIL (combined GAIA + SWE)

| | Full corpus | Held-out training |
|---|---|---|
| Traces | 148 | 123 |
| Suppes edges | 27 | 24 |
| Edges passing $\tau=0.35$ corr-union | **15** | **11** |
| Edges in both | 22 |
| Only in full (lost when 20% test hidden) | 5 |
| Only in held (gained when 20% test hidden) | 2 |

### Edges only in full (5)
*Dropped when 25 test traces excluded from graph construction.*

| Edge | Precedence | pr_delta | Score | Passes $\tau=0.35$? |
|---|---|---|---|---|
| Environment Setup Errors → Goal Deviation | 0.60 | 0.25 | 0.39 | **yes** |
| Environment Setup Errors → Language-only | 1.00 | 0.23 | 0.48 | **yes** |
| Environment Setup Errors → Formatting Errors | 0.60 | 0.06 | 0.19 | no |
| Resource Abuse → Authentication Errors | 0.67 | 0.05 | 0.19 | no |
| Resource Abuse → Incorrect Problem Identification | 0.57 | 0.08 | 0.21 | no |

2 of 5 dropped edges had been in the corr-union. Both originate from "Environment Setup Errors", a category with low support (likely some traces with this onset landed in the held-out 20%, pushing pr_delta below threshold).

### Edges only in held-out (2)
*Gained when 25 test traces excluded from graph construction.*

| Edge | Precedence | pr_delta | Score | Passes $\tau=0.35$? |
|---|---|---|---|---|
| Incorrect Problem Identification → Resource Abuse | 0.60 | 0.06 | 0.19 | no |
| Instruction Non-compliance → Context Handling Failures | 1.00 | 0.10 | 0.32 | no |

Neither gained edge passes $\tau=0.35$, so neither contributes to the held-out corr-union prior.

### Shared edges (22) with weight shifts

22 edges survive in both graphs, but every one of them has a different precedence and/or pr_delta. Two shared edges that **passed $\tau$ in full but fall below in held-out**:

| Edge | Full score | Held score |
|---|---|---|
| Poor Information Retrieval → Context Handling Failures | 0.35 | 0.26 |
| Resource Abuse → Context Handling Failures | 0.39 | 0.32 |

No shared edges cross the threshold in the other direction.

### Net effect on the corr-union prior
- 4 fewer edges in the held-out corr-union (15 → 11):
  - 2 from dropped edges (Environment Setup → Goal Deviation, Environment Setup → Language-only)
  - 2 from shared edges whose score fell below threshold (both ending at Context Handling Failures)

---

## MAST

| | Full corpus | Held-out training |
|---|---|---|
| Records (graph statistics) | 393 | 319 |
| Unique trace_ids | 142 | 116 |
| Suppes edges | 43 | 31 |
| Edges passing $\tau=0.35$ corr-union | **28** | **23** |
| Edges in both | 30 |
| Only in full (lost when 20% test hidden) | 13 |
| Only in held (gained when 20% test hidden) | 1 |

### Edges only in full (13)
*Dropped when 26 held-out trace_ids excluded from graph construction.*

| Edge | Precedence | pr_delta | Score | Passes $\tau=0.35$? |
|---|---|---|---|---|
| 1.2 → 1.4 | 1.00 | 0.59 | 0.77 | **yes** |
| 1.2 → 3.3 | 1.00 | 0.42 | 0.65 | **yes** |
| 1.2 → 3.1 | 1.00 | 0.23 | 0.48 | **yes** |
| 1.4 → 3.1 | 0.90 | 0.14 | 0.36 | **yes** |
| 1.5 → 3.3 | 0.90 | 0.09 | 0.28 | no |
| 2.2 → 2.4 | 1.00 | 0.08 | 0.28 | no |
| 2.3 → 3.3 | 0.93 | 0.07 | 0.26 | no |
| 2.3 → 2.4 | 0.67 | 0.11 | 0.27 | no |
| 1.1 → 2.4 | 1.00 | 0.07 | 0.26 | no |
| 2.2 → 1.1 | 0.75 | 0.09 | 0.26 | no |
| 1.5 → 3.1 | 0.76 | 0.06 | 0.22 | no |
| 2.6 → 1.4 | 0.81 | 0.06 | 0.22 | no |
| 3.1 → 3.3 | 0.78 | 0.07 | 0.22 | no |

4 of 13 dropped edges had been in the corr-union, all originating from category 1.2 or 1.4. 1.2 (Disobey Role Specification) is one of the more sparsely populated categories in MAST, so removing 26 trace_ids has a visible effect on its outgoing edges.

### Edges only in held-out (1)
*Gained when 26 held-out trace_ids excluded from graph construction.*

| Edge | Precedence | pr_delta | Score | Passes $\tau=0.35$? |
|---|---|---|---|---|
| 2.4 → 2.3 | 0.67 | 0.66 | 0.66 | **yes** |

The single gained edge has a strong corr-union score (0.66) and is well above $\tau$.

### Shared edges (30) with weight shifts

All 30 shared edges have shifted weights. Two crossed below $\tau$ in the held-out graph:

| Edge | Full score | Held score |
|---|---|---|
| 1.4 → 3.3 | 0.45 | 0.30 |
| 2.6 → 3.1 | 0.36 | 0.34 |

No shared edges cross into the corr-union from the other direction.

### Net effect on the corr-union prior
- 5 fewer edges in the held-out corr-union (28 → 23):
  - 4 lost from dropped edges (all from category 1.2 or 1.4)
  - 2 lost from shared edges crossing below threshold (1.4 → 3.3, 2.6 → 3.1)
  - 1 gained from new edge that passes (2.4 → 2.3, score 0.66)
  - Net: -4 -2 +1 = -5

---

## Interpretation

The held-out Suppes graphs are independent re-fits, not subsets. The 80% training side has noticeably fewer edges in the final corr-union prior:

| | Full corr-union ($\tau=0.35$) | Held-out corr-union ($\tau=0.35$) | $\Delta$ |
|---|---|---|---|
| TRAIL | 15 edges | 11 edges | -4 |
| MAST  | 28 edges | 23 edges | -5 |

Most dropped edges were already near the threshold in the full graph, so the drop reflects local statistical fluctuation rather than structural collapse. The held-out graph keeps the high-score backbone: every shared edge with full score above 0.45 remains above 0.45 in held-out, and the top-scoring edges (Tool Selection Errors → Goal Deviation 0.68, Incorrect Problem Identification → Tool Output Misinterpretation 0.59 on TRAIL; 2.1 → 1.4 = 0.93, 2.1 → 1.5 = 0.73, 1.4 → 1.5 = 0.76 on MAST) are stable across the two fits.

This is exactly the condition the held-out experiment is designed to test. The graph the detector sees at test time was built without access to the 20% held-out traces, so any +EDGE gain measured on those traces cannot be attributed to population-level leakage.
