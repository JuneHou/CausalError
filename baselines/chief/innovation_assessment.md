# Innovation Assessment: Our Method vs Published CHIEF

**Question:** Given that CHIEF (arXiv 2602.23701) is published and uses "hierarchical causal graphs for failure attribution in LLM-based multi-agent systems," is our method still innovative?

**Answer:** Yes. The two methods solve fundamentally different problems and the word "causal graph" papers over six concrete differences that make our contribution non-overlapping.

---

## 1. Different problem definition

| Axis | CHIEF | Ours (on TRAIL benchmark) |
|---|---|---|
| Task | Single-answer attribution: pick the one `(agent, step)` that caused a known final-task failure | Multi-label detection: enumerate all `(error_category, span_id)` pairs that exist anywhere in the trace |
| Output cardinality | Always exactly 1 | 0 to many |
| Required inputs | Conversation + question + **ground-truth final task answer** (oracle-guided in Steps 1, 5, 6) | Trace only at inference. GAIA's correct final answer exists in the dataset metadata and could be supplied to CHIEF, but our pipeline does not consume it. TRAIL error annotations live separately in `benchmarking/processed_annotations_gaia/<trace_id>.json` and are used only for scoring. |
| Benchmark | Who&When (multi-agent dialogue, 184 traces) | TRAIL (single-agent OTel spans, GAIA + SWE-Bench) |
| Eval metric | Top-1 agent accuracy + top-1 step accuracy (binary match) | W-F1 / Location Accuracy / Joint Accuracy (multi-label) |

Even if both methods used identical machinery, the *task* CHIEF solves does not exist in TRAIL: TRAIL traces have many error spans per trace, and there is no single "responsible agent" — there is one agent and a 19-category error taxonomy. CHIEF's metric (top-1 step exact match) and TRAIL's metric (W-F1 over 19 classes × span localization) are mutually incomparable, as our paper §6 already documents.

---

## 2. Different graph definition

| Axis | CHIEF graph | Our graph |
|---|---|---|
| Nodes | Subtasks + agents (extracted per-trace) | 19 fixed error categories |
| Node identity | Trace-specific, regenerated each run | Stable across all traces |
| Edge construction | LLM is asked "what depends on what?" — natural language output, regex-parsed | Statistical estimation: Suppes precedence + probability raising + BIC hill-climb DAG search + bootstrap stability + shuffle null |
| Edge validation | None — edge strength is an LLM confidence number | Counterfactual patching: actually rewrite trace at error span A and re-run agent, measure whether B's occurrence rate drops |
| Granularity | Within-trace dependency structure | Cross-trace error-type cascade |
| Edge range | Consecutive subtasks only (S_i → S_{i+1}) | Any A → B across the 19 categories |

CHIEF's "causal graph" is a structural decomposition of *one trace's execution* asserted by an LLM. Ours is a *population-level causal model* of error co-occurrence learned from observational data and validated by intervention experiments. The shared word "causal" hides that these are different objects: CHIEF's edges encode "this subtask depends on that one"; ours encode "removing error type A reduces error type B in expectation."

---

## 3. Different inference-time mechanism

CHIEF: 6 LLM calls per trace, building a fresh DAG each time, then using Steps 5–6 to rank candidates against three hand-written rules (Loop / Data / Irreversibility). The graph is not pre-trained — it is regenerated every run.

Ours (`run_eval_graph_inject.py`): 1–2 LLM calls per trace. The graph is loaded from disk (built once offline) and used as prompt context. Pass 1 detects errors holistically; Pass 2 fires only when Pass 1 found a graph-source category, and injects only the relevant subgraph A→B where A is detected and B is not yet detected.

Per-trace cost: CHIEF ≈ 19,500 to 55,000 tokens × 6 calls. Ours ≈ 2 calls with the graph as a small prompt prefix.

---

## 4. Different validation evidence

CHIEF: ablations and accuracy numbers on Who&When. No counterfactual or interventional evidence that the graph edges reflect actual causation.

Ours: counterfactual patching pipeline (`causal/patch/`, `causal/intervention/`) that rewrites traces at the source error span, re-runs the agent, and uses an LLM judge to measure whether the downstream error occurred. Edges with no measured intervention effect are rejected. This is a fundamentally stronger causal claim — Pearl-rung-2 vs. CHIEF's rung-1.

---

## 5. CHIEF cannot be applied to TRAIL without redesign

A CHIEF run on a TRAIL trace would fail or produce meaningless output for these concrete reasons:

1. **`ground_truth` is available but unused.** GAIA / SWE-Bench provide the correct final task answer in the dataset metadata, and TRAIL error annotations are stored in `benchmarking/processed_annotations_gaia/<trace_id>.json`. So a CHIEF run *could* be configured to pass the correct task answer into the prompt as it expects (Steps 1, 5, 6) — but this makes CHIEF an **oracle-guided** baseline, while our method and all TRAIL baselines run blind. The comparison would not be apples-to-apples unless we either (a) feed the same answer to both methods or (b) modify CHIEF to drop the oracle.
2. **No multi-agent name structure.** TRAIL's single-agent traces have one agent (a `CodeAgent` or `ToolCallingAgent`), so CHIEF's agent-level DAG (Step 4) collapses to a trivial graph.
3. **Output mismatch.** CHIEF produces one `(agent_name, step_index)`; TRAIL needs zero or more `(category, span_id)` from a 19-category taxonomy CHIEF does not know about.
4. **Span ID format.** TRAIL identifies error locations by hex `span_id` (not integer step index). CHIEF's regex parsers extract integer steps.
5. **Input format.** TRAIL traces are OTel span trees with `span_kind ∈ {LLM, TOOL, CHAIN}`, not `[{name, role, content}]` agent dialogue.

Adapting CHIEF to TRAIL requires deleting Steps 5–6, replacing the output schema, removing the oracle dependency, and reframing the agent-level graph. At that point, only Steps 1–4 (the per-trace DAG construction) remain, which is a structural-context-augmented prompt — orthogonal to a population-level causal graph learned from data.

---

## 6. Verdict

Our contribution stands as:

- **First** to learn a *cross-trace, intervention-validated* causal graph over the TRAIL error taxonomy
- **First** to apply such a graph as a graph-injection prompt mechanism for multi-label error detection
- **First** to evaluate causal graph-guided inference under TRAIL's W-F1 / Loc / Joint metrics
- **Different validation rung**: counterfactual interventions vs. CHIEF's purely observational LLM assertions

CHIEF's existence does not subsume any of these. We should keep CHIEF in §6 (Related Work) framed exactly as: per-trace LLM-asserted DAG for single-answer attribution on Who&When, contrasted with our cross-trace, statistically estimated, intervention-validated DAG for multi-label detection on TRAIL.

---
