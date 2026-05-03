# Step-by-Step Walkthrough: Processing a TRAIL Trace with CHIEF

**Goal:** Trace, function-by-function, what would happen if we feed a TRAIL benchmark trace (OTel span tree, e.g. `benchmarking/data/GAIA/0035f455b3ff2295167a844f04d85d34.json`) into CHIEF's pipeline (`/data/wang/junh/githubs/CHIEF/CHIEF.py`), and identify which adapters we already have and which we still need to build.

**Convention:** "CHIEF.py:NNN" = line number in `/data/wang/junh/githubs/CHIEF/CHIEF.py`. "TRAIL./..." prefixes refer to files under `/data/wang/junh/githubs/trail-benchmark/`.

---

## 0. Findings that change the original analysis

After re-reading the existing TRAIL pipeline, the Who&When adaptation, and CHIEF's RAG code, five claims from the earlier draft were wrong or pessimistic:

1. **Cardinality is solvable.** Who&When solved the same single-answer→multi-label gap by replacing CHIEF's "rank candidates and pick one" with "decide yes/no for each of the 19 categories independently" (W1) or "loop CHIEF's algorithm per label" (W3). The same pattern applies to CHIEF's Step 5/6 — see §6 below. Cardinality is no longer a structural blocker.
2. **Adapter A is mostly already built.** `benchmarking/span_level_parser.py:parse_trace_to_step_level` already flattens OTel span trees to ordered step-spans. `benchmarking/compress_traces.py` (with `--dedup`) already collapses redundant prefixes (−67% on GAIA, −81% on SWE Bench). And `causal/graph/preprocess/trail_*.py` already builds onset/span-order tables. The only new code we need is a thin formatter that emits CHIEF's `[{"name", "role", "content"}, ...]` schema from these intermediates.
3. **CHIEF's "RAG KG" is not a knowledge graph.** It is a flat sentence-embedding index over GAIA `Question + Annotator Steps` (165 entries) and AssistantBench `Task + Explanation`, embedded with `all-MiniLM-L6-v2` and stored as `faiss.IndexFlatIP`. There is no graph structure (`build_gaia_kb.py`, `build_assistantbench_kb_faiss.py`).
4. **CHIEF's "causal graph" is a per-trace LLM-asserted DAG, not a learned graph.** No statistical estimation. No intervention. The "causal" label refers only to the words in CHIEF's prompts (Steps 2 and 4 say "construct causal edges" and "you are an expert in causal reasoning"). Edge weights are LLM confidence numbers. Failure modes come from a fixed 3-type vocabulary (`loop_issue / data_issue / irrecoverability_issue`) instructed in the prompt.
5. **Step int vs span_id is not a real obstacle.** Both are unique identifiers. As long as Adapter A maintains a `step_index ↔ span_id` table, CHIEF can run on integer indices and we can reverse-map to span_id at output time.
6. **"Multi-agent" can be reused, not abandoned.** TRAIL traces are multi-turn within one agent harness, but each turn has a distinct role/span_name (`CodeAgent.run`, `ToolCallingAgent.run`, planning step, individual tool spans, LLM spans). Treating these turn names as "agent" identifiers gives CHIEF's agent-level graph (Step 4) a non-trivial structure without conceptual abuse.

The rest of this document rewrites the walkthrough using these corrections.

---

## 1. Stage 0 — Loading the trace

### CHIEF's expectation (CHIEF.py:1060–1152)
JSON file with `question`, `ground_truth`, `history=[{name,role,content},...]`, `mistake_agent`, `mistake_step`.

### Where each field comes from for a TRAIL trace
| CHIEF field | TRAIL source | Status |
|---|---|---|
| `question` | `span_attributes["task"]` of `answer_single_question` span, or GAIA dataset metadata | Available |
| `ground_truth` | GAIA / SWE-Bench dataset gold final answer | Available (oracle-guided regime) |
| `history` | Step-spans from `span_level_parser.parse_trace_to_step_level`; each turn becomes `{name=span_name, role=("assistant"|"tool"|"system"), content=span content}` | **Adapter A**: thin formatter on top of existing pipeline |
| `mistake_agent` | Span-level annotations: agent identity at the annotated `location`. For TRAIL's single-harness traces, this is the `span_name` of the LLM/TOOL span where the error fires. | Derived from annotation |
| `mistake_step` | Annotations' `location` (span_id) → integer index of enclosing Step span via `span_level_parser.map_annotation_to_step` | Derived from annotation |

TRAIL annotations live at `benchmarking/processed_annotations_gaia/<trace_id>.json` (and `processed_annotations_swe/...`) with schema `{errors: [{category, location, evidence, description, impact}], scores: [...]}`. Each entry is a `(category, span_id)` pair.

### Adapter A — what we still need to write
We already have:
- `span_level_parser.parse_trace_to_step_level` → ordered step-spans + their LLM/TOOL children
- `compress_traces.py --dedup` → drops noise keys, hoists tool schemas, collapses prefix duplicates
- `causal/graph/preprocess/trail_2_build_span_order.py` → builds `span_rank` (span_id → integer rank)
- `causal/graph/preprocess/trail_3_build_onsets.py` → already converts `(category, span_id)` to `(category, integer_rank)` per trace

What's left for CHIEF: a ~50-line wrapper that takes the deduped trace + span_rank + GAIA dataset metadata and emits CHIEF's input JSON. Effort: low.

---

## 2. Stage 1 — `step1_generate_subtasks` (CHIEF.py:83–186)

### What it does
1. **CHIEF.py:84–97** — RAG retrieval: `retriever.search(question, top_k=2)`. Backed by `rag/rag_search.py` over `kb/gaia_kb.json` (165 examples of `Question + Annotator Steps`) and `kb/assistantbench_kb.json`. Embedding model `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP` with L2-normalized vectors. **This is a flat sentence index, not a knowledge graph** — the word "KG" was wrong in earlier descriptions.
2. **CHIEF.py:99–132** — Prompt the LLM to decompose the trace into semantic subtasks with `{id, name, step_range, oracle, evidence, loop_info}`. Self-reflection chain: Draft → Evidence → Final.
3. **CHIEF.py:134** — Single LLM call (temperature 0).
4. **CHIEF.py:136–186** — Regex-parse output.

### TRAIL fit
- The retrieved demonstrations come from CHIEF's GAIA KB (built from the full-question-answering Annotator Steps), not from TRAIL's error annotations. They show *how to solve a GAIA task*, not *how to attribute errors*. Two options:
  - **Option 1 (drop-in):** keep CHIEF's KB; demonstrations are off-topic but harmless.
  - **Option 2 (KG rebuild):** rebuild the KB from `processed_annotations_gaia/*.json` (each entry = `Question + flatten(errors)`), so retrieved demonstrations show *error-attribution patterns* on similar tasks. Effort: ~30 lines of Python following `build_gaia_kb.py`.
- Step ranges use integer indices (CHIEF.py:137 regex `Step Range:\s*(?:step)?(\d+)\s*-\s*(?:step)?(\d+)`). Adapter A's `step_index ↔ span_id` table makes this trivial.
- `oracle` field requires the LLM to know the correct intermediate output of each subtask. With GAIA `ground_truth` supplied (oracle-guided regime), CHIEF can hallucinate plausible per-subtask oracles. Without it, this field becomes vacuous.

### Output
Subtasks list, e.g. `[{id:"S1", name:"Parse question", step_range:"0-1", oracle:"...", loop_info:{is_loop_related:false,...}}, ...]`. For TRAIL GAIA traces typically 2–5 subtasks.

---

## 3. Stage 2 — `step2_generate_subtasks_edges` (CHIEF.py:188–398)

### What it does
- LLM-builds **consecutive** subtask edges (S_i → S_{i+1}). Each edge: `{from, to, type∈{data_dependency,logical_prereq}, strength, explanation, data_transfer{upstream_output, downstream_usage, consistency_score}, failure_modes:[{type∈{loop_issue,data_issue,irrecoverability_issue}, description, severity, if_from_wrong, likely_effect_on_to}]}`.
- The "causal graph" here is **constructed by the LLM in one prompt**, not learned, not validated. Edge `strength` is an LLM confidence number.
- `failure_modes` is a fixed 3-type vocabulary specified in the prompt itself (CHIEF.py:221). It is NOT derived from the subtasks empirically — the LLM is instructed to choose from `{loop_issue, data_issue, irrecoverability_issue}` and emit a description.

### TRAIL fit
- The `failure_modes` 3-type vocabulary is disjoint from TRAIL's 19-category taxonomy. Two options:
  - **Option 1 (keep CHIEF's failure modes as auxiliary metadata):** they are not used for prediction in Steps 5/6 except as soft hints; we can ignore them in the final output.
  - **Option 2 (replace failure modes with TRAIL's 19 categories):** modify the Step 2 prompt to ask `failure_modes:[{type ∈ TRAIL_TAXONOMY, ...}]`. This re-purposes the field for direct multi-label prediction, but conflates "edge-level failure annotation" with "trace-level error detection."
- Long-range data dependencies (S1 produces, S4 consumes) are not captured — only consecutive edges. For TRAIL with typically ≤5 subtasks, this is a smaller problem than for CHIEF's longer Who&When traces.

---

## 4. Stage 3 — `step3_generate_agents` (CHIEF.py:401–566)

### What it does
- Per subtask, identify acting agents and emit `Action / Observation / Thought / Result` per agent, plus intra-subtask `data_flow:[{from_step, to_step, source_agent, target_agent, data_item, ...}]`.

### TRAIL fit — multi-agent reframing
TRAIL traces are not single-agent in the trivial sense. The trace tree contains:
- `CodeAgent.run` / `ToolCallingAgent.run` (planner / executor)
- Distinct LLM spans (planning calls, reasoning calls)
- Distinct TOOL spans (each tool invocation: `web_search`, `python_executor`, `final_answer`, etc.)

Each span has a unique `span_name`. **We can treat each unique `span_name` as an agent identifier**:
- `CodeAgent.run` = "planner agent"
- Each tool span name (`web_search`, `python_executor`, ...) = a distinct "tool agent"
- Each `LLM` span name = a distinct "reasoning agent"

This gives CHIEF's Step 3 / Step 4 a non-trivial multi-agent graph to operate over, with no conceptual abuse — TRAIL traces really are multi-turn interactions between a planner LLM and a set of tools, which is structurally similar to Who&When's multi-agent dialogues.

Adapter A's `history` formatter should set `name = span_name` for each entry so this naming flows through automatically.

---

## 5. Stage 4 — `step4_generate_agents_edges` (CHIEF.py:568–700)

### What it does
- Per subtask, LLM-emits agent→agent edges with `agent_dependency_type ∈ {obs_dependency, reasoning_continuation, decision_dependency, environment_feedback, memory_ref, loop_control}`, `agent_strength`, `agent_explanation`, and `agent_failure_modes:[{type, description, severity, if_from_agent_wrong, likely_effect_on_to_agent}]`.

### TRAIL fit
With the multi-agent reframing from §4, agent edges become meaningful: `CodeAgent.run → web_search` (decision_dependency: planner decided to invoke search), `web_search → CodeAgent.run` (obs_dependency: planner ingests tool output), etc. These edges describe the planner-tool interaction structure of a single TRAIL trace.

`agent_failure_modes` is again the 3-type CHIEF vocabulary. Same trade-off as in §3 (keep as metadata or replace with TRAIL taxonomy).

---

## 6. Stages 5 & 6 — Single-answer → multi-label (cardinality fix)

### What CHIEF does (CHIEF.py:702–997)
- **Step 5:** prompt with full DAG + 3 rules (Loop / Data Tracing / Irreversibility) → list of ≥5 candidate steps with rich `loop_issue / data_issue / irrecoverability_issue / impact` annotations.
- **Step 6:** rank candidates → output a single `{mistake_agent, mistake_step, reason}`.

### Multi-label adaptation, mirroring Who&When
The Who&When plan (`baselines/who_and_when/plan.md`) shows three patterns for converting a single-answer attribution method to TRAIL's multi-label setting. We apply the same transforms to CHIEF Step 5/6:

#### Variant CHIEF-W1 (all-at-once, multi-label) — analog of W1
- Run Steps 1–4 once to build the per-trace DAG.
- **Replace Step 5+6 with a single prompt** that includes the DAG + the 3 rules + the 19-category TRAIL taxonomy, asking: *"For each of the 19 categories independently, decide whether it is present, and if so at which span_id (mapped from the candidate step_id)."*
- Output: standard TRAIL JSON `{errors: [{category, location, evidence, description, impact}, ...], scores: []}`.
- LLM calls per trace: 4 (Steps 1–4) + 1 (multi-label decision) = **5**.

#### Variant CHIEF-W2 (per-step scan, multi-label) — analog of W2
- Run Steps 1–4 once.
- For each step span in the trace, prompt: *"Given the DAG and the 3 rules, does THIS step contain any TRAIL category? Output 0–N (category, evidence) pairs."*
- Aggregate, dedupe by `(category, span_id)`.
- LLM calls per trace: 4 + N (≈8 for GAIA) = **~12**.

#### Variant CHIEF-W3 (per-label bisection) — analog of W3
- Run Steps 1–4 once.
- For each of the 19 labels, run CHIEF's bisection-style search over candidate step ids using the DAG as context. Allow both halves to be positive (multi-label).
- LLM calls per trace: 4 + 19 × ⌈log₂(N)⌉ ≈ **~70**.

#### Variant CHIEF-original-extended (closest to CHIEF)
- Keep Steps 1–6 unchanged but **loop Step 6 N times**: each iteration removes the previously selected `(agent, step)` from the candidate set and asks for the next-most-responsible step. Stop when Step 6 reports "no further responsible step" or after K iterations.
- LLM calls per trace: 4 + 1 (Step 5) + K (Step 6 loops, K≈5) = **~10**.
- Pros: minimal modification to CHIEF's code. Cons: K is a hyperparameter; precision may suffer because the rules were written for *the* most responsible step.

**Recommendation:** start with CHIEF-W1 (lowest cost, cleanest comparison), and add CHIEF-W2 or CHIEF-W3 if needed for ablations.

---

## 7. Stage 7 — Evaluation

CHIEF's binary `acc_agent / acc_step` is replaced with TRAIL's `calculate_scores.py` (W-F1, Location Accuracy, Joint Accuracy). The CHIEF-W{1,2,3} variants above all produce standard TRAIL JSON, so scoring is unchanged.

---

## 8. RAG / KG question

CHIEF does not use a knowledge graph. The RAG component (`rag/rag_search.py`) is:
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (general-purpose 384-d sentence encoder, no graph training)
- **Index:** `faiss.IndexFlatIP` over L2-normalized embeddings (flat inner-product search, no hierarchy)
- **Knowledge base:**
  - `kb/gaia_kb.json` (165 entries): each = `{id, question, steps, combined_text}` with `combined_text = "Question: {Q}\nSteps: {Annotator Steps}"`. Built from the GAIA dataset's `Annotator Metadata.Steps` field — i.e., the human-written reference solution steps for each task.
  - `kb/assistantbench_kb.json`: each = `{id, text}` with `text = "Task: ...\nExplanation: ..."` from AssistantBench dev set.
- **Build pipeline:** `rag/build_gaia_kb.py` and `rag/build_assistantbench_kb_faiss.py`. Both: load parquet/jsonl → format as text → embed → write FAISS index + JSON dump. ~50 lines each. Not benchmark-specific; trivially repointable to TRAIL data.

If we want CHIEF's Stage 1 demonstrations to be relevant to error attribution (rather than task solving), we should rebuild the KB with `combined_text = "Question: {Q}\nErrors: {flattened TRAIL annotations}"` from `processed_annotations_gaia/*.json`. This is a one-shot offline build, not part of inference.

---

## 9. CHIEF's causal graph — what the term means in their paper

The phrase "causal graph" in CHIEF refers to two things, both LLM-asserted:

1. **Subtask-level edges (Step 2):** the prompt says "you are an expert in causal reasoning" and "construct causal edges between consecutive subtask pairs." The LLM emits `data_dependency` or `logical_prereq` edges with a strength score. There is no statistical estimation, no intervention, no validation. The "causality" is whatever the LLM says it is.

2. **Agent-level edges (Step 4):** same pattern at the within-subtask agent level. Edge types are limited to `obs_dependency / reasoning_continuation / decision_dependency / environment_feedback / memory_ref / loop_control`.

There is **no offline graph construction, no learning, no validation pipeline**. Compared to our `causal/graph/CAPRI/` (Suppes precedence + BIC hill-climb + bootstrap stability + shuffle null) and `causal/patch/` (counterfactual rerun + LLM judge), CHIEF's "causal graph" is at a different rung of the causal-inference hierarchy.

### Where CHIEF's failure modes come from
The 3-type vocabulary `{loop_issue, data_issue, irrecoverability_issue}` is **defined in the Step 2 and Step 4 prompts themselves** (CHIEF.py:221, 600–605). It is not derived empirically from subtasks or learned from data — it's a hand-designed taxonomy chosen to support the 3 attribution rules in Step 5/6. Each subtask edge / agent edge can have zero or more failure modes attached, with the LLM choosing the type.

---

## 10. Summary of adapters required (revised)

| Adapter | Purpose | Status |
|---|---|---|
| **A — Trace ingestion** | OTel span tree → CHIEF's `[{name, role, content}]` history | Mostly built (in `span_level_parser.py` + `compress_traces.py` + `causal/graph/preprocess/`); ~50 lines of new code |
| **B — Annotation reduction (single-answer eval only)** | Multi-label TRAIL annotations → single `(mistake_agent, mistake_step)` for CHIEF's binary scoring | Not needed if we use CHIEF-W{1,2,3} — they output multi-label directly and score with `calculate_scores.py` |
| **C — Multi-agent reframing** | Use `span_name` of each TRAIL turn as agent identifier | Trivial (in Adapter A's history formatter) |
| **D — step↔span_id reverse map** | Step indices in CHIEF output → span_id hex for TRAIL JSON | Trivial (build the table in Adapter A, use at output time) |
| **E — Multi-label wrapper for Steps 5/6** | Convert single-answer ranking to per-label or per-step multi-label | Implement one of CHIEF-W1 / W2 / W3 as in §6 |
| **F — KB repointing (optional)** | Rebuild CHIEF's RAG KB from TRAIL annotations instead of GAIA Annotator Steps | Optional; ~30 lines of Python following `build_gaia_kb.py` |

None of these are conceptually hard. The earlier draft overstated the difficulty.

---

## 11. Files

```
baselines/chief/
├── plan.md                       ← this file
└── innovation_assessment.md      ← whether CHIEF's publication subsumes our method
```
