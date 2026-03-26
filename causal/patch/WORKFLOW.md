# Causal Patch Workflow: do(A=0) Counterfactual Intervention Pipeline

This document describes the full causal intervention pipeline and includes a concrete
trace walkthrough showing every LLM query, its exact inputs, and expected outputs.

The goal is to estimate Δ(A→B) per edge using LLM-generated patches, counterfactual
reruns, and a two-stage LLM judge — without requiring human repair.

---

## Overview

```
TRAIL Traces + Annotations
        │
        ▼
[Step 0] filter_traces     → eligible_traces.json
        │
        ▼
[Step 1] case_builder      → a_instances.jsonl  (one per unique A-instance)
                           → edge_pairs.jsonl   (one per A-instance × B-type)
        │
        ▼
[Step 2] patch_generator   → patch_results.jsonl       (one per A-instance)
                           → postcheck_failures.jsonl
        │
        ▼
[Step 3] rerun_harness     → rerun_results.jsonl        (one per A-instance)
        │
        ▼
[Step 4] judge_a_resolved  → a_resolved.jsonl           (one per A-instance)
        │
        ▼
[Step 5] judge_b_effect    → b_effect.jsonl             (one per EdgePair — fan-out)
        │
        ▼
[Step 6] effect_aggregator → effect_edges.json
```

**Key deduplication invariant**: Steps 2–4 run exactly once per A-instance
(keyed by `error_id`). Step 5 fans out to all EdgePairs that share the same A-instance,
reusing the single RerunResult.

---

## File Layout

```
causal/patch/
├── WORKFLOW.md              ← this file
├── patch_library.json       ← TemplateSpec entries keyed by TRAIL A-category
├── filter_traces.py         ← Step 0: eligibility filtering
├── sample_coverage.py       ← greedy set cover sampler (for test runs)
├── case_builder.py          ← Step 1: build AInstanceRecords + EdgePairs
├── patch_generator.py       ← Step 2: slot extraction + patch generation + postcheck
├── patch_generator_llm.py   ← shared _call_llm() helper
├── rerun_harness.py         ← Step 3: counterfactual LLM rerun
├── judge_a_resolved.py      ← Step 4: Judge 1 (treatment validity)
├── judge_b_effect.py        ← Step 5: Judge 2 (outcome label)
├── effect_aggregator.py     ← Step 6: compute Δ(A→B), validate, placebo
└── run_pipeline.py          ← CLI entry point: runs Steps 0–6 end-to-end
```

---

## Causal Graph (12 AIC-pruned edges from `capri_graph.json`)

| A (source) | B (target) |
|---|---|
| Formatting Errors | Context Handling Failures |
| Formatting Errors | Incorrect Problem Identification |
| Formatting Errors | Resource Abuse |
| Formatting Errors | Tool Output Misinterpretation |
| Incorrect Problem Identification | Tool Output Misinterpretation |
| Poor Information Retrieval | Resource Abuse |
| Resource Abuse | Authentication Errors |
| Resource Abuse | Tool-related |
| Task Orchestration | Context Handling Failures |
| Tool Selection Errors | Goal Deviation |
| Tool Selection Errors | Language-only |
| Tool Selection Errors | Task Orchestration |

---

## Step 1 — Build Cases (`case_builder.py`)

**Two output dataclasses** (deduplication split):

```python
@dataclass
class AInstanceRecord:
    trace_id: str
    error_id: str            # hash of (trace_id, location)
    a_instance: dict         # {category, location (span_id), description, evidence, impact}
    local_snippet: str       # span output.value (or input.value) to be replaced
    patch_side: str          # "replace_span_output" | "replace_span_input"
    prefix_context: str      # system prompt + task description
    user_requirements: str   # task goal extracted from first user message
    tools_available: list    # tool names available to the agent
    suffix_window_spec: dict # {"mode": "until_end"}
    b_types: list            # informational: all B-types this A-instance covers

@dataclass
class EdgePair:
    trace_id: str
    error_id: str            # FK → AInstanceRecord.error_id
    edge: dict               # {"a": "Formatting Errors", "b": "Context Handling Failures"}
    b_def: dict              # TRAIL taxonomy definition for B
    b_present_baseline: bool # whether B appears after t_A in the original trace
    b_onset_baseline: int    # annotation rank of first B after t_A (-1 if absent)
```

**Logic**: For each trace, for each A-instance, create one `AInstanceRecord`.
Then for each `(A, B)` in the graph where B also appears, create one `EdgePair`.
A-instances are deduplicated by `error_id` — the same span is never patched twice.

---

## Step 2 — Patch Generation (`patch_generator.py`)

**One LLM call per A-instance.** Output keyed by `error_id`.

```python
@dataclass
class PatchResult:
    trace_id: str
    error_id: str
    location: str            # t_A span_id
    error_type: str          # A category
    template_used: str       # category key from patch_library
    patch_side: str
    patch_payload: str       # replacement text
    slot_values: dict
    postcheck_passed: bool
    postcheck_failures: list
    attempts: int
    patch_reason: str
    b_types: list
```

**Prompt structure** (single combined slot-extraction + patch call):

```
System: [PATCH_GENERATION_SYSTEM from patch_generator.py]
  — Describes the repair contract, format, and output schema.

User:
  ERROR_TYPE: {a_instance.category}
  TEMPLATE_SPEC:
    patch_side: replace_span_output
    repair_instruction: Make the smallest possible structural edit...
    forbidden_actions: [Do not add new factual content, ...]
    postcheck: [All grounded REQUIRED_MARKERS appear exactly, ...]
  SLOT_SCHEMA: {slot_schema from patch_library}
  ERROR_DESCRIPTION: {a_instance.description}
  ERROR_EVIDENCE: {a_instance.evidence}
  USER_REQUIREMENTS: {user_requirements}
  TOOLS_AVAILABLE: [page_down, find_on_page_ctrl_f, ...]
  DOWNSTREAM_ERROR_TYPES (do NOT directly fix any of these): {b_types}
  LOCAL_SNIPPET:
  <<<
  {local_snippet}
  >>>
  ...
  Respond JSON: {"reason": "...", "slot_values": {...}, "patch_payload": "..."}
```

**Postcheck** (rule-based, up to `max_retries=3`):
- Required markers in `patch_payload`
- No ungrounded novel `<...>` tokens
- Semantic content preserved
- `patch_payload` differs from `local_snippet`

---

## Step 3 — Counterfactual Rerun (`rerun_harness.py`)

**One live LLM rerun per A-instance.** Uses `litellm` to continue the agent trace
from `t_A` with the patched content.

### Message format conversion

TRAIL traces use smolagents roles (`tool-call`, `tool-response`) that must be
converted to OpenAI format before sending to the LLM:

```
smolagents → OpenAI:

role=tool-call  content="Calling tools:\n[{'id': 'call_xxx', ...}]"
    → role=assistant  tool_calls=[{id, function:{name, arguments (JSON string)}}]

role=tool-response  content="Call id: call_xxx\nObservation:\n..."
    → role=tool  tool_call_id="call_xxx"  content="<observation>"
```

### Rerun protocol

```
rerun_status values:
  live_rerun_success    — LLM ran, rerun_suffix_spans populated
  rerun_missing_suffix  — rerun failed or produced no output
```

1. Load full message history for the agent span containing `t_A`.
2. Convert all messages to OpenAI format.
3. Apply patch at `t_A`:
   - `replace_span_output`: replace the assistant message output at `t_A` with `patch_payload`
   - `replace_span_input`: replace the user/system message input at `t_A`
4. Replay original tool results for any tool calls in the patched message
   (original TOOL spans after `t_A` are indexed by tool name in timestamp order).
5. Loop up to `max_steps_after`:
   - Call LLM with current message history
   - If LLM output has tool calls → look up replayed tool result → append tool response
   - If no tool calls → planning/final-answer step → append directly
   - Collect each LLM output as a new suffix span
6. Set `rerun_status = live_rerun_success` if at least one suffix span was generated.

### max_steps_after hyperparameter

Distance analysis across all 116 GAIA traces with eligible A→B pairs:

```
Unit: LLM steps (one LiteLLMModel.__call__ per step)

                                                       N  min  p25  p50  p75  p90  max  mean
Formatting Errors -> Context Handling Failures        21    1    3    5    7   10   25   6.0
Formatting Errors -> Resource Abuse                   15    1    1    2    6    9   13   3.7
Formatting Errors -> Tool Output Misinterpretation    12    1    5    6    9   11   12   6.0
Incorrect Problem Identification -> TOMisinterpret.    7    4    4    5    7   10   10   5.9
Resource Abuse -> Tool-related                         3    5    5   21   26   26   26  17.3
Task Orchestration -> Context Handling Failures        7    1    1    2    3   10   10   2.9
Tool Selection Errors -> Goal Deviation               15    1    1    2    3    4    5   2.1
Tool Selection Errors -> Language-only                 7    1    1    2    4    5    5   2.6

Overall (102 A→B pairs):
  p50=3  p75=6  p90=10  p95=13  max=26

Coverage:
  max_steps_after=8   → covers 87% of A→B pairs
  max_steps_after=12  → covers 94%
  max_steps_after=15  → covers 96%
```

**Recommended**: `max_steps_after=12` for production runs (covers p90).
The default of 8 is a reasonable budget-constrained choice (covers p75+).

---

## Step 4 — Judge 1: A-resolved (`judge_a_resolved.py`)

**One LLM call per A-instance.**

```
System: [JUDGE_A_SYSTEM]
  "You are verifying whether a source error of type A has been eliminated by a patch.
   Be strict: if the error criterion is still met (even partially), mark resolved=false.
   Return ONLY JSON: {resolved, confidence, evidence_excerpt}"

User:
  SOURCE_ERROR_TYPE: Formatting Errors
  ERROR_DESCRIPTION: {a_instance.description[:800]}
  ERROR_EVIDENCE: {a_instance.evidence[:800]}

  ORIGINAL_SPAN:
  <<<
  {local_snippet[:2000]}
  >>>

  PATCHED_SPAN:
  <<<
  {patch_payload[:2000]}
  >>>

  RERUN_SUFFIX (first spans after t_A):
  <<<
  {rerun_suffix_spans[:3] joined with ---}
  >>>

  Has the labeled source error A been eliminated in the patched span?
  Respond with JSON only: {"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}
```

**Decision**:
- `resolved=True` → valid do(A=0) intervention → EdgePairs for this error_id proceed to Judge 2
- `resolved=False` → invalid patch → excluded from Δ(A→B) estimation

---

## Step 5 — Judge 2: B-effect label (`judge_b_effect.py`)

**Fan-out: one LLM call per EdgePair** where Judge 1 returned `resolved=True`.
All EdgePairs with the same `error_id` reuse the same `RerunResult`.

```
System: [JUDGE_B_SYSTEM]
  "You are evaluating the downstream effect of a do(A=0) intervention on error type B.
   Effect labels: disappeared | delayed | unchanged | earlier | weakened |
                  strengthened | emerged | not_observable
   Return ONLY JSON."

User:
  SOURCE_ERROR_TYPE: Formatting Errors
  TARGET_ERROR_TYPE: Context Handling Failures

  TARGET_ERROR_DEFINITION:
  Window overflow / state tracking / forgetting important context. The agent lost
  track of key prior information, failed to carry over required state...

  ORIGINAL_TRACE_SUFFIX:
  <<<
  {original_suffix_spans[:8] joined with ---}
  >>>

  ORIGINAL_ONSET_REF: annotation index 33   (or "not present in baseline")

  RERUN_TRACE_SUFFIX_AFTER_DO_A_0:
  <<<
  {rerun_suffix_spans[:8] joined with ---}
  >>>

  Required output schema:
  {
    "source_error_type": "string",
    "target_error_type": "string",
    "effect_label": "disappeared|delayed|unchanged|...",
    "target_present_after": true,
    "original_onset_ref": "string|null",
    "rerun_onset_ref": "string|null",
    "confidence": "high|medium|low",
    "evidence": "string"
  }
```

---

## Step 6 — Aggregation (`effect_aggregator.py`)

```
Δ(A→B) = mean(target_present_after | resolved=True) - mean(b_present_baseline | resolved=True)
```

A negative Δ confirms the causal edge (fixing A reduces B).
Edge is `validated=True` if `Δ < -threshold` (default 0.15) and `n >= min_n` (default 3).

---

## Concrete Trace Walkthrough

**Trace**: `0140b3f657eddf76ca82f72c49ac8e58`
**Task**: Find the surname of an equine veterinarian in LibreText 1.E Exercises
**Edge under test**: `Formatting Errors → Context Handling Failures`

### Trace structure (ToolCallingAgent children)

```
Rank  Span ID           Name                    Kind   Annotation
─────────────────────────────────────────────────────────────────────
 ...  (earlier steps: search, visit, find)
 15   07acf0e196af00a9  Step 4                  CHAIN
 16   7a97ce1ac1c66524  LiteLLMModel.__call__   LLM    ← A: Formatting Errors
 17   9996caac66d1f76e  PageDownTool            TOOL   (output: empty — TypeError)
 18   d54d914af52f93fc  LiteLLMModel.__call__   LLM
 19   786320c6301586ed  LiteLLMModel.__call__   LLM
 20   1c8d835f50401c1f  Step 5                  CHAIN
 21   89ac0ff9f2132ee0  LiteLLMModel.__call__   LLM    ← A: Formatting Errors
 ...  (Steps 6–8: more wrong page_down calls)
 30   32f1cdbcc6561422  LiteLLMModel.__call__   LLM    ← A: Formatting Errors + Resource Abuse
 31   7b86b040d6109661  PageDownTool            TOOL
 32   724da7e6dab3cf90  LiteLLMModel.__call__   LLM
 33   74dc4786aeb4908f  LiteLLMModel.__call__   LLM    ← B: Context Handling Failures
 ...
```

**A error** (span `7a97ce1ac1c66524`, rank 16 — first Formatting Error):
```
Evidence:   Error when executing tool page_down with arguments {'': '', 'arguments': {}}:
            TypeError: PageDownTool.forward() got an unexpected keyword argument ''
Description: The page_down tool (takes no arguments) was called with incorrect kwargs.

LLM output (patch_side=replace_span_output):
  role: assistant
  content: null
  tool_calls: [{'function': {'arguments': {'': '', 'arguments': {}}, 'name': 'page_down'},
                'id': 'call_ZY0nX0mwddzBA5JOyogxkOF9'}]
```

**B error** (span `74dc4786aeb4908f`, rank 33 — 5 LLM steps after A):
```
Evidence:   [PLAN]: Step 1: Scroll further down the '1.E Exercises' page a few pages
            using the page_down tool to load additional content...
Description: The agent generated a new plan still including page_down (Step 1) without
             correcting the argument usage that caused repeated errors — it failed to
             learn from prior execution failures.

LLM output:
  "1. Use the find_on_page_ctrl_f tool with alternative search strings (e.g., 'veterin',
   'vet', 'equine', and 'horse') on the '1.E Exercises' page...
   2. If a match is found, examine the surrounding excerpt to confirm that it mentions
   an equine veterinarian and contains a full name..."
```

---

### LLM Query 1 — Patch Generation (Step 2)

**When**: once for error_id of span `7a97ce1ac1c66524`

**System prompt** (abbreviated):
```
You are a precise patch generator for LLM agent traces. Your task is to fix a specific
labeled error in one agent span by applying the minimal correction described in the
template spec. ...
Respond with JSON only: {"reason": "...", "slot_values": {...}, "patch_payload": "..."}
```

**User message** (abbreviated):
```
ERROR_TYPE: Formatting Errors
TEMPLATE_SPEC:
  patch_side: replace_span_output
  repair_instruction: Make the smallest possible structural edit that satisfies the
    required format contract. Do not change semantic content unless needed.
  forbidden_actions: [Do not add new factual content, Do not add tool calls not present,
                      Do not introduce special tokens not grounded in evidence]
  postcheck: [No ungrounded special tokens introduced, Original semantic content preserved,
              patch_payload differs from local_snippet]
SLOT_SCHEMA: {REQUIRED_MARKERS, REQUIRED_FORMAT_RULES, CORRECT_CALL_SIGNATURE}
ERROR_DESCRIPTION: The page_down tool, which takes no arguments, was called with an
  incorrect dictionary structure {'': '', 'arguments': {}}.
ERROR_EVIDENCE: TypeError: PageDownTool.forward() got an unexpected keyword argument ''
USER_REQUIREMENTS: Find the surname of the equine veterinarian in 1.E Exercises.
TOOLS_AVAILABLE: [page_down, find_on_page_ctrl_f, find_next, visit_webpage, ...]
DOWNSTREAM_ERROR_TYPES (do NOT directly fix): [Context Handling Failures, Resource Abuse, ...]
LOCAL_SNIPPET:
<<<
{'role': 'assistant', 'content': None,
 'tool_calls': [{'function': {'arguments': {'': '', 'arguments': {}}, 'name': 'page_down'},
                 'id': 'call_ZY0nX0mwddzBA5JOyogxkOF9', 'type': 'function'}]}
>>>
```

**Expected output** (JSON):
```json
{
  "reason": "page_down takes no arguments; remove the malformed kwargs dict.",
  "slot_values": {"REQUIRED_FORMAT_RULES": "page_down called with empty arguments {}",
                  "CORRECT_CALL_SIGNATURE": "page_down({})"},
  "patch_payload": "{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"function\": {\"arguments\": {}, \"name\": \"page_down\"}, \"id\": \"call_ZY0nX0mwddzBA5JOyogxkOF9\", \"type\": \"function\"}]}"
}
```

---

### LLM Query 2 — Counterfactual Rerun (Step 3)

**When**: once for this A-instance, using `litellm`

**Setup**:
```
1. Load full message history at span 7a97ce1ac1c66524 (10 messages).
2. Convert smolagents roles to OpenAI format:
     tool-call     → assistant + tool_calls
     tool-response → tool + tool_call_id
3. Replace assistant message at rank 16 with patched version:
     BEFORE: tool_calls: [{'arguments': {'': '', 'arguments': {}}, 'name': 'page_down'}]
     AFTER:  tool_calls: [{'arguments': {}, 'name': 'page_down'}]
4. Replay original PageDownTool result (rank 17) as tool response message.
```

**LLM call structure** (first continuation step):
```
messages: [
  {role: system, content: "You are an expert assistant who can solve any task..."},
  {role: user,   content: "New task: Find the surname of the equine veterinarian..."},
  ... (8 more messages: prior tool-calls and tool-responses) ...
  {role: assistant, tool_calls: [{name: page_down, arguments: {}}]},  ← PATCHED
  {role: tool,      content: "(page_down output)", tool_call_id: "call_ZY0..."}  ← replayed
]
→ LLM generates next step (e.g., tries a different search approach)
```

**Repeated** up to `max_steps_after=12` (recommended) times.

**Output** (`rerun_status = live_rerun_success` if LLM responds):
```json
{
  "trace_id": "0140b3f657eddf76ca82f72c49ac8e58",
  "error_id": "...",
  "a_location": "7a97ce1ac1c66524",
  "rerun_status": "live_rerun_success",
  "rerun_suffix_spans": [
    {"role": "assistant", "content": "I'll try page_down without any arguments..."},
    ...
  ],
  "original_suffix_spans": [
    {"span_id": "74dc4786aeb4908f", "content": "1. Use find_on_page_ctrl_f with ..."}
  ]
}
```

---

### LLM Query 3 — Judge 1: A-resolved (Step 4)

**When**: once for this A-instance, after rerun

**System**:
```
You are verifying whether a source error of type A has been eliminated by a patch.
Be strict: if the error criterion is still met (even partially), mark resolved=false.
Return ONLY JSON: {"resolved": true, "confidence": 0.0, "evidence_excerpt": "string"}
```

**User**:
```
SOURCE_ERROR_TYPE: Formatting Errors
ERROR_DESCRIPTION: The page_down tool was called with incorrect kwargs {'': '', 'arguments': {}}.
ERROR_EVIDENCE: TypeError: PageDownTool.forward() got an unexpected keyword argument ''

ORIGINAL_SPAN:
<<<
{'role': 'assistant', 'tool_calls': [{'function': {'arguments': {'': '', 'arguments': {}},
  'name': 'page_down'}, 'id': 'call_ZY0...'}]}
>>>

PATCHED_SPAN:
<<<
{"role": "assistant", "tool_calls": [{"function": {"arguments": {}, "name": "page_down"},
  "id": "call_ZY0..."}]}
>>>

RERUN_SUFFIX (first spans after t_A):
<<<
[LLM continuation step 1 content...]
---
[LLM continuation step 2 content...]
>>>

Has the labeled source error A been eliminated in the patched span?
Respond with JSON only: {"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}
```

**Expected output**:
```json
{"resolved": true, "confidence": 0.92,
 "evidence_excerpt": "Patched span calls page_down with empty arguments {}, eliminating the TypeError."}
```

---

### LLM Query 4 — Judge 2: B-effect (Step 5)

**When**: one call per EdgePair (fan-out) — here for edge `Formatting Errors → Context Handling Failures`
Reuses the same `rerun_suffix_spans` from Query 2.

**System**:
```
You are evaluating the downstream effect of a do(A=0) intervention on error type B.
The source error A was locally patched at one labeled span.
Effect labels: disappeared | delayed | unchanged | earlier | weakened | strengthened | emerged | not_observable
Return ONLY JSON.
```

**User**:
```
SOURCE_ERROR_TYPE: Formatting Errors
TARGET_ERROR_TYPE: Context Handling Failures

TARGET_ERROR_DEFINITION:
Window overflow / state tracking / forgetting important context. The agent lost track
of key prior information, failed to carry over required state, or exceeded its context
window leading to incorrect behavior.

ORIGINAL_TRACE_SUFFIX:
<<<
[span 74dc4786aeb4908f]: "1. Use find_on_page_ctrl_f with alternative search strings
 ('veterin', 'vet', 'equine', 'horse') on the 1.E Exercises page..."  ← B present
---
[further original spans...]
>>>

ORIGINAL_ONSET_REF: annotation index 33

RERUN_TRACE_SUFFIX_AFTER_DO_A_0:
<<<
[rerun step 1]: "Calling page_down with corrected arguments..."
---
[rerun step 2]: "The page scrolled. Now searching for equine veterinarian..."
---
[rerun step 3]: ...
>>>

Judge how B changed after the do(A=0) intervention.
```

**Expected output**:
```json
{
  "source_error_type": "Formatting Errors",
  "target_error_type": "Context Handling Failures",
  "effect_label": "disappeared",
  "target_present_after": false,
  "original_onset_ref": "annotation index 33",
  "rerun_onset_ref": null,
  "confidence": "medium",
  "evidence": "After the page_down argument fix, the agent correctly proceeds to search
               the page rather than generating a new plan that repeats the broken tool call."
}
```

---

## Coverage Test Run (Before Full Dataset)

Before running on the full dataset, use `sample_coverage.py` to select the minimum
set of traces that covers all 12 causal edges. This lets you catch template failures,
patch generation bugs, and annotation mismatches cheaply.

### Step 1 — Sample traces (no API calls)

```bash
cd benchmarking/

# 1. Filter eligible traces once — saves alongside the causal graph
python causal/patch/filter_traces.py \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_AIC/capri_graph.json \
    --min_errors       2 \
    --strict
# → data/trail_causal_outputs_AIC/eligible_traces.json

# 2. Sample a minimal covering set from those eligible traces
python causal/patch/sample_coverage.py \
    --eligible_file data/trail_causal_outputs_AIC/eligible_traces.json \
    --causal_graph  data/trail_causal_outputs_AIC/capri_graph.json \
    --out_dir       outputs/test_run \
    --min_backup    1
# → outputs/test_run/eligible_traces_test.json  (expect ~5–10 traces)
```

`filter_traces.py` is a standalone script — run it once and the result lives with the
causal graph data. `sample_coverage.py` samples from that fixed pool (no re-scanning).
`--min_backup 1` = one trace per edge minimum (smallest possible test set).
Use `--min_backup 2` for a backup trace per edge in case one fails postcheck.

Output: `outputs/test_run/eligible_traces_test.json`

The script prints:
```
Per-edge trace availability:
    1  Resource Abuse -> Authentication Errors   *** RARE ***
    8  Resource Abuse -> Tool-related
   ...
Greedy cover: N traces cover all reachable edges
After backup (min_backup=2): M traces
Final per-edge coverage in sampled set:
    2  Formatting Errors -> Context Handling Failures
    2  Resource Abuse -> Authentication Errors
   ...
```

**Important**: if any edge shows `<<< MISSING`, it has zero traces in the full annotation
set and cannot be tested. Check whether the edge's A-type or B-type is annotated.

### Step 2 — Run pipeline on the sampled set

```bash
python causal/patch/run_pipeline.py \
    --trace_dir        data/GAIA \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_AIC/capri_graph.json \
    --eligible_file    outputs/test_run/eligible_traces_test.json \
    --out_dir          outputs/test_run \
    --model            openai/gpt-4o \
    --rerun_model      openai/o3-mini \
    --max_steps_after  12
```

`--eligible_file` bypasses Step 0 and uses the sampled file directly.
Steps 1–6 run on the sampled traces only.

### Step 3 — Review results

After the run, check for failures:

```bash
# Patch failures (wrong template, annotation mismatch, etc.)
cat outputs/test_run/postcheck_failures.jsonl | python3 -c "
import json,sys
for l in sys.stdin:
    r = json.loads(l)
    print(r['error_id'][-30:], r['postcheck_failures'])
"

# Rerun failures
python3 -c "
import json
for l in open('outputs/test_run/rerun_results.jsonl'):
    r = json.loads(l)
    if r['rerun_status'] != 'live_rerun_success':
        print(r['trace_id'][:12], r['error_id'][-20:], r['rerun_status'])
"

# Judge-A resolution rate
python3 -c "
import json
rs = [json.loads(l) for l in open('outputs/test_run/a_resolved.jsonl')]
ok = sum(1 for r in rs if r['resolved'])
print(f'{ok}/{len(rs)} A-instances resolved')
"

# Edge coverage in final output
cat outputs/test_run/effect_edges.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for k,v in d['edges'].items():
    print(f'{v[\"n_valid_interventions\"]:2d} interventions  {k}  delta={v[\"delta\"]}  validated={v[\"validated\"]}')
"
```

Common failure patterns to look for:
| Symptom | Root cause |
|---|---|
| `patch_payload is identical to local_snippet` | Annotation points to wrong span (plan text, not tool call); verify span content against description |
| `patch_payload == null` | LLM refused or template spec doesn't match error type; check `patch_library.json` |
| `rerun_status == rerun_missing_suffix` | Rerun loop terminated early; check `max_steps_after` or message format conversion |
| `resolved=false` for all instances of one A-type | Template generates wrong patch; check repair instruction in `patch_library.json` |
| Edge `n_valid_interventions == 0` | All patches failed or all A-instances unresolved for that A-type |

Fix issues, then re-run with `--skip_filter --skip_cases` to resume from Step 2.

---

## Full Dataset Run

Once the test run is clean, run on all eligible traces:

```bash
python causal/patch/run_pipeline.py \
    --trace_dir        data/GAIA \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_AIC/capri_graph.json \
    --out_dir          outputs/interventions \
    --model            openai/gpt-4o \
    --rerun_model      openai/o3-mini \
    --max_steps_after  12 \
    --max_retries      3 \
    --min_errors       2 \
    --strict_filter \
    --threshold        0.15 \
    --min_n            3
```

---

## CLI Reference

```bash
python causal/patch/run_pipeline.py \
    --trace_dir        data/GAIA \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_AIC/capri_graph.json \
    --out_dir          outputs/interventions \
    --model            openai/gpt-4o \
    --rerun_model      openai/o3-mini \
    --max_steps_after  12 \
    --max_retries      3 \
    --min_errors       2 \
    --threshold        0.15 \
    --min_n            3
```

| Argument | Default | Used by |
|---|---|---|
| `--model` | `openai/gpt-4o` | patch generation, Judge-A, Judge-B |
| `--rerun_model` | `openai/o3-mini` | counterfactual rerun only (Step 3) |
| `--eligible_file` | _(none)_ | override Step 0 with a pre-sampled trace list |

**Skip flags** (to resume partial runs):
```
--skip_filter    --skip_cases    --skip_patches    --skip_rerun
--skip_judge_a   --skip_judge_b
```

---

## Data Flow Summary

```
data/GAIA/*.json + processed_annotations_gaia/*.json
    │
    │   [TEST RUN]                              [FULL RUN]
    │   sample_coverage.py                      run_pipeline.py --strict_filter
    │         ↓                                       ↓
    │   eligible_traces_test.json          eligible_traces.json     (Step 0)
    │         ↓  (--eligible_file)               ↓
    └─────────┴──────────────────────────────────┘
              │
    ├─► outputs/*/a_instances.jsonl        (Step 1: one per A-instance)
    ├─► outputs/*/edge_pairs.jsonl         (Step 1: one per A×B edge)
    │
    ├─► outputs/*/patch_results.jsonl      (Step 2: one per A-instance)
    ├─► outputs/*/postcheck_failures.jsonl
    │
    ├─► outputs/*/rerun_results.jsonl      (Step 3: one per A-instance)
    │
    ├─► outputs/*/a_resolved.jsonl         (Step 4: one per A-instance)
    │
    ├─► outputs/*/b_effect.jsonl           (Step 5: one per EdgePair)
    │
    └─► outputs/*/effect_edges.json        (Step 6: Δ(A→B) per edge)
```
