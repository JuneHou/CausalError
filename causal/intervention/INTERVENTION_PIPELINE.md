# TRAIL Pipeline — Usage Guide

This pipeline has **three phases**: (1) **operator-family aggregation** (action primitive dictionary), (2) **intervention** (apply patches and optionally re-run counterfactuals), and (3) **evaluation** (prevention effects Δ(A→B)). Run them in order.

| Phase | Script | Purpose |
|-------|--------|---------|
| 1. Aggregation | `action_primitive_library.py` | Build tool/control-flow primitives, templates, and primitive↔error co-occurrence from traces + annotations. Informs which operator families exist and how they correlate with error types. |
| 2. Intervention | `causal/intervention/intervene.py` or `rerun_intervention.py` | **Static**: apply patch specs to snippets, write patch log. **Re-run**: for each error A_i create single-intervention run do(A_i≈0), save counterfactual traces in GAIA format for per-edge attribution. |
| 3. Evaluation | `causal/intervention/effect_eval.py` | Compute Δ(A→B) and edge validation from patch log or rerun log and baseline annotations. |

---

## Directory layout

```
benchmarking/
├── action_primitive_library.py  # Phase 1: build action primitive dictionary
├── trail_io.py                  # TRAIL trace adapter: load trace + annotations → TraceObj
├── span_level_parser.py         # (dependency) step-level parsing, annotation mapping
├── intervene.py                 # Entry point → causal/intervention/intervene.py
├── effect_eval.py               # Entry point → causal/intervention/effect_eval.py
├── rerun_intervention.py       # Entry point → causal/intervention/rerun_intervention.py
├── causal/
│   └── intervention/           # All intervention code
│       ├── patch_apply.py      # Patch library loader, spec instantiation, apply_patch
│       ├── intervene.py        # Main intervention loop (static + optional rerun)
│       ├── rerun_intervention.py  # Single-intervention re-runs, GAIA-format output
│       ├── trace_replay.py     # Conversation before span, tool replay for rerun
│       └── effect_eval.py      # Δ(A→B) and edge validation
├── data/
│   ├── patches/                 # 8 operator-family JSON specs (required for Phase 2)
│   │   ├── BUDGET_GUARD_STOP_CONDITION.json
│   │   ├── CONTEXT_STATE_CARRYOVER.json
│   │   ├── EXECUTE_INSTEAD_OF_DESCRIBE.json
│   │   ├── GOAL_CONSTRAINT_CHECK.json
│   │   ├── OUTPUT_INTERPRETATION_VERIFY.json
│   │   ├── RETRIEVAL_REQUERY.json
│   │   ├── TOOL_SCHEMA_REPAIR.json
│   │   └── TOOL_SELECTION_SWAP.json
│   └── GAIA/                    # (example) trace JSONs: <trace_id>.json
├── processed_annotations_gaia/  # (example) annotation JSONs: <trace_id>.json
├── action_primitive_artifacts/  # Phase 1 outputs
│   ├── action_primitives.json
│   ├── primitive_error_stats.json
│   ├── templates.json
│   └── primitive_error_comparison.json   # (if --window_sizes has multiple values)
├── data/GAIA_interventions/     # counterfactual traces (full trace, one span patched), same format as GAIA
│   └── <trace_id>_do_<i>_<id>.json
├── data/GAIA_interventions_rerun/  # (when --mode rerun only) one *_rerun.json per intervention (all rerun data inside)
└── outputs/
    └── interventions/           # Phase 2 & 3 outputs
        ├── patch_log.jsonl
        ├── patched_traces.jsonl
        ├── rerun_log.jsonl      # (when using --rerun) one line per do(A_i≈0) run
        ├── intervention_stats.json
        └── effect_edges.json
```

---

## Prerequisites

- **Trace JSONs**: One file per run (e.g. `data/GAIA/<trace_id>.json`), OpenInference-style with `spans`, `span_attributes`, `output.value`, etc.
- **Annotation JSONs**: One file per trace (e.g. `processed_annotations_gaia/<trace_id>.json`), each with an `errors` array. Each error must have:
  - `location` (span_id where the error occurs)
  - `category` (error type; used in Phase 1 for co-occurrence and in Phase 2 for routing)
  - Optional: `evidence`, `description`, `impact`
- **Patch specs (Phase 2 only)**: All 8 JSON files in `data/patches/`. The intervention loop loads every `*.json` in that directory.

Run all scripts from the **`benchmarking/`** directory so that imports (`span_level_parser`, `trail_io`, `patch_apply`) resolve.

---

## How to run the pipeline (three phases)

Run from the **`benchmarking/`** directory. Use the same trace and annotation directories across phases.

### Phase 1: Operator-family aggregation (action primitive dictionary)

Builds the vocabulary of actions from traces: tool primitives, control-flow primitives, canonical templates, and primitive↔error co-occurrence (windowed). Use this to understand which tools and behaviors correlate with which error types before designing or running interventions.

```bash
cd benchmarking

python action_primitive_library.py \
  --trace_dir       data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --out_dir         action_primitive_artifacts \
  --max_traces      100
```

- **First run**: Use `--max_traces 5` or `10` to sanity-check.
- **Full run**: Omit `--max_traces` to use all trace/annotation pairs.
- **Single trace**: `--trace_ids b241cb7deedf9646f01fa15095ed96d2`
- **Multi-window comparison**: `--window_sizes 1 2 3` to produce `primitive_error_comparison.json` (stability of primitive↔error stats across window sizes).

**Outputs (Phase 1):**

| File | Description |
|------|-------------|
| `action_primitive_artifacts/action_primitives.json` | Tools, arg keys, control_flow counts, code_actions, templates. |
| `action_primitive_artifacts/primitive_error_stats.json` | Primitive/tool counts per error type (windowed). |
| `action_primitive_artifacts/templates.json` | Generic + literal templates per tool. |
| `action_primitive_artifacts/primitive_error_comparison.json` | Only if `--window_sizes` has multiple values; stability report across windows. |

---

### Phase 2: Intervention (apply patches)

Loads traces and annotations, routes each error’s category to an operator family, instantiates the patch spec, applies it to the span snippet, validates, and writes the patch log and patched traces.

```bash
python intervene.py \
  --trace_dir       data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --patch_specs_dir data/patches \
  --out_dir         outputs/interventions \
  --max_traces      5 \
  --window          0
```

- **First run**: Use `--max_traces 5` to confirm everything works.
- **Full run**: Omit `--max_traces` to process all trace/annotation pairs.
- **Snippet window**: `--window 0` = only the error span; `--window 1` = ±1 sibling span for more context.

**Outputs (Phase 2):**

| File | Description |
|------|-------------|
| `outputs/interventions/patch_log.jsonl` | One JSON object per line: each applied patch (trace_id, error_id, operator_family, location, original_text, patched_text, validation, success). |
| `outputs/interventions/patched_traces.jsonl` | One JSON object per trace: trace_id and list of patch records for that trace. |
| `outputs/interventions/intervention_stats.json` | Counts: traces processed, errors seen, patches attempted/succeeded/failed, skipped (no family/location), and counts by operator family. |

---

### Phase 2b: Re-run intervention (single-intervention counterfactuals)

Instead of (or in addition to) static patching, you can run **single-intervention counterfactuals**: for each error A_i at time t_Ai, create one run **do(A_i≈0)** that fixes that error and saves a full trace in GAIA format. This gives clean per-edge attribution for effect evaluation.

**What the re-run does:**

1. **No truncation for causal evaluation**: The counterfactual is the **full trace** with only the error span replaced (do A_i≈0). We **do not** truncate after the patch — truncation would remove later steps and break causal evaluation (later errors/sequence). Use `trace_replay.truncate_trace_after_span` only for patch-validity checks, not for rerun.
2. **Finding the patched conversation**: Each saved JSON has **`intervention_span_id`** — the `span_id` of the patched turn. Use `trace_replay.get_patched_span_content(trace_data, trace_data["intervention_span_id"])` to read the patched text.
3. **One run per error**: For errors A1, A2, A3 you get three counterfactual traces: `Run(do(A1≈0))`, etc., each a full trace with one span patched.
4. **Step ordering**: `trace_replay.get_ordered_steps()` uses **execution order** (timestamps: `start_time_unix_nano` or `timestamp`), not DFS, so before/after span slicing and tool-output tape are consistent.

**How history is input**

- The **trace** holds history: each LLM span has `llm.input_messages.*` — the messages sent to that call.
- For **Option 2 rerun** (prefix + patched turn + observation tape), use **`trace_replay.get_llm_input_messages_for_span(trace_data, span_id)`** to get the **exact** message list for the LLM call at (or nearest before) the intervention span — no duplication. Do **not** use `get_conversation_before_span` for the rerun prefix (it concatenates each LLM’s full input and duplicates prior turns).

**Option A — Re-run via intervene.py (patch + rerun in one go):**

```bash
python intervene.py \
  --trace_dir       data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --patch_specs_dir data/patches \
  --out_dir         outputs/interventions \
  --gaia_output_dir data/GAIA_interventions \
  --max_traces      5 \
  --rerun
```

**Option B — Re-run only (standalone):**

```bash
# Static patch only (default): full trace with patched span, no model call
python rerun_intervention.py \
  --trace_dir       data/GAIA \
  --gaia_output_dir data/GAIA_interventions \
  --mode patch_only --max_traces 5

# Option 2 rerun: prefix + tape + replay driver, write transcript to GAIA_interventions_rerun
python rerun_intervention.py \
  --mode rerun \
  --gaia_rerun_dir  data/GAIA_interventions_rerun \
  --max_traces      5
```

**Outputs (Phase 2b):**

| File / directory | Description |
|------------------|-------------|
| `outputs/interventions/rerun_log.jsonl` | One JSON object per intervention: trace_id, error_id, location, output_path, success; if `--mode rerun`, also `rerun_paths`. |
| `data/GAIA_interventions/<trace_id>_do_<i>_<id>.json` | Only when `--mode patch_only`. Readable JSON: `trace_id`, `steps`, `intervention_annotation`, `raw_trace`. |
| `data/GAIA_interventions_rerun/` (when `--mode rerun`) | One `*_rerun.json` per intervention (trace_id, intervention_annotation, messages_prefix, observation_tape, rerun_transcript). No patch-only folder. |

---

### Understanding rerun outputs (example: trace 0035f455b3ff2295167a844f04d85d34)

**Filename pattern:**  
`<trace_id>_do_<error_index>_<safe_error_id>.*`  
Example: `0035f455b3ff2295167a844f04d85d34_do_0_..._Instruction_No` = trace 0035f…, **do(error_0)** (first error). So **do_0** = fix error 0 only, **do_1** = fix error 1 only, **do_2** = fix error 2 only.  
New runs write one file per intervention: `*_rerun.json` (keys: `messages_prefix`, `observation_tape`, `rerun_transcript`, `intervention_annotation`). Older runs may have three separate files: `*_messages_prefix.json`, `*_observation_tape.json`, `*_rerun_transcript.jsonl`; the meaning is the same.

**What each piece is:**

| Artifact | Meaning |
|----------|--------|
| **messages_prefix** | Exact input to the model at the intervention point: system/user/assistant history **plus** the **patched** assistant message (the corrected turn). So the **last message** in this list is the **patch output** (what we injected). |
| **observation_tape** | Tool outputs from the **original** trace **after** the intervention span, replayed in order. The model sees these as “Observation (tool X): …” after each turn. |
| **rerun_transcript** | Conversation **after** the patch: turn 0 = patched assistant message (`"source": "patched"`), then model-generated turns and tape observations. Use this to see how the run continued once we fixed that one error. |

**How to check patch input vs output**

- **Patch input** = the **original** bad content at the intervention span (e.g. plan without `<end_plan>`). You get it from the **baseline trace** or from **GAIA_interventions** (patch_only) in `steps` where `is_patched: false` for that span before the patch, or from the annotation’s `evidence`.
- **Patch output** = the corrected content we inject. In rerun outputs it is:
  - **messages_prefix**: last element = `{"role": "assistant", "content": "<patched text>"}`.
  - **rerun_transcript**: first line, `"source": "patched"`, same content.
- In **GAIA_interventions** (patch_only): in `steps`, the step with `"is_patched": true` is the patch output; `intervention_annotation` describes the error we patched (error_type, evidence, description).

**How to evaluate effects using these logs**

1. **Per intervention (e.g. do_0):**  
   - Read **rerun_transcript**: did the model reach a final answer? Did it stay on task? Any new errors (e.g. wrong format, tool misuse)?  
   - Compare **observation_tape** with what the model requested: are replayed observations consistent with the new behavior?

2. **Across interventions (do_0 vs do_1 vs do_2):**  
   - **do_0** fixes “Instruction Non-compliance” at span 98fa… (missing `<end_plan>`). Rerun transcript shows: patched plan → tape (final_answer) → model continues and gives answer 33040 / 33149.  
   - **do_1** fixes a later error (e.g. Tool-related); **do_2** fixes Goal Deviation. Compare final answers and trajectory: does fixing error 0 change downstream errors or the final answer?

3. **Causal effect:**  
   - Run **effect_eval.py** with `rerun_log.jsonl` to get Δ(A→B).  
   - Manually: for each error type B, check whether in the **do(A≈0)** rerun transcript, B still appears (or appears earlier/later). Presence drop or onset shift = causal effect of A on B.

**Effect evaluation with re-run:** Use the same `effect_eval.py`; it accepts either `patch_log.jsonl` or `rerun_log.jsonl` (or the path to `rerun_log.jsonl` via `--patch_log`). If you pass the interventions dir, effect_eval will use the rerun log when present.

```bash
python effect_eval.py \
  --patch_log       outputs/interventions/rerun_log.jsonl \
  --annotations_dir processed_annotations_gaia \
  --out             outputs/interventions/effect_edges.json
```

---

### Phase 3: Evaluation (prevention effects)

Reads the patch log and baseline annotations to compute Δ(A→B) and edge validation. Does not re-execute the agent; uses annotation order to estimate downstream presence and timing.

```bash
python effect_eval.py \
  --patch_log       outputs/interventions/patch_log.jsonl \
  --annotations_dir processed_annotations_gaia \
  --out             outputs/interventions/effect_edges.json \
  --threshold       0.1
```

- **Optional**: `--stage1_edges path/to/aic_graph.json` to reweight edges from an existing causal graph (AIC/BIC).
- **`--threshold`**: An edge A→B is **validated** when Δ(A→B) ≥ this value (default 0.1).

**Outputs (Phase 3):**

| File | Description |
|------|-------------|
| `outputs/interventions/effect_edges.json` | presence_drop_matrix, timing_shift_matrix, edge_validation, top_effects per error type, and (if provided) reweighted_stage1_edges. |

The script also prints a short summary to stdout: total patches, success rate, and top prevention effects Δ(A→B) with validation marks.

---

### One-shot example (all three phases, small scale)

```bash
cd benchmarking

# 1. Aggregation (e.g. 10 traces, single window)
python action_primitive_library.py \
  --trace_dir data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --out_dir action_primitive_artifacts \
  --max_traces 10

# 2. Intervention (same 10 traces by default discovery)
# Static only:
python intervene.py \
  --trace_dir data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --patch_specs_dir data/patches \
  --out_dir outputs/interventions \
  --max_traces 10 \
  --window 0
# With re-run (counterfactual traces in data/GAIA_interventions):
# python intervene.py ... --rerun --gaia_output_dir data/GAIA_interventions

# 3. Evaluation
python effect_eval.py \
  --patch_log outputs/interventions/patch_log.jsonl \
  --annotations_dir processed_annotations_gaia \
  --out outputs/interventions/effect_edges.json \
  --threshold 0.1
```

---

## Error type → operator family routing

Each annotation `category` is mapped to exactly one operator family (top-1). Routing is defined in `intervene.py`; main mappings:

| Error type (annotation) | Operator family |
|-------------------------|-----------------|
| Resource Abuse, Authentication Errors | BUDGET_GUARD_STOP_CONDITION |
| Poor Information Retrieval | RETRIEVAL_REQUERY |
| Formatting Errors | TOOL_SCHEMA_REPAIR |
| Tool Selection Errors | TOOL_SELECTION_SWAP |
| Context Handling Failures | CONTEXT_STATE_CARRYOVER |
| Tool-Related | OUTPUT_INTERPRETATION_VERIFY |
| Goal Deviation, Language-Only, Instruction Non-compliance, Incorrect Problem Identification | GOAL_CONSTRAINT_CHECK |
| Task Orchestration | EXECUTE_INSTEAD_OF_DESCRIBE |

If an error type has no entry, it is skipped and counted under “Skipped (no fam)” in the stats.

---

## Interpreting outputs

### `patch_log.jsonl`

Each line is a patch attempt. Important fields:

- `trace_id`, `error_id`, `operator_family`, `location` (span_id)
- `original_text`, `patched_text`: the snippet before/after the patch (truncated in the log).
- `validation.ok` / `validation.reasons`: whether the patch passed the cheap checks (something changed, minimal diff, no fabricated Observation, guard keywords for budget family).
- `success`: true only if validation passed.

### `intervention_stats.json`

- **patches_succeeded** / **patches_failed**: number of patches that passed or failed validation.
- **skipped_no_family**: errors whose category had no routing.
- **skipped_no_location**: errors with no `location` (span_id).
- **by_family**: number of patches applied per operator family.

### `effect_edges.json`

- **presence_drop_matrix[A][B]**: fraction of traces (where we patched A) in which B appeared *downstream* of A. This is an annotation-based proxy for “if A is removed, how often B was downstream.”
- **timing_shift_matrix[A][B]**: average span-index gap (how many “steps” after A does B first appear).
- **edge_validation[A][B]**: for each A→B, `delta`, `validated` (true if delta ≥ threshold), and optionally `in_stage1_graph`.
- **top_effects**: for each patched error type A, the top-5 downstream B types by Δ(A→B).

When using **static patch log** only, “effect” is an annotation-based proxy: *given baseline annotations, how often B occurs after A*. When using **re-run intervention** (Phase 2b), counterfactual traces are written in GAIA format; you can run task evaluation or annotation on those traces to measure how B actually changes under do(A_i≈0), then use `effect_eval.py` with `rerun_log.jsonl` for per-edge attribution.

---

## CLI reference

### action_primitive_library.py (Phase 1 — aggregation)

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_dir` | `data/GAIA` | Directory of trace JSON files. |
| `--annotations_dir` | `processed_annotations_gaia` | Directory of per-trace annotation JSON files. |
| `--out_dir` | `action_primitive_artifacts` | Output directory for action_primitives.json, primitive_error_stats.json, templates.json. |
| `--trace_ids` | (all with annotations) | Optional: space-separated list of trace IDs. |
| `--max_traces` | None | Cap number of traces to process. |
| `--window_sizes` | `1` | Window size(s) for primitive-error stats. Use `--window_sizes 1 2 3` to also produce primitive_error_comparison.json. |

### intervene.py (Phase 2 — intervention)

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_dir` | `data/GAIA` | Directory of trace JSON files. |
| `--annotations_dir` | `processed_annotations_gaia` | Directory of annotation JSON files (same basename as trace). |
| `--patch_specs_dir` | `data/patches` | Directory containing operator-family JSON specs. |
| `--out_dir` | `outputs/interventions` | Where to write patch_log.jsonl, patched_traces.jsonl, intervention_stats.json. |
| `--trace_ids` | (all) | Optional: space-separated list of trace IDs to process. |
| `--max_traces` | None | Cap number of traces (e.g. 5 for testing). |
| `--window` | 0 | Snippet expansion: 0 = only error span; 1 = ±1 sibling span. |
| `--rerun` | false | Also run single-intervention counterfactuals and write GAIA-format traces. |
| `--gaia_output_dir` | `data/GAIA_interventions` | Output directory for counterfactual traces (used when `--rerun`). |

### rerun_intervention.py (Phase 2b — re-run only)

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_dir` | `data/GAIA` | Directory of trace JSON files. |
| `--annotations_dir` | `processed_annotations_gaia` | Directory of annotation JSON files. |
| `--patch_specs_dir` | `data/patches` | Directory containing operator-family JSON specs. |
| `--out_dir` | `outputs/interventions` | Where to write rerun_log.jsonl. |
| `--gaia_output_dir` | `data/GAIA_interventions` | Output directory for counterfactual traces (GAIA format). |
| `--trace_ids` | (all) | Optional: list of trace IDs to process. |
| `--max_traces` | None | Cap number of traces. |
| `--window` | 0 | Snippet expansion for patch (0 = error span only). |
| `--mode` | `patch_only` | `patch_only`: write to gaia_output_dir only. `rerun`: write to gaia_rerun_dir only (one JSON per intervention; no patch output). |
| `--gaia_rerun_dir` | `data/GAIA_interventions_rerun` | For `--mode rerun`: single output dir; each intervention → one `*_rerun.json`. |

### effect_eval.py (Phase 3 — evaluation)

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch_log` | `outputs/interventions/patch_log.jsonl` | Patch log from Step 1. |
| `--annotations_dir` | `processed_annotations_gaia` | Same annotations used for intervention (for baseline error order). |
| `--stage1_edges` | None | Optional path to causal graph JSON for edge reweighting. |
| `--out` | `outputs/interventions/effect_edges.json` | Output path for effect table. |
| `--threshold` | 0.1 | Δ threshold for marking an edge as validated. |

---

## Programmatic use

You can run each phase from Python:

```python
# Phase 1: aggregation
from action_primitive_library import build_library
build_library(
    trace_dir="data/GAIA",
    annotations_dir="processed_annotations_gaia",
    out_dir="action_primitive_artifacts",
    max_traces=10,
    window_sizes=[1],
)

# Phase 2 & 3: intervention and evaluation
from trail_io import load_trail_trace, get_expanded_snippet
from patch_apply import load_patch_specs, apply_patch
from intervene import route_error_to_family, run_interventions
from effect_eval import compute_effects

# Single trace + one error
trace_obj = load_trail_trace("data/GAIA/abc123.json", "processed_annotations_gaia/abc123.json")
specs = load_patch_specs("data/patches")
err = trace_obj.errors[0]
family = route_error_to_family(err.get("category", ""))
if family and family in specs:
    record = apply_patch(trace_obj, err, specs[family], window=0)
    print(record.success, record.patched_text[:200])

# Full intervention run (Phase 2)
run_interventions(
    trace_dir="data/GAIA",
    annotations_dir="processed_annotations_gaia",
    patch_specs_dir="data/patches",
    out_dir="outputs/interventions",
    max_traces=10,
    window=0,
)

# Effect evaluation (Phase 3)
compute_effects(
    patch_log_path="outputs/interventions/patch_log.jsonl",
    annotations_dir="processed_annotations_gaia",
    out_path="outputs/interventions/effect_edges.json",
    stage1_edges_path=None,
    validation_threshold=0.1,
)
```

---

## Troubleshooting

- **“no routing for error_type”**: Add that category to `ERROR_TYPE_TO_FAMILY` in `intervene.py`, or normalise the label (e.g. singular/plural) in `_NORM_MAP`.
- **“no_change: patched text is identical”**: The instantiator could not find an anchor (e.g. no “I will call” phrase, no bad keyword in evidence). Patch is skipped; check evidence/location for that error.
- **“fabricated_output”**: The patch inserted text that looks like new “Observation:” content; validation rejects it.
- **“guard_missing”**: A BUDGET_GUARD patch was applied but the patched text does not contain the required guard keywords (max_retries, stop_condition, retry, etc.); check the instantiator output.
- **Malformed annotation JSON**: Some annotation files may be invalid JSON; `effect_eval.py` skips them and continues. Fix or remove bad files if you need those traces in the effect table.
