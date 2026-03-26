# LLM-based patch creation pipeline

This is an **alternative to the operator-family mapping**: patches are created **one by one with an LLM** (diagnosis + minimal repair), with no predefined operator families.

## Flow

1. **Input to patch generator**  
   For each error: annotation (type, evidence, description) + problematic span text + minimal local context (neighbour spans).

2. **Mechanism diagnosis**  
   Ask the LLM to identify the proximal mechanism and choose the repair target:
   - **input_context** — fix or add context (prompt, prior output, tool result).
   - **local_decision_policy** — change the agent’s decision (tool choice, reasoning, add guard).
   - **output_surface** — fix only the output of this span (format, missing tag, malformed JSON).

3. **Single-error patch generation**  
   Ask the LLM to produce the **smallest repair** that fixes only this error in this span and does not alter unrelated behavior.

4. **Rerun request**  
   Each successful patch is written as a **rerun request** (trace_id, intervention_span_id, patch_payload, apply_mode, etc.) to `rerun_requests.jsonl`. Patches are **applied and the trace is rerun by different code** after this step; this pipeline only generates the patch payloads and request records.

5. **Evaluation**  
   - **Compare downstream B**: after you have rerun results, use `compare_baseline_rerun_errors(baseline_errors, rerun_errors)` to get delta presence by error type.

## Requirements

- **litellm** (same as `run_eval.py`). Install with: `pip install litellm`
- **OpenAI API key** (or other provider) set in the environment as expected by litellm (e.g. `OPENAI_API_KEY`).

## How to run

From the `benchmarking/` directory:

```bash
# All traces with annotations, default model gpt-4o
python patch_generator_llm.py \
  --trace_dir data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --out_dir outputs/llm_patches

# Limit traces and errors per trace (for testing)
python patch_generator_llm.py \
  --trace_dir data/GAIA \
  --annotations_dir processed_annotations_gaia \
  --out_dir outputs/llm_patches \
  --max_traces 3 \
  --max_errors_per_trace 2

# Use a different model
python patch_generator_llm.py \
  --model openai/gpt-4o-mini \
  --out_dir outputs/llm_patches
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/llm_patches/patch_log.jsonl` | One JSON object per error: trace_id, error_id, location, error_type, diagnosis_target, diagnosis_reasoning, success, error_message. |
| `outputs/llm_patches/rerun_requests.jsonl` | One JSON object per **successful** patch: trace_id, intervention_span_id, error_id, error_type, diagnosis_target, diagnosis_reasoning, **apply_mode** (replace_span_output | prepend_context_before_rerun | inject_local_policy_before_rerun), original_span_text, **patch_payload** (replacement span or context/policy snippet), and **instruction** for the rerun harness. |

## Counterfactual rerun

Each request has **apply_mode** and **patch_payload**: `replace_span_output` (use payload as new span content), `prepend_context_before_rerun` (insert payload as context before the span), or `inject_local_policy_before_rerun` (insert payload as a local rule before the span). **Apply and rerun are done by separate code** after this step; that code should:

1. Read each line of `rerun_requests.jsonl`.
2. Load the trace for `trace_id`.
3. Apply the patch per `apply_mode`: use `patch_payload` to replace span content, prepend context, or inject local policy before rerunning the intervention span.
4. Regenerate the trace suffix from that intervention point (your execution environment may support “resume from span” or “replay from checkpoint”).

The pipeline does **not** run the agent; it only produces the patch and the rerun request.

## Evaluation after rerun

1. **Test whether downstream B changes**  
   Load baseline annotations for the trace and annotations (or error list) from the **rerun** trace. Then:

   ```python
   from patch_generator_llm import compare_baseline_rerun_errors

   baseline_errors = [...]  # list of { "category": "...", ... } from original annotation
   rerun_errors = [...]     # list from rerun trace annotation or your evaluator

   result = compare_baseline_rerun_errors(baseline_errors, rerun_errors)
   # result["delta_presence"]: { "ErrorType": baseline_count - rerun_count }
   # result["summary"]: short text summary
   ```

Use `delta_presence` to see which error types decreased (positive delta = fewer in rerun) or increased after the intervention.

## Programmatic use

```python
from trail_io import load_trail_trace
from patch_generator_llm import (
    build_patch_input,
    diagnose_mechanism,
    generate_single_error_patch,
    run_llm_patch_pipeline,
    compare_baseline_rerun_errors,
)

trace_obj = load_trail_trace("data/GAIA/abc.json", "processed_annotations_gaia/abc.json")
err = trace_obj.errors[0]

# Full pipeline for one error
result = run_llm_patch_pipeline(trace_obj, err, model="openai/gpt-4o")
if result.success:
    print(result.diagnosis.target, result.patched_span_text[:200])
    # result.rerun_request for your rerun harness
```

## CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_dir` | `data/GAIA` | Directory of trace JSONs. |
| `--annotations_dir` | `processed_annotations_gaia` | Directory of annotation JSONs. |
| `--out_dir` | `outputs/llm_patches` | Output for patch_log.jsonl and rerun_requests.jsonl. |
| `--trace_ids` | (all) | Optional list of trace IDs. |
| `--max_traces` | None | Cap number of traces. |
| `--max_errors_per_trace` | None | Cap errors per trace. |
| `--model` | `openai/gpt-4o` | Model for litellm. |
| `--context_window` | 1 | Neighbour spans for local context (0 = span only). |
