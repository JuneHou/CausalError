# Test Run Analysis

**Run**: 26 A-instances across 5 sampled traces (covering all 12 edges with min_backup=1)
**Results**: 25/26 patches passed postcheck, 20/25 reruns succeeded, 5 rerun_missing_suffix

---

## Summary Table

| # | Category | Error Type | Span | Root Cause | Severity |
|---|---|---|---|---|---|
| 1 | Postcheck bug | Formatting Errors | `b7d289f851318bab` | REQUIRED_MARKERS checked as one string, not split | Bug in postcheck |
| 2 | Rerun failure (×5) | Resource Abuse, Poor Information Retrieval | TOOL spans | `replace_span_input` on TOOL span has no message history | Design gap in case_builder |
| 3 | Annotation error | Formatting Errors | `bb4882dc2e1e1f30` | Annotator pointed to planning span instead of error span | Data quality (prior run) |

---

## Error 1 — Postcheck: REQUIRED_MARKERS treated as one string

**Trace**: `b1f9b9baefa4c69d` | **Error**: `Formatting Errors|12` | **Span**: `b7d289f851318bab`

**Symptom**: Patch failed postcheck after 3 attempts. FAIL message:
```
Required marker '### 1. Task outcome (short version):, ### 2. Task outcome
(extremely detailed version):, ### 3. Additional context (if relevant):'
not found in patch_payload.
```

**What the LLM generated** (patch_payload — correct content):
```
final_answer("### 1. Task outcome (short version):\n...\n\n### 2. Task outcome
(extremely detailed version):\n...\n\n### 3. Additional context (if relevant):\n...")
```

**Note on `local_snippet`**: `PatchResult` does not store `local_snippet` as a field. Reading it
from `postcheck_failures.jsonl` returns `''` by default — this was a misread. Confirmed via
`a_instances.jsonl`: `local_snippet` for this span is correctly populated with the `Thought:...Code:...`
content. `trail_io` handles CodeAgent-style spans (`content` string, `tool_calls: null`) correctly
via `_decode_value` → `data.get("content")`. No trail_io fix needed.

**Root cause**: The `REQUIRED_MARKERS` slot value is a comma-separated list of markers:
```
### 1. Task outcome (short version):, ### 2. Task outcome (extremely detailed version):, ### 3. Additional context (if relevant):
```
The postcheck treated this **entire string** (including commas) as a single required marker to find
as a literal substring in `patch_payload`. It is not there — each marker appears individually separated
by `\n\n`. Each individual marker IS present; only the combined string is not.

---

## Error 2 — Rerun Failure: `replace_span_input` on TOOL spans (×5)

**Trace**: `b1f9b9baefa4c69d` | **Error types**: Resource Abuse (×4), Poor Information Retrieval (×1)

**Spans affected** (all are `SearchInformationTool` TOOL spans):
```
d7f4a450e1b0e2c6  Resource Abuse|1             SearchInformationTool
d7f4a450e1b0e2c6  Poor Information Retrieval|2  SearchInformationTool (same span, 2 A-types)
9a9289c15d6fd9c9  Resource Abuse|6             SearchInformationTool
797e9814dfc62698  Resource Abuse|8             SearchInformationTool
b72be6cc8b945d04  Resource Abuse|10            SearchInformationTool
```

**Error from rerun_harness**:
```
rerun_error: No message history at span d7f4a450e1b0e2c6
```

**Root cause**: `patch_library.json` defines `Resource Abuse` and `Poor Information Retrieval` with
`patch_side_default: replace_span_input`. When the annotation points to a TOOL span (the actual search
call), `case_builder.py` uses that TOOL span as the intervention point. `rerun_harness.py` needs a
message history at the intervention span — TOOL spans have no message history (only raw `kwargs`).
The tool call arguments live in the output (`tool_calls`) of the **parent LLM span**, not in the
TOOL span. The intervention must happen there.

---

## Error 3 — Annotation Error: Planning span annotated as Formatting Error (prior run)

**Trace**: `01c5727165fc43899b3b594b9bef5f19` | **Error**: `Formatting Errors|4` | **Span**: `bb4882dc2e1e1f30`

**Symptom**: `patch_payload is identical to local_snippet — no change made.` after 3 attempts.

**Root cause**: Span `bb4882dc2e1e1f30` is a standalone planning LLM call outputting a `[FACTS LIST]`.
The annotation description claims "page_down called with unexpected keyword argument", but the actual
malformed page_down calls are at `2e4f1555d3662147` (Formatting Errors|2) and `dbf627066c19ae3c`
(Formatting Errors|3) — both already correctly annotated. Error index 4 is a duplicate annotation
pointing to the wrong span.

**This is an annotation error, not a pipeline bug.** The pipeline correctly surfaces it via postcheck
(the LLM found nothing to fix in a [FACTS LIST] and returned the same content).

---

## Rerun Status Summary

```
live_rerun_success  : 20  (Formatting Errors, Tool Selection Errors, Task Orchestration,
                           Incorrect Problem Identification all work correctly)
rerun_missing_suffix:  5  (all Resource Abuse / Poor Information Retrieval → TOOL spans)
postcheck_fail      :  1  (Formatting Errors|12 → REQUIRED_MARKERS postcheck bug)
```

The 5 `rerun_missing_suffix` cases are all from one trace (`b1f9b9ba`) and all share the same
structural pattern: `replace_span_input` patching a TOOL span with no message history.

---

## Fix Plan

### Fix 1 — `patch_generator.py`: Normalize REQUIRED_MARKERS before postcheck ✅ DONE

**Problem**: Postcheck compared the entire `REQUIRED_MARKERS` slot value as one substring. The slot
can return a plain string with comma-separated entries, a list of strings, or a mixed list.

**Implementation**: Added `_normalize_required_markers(value)` that handles all shapes:
```python
def _normalize_required_markers(value) -> List[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out
```
`_run_postcheck` calls this and checks each marker independently.

---

### Fix 2 — `case_builder.py`: Remap TOOL spans to parent LLM span for `replace_span_input` ✅ DONE

**Problem**: `Resource Abuse` and `Poor Information Retrieval` use `patch_side_default: replace_span_input`.
When the annotation points to a TOOL span, `rerun_harness.py` cannot find a message history.
Tool call arguments are authored by the parent LLM span's `tool_calls` output — that is the correct
intervention point.

**Implementation**:

*Two new helpers*:
- `_get_span_kind(trace_obj, span_id)` — reads `openinference.span.kind`
- `_find_sibling_llm_span(trace_obj, span_id)` — finds the LLM sibling within the same CHAIN step.
  The LLM span is NOT a parent of the TOOL span; it is a **sibling**. In smolagents, each CHAIN step
  contains one LLM child followed by one TOOL child:
  ```
  CHAIN (Step N)
    ├── LLM (LiteLLMModel.__call__)  ← generates tool_calls → correct intervention point
    └── TOOL (SearchInformationTool) ← annotated span
  ```
  The helper walks to the parent span and searches its `child_spans` for an LLM sibling.
  If not found at the immediate parent, it walks one level further up.
  Initial implementation (`_find_parent_llm_span`) incorrectly walked up the ancestor chain
  instead of searching siblings — fixed in test_run_2.

*Four new fields on `AInstanceRecord`* — keeps both locations for auditing; does not silently overwrite:
```
annotated_location:    str  # span_id from annotation (may be a TOOL span)
intervention_location: str  # span_id where patch is applied (always an LLM span)
annotated_span_kind:   str  # e.g. "TOOL", "LLM"
intervention_span_kind:str  # e.g. "LLM"
```
`a_instance["location"]` retains the original annotation location for manual audit.

*Remapping logic*: When `patch_side == replace_span_input` and `annotated_kind == TOOL`,
walk to parent LLM span, set `intervention_location`, flip `effective_patch_side = replace_span_output`.

*`patch_generator.py`* updated to use `case.get("intervention_location")` for `PatchResult.location`
so `rerun_harness.py` receives the correct LLM span ID.

---

### Fix 3 — Annotation correction: `bb4882dc2e1e1f30` ✅ DONE

**Problem**: Error index 4 in `01c5727165fc43899b3b594b9bef5f19.json` has `location: bb4882dc2e1e1f30`
(a planning [FACTS LIST] span). The description says "page_down with unexpected kwarg" — the actual
malformed call is at `dbf627066c19ae3c` (already annotated as Formatting Errors|3).

**Implementation**: Change `location` of error index 4 from `bb4882dc2e1e1f30` to `dbf627066c19ae3c`.
