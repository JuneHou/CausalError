# Test Run 2 Analysis

**Run**: 26 A-instances across 5 sampled traces (all 3 fixes from test_run_1 applied)
**Results**: 26/26 rerun success (Fix 2 confirmed working), 15/26 Judge-A RESOLVED, 11/26 UNRESOLVED

---

## Summary Table

| # | Category | Error Type | Span | Root Cause | Count |
|---|---|---|---|---|---|
| 1 | Hallucinated tool_call fields | Resource Abuse | TOOL→LLM remap | ERROR_TYPE_SPEC written for input-side, applied to tool_calls output | 3 |
| 2 | Shared location conflict | Poor Information Retrieval | TOOL→LLM remap | Shares intervention_location with Resource Abuse; patch also ineffective | 1 |
| 3 | Judge-A confused by rerun suffix | Formatting Errors | LLM | RERUN_SUFFIX text bleeds into patch quality judgment | 1 |
| 4 | Input-side patch vs output behavior | Tool Selection Errors, Task Orchestration | LLM | Judge-A can't confirm error removal from input comparison alone | 2 |
| 5 | Nearly-unchanged patch on coding trace | Formatting Errors | LLM (CodeAgent) | Patch spec produces minimal changes; errors may be misclassified | 2 |
| 6 | Input-side patch on LLM span | Resource Abuse | LLM | Editing historical tool record in message history doesn't prevent repeat | 1 |
| 7 | Incorrect Problem Identification on coding trace | Incorrect Problem Identification | LLM | Patch barely changed reasoning; error description matches IPI but logic unchanged | 1 |

---

## Error Group 1 — Resource Abuse: Hallucinated `stop_criterion` in tool_calls JSON (×3)

**Traces/spans**:
```
b1f9b9ba  Resource Abuse|1   intervention_location=2ed1fc6345114c29 (remapped from TOOL d7f4a450)
b1f9b9ba  Resource Abuse|6   intervention_location=1a782ec769de40f1 (remapped from TOOL 9a9289c1)
b1f9b9ba  Resource Abuse|10  intervention_location=4ea5cfbbfa357d36 (remapped from TOOL b72be6cc)
```

**Symptom**: All three patches added a hallucinated field to the tool_calls JSON object:
```json
{..., "stop_criterion": "Do not repeat identical tool calls if the result is already obtained."}
```
or
```json
{..., "stop_condition": "..."}
```

**What went wrong**: After the TOOL→LLM sibling remap (Fix 2), `effective_patch_side` becomes
`replace_span_output` and `local_snippet` is the LLM's tool_calls JSON. But the `error_type_spec_text`
for `Resource Abuse` in `patch_library.json` was written for `patch_side_default: replace_span_input`
— it instructs the LLM to "rewrite the local step to avoid repeating already-completed actions".
Receiving a tool_calls JSON as `LOCAL_SNIPPET` with input-side instructions, the LLM adds a
`stop_criterion` field that does not exist in the OpenAI tool_calls schema.

**Postcheck gap**: The `_run_postcheck` function has no rule for Resource Abuse. No validation
catches the hallucinated key in tool_calls JSON.

**Judge verdict**: Correct UNRESOLVED — the underlying search query (with `filter_year`) is
unchanged; only a phantom field was added.

---

## Error Group 2 — Poor Information Retrieval: Shared intervention_location conflict (×1)

**Span**: `b1f9b9ba` Poor Information Retrieval|2 at `2ed1fc6345114c29` (same LLM sibling as
Resource Abuse|1).

**Symptom**: `filter_year: "2022"` remains in patch_payload; Judge correctly marks UNRESOLVED.

**Root cause**: Two separate A-instances (Resource Abuse|1 and Poor Information Retrieval|2)
are both annotated on the same TOOL span `d7f4a450e1b0e2c6`, and both remap to the same LLM
sibling `2ed1fc6345114c29`. Two separate patches are generated for the same span. The rerun
applies one patch per `location` key; the behavior of the second patch (which wins?) is not
well-defined. The Poor Information Retrieval patch still retained `filter_year`.

**Note**: When multiple A-types share the same intervention_location, each gets its own
`AInstanceRecord` and its own separate patch. Since both patches target the same span, one
likely overwrites the other in the rerun. This needs explicit handling.

---

## Error Group 3 — Judge-A Confused by Rerun Suffix (×1)

**Span**: `b1f9b9ba` Formatting Errors|5 at `1a782ec769de40f1` (LLM span)

**Symptom**: Judge-A evidence_excerpt reads:
```
"I'm having trouble locating an archived version of the official Flavor Graveyard page
for December 31, 2022. It appears that https://www.benjerry.com/flavors/flavor-graveyard
wasn't directly archived on that date."
```
This text is from the **rerun suffix** (the agent's counterfactual output), not from the
patch-vs-original comparison. The patch DID make a change (date `20221230` → `20221231`).

**Root cause**: The JUDGE_A prompt provides RERUN_SUFFIX alongside ORIGINAL_SPAN and PATCHED_SPAN.
When the rerun still shows difficulties, the judge uses that evidence to mark UNRESOLVED —
even though the patch correctly fixed the Formatting Error in the span itself.

**Impact**: At least 1 false UNRESOLVED. May affect more cases if rerun encounters task failures.

---

## Error Group 4 — Input-Side Patches: Judge-A Comparison Mismatch (×2)

**Spans**:
```
2713fa0a  Tool Selection Errors  intervention_location=f84f4bfbb36d8ee9  patch_side=replace_span_input
a99faf78  Task Orchestration     intervention_location=63610e7455b22b1e  patch_side=replace_span_input
```

**Symptom**: Judge-A says patches still don't show the correct behavior. For example:
```
"The patched span still does not show the use of a search tool to verify information."
```

**Root cause**: For `replace_span_input`, `local_snippet` is the **message history input** to
the LLM (previous messages, tool call/response records). The patch modifies this context.
But the error was observed in the LLM's **output** behavior (which tool it selected, what it
decided). The Judge-A prompt shows ORIGINAL_SPAN (input) vs PATCHED_SPAN (input) and asks
"is error A eliminated?" — but the error is in the output, which neither span shows.

This is a fundamental mismatch for input-side patches:
- `replace_span_output`: compare original output vs patched output → can directly check
- `replace_span_input`: compare original input vs patched input → can only verify the
  error-triggering context is removed, NOT that the LLM would now behave correctly

**What Judge-A should check for input-side patches**: Whether the patched input removes the
specific context that caused the error (e.g., the wrong tool call was removed from history,
or a correct guidance was added), not whether the patched text contains the correct behavior.

---

## Error Group 5 — Formatting Errors on CodeAgent Traces (×2)

**Spans**:
```
f04b425c  Formatting Errors|0  intervention_location=983bb8ef7929be6d  replace_span_output
f04b425c  Formatting Errors|4  intervention_location=988b9b4c3f058911  replace_span_output
```

**Symptom**: Both patches are nearly identical to the originals (no meaningful change). Judge
correctly marks UNRESOLVED: "patched span still includes same task request to retrieve file
content."

**Root cause**: These are CodeAgent "Thought: ... Code: ..." spans. The errors are annotated as
`Formatting Errors` but the descriptions say things like "file could not be loaded using
inspect_file_as_text, so I will ask our search_agent team member". This is not a formatting
issue — it is poor tool choice. The Formatting Errors ERROR_TYPE_SPEC targets markers like
`<end_plan>` tags or tool call argument syntax; it cannot meaningfully patch a planning
paragraph about which tool to use.

**Postcheck gap**: `patch_payload is identical to local_snippet` rule should catch these, but
the patches had minor wording tweaks (e.g., "retrieve" vs "retrieve the complete text") that
passed the identity check while leaving the core error unchanged.

**Suspected annotation issue**: These may be misclassified errors — they fit `Incorrect
Problem Identification` better than `Formatting Errors`.

---

## Error Group 6 — Resource Abuse: Input-side patch doesn't break repetition cycle (×1)

**Span**: `c60ad860` Resource Abuse|2 at `28be2379ec209772` (`patch_side=replace_span_input`, LLM span)

**Symptom**: Judge-A evidence_excerpt shows:
```
Error when executing tool page_down with arguments {}: TypeError: PageDownTool.forward()
got an unexpected keyword argument 'arguments'
```
The rerun suffix shows the same tool error persisting.

**Root cause**: The patch correctly fixed the historical tool call record in the message
history (`{'arguments': {'arguments': {}}}` → `{}`). But this modifies the **context** seen
by the next LLM call, not the decision-making that generated the wrong call. In the
counterfactual rerun, the LLM still generates `{'arguments': {}}` for a later `page_down`
call, which triggers `TypeError: unexpected keyword argument 'arguments'`.

**Note**: Fixing the history doesn't prevent the model from repeating the mistake in a
future step. Input-side patches for Resource Abuse have limited effectiveness when the
model systematically misuses a tool.

---

## Error Group 7 — Incorrect Problem Identification on coding trace (×1)

**Span**: `f04b425c` Incorrect Problem Identification|1 at `983bb8ef7929be6d` (`replace_span_input`)

**Symptom**: Judge-A says "patched span still relies on search_agent to retrieve file content
after initial failure with inspect_file_as_text, indicating same fundamental problem."

**Root cause**: The input shows the message history where the agent has already decided to ask
search_agent. The patch modified the input slightly ("I will first attempt to read..." vs "I
will first read..."). The underlying decision to delegate to search_agent is unchanged.

---

## Tool-Calling vs Coding Trace Distinction

The 11 UNRESOLVED cases cluster differently by trace type:

**Tool-calling traces** (`b1f9b9ba`, `c60ad860`):
- Patches target tool_calls JSON (replace_span_output after remap)
- LLM generates hallucinated fields (`stop_criterion`)
- Judge-A confusion from rerun context
- Issue: ERROR_TYPE_SPEC for input-side errors doesn't match tool_calls output format

**Coding traces** (`f04b425c`):
- Patches target CodeAgent "Thought:...Code:..." text (replace_span_output)
- Patch barely changes wording; underlying decision unchanged
- Issue: Formatting Errors spec not suited for planning-text patches in CodeAgent format

**Mixed input-side** (`2713fa0a`, `a99faf78`):
- Patches modify message history context
- Judge-A incorrectly checks input for error absence when error is in output
- Issue: Judge-A prompt does not distinguish patch_side

---

## Fix Plan

### Fix A — Resource Abuse Postcheck: Validate tool_calls schema (gated on remapped case) ✅ DONE

**Problem**: Patches add hallucinated keys (`stop_criterion`, `stop_condition`) to tool_calls JSON.

**Gate condition**: Only apply this rule when the case is a remapped TOOL→LLM intervention —
i.e., `annotated_span_kind == "TOOL"` and `patch_side == "replace_span_output"`. This is
indicated by `snippet_mode == "tool_call_json"` passed from `AInstanceRecord`. Without the
gate, the rule would fire on non-tool-call Resource Abuse patches (e.g., pure text rewrites)
and generate false positives.

**Implementation**: Pass `snippet_mode` to `_run_postcheck`. Set it to `"tool_call_json"` in
`generate_patch` when `case["annotated_span_kind"] == "TOOL"` and
`case["patch_side"] == "replace_span_output"`:

```python
elif a_cat == "Resource Abuse":
    if snippet_mode == "tool_call_json":
        # Patch targets tool_calls JSON output of a remapped LLM span.
        # Validate no novel keys in tool_call entries (allowed: id, type, function).
        try:
            parsed = json.loads(patch_payload)
            tool_calls = parsed.get("tool_calls", []) if isinstance(parsed, dict) else parsed
            if isinstance(tool_calls, list):
                allowed_keys = {"id", "type", "function"}
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        novel = set(tc.keys()) - allowed_keys
                        if novel:
                            failures.append(f"Novel keys in tool_call entry: {novel}")
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON tool_calls — skip schema check
```

### Fix B — Resource Abuse ERROR_TYPE_SPEC: Add output-side variant ✅ DONE

**Problem**: Spec written for `replace_span_input` but after TOOL→LLM remap it is used with
`replace_span_output`. The LLM doesn't know to modify query arguments, not add phantom fields.

**Implementation**: Add a note to the Resource Abuse ERROR_TYPE_SPEC explaining the output-side case:
> When LOCAL_SNIPPET is a tool_calls JSON object, fix the repetition by modifying the tool
> call arguments (e.g., change the query to be more specific, remove redundant parameters
> like filter_year) or replacing the tool call with a more targeted one. Do NOT add new
> fields (stop_criterion, stop_condition, etc.) to the tool_calls JSON — these are not valid
> OpenAI tool call schema fields.

### Fix C — Judge-A: Differentiate replace_span_output vs replace_span_input evaluation ✅ DONE

**Problem**: For `replace_span_input`, Judge-A checks the input for error absence when the
error is in the output. For `replace_span_output`, the current check is appropriate.

**Implementation**: Two separate user templates selected by `patch_side` read from
`a_instance_record["patch_side"]`:

- `JUDGE_A_USER_OUTPUT_TEMPLATE` (replace_span_output): Compare ORIGINAL_SPAN vs PATCHED_SPAN,
  check if the error is removed from the content.
- `JUDGE_A_USER_INPUT_TEMPLATE` (replace_span_input): The patched content is the MESSAGE HISTORY
  INPUT to the LLM. Check whether the error-causing context is removed or corrected in the
  patched input. Explicit NOTE: do NOT require the patched input to contain the correct output
  behavior — it is an input, not an output.

### Fix D — Judge-A: Reduce RERUN_SUFFIX influence ✅ DONE

**Problem**: Judge-A was using rerun failure evidence to mark patches UNRESOLVED even when
the patch correctly addressed the labeled error (e.g., rerun shows `filter_year` in OTHER spans,
or rerun shows a tool error in a subsequent span, not the patched span).

**Implementation**:
- JUDGE_A_SYSTEM updated with explicit IMPORTANT RULES:
  - Primary verdict from ORIGINAL_SPAN vs PATCHED_SPAN comparison.
  - RERUN_SUFFIX is supplementary context only.
  - "If the patched span itself resolves error A, mark resolved=true even if the rerun
    encountered downstream difficulties or repeated errors in OTHER spans."
- RERUN_SUFFIX in user message labeled as "supplementary context — first 3 spans after t_A
  in counterfactual run" to make clear it is not the primary evidence.

### Fix E — Shared intervention_location: Keep one, skip the rest with conflict record ✅ DONE

**Problem**: Multiple A-types with the same `intervention_location` generate separate patches
for the same span. The rerun applies one patch per `location` key — it is undefined which
wins and which is silently discarded.

**Why not merge**: Merging descriptions changes the intervention estimand — you move from
"remove one A-instance" to "jointly repair multiple A-types". The patch becomes harder to
interpret causally, and Judge-A / Judge-B assumptions no longer cleanly match a single
source error.

**Implementation**: In `case_builder.py`, after all A-instances are built, apply a
post-dedup pass keyed on `intervention_location`. For each group sharing the same location:
1. Keep the A-instance with the lowest `annotation_index` as the **active** case.
2. Skip all others — write them to `intervention_location_conflicts.jsonl` in `out_dir`
   with a `conflict_reason: "shared_intervention_location"` field and the `kept_error_id`
   of the active case.

Priority rule: lowest `annotation_index` wins (first annotation in trace order).
Rationale: deterministic, reproducible, requires no semantic judgment.

The conflict file provides a full audit trail. Cases in the conflict file are excluded from
patch generation, rerun, and Judge-A/B.

```python
# After build_cases returns a_instances:
seen_locations: Dict[str, str] = {}   # intervention_location → kept error_id
active: List[AInstanceRecord] = []
conflicts: List[dict] = []

for rec in a_instances:
    loc = rec.intervention_location
    if loc not in seen_locations:
        seen_locations[loc] = rec.error_id
        active.append(rec)
    else:
        conflicts.append({
            **asdict(rec),
            "conflict_reason": "shared_intervention_location",
            "kept_error_id": seen_locations[loc],
        })

# Write conflicts to out_dir/intervention_location_conflicts.jsonl
# Proceed with active list only
```

---

## Rerun/Judge Status Summary

```
rerun_success               : 26  (all cases — Fix 2 confirmed working)
judge_a_resolved            : 15
judge_a_unresolved          : 11
  - hallucinated tool_call keys (Resource Abuse, remapped): 3
  - shared location / same issue             (PIR):        1
  - Judge-A rerun confusion                  (FE):         1
  - input-side mismatch  (TSE, TaskOrch):                  2
  - near-unchanged patch on coding trace (FE):             2
  - input-patch doesn't break cycle  (RA):                 1
  - IPI near-unchanged on coding trace:                    1
```

The 15 RESOLVED cases demonstrate the pipeline works correctly for:
- Formatting Errors (output-side, tool-calling traces)
- Resource Abuse (output-side, when patch correctly changes query)
- Incorrect Problem Identification (input-side, when patch meaningfully changes reasoning)
- Task Orchestration (input-side, when patch provides clear plan guidance)
