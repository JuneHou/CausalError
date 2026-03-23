#!/usr/bin/env python3
"""
Steps 2-4: Patch generation using the shared scaffold + ERROR_TYPE_SPEC from patch_library.

Flow per AInstanceRecord:
  1. Look up ERROR_TYPE_SPEC from patch_library.json (keyed by A category).
  2. Build prompt: shared scaffold SYSTEM + USER template with ERROR_TYPE_SPEC substituted.
  3. Call LLM → parse JSON response (slot_values, patch_payload, postcheck).
  4. Run external rule-based postcheck. Retry up to max_retries if postcheck fails.

Input:  a_instances.jsonl  (one record per unique A-instance)
Output: patch_results.jsonl, postcheck_failures.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from patch_generator_llm import _call_llm  # reuse existing litellm wrapper


# ---------------------------------------------------------------------------
# Shared scaffold (SYSTEM prompt — identical for all A types)
# ---------------------------------------------------------------------------

PATCH_SYSTEM = """You are generating a localized intervention corresponding to do(A=0) for one annotated source error instance.

Goal:
Remove ONLY the annotated source error instance of type A by minimally replacing the exact labeled span.
This is a causal intervention for testing whether fixing A changes downstream error B.
You must not directly repair B.

Hard constraints:
1. Modify ONLY the provided LOCAL_SNIPPET.
2. Target ONLY the annotated source error type A.
3. Do NOT directly fix, mention, or optimize for the downstream error type B.
4. Preserve as much original meaning and content as possible.
5. Do NOT invent new tags, markers, tool outputs, resources, credentials, or facts unless they appear verbatim in ERROR_DESCRIPTION, ERROR_EVIDENCE, USER_REQUIREMENTS, or LOCAL_SNIPPET.
6. If the intervention is a formatting fix, make the smallest possible structural edit only.
7. If the intervention is a planning/reasoning fix, rewrite only the local decision or local text needed to eliminate A.
8. Return ONLY the JSON object in the required schema.

Required output schema:
{
  "source_error_type": "string",
  "downstream_error_type": "string",
  "patch_side": "replace_span_output|replace_span_input",
  "slot_values": { "key": "value or null" },
  "patch_payload": "string",
  "postcheck": {
    "passed": true,
    "checks": ["string"],
    "notes": "string"
  }
}"""


# ---------------------------------------------------------------------------
# USER template
# ---------------------------------------------------------------------------

PATCH_USER_TEMPLATE = """\
ERROR_TYPE_SPEC:
<<<
{ERROR_TYPE_SPEC}
>>>

SOURCE_ERROR_TYPE: {A}
DOWNSTREAM_ERROR_TYPES (do NOT directly fix any of these): {B_LIST}
ERROR_DESCRIPTION:
{ERROR_DESCRIPTION}
ERROR_EVIDENCE:
{ERROR_EVIDENCE}

LOCAL_SNIPPET (exact span text to replace):
<<<
{LOCAL_SNIPPET}
>>>

USER_REQUIREMENTS:
<<<
{USER_REQUIREMENTS}
>>>

Task:
Generate a minimal patch that implements do(A=0): remove the source error instance only.\
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    trace_id: str
    error_id: str
    location: str
    patch_side: str
    template_used: str       # A category key from patch_library
    slot_values: dict
    patch_payload: str
    postcheck_passed: bool
    postcheck_failures: list
    attempts: int
    patch_reason: str        # from LLM's postcheck.notes or summary
    llm_postcheck: dict      # raw postcheck block from LLM response


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_required_markers(value) -> List[str]:
    """
    Normalize the REQUIRED_MARKERS slot value into a flat list of individual marker strings.

    Handles:
    - None / empty → []
    - A plain string that may contain multiple markers separated by ", "
    - A list of strings (each element may itself contain commas)
    - Any mix of the above with inconsistent spacing or punctuation
    """
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


# External (rule-based) postcheck
# ---------------------------------------------------------------------------

def _run_postcheck(
    patch_payload: str,
    local_snippet: str,
    slot_values: dict,
    lib_entry: dict,
    snippet_mode: str = "",
) -> tuple[bool, list]:
    """
    Run rule-based postcheck on patch_payload.
    Returns (passed, list_of_failure_messages).

    snippet_mode: optional hint about the LOCAL_SNIPPET format.
      "tool_call_json" — LOCAL_SNIPPET is a tool_calls JSON object from a remapped
                         TOOL→LLM span (annotated_span_kind="TOOL", patch_side="replace_span_output").
                         Enables schema validation for Resource Abuse patches.
    """
    failures = []
    a_cat = lib_entry.get("category", "")

    # Universal: patch must differ from original
    if patch_payload.strip() == local_snippet.strip():
        failures.append("patch_payload is identical to local_snippet — no change made.")

    # Universal: patch must be non-empty
    if not patch_payload.strip():
        failures.append("patch_payload is empty.")

    # Category-specific checks
    if a_cat == "Formatting Errors":
        required_markers = _normalize_required_markers(slot_values.get("REQUIRED_MARKERS"))
        for marker in required_markers:
            if marker not in patch_payload:
                failures.append(f"Required marker '{marker}' not found in patch_payload.")
        # No novel <...> tokens (tokens not in original snippet or required_markers)
        original_tokens = set(re.findall(r"<[^>]+>", local_snippet))
        required_token_set = set(required_markers)
        patch_tokens = set(re.findall(r"<[^>]+>", patch_payload))
        novel = patch_tokens - original_tokens - required_token_set
        if novel:
            failures.append(f"Novel ungrounded tokens introduced: {novel}")

    elif a_cat == "Tool Selection Errors":
        wrong_tool = (slot_values.get("WRONG_TOOL") or "").strip()
        if wrong_tool and wrong_tool in patch_payload:
            failures.append(f"Wrong tool '{wrong_tool}' still present in patch_payload.")
        correct_tool = (slot_values.get("CORRECT_TOOL_HINT") or "").strip()
        if correct_tool and correct_tool not in patch_payload:
            failures.append(
                f"Correct tool '{correct_tool}' not found in patch_payload. "
                f"The patched input must explicitly direct the agent to use '{correct_tool}'."
            )

    elif a_cat == "Resource Abuse" and snippet_mode == "tool_call_json":
        # Patch targets tool_calls JSON from a remapped TOOL→LLM span.
        # No novel keys are allowed in tool_call entries beyond the OpenAI schema fields.
        allowed_tc_keys = {"id", "type", "function"}
        try:
            parsed = json.loads(patch_payload)
            tool_calls = parsed.get("tool_calls", []) if isinstance(parsed, dict) else parsed
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        novel = set(tc.keys()) - allowed_tc_keys
                        if novel:
                            failures.append(
                                f"Novel keys in tool_call entry (not in OpenAI schema): {sorted(novel)}. "
                                f"Fix the tool call arguments instead of adding phantom fields."
                            )
        except (json.JSONDecodeError, TypeError):
            pass  # Not parseable as JSON tool_calls — skip schema check

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_patch(
    case: dict,
    patch_library: dict,
    model: str = "openai/gpt-4o",
    max_retries: int = 3,
) -> PatchResult:
    """
    Generate a patch for one AInstanceRecord dict. Returns PatchResult.
    """
    a_cat = case["a_instance"]["category"]
    b_list = ", ".join(case.get("b_types") or []) or "unknown"
    lib_entry = patch_library.get(a_cat, {})
    error_type_spec = lib_entry.get("error_type_spec_text", f"error_type: {a_cat}")

    # Determine snippet_mode for postcheck gating.
    # "tool_call_json": this is a remapped TOOL→LLM case where local_snippet is the LLM's
    # tool_calls JSON output. Enables schema validation in Resource Abuse postcheck.
    snippet_mode = ""
    if (case.get("annotated_span_kind") == "TOOL"
            and case.get("patch_side") == "replace_span_output"):
        snippet_mode = "tool_call_json"

    user_msg = PATCH_USER_TEMPLATE.format(
        ERROR_TYPE_SPEC=error_type_spec,
        A=a_cat,
        B_LIST=b_list,
        ERROR_DESCRIPTION=(case["a_instance"].get("description") or "")[:1200],
        ERROR_EVIDENCE=(case["a_instance"].get("evidence") or "")[:1200],
        LOCAL_SNIPPET=(case["local_snippet"] or "")[:4000],
        USER_REQUIREMENTS=(case.get("user_requirements") or "")[:1500],
    )

    patch_payload = ""
    slot_values: dict = {}
    postcheck_passed = False
    postcheck_failures: list = []
    patch_reason = ""
    llm_postcheck: dict = {}
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        retry_note = ""
        if attempt > 1 and postcheck_failures:
            retry_note = (
                f"\n\nPREVIOUS ATTEMPT FAILED POSTCHECK:\n"
                + "\n".join(f"- {f}" for f in postcheck_failures)
                + "\nPlease fix these issues in your new response."
            )

        try:
            raw = _call_llm(
                PATCH_SYSTEM,
                user_msg + retry_note,
                model=model,
                max_tokens=2048,
            )
        except Exception as e:
            postcheck_failures = [f"LLM call failed: {e}"]
            continue

        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON substring
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    postcheck_failures = ["LLM response is not valid JSON."]
                    continue
            else:
                postcheck_failures = ["LLM response contains no JSON object."]
                continue

        if not isinstance(parsed, dict):
            postcheck_failures = ["LLM JSON is not a dict."]
            continue

        patch_payload = (parsed.get("patch_payload") or "").strip()
        slot_values = parsed.get("slot_values") or {}
        llm_postcheck = parsed.get("postcheck") or {}
        patch_reason = llm_postcheck.get("notes") or ""

        # External rule-based postcheck
        postcheck_passed, postcheck_failures = _run_postcheck(
            patch_payload, case["local_snippet"], slot_values, lib_entry,
            snippet_mode=snippet_mode,
        )
        if postcheck_passed:
            break

    return PatchResult(
        trace_id=case["trace_id"],
        error_id=case["a_instance"].get("error_id", ""),
        location=case.get("intervention_location") or case["a_instance"].get("location", ""),
        patch_side=case["patch_side"],
        template_used=a_cat,
        slot_values=slot_values,
        patch_payload=patch_payload,
        postcheck_passed=postcheck_passed,
        postcheck_failures=postcheck_failures,
        attempts=attempts,
        patch_reason=patch_reason,
        llm_postcheck=llm_postcheck,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate do(A=0) patches for all AInstanceRecords."
    )
    parser.add_argument("--cases", default="outputs/interventions/a_instances.jsonl")
    parser.add_argument("--patch_library", default="causal/patch/patch_library.json")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    with open(args.patch_library, "r", encoding="utf-8") as f:
        patch_library = json.load(f)

    cases = []
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "patch_results.jsonl")
    failures_path = os.path.join(args.out_dir, "postcheck_failures.jsonl")

    n_ok = n_fail = 0
    with open(results_path, "w", encoding="utf-8") as rf, \
         open(failures_path, "w", encoding="utf-8") as ff:
        for case in cases:
            result = generate_patch(case, patch_library, model=args.model,
                                    max_retries=args.max_retries)
            rec = asdict(result)
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if result.postcheck_passed:
                status = "OK"
            else:
                err_detail = result.postcheck_failures[0][:120] if result.postcheck_failures else "unknown"
                status = f"FAIL: {err_detail}"
            print(f"  [{status}] {result.trace_id[:8]} {result.error_id[-20:]} attempts={result.attempts}")
            if result.postcheck_passed:
                n_ok += 1
            else:
                ff.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_fail += 1

    print(f"\nWrote {results_path}. OK={n_ok} FAIL={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
