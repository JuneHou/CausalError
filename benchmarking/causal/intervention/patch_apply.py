#!/usr/bin/env python3
"""
Patch library loader + application engine (A1, A3, A5, A6, D).

Responsibilities:
  - load_patch_specs(dir) → {family: PatchSpec}
  - instantiate_spec(spec, error, snippet, trace_obj) → concrete dict with patched_text
  - apply_patch(trace_obj, error, spec, window) → PatchRecord
  - validate_patch_record(original, patched, ...) → (ok, reasons)

Per-family instantiation strategies (rule-based, no LLM needed):

  BUDGET_GUARD_STOP_CONDITION  – detect repeated tool calls; prepend guard block
  TOOL_SCHEMA_REPAIR           – extract bad arg key from error evidence; remove it
  EXECUTE_INSTEAD_OF_DESCRIBE  – find intent phrase; replace with action marker
  TOOL_SELECTION_SWAP          – find wrong tool call; swap to better tool
  CONTEXT_STATE_CARRYOVER      – append state-capture annotation at end
  GOAL_CONSTRAINT_CHECK        – insert checkpoint before final_answer (or at top)
  OUTPUT_INTERPRETATION_VERIFY – insert verify block after Observation:
  RETRIEVAL_REQUERY            – find query string; append specificity terms
"""
from __future__ import annotations

import os
import sys
_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trail_io import TraceObj, get_expanded_snippet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PatchSpec:
    """Normalised representation of one operator-family JSON file."""
    operator_family: str
    error_type: str        # "{ERROR_TYPE}" means applicable to multiple types
    patch_type: str
    constraints: dict
    raw: dict              # full original JSON for traceability


@dataclass
class PatchRecord:
    """Result of one apply_patch call."""
    trace_id: str
    error_id: str
    operator_family: str
    location: str           # annotated span_id
    original_text: str
    patched_text: str
    diff_lines: int
    instantiated_spec: dict
    validation: dict        # {ok: bool, reasons: [str]}
    success: bool


# ---------------------------------------------------------------------------
# Spec loader (A1)
# ---------------------------------------------------------------------------


def load_patch_specs(patch_dir: str) -> Dict[str, PatchSpec]:
    """
    Read every *.json file in patch_dir into a PatchSpec keyed by operator_family.
    Normalises: operator_family, error_type, patch_type, constraints_checked.
    """
    specs: Dict[str, PatchSpec] = {}
    for fname in sorted(os.listdir(patch_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(patch_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        family = data.get("operator_family") or os.path.splitext(fname)[0]
        specs[family] = PatchSpec(
            operator_family=family,
            error_type=data.get("error_type", "{ERROR_TYPE}"),
            patch_type=data.get("patch_type", ""),
            constraints=data.get("constraints_checked", {}),
            raw=data,
        )
    return specs


# ---------------------------------------------------------------------------
# Shared regex patterns
# ---------------------------------------------------------------------------

# Tool call JSON: {"name": "X", "arguments": {...}}
_TOOL_CALL_JSON_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"(?P<tool>[^"]+)"[^{}]*"arguments"\s*:\s*(?P<args_open>\{)',
    re.DOTALL,
)
# Python-literal tool call: {'name': 'X', 'arguments': {...}}
_TOOL_CALL_PY_RE = re.compile(
    r"\{[^{}]*'name'\s*:\s*'(?P<tool>[^']+)'[^{}]*'arguments'\s*:\s*",
)
# Intent phrase ("I will call ...", "Let's use ...", etc.)
_INTENT_RE = re.compile(
    r"(I\s+will|I'll|Now\s+I\s+will|Let's|I\s+plan\s+to)\s+"
    r"(call|use|invoke|ask|run|search|visit|find)\b[^\n.!?]{0,120}[.!?]?",
    re.IGNORECASE,
)
# TypeError bad keyword argument
_BAD_KWARG_RE = re.compile(
    r"TypeError:\s*\S+\.forward\(\)\s+got\s+an\s+unexpected\s+keyword\s+argument\s+'(?P<key>[^']+)'",
    re.IGNORECASE,
)
# Error-when-executing message
_EXEC_ERR_RE = re.compile(
    r"Error\s+when\s+executing\s+tool\s+(?P<tool>\w+)\s+with\s+arguments\s+"
    r"(?P<args>\{[^}]*\})",
    re.IGNORECASE,
)
# Query key patterns
_QUERY_RE = re.compile(
    r'(?:"query"\s*:\s*"(?P<q1>[^"]{3,})"'
    r'|"search_string"\s*:\s*"(?P<q2>[^"]{3,})"'
    r"|'query'\s*:\s*'(?P<q3>[^']{3,})'"
    r"|'search_string'\s*:\s*'(?P<q4>[^']{3,})')"
)
# final_answer invocation
_FINAL_ANSWER_RE = re.compile(r"\bfinal_answer\b", re.IGNORECASE)
# Observation block
_OBSERVATION_RE = re.compile(r"(Observation\s*:\s*\n?)", re.IGNORECASE)
# Guard keywords (for validation)
_GUARD_KEYWORDS_RE = re.compile(
    r"\b(max_retries|stop_condition|retry|GUARD|MAX[_ ]RETRY|RETRY_COUNTER|BUDGET)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-family instantiators (D — concrete spec instantiation)
# ---------------------------------------------------------------------------


def _find_tool_name_in_text(text: str, tools_available: List[str]) -> Optional[str]:
    """Return the first tool name from tools_available that appears in text."""
    for t in (tools_available or []):
        if t and re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE):
            return t
    return None


def _count_tool_calls(text: str) -> Dict[str, int]:
    """Count occurrences of each tool name across JSON and Python-literal calls."""
    counts: Dict[str, int] = {}
    for m in re.finditer(r"""['"](name)['"]\s*:\s*['"]([^'"]+)['"]""", text):
        t = m.group(2)
        counts[t] = counts.get(t, 0) + 1
    return counts


def _instantiate_budget_guard(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    BUDGET_GUARD_STOP_CONDITION: detect repeated tool calls; prepend a guard block.
    Concrete patched_text = guard directive + original snippet.
    """
    counts = _count_tool_calls(snippet)
    repeated = {t: c for t, c in counts.items() if c >= 2}
    target_tool = next(iter(repeated), "the same action")
    max_retries = 3

    guard = (
        f"[GUARD: Retry limit = {max_retries} for '{target_tool}'. "
        f"After {max_retries} attempts with no new information, "
        f"stop and switch to an alternative strategy.]\n"
    )
    return {
        "patched_text": guard + snippet,
        "guard_policy": {
            "guard_type": "MAX_RETRY_COUNTER",
            "max_retries": max_retries,
            "target_tool": target_tool,
            "stop_condition": "no new information or same failure repeats",
            "fallback_action": "stop retrying and switch to alternative step",
        },
    }


def _instantiate_tool_schema_repair(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    TOOL_SCHEMA_REPAIR: extract bad arg key from evidence; remove it from snippet.
    """
    evidence = error.get("evidence") or ""
    description = error.get("description") or ""

    bad_key: Optional[str] = None
    for text in (evidence, description, snippet):
        m = _BAD_KWARG_RE.search(text)
        if m:
            bad_key = m.group("key")
            break
        m2 = _EXEC_ERR_RE.search(text)
        if m2:
            # Try to extract from the args dict shown in the error
            args_text = m2.group("args")
            km = re.search(r"'([^']+)'\s*:", args_text)
            if km:
                bad_key = km.group(1)
                break

    if bad_key:
        # Remove "bad_key": <value> in both JSON and Python-literal forms
        bad_json_re = re.compile(
            r"""[,\s]*['"]""" + re.escape(bad_key) + r"""['"]\s*:\s*(?:'[^']*'|"[^"]*"|\{\}|\[\]|\d+|\w+)""",
        )
        patched = bad_json_re.sub("", snippet)
        # Clean up resulting empty dicts
        patched = re.sub(r"\{\s*\}", "{}", patched)
        return {
            "patched_text": patched,
            "original_call": {"bad_key": bad_key},
            "repaired_call": {"dropped_keys": [bad_key]},
            "repair_note": f"Removed invalid keyword argument '{bad_key}' from tool call.",
        }

    # Fallback: annotate for manual repair
    note = "\n[SCHEMA REPAIR: Remove the invalid keyword argument from the tool call above.]"
    return {
        "patched_text": snippet + note,
        "repair_note": "Could not auto-identify bad key; manual repair needed.",
    }


def _instantiate_execute_instead_of_describe(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    EXECUTE_INSTEAD_OF_DESCRIBE: replace 'I will call X' intent phrase with
    an explicit action marker referencing the intended tool.
    """
    m = _INTENT_RE.search(snippet)
    if m:
        tool = _find_tool_name_in_text(m.group(0), tools_available) or "<TOOL>"
        replacement = f"[ACTION: Invoke `{tool}` with appropriate arguments.]"
        patched = snippet[: m.start()] + replacement + snippet[m.end() :]
        return {
            "patched_text": patched,
            "tool_invocation": {"tool": tool, "args": {"task": "<TASK_TEXT>"}},
            "replace_note": f"Replaced descriptive intent phrase with action call for '{tool}'.",
        }

    note = "\n[EXECUTE: Replace the descriptive statement above with an actual tool invocation.]"
    return {
        "patched_text": snippet + note,
        "replace_note": "Intent phrase not matched; appended directive.",
    }


def _instantiate_tool_selection_swap(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    TOOL_SELECTION_SWAP: identify the wrong tool call in snippet; swap to a
    better-matched tool from tools_available based on error description.
    """
    description = error.get("description") or ""
    evidence = error.get("evidence") or ""

    # Identify current (wrong) tool
    current_tool: Optional[str] = None
    for m in re.finditer(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", snippet):
        current_tool = m.group(1)
        break

    # Suggest better tool: prefer one mentioned in description/evidence
    better_tool = _find_tool_name_in_text(description + " " + evidence, tools_available)
    if not better_tool or better_tool == current_tool:
        # Fall back to a sensible default retrieval tool
        for preferred in ("web_search", "visit_page", "find_on_page_ctrl_f"):
            if preferred in (tools_available or []) and preferred != current_tool:
                better_tool = preferred
                break

    if current_tool and better_tool and current_tool != better_tool:
        patched = snippet.replace(f'"name": "{current_tool}"', f'"name": "{better_tool}"')
        patched = patched.replace(f"'name': '{current_tool}'", f"'name': '{better_tool}'")
        return {
            "patched_text": patched,
            "original_call": {"tool": current_tool},
            "repaired_call": {"tool": better_tool},
            "swap_reason": {
                "why_original_wrong": (description[:100] if description else "tool does not match subgoal"),
                "why_new_tool_better": f"'{better_tool}' better matches the retrieval/navigation intent.",
            },
        }

    note = (
        f"\n[TOOL SWAP: Replace '{current_tool or 'current tool'}' "
        f"with a better-matched tool from {tools_available[:4]}.]"
    )
    return {
        "patched_text": snippet + note,
        "swap_note": "Could not auto-swap; manual selection needed.",
    }


def _instantiate_context_state_carryover(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    CONTEXT_STATE_CARRYOVER: append a state-capture annotation so the key
    finding from this step is explicitly preserved for the next step.
    """
    evidence = (error.get("evidence") or "")[:80].strip()
    state_hint = evidence if evidence else "<key finding from prior step>"

    carryover = (
        f"\n[STATE CARRYOVER: Capture → prior_result = '{state_hint}'. "
        f"The next step MUST reference this value instead of re-querying.]"
    )
    return {
        "patched_text": snippet + carryover,
        "state_update": {
            "update_mode": "EXPLICIT_REFERENCE_TO_PRIOR_OUTPUT",
            "state_key": "prior_result",
            "state_value": state_hint,
            "used_by_next_step": "reference prior_result directly",
        },
    }


def _instantiate_goal_constraint_check(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    GOAL_CONSTRAINT_CHECK: insert a requirement checkpoint before final_answer
    (or at the top of the snippet if no final_answer is present).
    """
    evidence = (error.get("evidence") or "")[:100].strip()
    req = evidence if evidence else "<task requirement>"

    checkpoint = (
        f"[CHECKPOINT: Before proceeding, verify '{req}' is satisfied. "
        f"If not, execute the missing action first.]\n"
    )
    m = _FINAL_ANSWER_RE.search(snippet)
    if m:
        patched = snippet[: m.start()] + checkpoint + snippet[m.start() :]
    else:
        patched = checkpoint + snippet

    return {
        "patched_text": patched,
        "checkpoint_step": {
            "check_mode": "REQUIREMENT_CHECKLIST_BEFORE_FINAL_ANSWER",
            "requirement": req,
            "fail_action": "execute missing action or revise output",
        },
    }


def _instantiate_output_interpretation_verify(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    OUTPUT_INTERPRETATION_VERIFY: insert a parse-and-verify step after the
    Observation block (or at end if no Observation found).
    """
    verify_block = (
        "\n[VERIFY: Parse tool output — check if empty or contains failure indicator. "
        "If failure: stop or switch strategy. If success: extract key fields and proceed.]\n"
    )
    m = _OBSERVATION_RE.search(snippet)
    if m:
        # Insert after the first line following "Observation:"
        insert_at = snippet.find("\n", m.end())
        insert_at = insert_at + 1 if insert_at != -1 else len(snippet)
        patched = snippet[:insert_at] + verify_block + snippet[insert_at:]
    else:
        patched = snippet + verify_block

    return {
        "patched_text": patched,
        "verification_step": {
            "verify_mode": "CHECK_FAILURE_SIGNAL_THEN_PROCEED",
            "failure_check": "output is empty or contains failure indicator",
            "success_action": "proceed using extracted fields",
            "failure_action": "stop or switch to alternative retrieval step",
        },
    }


def _instantiate_retrieval_requery(
    spec: dict, error: dict, snippet: str, tools_available: List[str]
) -> dict:
    """
    RETRIEVAL_REQUERY: find the query/search_string; append specificity terms
    derived from the error evidence.
    """
    evidence = (error.get("evidence") or "")[:80].strip()
    m = _QUERY_RE.search(snippet)
    if m:
        orig_query = m.group("q1") or m.group("q2") or m.group("q3") or m.group("q4") or ""
        # Append context from evidence if it adds new information
        if evidence and evidence not in orig_query:
            new_query = f"{orig_query} {evidence}".strip()[:200]
        else:
            new_query = orig_query + " [more specific]"
        patched = snippet.replace(orig_query, new_query, 1)
        return {
            "patched_text": patched,
            "original_call": {"query": orig_query},
            "repaired_call": {"query": new_query},
            "rewrite_note": "Made query more specific by appending evidence context.",
        }

    note = "\n[REQUERY: Rewrite the retrieval query above to be more specific and task-grounded.]"
    return {
        "patched_text": snippet + note,
        "rewrite_note": "Query pattern not found; manual rewrite needed.",
    }


_INSTANTIATORS = {
    "BUDGET_GUARD_STOP_CONDITION":    _instantiate_budget_guard,
    "TOOL_SCHEMA_REPAIR":             _instantiate_tool_schema_repair,
    "EXECUTE_INSTEAD_OF_DESCRIBE":    _instantiate_execute_instead_of_describe,
    "TOOL_SELECTION_SWAP":            _instantiate_tool_selection_swap,
    "CONTEXT_STATE_CARRYOVER":        _instantiate_context_state_carryover,
    "GOAL_CONSTRAINT_CHECK":          _instantiate_goal_constraint_check,
    "OUTPUT_INTERPRETATION_VERIFY":   _instantiate_output_interpretation_verify,
    "RETRIEVAL_REQUERY":              _instantiate_retrieval_requery,
}


# ---------------------------------------------------------------------------
# Spec instantiation (D)
# ---------------------------------------------------------------------------


def instantiate_spec(
    spec: PatchSpec,
    error_instance: dict,
    snippet: str,
    trace_obj: TraceObj,
) -> dict:
    """
    Fill all placeholders in spec.raw and produce a concrete patched_text.

    Placeholder substitutions:
      {ERROR_TYPE}     → error_instance.error_type / category
      {LOCAL_SNIPPET}  → snippet
      <NUM>            → sensible default (e.g. 3 for max_retries)

    Returns a dict that always contains at least:
      "operator_family", "error_type", "local_snippet", "patch_type", "patched_text"
    plus family-specific structured fields.
    """
    family = spec.operator_family
    tools = trace_obj.tools_available if trace_obj else []
    instantiator = _INSTANTIATORS.get(family)
    if instantiator:
        result = instantiator(spec.raw, error_instance, snippet, tools)
    else:
        result = {"patched_text": snippet + f"\n[PATCH: Apply {family} — manual intervention needed.]"}

    result.setdefault("operator_family", family)
    result.setdefault("error_type", error_instance.get("error_type") or error_instance.get("category") or "")
    result.setdefault("local_snippet", snippet[:500])
    result.setdefault("patch_type", spec.patch_type)
    return result


# ---------------------------------------------------------------------------
# apply_patch (A5)
# ---------------------------------------------------------------------------


def apply_patch(
    trace_obj: TraceObj,
    error_instance: dict,
    patch_spec: PatchSpec,
    window: int = 0,
) -> PatchRecord:
    """
    Core operation (A5):
      1. Extract the local snippet at error_instance["annotated_span_id"]
         (optionally expanded by `window` sibling spans).
      2. Instantiate the spec with concrete values.
      3. Build and return a PatchRecord.

    Does NOT mutate trace_obj.  Creates a copy implicitly through string ops.
    """
    location = (
        error_instance.get("annotated_span_id")
        or error_instance.get("location")
        or ""
    )

    if not location:
        return PatchRecord(
            trace_id=trace_obj.trace_id,
            error_id=error_instance.get("error_id", ""),
            operator_family=patch_spec.operator_family,
            location="",
            original_text="",
            patched_text="",
            diff_lines=0,
            instantiated_spec={},
            validation={"ok": False, "reasons": ["missing_location"]},
            success=False,
        )

    original = get_expanded_snippet(trace_obj, location, window=window)
    inst = instantiate_spec(patch_spec, error_instance, original, trace_obj)
    patched = inst.get("patched_text", original)

    diff_lines = _count_diff_lines(original, patched)
    ok, reasons = validate_patch_record(original, patched, error_instance, patch_spec)

    return PatchRecord(
        trace_id=trace_obj.trace_id,
        error_id=error_instance.get("error_id", ""),
        operator_family=patch_spec.operator_family,
        location=location,
        original_text=original,
        patched_text=patched,
        diff_lines=diff_lines,
        instantiated_spec=inst,
        validation={"ok": ok, "reasons": reasons},
        success=ok,
    )


# ---------------------------------------------------------------------------
# Validation (A6)
# ---------------------------------------------------------------------------


# Detect if new Observation: content was fabricated
_FABRICATED_OBS_RE = re.compile(r"Observation\s*:\s*\n\s*\w", re.IGNORECASE)


def _count_diff_lines(original: str, patched: str) -> int:
    return sum(
        1
        for line in difflib.unified_diff(
            original.splitlines(), patched.splitlines()
        )
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def validate_patch_record(
    original: str,
    patched: str,
    error_instance: dict,
    spec: PatchSpec,
    max_diff_lines: int = 60,
) -> Tuple[bool, List[str]]:
    """
    Cheap validations (A6).  Returns (is_valid, failure_reasons).

    Rules:
      1. Patch must change something.
      2. Change must be minimal (< max_diff_lines).
      3. No fabricated Observation: content in new lines.
      4. BUDGET_GUARD patches must contain guard keywords.
    """
    reasons: List[str] = []

    # 1. Must change something
    if original.strip() == patched.strip():
        reasons.append("no_change: patched text is identical to original")

    # 2. Must be minimal
    diff_lines = _count_diff_lines(original, patched)
    if diff_lines > max_diff_lines:
        reasons.append(f"too_large: {diff_lines} changed lines > {max_diff_lines}")

    # 3. No fabricated observation content
    orig_obs = {m.start() for m in _FABRICATED_OBS_RE.finditer(original)}
    patched_obs = {m.start() for m in _FABRICATED_OBS_RE.finditer(patched)}
    if patched_obs - orig_obs:
        reasons.append("fabricated_output: patch inserts Observation: content")

    # 4. Guard keywords required for budget guard
    if spec.operator_family == "BUDGET_GUARD_STOP_CONDITION":
        if not _GUARD_KEYWORDS_RE.search(patched):
            reasons.append("guard_missing: budget guard patch lacks guard keywords")

    return len(reasons) == 0, reasons
