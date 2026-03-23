#!/usr/bin/env python3
"""
Build an Action Primitive Dictionary from GAIA/TRAIL agent logs and error annotations.

From raw traces + error annotations, constructs:
- Tool primitives: tool names, argument keys, common argument shapes
- Control-flow primitives: CALL_TOOL, DESCRIBE_TOOL_NO_CALL, RETRY_LOOP, PLAN_ONLY, VERIFY_STEP, etc.
- State/code primitives: assign, print, call, loop
- Canonical templates for counterfactual replacement
- Primitive ↔ error_type co-occurrence for intervenable knobs

Parsing strategy (TRAIL-style trace):
- Prefer structured: LLM tool_calls (flattened keys), TOOL spans (tool.name, input.value)
- Fallback: regex on "Calling tools:", "Call id:", "Action:" blobs in output.value
- Intent heuristics: I will call, retry, plan:, verify, etc.

Usage:
  # From benchmarking/ directory (so span_level_parser is importable):
  python action_primitive_library.py --trace_dir data/GAIA --annotations_dir processed_annotations_gaia --out_dir action_primitive_artifacts
  # Single trace:
  python action_primitive_library.py --trace_dir data/GAIA --annotations_dir processed_annotations_gaia --out_dir action_primitive_artifacts --trace_ids b241cb7deedf9646f01fa15095ed96d2
  # Cap traces:
  python action_primitive_library.py --trace_dir data/GAIA --annotations_dir processed_annotations_gaia --out_dir action_primitive_artifacts --max_traces 100

  Programmatically:
  from action_primitive_library import build_library
  build_library("data/GAIA", "processed_annotations_gaia", "action_primitive_artifacts", max_traces=10)
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from span_level_parser import (
    parse_trace_to_step_level,
    map_annotation_to_step,
    build_error_annotation_output,
    get_span_kind,
)

# ---------------------------------------------------------------------------
# Regex patterns (unstructured extraction)
# ---------------------------------------------------------------------------

CALLING_TOOLS_RE = re.compile(r"Calling tools:\s*\n(?P<blob>\[.*?\])", re.DOTALL)
CALL_ID_RE = re.compile(r"Call id:\s*(?P<call_id>\S+)\s*\nObservation:\s*\n(?P<obs>.*)", re.DOTALL)
ACTION_BLOB_RE = re.compile(r"Action:\s*(\{.*?\})\s*(?:Observation:|$)", re.DOTALL)
INTENT_TOOL_RE = re.compile(
    r"\b(I will|I'll|Now I will|Let's)\b.*\b(call|use|run|invoke|ask)\b", re.IGNORECASE
)
RETRY_RE = re.compile(
    r"\b(retry|try again|re-run|search again|no result|not found)\b", re.IGNORECASE
)
VERIFY_RE = re.compile(
    r"\b(check|validate|confirm|verify|unit test|assert)\b", re.IGNORECASE
)
PLAN_MARKER_RE = re.compile(
    r"\b(plan:|step\s*1|step-by-step|step\s*\d+)\b", re.IGNORECASE
)
# Code block extraction
CODE_BLOCK_RE = re.compile(r"```(?:py|python)?\s*\n(.*?)```", re.DOTALL)
PRINT_PLUS_TOOL_RE = re.compile(
    r"print\s*\(.*\b(send|call|use)\s+(?:the\s+)?tool\b", re.IGNORECASE
)

# Span kinds we treat as actionable
ACTIONABLE_KINDS = {"AGENT", "LLM", "TOOL", "CHAIN"}

# Keys injected by smolagents/opentelemetry instrumentation that are NOT user-level tool args.
# "arguments" appears as {"arguments": {}} in TOOL span input.value for no-arg tools.
_INSTRUMENTATION_KEYS = frozenset({"sanitize_inputs_outputs", "args", "arguments"})


def _canonicalize_args(raw_args: Any, tool_name: str = "") -> Dict[str, Any]:
    """
    Unwrap smolagents instrumentation wrapper and strip non-user-level fields.

    Rules (in order):
      1. If raw_args has a "kwargs" key whose value is a dict → use that inner dict.
         (Covers {kwargs: {...}, args: [...], sanitize_inputs_outputs: ...} wrapper.)
      2. If raw_args has "arguments" as the *only* meaningful key and its value is a dict
         → unwrap it (handles {"arguments": {}} TOOL-span wrapper for no-arg tools).
      3. Drop _INSTRUMENTATION_KEYS ("sanitize_inputs_outputs", bare "args", "arguments").
      4. Drop any key that is None, empty, or whitespace-only.
      5. Drop any key that equals the tool_name itself (guards against malformed calls
         like page_down({"page_down": ""}) that are instrumentation/agent bugs).
    """
    if not isinstance(raw_args, dict):
        return {}
    args = raw_args
    # Rule 1: unwrap kwargs wrapper
    if "kwargs" in args and isinstance(args.get("kwargs"), dict):
        args = args["kwargs"]
    # Rule 2: unwrap bare "arguments" wrapper (e.g. {"arguments": {}} from TOOL spans)
    elif (
        "arguments" in args
        and isinstance(args.get("arguments"), dict)
        and all(k in _INSTRUMENTATION_KEYS or not str(k).strip() for k in args if k != "arguments")
    ):
        args = args["arguments"]
    return {
        k: v for k, v in args.items()
        if k
        and isinstance(k, str)
        and k.strip()
        and k not in _INSTRUMENTATION_KEYS
        and k != tool_name  # never allow the tool name itself as an arg key
    }


# ---------------------------------------------------------------------------
# Step 0 — Turn object: build unified records per turn (step-level)
# ---------------------------------------------------------------------------


def _get_attr(span: Dict[str, Any], key: str) -> Any:
    """Get value from span_attributes (or attributes)."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    return attrs.get(key) if isinstance(attrs, dict) else None


def _get_output_text(span: Dict[str, Any]) -> str:
    """Concatenate output.value and relevant log bodies into one text blob."""
    parts = []
    out_val = _get_attr(span, "output.value")
    if isinstance(out_val, str):
        parts.append(out_val)
    for log in span.get("logs") or []:
        body = log.get("body") or {}
        if isinstance(body, dict):
            for k in ("output.value", "function.output", "message.content"):
                v = body.get(k)
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item.get("text", ""))
    return "\n".join(parts)


def _get_input_text(span: Dict[str, Any]) -> str:
    """Get input.value and input message content as text."""
    parts = []
    inv = _get_attr(span, "input.value")
    if isinstance(inv, str):
        parts.append(inv)
    # Flattened keys like llm.input_messages.0.message.content
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            if "input_messages" in k and "content" in k and isinstance(v, str):
                parts.append(v)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action-span identification helpers (for CHAIN memory_step granularity)
# ---------------------------------------------------------------------------


def _is_agent_action_span(span: Dict[str, Any]) -> bool:
    """True for CHAIN spans that represent one individual agent step (Thought+Code+Observation)."""
    kind = _get_attr(span, "openinference.span.kind")
    if kind != "CHAIN":
        return False
    inv = _get_attr(span, "input.value")
    return isinstance(inv, str) and "memory_step" in inv


def _span_sort_key(span: Dict[str, Any]) -> Tuple[str, str]:
    """Sortable (timestamp, span_id) for consistent ordering."""
    ts = str(span.get("timestamp") or span.get("start_time") or "")
    return (ts, span.get("span_id") or "")


def _find_all_action_spans(span_by_id: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All CHAIN-memory_step spans sorted by timestamp (= individual agent action steps)."""
    result = [s for s in span_by_id.values() if _is_agent_action_span(s)]
    result.sort(key=_span_sort_key)
    return result


def _find_action_span_ancestor(
    span_id: str,
    span_by_id: Dict[str, Any],
    parent_of: Dict[str, str],
    action_span_ids: set,
) -> Optional[str]:
    """Walk parent_of from span_id upward to find the nearest CHAIN-action ancestor (inclusive)."""
    cur: Optional[str] = span_id
    while cur:
        if cur in action_span_ids:
            return cur
        cur = parent_of.get(cur)
    return None


def _collect_descendant_span_ids(step_span_id: str, parent_of: Dict[str, str]) -> List[str]:
    """Return all span_ids that have step_span_id as ancestor (step plus all descendants)."""
    out = [step_span_id]
    added = {step_span_id}
    while True:
        more = [sid for sid, pid in parent_of.items() if pid in added and sid not in added]
        if not more:
            break
        for sid in more:
            added.add(sid)
            out.append(sid)
    return out


def _collect_descendant_spans(
    step_span: Dict[str, Any],
    span_by_id: Dict[str, Dict],
    parent_of: Dict[str, str],
    step_span_ids: List[str],
) -> List[Dict[str, Any]]:
    """Collect this step span and all descendant spans (for aggregating turn content)."""
    step_id = step_span.get("span_id")
    if step_id not in step_span_ids:
        return []
    ids_order = _collect_descendant_span_ids(step_id, parent_of)
    return [span_by_id[sid] for sid in ids_order if sid in span_by_id]


def _extract_tool_calls_from_llm_span(span: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from LLM span: flattened keys or output.value JSON."""
    calls = []
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return calls

    # 1) Flattened keys: llm.output_messages.0.message.tool_calls.N.tool_call.function.*
    idx = 0
    while True:
        name_key = f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.name"
        args_key = f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.arguments"
        name = attrs.get(name_key)
        args_str = attrs.get(args_key)
        if name is None and args_str is None:
            break
        tool_name = str(name) if name is not None else ""
        raw_args: Any = {}
        if args_str and isinstance(args_str, str):
            try:
                raw_args = json.loads(args_str)
            except json.JSONDecodeError:
                pass
        canonical = _canonicalize_args(raw_args, tool_name=tool_name)
        calls.append({
            "tool": tool_name,
            "args": canonical,
            "raw": json.dumps(canonical),
            "is_valid_json": True,
            "source": "LLM_tool_call",
        })
        idx += 1

    # 2) output.value with tool_calls array (fallback when flattened keys absent)
    out_val = attrs.get("output.value")
    if isinstance(out_val, str) and "tool_calls" in out_val and not calls:
        try:
            data = json.loads(out_val)
            tc = data.get("tool_calls") or []
            for t in tc:
                if isinstance(t, dict):
                    fn = t.get("function") or t
                    tool_name = fn.get("name") or t.get("name") or ""
                    ar = fn.get("arguments")
                    if isinstance(ar, str):
                        try:
                            ar = json.loads(ar)
                        except json.JSONDecodeError:
                            ar = {}
                    canonical = _canonicalize_args(ar, tool_name=tool_name)
                    calls.append({
                        "tool": tool_name,
                        "args": canonical,
                        "raw": json.dumps(canonical),
                        "is_valid_json": True,
                        "source": "LLM_output_value",
                    })
        except json.JSONDecodeError:
            pass

    return calls


def _extract_tool_call_from_tool_span(span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract one tool call from TOOL span: tool.name or span_name, input.value."""
    kind = _get_attr(span, "openinference.span.kind")
    if kind != "TOOL":
        return None
    tool_name = _get_attr(span, "tool.name")
    if not tool_name:
        span_name = (span.get("span_name") or "").strip()
        # Map known span_name → canonical tool name
        _SPAN_TO_TOOL = {
            "VisitTool": "visit_page",
            "FinderTool": "find_on_page_ctrl_f",
        }
        tool_name = _SPAN_TO_TOOL.get(span_name, span_name or None)
    if not tool_name:
        return None
    input_val = _get_attr(span, "input.value")
    raw_args: Any = {}
    if isinstance(input_val, str):
        try:
            raw_args = json.loads(input_val)
        except json.JSONDecodeError:
            pass
    canonical = _canonicalize_args(raw_args, tool_name=str(tool_name))
    return {
        "tool": str(tool_name),
        "args": canonical,
        "raw": json.dumps(canonical),
        "is_valid_json": True,
        "source": "TOOL_span",
    }


def _parse_calling_tools_blob(text: str) -> List[Dict[str, Any]]:
    """Parse 'Calling tools:\n[...]' blob; content may be Python literal (single quotes)."""
    calls = []
    m = CALLING_TOOLS_RE.search(text)
    if not m:
        return calls
    blob = m.group("blob").strip()
    try:
        blob_json = blob.replace("'", '"')
        arr = json.loads(blob_json)
    except json.JSONDecodeError:
        try:
            arr = ast.literal_eval(blob)
        except (ValueError, SyntaxError):
            return calls
    if not isinstance(arr, list):
        return calls
    for item in arr:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or item
        name = fn.get("name") or item.get("name")
        ar = fn.get("arguments")
        if isinstance(ar, str):
            try:
                ar = json.loads(ar)
            except json.JSONDecodeError:
                ar = {}
        if name:
            canonical = _canonicalize_args(ar, tool_name=str(name))
            calls.append({
                "tool": str(name),
                "args": canonical,
                "raw": json.dumps(canonical),
                "is_valid_json": True,
                "source": "CALLING_TOOLS_text",
            })
    return calls


def _parse_action_blobs(text: str) -> List[Dict[str, Any]]:
    """Parse 'Action: {...}' JSON blobs."""
    calls = []
    for m in ACTION_BLOB_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
            name = data.get("name")
            ar = data.get("arguments") or {}
            if name:
                canonical = _canonicalize_args(ar, tool_name=str(name))
                calls.append({
                    "tool": str(name),
                    "args": canonical,
                    "raw": json.dumps(canonical),
                    "is_valid_json": True,
                    "source": "ACTION_blob",
                })
        except json.JSONDecodeError:
            pass
    return calls


def _extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Combine Calling tools + Action blobs from unstructured text."""
    calls = []
    calls.extend(_parse_calling_tools_blob(text))
    calls.extend(_parse_action_blobs(text))
    return calls


def _extract_code_blocks(text: str) -> List[str]:
    """Extract code snippets from markdown fences."""
    return CODE_BLOCK_RE.findall(text)


def _infer_code_actions(code: str) -> List[Dict[str, Any]]:
    """Lightweight code action extraction: call, assign, print, loop, return, import."""
    actions = []
    # Simple patterns (avoid full AST for robustness)
    for line in code.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^\s*print\s*\(", line):
            actions.append({"kind": "print", "callee": None, "args_sig": None, "raw": line[:200]})
        elif re.match(r"^\s*(for|while)\s+", line):
            actions.append({"kind": "loop", "callee": None, "args_sig": None, "raw": line[:200]})
        elif re.match(r"^\s*return\s+", line):
            actions.append({"kind": "return", "callee": None, "args_sig": None, "raw": line[:200]})
        elif re.match(r"^\s*import\s+|^\s*from\s+", line):
            actions.append({"kind": "import", "callee": None, "args_sig": None, "raw": line[:200]})
        elif "=" in line and re.match(r"^\s*\w+\s*=", line):
            actions.append({"kind": "assign", "callee": None, "args_sig": None, "raw": line[:200]})
        else:
            # call: foo(...) or obj.method(...)
            call_m = re.match(r"^\s*(\w+(?:\.\w+)*)\s*\(", line)
            if call_m:
                callee = call_m.group(1)
                actions.append({"kind": "call", "callee": callee, "args_sig": "(...)", "raw": line[:200]})
    return actions


def _extract_intent(text: str) -> Dict[str, Any]:
    """Intent flags from unstructured text."""
    mentions_tool = bool(INTENT_TOOL_RE.search(text))
    mentions_verify = bool(VERIFY_RE.search(text))
    mentions_plan = bool(PLAN_MARKER_RE.search(text))
    mentions_retry = bool(RETRY_RE.search(text))
    mentions_tool_name = None
    # Heuristic: try to find a tool name after "call" / "use"
    for m in re.finditer(r"\b(call|use|invoke)\s+(\w+(?:_\w+)*)", text, re.IGNORECASE):
        mentions_tool_name = m.group(2)
        break
    return {
        "mentions_tool": mentions_tool,
        "mentions_tool_name": mentions_tool_name,
        "mentions_verify": mentions_verify,
        "mentions_plan": mentions_plan,
        "mentions_retry": mentions_retry,
    }


def build_action_turns(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build one Turn per individual agent action (CHAIN memory_step span), sorted by timestamp.
    Falls back to level-1 step turns when no CHAIN-action spans exist (e.g. different trace format).

    Each turn contains:
      - tool_calls: from direct LLM child spans (canonical args); TOOL child spans as fallback
      - code_blocks: from LLM output code fences
      - text: CHAIN output + LLM output text (for intent/control-flow detection)
      - errors: error annotations whose annotated span falls under this action span
    """
    trace_id = parsed.get("trace_id") or ""
    span_by_id = parsed.get("span_by_id") or {}
    parent_of = parsed.get("parent_of") or {}
    error_annotations = parsed.get("error_annotations") or []

    action_spans = _find_all_action_spans(span_by_id)

    # Fallback: level-1 step spans when trace has no CHAIN-memory_step structure
    if not action_spans:
        return _build_step_level_turns(parsed)

    action_span_ids = {s.get("span_id") for s in action_spans if s.get("span_id")}

    # Map annotated_span_id → action span (walk parent_of upward)
    action_turn_of_error: Dict[str, str] = {}
    for rec in error_annotations:
        ann_sid = rec.get("annotated_span_id")
        if ann_sid:
            act_id = _find_action_span_ancestor(ann_sid, span_by_id, parent_of, action_span_ids)
            if act_id:
                action_turn_of_error[ann_sid] = act_id

    # Build action_span_id → list of error records
    action_errors: Dict[str, List] = defaultdict(list)
    for rec in error_annotations:
        ann_sid = rec.get("annotated_span_id")
        act_id = action_turn_of_error.get(ann_sid) if ann_sid else None
        if act_id:
            action_errors[act_id].append(rec)

    # Direct children: span_id → list of child spans
    children_of: Dict[str, List] = defaultdict(list)
    for sid, sp in span_by_id.items():
        pid = parent_of.get(sid)
        if pid:
            children_of[pid].append(sp)

    turns = []
    for turn_idx, action_span in enumerate(action_spans):
        action_span_id = action_span.get("span_id")
        children = children_of.get(action_span_id, [])

        all_tool_calls: List[Dict] = []
        all_code_blocks: List[str] = []
        text_parts: List[str] = []

        # CHAIN span's own output (Execution logs / observations)
        chain_text = _get_output_text(action_span)
        text_parts.append(chain_text)
        # Regex fallback on CHAIN output (e.g. "Calling tools:" blobs in code agent)
        all_tool_calls.extend(_extract_tool_calls_from_text(chain_text))
        all_code_blocks.extend(_extract_code_blocks(chain_text))

        for child in children:
            kind = _get_attr(child, "openinference.span.kind")
            if kind == "LLM":
                tc_list = _extract_tool_calls_from_llm_span(child)
                all_tool_calls.extend(tc_list)
                child_text = _get_output_text(child)
                text_parts.append(child_text)
                all_code_blocks.extend(_extract_code_blocks(child_text))
            elif kind == "TOOL":
                tc = _extract_tool_call_from_tool_span(child)
                if tc:
                    all_tool_calls.append(tc)
                text_parts.append(_get_output_text(child))

        text = "\n".join(text_parts)

        # Deduplicate tool calls by (tool, raw) — LLM_tool_call preferred over regex
        seen_tc: set = set()
        deduped: List[Dict] = []
        # Prefer structured sources first
        for priority in ("LLM_tool_call", "LLM_output_value", "TOOL_span", "CALLING_TOOLS_text", "ACTION_blob"):
            for tc in all_tool_calls:
                if tc.get("source") != priority:
                    continue
                key = (tc.get("tool", ""), tc.get("raw", ""))
                if key not in seen_tc:
                    seen_tc.add(key)
                    deduped.append(tc)

        turns.append({
            "trace_id": trace_id,
            "turn_id": turn_idx + 1,
            "action_span_id": action_span_id,
            "role": "agent",
            "text": text,
            "tool_calls": deduped,
            "code_blocks": all_code_blocks,
            "errors": action_errors.get(action_span_id, []),
        })

    return turns


def _build_step_level_turns(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fallback: one turn per level-1 step span (when trace lacks CHAIN-memory_step spans).
    Aggregates all tool calls and text across all descendants.
    """
    trace_id = parsed.get("trace_id") or ""
    step_spans = parsed.get("step_spans") or []
    span_by_id = parsed.get("span_by_id") or {}
    parent_of = parsed.get("parent_of") or {}
    step_span_ids = parsed.get("step_span_ids") or []
    error_annotations = parsed.get("error_annotations") or []

    step_errors: Dict[str, List] = defaultdict(list)
    for rec in error_annotations:
        sid = rec.get("step_span_id")
        if sid:
            step_errors[sid].append(rec)

    turns = []
    for step_entry in step_spans:
        step_span = step_entry["span"]
        step_index = step_entry["step_index"]
        step_span_id = step_span.get("span_id")
        descendants = _collect_descendant_spans(step_span, span_by_id, parent_of, step_span_ids)

        all_tool_calls: List[Dict] = []
        all_code_blocks: List[str] = []
        text_parts: List[str] = []
        for sp in descendants:
            kind = _get_attr(sp, "openinference.span.kind")
            if kind == "LLM":
                all_tool_calls.extend(_extract_tool_calls_from_llm_span(sp))
                text_parts.append(_get_output_text(sp))
            elif kind == "TOOL":
                tc = _extract_tool_call_from_tool_span(sp)
                if tc:
                    all_tool_calls.append(tc)
                text_parts.append(_get_output_text(sp))
            elif kind in ("CHAIN", "AGENT"):
                txt = _get_output_text(sp)
                text_parts.append(txt)
                all_tool_calls.extend(_extract_tool_calls_from_text(txt))
                all_code_blocks.extend(_extract_code_blocks(txt))
        text = "\n".join(text_parts)
        if not all_tool_calls:
            all_tool_calls = _extract_tool_calls_from_text(text)

        turns.append({
            "trace_id": trace_id,
            "turn_id": step_index,
            "action_span_id": step_span_id,
            "role": "agent",
            "text": text,
            "tool_calls": all_tool_calls,
            "code_blocks": all_code_blocks,
            "errors": step_errors.get(step_span_id, []),
        })

    return turns


# ---------------------------------------------------------------------------
# Step 1 — Events per turn: tool calls (done above), code actions, intents
# ---------------------------------------------------------------------------


def extract_events_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    """From one turn, produce tool_calls (already in turn), code_actions, intent."""
    code_actions: List[Dict] = []
    for block in turn.get("code_blocks") or []:
        code_actions.extend(_infer_code_actions(block))
    intent = _extract_intent(turn.get("text") or "")
    return {
        "tool_calls": turn.get("tool_calls") or [],
        "code_actions": code_actions,
        "intent": intent,
    }


# ---------------------------------------------------------------------------
# Step 2 — Aggregate primitives into dictionaries
# ---------------------------------------------------------------------------


def _arg_value_shape(val: Any) -> str:
    """Bucket argument value for schema stats."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        if len(val) <= 20:
            return "str_0_20"
        if len(val) <= 100:
            return "str_21_100"
        return "str_100+"
    if isinstance(val, list):
        return f"list_{len(val)}"
    if isinstance(val, dict):
        return f"dict_{len(val)}"
    return "other"


def aggregate_primitive_stats(
    all_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate tools, arg_key_stats, control_flow, code_actions, and initial templates.
    all_turns = list of turn dicts (each with tool_calls, code_blocks, text, errors)
    from one or more traces.
    """
    tools: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "arg_keys": defaultdict(int),
        "arg_examples": defaultdict(list),
        "arg_value_shapes": defaultdict(lambda: defaultdict(int)),
        "sources": defaultdict(int),
    })
    arg_key_stats: Dict[str, int] = defaultdict(int)
    control_flow: Dict[str, int] = defaultdict(int)
    code_actions: Dict[str, int] = defaultdict(int)
    # Raw canonical call strings per tool (consumed by mine_templates)
    raw_calls_per_tool: Dict[str, List[str]] = defaultdict(list)

    for turn in all_turns:
        events = extract_events_from_turn(turn)
        tool_calls = events["tool_calls"]
        code_actions_list = events["code_actions"]
        intent = events["intent"]
        text = turn.get("text") or ""

        # --- Tool primitives ---
        for tc in tool_calls:
            tname = (tc.get("tool") or "").strip()
            if not tname:
                continue
            tools[tname]["count"] += 1
            tools[tname]["sources"][tc.get("source", "unknown")] += 1
            args = tc.get("args") or {}
            for k, v in args.items():
                if k is None or not str(k).strip():
                    continue
                tools[tname]["arg_keys"][k] += 1
                arg_key_stats[k] += 1
                shape = _arg_value_shape(v)
                tools[tname]["arg_value_shapes"][k][shape] += 1
                ex = tools[tname]["arg_examples"][k]
                if len(ex) < 5 and v is not None:
                    ex.append(v if not isinstance(v, (str, list, dict)) or (isinstance(v, str) and len(v) <= 80) else str(v)[:80])

            raw = tc.get("raw") or ""
            if raw and len(raw) < 500:
                raw_calls_per_tool[tname].append(raw)

        # --- Control-flow primitives ---
        has_tool_call = len(tool_calls) > 0
        if has_tool_call:
            control_flow["CALL_TOOL"] += 1
        if intent["mentions_tool"] and not has_tool_call:
            control_flow["DESCRIBE_TOOL_NO_CALL"] += 1
        if PRINT_PLUS_TOOL_RE.search(text) and intent["mentions_tool"]:
            control_flow["PRINT_INSTEAD_OF_EXECUTE"] += 1
        if intent["mentions_plan"] and not has_tool_call and not re.search(r"\b(use|call|invoke)\s+\w+", text, re.IGNORECASE):
            control_flow["PLAN_ONLY"] += 1
        if intent["mentions_verify"]:
            control_flow["VERIFY_STEP"] += 1
        if intent["mentions_retry"]:
            control_flow["RETRY_LOOP"] += 1

        # --- Code action counts ---
        for ca in code_actions_list:
            kind = ca.get("kind") or "other"
            if kind == "call" and ca.get("callee"):
                code_actions[f"call:{ca['callee']}"] += 1
            else:
                code_actions[kind] += 1

    # Build final tools structure with serializable types
    tools_out = {}
    for tname, tdata in tools.items():
        tools_out[tname] = {
            "count": tdata["count"],
            "arg_keys": dict(tdata["arg_keys"]),
            "arg_examples": {k: list(v)[:5] for k, v in tdata["arg_examples"].items()},
            "arg_value_shapes": {k: dict(v) for k, v in tdata["arg_value_shapes"].items()},
            "sources": dict(tdata["sources"]),
        }

    return {
        "tools": tools_out,
        "arg_key_stats": dict(arg_key_stats),
        "control_flow": dict(control_flow),
        "code_actions": dict(code_actions),
        "_raw_calls_per_tool": dict(raw_calls_per_tool),
    }


# ---------------------------------------------------------------------------
# Step 3 — Template mining (refine templates)
# ---------------------------------------------------------------------------


def mine_templates(
    stats: Dict[str, Any],
    max_generic: int = 10,
    max_literals_per_key: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Two-tier template mining from canonical call raws stored in stats["_raw_calls_per_tool"].

    Tier 1 — generic (preferred for patching):
      One entry per unique set of argument keys, with all values replaced by type placeholders.
      Example: find_on_page_ctrl_f({"search_string": "<TEXT>"})

    Tier 2 — literals (observed frequent values for short-string args):
      Top-N observed literal values per argument key (only for strings <= 20 chars).
      Example: {"search_string": ["Aquinas", "File:", "Thomas"]}

    Returns:
      {
        "tool_name": {
          "generic": ["tool_name({\"key\": \"<TEXT>\"})"],
          "literals": {"key": ["val1", "val2", ...]}
        }
      }
    """
    raw_calls = stats.get("_raw_calls_per_tool") or {}
    result: Dict[str, Dict[str, Any]] = {}

    for tname, raw_list in raw_calls.items():
        generic_sigs: Dict[frozenset, str] = {}  # keyset → template string
        literal_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for raw in raw_list[:200]:
            if not raw.strip().startswith("{"):
                continue
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue

            # Collect literal frequencies for short string values
            for k, v in args.items():
                if isinstance(v, str) and 0 < len(v) <= 20:
                    literal_counts[k][v] += 1

            # Build generic template (one per unique key-set)
            sig = frozenset(args.keys())
            if sig not in generic_sigs and len(generic_sigs) < max_generic:
                placeholder: Dict[str, str] = {}
                for k in sorted(args.keys()):
                    v = args[k]
                    if isinstance(v, str):
                        placeholder[k] = "<TEXT>"
                    elif isinstance(v, bool):
                        placeholder[k] = "<BOOL>"
                    elif isinstance(v, (int, float)):
                        placeholder[k] = "<NUM>"
                    elif v is None:
                        placeholder[k] = "<NULL>"
                    else:
                        placeholder[k] = "<JSON>"
                generic_sigs[sig] = f"{tname}({json.dumps(placeholder)})"

        generic = list(generic_sigs.values()) or [f"{tname}(<ARGS>)"]
        literals = {
            k: [v for v, _ in sorted(vc.items(), key=lambda x: -x[1])[:max_literals_per_key]]
            for k, vc in literal_counts.items()
            if vc
        }
        result[tname] = {"generic": generic, "literals": literals}

    return result


# ---------------------------------------------------------------------------
# Error type normalization (singular/plural/casing deduplication)
# ---------------------------------------------------------------------------

# Canonical form for each known variant.
# Keys are title-cased; add new pairs as you encounter them.
_ERROR_NORM_MAP: Dict[str, str] = {
    # Singular ↔ Plural merges → keep plural as canonical
    "Context Handling Failure": "Context Handling Failures",
    "Tool Selection Error": "Tool Selection Errors",
    "Formatting Error": "Formatting Errors",
    "Resource Abuse Error": "Resource Abuse",
    # Casing / punctuation variants
    "Poor Information-Retrieval": "Poor Information Retrieval",
    "Goal-Deviation": "Goal Deviation",
    "Task Orchestration Error": "Task Orchestration",
    "Instruction Non Compliance": "Instruction Non-compliance",
    "Instruction Noncompliance": "Instruction Non-compliance",
    # Add more as encountered
}


def _normalize_error_type(etype: str) -> str:
    """
    Normalize an error type string:
      1. Strip whitespace, title-case each word.
      2. Look up in _ERROR_NORM_MAP (keyed by title-cased form).
      3. If not found in map, return the title-cased form as-is.

    To extend: add new entries to _ERROR_NORM_MAP.
    """
    if not etype or not etype.strip():
        return "Unknown"
    # Normalize whitespace and title-case
    title = " ".join(etype.strip().split()).title()
    return _ERROR_NORM_MAP.get(title, title)


# ---------------------------------------------------------------------------
# Step 4 — Join with error annotations (primitive ↔ error_type)
# ---------------------------------------------------------------------------


def _turn_control_flow_flags(turn: Dict[str, Any]) -> Dict[str, bool]:
    """Compute boolean control-flow flags for a single turn."""
    events = extract_events_from_turn(turn)
    tool_calls = events["tool_calls"]
    intent = events["intent"]
    has_tool_call = len(tool_calls) > 0
    return {
        "CALL_TOOL": has_tool_call,
        "DESCRIBE_TOOL_NO_CALL": intent["mentions_tool"] and not has_tool_call,
        "VERIFY_STEP": intent["mentions_verify"],
        "RETRY_LOOP": intent["mentions_retry"],
        "PLAN_ONLY": intent["mentions_plan"] and not has_tool_call,
    }


def compute_primitive_error_stats(
    all_turns: List[Dict[str, Any]],
    stats: Dict[str, Any],
    window: int = 1,
) -> Dict[str, Any]:
    """
    For each error instance, count primitives and tools within a local context window
    of ±window turns (within the same trace), not across the whole trace.

    This prevents the "everything correlates with everything" artifact that occurs
    when you count all tools in the trace for each error.

    window=1 means turns [t-1, t, t+1] are counted per error occurrence at turn t.
    """
    error_primitive_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    error_tool_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    error_type_total: Dict[str, int] = defaultdict(int)

    # Group turns by trace_id, sort by turn_id within each trace
    turns_by_trace: Dict[str, List] = defaultdict(list)
    for turn in all_turns:
        turns_by_trace[turn["trace_id"]].append(turn)

    for trace_id, trace_turns in turns_by_trace.items():
        trace_turns_sorted = sorted(trace_turns, key=lambda t: t.get("turn_id", 0))

        for i, turn in enumerate(trace_turns_sorted):
            errors = turn.get("errors") or []
            if not errors:
                continue

            # Context window: [i-window, ..., i, ..., i+window] clamped to trace bounds
            lo = max(0, i - window)
            hi = min(len(trace_turns_sorted), i + window + 1)
            window_turns = trace_turns_sorted[lo:hi]

            # Accumulate primitives + tools across the window (once per unique primitive)
            window_flags: Dict[str, bool] = defaultdict(bool)
            window_tools: Dict[str, bool] = defaultdict(bool)
            for wt in window_turns:
                for prim, on in _turn_control_flow_flags(wt).items():
                    if on:
                        window_flags[prim] = True
                for tc in (extract_events_from_turn(wt)["tool_calls"]):
                    tname = (tc.get("tool") or "").strip()
                    if tname:
                        window_tools[tname] = True

            for err in errors:
                raw_etype = err.get("error_type") or err.get("category") or "unknown"
                etype = _normalize_error_type(raw_etype)
                error_type_total[etype] += 1
                for prim, present in window_flags.items():
                    if present:
                        error_primitive_counts[etype][prim] += 1
                for tname in window_tools:
                    error_tool_counts[etype][tname] += 1

    return {
        "window_size": window,
        "error_type_totals": dict(error_type_total),
        "primitive_given_error": {e: dict(p) for e, p in error_primitive_counts.items()},
        "tool_given_error": {e: dict(t) for e, t in error_tool_counts.items()},
        "top_primitives_per_error": {
            e: sorted(p.keys(), key=lambda k: -p[k])[:10]
            for e, p in error_primitive_counts.items()
        },
        "top_tools_per_error": {
            e: sorted(t.keys(), key=lambda k: -t[k])[:10]
            for e, t in error_tool_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Step 4b — Multi-window comparison (stability across window sizes)
# ---------------------------------------------------------------------------


def compare_window_sizes(
    all_turns: List[Dict[str, Any]],
    stats: Dict[str, Any],
    window_sizes: List[int],
    out_dir: str,
) -> Dict[str, Any]:
    """
    Run compute_primitive_error_stats for each window size in window_sizes.
    Produce a comparison JSON that shows:
      - per-window stats (error_type_totals, top_primitives_per_error, top_tools_per_error)
      - stability report: for each error type + primitive, the rank across windows
        (stable = rank doesn't change; unstable = appears only in wider windows)

    Writes: primitive_error_comparison.json
    Returns the comparison dict.
    """
    per_window: Dict[str, Any] = {}
    for w in window_sizes:
        per_window[str(w)] = compute_primitive_error_stats(all_turns, stats, window=w)

    # Build stability report: error_type → primitive → {wN: rank, ...}
    all_etypes: set = set()
    for wres in per_window.values():
        all_etypes.update((wres.get("primitive_given_error") or {}).keys())

    stability: Dict[str, Dict[str, Dict[str, int]]] = {}
    for etype in sorted(all_etypes):
        prim_rank: Dict[str, Dict[str, int]] = {}
        for w_str, wres in per_window.items():
            ranked = (wres.get("top_primitives_per_error") or {}).get(etype, [])
            for rank, prim in enumerate(ranked, start=1):
                if prim not in prim_rank:
                    prim_rank[prim] = {}
                prim_rank[prim][f"w{w_str}"] = rank
        # Sort primitives by best (lowest) average rank
        stability[etype] = dict(
            sorted(prim_rank.items(), key=lambda x: sum(x[1].values()) / len(x[1]))
        )

    # Stability summary: "stable" if a primitive appears in ALL windows for that error type
    stable_summary: Dict[str, List[str]] = {}
    for etype, prim_ranks in stability.items():
        n_windows = len(window_sizes)
        stable_summary[etype] = [
            p for p, ranks in prim_ranks.items() if len(ranks) == n_windows
        ]

    comparison = {
        "window_sizes_tested": window_sizes,
        "per_window": per_window,
        "stability": stability,
        "stable_primitives_per_error": stable_summary,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "primitive_error_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print(f"\n--- Window comparison summary (windows={window_sizes}) ---")
    for etype in sorted(stable_summary):
        stable = stable_summary[etype]
        total = (per_window[str(window_sizes[-1])].get("error_type_totals") or {}).get(etype, "?")
        print(f"  {etype} (n={total}): stable primitives = {stable}")
    print(f"\nWrote: {path}")

    return comparison


# ---------------------------------------------------------------------------
# Step 5 — Output artifacts and sanity prints
# ---------------------------------------------------------------------------


def save_artifacts(
    stats: Dict[str, Any],
    primitive_error_stats: Dict[str, Any],
    templates: Dict[str, Any],
    out_dir: str,
) -> Tuple[str, str, str]:
    """Write action_primitives.json, primitive_error_stats.json, templates.json. Return paths."""
    os.makedirs(out_dir, exist_ok=True)
    # Drop internal key
    stats_export = {k: v for k, v in stats.items() if not k.startswith("_")}
    path_prim = os.path.join(out_dir, "action_primitives.json")
    path_err = os.path.join(out_dir, "primitive_error_stats.json")
    path_tpl = os.path.join(out_dir, "templates.json")

    with open(path_prim, "w", encoding="utf-8") as f:
        json.dump(stats_export, f, indent=2)

    with open(path_err, "w", encoding="utf-8") as f:
        json.dump(primitive_error_stats, f, indent=2)

    with open(path_tpl, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2)

    return path_prim, path_err, path_tpl


def print_sanity(
    stats: Dict[str, Any],
    primitive_error_stats: Dict[str, Any],
) -> None:
    """Quick sanity prints: top tools, top arg keys, top primitives, top primitives per error type."""
    tools = stats.get("tools") or {}
    arg_key_stats = stats.get("arg_key_stats") or {}
    control_flow = stats.get("control_flow") or {}
    templates = stats.get("templates") or {}

    print("\n--- Top tools by count ---")
    for tname, tdata in sorted(tools.items(), key=lambda x: -x[1].get("count", 0))[:15]:
        print(f"  {tname}: {tdata.get('count', 0)}")

    print("\n--- Top arg keys (global) ---")
    for k, c in sorted(arg_key_stats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {c}")

    print("\n--- Top control-flow primitives ---")
    for p, c in sorted(control_flow.items(), key=lambda x: -x[1])[:15]:
        print(f"  {p}: {c}")

    print("\n--- Generic templates per tool ---")
    for tname, tdata in sorted(templates.items(), key=lambda x: -tools.get(x[0], {}).get("count", 0))[:10]:
        generics = tdata.get("generic") or []
        print(f"  {tname}: {generics[:2]}")

    print("\n--- Top 5 primitives per error type (windowed) ---")
    for etype, prims in (primitive_error_stats.get("top_primitives_per_error") or {}).items():
        total = (primitive_error_stats.get("error_type_totals") or {}).get(etype, "?")
        print(f"  {etype} (n={total}): {prims[:5]}")


# ---------------------------------------------------------------------------
# Full pipeline: load traces + annotations, build turns, aggregate, save
# ---------------------------------------------------------------------------


def load_annotations_for_trace(
    annotations_path: str,
    trace_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Load annotations from processed_annotations_gaia style file (errors[].location = span_id)."""
    if not os.path.isfile(annotations_path):
        return None
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[action_primitive_library] Skip malformed annotation file: {annotations_path} ({e})", file=sys.stderr)
        return None
    errors = data.get("errors") or []
    if not errors:
        return None
    # Normalize to list of { category, location, description, evidence, impact }
    return [{"category": e.get("category"), "location": e.get("location"), "description": e.get("description"), "evidence": e.get("evidence"), "impact": e.get("impact")} for e in errors]


def build_library(
    trace_dir: str,
    annotations_dir: str,
    out_dir: str,
    trace_ids: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
    window_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Build action primitive dictionary from GAIA trace dir + processed_annotations_gaia dir.
    If trace_ids is None, discovers all trace IDs from trace_dir that have a matching annotation file.
    """
    all_turns: List[Dict[str, Any]] = []

    if trace_ids is None:
        trace_ids = []
        for fname in os.listdir(trace_dir):
            if fname.endswith(".json"):
                tid = os.path.splitext(fname)[0]
                ann_path = os.path.join(annotations_dir, fname)
                if os.path.isfile(ann_path):
                    trace_ids.append(tid)
        if max_traces is not None:
            trace_ids = trace_ids[: max_traces]
    else:
        if max_traces is not None:
            trace_ids = trace_ids[: max_traces]

    for tid in trace_ids:
        trace_path = os.path.join(trace_dir, tid + ".json")
        ann_path = os.path.join(annotations_dir, tid + ".json")
        if not os.path.isfile(trace_path):
            continue
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        trace_data["trace_id"] = tid
        annotations = load_annotations_for_trace(ann_path, tid)

        parsed = parse_trace_to_step_level(trace_data)
        parsed["trace_id"] = tid
        if annotations:
            error_records = []
            for ann in annotations:
                span_id = ann.get("location")
                if not span_id:
                    continue
                step_mapping = map_annotation_to_step(parsed, span_id)
                rec = build_error_annotation_output(tid, span_id, ann, step_mapping)
                error_records.append(rec)
            parsed["error_annotations"] = error_records

        turns = build_action_turns(parsed)
        all_turns.extend(turns)

    stats = aggregate_primitive_stats(all_turns)
    templates = mine_templates(stats)
    stats["templates"] = templates
    # Default window for the main artifact
    primary_window = (window_sizes or [1])[0]
    primitive_error_stats = compute_primitive_error_stats(all_turns, stats, window=primary_window)

    path_prim, path_err, path_tpl = save_artifacts(
        stats, primitive_error_stats, templates, out_dir
    )
    print_sanity(stats, primitive_error_stats)
    print(f"\nWrote: {path_prim}, {path_err}, {path_tpl}")

    # Multi-window comparison (if more than one window requested)
    comparison: Optional[Dict[str, Any]] = None
    if window_sizes and len(window_sizes) > 1:
        comparison = compare_window_sizes(all_turns, stats, window_sizes, out_dir)

    return {
        "stats": stats,
        "primitive_error_stats": primitive_error_stats,
        "comparison": comparison,
        "templates": templates,
        "num_traces": len(trace_ids),
        "num_turns": len(all_turns),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build action primitive dictionary from GAIA traces + annotations")
    parser.add_argument("--trace_dir", default="data/GAIA", help="Directory of trace JSON files")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia",
                        help="Directory of per-trace annotation JSON (errors[].location)")
    parser.add_argument("--out_dir", default="action_primitive_artifacts",
                        help="Output directory for JSON artifacts")
    parser.add_argument("--trace_ids", nargs="*",
                        help="Optional list of trace IDs; default: all with annotations")
    parser.add_argument("--max_traces", type=int, default=None,
                        help="Cap number of traces to process")
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[1],
                        help="Window size(s) for primitive-error stats. "
                             "Pass multiple (e.g. --window_sizes 1 2 3) to produce "
                             "primitive_error_comparison.json alongside the default artifact "
                             "(first value used for primitive_error_stats.json).")
    args = parser.parse_args()

    build_library(
        trace_dir=args.trace_dir,
        annotations_dir=args.annotations_dir,
        out_dir=args.out_dir,
        trace_ids=args.trace_ids or None,
        max_traces=args.max_traces,
        window_sizes=args.window_sizes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
