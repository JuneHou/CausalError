#!/usr/bin/env python3
"""
TRAIL trace adapter (A2).

Loads a TRAIL/GAIA trace JSON + optional annotations file into a TraceObj that
exposes everything downstream patching needs:

  - span_by_id          : flat map span_id → span dict
  - parent_of           : child span_id → parent span_id
  - text_by_location    : span_id → decoded text snippet (output.value decoded)
  - input_by_location   : span_id → exact input (messages for LLM, input_value for TOOL/CHAIN/AGENT)
  - output_by_location  : span_id → exact output (output_text + output_value_raw)
  - tool_calls_by_location: span_id → list of {tool, args} extracted from that span
  - tools_available     : tool names declared in the AGENT span
  - managed_agents      : sub-agent names (also callable as tools)
  - errors              : list of normalised error records with error_id, annotated_span_id, ...
  - step_spans          : agent-step spans from span_level_parser (CodeAgent.run, ToolCallingAgent.run, Step N)

Key design: "location" in TRAIL annotations is a span_id.  For each annotated span you can
recover the exact input messages into that span and the exact output of that span (so
input_context patches are not approximate).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from span_level_parser import (
    parse_trace_to_step_level,
    map_annotation_to_step,
    build_error_annotation_output,
)


# ---------------------------------------------------------------------------
# TraceObj dataclass
# ---------------------------------------------------------------------------


@dataclass
class TraceObj:
    trace_id: str
    span_by_id: Dict[str, dict]
    parent_of: Dict[str, str]
    # span_id → decoded text (output.value, last LLM message, observation, ...)
    text_by_location: Dict[str, str]
    # span_id → exact input: { "kind", "messages" (LLM only), "input_value", "input_value_raw" }
    input_by_location: Dict[str, dict]
    # span_id → exact output: { "output_text", "output_value_raw" }
    output_by_location: Dict[str, dict]
    # span_id → list of tool calls parsed from that span
    tool_calls_by_location: Dict[str, List[dict]]
    # All tool names available to this trace's agent(s)
    tools_available: List[str]
    managed_agents: List[str]
    # Normalised error records (from processed_annotations file)
    errors: List[dict]
    # Agent-step spans (CodeAgent.run, ToolCallingAgent.run, Step N at any depth)
    step_spans: List[dict]
    # Original JSON (kept for traceability)
    raw_trace: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Low-level span text helpers
# ---------------------------------------------------------------------------


def _get_attr(span: dict, key: str) -> Any:
    """Read from span_attributes (or attributes fallback)."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    return attrs.get(key) if isinstance(attrs, dict) else None


def _decode_value(val: Any, max_len: int = 8000) -> str:
    """
    Decode an output.value / input.value field into plain text.
    These fields are often JSON-encoded strings; we try to extract the
    human-readable content (message.content, function.output, etc.).
    """
    if val is None:
        return ""
    if not isinstance(val, str):
        return str(val)[:max_len]
    # Try parsing as JSON
    try:
        data = json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return val[:max_len]
    if not isinstance(data, dict):
        return val[:max_len]
    # LLM response: {role, content, tool_calls}
    content = data.get("content")
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return " ".join(parts)[:max_len]
    # AGENT input: {task, ...}
    task = data.get("task")
    if isinstance(task, str):
        return task[:max_len]
    # Fallback: raw string
    return val[:max_len]


def _extract_span_text(span: dict) -> str:
    """
    Build a human-readable text block for a span:
      - output.value (decoded)
      - last LLM assistant message content (llm.output_messages.0.message.content)
      - log bodies (function.output)
    Input is intentionally kept short to avoid embedding the whole prompt.
    """
    parts: List[str] = []

    out_val = _get_attr(span, "output.value")
    if out_val:
        parts.append(_decode_value(out_val))

    # LLM flattened output message content (direct text, not JSON)
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if isinstance(attrs, dict):
        msg_content = attrs.get("llm.output_messages.0.message.content")
        if isinstance(msg_content, str) and msg_content not in parts:
            parts.append(msg_content[:4000])

    # Log bodies (function.output)
    for log in span.get("logs") or []:
        body = log.get("body") or {}
        if isinstance(body, dict):
            for k in ("function.output", "output.value"):
                v = body.get(k)
                if isinstance(v, str) and len(v) > 0:
                    parts.append(v[:800])
                    break

    return "\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Exact input/output per span (for input_context and precise patch targeting)
# ---------------------------------------------------------------------------

# Max lengths for raw values stored (avoid huge payloads)
MAX_INPUT_RAW_CHARS = 12000
MAX_OUTPUT_RAW_CHARS = 12000


def _extract_span_input(span: dict) -> dict:
    """
    Extract the exact input into this span. Returns a dict:
      - kind: "llm" | "tool" | "chain" | "agent" | "other"
      - messages: list of {"role", "content"} for LLM spans (exact input_messages)
      - input_value: human-readable summary of input (decoded)
      - input_value_raw: raw input.value string when present (exact recovery)
    """
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return {"kind": "other", "messages": [], "input_value": "", "input_value_raw": ""}

    kind_key = attrs.get("openinference.span.kind", "")
    if kind_key == "LLM" or span.get("span_name") == "LiteLLMModel.__call__":
        kind = "llm"
        messages: List[dict] = []
        i = 0
        while True:
            content_key = f"llm.input_messages.{i}.message.content"
            role_key = f"llm.input_messages.{i}.message.role"
            if content_key not in attrs:
                break
            content = attrs.get(content_key)
            role = attrs.get(role_key, "user")
            # content can be str or list of content parts
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                content = " ".join(text_parts) if text_parts else ""
            messages.append({"role": str(role), "content": content or ""})
            i += 1
        input_value_raw = attrs.get("input.value")
        if isinstance(input_value_raw, str):
            input_value_raw = input_value_raw[:MAX_INPUT_RAW_CHARS]
        else:
            input_value_raw = ""
        input_value = _decode_value(input_value_raw, max_len=8000) if input_value_raw else ""
        if not input_value and messages:
            input_value = "\n".join(f"[{m['role']}]: {m['content'][:2000]}" for m in messages)
        return {
            "kind": kind,
            "messages": messages,
            "input_value": input_value,
            "input_value_raw": input_value_raw,
        }

    if kind_key == "TOOL":
        kind = "tool"
    elif kind_key == "CHAIN":
        kind = "chain"
    elif kind_key == "AGENT":
        kind = "agent"
    else:
        kind = "other"

    input_value_raw = attrs.get("input.value")
    if isinstance(input_value_raw, str):
        input_value_raw = input_value_raw[:MAX_INPUT_RAW_CHARS]
    else:
        input_value_raw = ""
    input_value = _decode_value(input_value_raw, max_len=8000) if input_value_raw else ""
    return {
        "kind": kind,
        "messages": [],
        "input_value": input_value,
        "input_value_raw": input_value_raw,
    }


def _extract_span_output(span: dict) -> dict:
    """
    Extract the exact output of this span. Returns a dict:
      - output_text: human-readable output (decoded, same logic as _extract_span_text for output)
      - output_value_raw: raw output.value string when present (exact recovery)
    """
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return {"output_text": "", "output_value_raw": ""}

    output_value_raw = attrs.get("output.value")
    if isinstance(output_value_raw, str):
        output_value_raw = output_value_raw[:MAX_OUTPUT_RAW_CHARS]
    else:
        output_value_raw = ""

    output_text = _decode_value(output_value_raw, max_len=8000) if output_value_raw else ""

    # LLM: prefer llm.output_messages.0.message.content as the primary output text
    msg_content = attrs.get("llm.output_messages.0.message.content")
    if isinstance(msg_content, str) and msg_content:
        output_text = msg_content[:8000] if output_text != msg_content else output_text

    # Log bodies (function.output) as part of output text for TOOL spans
    for log in span.get("logs") or []:
        body = log.get("body") or {}
        if isinstance(body, dict):
            for k in ("function.output", "output.value"):
                v = body.get(k)
                if isinstance(v, str) and len(v) > 0:
                    if output_text and v[:500] not in output_text:
                        output_text = output_text + "\n" + v[:2000]
                    elif not output_text:
                        output_text = v[:8000]
                    break

    return {
        "output_text": output_text.strip(),
        "output_value_raw": output_value_raw,
    }


def _extract_tool_calls_from_span(span: dict) -> List[dict]:
    """
    Extract structured tool calls from a span (LLM or TOOL kind).
    Returns list of {tool, args, source}.
    """
    calls: List[dict] = []
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return calls

    kind = attrs.get("openinference.span.kind")

    # LLM span: flattened tool_calls keys
    if kind == "LLM":
        idx = 0
        while True:
            name_key = f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.name"
            args_key = f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.arguments"
            name = attrs.get(name_key)
            args_str = attrs.get(args_key)
            if name is None:
                break
            args: Any = {}
            if args_str:
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    pass
            calls.append({"tool": str(name), "args": args if isinstance(args, dict) else {}, "source": "LLM"})
            idx += 1

    # TOOL span
    if kind == "TOOL":
        tool_name = attrs.get("tool.name") or span.get("span_name", "")
        _SPAN_NAME_MAP = {"VisitTool": "visit_page", "FinderTool": "find_on_page_ctrl_f"}
        tool_name = _SPAN_NAME_MAP.get(tool_name, tool_name)
        in_val = attrs.get("input.value")
        args = {}
        if in_val:
            try:
                raw = json.loads(in_val)
                args = raw.get("kwargs", raw) if isinstance(raw, dict) else {}
            except json.JSONDecodeError:
                pass
        if tool_name:
            calls.append({"tool": tool_name, "args": args if isinstance(args, dict) else {}, "source": "TOOL"})

    return calls


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_trail_trace(
    trace_path: str,
    annotations_path: Optional[str] = None,
) -> TraceObj:
    """
    Load one TRAIL trace JSON (+ optional annotation file) into a TraceObj.

    trace_path        : path to the trace JSON (e.g. data/GAIA/<trace_id>.json)
    annotations_path  : path to the per-trace annotation JSON
                        (e.g. processed_annotations_gaia/<trace_id>.json).
                        errors[].location is the span_id.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    trace_id = (
        trace_data.get("trace_id")
        or os.path.splitext(os.path.basename(trace_path))[0]
    )
    trace_data["trace_id"] = trace_id

    # --- Structural parsing via span_level_parser ---
    parsed = parse_trace_to_step_level(trace_data)
    span_by_id: Dict[str, dict] = parsed.get("span_by_id") or {}
    parent_of: Dict[str, str] = parsed.get("parent_of") or {}
    step_spans: List[dict] = parsed.get("step_spans") or []

    # --- Text, exact input/output, and tool-call maps ---
    text_by_location: Dict[str, str] = {}
    input_by_location: Dict[str, dict] = {}
    output_by_location: Dict[str, dict] = {}
    tool_calls_by_location: Dict[str, List[dict]] = {}
    for sid, span in span_by_id.items():
        text_by_location[sid] = _extract_span_text(span)
        input_by_location[sid] = _extract_span_input(span)
        output_by_location[sid] = _extract_span_output(span)
        tool_calls_by_location[sid] = _extract_tool_calls_from_span(span)

    # --- Tools available (from AGENT spans) ---
    tools_available: List[str] = []
    managed_agents: List[str] = []
    for span in span_by_id.values():
        kind = _get_attr(span, "openinference.span.kind")
        if kind != "AGENT":
            continue
        tn_raw = _get_attr(span, "smolagents.tools_names")
        if isinstance(tn_raw, str):
            try:
                names = json.loads(tn_raw.replace("'", '"'))
                if isinstance(names, list):
                    tools_available = [str(n) for n in names if n]
            except Exception:
                pass
        for i in range(10):
            ma = _get_attr(span, f"smolagents.managed_agents.{i}.name")
            if ma:
                managed_agents.append(str(ma))
            else:
                break
        if tools_available:
            break

    # --- Error annotations ---
    errors: List[dict] = []
    if annotations_path and os.path.isfile(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
        for i, err in enumerate(ann_data.get("errors") or []):
            span_id = err.get("location") or ""
            step_mapping = map_annotation_to_step(parsed, span_id) if span_id else None
            rec = build_error_annotation_output(trace_id, span_id, err, step_mapping)
            rec["error_id"] = f"{trace_id}|{span_id}|{err.get('category', 'unknown')}|{i}"
            rec["impact"] = err.get("impact", "")
            errors.append(rec)

    return TraceObj(
        trace_id=trace_id,
        span_by_id=span_by_id,
        parent_of=parent_of,
        text_by_location=text_by_location,
        input_by_location=input_by_location,
        output_by_location=output_by_location,
        tool_calls_by_location=tool_calls_by_location,
        tools_available=tools_available + managed_agents,
        managed_agents=managed_agents,
        errors=errors,
        step_spans=step_spans,
        raw_trace=trace_data,
    )


# ---------------------------------------------------------------------------
# Snippet extraction with optional neighbour expansion
# ---------------------------------------------------------------------------


def get_span_io(trace_obj: TraceObj, location: str) -> dict:
    """
    Return exact input and output for an annotated span (for precise input_context patches).
    Returns:
      {
        "input": { "kind", "messages", "input_value", "input_value_raw" },
        "output": { "output_text", "output_value_raw" },
      }
    """
    return {
        "input": trace_obj.input_by_location.get(location) or {"kind": "", "messages": [], "input_value": "", "input_value_raw": ""},
        "output": trace_obj.output_by_location.get(location) or {"output_text": "", "output_value_raw": ""},
    }


def get_expanded_snippet(
    trace_obj: TraceObj,
    location: str,
    window: int = 0,
) -> str:
    """
    Return the text snippet for a given span location.

    window=0  : just this span's text
    window>=1 : this span + up to `window` sibling spans before and after
                (siblings = other children of the same parent span)

    The snippet is the anchor for apply_patch's string surgery.
    """
    base = trace_obj.text_by_location.get(location, "")
    if window == 0 or not location:
        return base

    parent_id = trace_obj.parent_of.get(location)
    if not parent_id:
        return base

    # Siblings = all children of the same parent
    siblings = [
        sid for sid, pid in trace_obj.parent_of.items() if pid == parent_id
    ]

    def _ts(sid: str) -> str:
        sp = trace_obj.span_by_id.get(sid) or {}
        return str(sp.get("timestamp") or sp.get("start_time") or "")

    siblings.sort(key=_ts)
    try:
        idx = siblings.index(location)
    except ValueError:
        return base

    lo = max(0, idx - window)
    hi = min(len(siblings), idx + window + 1)
    parts = [
        trace_obj.text_by_location.get(s, "")
        for s in siblings[lo:hi]
        if trace_obj.text_by_location.get(s, "").strip()
    ]
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience: iterate all (trace_path, ann_path) pairs in a directory
# ---------------------------------------------------------------------------


def iter_trail_traces(
    trace_dir: str,
    annotations_dir: str,
    trace_ids: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
):
    """
    Yield (trace_path, ann_path) tuples for traces that have a matching annotation file.
    If trace_ids is None, discovers all matching pairs.
    """
    if trace_ids is None:
        trace_ids = []
        for fname in sorted(os.listdir(trace_dir)):
            if fname.endswith(".json"):
                tid = os.path.splitext(fname)[0]
                ann = os.path.join(annotations_dir, fname)
                if os.path.isfile(ann):
                    trace_ids.append(tid)
    if max_traces is not None:
        trace_ids = trace_ids[:max_traces]
    for tid in trace_ids:
        tp = os.path.join(trace_dir, tid + ".json")
        ap = os.path.join(annotations_dir, tid + ".json")
        if os.path.isfile(tp):
            yield tp, ap if os.path.isfile(ap) else None
