#!/usr/bin/env python3
"""
Trace replay for counterfactual intervention runs.

Extract from a GAIA trace:
- Ordered execution steps (LLM and TOOL spans in DFS/timestamp order)
- Conversation history before a given span (for re-run prefix)
- Tool outputs from original trace (for replay; keep observations fixed)

Used by rerun_intervention.py to build single-intervention runs:
  do(A_i ≈ 0): prefix = history before t_Ai, at t_Ai inject patched content,
  suffix = replayed tool outputs + re-generated agent responses.
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ordered steps (LLM + TOOL) in execution order — by timestamp, not DFS
# ---------------------------------------------------------------------------

# OTEL/GAIA: execution order should use start time, not tree DFS.
# Use start_time_unix_nano, or timestamp (ISO), or start_time; fallback to DFS order if missing.


def _get_attr(span: dict, key: str) -> Any:
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    return attrs.get(key) if isinstance(attrs, dict) else None


def _is_llm_span(span: dict) -> bool:
    """GAIA: LiteLLMModel.__call__ or openinference.span.kind == LLM."""
    if span.get("span_name") == "LiteLLMModel.__call__":
        return True
    return _get_attr(span, "openinference.span.kind") == "LLM"


def _is_tool_span(span: dict) -> bool:
    return _get_attr(span, "openinference.span.kind") == "TOOL"


def _span_start_time_sort_key(span: dict) -> Tuple[float, str]:
    """
    Return (start_time_float, span_id) for stable ordering.
    Prefer start_time_unix_nano, then parse timestamp (ISO), then start_time; fallback (0, span_id).
    """
    sid = span.get("span_id") or ""
    nano = span.get("start_time_unix_nano")
    if nano is not None:
        try:
            return (float(nano) / 1e9, sid)
        except (TypeError, ValueError):
            pass
    ts = span.get("timestamp") or span.get("start_time")
    if ts:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return (dt.timestamp(), sid)
        except (ValueError, TypeError):
            pass
    return (0.0, sid)


def _collect_llm_tool_spans_flat(spans: List[dict], out: List[Dict[str, Any]]) -> None:
    """Flatten LLM and TOOL spans from tree (no order yet)."""
    for span in spans:
        sid = span.get("span_id")
        if not sid:
            _collect_llm_tool_spans_flat(span.get("child_spans") or [], out)
            continue
        if _is_llm_span(span):
            out.append({"span_id": sid, "kind": "llm", "span": span, "start_key": _span_start_time_sort_key(span)})
        elif _is_tool_span(span):
            out.append({"span_id": sid, "kind": "tool", "span": span, "start_key": _span_start_time_sort_key(span)})
        _collect_llm_tool_spans_flat(span.get("child_spans") or [], out)


def get_ordered_steps(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return list of execution steps (LLM and TOOL) in execution order.
    Ordering: by start_time_unix_nano, or timestamp/start_time (parsed), then span_id.
    Fallback: if no timestamps, use DFS order (stable but not guaranteed execution order).
    Each element: { span_id, kind, span, step_index }.
    """
    flat: List[Dict[str, Any]] = []
    for root in trace_data.get("spans") or []:
        _collect_llm_tool_spans_flat([root], flat)
    flat.sort(key=lambda x: x.get("start_key", (0.0, x.get("span_id", ""))))
    return [
        {"span_id": s["span_id"], "kind": s["kind"], "span": s["span"], "step_index": i}
        for i, s in enumerate(flat)
    ]


def get_step_index_for_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> Optional[int]:
    """Return the step_index of the first step whose span_id equals span_id, or None."""
    steps = get_ordered_steps(trace_data)
    for s in steps:
        if s.get("span_id") == span_id:
            return s.get("step_index")
    return None


def get_steps_after_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> List[Dict[str, Any]]:
    """Return steps that come strictly after span_id in execution order (for suffix re-execution)."""
    steps = get_ordered_steps(trace_data)
    found = False
    out: List[Dict[str, Any]] = []
    for s in steps:
        if s.get("span_id") == span_id:
            found = True
            continue
        if found:
            out.append(s)
    return out


def get_run_config_from_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract run configuration from the trace: same model and settings as original GAIA run.
    Returns dict with model_id (required), temperature (default 0), and optional extra params.
    Raises ValueError if no LLM span (hence no model) is found.
    """
    steps = get_ordered_steps(trace_data)
    for s in steps:
        if s.get("kind") != "llm":
            continue
        span = s.get("span") or {}
        attrs = span.get("span_attributes") or span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        model_id = attrs.get("llm.model_name")
        if not model_id:
            continue
        # Temperature may be in trace; GAIA replication uses 0
        temperature = 0
        if "llm.temperature" in attrs:
            try:
                temperature = int(attrs["llm.temperature"])
            except (TypeError, ValueError):
                pass
        return {
            "model_id": str(model_id),
            "temperature": temperature,
        }
    raise ValueError("Trace has no LLM span (LiteLLMModel.__call__ with llm.model_name); cannot re-run with same model.")


# ---------------------------------------------------------------------------
# Prefix at intervention: exact messages for that LLM call (no duplication)
# ---------------------------------------------------------------------------
#
# For Option 2 rerun you want the exact prefix the model saw at the intervention
# point — i.e. the input_messages of the LLM step at (or immediately before) that
# span. Use get_llm_input_messages_for_span(); do not concatenate per-call inputs
# (that would duplicate prior turns).


def _llm_messages_from_span(span: dict) -> List[Dict[str, str]]:
    """Extract messages list from LiteLLMModel.__call__ span (input_messages)."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return []
    messages = []
    i = 0
    while True:
        content_key = f"llm.input_messages.{i}.message.content"
        role_key = f"llm.input_messages.{i}.message.role"
        if content_key not in attrs:
            break
        content = attrs.get(content_key)
        role = attrs.get(role_key, "user")
        messages.append({"role": str(role), "content": content or ""})
        i += 1
    return messages


def _llm_output_from_span(span: dict) -> str:
    """Extract assistant output content from LLM span."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return ""
    return attrs.get("llm.output_messages.0.message.content") or ""


def _tool_output_from_span(span: dict) -> str:
    """Extract tool result text from TOOL span (output.value or function.output)."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if isinstance(attrs, dict) and attrs.get("output.value"):
        return str(attrs["output.value"])[:16000]
    for log in span.get("logs") or []:
        body = log.get("body") or {}
        if isinstance(body, dict) and body.get("function.output") is not None:
            return str(body["function.output"])[:16000]
    return ""


def _tool_name_and_args_from_span(span: dict) -> Tuple[str, dict]:
    """Tool name and args from TOOL span."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return "", {}
    name = attrs.get("tool.name") or span.get("span_name") or ""
    if name == "VisitTool":
        name = "visit_page"
    elif name == "FinderTool":
        name = "find_on_page_ctrl_f"
    inv = attrs.get("input.value")
    args = {}
    if isinstance(inv, str):
        try:
            raw = json.loads(inv)
            args = raw.get("kwargs", raw) if isinstance(raw, dict) else {}
        except json.JSONDecodeError:
            pass
    return str(name), args


def get_llm_input_messages_for_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> List[Dict[str, str]]:
    """
    Return the exact message list that was fed to the LLM call at the intervention point.

    Finds the LLM span with span_id, or the nearest LLM span immediately before span_id
    in (timestamp) execution order. Returns that span's llm.input_messages.* as
    [{"role": ..., "content": ...}, ...]. No duplication: this is the single prefix
    context the model saw for that call. Use this for Option 2 rerun (prefix + patched
    assistant + observation tape).
    """
    steps = get_ordered_steps(trace_data)
    # Which step is the intervention? If span_id is an LLM span, use it; else use last LLM before it
    target_llm_step: Optional[Dict[str, Any]] = None
    for step in steps:
        if step.get("span_id") == span_id and step.get("kind") == "llm":
            target_llm_step = step
            break
        if step.get("kind") == "llm":
            target_llm_step = step
        if step.get("span_id") == span_id:
            # span_id is not LLM; use last LLM we saw
            break
    if not target_llm_step:
        return []
    span = target_llm_step.get("span") or {}
    return _llm_messages_from_span(span)


def get_conversation_before_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> List[Dict[str, str]]:
    """
    Build conversation (messages) before span_id by walking steps and appending
    each LLM's input_messages + output and each TOOL's observation.

    Note: this duplicates prior turns (each LLM call's input_messages contains
    full history). For Option 2 rerun prefix use get_llm_input_messages_for_span()
    instead, then append the patched assistant turn once.
    """
    steps = get_ordered_steps(trace_data)
    messages: List[Dict[str, str]] = []
    for step in steps:
        if step["span_id"] == span_id:
            break
        kind = step.get("kind")
        span = step.get("span") or {}
        if kind == "llm":
            for m in _llm_messages_from_span(span):
                messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            out = _llm_output_from_span(span)
            if out:
                messages.append({"role": "assistant", "content": out})
        elif kind == "tool":
            out = _tool_output_from_span(span)
            name, _ = _tool_name_and_args_from_span(span)
            messages.append({
                "role": "user",
                "content": f"Observation (tool {name}): {out}" if out else f"Observation (tool {name}): (no output)",
            })
    return messages


def get_tool_outputs_after_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> List[Dict[str, Any]]:
    """
    Return tool outputs from the original trace after the given span_id (for replay).
    Each element: { tool, args, output } so the re-run can inject these as fixed observations.
    """
    steps = get_ordered_steps(trace_data)
    found = False
    result: List[Dict[str, Any]] = []
    for step in steps:
        if step["span_id"] == span_id:
            found = True
            continue
        if not found or step.get("kind") != "tool":
            continue
        span = step.get("span") or {}
        name, args = _tool_name_and_args_from_span(span)
        out = _tool_output_from_span(span)
        result.append({"tool": name, "args": args, "output": out})
    return result


def get_full_ordered_tool_outputs(
    trace_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """All tool steps in order: [{ tool, args, output }, ...]. Useful for full suffix replay."""
    steps = get_ordered_steps(trace_data)
    result = []
    for step in steps:
        if step.get("kind") != "tool":
            continue
        span = step.get("span") or {}
        name, args = _tool_name_and_args_from_span(span)
        result.append({"tool": name, "args": args, "output": _tool_output_from_span(span)})
    return result


# ---------------------------------------------------------------------------
# Readable output: execution-order steps with content (for human inspection)
# ---------------------------------------------------------------------------

def build_readable_steps(
    trace_data: Dict[str, Any],
    intervention_span_id: Optional[str] = None,
    max_content_len: int = 8000,
) -> List[Dict[str, Any]]:
    """
    Build a human-readable list of steps (LLM + TOOL) in execution order.
    Each step: { step_index, kind, span_id, content, is_patched?, tool_name? }.
    Content is truncated to max_content_len for readability.
    """
    steps = get_ordered_steps(trace_data)
    out: List[Dict[str, Any]] = []
    for step in steps:
        span = step.get("span") or {}
        sid = step.get("span_id", "")
        kind = step.get("kind", "")
        is_patched = sid == intervention_span_id
        entry: Dict[str, Any] = {
            "step_index": step.get("step_index", len(out)),
            "kind": kind,
            "span_id": sid,
            "is_patched": is_patched,
        }
        if kind == "llm":
            content = _llm_output_from_span(span)
            if len(content) > max_content_len:
                content = content[:max_content_len] + "\n... [truncated]"
            entry["content"] = content
        elif kind == "tool":
            name, args = _tool_name_and_args_from_span(span)
            tool_out = _tool_output_from_span(span)
            content = f"[{name}] " + (json.dumps(args)[:200] if args else "") + " -> " + (tool_out[:max_content_len] or "(no output)")
            if len(tool_out) > max_content_len:
                content += " ... [truncated]"
            entry["tool_name"] = name
            entry["content"] = content
        else:
            entry["content"] = ""
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Clone trace and replace span content (for static patch write-back)
# ---------------------------------------------------------------------------


def _find_span_in_tree(spans: List[dict], span_id: str) -> Optional[dict]:
    """Return the span dict with span_id in the nested tree (first match)."""
    for span in spans:
        if span.get("span_id") == span_id:
            return span
        found = _find_span_in_tree(span.get("child_spans") or [], span_id)
        if found is not None:
            return found
    return None


def _set_span_output(span: dict, new_content: str) -> None:
    """
    Set the main output content for this span so the trace reflects patched output.
    Modifies span in place. Uses span kind: LLM -> llm.output_messages.0.message.content,
    TOOL -> output.value (and function.output in logs if present), else output.value.
    """
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    attrs = copy.deepcopy(attrs)
    kind = attrs.get("openinference.span.kind")
    if kind == "LLM" or span.get("span_name") == "LiteLLMModel.__call__":
        attrs["llm.output_messages.0.message.content"] = new_content
    attrs["output.value"] = new_content
    span["span_attributes"] = attrs
    # If span has logs with function.output, set first one for TOOL spans
    if kind == "TOOL":
        for log in span.get("logs") or []:
            body = log.get("body")
            if isinstance(body, dict) and "function.output" in body:
                log["body"] = {**body, "function.output": new_content}
                break


def replace_span_output_in_trace(
    trace_data: Dict[str, Any],
    span_id: str,
    new_content: str,
) -> None:
    """Replace the output content of the span with span_id in trace_data (mutates in place)."""
    for root in trace_data.get("spans") or []:
        span = _find_span_in_tree([root], span_id)
        if span is not None:
            _set_span_output(span, new_content)
            return
    raise KeyError(f"span_id {span_id!r} not found in trace")


def clone_trace_and_patch_span(
    trace_data: Dict[str, Any],
    span_id: str,
    patched_text: str,
) -> Dict[str, Any]:
    """
    Deep-clone the trace and replace the content of the span with span_id by patched_text.
    Used for single-intervention static patch: do(A≈0) = same trace with that span's output replaced.
    Returns new trace dict (same structure as GAIA original).
    """
    cloned = copy.deepcopy(trace_data)
    for root in cloned.get("spans") or []:
        span = _find_span_in_tree([root], span_id)
        if span is not None:
            _set_span_output(span, patched_text)
            break
    return cloned


def format_tool_observation_message(step: Dict[str, Any]) -> Dict[str, str]:
    """Build the user message for a tool step (observation) to append to conversation."""
    span = step.get("span") or {}
    out = _tool_output_from_span(span)
    name, _ = _tool_name_and_args_from_span(span)
    content = f"Observation (tool {name}): {out}" if out else f"Observation (tool {name}): (no output)"
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Truncate trace after a span (for patch validity checks only; do NOT use for causal rerun)
# ---------------------------------------------------------------------------
# Truncation removes everything after the span, so later errors/sequence cannot be
# evaluated. Use only when you need "trace ends at patch" for validation. For
# causal evaluation, save the full trace with the patched span only (no truncation).


def _dfs_span_ids(spans: List[dict]) -> List[str]:
    """Return all span_ids in DFS order (root to leaves)."""
    out: List[str] = []
    for span in spans:
        sid = span.get("span_id")
        if sid:
            out.append(sid)
        out.extend(_dfs_span_ids(span.get("child_spans") or []))
    return out


def truncate_trace_after_span(
    trace_data: Dict[str, Any],
    span_id: str,
) -> Dict[str, Any]:
    """
    Return a copy of the trace that keeps only spans up to and including span_id in DFS order.
    Everything after the patched span is removed, so the patched conversation is the last thing
    and we avoid including any subsequent (unpatched or re-generated) content.
    """
    all_ids: List[str] = []
    for root in trace_data.get("spans") or []:
        all_ids.extend(_dfs_span_ids([root]))

    if span_id not in all_ids:
        raise KeyError(f"span_id {span_id!r} not found in trace")
    keep_ids = set(all_ids[: all_ids.index(span_id) + 1])

    cloned = copy.deepcopy(trace_data)

    def filter_child_spans(node: dict) -> None:
        children = node.get("child_spans") or []
        node["child_spans"] = [c for c in children if (c.get("span_id") or "") in keep_ids]
        for c in node["child_spans"]:
            filter_child_spans(c)

    for root in cloned.get("spans") or []:
        filter_child_spans(root)

    return cloned


def get_patched_span_content(trace_data: Dict[str, Any], span_id: str) -> Optional[str]:
    """
    Return the output content of the span with span_id (e.g. to confirm it's the patched turn).
    In a counterfactual trace written by rerun_intervention, the span with intervention_span_id
    holds the patched conversation.
    """
    span = _find_span_in_tree(trace_data.get("spans") or [], span_id)
    if not span:
        return None
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if isinstance(attrs, dict):
        return attrs.get("llm.output_messages.0.message.content") or attrs.get("output.value") or None
    return None
