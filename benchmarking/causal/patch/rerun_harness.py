#!/usr/bin/env python3
"""
Step 5: Counterfactual rerun harness.

Applies the validated patch to span t_A and reruns the agent forward by calling
the LLM for up to max_steps_after turns, replaying original trace tool results.

One rerun per A-instance (not per edge). Results are shared across all EdgePairs
that reference the same error_id.

Message format:
  TRAIL traces use smolagents roles "tool-call" / "tool-response".
  These are converted to OpenAI API format (assistant+tool_calls / tool+tool_call_id)
  before sending to the LLM.

Patch application:
  replace_span_output — inject patch_payload as the assistant message at t_A,
                        add the original tool results for any tool calls in the patch,
                        then call the LLM for subsequent steps.
  replace_span_input  — replace the last user message at t_A with patch_payload,
                        call the LLM to re-generate t_A's response,
                        then continue forward.

Tool results:
  Original tool results from the trace are replayed in order (keyed by tool name).
  For tool calls where no matching result exists, a placeholder is inserted.

Loop termination:
  The continuation loop runs for up to max_steps_after LLM calls. Each call is one
  step, whether it produces a tool call or plain text (planning/thinking spans are
  valid intermediate steps and do not stop the loop). The loop stops early only when
  the agent calls the "final_answer" tool.

rerun_status:
  "live_rerun_success"    — LLM rerun completed; rerun_suffix_spans is populated.
  "rerun_missing_suffix"  — trace load failed or no message history at t_A.

Input:  patch_results.jsonl  (one record per unique A-instance)
Output: rerun_results.jsonl  (one record per unique A-instance)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from trail_io import load_trail_trace


# ---------------------------------------------------------------------------
# rerun_status constants
# ---------------------------------------------------------------------------

LIVE_RERUN_SUCCESS = "live_rerun_success"
RERUN_MISSING_SUFFIX = "rerun_missing_suffix"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RerunResult:
    trace_id: str
    error_id: str
    a_location: str              # span_id of intervention point t_A
    patch_side: str
    patch_payload: str
    rerun_status: str            # LIVE_RERUN_SUCCESS | RERUN_MISSING_SUFFIX
    rerun_success: bool          # True for live_rerun_success
    rerun_error: str
    original_suffix_spans: list  # span texts from t_A onward in original trace (baseline)
    rerun_suffix_spans: list     # LLM-generated texts from t_A onward (counterfactual)


# ---------------------------------------------------------------------------
# Smolagents → OpenAI message conversion
# ---------------------------------------------------------------------------

def _parse_tool_call_content(content: str) -> List[dict]:
    """
    Parse smolagents 'tool-call' message content:
      "Calling tools:\n[{'id': ..., 'type': 'function', 'function': {...}}]"
    Returns OpenAI-format tool_calls list.
    """
    content = content.strip()
    for prefix in ("Calling tools:\n", "Calling tools:"):
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            break
    try:
        calls = ast.literal_eval(content)
        if not isinstance(calls, list):
            return []
    except Exception:
        return []

    result = []
    for tc in calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"
        result.append({
            "id": tc.get("id", f"call_{len(result)}"),
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": args_str,
            },
        })
    return result


def _parse_tool_response_content(content: str) -> Tuple[str, str]:
    """
    Parse smolagents 'tool-response' message content:
      "Call id: call_xxx\nObservation:\n...result..."
    or
      "Call id: call_xxx\nError:\n...error..."
    Returns (call_id, observation_text).
    """
    lines = content.strip().split("\n")
    call_id = ""
    obs_start = 1  # default: everything after the first line
    for i, line in enumerate(lines):
        if line.startswith("Call id:"):
            call_id = line[len("Call id:"):].strip()
        elif line.strip() in ("Observation:", "Error:") or \
                line.startswith("Observation:") or line.startswith("Error:"):
            obs_start = i + 1
            break
    observation = "\n".join(lines[obs_start:]).strip() or content
    return call_id, observation


def _to_openai_messages(messages: List[dict]) -> List[dict]:
    """
    Convert smolagents message list (tool-call / tool-response roles) to
    OpenAI API format (assistant + tool_calls / tool + tool_call_id).
    """
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""

        if role == "tool-call":
            tool_calls = _parse_tool_call_content(content)
            if tool_calls:
                result.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "tool-response":
            call_id, observation = _parse_tool_response_content(content)
            result.append({
                "role": "tool",
                "tool_call_id": call_id or "call_unknown",
                "content": observation,
            })

        else:
            # system, user, assistant — pass through
            result.append({"role": role, "content": content})

    return result


# ---------------------------------------------------------------------------
# Patch payload parsing
# ---------------------------------------------------------------------------

def _normalize_tool_calls(tool_calls: list) -> List[dict]:
    """Ensure tool_call arguments are JSON strings (required by OpenAI API)."""
    result = []
    for tc in tool_calls:
        tc = dict(tc)
        if "function" in tc:
            fn = dict(tc["function"])
            args = fn.get("arguments")
            if isinstance(args, dict):
                fn["arguments"] = json.dumps(args)
            elif args is None:
                fn["arguments"] = "{}"
            tc["function"] = fn
        tc.pop("description", None)  # remove non-standard fields
        result.append(tc)
    return result


def _parse_assistant_message(patch_payload: str) -> dict:
    """
    Parse patch_payload (the corrected span output) as an OpenAI assistant message.
    patch_payload may be:
      - JSON string of {"role": "assistant", "content": ..., "tool_calls": [...]}
      - Plain text (formatting fix, planning text, etc.)
    """
    try:
        obj = json.loads(patch_payload)
        if isinstance(obj, dict) and obj.get("role") == "assistant":
            if obj.get("tool_calls"):
                obj["tool_calls"] = _normalize_tool_calls(obj["tool_calls"])
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    return {"role": "assistant", "content": patch_payload}


# ---------------------------------------------------------------------------
# Tool result queue from original trace
# ---------------------------------------------------------------------------

def _build_tool_result_queue(trace_obj, a_location: str) -> Dict[str, List[str]]:
    """
    Build a per-tool-name queue of result texts for all TOOL spans that appear
    after t_A in the original trace (sorted by timestamp).
    Used to replay tool results during the counterfactual LLM loop.
    """
    ta_ts = ""
    if a_location in trace_obj.span_by_id:
        sp = trace_obj.span_by_id[a_location]
        ta_ts = sp.get("timestamp") or sp.get("start_time") or ""

    timed: List[Tuple[str, str, str]] = []  # (ts, tool_name, result_text)
    for sid, span in trace_obj.span_by_id.items():
        attrs = span.get("span_attributes") or span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        if attrs.get("openinference.span.kind") != "TOOL":
            continue
        ts = span.get("timestamp") or span.get("start_time") or ""
        if ts <= ta_ts:
            continue
        tool_name = attrs.get("tool.name") or span.get("span_name", "")
        out = trace_obj.output_by_location.get(sid, {})
        result_text = (out.get("output_text") or out.get("output_value_raw") or "").strip()
        timed.append((ts, tool_name, result_text))

    timed.sort(key=lambda x: x[0])
    queue: Dict[str, List[str]] = defaultdict(list)
    for _, tool_name, result_text in timed:
        if tool_name:
            queue[tool_name].append(result_text[:3000])
    return dict(queue)


# ---------------------------------------------------------------------------
# LLM call with full message list
# ---------------------------------------------------------------------------

def _call_messages(
    model: str,
    messages: List[dict],
    max_tokens: int = 1024,
) -> dict:
    """
    Call litellm with a full OpenAI-format message list.
    Returns the assistant message dict (role, content, tool_calls if any).
    """
    try:
        from litellm import completion, RateLimitError
    except ImportError:
        raise RuntimeError("litellm not installed. pip install litellm")

    params: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if any(x in model for x in ("o1", "o3", "o4")):
        params["reasoning_effort"] = "medium"
        params.pop("temperature", None)

    def _do_call():
        resp = completion(**params)
        msg = resp.choices[0].message
        result = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            result["content"] = None
        return result

    try:
        return _do_call()
    except RateLimitError:
        import time
        time.sleep(30)
        return _do_call()


# ---------------------------------------------------------------------------
# Span ordering helper (for baseline suffix)
# ---------------------------------------------------------------------------

def _spans_after(trace_obj, a_location: str) -> List[str]:
    """Return span output texts for all spans after a_location in timestamp order."""
    timed: List[tuple] = []
    for sid, span in trace_obj.span_by_id.items():
        ts = span.get("timestamp") or span.get("start_time") or ""
        text = trace_obj.text_by_location.get(sid, "").strip()
        if text:
            timed.append((ts, sid, text))
    timed.sort(key=lambda x: x[0])
    ids_ordered = [sid for _, sid, _ in timed]
    try:
        idx = ids_ordered.index(a_location)
    except ValueError:
        idx = -1
    return [text for _, sid, text in timed[idx + 1:]]


# ---------------------------------------------------------------------------
# Real rerun
# ---------------------------------------------------------------------------

def _real_rerun(
    patch_result: dict,
    trace_obj,
    model: str,
    max_steps_after: int,
) -> RerunResult:
    """
    Apply the patch at t_A and call the LLM for up to max_steps_after turns,
    replaying original trace tool results in order.

    Each LLM call counts as one step regardless of whether it produces a tool call
    or plain text. Plain-text responses (planning/thinking spans) are valid
    intermediate steps. The loop stops early only on a "final_answer" tool call.
    """
    a_location = patch_result.get("location", "")
    patch_side = patch_result.get("patch_side", "replace_span_output")
    patch_payload = patch_result.get("patch_payload", "")

    # 1. Reconstruct message history up to t_A in OpenAI format
    raw_messages = trace_obj.input_by_location.get(a_location, {}).get("messages", [])
    if not raw_messages:
        raise ValueError(f"No message history at span {a_location}")
    history = _to_openai_messages(raw_messages)

    # 2. Build tool result queue from original trace
    tool_queue: Dict[str, List[str]] = _build_tool_result_queue(trace_obj, a_location)

    def _pop_result(tool_name: str) -> str:
        q = tool_queue.get(tool_name, [])
        if q:
            r = q.pop(0)
            tool_queue[tool_name] = q
            return r
        return f"[{tool_name}: no result in original trace]"

    def _add_tool_results(tool_calls: list) -> None:
        for tc in tool_calls:
            tc_id = tc.get("id", "call_unknown")
            tc_name = tc.get("function", {}).get("name", "")
            history.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": _pop_result(tc_name),
            })

    # 3. Apply patch at t_A
    if patch_side == "replace_span_output":
        patched_msg = _parse_assistant_message(patch_payload)
        history.append(patched_msg)
        _add_tool_results(patched_msg.get("tool_calls") or [])

    elif patch_side == "replace_span_input":
        # Replace last user message with patch_payload, then re-call LLM at t_A
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                history[i] = {"role": "user", "content": patch_payload}
                break
        ta_resp = _call_messages(model, history, max_tokens=1024)
        history.append(ta_resp)
        _add_tool_results(ta_resp.get("tool_calls") or [])

    # 4. Agent continuation loop
    # Each iteration = one LLM call = one step.
    # Plain-text responses (no tool calls) are intermediate planning steps — do not stop.
    # Stop only on final_answer tool call or when max_steps_after is exhausted.
    new_spans: List[str] = []
    for _ in range(max_steps_after):
        resp = _call_messages(model, history, max_tokens=1024)
        history.append(resp)

        span_text = resp.get("content") or json.dumps(resp.get("tool_calls") or [])
        new_spans.append(span_text)

        tc_list = resp.get("tool_calls") or []
        if tc_list:
            if any(tc.get("function", {}).get("name") == "final_answer" for tc in tc_list):
                break  # agent reached final answer
            _add_tool_results(tc_list)
        # else: planning/thinking step — continue to next iteration

    return RerunResult(
        trace_id=patch_result["trace_id"],
        error_id=patch_result.get("error_id", ""),
        a_location=a_location,
        patch_side=patch_side,
        patch_payload=patch_payload,
        rerun_status=LIVE_RERUN_SUCCESS,
        rerun_success=True,
        rerun_error="",
        original_suffix_spans=_spans_after(trace_obj, a_location),
        rerun_suffix_spans=new_spans,
    )


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------

def run_rerun(
    patch_result: dict,
    trace_dir: str,
    annotations_dir: str,
    model: str = "openai/o3-mini",
    max_steps_after: int = 12,
) -> RerunResult:
    trace_id = patch_result["trace_id"]
    trace_path = os.path.join(trace_dir, trace_id + ".json")
    ann_path = os.path.join(annotations_dir, trace_id + ".json")
    a_location = patch_result.get("location", "")

    try:
        trace_obj = load_trail_trace(
            trace_path,
            ann_path if os.path.isfile(ann_path) else None,
        )
    except Exception as e:
        return RerunResult(
            trace_id=trace_id,
            error_id=patch_result.get("error_id", ""),
            a_location=a_location,
            patch_side=patch_result.get("patch_side", ""),
            patch_payload=patch_result.get("patch_payload", ""),
            rerun_status=RERUN_MISSING_SUFFIX,
            rerun_success=False,
            rerun_error=f"trace load failed: {e}",
            original_suffix_spans=[],
            rerun_suffix_spans=[],
        )

    try:
        return _real_rerun(patch_result, trace_obj, model=model,
                           max_steps_after=max_steps_after)
    except Exception as e:
        return RerunResult(
            trace_id=trace_id,
            error_id=patch_result.get("error_id", ""),
            a_location=a_location,
            patch_side=patch_result.get("patch_side", ""),
            patch_payload=patch_result.get("patch_payload", ""),
            rerun_status=RERUN_MISSING_SUFFIX,
            rerun_success=False,
            rerun_error=str(e),
            original_suffix_spans=_spans_after(trace_obj, a_location),
            rerun_suffix_spans=[],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply patches and rerun agent via LLM continuation."
    )
    parser.add_argument("--patch_results",
                        default="outputs/interventions/patch_results.jsonl")
    parser.add_argument("--trace_dir", default="data/GAIA")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/o3-mini",
                        help="LLM for rerun continuation (should match original trace model)")
    parser.add_argument("--max_steps_after", type=int, default=12,
                        help="Max LLM turns to generate after t_A (covers ~94%% of A→B distances)")
    args = parser.parse_args()

    with open(args.patch_results, "r", encoding="utf-8") as f:
        patch_results = [json.loads(l) for l in f if l.strip()]

    to_rerun = [p for p in patch_results if p.get("postcheck_passed")]
    print(f"Rerunning {len(to_rerun)} / {len(patch_results)} patches (postcheck passed)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "rerun_results.jsonl")

    from collections import Counter
    counts: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for pr in to_rerun:
            result = run_rerun(
                pr, args.trace_dir, args.annotations_dir,
                model=args.model, max_steps_after=args.max_steps_after,
            )
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            counts[result.rerun_status] += 1
            n_new = len(result.rerun_suffix_spans)
            print(f"  [{result.rerun_status}] {result.trace_id[:8]} "
                  f"err={result.error_id[-20:]} new_spans={n_new}")

    print(f"\nWrote {out_path}.")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
