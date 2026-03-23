#!/usr/bin/env python3
"""
Parse trace JSON to a desired level of span (step level) and map annotations to steps.

Implements the concrete parsing pipeline:
  Step 0 — Load one trace JSON (trace_data['spans'] = list of main spans; each span has span_id, timestamps, child_spans)
  Step 1 — Flatten spans + build parent pointers (span_by_id, parent_of)
  Step 2 — Identify root "main" span (single "main" or earliest-start top-level)
  Step 3 — Define step spans = main.child_spans (level-1 only), assign step indices by start_time
  Step 4 — Map each annotated span_id to a step (walk parent_of until level-1 under main)
  Step 5 — Store for each error annotation: trace_id, annotated_span_id, annotated_span_kind, step_span_id, step_index, evidence/error type

Compatible with GAIA / OpenInference-style trace JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Step 0: Load trace (caller loads; we accept trace_data dict)
# ---------------------------------------------------------------------------


def _parse_timestamp(span: Dict[str, Any]) -> str:
    """Return sortable timestamp string from span (start_time or timestamp)."""
    ts = span.get("start_time") or span.get("timestamp") or ""
    if not ts:
        return datetime.min.isoformat()
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        return datetime.min.isoformat()


def _span_name(span: Dict[str, Any]) -> str:
    """Return span name (span_name or name)."""
    return str(span.get("span_name") or span.get("name") or "")


# ---------------------------------------------------------------------------
# Step 1: Flatten spans + build parent pointers
# ---------------------------------------------------------------------------


def flatten_spans_and_parents(trace_data: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, str], set]:
    """
    DFS over trace_data['spans']; build span_by_id and parent_of.
    Top-level spans have no parent (not in parent_of).

    Returns:
        span_by_id: span_id -> span object
        parent_of: child_span_id -> parent_span_id (only for non-top-level)
        top_level_ids: set of span_ids that are roots (in trace_data['spans'])
    """
    span_by_id: Dict[str, Dict[str, Any]] = {}
    parent_of: Dict[str, str] = {}
    top_level_ids: set = set()

    def visit(span: Dict[str, Any], parent_id: Optional[str]) -> None:
        sid = span.get("span_id")
        if sid is None:
            return
        span_by_id[sid] = span
        if parent_id is not None:
            parent_of[sid] = parent_id
        for child in span.get("child_spans") or []:
            visit(child, sid)

    for root in trace_data.get("spans") or []:
        sid = root.get("span_id")
        if sid is not None:
            top_level_ids.add(sid)
            visit(root, None)

    return span_by_id, parent_of, top_level_ids


# ---------------------------------------------------------------------------
# Step 2: Identify root "main" span
# ---------------------------------------------------------------------------


def identify_main_span(
    trace_data: Dict[str, Any],
    span_by_id: Dict[str, Dict],
    top_level_ids: set,
) -> Optional[Dict[str, Any]]:
    """
    If there is exactly one top-level span named "main", use it.
    If multiple top-level spans exist, choose one named "main"; otherwise earliest-start top-level.
    """
    roots = [span_by_id[sid] for sid in top_level_ids if sid in span_by_id]
    if not roots:
        return None

    named_main = [s for s in roots if _span_name(s) == "main"]
    if len(roots) == 1:
        return roots[0]
    if named_main:
        # Prefer single "main"; if multiple "main", take earliest start
        if len(named_main) == 1:
            return named_main[0]
        named_main.sort(key=lambda s: (_parse_timestamp(s), s.get("span_id", "")))
        return named_main[0]
    # No "main" -> earliest-start top-level
    roots.sort(key=lambda s: (_parse_timestamp(s), s.get("span_id", "")))
    return roots[0]


# ---------------------------------------------------------------------------
# Step 3: Define step spans (level-1 children of main)
# ---------------------------------------------------------------------------


def build_step_spans(
    main_span: Dict[str, Any],
    order_by_start_time: bool = True,
) -> List[Dict[str, Any]]:
    """
    step_spans = main.child_spans (level-1 only).
    Assign step indices by start_time (if order_by_start_time) or original order.
    Returns list of step span dicts in step order; step_index will be 1..K in caller.
    """
    steps = list(main_span.get("child_spans") or [])
    if order_by_start_time and steps:
        steps = sorted(steps, key=lambda s: (_parse_timestamp(s), s.get("span_id", "")))
    return steps


# ---------------------------------------------------------------------------
# Step 4: Map annotated span_id to step
# ---------------------------------------------------------------------------


def annotated_span_to_step(
    annotated_span_id: str,
    main_span_id: str,
    parent_of: Dict[str, str],
    top_level_ids: set,
    step_span_ids: List[str],
) -> Optional[Tuple[str, int]]:
    """
    Walk parent_of from annotated_span_id until we reach a level-1 child of main.
    Return (step_span_id, step_index) or None if not under main.
    step_index is 1-based.
    """
    cur = annotated_span_id
    if cur not in parent_of and cur not in step_span_ids:
        # Might be top-level or unknown
        if cur in step_span_ids:
            idx = step_span_ids.index(cur) + 1
            return (cur, idx)
        return None

    while cur in parent_of:
        parent_id = parent_of[cur]
        if parent_id == main_span_id:
            # cur is level-1 child = step span
            if cur in step_span_ids:
                step_index = step_span_ids.index(cur) + 1
                return (cur, step_index)
            return None
        cur = parent_id

    # cur is now top-level (not under main)
    if cur == main_span_id:
        # Annotation on main span itself: assign to first step by convention or leave unset
        if step_span_ids:
            return (step_span_ids[0], 1)
        return None
    return None


def get_span_kind(span: Dict[str, Any]) -> Optional[str]:
    """OpenInference span kind from span_attributes or attributes."""
    attrs = span.get("span_attributes") or span.get("attributes")
    if isinstance(attrs, dict):
        return attrs.get("openinference.span.kind")
    return None


# ---------------------------------------------------------------------------
# Step 5: Full pipeline — parse one trace and (optionally) map annotations
# ---------------------------------------------------------------------------


def parse_trace_to_step_level(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Steps 0–3 on one trace. Caller provides trace_data (from JSON).

    Returns a structure:
      trace_id
      main_span_id
      main_span (the root span object, or None)
      step_spans: [ { span, step_index 1..K }, ... ]
      span_by_id, parent_of, top_level_ids (for later annotation mapping)
    """
    trace_id = trace_data.get("trace_id") or ""
    span_by_id, parent_of, top_level_ids = flatten_spans_and_parents(trace_data)
    main_span = identify_main_span(trace_data, span_by_id, top_level_ids)
    if main_span is None:
        return {
            "trace_id": trace_id,
            "main_span_id": None,
            "main_span": None,
            "step_spans": [],
            "step_span_ids": [],
            "span_by_id": span_by_id,
            "parent_of": parent_of,
            "top_level_ids": top_level_ids,
        }

    step_spans = build_step_spans(main_span, order_by_start_time=True)
    main_span_id = main_span.get("span_id")
    step_span_ids = [s.get("span_id") for s in step_spans if s.get("span_id")]

    return {
        "trace_id": trace_id,
        "main_span_id": main_span_id,
        "main_span": main_span,
        "step_spans": [{"span": s, "step_index": i + 1} for i, s in enumerate(step_spans)],
        "step_span_ids": step_span_ids,
        "span_by_id": span_by_id,
        "parent_of": parent_of,
        "top_level_ids": top_level_ids,
    }


def map_annotation_to_step(
    parsed: Dict[str, Any],
    annotated_span_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Given result of parse_trace_to_step_level and an annotated span_id,
    return the step mapping: step_span_id, step_index, and annotated span kind.
    """
    main_span_id = parsed.get("main_span_id")
    if main_span_id is None:
        return None
    parent_of = parsed.get("parent_of") or {}
    top_level_ids = parsed.get("top_level_ids") or set()
    step_span_ids = parsed.get("step_span_ids") or []
    span_by_id = parsed.get("span_by_id") or {}

    res = annotated_span_to_step(
        annotated_span_id,
        main_span_id,
        parent_of,
        top_level_ids,
        step_span_ids,
    )
    if res is None:
        return None
    step_span_id, step_index = res
    span_obj = span_by_id.get(annotated_span_id, {})
    kind = get_span_kind(span_obj)
    return {
        "step_span_id": step_span_id,
        "step_index": step_index,
        "annotated_span_kind": kind,
    }


def build_error_annotation_output(
    trace_id: str,
    annotated_span_id: str,
    annotation: Dict[str, Any],
    step_mapping: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the stored record for one error annotation (Step 5).
    annotation may have: category, location, description, evidence, impact, etc.
    """
    span_kind = step_mapping.get("annotated_span_kind") if step_mapping else None
    return {
        "trace_id": trace_id,
        "annotated_span_id": annotated_span_id,
        "annotated_span_kind": span_kind,
        "step_span_id": step_mapping.get("step_span_id") if step_mapping else None,
        "step_index": step_mapping.get("step_index") if step_mapping else None,
        "error_type": annotation.get("category"),
        "description": annotation.get("description"),
        "evidence": annotation.get("evidence"),
        "impact": annotation.get("impact"),
    }


# ---------------------------------------------------------------------------
# Batch: trace file + annotations list
# ---------------------------------------------------------------------------


def process_trace_with_annotations(
    trace_path: str,
    trace_id: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Load trace JSON, run parse_trace_to_step_level, then map each annotation
    (with 'location' or 'annotated_span_id' as span_id) to step. Return
    parsed structure plus list of stored error records.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    tid = trace_id or trace_data.get("trace_id") or os.path.splitext(os.path.basename(trace_path))[0]
    trace_data["trace_id"] = tid

    parsed = parse_trace_to_step_level(trace_data)
    parsed["trace_id"] = tid

    error_records: List[Dict[str, Any]] = []
    if annotations:
        for ann in annotations:
            annotation_body = ann.get("annotation", ann)
            span_id = annotation_body.get("location") or annotation_body.get("annotated_span_id") or ann.get("location")
            if not span_id:
                continue
            step_mapping = map_annotation_to_step(parsed, span_id)
            rec = build_error_annotation_output(tid, span_id, annotation_body, step_mapping)
            error_records.append(rec)
    parsed["error_annotations"] = error_records
    return parsed


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Parse trace to step-level and optionally map annotations")
    parser.add_argument("--trace_file", required=True, help="Path to trace JSON")
    parser.add_argument("--annotations_file", default=None, help="Optional JSON file with list of annotations (or report with traces[].annotated_errors)")
    parser.add_argument("--trace_id", default=None, help="Override trace_id (default: from trace or filename)")
    parser.add_argument("--out", default=None, help="Write result JSON here")
    args = parser.parse_args()

    # Resolve trace_id from file if not overridden
    with open(args.trace_file, "r", encoding="utf-8") as f:
        _trace_data = json.load(f)
    trace_id_from_file = _trace_data.get("trace_id") or os.path.splitext(os.path.basename(args.trace_file))[0]
    trace_id = args.trace_id or trace_id_from_file

    annotations = None
    if args.annotations_file and os.path.isfile(args.annotations_file):
        with open(args.annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "traces" in data:
            for t in data["traces"]:
                if t.get("trace_id") == trace_id:
                    annotations = [e for e in t.get("annotated_errors") or []]
                    break
            if annotations is None and data["traces"]:
                annotations = (data["traces"][0].get("annotated_errors") or [])
        elif isinstance(data, list):
            annotations = data
        else:
            annotations = data.get("annotated_errors") or []

    result = process_trace_with_annotations(args.trace_file, trace_id=trace_id, annotations=annotations)
    num_steps = len(result.get("step_spans") or [])
    span_by_id = result.get("span_by_id") or {}
    total_spans = len(span_by_id)
    step_level_spans = 1 + num_steps  # main + level-1 only (GAIA / Table 5 method)
    # Human-readable summary (exclude large raw objects)
    out = {
        "trace_id": result["trace_id"],
        "main_span_id": result.get("main_span_id"),
        "num_steps": num_steps,
        "total_spans_in_trace": total_spans,
        "step_level_spans": step_level_spans,
        "step_spans": [{"step_index": x["step_index"], "span_id": x["span"].get("span_id"), "span_name": _span_name(x["span"])} for x in (result.get("step_spans") or [])],
        "error_annotations": result.get("error_annotations"),
    }
    print(json.dumps(out, indent=2))
    print(f"\nSpans in trace (GAIA): {total_spans} total ({step_level_spans} at step level: 1 main + {num_steps} steps)", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote full result to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
