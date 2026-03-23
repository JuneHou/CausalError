#!/usr/bin/env python3
"""
Step 1: Build AInstanceRecords and EdgePairs from TRAIL traces + annotations.

AInstanceRecord — one per unique (trace_id, error_id) A-instance.
EdgePair        — one per (AInstanceRecord × B-type) graph edge.

Steps 2–6 (patch, rerun, Judge A) operate on AInstanceRecord.
Step 7 (Judge B) fans out to EdgePair using the shared rerun result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from trail_io import load_trail_trace, iter_trail_traces


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AInstanceRecord:
    """One per unique (trace_id, error_id) A-instance. Used for patch → rerun → Judge A."""
    trace_id: str
    error_id: str
    a_instance: dict         # category, location, description, evidence, impact, annotation_index
    local_snippet: str       # exact span text to replace (output or input side per library)
    patch_side: str          # "replace_span_output" | "replace_span_input"
    annotated_location: str  # span_id from the annotation (may be a TOOL span)
    intervention_location: str  # span_id where the patch is actually applied (always LLM span)
    annotated_span_kind: str    # openinference.span.kind of the annotated span (e.g. "TOOL","LLM")
    intervention_span_kind: str # openinference.span.kind of the intervention span
    prefix_context: str      # system prompt text from root AGENT span
    user_requirements: str   # task description from root AGENT span
    tools_available: list    # tool names available to the agent
    suffix_window_spec: dict # {"mode": "until_end"}
    b_types: list            # B-type categories this A-instance serves (informational)


@dataclass
class EdgePair:
    """One per (AInstanceRecord × B-type) graph edge. Used for Judge B and aggregation."""
    trace_id: str
    error_id: str            # FK → AInstanceRecord
    edge: dict               # {"a": ..., "b": ...}
    b_def: dict              # {"category": b_cat}
    b_present_baseline: bool # whether B appears after t_A in original annotations
    b_onset_baseline: int    # annotation list index of first B after A (-1 if absent)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def load_graph_edges(graph_path: str) -> Tuple[Set[Tuple[str, str]], Dict[str, List[str]]]:
    """
    Load capri_graph.json.
    Returns:
      allowed_edges: set of (a_category, b_category) tuples
      a_to_bs: dict a_category -> [b_category, ...]
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    edges = [(e["a"], e["b"]) for e in g.get("edges", [])]
    allowed_edges: Set[Tuple[str, str]] = set(edges)
    a_to_bs: Dict[str, List[str]] = {}
    for a, b in edges:
        a_to_bs.setdefault(a, []).append(b)
    return allowed_edges, a_to_bs


# ---------------------------------------------------------------------------
# Trace context helpers
# ---------------------------------------------------------------------------


def _get_span_kind(trace_obj, span_id: str) -> str:
    """Return openinference.span.kind for a span, or '' if unknown."""
    span = trace_obj.span_by_id.get(span_id, {})
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if isinstance(attrs, dict):
        return attrs.get("openinference.span.kind", "")
    return ""


def _find_sibling_llm_span(trace_obj, span_id: str) -> Optional[str]:
    """
    For a TOOL span, find the LLM span that generated its tool call.

    In smolagents traces, each CHAIN step contains exactly one LLM child
    (LiteLLMModel.__call__) followed by one TOOL child. The LLM span authors
    the tool_calls that drive the tool execution, so the intervention must
    happen at the LLM span, not the TOOL span.

    Strategy:
      1. Get the immediate parent of the TOOL span (should be a CHAIN step).
      2. Among the parent's child_spans, find the LLM sibling.
      3. If not found at the immediate parent, walk one level further up and retry.
         (Handles edge cases where the parent is not a CHAIN but an AGENT.)

    Returns the span_id of the LLM sibling, or None if not found.
    """
    span = trace_obj.span_by_id.get(span_id)
    if span is None:
        return None

    parent_id = span.get("parent_span_id") or span.get("parentSpanId")
    while parent_id:
        parent_span = trace_obj.span_by_id.get(parent_id)
        if parent_span is None:
            break
        # Look for an LLM child among the parent's children
        for child in parent_span.get("child_spans") or []:
            child_id = child.get("span_id")
            if child_id and child_id != span_id:
                child_kind = _get_span_kind(trace_obj, child_id)
                if child_kind == "LLM":
                    return child_id
        # Not found at this level — move up one more
        parent_id = parent_span.get("parent_span_id") or parent_span.get("parentSpanId")
    return None


def _extract_root_agent_context(trace_obj) -> Tuple[str, str]:
    """
    Find the root AGENT span and extract:
      - prefix_context : system prompt (first system message)
      - user_requirements : task/user message (first user message or input_value)
    """
    prefix_context = ""
    user_requirements = ""

    for span_id, span in trace_obj.span_by_id.items():
        attrs = span.get("span_attributes") or span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        if attrs.get("openinference.span.kind") != "AGENT":
            continue
        is_root = span_id not in trace_obj.parent_of
        inp = trace_obj.input_by_location.get(span_id, {})
        messages = inp.get("messages", [])

        sys_msgs = [m.get("content", "") for m in messages if m.get("role") == "system"]
        usr_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]

        if sys_msgs:
            prefix_context = str(sys_msgs[0])[:3000]
        if usr_msgs:
            user_requirements = str(usr_msgs[0])[:3000]
        elif inp.get("input_value"):
            user_requirements = str(inp["input_value"])[:3000]

        if is_root and (prefix_context or user_requirements):
            break

    return prefix_context, user_requirements


def _get_local_snippet(trace_obj, span_id: str, patch_side: str) -> str:
    """Return the exact span text on the appropriate side."""
    if patch_side == "replace_span_output":
        out = trace_obj.output_by_location.get(span_id, {})
        return (out.get("output_text") or out.get("output_value_raw") or "")[:6000]
    else:
        inp = trace_obj.input_by_location.get(span_id, {})
        if inp.get("messages"):
            msgs = inp["messages"]
            parts = [f"[{m.get('role','')}]: {str(m.get('content',''))[:1500]}" for m in msgs[-3:]]
            return "\n".join(parts)[:6000]
        return (inp.get("input_value") or inp.get("input_value_raw") or "")[:6000]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_cases(
    trace_dir: str,
    annotations_dir: str,
    graph_path: str,
    patch_library_path: str,
    eligible_trace_ids: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
) -> Tuple[List[AInstanceRecord], List[EdgePair]]:
    """
    Build AInstanceRecords and EdgePairs for all (A-instance, B-type) pairs in the
    causal graph.

    Returns:
      a_instances : one per unique (trace_id, error_id) A-instance
      edge_pairs  : one per (A-instance × B-type) graph edge
    """
    allowed_edges, a_to_bs = load_graph_edges(graph_path)
    a_types: Set[str] = set(a for a, _ in allowed_edges)

    with open(patch_library_path, "r", encoding="utf-8") as f:
        patch_library: Dict[str, dict] = json.load(f)

    a_instances: List[AInstanceRecord] = []
    edge_pairs: List[EdgePair] = []
    seen_error_ids: Set[str] = set()

    pairs = list(iter_trail_traces(
        trace_dir, annotations_dir,
        trace_ids=eligible_trace_ids,
        max_traces=max_traces,
    ))

    for trace_path, ann_path in pairs:
        if not ann_path or not os.path.isfile(ann_path):
            continue
        try:
            trace_obj = load_trail_trace(trace_path, ann_path)
        except Exception as e:
            print(f"[WARN] load failed {trace_path}: {e}", file=sys.stderr)
            continue

        errors = trace_obj.errors
        prefix_context, user_requirements = _extract_root_agent_context(trace_obj)

        trace_b_types: Set[str] = {
            (e.get("error_type") or e.get("category") or "").strip()
            for e in errors
        }

        for i, err in enumerate(errors):
            a_cat = (err.get("error_type") or err.get("category") or "").strip()
            if a_cat not in a_types:
                continue

            span_id = err.get("annotated_span_id") or err.get("location", "")
            if not span_id:
                continue

            b_targets = [b for b in a_to_bs.get(a_cat, []) if b in trace_b_types]
            if not b_targets:
                continue

            error_id = err.get("error_id", "")
            lib_entry = patch_library.get(a_cat, {})
            patch_side = lib_entry.get("patch_side_default", "replace_span_output")

            # Determine intervention point.
            # If the annotation points to a TOOL span and patch_side is replace_span_input,
            # walk up to the parent LLM span — tool call inputs are authored by the LLM output
            # (tool_calls), so the intervention must happen at the LLM span.
            # We keep both annotated_location and intervention_location for auditing.
            annotated_span_id = span_id
            annotated_kind = _get_span_kind(trace_obj, annotated_span_id)
            intervention_span_id = annotated_span_id
            intervention_kind = annotated_kind
            effective_patch_side = patch_side

            if patch_side == "replace_span_input" and annotated_kind == "TOOL":
                parent_llm = _find_sibling_llm_span(trace_obj, annotated_span_id)
                if parent_llm:
                    intervention_span_id = parent_llm
                    intervention_kind = "LLM"
                    # The tool call arguments live in the LLM span's output (tool_calls list)
                    effective_patch_side = "replace_span_output"

            local_snippet = _get_local_snippet(trace_obj, intervention_span_id, effective_patch_side)

            a_instance_dict = {
                "category": a_cat,
                "location": annotated_span_id,   # original annotation — preserved for audit
                "description": err.get("description", ""),
                "evidence": err.get("evidence", ""),
                "impact": err.get("impact", ""),
                "error_id": error_id,
                "annotation_index": i,
            }

            # One AInstanceRecord per unique error_id
            if error_id not in seen_error_ids:
                seen_error_ids.add(error_id)
                a_instances.append(AInstanceRecord(
                    trace_id=trace_obj.trace_id,
                    error_id=error_id,
                    a_instance=a_instance_dict,
                    local_snippet=local_snippet,
                    patch_side=effective_patch_side,
                    annotated_location=annotated_span_id,
                    intervention_location=intervention_span_id,
                    annotated_span_kind=annotated_kind,
                    intervention_span_kind=intervention_kind,
                    prefix_context=prefix_context,
                    user_requirements=user_requirements,
                    tools_available=list(trace_obj.tools_available or [])[:30],
                    suffix_window_spec={"mode": "until_end"},
                    b_types=b_targets,
                ))

            # One EdgePair per (error_id, b_type)
            for b_cat in b_targets:
                b_present = False
                b_onset = -1
                for j, other_err in enumerate(errors):
                    if j <= i:
                        continue
                    if (other_err.get("error_type") or other_err.get("category") or "").strip() == b_cat:
                        b_present = True
                        b_onset = j
                        break

                edge_pairs.append(EdgePair(
                    trace_id=trace_obj.trace_id,
                    error_id=error_id,
                    edge={"a": a_cat, "b": b_cat},
                    b_def={"category": b_cat},
                    b_present_baseline=b_present,
                    b_onset_baseline=b_onset,
                ))

    return a_instances, edge_pairs


# ---------------------------------------------------------------------------
# Intervention-location conflict resolution
# ---------------------------------------------------------------------------


def dedup_by_intervention_location(
    a_instances: List[AInstanceRecord],
    conflicts_path: Optional[str] = None,
) -> List[AInstanceRecord]:
    """
    When multiple A-instances share the same intervention_location, keep only the
    first by annotation_index and write the rest to conflicts_path (if provided).

    Rationale: the rerun applies one patch per location; undefined overwrite behavior
    would otherwise make results non-reproducible. Merging patches changes the causal
    estimand, so we keep only one active intervention per location.

    Returns: active list (intervention_location unique).
    """
    seen: Dict[str, str] = {}   # intervention_location → kept error_id
    active: List[AInstanceRecord] = []
    conflict_records: List[dict] = []

    for rec in a_instances:
        loc = rec.intervention_location
        if loc not in seen:
            seen[loc] = rec.error_id
            active.append(rec)
        else:
            conflict_records.append({
                **asdict(rec),
                "conflict_reason": "shared_intervention_location",
                "kept_error_id": seen[loc],
            })

    if conflict_records and conflicts_path:
        os.makedirs(os.path.dirname(os.path.abspath(conflicts_path)), exist_ok=True)
        with open(conflicts_path, "w", encoding="utf-8") as f:
            for rec in conflict_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(conflict_records)} location-conflict records → {conflicts_path}")

    if conflict_records:
        print(f"[dedup] Kept {len(active)}/{len(active) + len(conflict_records)} A-instances "
              f"({len(conflict_records)} skipped: shared_intervention_location)")

    return active


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build AInstanceRecords and EdgePairs for causal intervention pipeline."
    )
    parser.add_argument("--trace_dir", default="data/GAIA")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia")
    parser.add_argument("--causal_graph",
                        default="data/trail_causal_outputs_AIC/capri_graph.json")
    parser.add_argument("--patch_library",
                        default="causal/patch/patch_library.json")
    parser.add_argument("--eligible_traces", default=None,
                        help="Path to eligible_traces.json from filter_traces.py")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--max_traces", type=int, default=None)
    args = parser.parse_args()

    eligible_ids = None
    if args.eligible_traces and os.path.isfile(args.eligible_traces):
        with open(args.eligible_traces, "r", encoding="utf-8") as f:
            et = json.load(f)
        eligible_ids = [t["trace_id"] for t in et.get("eligible", [])]
        print(f"Using {len(eligible_ids)} eligible traces from {args.eligible_traces}")

    os.makedirs(args.out_dir, exist_ok=True)
    a_instances, edge_pairs = build_cases(
        args.trace_dir, args.annotations_dir,
        args.causal_graph, args.patch_library,
        eligible_trace_ids=eligible_ids,
        max_traces=args.max_traces,
    )

    conflicts_path = os.path.join(args.out_dir, "intervention_location_conflicts.jsonl")
    a_instances = dedup_by_intervention_location(a_instances, conflicts_path=conflicts_path)

    a_path = os.path.join(args.out_dir, "a_instances.jsonl")
    e_path = os.path.join(args.out_dir, "edge_pairs.jsonl")

    with open(a_path, "w", encoding="utf-8") as f:
        for rec in a_instances:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    with open(e_path, "w", encoding="utf-8") as f:
        for rec in edge_pairs:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    from collections import Counter
    edge_counts = Counter(f"{ep.edge['a']} -> {ep.edge['b']}" for ep in edge_pairs)
    b_rate = sum(1 for ep in edge_pairs if ep.b_present_baseline) / max(len(edge_pairs), 1)
    print(f"Built {len(a_instances)} A-instances → {a_path}")
    print(f"Built {len(edge_pairs)} edge pairs  → {e_path}")
    print(f"b_present_baseline rate: {b_rate:.2%}")
    print("Edge pairs per edge:")
    for edge, count in sorted(edge_counts.items()):
        print(f"  {count:3d}  {edge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
