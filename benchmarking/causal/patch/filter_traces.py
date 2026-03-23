#!/usr/bin/env python3
"""
Preprocessing: filter trace files eligible for causal intervention.

A trace is eligible if it has:
  1. >= min_errors total annotated errors, AND
  2. At least one error whose type is an A-type (source node) in the causal graph.

Optionally also requires at least one B-type error appearing AFTER the A-type
(strict mode: ensures b_present_baseline=True for at least one (A,B) pair).

Output:
  eligible_traces.json  — {
    "n_total": int,
    "n_eligible": int,
    "a_types": [...],
    "eligible": [
      { "trace_id": str, "n_errors": int,
        "a_errors": [{"type": str, "index": int}],
        "b_errors_after_a": [{"type": str, "index": int}],
        "covered_edges": [{"a": str, "b": str}] }
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph(graph_path: str) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    """Returns a_types, b_types, edge_set."""
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    edges = [(e["a"], e["b"]) for e in g.get("edges", [])]
    a_types = {a for a, _ in edges}
    b_types = {b for _, b in edges}
    return a_types, b_types, set(edges)


# ---------------------------------------------------------------------------
# Annotation loader (raw, no trail_io — fast scan)
# ---------------------------------------------------------------------------

def _load_errors(ann_path: str) -> List[dict]:
    """Load raw annotation errors. Returns list of {error_type, index}."""
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    errors = []
    for i, e in enumerate(data.get("errors") or []):
        cat = (e.get("category") or e.get("error_type") or "").strip()
        errors.append({"type": cat, "index": i, "raw": e})
    return errors


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def filter_traces(
    annotations_dir: str,
    graph_path: str,
    min_errors: int = 2,
    strict: bool = False,
) -> dict:
    """
    Scan all annotation files and return eligible trace metadata.

    strict=False : require only ≥1 A-type error (b_present_baseline may be False)
    strict=True  : require at least one (A,B) pair where B appears after A
    """
    a_types, b_types, edge_set = load_graph(graph_path)
    a_to_bs: Dict[str, List[str]] = defaultdict(list)
    for a, b in edge_set:
        a_to_bs[a].append(b)

    eligible = []
    n_total = 0

    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith(".json"):
            continue
        trace_id = os.path.splitext(fname)[0]
        errors = _load_errors(os.path.join(annotations_dir, fname))
        n_total += 1

        if len(errors) < min_errors:
            continue

        # Find A-type errors
        a_errors = [e for e in errors if e["type"] in a_types]
        if not a_errors:
            continue

        # Find B-type errors after each A
        covered_edges = []
        b_after_a = []
        for ae in a_errors:
            for be in errors:
                if be["index"] <= ae["index"]:
                    continue
                if be["type"] in b_types and (ae["type"], be["type"]) in edge_set:
                    b_after_a.append({"type": be["type"], "index": be["index"]})
                    edge = {"a": ae["type"], "b": be["type"]}
                    if edge not in covered_edges:
                        covered_edges.append(edge)

        if strict and not covered_edges:
            continue

        eligible.append({
            "trace_id": trace_id,
            "n_errors": len(errors),
            "a_errors": [{"type": e["type"], "index": e["index"]} for e in a_errors],
            "b_errors_after_a": b_after_a,
            "covered_edges": covered_edges,
        })

    return {
        "n_total": n_total,
        "n_eligible": len(eligible),
        "a_types": sorted(a_types),
        "b_types": sorted(b_types),
        "strict_mode": strict,
        "min_errors": min_errors,
        "eligible": eligible,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter traces eligible for do(A=0) causal intervention."
    )
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia")
    parser.add_argument("--causal_graph",
                        default="data/trail_causal_outputs_AIC/capri_graph.json")
    parser.add_argument("--out_dir", default=None,
                        help="Directory to write eligible_traces.json. "
                             "Defaults to the same directory as --causal_graph.")
    parser.add_argument("--min_errors", type=int, default=2,
                        help="Minimum total errors in trace (default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="Also require at least one B-type error after an A-type error")
    args = parser.parse_args()

    result = filter_traces(
        args.annotations_dir, args.causal_graph,
        min_errors=args.min_errors, strict=args.strict,
    )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.causal_graph))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eligible_traces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nA-types in graph : {result['a_types']}")
    print(f"Total traces     : {result['n_total']}")
    print(f"Eligible traces  : {result['n_eligible']} "
          f"(min_errors={args.min_errors}, strict={args.strict})")

    # Edge coverage
    edge_counts: Dict[str, int] = defaultdict(int)
    for t in result["eligible"]:
        for e in t["covered_edges"]:
            edge_counts[f"{e['a']} -> {e['b']}"] += 1

    if edge_counts:
        print("\nTraces covering each graph edge:")
        for edge, count in sorted(edge_counts.items()):
            print(f"  {count:3d}  {edge}")
    else:
        print("\nNo (A→B) pairs found in annotations (try without --strict).")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
