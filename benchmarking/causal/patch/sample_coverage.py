#!/usr/bin/env python3
"""
Sample a minimal set of traces that covers all graph edges for pipeline testing.

Reads from an already-filtered eligible_traces.json (output of filter_traces.py),
so the pool is always the same 64 eligible traces and no re-scanning is needed.

Algorithm:
  1. Load eligible traces from eligible_traces.json (already has covered_edges).
  2. Greedy set cover — pick traces that cover the most uncovered edges.
  3. Backup pass — for each edge that still has fewer than --min_backup
     traces in the selected set, add more traces until the target is met.
  4. Write output as eligible_traces_test.json (same schema as eligible_traces.json).

Usage (from benchmarking/):
    python causal/patch/sample_coverage.py \\
        --eligible_file outputs/interventions/eligible_traces.json \\
        --causal_graph  data/trail_causal_outputs_AIC/capri_graph.json \\
        --out_dir       outputs/test_run \\
        --min_backup    1

Then pass the output to run_pipeline.py:
    python causal/patch/run_pipeline.py \\
        --eligible_file outputs/test_run/eligible_traces_test.json \\
        ...other args...
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
# Helpers
# ---------------------------------------------------------------------------

def load_graph(graph_path: str):
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    edges = [(e["a"], e["b"]) for e in g.get("edges", [])]
    a_types = {a for a, _ in edges}
    b_types = {b for _, b in edges}
    return a_types, b_types, set(edges), edges


def load_eligible(eligible_path: str) -> Dict[str, dict]:
    """
    Load trace_info from an eligible_traces.json file.
    Returns dict: trace_id -> trace record (already has covered_edges, n_errors, etc.)
    Only includes traces that have at least one covered edge.
    """
    with open(eligible_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        t["trace_id"]: t
        for t in data.get("eligible", [])
        if t.get("covered_edges")
    }


# ---------------------------------------------------------------------------
# Greedy set cover
# ---------------------------------------------------------------------------

def greedy_cover(
    all_edges: List[Tuple[str, str]],
    trace_info: Dict[str, dict],
) -> Tuple[List[str], Set[Tuple[str, str]]]:
    """
    Greedy set cover: iteratively pick the trace that covers the most
    currently uncovered edges. Tie-break: more total errors (more signal).
    Returns (selected_ids, still_uncovered).
    """
    uncovered = set(all_edges)
    selected: List[str] = []

    edge_sets: Dict[str, Set[Tuple[str, str]]] = {
        tid: {(e["a"], e["b"]) for e in info["covered_edges"]}
        for tid, info in trace_info.items()
    }

    while uncovered:
        best_tid = None
        best_gain = 0
        best_n_errors = 0
        for tid, es in edge_sets.items():
            if tid in selected:
                continue
            gain = len(es & uncovered)
            n_err = trace_info[tid]["n_errors"]
            if gain > best_gain or (gain == best_gain and n_err > best_n_errors):
                best_tid = tid
                best_gain = gain
                best_n_errors = n_err
        if best_tid is None or best_gain == 0:
            break
        selected.append(best_tid)
        uncovered -= edge_sets[best_tid]

    return selected, uncovered


def add_backups(
    all_edges: List[Tuple[str, str]],
    selected: List[str],
    trace_info: Dict[str, dict],
    min_backup: int,
) -> List[str]:
    """
    For each edge, ensure at least min_backup traces cover it in selected set.
    Add traces greedily (prefer traces covering the most under-covered edges).
    """
    edge_sets: Dict[str, Set[Tuple[str, str]]] = {
        tid: {(e["a"], e["b"]) for e in info["covered_edges"]}
        for tid, info in trace_info.items()
    }
    selected_set = set(selected)

    def coverage_counts():
        counts: Dict[Tuple, int] = defaultdict(int)
        for tid in selected_set:
            for e in edge_sets.get(tid, set()):
                counts[e] += 1
        return counts

    while True:
        counts = coverage_counts()
        need_more = {e for e in all_edges if counts.get(e, 0) < min_backup}
        if not need_more:
            break
        best_tid = None
        best_gain = 0
        best_n_errors = 0
        for tid, es in edge_sets.items():
            if tid in selected_set:
                continue
            gain = len(es & need_more)
            n_err = trace_info[tid]["n_errors"]
            if gain > best_gain or (gain == best_gain and n_err > best_n_errors):
                best_tid = tid
                best_gain = gain
                best_n_errors = n_err
        if best_tid is None or best_gain == 0:
            break
        selected_set.add(best_tid)
        selected.append(best_tid)

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample minimal trace set covering all graph edges for pipeline testing."
    )
    parser.add_argument("--eligible_file",
                        default="data/trail_causal_outputs_AIC/eligible_traces.json",
                        help="Path to eligible_traces.json from filter_traces.py")
    parser.add_argument("--causal_graph",
                        default="data/trail_causal_outputs_AIC/capri_graph.json")
    parser.add_argument("--out_dir", default="outputs/test_run")
    parser.add_argument("--out_file", default="eligible_traces_test.json",
                        help="Output filename inside out_dir")
    parser.add_argument("--min_backup", type=int, default=1,
                        help="Min traces per edge in the sampled set (default: 1)")
    args = parser.parse_args()

    if not os.path.isfile(args.eligible_file):
        print(f"ERROR: eligible_file not found: {args.eligible_file}")
        print("Run filter_traces.py first to generate it.")
        return 1

    # Load graph (for full edge list)
    a_types, b_types, _, all_edges = load_graph(args.causal_graph)
    print(f"Graph: {len(all_edges)} edges, {len(a_types)} A-types, {len(b_types)} B-types")

    # Load eligible traces (already filtered — our pool)
    trace_info = load_eligible(args.eligible_file)
    print(f"Eligible traces with ≥1 covered edge: {len(trace_info)}")

    # Summarize per-edge trace counts within the eligible pool
    print("\nPer-edge trace availability (in eligible pool):")
    edge_trace_counts: Dict[Tuple, List[str]] = defaultdict(list)
    for tid, info in trace_info.items():
        for e in info["covered_edges"]:
            edge_trace_counts[(e["a"], e["b"])].append(tid)
    for e in sorted(all_edges):
        count = len(edge_trace_counts.get(e, []))
        flag = "  *** RARE ***" if count < 3 else ""
        print(f"  {count:3d}  {e[0]} -> {e[1]}{flag}")

    # Greedy cover
    selected, still_uncovered = greedy_cover(all_edges, trace_info)
    if still_uncovered:
        print(f"\nWARNING: {len(still_uncovered)} edges have NO eligible trace:")
        for e in sorted(still_uncovered):
            print(f"  {e[0]} -> {e[1]}")
    print(f"\nGreedy cover: {len(selected)} traces cover all reachable edges")

    # Backup pass
    selected = add_backups(all_edges, selected, trace_info, args.min_backup)
    print(f"After backup (min_backup={args.min_backup}): {len(selected)} traces")

    # Verify final coverage counts
    print("\nFinal per-edge coverage in sampled set:")
    final_counts: Dict[Tuple, int] = defaultdict(int)
    for tid in selected:
        for e in trace_info[tid]["covered_edges"]:
            final_counts[(e["a"], e["b"])] += 1
    for e in sorted(all_edges):
        cnt = final_counts.get(e, 0)
        flag = "  <<< MISSING" if cnt == 0 else ("  (only 1)" if cnt == 1 else "")
        print(f"  {cnt:3d}  {e[0]} -> {e[1]}{flag}")

    # Build output in same schema as eligible_traces.json
    with open(args.eligible_file) as f:
        source = json.load(f)
    result = {
        "n_total": source.get("n_total", 0),
        "n_eligible": len(selected),
        "a_types": sorted(a_types),
        "b_types": sorted(b_types),
        "strict_mode": source.get("strict_mode", True),
        "min_errors": source.get("min_errors", 0),
        "sampled": True,
        "min_backup": args.min_backup,
        "all_edges_covered": len(still_uncovered) == 0,
        "eligible": [trace_info[tid] for tid in selected],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSelected {len(selected)} traces → {out_path}")
    print("\nSelected trace IDs:")
    for tid in selected:
        n_edges = len(trace_info[tid]["covered_edges"])
        edge_labels = ", ".join(
            f"{e['a'][:10]}→{e['b'][:10]}"
            for e in trace_info[tid]["covered_edges"]
        )
        print(f"  {tid}  [{n_edges} edges: {edge_labels}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
