#!/usr/bin/env python3
"""
Step 8: Effect aggregation — compute Δ(A→B) per causal graph edge.

Δ(A→B) = E[B(0)] - E[B(1)]
        = mean(target_present_after | resolved=True) - mean(b_present_baseline | resolved=True)

A negative Δ means do(A=0) reduced downstream B (causal effect confirmed).

Also computes:
  - effect_label distribution per edge
  - patch_failure_rate per A category
  - placebo estimate (cross-edge baseline reassignment null)
  - edge validation: validated=True if delta < -threshold and n >= min_n

Output: effect_edges.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_graph_edges(graph_path: str) -> List[Tuple[str, str]]:
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    return [(e["a"], e["b"]) for e in g.get("edges", [])]


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate(
    b_effect_path: str,
    a_resolved_path: str,
    patch_results_path: str,
    graph_path: str,
    threshold: float = 0.15,
    min_n: int = 3,
    placebo_seeds: int = 100,
) -> dict:
    graph_edges = _load_graph_edges(graph_path)
    graph_edge_set = {(a, b) for a, b in graph_edges}

    b_verdicts = _load_jsonl(b_effect_path)
    a_verdicts = _load_jsonl(a_resolved_path)
    patch_results = _load_jsonl(patch_results_path)

    # Index Judge 1 verdicts: (trace_id, error_id) -> resolved bool
    # AResolvedVerdict is per A-instance (no edge field); edge info lives in b_effect.jsonl
    resolved_idx: Dict[Tuple, bool] = {}
    for v in a_verdicts:
        key = (v["trace_id"], v.get("error_id", ""))
        resolved_idx[key] = bool(v.get("resolved", False))

    # Patch failure rate per A category — PatchResult uses template_used (no edge field)
    patch_fail: Dict[str, List[bool]] = defaultdict(list)
    for p in patch_results:
        a_cat = p.get("template_used", "")
        patch_fail[a_cat].append(not bool(p.get("postcheck_passed", False)))

    patch_failure_rate = {
        cat: sum(fails) / len(fails)
        for cat, fails in patch_fail.items() if fails
    }

    # Per-edge data collection
    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {
        (a, b): {"b_present_baseline": [], "b_present_rerun": [], "effect_labels": []}
        for a, b in graph_edges
    }

    for v in b_verdicts:
        edge = v.get("edge", {})
        a_cat = edge.get("a", "")
        b_cat = edge.get("b", "")
        key_pair = (a_cat, b_cat)
        if key_pair not in graph_edge_set:
            continue

        a_instance_key = (v["trace_id"], v.get("error_id", ""))
        if not resolved_idx.get(a_instance_key, False):
            continue  # exclude invalid do(A=0)

        b_baseline = bool(v.get("b_present_baseline", False))
        b_rerun = bool(v.get("target_present_after", False))
        label = v.get("effect_label", "not_observable")

        edge_data[key_pair]["b_present_baseline"].append(b_baseline)
        edge_data[key_pair]["b_present_rerun"].append(b_rerun)
        edge_data[key_pair]["effect_labels"].append(label)

    def _delta(base_list: List[bool], rerun_list: List[bool]) -> float:
        if not base_list:
            return 0.0
        return sum(rerun_list) / len(rerun_list) - sum(base_list) / len(base_list)

    # Placebo: cross-edge baseline reassignment null.
    # For each edge, keep its rerun vector fixed; draw a fake baseline from
    # all *other* edges' baseline labels.  This breaks edge specificity while
    # preserving the true rerun outcomes.
    placebo_deltas: List[float] = []
    placebo_by_edge: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    rng = random.Random(42)

    all_edge_baselines = {
        edge_key: data["b_present_baseline"][:]
        for edge_key, data in edge_data.items()
        if data["b_present_baseline"]
    }

    for edge_key, data in edge_data.items():
        rr = data["b_present_rerun"][:]
        if not rr:
            continue

        other_pool: List[bool] = []
        for other_edge, other_bl in all_edge_baselines.items():
            if other_edge != edge_key:
                other_pool.extend(other_bl)

        if not other_pool:
            continue

        for _ in range(placebo_seeds):
            if len(other_pool) >= len(rr):
                fake_bl = rng.sample(other_pool, k=len(rr))
            else:
                fake_bl = [rng.choice(other_pool) for _ in range(len(rr))]

            d = _delta(fake_bl, rr)
            placebo_deltas.append(d)
            placebo_by_edge[edge_key].append(d)

    if placebo_deltas:
        placebo_mean = sum(placebo_deltas) / len(placebo_deltas)
        placebo_std = (
            sum((x - placebo_mean) ** 2 for x in placebo_deltas) / len(placebo_deltas)
        ) ** 0.5
    else:
        placebo_mean = placebo_std = 0.0

    # Build output
    edges_out = {}
    for (a_cat, b_cat) in graph_edges:
        data = edge_data.get((a_cat, b_cat), {})
        bl = data.get("b_present_baseline", [])
        rr = data.get("b_present_rerun", [])
        labels = data.get("effect_labels", [])

        n = len(bl)
        if n == 0:
            delta = None
            b_base_rate = None
            b_rerun_rate = None
            validated = False
        else:
            b_base_rate = sum(bl) / n
            b_rerun_rate = sum(rr) / n
            delta = b_rerun_rate - b_base_rate
            validated = (delta is not None and delta < -threshold and n >= min_n)

        edge_key = f"{a_cat} -> {b_cat}"
        edge_placebos = placebo_by_edge.get((a_cat, b_cat), [])
        if edge_placebos:
            ep_mean = sum(edge_placebos) / len(edge_placebos)
            ep_std = (
                sum((x - ep_mean) ** 2 for x in edge_placebos) / len(edge_placebos)
            ) ** 0.5
        else:
            ep_mean = None
            ep_std = None

        edges_out[edge_key] = {
            "a": a_cat,
            "b": b_cat,
            "n_valid_interventions": n,
            "patch_failure_rate": patch_failure_rate.get(a_cat),
            "b_present_baseline_rate": b_base_rate,
            "b_present_rerun_rate": b_rerun_rate,
            "delta": delta,
            "placebo_mean": round(ep_mean, 4) if ep_mean is not None else None,
            "placebo_std": round(ep_std, 4) if ep_std is not None else None,
            "effect_label_distribution": dict(Counter(labels)),
            "validated": validated,
            "in_capri_graph": True,
        }

    return {
        "edges": edges_out,
        "placebo": {
            "null_delta_mean": round(placebo_mean, 4),
            "null_delta_std": round(placebo_std, 4),
            "n_placebo_samples": len(placebo_deltas),
        },
        "patch_failure_by_category": patch_failure_rate,
        "validation_threshold": threshold,
        "min_n": min_n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Δ(A→B) per edge from judge verdicts."
    )
    parser.add_argument("--b_effect",
                        default="outputs/interventions/b_effect.jsonl")
    parser.add_argument("--a_resolved",
                        default="outputs/interventions/a_resolved.jsonl")
    parser.add_argument("--patch_results",
                        default="outputs/interventions/patch_results.jsonl")
    parser.add_argument("--causal_graph",
                        default="data/trail_causal_outputs_AIC/capri_graph.json")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Δ < -threshold required to validate edge")
    parser.add_argument("--min_n", type=int, default=3,
                        help="Minimum valid interventions to validate edge")
    args = parser.parse_args()

    result = aggregate(
        args.b_effect,
        args.a_resolved,
        args.patch_results,
        args.causal_graph,
        threshold=args.threshold,
        min_n=args.min_n,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "effect_edges.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nEffect edges summary (threshold={args.threshold}, min_n={args.min_n}):")
    print(f"{'Edge':<55} {'n':>4}  {'Δ':>7}  {'validated'}")
    print("-" * 80)
    for edge_key, info in result["edges"].items():
        n = info["n_valid_interventions"]
        delta = info["delta"]
        delta_str = f"{delta:+.3f}" if delta is not None else "  N/A "
        val = "YES" if info["validated"] else "no"
        print(f"{edge_key:<55} {n:>4}  {delta_str}  {val}")

    pl = result["placebo"]
    print(f"\nPlacebo null: mean={pl['null_delta_mean']:.4f}  std={pl['null_delta_std']:.4f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
