#!/usr/bin/env python3
"""
One-time script: recompute cross-edge placebo null from existing run outputs.

Uses only:
  b_effect.jsonl       — Judge-B verdicts
  a_resolved.jsonl     — Judge-A verdicts
  capri_graph.json     — graph edge list (for ordering output)

Does NOT rerun patching, rerun harness, or judges.

Output: new_placebo_delta.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_graph_edges(graph_path: str) -> List[Tuple[str, str]]:
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    return [(e["a"], e["b"]) for e in g.get("edges", [])]


def _delta(base_list: List[bool], rerun_list: List[bool]) -> float:
    if not base_list:
        return 0.0
    return sum(rerun_list) / len(rerun_list) - sum(base_list) / len(base_list)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def recompute(
    b_effect_path: str,
    a_resolved_path: str,
    graph_path: str,
    placebo_seeds: int = 100,
) -> dict:
    graph_edges = _load_graph_edges(graph_path)
    graph_edge_set = {(a, b) for a, b in graph_edges}

    b_verdicts = _load_jsonl(b_effect_path)
    a_verdicts = _load_jsonl(a_resolved_path)

    # Build resolved index: (trace_id, error_id) -> bool
    resolved_idx: Dict[Tuple[str, str], bool] = {}
    for v in a_verdicts:
        key = (v["trace_id"], v.get("error_id", ""))
        resolved_idx[key] = bool(v.get("resolved", False))

    # Collect per-edge vectors (resolved cases only)
    edge_data: Dict[Tuple[str, str], Dict] = {
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
        a_key = (v["trace_id"], v.get("error_id", ""))
        if not resolved_idx.get(a_key, False):
            continue

        edge_data[key_pair]["b_present_baseline"].append(bool(v.get("b_present_baseline", False)))
        edge_data[key_pair]["b_present_rerun"].append(bool(v.get("target_present_after", False)))
        edge_data[key_pair]["effect_labels"].append(v.get("effect_label", "not_observable"))

    # Cross-edge placebo null
    all_edge_baselines = {
        edge_key: data["b_present_baseline"][:]
        for edge_key, data in edge_data.items()
        if data["b_present_baseline"]
    }

    placebo_deltas: List[float] = []
    placebo_by_edge: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    rng = random.Random(42)

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

    # Pooled placebo stats
    if placebo_deltas:
        pool_mean = sum(placebo_deltas) / len(placebo_deltas)
        pool_std = (
            sum((x - pool_mean) ** 2 for x in placebo_deltas) / len(placebo_deltas)
        ) ** 0.5
    else:
        pool_mean = pool_std = 0.0

    # Build per-edge output
    edges_out = {}
    for (a_cat, b_cat) in graph_edges:
        data = edge_data[(a_cat, b_cat)]
        bl = data["b_present_baseline"]
        rr = data["b_present_rerun"]
        labels = data["effect_labels"]
        n = len(bl)

        if n == 0:
            real_delta = None
            b_base_rate = None
            b_rerun_rate = None
        else:
            b_base_rate = sum(bl) / n
            b_rerun_rate = sum(rr) / n
            real_delta = b_rerun_rate - b_base_rate

        ep = placebo_by_edge.get((a_cat, b_cat), [])
        if ep:
            ep_mean = sum(ep) / len(ep)
            ep_std = (sum((x - ep_mean) ** 2 for x in ep) / len(ep)) ** 0.5
        else:
            ep_mean = None
            ep_std = None

        edge_key = f"{a_cat} -> {b_cat}"
        edges_out[edge_key] = {
            "a": a_cat,
            "b": b_cat,
            "n_valid_interventions": n,
            "b_present_baseline_rate": round(b_base_rate, 4) if b_base_rate is not None else None,
            "b_present_rerun_rate": round(b_rerun_rate, 4) if b_rerun_rate is not None else None,
            "delta": round(real_delta, 4) if real_delta is not None else None,
            "placebo_mean": round(ep_mean, 4) if ep_mean is not None else None,
            "placebo_std": round(ep_std, 4) if ep_std is not None else None,
            "n_placebo_samples": len(ep),
            "effect_label_distribution": dict(Counter(labels)),
        }

    return {
        "edges": edges_out,
        "placebo_pooled": {
            "null_delta_mean": round(pool_mean, 4),
            "null_delta_std": round(pool_std, 4),
            "n_placebo_samples": len(placebo_deltas),
            "method": "cross_edge_baseline_reassignment",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute cross-edge placebo null from existing run outputs."
    )
    parser.add_argument("--b_effect",    required=True, help="Path to b_effect.jsonl")
    parser.add_argument("--a_resolved",  required=True, help="Path to a_resolved.jsonl")
    parser.add_argument("--causal_graph", required=True, help="Path to capri_graph.json")
    parser.add_argument("--out",         default="new_placebo_delta.json",
                        help="Output file (default: new_placebo_delta.json)")
    parser.add_argument("--placebo_seeds", type=int, default=100)
    args = parser.parse_args()

    result = recompute(
        args.b_effect,
        args.a_resolved,
        args.causal_graph,
        placebo_seeds=args.placebo_seeds,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    pl = result["placebo_pooled"]
    print(f"\nCross-edge placebo null (pooled): mean={pl['null_delta_mean']:.4f}  std={pl['null_delta_std']:.4f}  n={pl['n_placebo_samples']}")
    print(f"\n{'Edge':<55} {'n':>4}  {'Δ':>7}  {'placebo_mean':>13}  {'placebo_std':>11}")
    print("-" * 95)
    for edge_key, info in result["edges"].items():
        n = info["n_valid_interventions"]
        d = info["delta"]
        pm = info["placebo_mean"]
        ps = info["placebo_std"]
        print(
            f"{edge_key:<55} {n:>4}  "
            f"{(f'{d:+.3f}' if d is not None else '   N/A'):>7}  "
            f"{(f'{pm:+.4f}' if pm is not None else '         N/A'):>13}  "
            f"{(f'{ps:.4f}' if ps is not None else '        N/A'):>11}"
        )

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
