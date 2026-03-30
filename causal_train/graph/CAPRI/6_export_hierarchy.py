"""
Export hierarchical levels from pruned causal graph.

Computes topological levels from the pruned graph. If cycles exist, breaks them
by repeatedly removing the edge in a cycle with the lowest stability score
(so the resulting graph is a DAG).

Usage:
    python causal_explore/CAPRI/6_export_hierarchy.py
    python causal_explore/CAPRI/6_export_hierarchy.py --stability_threshold 0.3
"""

import json
import os
import argparse
from collections import defaultdict


def find_cycle(edges, nodes):
    """
    If the graph has a cycle, return a list of edges that form a cycle (first cycle found).
    Uses DFS; returns None if no cycle.
    """
    out = defaultdict(list)
    for a, b in edges:
        out[a].append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    parent_edge = {}  # node -> (a,b) that we used to enter it

    def dfs(u, from_edge):
        color[u] = GRAY
        for v in out[u]:
            e = (u, v)
            if color[v] == GRAY:
                # Back edge (u,v): cycle is (u,v) plus path from v to u via parent_edge
                cycle_edges = [e]
                cur = u
                while cur != v:
                    if cur not in parent_edge:
                        break
                    cycle_edges.append(parent_edge[cur])
                    cur = parent_edge[cur][0]
                return cycle_edges
            if color[v] == WHITE:
                parent_edge[v] = e
                res = dfs(v, e)
                if res is not None:
                    return res
        color[u] = BLACK
        return None

    for n in nodes:
        if color[n] == WHITE:
            parent_edge.clear()
            cycle = dfs(n, None)
            if cycle is not None:
                return cycle
    return None


def break_cycles(edges, stability_scores=None):
    """
    Return an acyclic subset of edges by repeatedly finding a cycle and
    removing the edge in it with the lowest stability (or arbitrary if no scores).
    """
    edges = list(edges)
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    edge_set = set(edges)

    while True:
        cycle = find_cycle(edge_set, nodes)
        if cycle is None:
            break
        # Remove edge in cycle with lowest stability (higher score = more stable)
        if stability_scores:
            worst = min(cycle, key=lambda e: stability_scores.get(e, 0.0))
        else:
            worst = cycle[0]
        edge_set.discard(worst)
    return edge_set


def topological_levels(edges, stability_scores=None, min_stability=0.3):
    """
    Compute topological levels for a DAG.
    Filters by min_stability, breaks cycles, then assigns levels.
    """
    # Filter edges by stability (default min_stability=0.3)
    if stability_scores:
        filtered_edges = [
            (a, b) for a, b in edges
            if stability_scores.get((a, b), 1.0) >= min_stability
        ]
    else:
        filtered_edges = list(edges)

    # Break cycles by removing lowest-stability edges in each cycle
    dag_edges = break_cycles(filtered_edges, stability_scores)
    if len(dag_edges) < len(filtered_edges):
        n_broken = len(filtered_edges) - len(dag_edges)
        print(f"  Broke {n_broken} edge(s) to remove cycles")

    # Build adjacency lists from DAG
    children = defaultdict(list)
    parents = defaultdict(list)
    all_nodes = set()
    for a, b in dag_edges:
        children[a].append(b)
        parents[b].append(a)
        all_nodes.add(a)
        all_nodes.add(b)

    levels = {}
    assigned = set()
    for node in all_nodes:
        if not parents[node]:
            levels[node] = 0
            assigned.add(node)

    current_level = 0
    while len(assigned) < len(all_nodes):
        current_level += 1
        next_level = []
        for node in all_nodes:
            if node in assigned:
                continue
            if not parents[node]:
                levels[node] = 0
                assigned.add(node)
                next_level.append(node)
                continue
            parent_levels = [levels[p] for p in parents[node] if p in assigned]
            if len(parent_levels) == len(parents[node]):
                levels[node] = max(parent_levels) + 1
                assigned.add(node)
                next_level.append(node)
        if not next_level:
            for node in all_nodes:
                if node not in assigned:
                    levels[node] = current_level
                    assigned.add(node)
            break

    levels_grouped = defaultdict(list)
    for node, level in levels.items():
        levels_grouped[level].append(node)
    return dict(levels_grouped)


def main():
    ap = argparse.ArgumentParser(description="Export hierarchical levels")
    ap.add_argument("--capri_path", default="outputs/capri_graph.json",
                    help="Input pruned graph path")
    ap.add_argument("--stability_path", default="outputs/edge_stability.json",
                    help="Edge stability scores (optional)")
    ap.add_argument("--out_path", default="outputs/hierarchy_levels.json",
                    help="Output hierarchy levels path")
    ap.add_argument("--stability_threshold", type=float, default=0.3,
                    help="Minimum stability score to include edge (default: 0.3)")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Load pruned graph
    print(f"Loading pruned graph from {args.capri_path}...")
    capri = json.load(open(args.capri_path, "r"))
    edges = [(e["a"], e["b"]) for e in capri["edges"]]
    print(f"Loaded {len(edges)} edges")

    # Load stability scores if available
    stability_scores = {}
    if os.path.exists(args.stability_path):
        print(f"Loading stability scores from {args.stability_path}...")
        stability = json.load(open(args.stability_path, "r"))
        for e in stability.get("edges", []):
            stability_scores[(e["a"], e["b"])] = e["frequency"]
        print(f"Loaded {len(stability_scores)} stability scores")
    else:
        print("No stability scores found (using all edges)")

    # Compute levels
    print(f"\nComputing topological levels (min_stability={args.stability_threshold})...")
    levels = topological_levels(edges, stability_scores, args.stability_threshold)

    # Sort levels
    levels_sorted = {
        f"level_{i}": sorted(nodes) 
        for i, nodes in sorted(levels.items())
    }

    # Create output
    output = {
        "params": {
            "capri_path": args.capri_path,
            "stability_path": args.stability_path,
            "stability_threshold": args.stability_threshold
        },
        "n_levels": len(levels),
        "n_nodes": sum(len(nodes) for nodes in levels.values()),
        "levels": levels_sorted
    }

    # Save output
    with open(args.out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HIERARCHY LEVELS")
    print(f"{'='*60}")
    print(f"Total levels: {len(levels)}")
    print(f"Total nodes: {sum(len(nodes) for nodes in levels.values())}")
    
    for i in sorted(levels.keys()):
        nodes = levels[i]
        print(f"\nLevel {i} ({len(nodes)} nodes):")
        print(f"  {', '.join(sorted(nodes)[:10])}{' ...' if len(nodes) > 10 else ''}")
    
    print(f"\n✓ Saved to {args.out_path}")


if __name__ == "__main__":
    main()
