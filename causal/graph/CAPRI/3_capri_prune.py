"""
CAPRI-style structure learning: score-based DAG search with BIC/AIC.

- Candidate edges from Suppes; hill-climbing (add/remove/reverse) over DAGs.
- Local likelihood: discrete BN CPT (MLE P(B|parent config) per config); k_params = 2^k.
- Incremental score: cache local_score(node, parent_set); only rescore affected nodes per move.

Usage:
    python causal_explore/CAPRI/3_capri_prune.py
    python causal_explore/CAPRI/3_capri_prune.py --criterion AIC --max_parents 5
"""

import json
import os
import argparse
from collections import defaultdict
import math


def local_score_cpt(node, parent_list, X, mode_to_idx, n_traces, criterion="BIC", cache=None):
    """
    Local BIC/AIC for one node using CPT (conditional probability table).
    Binary node B with parents pa: 2^|pa| configs, each has MLE P(B=1|config) with Laplace smoothing.
    k_params = 2^|pa|. Returns score (lower is better).
    """
    key = (node, frozenset(parent_list))
    if cache is not None and key in cache:
        return cache[key]

    col_b = mode_to_idx[node]
    k = len(parent_list)
    # Fixed order for parent configs (for consistent indexing)
    pa_ordered = sorted(parent_list)
    pa_cols = [mode_to_idx[p] for p in pa_ordered]

    # Count (config_tuple) -> (n_total, n_b1)
    counts = defaultdict(lambda: [0, 0])
    for row in X:
        config = tuple(row[c] for c in pa_cols) if pa_cols else ()
        counts[config][0] += 1
        if row[col_b] == 1:
            counts[config][1] += 1

    # -2 * log L = sum over configs of -2 * [ n1*log(p) + n0*log(1-p) ] with Laplace p = (n1+0.5)/(n+1)
    neg2ll = 0.0
    for config, (n, n1) in counts.items():
        n0 = n - n1
        p = (n1 + 0.5) / (n + 1.0)
        p = max(1e-9, min(1 - 1e-9, p))
        neg2ll += -2 * (n1 * math.log(p) + n0 * math.log(1 - p))

    k_params = 2 ** k
    if criterion == "BIC":
        score = neg2ll + k_params * math.log(n_traces)
    else:
        score = neg2ll + 2 * k_params

    if cache is not None:
        cache[key] = score
    return score


def edges_to_parents(edges):
    """edges = set of (a,b). Return dict: node -> list of parents (sorted for consistency)."""
    parents_of = defaultdict(list)
    for a, b in edges:
        parents_of[b].append(a)
    for b in parents_of:
        parents_of[b] = sorted(parents_of[b])
    return parents_of


def graph_total_score(edges, modes, X, mode_to_idx, n_traces, criterion="BIC", cache=None):
    """Total score = sum of local CPT scores. Optionally use and fill cache."""
    parents_of = edges_to_parents(edges)
    total = 0.0
    for b in modes:
        par = parents_of.get(b, [])
        total += local_score_cpt(b, par, X, mode_to_idx, n_traces, criterion, cache)
    return total


def has_cycle(edges, modes):
    """True if the directed graph contains a cycle."""
    out = defaultdict(list)
    for a, b in edges:
        out[a].append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {m: WHITE for m in modes}

    def dfs(u):
        color[u] = GRAY
        for v in out[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for m in modes:
        if color[m] == WHITE and dfs(m):
            return True
    return False


def get_neighbors(edges, candidate_set, modes, max_parents=None):
    """Neighbors: add / remove / reverse (only if acyclic and within candidate set)."""
    edges = set(edges)
    out = []
    parents_of = defaultdict(set)
    for a, b in edges:
        parents_of[b].add(a)

    for a, b in candidate_set:
        if (a, b) in edges:
            continue
        if max_parents is not None and len(parents_of[b]) >= max_parents:
            continue
        new_e = edges | {(a, b)}
        if not has_cycle(new_e, modes):
            out.append(("add", (a, b), new_e))

    for e in edges:
        new_e = edges - {e}
        out.append(("remove", e, new_e))

    for a, b in list(edges):
        if (b, a) not in candidate_set or (b, a) in edges:
            continue
        new_e = (edges - {(a, b)}) | {(b, a)}
        if max_parents is not None:
            if sum(1 for _a, _b in new_e if _b == a) > max_parents:
                continue
        if not has_cycle(new_e, modes):
            out.append(("reverse", (a, b), new_e))

    return out


def affected_nodes(move, new_edges):
    """Return set of nodes whose parent set changed. move = ('add'|'remove'|'reverse', e, new_edges)."""
    op, e, _ = move
    if op == "add" or op == "remove":
        return {e[1]}  # child
    else:
        return {e[0], e[1]}  # reverse (a,b)->(b,a): both a and b change parents


def parents_of_in(new_edges, node):
    """List of parents of node in edge set (sorted)."""
    return sorted(a for a, b in new_edges if b == node)


def hill_climb(candidate_set, modes, X, mode_to_idx, n_traces, criterion="BIC", max_parents=None, max_iters=500):
    """
    Hill-climb with incremental score: cache local_score(node, parent_set);
    for each neighbor only rescore affected nodes.
    """
    cache = {}
    current_edges = set()
    # Current total and per-node local scores
    parents_of = edges_to_parents(current_edges)
    current_local = {}
    for b in modes:
        par = parents_of.get(b, [])
        current_local[b] = local_score_cpt(b, par, X, mode_to_idx, n_traces, criterion, cache)
    current_score = sum(current_local[b] for b in modes)
    history = [current_score]

    for it in range(max_iters):
        neighbors = get_neighbors(current_edges, candidate_set, modes, max_parents)
        best_score = current_score
        best_edges = None
        best_affected = None

        for move in neighbors:
            op, e, new_edges = move
            affected = affected_nodes(move, new_edges)
            # Delta: remove old local scores for affected, add new local scores
            delta = 0.0
            for n in affected:
                delta -= current_local[n]
                par_new = parents_of_in(new_edges, n)
                s = local_score_cpt(n, par_new, X, mode_to_idx, n_traces, criterion, cache)
                delta += s
            new_total = current_score + delta
            if new_total < best_score:
                best_score = new_total
                best_edges = new_edges
                best_affected = affected

        if best_edges is None:
            break

        # Update state
        current_edges = best_edges
        current_score = best_score
        for n in best_affected:
            par_new = parents_of_in(current_edges, n)
            current_local[n] = local_score_cpt(n, par_new, X, mode_to_idx, n_traces, criterion, cache)
        history.append(current_score)

    return current_edges, history


def main():
    ap = argparse.ArgumentParser(description="CAPRI: CPT-based DAG search (hill-climb + BIC/AIC, incremental score)")
    ap.add_argument("--onsets_path", default="data/derived/onsets.jsonl", help="Input onsets path")
    ap.add_argument("--suppes_path", default="outputs/suppes_graph.json", help="Suppes graph (candidate edges)")
    ap.add_argument("--out_path", default="outputs/capri_graph.json", help="Output pruned graph path")
    ap.add_argument("--max_parents", type=int, default=None,
                    help="Optional cap on parents per node (default: no cap)")
    ap.add_argument("--criterion", choices=["BIC", "AIC"], default="BIC", help="Score criterion (default: BIC)")
    ap.add_argument("--max_iters", type=int, default=500, help="Max hill-climbing iterations (default: 500)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    print(f"Loading onsets from {args.onsets_path}...")
    rows = []
    modes = set()
    with open(args.onsets_path, "r") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            modes.update((r.get("onset") or {}).keys())
    modes = sorted(modes)
    n_traces = len(rows)
    mode_to_idx = {m: i for i, m in enumerate(modes)}
    print(f"Loaded {n_traces} traces with {len(modes)} failure modes")

    X = []
    for r in rows:
        onset = r.get("onset") or {}
        X.append([1 if m in onset else 0 for m in modes])

    print(f"Loading Suppes graph from {args.suppes_path}...")
    suppes = json.load(open(args.suppes_path, "r"))
    candidate_set = set((e["a"], e["b"]) for e in suppes["edges"])
    print(f"Suppes candidate edges: {len(candidate_set)}")

    print(f"\nHill-climbing (CPT local + {args.criterion}, max_parents={args.max_parents or 'none'}, incremental score)...")
    best_edges, history = hill_climb(
        candidate_set, modes, X, mode_to_idx, n_traces,
        criterion=args.criterion, max_parents=args.max_parents, max_iters=args.max_iters
    )

    pruned_edges = [{"a": a, "b": b} for a, b in sorted(best_edges)]
    parent_counts = defaultdict(int)
    for a, b in best_edges:
        parent_counts[b] += 1

    output = {
        "params": {
            "criterion": args.criterion,
            "max_parents": args.max_parents,
            "max_iters": args.max_iters,
            "onsets_path": args.onsets_path,
            "suppes_path": args.suppes_path,
        },
        "n_traces": n_traces,
        "n_modes": len(modes),
        "suppes_n_edges": len(candidate_set),
        "pruned_n_edges": len(pruned_edges),
        "score_history": history,
        "edges": pruned_edges,
    }
    with open(args.out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CAPRI PRUNING RESULTS (CPT + {args.criterion})")
    print(f"{'='*60}")
    print(f"Suppes edges (candidates): {len(candidate_set)}")
    print(f"Pruned edges (output): {len(pruned_edges)}")
    print(f"Reduction: {100*(1 - len(pruned_edges)/max(1, len(candidate_set))):.1f}%")
    print(f"Score iterations: {len(history)}")
    print(f"\nParent count distribution:")
    for np in sorted(set(parent_counts.values()), reverse=True):
        nn = sum(1 for c in parent_counts.values() if c == np)
        print(f"  {np} parents: {nn} nodes")
    print(f"\n✓ Saved to {args.out_path}")


if __name__ == "__main__":
    main()
