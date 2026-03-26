#!/usr/bin/env python3
"""
Build graph data for GNN training from TRAIL Suppes causal graph.

Edge weight formula for each observed edge A→B:
    s_AB = (α · precedence_AB + (1-α) · ΔPR̃_AB) · n_AB / (n_AB + κ)

where:
    n_AB  = precedence_n  (# traces where both A and B co-occur with non-tied onsets)
    ΔPR̃  = min-max normalized pr_delta across all edges
    κ     = shrinkage constant (default 5)
    α     = precedence weight (default 0.4)

Validated causal edges override to weight = 1.0.
Pairs absent from the Suppes graph get weight = 0.

Outputs (in --out_dir):
    node_list.json          - {id: node_name, ...}
    edge_list.csv           - per-edge details with final weights
    adjacency_matrix.csv    - N×N matrix (node names as headers)
    adjacency_matrix.npy    - N×N float32 numpy array
    graph_data.json         - PyG-style: edge_index, edge_attr, node_names, metadata
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validated causal edges (A → B): assigned weight = 1.0
# ---------------------------------------------------------------------------
CAUSAL_EDGES = {
    ("Formatting Errors", "Tool Output Misinterpretation"),
    ("Formatting Errors", "Incorrect Problem Identification"),
    ("Formatting Errors", "Context Handling Failures"),
    ("Formatting Errors", "Resource Abuse"),
    ("Incorrect Problem Identification", "Tool Output Misinterpretation"),
    ("Poor Information Retrieval", "Resource Abuse"),
    ("Resource Abuse", "Authentication Errors"),
    ("Resource Abuse", "Tool-related"),
    ("Task Orchestration", "Context Handling Failures"),
    ("Tool Selection Errors", "Language-only"),
    ("Tool Selection Errors", "Goal Deviation"),
}


def main():
    ap = argparse.ArgumentParser(description="Build GNN-ready graph data from Suppes graph")
    ap.add_argument(
        "--suppes_path",
        default="data/trail_causal_outputs_fully_connected/suppes_graph.json",
        help="Path to suppes_graph.json (default: fully-connected run)",
    )
    ap.add_argument("--out_dir", default="graph/outputs", help="Output directory")
    ap.add_argument("--alpha", type=float, default=0.4, help="Precedence weight α (default 0.4)")
    ap.add_argument("--kappa", type=float, default=5.0, help="Shrinkage constant κ (default 5)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Suppes graph
    # ------------------------------------------------------------------
    with open(args.suppes_path) as f:
        suppes = json.load(f)

    raw_edges = suppes["edges"]
    print(f"Loaded {len(raw_edges)} Suppes edges from {args.suppes_path}")

    # ------------------------------------------------------------------
    # Collect all nodes (deterministic sorted order)
    # ------------------------------------------------------------------
    node_set = set()
    for e in raw_edges:
        node_set.add(e["a"])
        node_set.add(e["b"])
    # Also add any causal-edge nodes not in Suppes (edge case)
    for a, b in CAUSAL_EDGES:
        node_set.add(a)
        node_set.add(b)

    nodes = sorted(node_set)
    n = len(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    print(f"Nodes ({n}): {nodes}")

    # ------------------------------------------------------------------
    # Min-max normalise ΔPR across all 164 Suppes edges
    # ------------------------------------------------------------------
    pr_deltas = np.array([e["pr_delta"] for e in raw_edges], dtype=float)
    pr_min, pr_max = pr_deltas.min(), pr_deltas.max()
    pr_range = pr_max - pr_min if pr_max > pr_min else 1.0
    print(f"ΔPR range: [{pr_min:.4f}, {pr_max:.4f}]")

    # ------------------------------------------------------------------
    # Compute s_AB_obs for each Suppes edge
    # ------------------------------------------------------------------
    edge_records = []
    for e in raw_edges:
        a, b = e["a"], e["b"]
        prec = e["pr_delta"]
        pr_norm = (prec - pr_min) / pr_range  # but wait — need precedence vs pr_delta separately
        # Clarification: α weights the precedence fraction, (1-α) weights ΔPR̃
        precedence_frac = e["precedence"]       # fraction A precedes B
        pr_delta_norm = (e["pr_delta"] - pr_min) / pr_range
        n_ab = e["precedence_n"]

        score = (args.alpha * precedence_frac + (1 - args.alpha) * pr_delta_norm) * (
            n_ab / (n_ab + args.kappa)
        )
        is_causal = (a, b) in CAUSAL_EDGES
        final_weight = 1.0 if is_causal else score

        edge_records.append(
            {
                "src": a,
                "dst": b,
                "src_id": node_to_idx[a],
                "dst_id": node_to_idx[b],
                "precedence": e["precedence"],
                "pr_delta": e["pr_delta"],
                "pr_delta_norm": round(pr_delta_norm, 6),
                "precedence_n": n_ab,
                "s_obs": round(score, 6),
                "is_causal": is_causal,
                "weight": round(final_weight, 6),
            }
        )

    # ------------------------------------------------------------------
    # Build N×N adjacency matrix  (0 for absent edges)
    # ------------------------------------------------------------------
    adj = np.zeros((n, n), dtype=np.float32)
    for r in edge_records:
        adj[r["src_id"], r["dst_id"]] = r["weight"]

    # Override causal edges — also add them if somehow not in Suppes
    for a, b in CAUSAL_EDGES:
        i, j = node_to_idx[a], node_to_idx[b]
        adj[i, j] = 1.0

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # 1. Node list
    node_list_path = os.path.join(args.out_dir, "node_list.json")
    with open(node_list_path, "w") as f:
        json.dump({i: name for i, name in enumerate(nodes)}, f, indent=2)
    print(f"Saved node list → {node_list_path}")

    # 2. Edge list CSV
    df_edges = pd.DataFrame(edge_records)
    edge_list_path = os.path.join(args.out_dir, "edge_list.csv")
    df_edges.to_csv(edge_list_path, index=False)
    print(f"Saved edge list ({len(df_edges)} edges) → {edge_list_path}")

    # 3. Adjacency matrix CSV (with node names as row/col headers)
    df_adj = pd.DataFrame(adj, index=nodes, columns=nodes)
    adj_csv_path = os.path.join(args.out_dir, "adjacency_matrix.csv")
    df_adj.to_csv(adj_csv_path)
    print(f"Saved adjacency matrix ({n}×{n}) → {adj_csv_path}")

    # 4. Adjacency matrix .npy
    adj_npy_path = os.path.join(args.out_dir, "adjacency_matrix.npy")
    np.save(adj_npy_path, adj)
    print(f"Saved adjacency matrix .npy → {adj_npy_path}")

    # 5. PyG-style graph_data.json
    #    edge_index: [[src...], [dst...]]  — includes ALL non-zero edges (Suppes + causal)
    nonzero_mask = adj > 0
    src_ids, dst_ids = np.where(nonzero_mask)
    edge_index = [src_ids.tolist(), dst_ids.tolist()]
    edge_attr = adj[src_ids, dst_ids].tolist()

    # Flag which edges are validated causal
    causal_set_idx = {(node_to_idx[a], node_to_idx[b]) for a, b in CAUSAL_EDGES}
    edge_is_causal = [(int(s), int(d)) in causal_set_idx for s, d in zip(src_ids, dst_ids)]

    graph_data = {
        "meta": {
            "suppes_path": args.suppes_path,
            "n_nodes": n,
            "n_suppes_edges": len(raw_edges),
            "n_causal_edges": len(CAUSAL_EDGES),
            "n_total_edges": int(nonzero_mask.sum()),
            "alpha": args.alpha,
            "kappa": args.kappa,
            "pr_delta_min": float(pr_min),
            "pr_delta_max": float(pr_max),
        },
        "node_names": nodes,
        "node_to_idx": node_to_idx,
        "edge_index": edge_index,          # shape [2, E]
        "edge_attr": edge_attr,            # shape [E] — scalar weight
        "edge_is_causal": edge_is_causal,  # shape [E] — bool
    }

    graph_data_path = os.path.join(args.out_dir, "graph_data.json")
    with open(graph_data_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"Saved PyG-style graph data → {graph_data_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_nonzero = int(nonzero_mask.sum())
    n_zero = n * n - n - n_nonzero  # exclude diagonal
    weight_arr = adj[nonzero_mask]
    print(f"\n{'='*60}")
    print(f"Graph summary")
    print(f"{'='*60}")
    print(f"  Nodes:              {n}")
    print(f"  Non-zero edges:     {n_nonzero}  (weight > 0)")
    print(f"  Zero edges:         {n_zero}  (absent pairs get weight=0)")
    print(f"  Causal edges (w=1): {len(CAUSAL_EDGES)}")
    print(f"  Weight stats:  min={weight_arr.min():.4f}  max={weight_arr.max():.4f}  "
          f"mean={weight_arr.mean():.4f}  median={np.median(weight_arr):.4f}")
    print(f"\nOutputs in: {os.path.abspath(args.out_dir)}")
    print(f"  node_list.json")
    print(f"  edge_list.csv")
    print(f"  adjacency_matrix.csv")
    print(f"  adjacency_matrix.npy")
    print(f"  graph_data.json")


if __name__ == "__main__":
    main()
