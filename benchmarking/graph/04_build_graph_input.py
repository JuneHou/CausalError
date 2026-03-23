#!/usr/bin/env python3
"""
04_build_graph_input.py — Assemble the frozen graph + prototype features.

Loads graph_data.json (19 error nodes, 155 Suppes edges) and prototypes.pt
(20, 4096 — train-only mean embeddings), then saves graph_input.pt for use
in training.

The "Correct" node (index 19) is isolated — no edges. Its prototype is the
last row of prototypes.pt.

Output: graph/data/graph_input.pt
    {
        "x":            Tensor(20, 4096)  — prototype node features (fp32)
        "edge_index":   Tensor(2, E)      — directed edges (src, dst)
        "edge_weight":  Tensor(E,)        — Suppes / causal weights in [0,1]
        "edge_is_causal": Tensor(E,)      — bool, 1.0 for validated causal edges
        "n_nodes":      int = 20
        "n_edges":      int = E
        "node_names":   list[str]         — 20 names (index 0..18 error, 19 Correct)
    }
"""

import json
import logging
from pathlib import Path

import torch

BENCH_DIR   = Path(__file__).resolve().parent.parent
GRAPH_DATA  = BENCH_DIR / "graph" / "outputs" / "graph_data.json"
PROTO_FILE  = BENCH_DIR / "graph" / "data" / "prototypes.pt"
LABEL_MAP   = BENCH_DIR / "graph" / "data" / "label_to_node_idx.json"
OUTPUT_DIR  = BENCH_DIR / "graph" / "data"
OUTPUT_FILE = OUTPUT_DIR / "graph_input.pt"

CORRECT_NODE = "Correct"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph data (19 nodes, 155 Suppes edges)
    # ------------------------------------------------------------------
    if not GRAPH_DATA.exists():
        raise FileNotFoundError(f"{GRAPH_DATA} not found — run build_graph_data.py first")

    graph_data = json.loads(GRAPH_DATA.read_text())
    node_names: list[str] = graph_data["node_names"]          # 19 error types, sorted
    assert len(node_names) == 19, f"Expected 19 nodes, got {len(node_names)}"

    # Add Correct node
    all_names = node_names + [CORRECT_NODE]                    # index 19
    n_nodes = 20

    # ------------------------------------------------------------------
    # Verify alignment with label_to_node_idx
    # ------------------------------------------------------------------
    if LABEL_MAP.exists():
        label_map = json.loads(LABEL_MAP.read_text())
        for name, idx in label_map.items():
            if name == CORRECT_NODE:
                assert idx == 19, f"Correct node should be index 19, got {idx}"
            else:
                assert all_names[idx] == name, (
                    f"Node mismatch at idx {idx}: expected {all_names[idx]!r} got {name!r}"
                )
        log.info("Label map alignment verified (%d nodes)", len(label_map))

    # ------------------------------------------------------------------
    # Build edge tensors
    # ------------------------------------------------------------------
    edge_index_raw = graph_data["edge_index"]          # [[src...], [dst...]]
    edge_attr      = graph_data["edge_attr"]           # [w, ...]
    edge_is_causal = graph_data["edge_is_causal"]      # [bool, ...]

    edge_index    = torch.tensor(edge_index_raw, dtype=torch.long)   # (2, E)
    edge_weight   = torch.tensor(edge_attr, dtype=torch.float32)     # (E,)
    edge_is_causal_t = torch.tensor(
        [float(b) for b in edge_is_causal], dtype=torch.float32
    )                                                                  # (E,)

    n_edges = edge_index.shape[1]
    log.info("Graph: %d nodes, %d directed edges (%d causal)",
             n_nodes, n_edges, int(edge_is_causal_t.sum().item()))

    # ------------------------------------------------------------------
    # Load prototype features
    # ------------------------------------------------------------------
    if not PROTO_FILE.exists():
        raise FileNotFoundError(f"{PROTO_FILE} not found — run 03_encode_spans.py first")

    prototypes = torch.load(PROTO_FILE, weights_only=True)    # (20, 4096)
    assert prototypes.shape == (20, 4096), (
        f"Expected prototypes shape (20, 4096), got {tuple(prototypes.shape)}"
    )
    log.info("Prototypes loaded: shape=%s, dtype=%s", tuple(prototypes.shape), prototypes.dtype)

    # ------------------------------------------------------------------
    # Save graph_input.pt
    # ------------------------------------------------------------------
    graph_input = {
        "x":              prototypes,          # (20, 4096) fp32
        "edge_index":     edge_index,          # (2, E)
        "edge_weight":    edge_weight,         # (E,) in [0,1]
        "edge_is_causal": edge_is_causal_t,    # (E,) {0,1}
        "n_nodes":        n_nodes,
        "n_edges":        n_edges,
        "node_names":     all_names,
    }
    torch.save(graph_input, OUTPUT_FILE)
    log.info("Saved → %s", OUTPUT_FILE)

    # Summary
    print(f"\n{'='*50}")
    print("Graph input summary")
    print(f"{'='*50}")
    print(f"  Nodes:       {n_nodes}  (19 error + 1 Correct)")
    print(f"  Edges:       {n_edges}  (directed Suppes + causal)")
    print(f"  Causal:      {int(edge_is_causal_t.sum())}  (weight=1.0)")
    print(f"  Feat dim:    {prototypes.shape[1]}")
    print(f"  Weight min:  {edge_weight.min():.4f}")
    print(f"  Weight max:  {edge_weight.max():.4f}")
    print(f"\nNodes:")
    for i, name in enumerate(all_names):
        tag = " [Correct, isolated]" if i == 19 else ""
        print(f"  {i:2d}  {name}{tag}")


if __name__ == "__main__":
    main()
