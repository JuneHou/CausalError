"""
visualize_graphs.py — TRAIL-adapted visualizer for the causal-only graph.

Default output: single figure `graph_causal.png` of the intervention-validated
causal graph (weights = |Δ(A→B)|). This is what the +CG / +GI+SI causal-only
inference conditions actually inject into prompts.

Optional corr ≥ τ graph: pass `--corr_threshold τ` to additionally render
`graph_corr_t<τ>.png` showing the edge set used in the corr-threshold
ablation — Suppes edges with geomean ≥ τ unioned with the validated causal
edges, weighted by geomean. Mirrors the inference logic in
`benchmarking/eval/run_eval_graph_inject_vllm.py:load_graph_edges()`.

Layout: columns by topological level (from hierarchy_levels.json if provided,
otherwise computed via networkx topological_generations on the validated DAG).

Usage (from trail-benchmark/):

    # default: causal-only figure
    python causal/graph/visualize_graphs.py \\
        --effect_edges  benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json \\
        --hierarchy     benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/hierarchy_levels.json \\
        --out_dir       figures/graph

    # also render the corr ≥ 0.2 ablation graph
    python causal/graph/visualize_graphs.py \\
        --effect_edges  benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json \\
        --suppes        benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json \\
        --hierarchy     benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/hierarchy_levels.json \\
        --corr_threshold 0.2 \\
        --out_dir       figures/graph
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# TRAIL category → top-level branch
# ---------------------------------------------------------------------------
BRANCH = {
    # Reasoning Errors
    "Language-only":                    "reasoning",
    "Tool-related":                     "reasoning",
    "Poor Information Retrieval":       "reasoning",
    "Tool Output Misinterpretation":    "reasoning",
    "Incorrect Problem Identification": "reasoning",
    # System Execution Errors
    "Tool Selection Errors":            "system",
    "Formatting Errors":                "system",
    "Instruction Non-compliance":       "system",
    "Tool Definition Issues":           "system",
    "Environment Setup Errors":         "system",
    "Authentication Errors":            "system",
    "Service Errors":                   "system",
    "Resource Not Found":               "system",
    "Resource Exhaustion":              "system",
    "Timeout Issues":                   "system",
    "Rate Limiting":                    "system",
    # Planning and Coordination Errors
    "Context Handling Failures":        "planning",
    "Resource Abuse":                   "planning",
    "Goal Deviation":                   "planning",
    "Task Orchestration":               "planning",
}

BRANCH_COLORS = {
    "reasoning": "#4C9BE8",  # blue
    "system":    "#E87C4C",  # orange
    "planning":  "#5CBF7A",  # green
}

BRANCH_NAMES = {
    "reasoning": "Reasoning Errors",
    "system":    "System Execution Errors",
    "planning":  "Planning & Coordination Errors",
}


def short_label(name: str, width: int = 14) -> str:
    """Wrap a long category name onto multiple lines for display inside a node."""
    return "\n".join(textwrap.wrap(name, width=width))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_validated_causal(effect_edges_path):
    """Causal-only graph from intervention validation.
    Weight = |Δ(A→B)|. Only edges with validated=True are returned."""
    data = json.load(open(effect_edges_path))
    out = []
    for v in data["edges"].values():
        if not v.get("validated"):
            continue
        d = v.get("delta")
        if d is None:
            continue
        out.append({"a": v["a"], "b": v["b"], "weight": abs(d)})
    return out


def load_corr_graph(suppes_path, validated_pairs, corr_threshold):
    """Reproduce the corr ≥ τ injection set used at inference:
    Suppes edges with geomean ≥ τ ∪ validated causal edges, weighted by geomean.
    Mirrors `eval/run_eval_graph_inject_vllm.py:load_graph_edges()`.
    """
    data = json.load(open(suppes_path))
    out = []
    seen = set()
    # Pass 1: Suppes edges with geomean >= τ
    for e in data["edges"]:
        geomean = (max(0.0, e.get("p_b_given_a", 0.0)) *
                   max(0.0, e.get("pr_delta", 0.0))) ** 0.5
        if geomean >= corr_threshold:
            out.append({"a": e["a"], "b": e["b"], "weight": geomean})
            seen.add((e["a"], e["b"]))
    # Pass 2: union with validated edges, using their geomean lookup from Suppes
    suppes_lookup = {(e["a"], e["b"]): e for e in data["edges"]}
    for (a, b) in validated_pairs:
        if (a, b) in seen:
            continue
        s = suppes_lookup.get((a, b))
        if s is None:
            continue
        geomean = (max(0.0, s.get("p_b_given_a", 0.0)) *
                   max(0.0, s.get("pr_delta", 0.0))) ** 0.5
        out.append({"a": a, "b": b, "weight": geomean})
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def levels_from_hierarchy_file(path):
    h = json.load(open(path))
    out = {}
    for lvl_key, nodes in h["levels"].items():
        idx = int(lvl_key.split("_")[1])
        for n in nodes:
            out[n] = idx
    return out


def levels_from_dag(edges):
    """Compute topological levels from the validated causal edges."""
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["a"], e["b"])
    if not nx.is_directed_acyclic_graph(G):
        return {n: 0 for n in G.nodes()}
    out = {}
    for i, gen in enumerate(nx.topological_generations(G)):
        for n in gen:
            out[n] = i
    return out


def make_positions(all_nodes, level_map):
    n_levels = max(level_map.values()) + 1 if level_map else 1
    cols = {i: [] for i in range(n_levels + 1)}  # +1 spare column for orphans
    for n in all_nodes:
        cols[level_map.get(n, 0)].append(n)

    pos = {}
    used_levels = sorted([k for k, v in cols.items() if v])
    n_used = len(used_levels)
    for ci, lvl in enumerate(used_levels):
        # cluster within column by branch (reasoning top, system mid, planning bottom)
        nodes = cols[lvl]
        nodes.sort(key=lambda n: (
            list(BRANCH_COLORS.keys()).index(BRANCH.get(n, "system")), n
        ))
        m = len(nodes)
        ys = np.linspace(0.92, 0.08, m) if m > 1 else [0.5]
        x = ci / max(1, n_used - 1) * 4.0  # spread columns 0..4
        for n, y in zip(nodes, ys):
            pos[n] = (x, y)
    return pos


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def build_graph(edges, weight_key="weight"):
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["a"], e["b"], weight=e[weight_key])
    return G


def draw_graph(ax, G, pos, title, weight_range=(0.05, 0.87),
               weight_label="weight",
               show_edge_labels=True, label_threshold=0.0):
    xs = [p[0] for p in pos.values()]
    x_min, x_max = (min(xs), max(xs)) if xs else (0, 1)
    ax.set_xlim(x_min - 0.6, x_max + 0.6)
    ax.set_ylim(-0.05, 1.10)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Nodes
    node_colors = [BRANCH_COLORS[BRANCH.get(n, "system")] for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2600,
        alpha=0.92,
    )

    short = {n: short_label(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=short, ax=ax,
        font_size=6.5, font_weight="bold", font_color="white",
    )

    edges_data = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    if not edges_data:
        return

    w_min, w_max = weight_range
    cmap = plt.cm.YlOrRd
    for u, v, w in edges_data:
        norm_w = (w - w_min) / (w_max - w_min + 1e-9)
        norm_w = max(0.0, min(1.0, norm_w))
        color = cmap(0.25 + norm_w * 0.7)
        width = 1.0 + norm_w * 4.5
        rad = 0.18 if (v, u) in G.edges() else 0.10
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=[color],
            width=width,
            arrows=True,
            arrowsize=16,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=28,
            min_target_margin=28,
        )

    if show_edge_labels:
        edge_labels = {
            (u, v): f"{w:.2f}"
            for u, v, w in edges_data
            if w >= label_threshold
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
            label_pos=0.4,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label(weight_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def add_legend(ax):
    patches = [
        mpatches.Patch(color=BRANCH_COLORS[k], label=BRANCH_NAMES[k])
        for k in ["reasoning", "system", "planning"]
    ]
    ax.legend(handles=patches, loc="lower center",
              fontsize=8, framealpha=0.85, ncol=3,
              bbox_to_anchor=(0.5, -0.06))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect_edges",  required=True,
                        help="effect_edges.json from intervention validation (drives the causal-only plot)")
    parser.add_argument("--suppes",        default=None,
                        help="suppes_graph.json — only required if --corr_threshold is set")
    parser.add_argument("--hierarchy",     default=None,
                        help="Optional hierarchy_levels.json for layout; else derived from validated DAG")
    parser.add_argument("--corr_threshold", type=float, default=None,
                        help="If set, also render the corr ≥ τ ablation graph "
                             "(Suppes edges with geomean ≥ τ ∪ validated causal). "
                             "Requires --suppes.")
    parser.add_argument("--show_isolated", action="store_true",
                        help="Include all Suppes-universe error categories as nodes "
                             "(isolated if no validated edge connects them). Requires --suppes.")
    parser.add_argument("--out_dir",       required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    causal_edges = load_validated_causal(args.effect_edges)
    G_causal = build_graph(causal_edges)
    all_nodes = set(G_causal.nodes())

    corr_edges = None
    G_corr = None
    if args.corr_threshold is not None:
        if not args.suppes:
            parser.error("--corr_threshold requires --suppes")
        validated_pairs = {(e["a"], e["b"]) for e in causal_edges}
        corr_edges = load_corr_graph(args.suppes, validated_pairs, args.corr_threshold)
        G_corr = build_graph(corr_edges)
        all_nodes |= set(G_corr.nodes())

    # Optionally surface error categories that the pipeline considered but for
    # which no edge survived intervention validation. Helps the reader see the
    # full taxonomy under test, not just the connected subset.
    if args.show_isolated:
        if not args.suppes:
            parser.error("--show_isolated requires --suppes")
        suppes_data = json.load(open(args.suppes))
        suppes_nodes = ({e["a"] for e in suppes_data["edges"]} |
                        {e["b"] for e in suppes_data["edges"]})
        G_causal.add_nodes_from(suppes_nodes)
        if G_corr is not None:
            G_corr.add_nodes_from(suppes_nodes)
        all_nodes |= suppes_nodes

    all_nodes = sorted(all_nodes)

    # Layout
    if args.hierarchy and os.path.exists(args.hierarchy):
        level_map = levels_from_hierarchy_file(args.hierarchy)
    else:
        level_map = levels_from_dag(causal_edges)
    for n in all_nodes:
        level_map.setdefault(n, 0)

    pos = make_positions(all_nodes, level_map)

    causal_weights = [e["weight"] for e in causal_edges]
    causal_range = (min(causal_weights), max(causal_weights)) if causal_weights else (0, 1)

    CAUSAL_LABEL = r"|Δ(A→B)|  (intervention-validated effect)"

    # ---- Validated causal-only graph (default) ----
    fig, ax = plt.subplots(figsize=(13, 7))
    draw_graph(ax, G_causal, pos,
               title=f"Validated Causal Graph  ({len(causal_edges)} edges, weight=|Δ|)",
               weight_range=causal_range,
               weight_label=CAUSAL_LABEL,
               show_edge_labels=True,
               label_threshold=0.0)
    add_legend(ax)
    fig.tight_layout()
    out_c = os.path.join(args.out_dir, "graph_causal.png")
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_c}  ({len(causal_edges)} validated edges, weight = |Δ|)")

    # ---- Optional: corr ≥ τ ablation graph ----
    if corr_edges is not None:
        corr_weights = [e["weight"] for e in corr_edges]
        corr_range = (min(corr_weights), max(corr_weights)) if corr_weights else (0, 1)
        SUPPES_LABEL = r"geomean = $\sqrt{P(B|A)\cdot \Delta\mathrm{PR}}$"

        fig, ax = plt.subplots(figsize=(13, 7))
        draw_graph(ax, G_corr, pos,
                   title=(f"Corr ≥ {args.corr_threshold} Ablation Graph  "
                          f"({len(corr_edges)} edges, weight=geomean)"),
                   weight_range=corr_range,
                   weight_label=SUPPES_LABEL,
                   show_edge_labels=False)
        add_legend(ax)
        fig.tight_layout()
        tag = f"{args.corr_threshold}".replace(".", "p")
        out_corr = os.path.join(args.out_dir, f"graph_corr_t{tag}.png")
        fig.savefig(out_corr, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_corr}  ({len(corr_edges)} edges; "
              f"Suppes geomean ≥ {args.corr_threshold} ∪ validated)")


if __name__ == "__main__":
    main()
