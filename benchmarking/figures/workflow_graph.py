"""
TRAIL Benchmark — Main Workflow Figure
Prototype for the paper's main figure.

Two-phase layout (top → bottom):
  Phase 1 (left column):  Causal Graph Construction
  Phase 2 (right column): Causal Intervention & Validation

Run:
    python figures/workflow_graph.py          # saves workflow_graph.pdf + .png
    python figures/workflow_graph.py --show   # also pops up interactive window
"""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── colour palette ──────────────────────────────────────────────────────────
C_DATA    = "#E8F4FD"   # light blue  – data / artifact nodes
C_PHASE1  = "#EAF5EA"   # light green – causal graph steps
C_PHASE2  = "#FFF5E6"   # light orange – intervention steps
C_LLM     = "#F3E8FD"   # light purple – LLM-assisted steps
C_OUT     = "#FFF0F0"   # light red   – final output nodes
C_EDGE    = "#555555"
C_HEADER  = "#2C3E50"

FONT = "DejaVu Sans"


def rbox(ax, xy, w, h, text, color, fontsize=8.5, bold=False,
         sub=None, subsize=7.5, radius=0.015, alpha=0.92, zorder=3):
    """Draw a rounded rectangle with centred label (+ optional sub-label)."""
    x, y = xy
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=0.9, edgecolor="#888888",
                         facecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    dy = 0.008 if sub else 0
    ax.text(x, y + dy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            fontfamily=FONT, zorder=zorder + 1,
            wrap=True, multialignment="center")
    if sub:
        ax.text(x, y - 0.022, sub, ha="center", va="center",
                fontsize=subsize, color="#666666",
                fontfamily=FONT, zorder=zorder + 1,
                style="italic")
    return box


def arrow(ax, src, dst, color=C_EDGE, lw=1.1, style="->",
          rad=0.0, label="", lfs=7, zorder=2):
    """Draw a curved arrow between two (x, y) points."""
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}",
                                shrinkA=4, shrinkB=4),
                zorder=zorder)
    if label:
        mx = (src[0] + dst[0]) / 2 + 0.01
        my = (src[1] + dst[1]) / 2
        ax.text(mx, my, label, fontsize=lfs, color=color,
                ha="left", va="center", fontfamily=FONT, zorder=zorder + 1)


def phase_banner(ax, x, y, w, h, text, color, alpha=0.18):
    """Light background band behind a phase."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=0.02",
                         linewidth=1.2, edgecolor=color,
                         facecolor=color, alpha=alpha, zorder=1)
    ax.add_patch(box)
    ax.text(x + w/2, y + h + 0.01, text,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=color, fontfamily=FONT, zorder=2)


# ─── layout constants ─────────────────────────────────────────────────────────
BW, BH   = 0.28, 0.055   # default box width / height
BW_WIDE  = 0.32
GAP      = 0.095          # vertical gap between box centres

# x-centres for each column
CX1 = 0.22   # phase 1 (causal construction)
CX2 = 0.76   # phase 2 (intervention)
DX  = 0.50   # shared data / output column

# top y for each phase
Y0 = 0.94


def build_figure():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── phase background banners ──────────────────────────────────────────────
    phase_banner(ax, 0.03, 0.36, 0.44, 0.55,
                 "Phase 1 · Causal Graph Construction", "#2E86AB")
    phase_banner(ax, 0.53, 0.10, 0.44, 0.81,
                 "Phase 2 · Causal Intervention & Validation", "#E07B39")

    # ══════════════════════════════════════════════════════════════════════════
    # INPUTS  (top, spanning both phases)
    # ══════════════════════════════════════════════════════════════════════════
    Y_IN = 0.95
    rbox(ax, (0.24, Y_IN), 0.22, BH,
         "GAIA Agent Traces\n(OpenInference OTEL)", C_DATA,
         fontsize=8, bold=True)
    rbox(ax, (0.52, Y_IN), 0.22, BH,
         "TRAIL Annotations\n(per-span error labels)", C_DATA,
         fontsize=8, bold=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Preprocessing
    # ══════════════════════════════════════════════════════════════════════════
    ys = {}   # store y positions keyed by node id

    # Preprocessing sub-group
    pp_top   = Y_IN - 0.08
    pp_nodes = [
        ("filt",   "Filter & Align Traces\n(match trace ↔ annotation)",   pp_top),
        ("order",  "Build Span Timeline\n(LLM/TOOL/CHAIN spans, ranked)",  pp_top - GAP),
        ("onset",  "Build Onset Table\n(present / rank / ties per error)", pp_top - 2*GAP),
    ]
    for nid, label, y in pp_nodes:
        rbox(ax, (CX1, y), BW, BH, label, C_PHASE1, fontsize=8)
        ys[nid] = y

    # arrows: inputs → filter
    arrow(ax, (0.24, Y_IN - BH/2), (CX1, ys["filt"] + BH/2))
    arrow(ax, (0.52, Y_IN - BH/2), (CX1, ys["filt"] + BH/2), rad=0.15)

    # preprocessing chain
    arrow(ax, (CX1, ys["filt"] - BH/2),  (CX1, ys["order"] + BH/2))
    arrow(ax, (CX1, ys["order"] - BH/2), (CX1, ys["onset"] + BH/2))

    # intermediate artefact label
    ax.text(CX1 + BW/2 + 0.01, ys["order"], "onsets_gaia.jsonl",
            fontsize=6.5, color="#888", va="center", fontfamily=FONT)

    # ── CAPRI Pipeline ────────────────────────────────────────────────────────
    c_top = ys["onset"] - GAP
    capri_nodes = [
        ("opairs", "Order Pairs\n(A precedes B per trace)", c_top),
        ("suppes", "Suppes Screening\n(temporal priority + prob. raising)", c_top - GAP),
        ("capri",  "CAPRI Pruning\n(greedy DAG learning, BIC/AIC score)", c_top - 2*GAP),
        ("boot",   "Bootstrap Stability\n(100× resampling → edge freq.)",  c_top - 3*GAP),
        ("shuf",   "Shuffle Control\n(50× null, permuted onset ranks)",    c_top - 4*GAP),
    ]
    for nid, label, y in capri_nodes:
        rbox(ax, (CX1, y), BW, BH, label, C_PHASE1, fontsize=8)
        ys[nid] = y

    arrow(ax, (CX1, ys["onset"]  - BH/2), (CX1, ys["opairs"] + BH/2))
    arrow(ax, (CX1, ys["opairs"] - BH/2), (CX1, ys["suppes"] + BH/2))
    arrow(ax, (CX1, ys["suppes"] - BH/2), (CX1, ys["capri"]  + BH/2))
    arrow(ax, (CX1, ys["capri"]  - BH/2), (CX1, ys["boot"]   + BH/2))
    arrow(ax, (CX1, ys["boot"]   - BH/2), (CX1, ys["shuf"]   + BH/2))

    # ── Causal Graph output ───────────────────────────────────────────────────
    y_cg = ys["shuf"] - GAP
    rbox(ax, (CX1, y_cg), BW_WIDE, BH,
         "Causal Error Graph\n(12 edges · 3 hierarchy levels)",
         C_OUT, fontsize=8.5, bold=True)
    ys["cg"] = y_cg
    arrow(ax, (CX1, ys["shuf"] - BH/2), (CX1, y_cg + BH/2))

    # stability feeds back into graph (side arrow)
    arrow(ax, (CX1 + BW/2, ys["boot"]),
               (CX1 + BW_WIDE/2, y_cg + BH/2),
               color="#AAAAAA", lw=0.8, rad=-0.3,
               label="filter by\nstability", lfs=6)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Intervention Pipeline
    # ══════════════════════════════════════════════════════════════════════════
    i_top = Y_IN - 0.08

    int_nodes = [
        ("filt2",  "Filter Eligible Traces\n(A→B pairs with required errors)", C_PHASE2, i_top),
        ("cases",  "Build A-Instances & Edge Pairs\n(snippet · context · B-types)", C_PHASE2, i_top - GAP),
        ("patch",  "LLM Patch Generation\n(GPT-4o · patch_library templates + post-check)", C_LLM, i_top - 2*GAP),
        ("rerun",  "Counterfactual Rerun\n(o3-mini · replay original tool outputs)", C_LLM, i_top - 3*GAP),
        ("judA",   "Judge A — Error Resolved?\n(GPT-4o · strict binary verdict)", C_LLM, i_top - 4*GAP),
        ("judB",   "Judge B — Effect on B?\n(GPT-4o · fan-out per edge pair)", C_LLM, i_top - 5*GAP),
        ("agg",    "Effect Aggregation\nΔ(A→B) = Δ P(B present | A fixed)", C_PHASE2, i_top - 6*GAP),
    ]
    for nid, label, col, y in int_nodes:
        rbox(ax, (CX2, y), BW, BH, label, col, fontsize=8)
        ys[nid] = y

    # chain arrows
    prev = "filt2"
    for nid, _, _, _ in int_nodes[1:]:
        arrow(ax, (CX2, ys[prev] - BH/2), (CX2, ys[nid] + BH/2))
        prev = nid

    # inputs → filt2
    arrow(ax, (0.24, Y_IN - BH/2), (CX2, ys["filt2"] + BH/2), rad=-0.15)
    arrow(ax, (0.52, Y_IN - BH/2), (CX2, ys["filt2"] + BH/2))

    # Judge A gate: only resolved A-instances proceed
    ax.annotate("", xy=(CX2, ys["judB"] + BH/2),
                xytext=(CX2, ys["judA"] - BH/2),
                arrowprops=dict(arrowstyle="->", color="#E07B39",
                                lw=1.2, connectionstyle="arc3,rad=0",
                                shrinkA=4, shrinkB=4), zorder=2)
    ax.text(CX2 + BW/2 + 0.01, (ys["judA"] + ys["judB"]) / 2,
            "resolved=True\nonly", fontsize=6.5, color="#E07B39",
            ha="left", va="center", fontfamily=FONT)

    # ── Validated Edges output ────────────────────────────────────────────────
    y_ve = ys["agg"] - GAP
    rbox(ax, (CX2, y_ve), BW_WIDE, BH,
         "Validated Causal Edges\n(10 / 12 edges · Δ(A→B) < −0.15)",
         C_OUT, fontsize=8.5, bold=True)
    ys["ve"] = y_ve
    arrow(ax, (CX2, ys["agg"] - BH/2), (CX2, y_ve + BH/2))

    # ── Causal Graph feeds Phase 2 ────────────────────────────────────────────
    arrow(ax, (CX1 + BW_WIDE/2, ys["cg"]),
               (CX2 - BW/2, ys["filt2"]),
               color="#2E86AB", lw=1.4, rad=-0.18,
               label="causal graph\n(A→B candidates)", lfs=6.5)

    # ── Comparison arrow ──────────────────────────────────────────────────────
    arrow(ax, (CX1, ys["cg"] - BH/2), (DX - 0.03, 0.075),
               color="#AAAAAA", lw=0.9, rad=0.12)
    arrow(ax, (CX2, ys["ve"] - BH/2), (DX + 0.03, 0.075),
               color="#AAAAAA", lw=0.9, rad=-0.12)

    rbox(ax, (DX, 0.055), 0.36, BH,
         "Observational Graph vs. Interventional Evidence\n"
         "→ confirm / refute each hypothesised edge",
         "#FFFDE7", fontsize=8, bold=False)

    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    legend_items = [
        (C_DATA,   "Input data / artefact"),
        (C_PHASE1, "Causal graph step"),
        (C_PHASE2, "Intervention step"),
        (C_LLM,    "LLM-assisted step"),
        (C_OUT,    "Pipeline output"),
    ]
    lx, ly = 0.015, 0.27
    for i, (col, label) in enumerate(legend_items):
        patch = FancyBboxPatch((lx, ly - i * 0.028), 0.018, 0.018,
                               boxstyle="round,pad=0,rounding_size=0.003",
                               linewidth=0.7, edgecolor="#888888",
                               facecolor=col, alpha=0.9, zorder=4)
        ax.add_patch(patch)
        ax.text(lx + 0.024, ly - i * 0.028 + 0.009, label,
                fontsize=7.5, va="center", fontfamily=FONT, zorder=5)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.995,
            "TRAIL Benchmark — Causal Graph Construction & Intervention Pipeline",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=C_HEADER, fontfamily=FONT)

    fig.tight_layout(pad=0)
    return fig


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Display the figure interactively")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to save output files")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    fig = build_figure()

    out_pdf = os.path.join(args.out_dir, "workflow_graph.pdf")
    out_png = os.path.join(args.out_dir, "workflow_graph.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")

    if args.show:
        plt.show()
