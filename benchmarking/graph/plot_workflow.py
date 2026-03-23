#!/usr/bin/env python3
"""
plot_workflow.py — Generate a pipeline diagram for the graph-based error classification workflow.
Output: graph/outputs/workflow_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "outputs" / "workflow_diagram.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 20, 13
STEP_W, STEP_H = 2.8, 0.7        # box dimensions for pipeline steps
DATA_W, DATA_H = 2.4, 0.5        # data artifact boxes

C_STEP  = "#2C7BB6"   # blue  — pipeline steps
C_DATA  = "#5AAE61"   # green — data artifacts
C_MODEL = "#D7191C"   # red   — model / training
C_EVAL  = "#FDAE61"   # orange — evaluation
C_GRAPH = "#762A83"   # purple — graph / causal input
C_TEXT  = "white"

def box(ax, x, y, w, h, label, color, fontsize=9, lines=None):
    """Draw a rounded rectangle with centered text (optional multi-line)."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor="white",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    if lines:
        full = label + "\n" + "\n".join(lines)
        ax.text(x, y, full, ha="center", va="center", fontsize=fontsize - 1,
                color=C_TEXT, zorder=4, wrap=True,
                multialignment="center", fontweight="bold" if not lines else "normal")
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color=C_TEXT, zorder=4, multialignment="center", fontweight="bold")

def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=2)

def label_arrow(ax, x, y, text, fontsize=7.5, color="#333333"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#1A1A2E")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(FIG_W / 2, 12.5,
        "Graph-Based Error Classification Pipeline  (GAIA, 117 traces)",
        ha="center", va="center", fontsize=14, color="white", fontweight="bold")

# ===========================================================================
# ROW 1 — Data preparation (Steps 1–2)   y ≈ 10.5
# ===========================================================================
Y1 = 10.5

# ── Step 01 ──────────────────────────────────────────────────────────────
S1x = 3.0
box(ax, S1x, Y1, STEP_W, STEP_H,
    "Step 01\n01_make_splits.py", C_STEP, fontsize=9)
ax.text(S1x, Y1 - 0.6,
        "117 GAIA traces → 92 train / 12 val / 13 test\n"
        "Stratified: rare (≤4) → 1 test + rest train\n"
        "Anchor (5–8) → ≥1 train; rescue-test if needed",
        ha="center", va="top", fontsize=7, color="#AAAACC")

# Input annotation files
box(ax, S1x - 2.2, Y1, 1.8, DATA_H,
    "processed_annotations\n_gaia/*.json", C_DATA, fontsize=7.5)
arrow(ax, S1x - 1.3, Y1, S1x - STEP_W / 2, Y1)

# Output splits
box(ax, S1x + 2.5, Y1, 2.0, DATA_H,
    "splits/\n{train,val,test}_ids.json", C_DATA, fontsize=7.5)
arrow(ax, S1x + STEP_W / 2, Y1, S1x + 1.5, Y1)

# ── Step 02 ──────────────────────────────────────────────────────────────
S2x = 10.5
box(ax, S2x, Y1, STEP_W, STEP_H,
    "Step 02\n02_build_span_dataset.py", C_STEP, fontsize=9)
ax.text(S2x, Y1 - 0.6,
        "Rule 1: 1 LLM per Step N  (correct or annotated)\n"
        "Rule 2: annotated non-Step-N spans + LLM siblings\n"
        "1,062 spans  |  387 annotated  |  675 correct  |  99.3% coverage",
        ha="center", va="top", fontsize=7, color="#AAAACC")

# Input: GAIA traces
box(ax, S2x - 2.4, Y1, 2.0, DATA_H,
    "data/GAIA/\n*.json (traces)", C_DATA, fontsize=7.5)
arrow(ax, S2x - 1.4, Y1, S2x - STEP_W / 2, Y1)

# Output span_dataset
box(ax, S2x + 2.5, Y1, 2.2, DATA_H,
    "data/span_dataset.jsonl\n(1,062 spans)", C_DATA, fontsize=7.5)
arrow(ax, S2x + STEP_W / 2, Y1, S2x + 1.4, Y1)

# Arrow splits → step02
arrow(ax, S1x + 2.5 + 1.0, Y1, S2x - 2.4 - 1.0, Y1, color="#888888")

# ===========================================================================
# ROW 2 — Encoding (Step 3)   y ≈ 7.8
# ===========================================================================
Y2 = 7.8

S3x = 7.0
box(ax, S3x, Y2, STEP_W, STEP_H,
    "Step 03\n03_encode_spans.py", C_STEP, fontsize=9)
ax.text(S3x, Y2 - 0.65,
        "Encoder: Qwen3-Embedding-8B (frozen, 4096-dim)\n"
        "All splits encoded separately\n"
        "Prototypes: mean-pool train spans per node (20 × 4096)",
        ha="center", va="top", fontsize=7, color="#AAAACC")

# Input from span_dataset
arrow(ax, S2x + 2.5 + 1.1, Y1 - 0.25,   # from span_dataset
      S3x - STEP_W / 2, Y2 + 0.25,
      color="#888888")
label_arrow(ax, (S2x + 2.5 + 1.1 + S3x - STEP_W / 2) / 2,
            (Y1 - 0.25 + Y2 + 0.25) / 2 + 0.15,
            "span texts", 7)

# Outputs
box(ax, S3x + 2.8, Y2 + 0.4, 2.4, DATA_H,
    "span_embeddings_{train,\nval,test}.pt  (4096-d)", C_DATA, fontsize=7.5)
box(ax, S3x + 2.8, Y2 - 0.4, 2.0, DATA_H,
    "prototypes.pt\n(20 × 4096)", C_DATA, fontsize=7.5)
arrow(ax, S3x + STEP_W / 2, Y2 + 0.25,  S3x + 2.8 - 1.2, Y2 + 0.4)
arrow(ax, S3x + STEP_W / 2, Y2 - 0.15,  S3x + 2.8 - 1.0, Y2 - 0.4)

# ===========================================================================
# Causal graph (left side, between rows 2 and 3)
# ===========================================================================
Gx, Gy = 1.8, 6.4
box(ax, Gx, Gy, 2.6, 0.65,
    "Causal Graph\ngraph_data.json", C_GRAPH, fontsize=8.5)
ax.text(Gx, Gy - 0.6,
        "19 error-type nodes\n155 directed edges (Suppes test)\nEdge weight: precedence + ΔPR̃",
        ha="center", va="top", fontsize=7, color="#CCAAEE")

# ===========================================================================
# ROW 3 — Graph input construction (Step 4)   y ≈ 5.5
# ===========================================================================
Y3 = 5.5

S4x = 7.0
box(ax, S4x, Y3, STEP_W, STEP_H,
    "Step 04\n04_build_graph_input.py", C_STEP, fontsize=9)
ax.text(S4x, Y3 - 0.65,
        "Load frozen graph + add Correct node (index 19, isolated)\n"
        "Node features x = prototypes.pt  (20 × 4096)\n"
        "Output: graph_input.pt",
        ha="center", va="top", fontsize=7, color="#AAAACC")

# prototypes → step04
arrow(ax, S3x + 2.8, Y2 - 0.4 - DATA_H / 2,
      S4x - STEP_W / 2, Y3 + 0.15, color="#888888")
label_arrow(ax, (S3x + 2.8 + S4x - STEP_W / 2) / 2 + 0.3,
            (Y2 - 0.65 + Y3 + 0.15) / 2, "prototypes", 7)

# causal graph → step04
arrow(ax, Gx + 1.3, Gy, S4x - STEP_W / 2, Y3 + 0.1, color="#AA88CC")
label_arrow(ax, (Gx + 1.3 + S4x - STEP_W / 2) / 2,
            (Gy + Y3 + 0.1) / 2 + 0.1, "topology + weights", 7, "#CCAAEE")

# Output graph_input.pt
box(ax, S4x + 2.7, Y3, 2.2, DATA_H,
    "graph_input.pt\n(x, edge_index, edge_weight)", C_DATA, fontsize=7.5)
arrow(ax, S4x + STEP_W / 2, Y3, S4x + 2.7 - 1.1, Y3)

# ===========================================================================
# ROW 4 — Training (Step 5)   y ≈ 3.5
# ===========================================================================
Y4 = 3.5

S5x = 7.0
box(ax, S5x, Y4, STEP_W + 0.2, STEP_H + 0.1,
    "Step 05  —  GAT Training\n05_train_gat.py", C_MODEL, fontsize=9)
ax.text(S5x, Y4 - 0.75,
        "Linear proj (4096→256) → GATLayer×2 (K=4 heads) → Z (20×256)\n"
        "Score: bilinear(h,z) + cos_sim(h,z)/τ  (τ learned, init 0.07)\n"
        "Loss: BCE_span + 0.1 × BCE_graph_recon\n"
        "AdamW · batch=32 · early-stop on val F1 (patience=10)\n"
        "Val threshold sweep [0.10…0.50] → save best τ_val in checkpoint",
        ha="center", va="top", fontsize=7, color="#FFCCCC")

# Inputs
arrow(ax, S4x + 2.7, Y3 - DATA_H / 2,
      S5x - STEP_W / 2, Y4 + 0.2, color="#888888")
label_arrow(ax, (S4x + 2.7 + S5x - STEP_W / 2) / 2 + 0.25,
            (Y3 - 0.25 + Y4 + 0.2) / 2, "graph_input.pt", 7)

arrow(ax, S3x + 2.8 + 1.2, Y2 + 0.4,        # span_embeddings
      S5x + 1.6, Y4 + 0.3, color="#888888")
label_arrow(ax, S5x + 2.9, Y4 + 1.1, "span_embs\n(train+val)", 7)

# Output model
box(ax, S5x + 3.2, Y4, 2.2, DATA_H,
    "models/best_model.pt\n(weights + τ_val)", C_MODEL, fontsize=7.5)
arrow(ax, S5x + STEP_W / 2 + 0.1, Y4, S5x + 3.2 - 1.1, Y4)

# ===========================================================================
# ROW 5 — Evaluation (Step 6)   y ≈ 1.5
# ===========================================================================
Y5 = 1.5

S6x = 7.0
box(ax, S6x, Y5, STEP_W + 0.2, STEP_H + 0.1,
    "Step 06  —  Evaluation\n06_evaluate.py", C_EVAL, fontsize=9)
ax.text(S6x, Y5 - 0.75,
        "GAT forward → Z  |  score each test span vs 20 nodes  |  sigmoid\n"
        "Trace aggregation: trace_max[i] = max over spans (max-pool)\n"
        "Predict: {i : trace_max[i] > τ_val}  (i ∈ 0..18, excl. Correct)\n"
        "Metrics: weighted / macro / micro F1  (19 error types)",
        ha="center", va="top", fontsize=7, color="#333300")

# Inputs
arrow(ax, S5x + 3.2, Y4 - DATA_H / 2,
      S6x + 1.8, Y5 + 0.3, color="#888888")
label_arrow(ax, S5x + 4.0, Y5 + 1.1, "best_model.pt", 7)

arrow(ax, S3x + 2.8 + 1.2, Y2 + 0.4,
      S6x - 1.8, Y5 + 0.3, color="#888888")
label_arrow(ax, S6x - 3.5, Y5 + 1.1, "span_embs\n(test)", 7)

# Output results
box(ax, S6x + 3.3, Y5, 2.4, DATA_H,
    "eval_results_test.json\nWeighted F1=0.338  τ=0.15", C_EVAL, fontsize=7.5)
arrow(ax, S6x + STEP_W / 2 + 0.1, Y5, S6x + 3.3 - 1.2, Y5)

# ===========================================================================
# Legend
# ===========================================================================
legend_items = [
    mpatches.Patch(color=C_STEP,  label="Pipeline step"),
    mpatches.Patch(color=C_DATA,  label="Data artifact"),
    mpatches.Patch(color=C_MODEL, label="Model / training"),
    mpatches.Patch(color=C_EVAL,  label="Evaluation"),
    mpatches.Patch(color=C_GRAPH, label="Causal graph (frozen)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.3, facecolor="#333355", labelcolor="white",
          edgecolor="white", bbox_to_anchor=(0.0, 0.0))

# ===========================================================================
# Save
# ===========================================================================
plt.tight_layout(pad=0.3)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUTPUT_PATH}")
