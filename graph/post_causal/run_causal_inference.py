#!/usr/bin/env python3
"""
post_causal/run_causal_inference.py — Inference-time causal intervention on baseline predictions.

Inspired by: Tian et al. "Causal Multi-Label Learning for Image Classification",
Neural Networks, Vol.167, pp.626-637, 2023. DOI: 10.1016/j.neunet.2023.08.052

─────────────────────────────────────────────────────────────────────────────
CMLL PIPELINE SUMMARY (Tian et al. 2023)
─────────────────────────────────────────────────────────────────────────────
CMLL addresses multi-label image classification using a two-stream architecture
with causal intervention at inference time (not training time):

  Training:
    Global stream  — backbone CNN over full image → correlation-based predictions
    Local stream   — backbone CNN over high-confidence attention crop regions
    Both streams trained jointly; local stream additionally regularised with
    causal structure to break spurious co-occurrence bias.

  Inference:
    1. Global stream produces per-label attention maps and base probabilities.
    2. High-confidence regions (do(X=x) in Pearl's do-calculus notation) are
       selected from the attention maps.
    3. Local stream re-scores labels using only these causally identified crops.
    4. Final prediction = causal local scores, NOT the global correlational scores.

  Key idea — do-calculus framing:
    Standard observation: P(Y | X)        — may include spurious paths
    Causal intervention:  P(Y | do(X=x))  — only causal paths remain

    By intervening on which visual evidence is used (selecting only high-confidence
    causal crops), CMLL prevents spurious co-occurrence from inflating predictions.
    Concretely: label B is only predicted if there is direct visual evidence FOR B,
    not merely because B co-occurs with A in the training distribution.

─────────────────────────────────────────────────────────────────────────────
OUR ADAPTATION FOR TRAIL
─────────────────────────────────────────────────────────────────────────────
We adapt the core CMLL principle — applying causal structure at inference to
suppress spurious predictions — to our trace-level multi-label setting.

  Analogy to CMLL:
    Global stream        ↔  Baseline scorer  (bilinear+cosine on span embeddings)
    Causal graph         ↔  Suppes graph     (A→B: A causally precedes B)
    do(X=x) intervention ↔  Causal gate      (suppress B if A is not evidenced)
    Spurious co-occurrence ↔ Over-prediction (recall=0.85, precision=0.34)

  Mechanism (Causal Predecessor Gating):

    For each trace t and error type i:
      1. Compute baseline probability:  p_base(t, i)  ∈ [0,1]
      2. Identify causal predecessors:  pred(i) = {j : A[j,i] > 0}
      3. Compute predecessor support:
            gate(t, i) = Σ_j  A_col[j,i] · p_base(t, j)
         where A_col is column-normalised Suppes matrix (Σ_j A_col[j,i] = 1).
         gate(t,i) ∈ [0,1]: high when predecessors of i are confidently predicted.
      4. Apply causal intervention:
            p_causal(t, i) = p_base(t, i) · (β + (1-β) · gate(t, i))
         β ∈ [0,1]: minimum gate floor.
            β = 1.0  → no intervention  (p_causal = p_base)
            β = 0.0  → full suppression when no predecessor support
         For nodes with no Suppes predecessors, gate = 1.0 (no intervention).

  Contrast with additive Suppes propagation (run_suppes_inference.py):
    Additive:     p_final = p_base + α · A_norm · p_base   ← always increases probs
    Causal gate:  p_final = p_base · (β + (1-β) · gate)    ← only decreases probs

  The causal gate is precision-oriented: it suppresses error-type predictions that
  lack support from their causal predecessors, directly addressing the baseline's
  over-prediction problem (high recall, low precision).

  β and threshold are tuned jointly on the val split.

─────────────────────────────────────────────────────────────────────────────
Usage (from trail-benchmark/):
    python graph/post_causal/run_causal_inference.py
    python graph/post_causal/run_causal_inference.py --beta 0.3 --threshold 0.4
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPH_DIR    = Path(__file__).resolve().parent.parent   # trail-benchmark/graph/
BASELINE_DIR = GRAPH_DIR / "baseline"
POST_DIR     = Path(__file__).resolve().parent          # trail-benchmark/graph/post_causal/
DATA_DIR     = GRAPH_DIR / "data"
MODEL_DIR    = BASELINE_DIR / "models"
OUTPUT_DIR   = POST_DIR / "outputs"

DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CORRECT_IDX = 19

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import shared helpers
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_gat      = _load_module("gat_module",  GRAPH_DIR / "05_train_gat.py")
_eval_mod = _load_module("eval_module", GRAPH_DIR / "06_evaluate.py")
_base_mod = _load_module("base_module", BASELINE_DIR / "run_baseline.py")

build_flat_span_list = _gat.build_flat_span_list
build_adj_matrix     = _gat.build_adj_matrix
NoGraphBaseline      = _base_mod.NoGraphBaseline
compute_metrics      = _eval_mod.compute_metrics
print_metrics        = _eval_mod.print_metrics


# ---------------------------------------------------------------------------
# Build column-normalised Suppes matrix
# ---------------------------------------------------------------------------

def build_causal_matrix(gi: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        A_raw     : (19, 19) raw Suppes weights
        A_col     : (19, 19) column-normalised Suppes matrix
                    A_col[j, i] = A_raw[j, i] / sum_k A_raw[k, i]
                    Column i: normalised weights of i's causal predecessors.
                    Rows with column sum = 0 → no predecessors → gate = 1.
    """
    adj_full = build_adj_matrix(gi["edge_index"], gi["edge_weight"], gi["n_nodes"])
    A_raw = adj_full[:CORRECT_IDX, :CORRECT_IDX].numpy().astype(np.float32)

    col_sums = A_raw.sum(axis=0, keepdims=True)          # (1, 19)
    no_pred  = (col_sums == 0)                           # labels with no predecessors
    col_sums = np.where(no_pred, 1.0, col_sums)         # avoid div-by-zero
    A_col    = A_raw / col_sums

    n_labels_with_pred = int((~no_pred).sum())
    log.info("Suppes matrix: %d / %d error types have causal predecessors",
             n_labels_with_pred, CORRECT_IDX)
    return A_raw, A_col


# ---------------------------------------------------------------------------
# Get raw trace-level probabilities from the baseline model
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_trace_probs(
    model: "NoGraphBaseline",
    spans: list[dict],
    x: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        p_base  : (T, 19) raw sigmoid probabilities (before threshold)
        y_true  : (T, 19) ground-truth multi-label binary matrix
    """
    from collections import defaultdict
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))

    by_trace = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    p_list, y_list = [], []

    for tid in sorted(by_trace.keys()):
        trace_spans = by_trace[tid]
        embs   = torch.stack([sp["emb"] for sp in trace_spans]).to(device)
        scores = model.score_spans(embs, Z)
        probs  = torch.sigmoid(scores).cpu().numpy()       # (K, 20)

        trace_max = probs[:, :CORRECT_IDX].max(axis=0)    # (19,)
        p_list.append(trace_max)

        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in trace_spans:
            for idx, v in enumerate(sp["label_vec"][:CORRECT_IDX].tolist()):
                if v > 0:
                    gt[idx] = 1
        y_list.append(gt)

    return np.array(p_list, dtype=np.float32), np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Causal predecessor gating (core intervention)
# ---------------------------------------------------------------------------

def causal_gate_intervention(
    p_base: np.ndarray,   # (T, 19)
    A_col: np.ndarray,    # (19, 19) column-normalised Suppes
    A_raw: np.ndarray,    # (19, 19) raw Suppes (used to detect no-predecessor nodes)
    beta: float,          # gate floor: 0 = full suppression, 1 = no intervention
) -> np.ndarray:
    """
    Apply causal predecessor gating:
        gate(t, i)     = Σ_j  A_col[j,i] · p_base(t, j)   (predecessor support)
        p_causal(t, i) = p_base(t, i) · (β + (1-β) · gate(t, i))

    Nodes with no predecessors (col sum = 0 in A_raw) are left unchanged (gate=1).

    Returns p_causal: (T, 19)
    """
    # gate(t, i) = p_base(t,:) @ A_col[:, i]  for each trace
    gate = p_base @ A_col          # (T, 19): predecessor support per trace per label

    # Identify labels with no predecessors — skip intervention for those
    has_predecessors = (A_raw.sum(axis=0) > 0)          # (19,) bool
    gate[:, ~has_predecessors] = 1.0                    # no intervention

    # Apply multiplicative gate
    effective_gate = beta + (1.0 - beta) * gate         # (T, 19) in [β, 1]
    p_causal = p_base * effective_gate
    return np.clip(p_causal, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Span-level location and joint accuracy (error step detection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_span_location_metrics(
    model: "NoGraphBaseline",
    spans: list[dict],
    x: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
    threshold: float,
    A_col: np.ndarray,
    A_raw: np.ndarray,
    beta: float,
    label_map: dict,
) -> tuple[float, float]:
    """
    Compute Loc. Acc. and Joint Acc. after applying the causal gate at span level.

    The causal gate is trace-level (gate(t, i) derived from trace max-pool probs),
    but applied multiplicatively to each span's raw probabilities:
        p_causal_span(k, i) = sigmoid(score(k, i)) * (β + (1-β) * gate(t, i))

    Returns:
        loc_acc   — average fraction of GT error spans correctly identified
        joint_acc — average fraction of GT (span_id, category) pairs correctly predicted
    """
    from collections import defaultdict
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))

    by_trace = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    loc_acc_list, joint_acc_list = [], []

    for tid in sorted(by_trace.keys()):
        trace_spans = by_trace[tid]
        embs   = torch.stack([sp["emb"] for sp in trace_spans]).to(device)
        scores = model.score_spans(embs, Z)
        span_probs = torch.sigmoid(scores).cpu().numpy()[:, :CORRECT_IDX]  # (K, 19)

        # Trace-level max-pool used for the causal gate
        trace_max = span_probs.max(axis=0, keepdims=True)  # (1, 19)
        gate = trace_max @ A_col                           # (1, 19) predecessor support
        has_pred = A_raw.sum(axis=0) > 0
        gate[:, ~has_pred] = 1.0
        eff_gate = beta + (1.0 - beta) * gate              # (1, 19) in [β, 1]
        span_probs_causal = span_probs * eff_gate          # (K, 19) broadcast

        # Location accuracy: |GT_error_spans ∩ pred_error_spans| / |GT_error_spans|
        pred_error = span_probs_causal.max(axis=1) > threshold  # (K,)
        gt_span_ids   = {sp["span_id"] for sp in trace_spans if not sp["is_correct"]}
        pred_span_ids = {sp["span_id"] for sp, is_err in zip(trace_spans, pred_error) if is_err}
        loc_acc = len(gt_span_ids & pred_span_ids) / len(gt_span_ids) if gt_span_ids else (1.0 if not pred_span_ids else 0.0)
        loc_acc_list.append(loc_acc)

        # Joint accuracy: |GT_(span_id,cat) pairs ∩ pred_pairs| / |GT_pairs|
        gt_pairs, pred_pairs = set(), set()
        for k, sp in enumerate(trace_spans):
            sid = sp["span_id"]
            if not sp["is_correct"]:
                for cat in sp["labels"]:
                    gt_pairs.add((sid, cat))
            for cat, cat_idx in label_map.items():
                if cat_idx < CORRECT_IDX and span_probs_causal[k, cat_idx] > threshold:
                    pred_pairs.add((sid, cat))

        joint_acc = len(gt_pairs & pred_pairs) / len(gt_pairs) if gt_pairs else (1.0 if not pred_span_ids else 0.0)
        joint_acc_list.append(joint_acc)

    return float(np.mean(loc_acc_list)), float(np.mean(joint_acc_list))


# ---------------------------------------------------------------------------
# Val grid-search over β (threshold fixed at 0.5)
# ---------------------------------------------------------------------------

def sweep_beta(
    p_base: np.ndarray,
    y_true: np.ndarray,
    A_col: np.ndarray,
    A_raw: np.ndarray,
    betas: list[float],
) -> tuple[float, float]:
    """Sweep β only; threshold is fixed at 0.5."""
    best_f1, best_beta = 0.0, 1.0
    for beta in betas:
        p_causal = causal_gate_intervention(p_base, A_col, A_raw, beta)
        y_pred = (p_causal > 0.5).astype(int)
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if f1 > best_f1:
            best_f1, best_beta = f1, beta
    return best_beta, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inference-time causal predecessor gating (CMLL-inspired)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--beta",      type=float, default=None,
                    help="Gate floor β (0=full suppression, 1=no intervention). "
                         "Default: sweep on val.")
    ap.add_argument("--gpu",       type=int,   default=0)
    ap.add_argument("--split_tag", default="",
                    help="Tag suffix for data/model/output dirs (e.g. '712' → data_712/, "
                         "models_712/, outputs_712/). Empty = default dirs.")
    args = ap.parse_args()

    # Apply split_tag to module-level path variables
    if args.split_tag:
        tag = f"_{args.split_tag}"
        global DATA_DIR, MODEL_DIR, OUTPUT_DIR
        global DATASET_FILE, GRAPH_INPUT, LABEL_MAP_FILE
        DATA_DIR       = GRAPH_DIR / f"data{tag}"
        MODEL_DIR      = BASELINE_DIR / f"models{tag}"
        OUTPUT_DIR     = POST_DIR / f"outputs{tag}"
        DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
        GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
        LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph input + causal matrix
    # ------------------------------------------------------------------
    gi     = torch.load(GRAPH_INPUT, weights_only=False)
    x      = gi["x"].float().to(device)
    n_nodes = gi["n_nodes"]
    node_names = gi["node_names"]
    adj    = build_adj_matrix(gi["edge_index"], gi["edge_weight"], n_nodes).to(device)

    A_raw, A_col = build_causal_matrix(gi)

    # ------------------------------------------------------------------
    # Load baseline model
    # ------------------------------------------------------------------
    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} — run graph/baseline/run_baseline.py first")

    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved    = ckpt["args"]
    feat_dim = ckpt.get("feat_dim", 4096)
    base_thr = 0.5
    log.info("Loaded baseline model: epoch=%d  val_F1=%.4f  fixed_threshold=0.50",
             ckpt["epoch"], ckpt["val_f1"])

    model = NoGraphBaseline(
        feat_dim   = feat_dim,
        hidden_dim = saved["hidden_dim"],
        n_nodes    = n_nodes,
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ------------------------------------------------------------------
    # Load label map
    # ------------------------------------------------------------------
    label_map   = json.loads(LABEL_MAP_FILE.read_text())
    error_names = node_names[:CORRECT_IDX]

    # ------------------------------------------------------------------
    # Compute raw trace-level probabilities for val + test
    # ------------------------------------------------------------------
    splits = {}
    for split in ("val", "test"):
        emb_dict = torch.load(DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True)
        spans    = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)
        p_base, y_true = get_trace_probs(model, spans, x, adj, device)
        splits[split]  = {"p_base": p_base, "y_true": y_true, "spans": spans}
        log.info("%s: %d traces, %d positive labels",
                 split, len(p_base), y_true.sum())

    # ------------------------------------------------------------------
    # Tune β on val (threshold fixed at 0.5)
    # ------------------------------------------------------------------
    best_thr = 0.5
    betas    = [round(b, 2) for b in np.arange(0.0, 1.05, 0.05)]

    if args.beta is not None:
        best_beta = args.beta
        p_val = causal_gate_intervention(
            splits["val"]["p_base"], A_col, A_raw, best_beta
        )
        best_val_f1 = float(f1_score(
            splits["val"]["y_true"],
            (p_val > best_thr).astype(int),
            average="weighted", zero_division=0,
        ))
        log.info("Using fixed β=%.2f, threshold=0.50 → val weighted F1=%.4f",
                 best_beta, best_val_f1)
    else:
        log.info("Sweeping β ∈ [0..1] on val (%d values, threshold fixed at 0.50)...",
                 len(betas))
        best_beta, best_val_f1 = sweep_beta(
            splits["val"]["p_base"], splits["val"]["y_true"],
            A_col, A_raw, betas,
        )
        log.info("Best val: β=%.2f, threshold=0.50 → weighted F1=%.4f",
                 best_beta, best_val_f1)

    # ------------------------------------------------------------------
    # Evaluate both splits
    # ------------------------------------------------------------------
    all_results = {}

    for split in ("val", "test"):
        p_base  = splits[split]["p_base"]
        y_true  = splits[split]["y_true"]
        spans   = splits[split]["spans"]

        p_causal = causal_gate_intervention(p_base, A_col, A_raw, best_beta)
        y_pred   = (p_causal > best_thr).astype(int)

        # Baseline (no intervention) for comparison
        y_pred_base = (p_base > base_thr).astype(int)
        base_metrics = compute_metrics(y_true, y_pred_base, error_names)

        metrics  = compute_metrics(y_true, y_pred, error_names)
        delta_f1 = metrics["f1_weighted"] - base_metrics["f1_weighted"]

        # Span-level location and joint accuracy (error step detection)
        loc_acc, joint_acc = compute_span_location_metrics(
            model, spans, x, adj, device, best_thr,
            A_col, A_raw, best_beta, label_map,
        )

        print(f"\n{'='*60}")
        print(f"Causal gate intervention (β={best_beta:.2f}, thr={best_thr:.2f}) — {split} set")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=error_names, zero_division=0))
        print_metrics(metrics, header="--- Causal gate ---")
        print_metrics(base_metrics, header=f"\n--- Baseline (no intervention, thr={base_thr:.2f}) ---")
        print(f"\nΔ F1 weighted vs baseline: {delta_f1:+.4f}")
        print(f"Location Accuracy (Loc. Acc.):  {loc_acc:.4f}")
        print(f"Joint Accuracy    (Joint Acc.): {joint_acc:.4f}")

        all_results[split] = {
            "model":                "Baseline+CausalGate",
            "split":                split,
            "beta":                 best_beta,
            "threshold":            best_thr,
            "n_traces":             int(len(p_base)),
            **{k: v for k, v in metrics.items() if k != "per_class"},
            "location_accuracy":    loc_acc,
            "joint_accuracy":       joint_acc,
            "per_class":            metrics["per_class"],
            "baseline_f1_weighted": base_metrics["f1_weighted"],
            "delta_f1_weighted":    delta_f1,
        }

        out_path = OUTPUT_DIR / f"eval_results_causal_gate_{split}.json"
        out_path.write_text(json.dumps(all_results[split], indent=2))
        log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
