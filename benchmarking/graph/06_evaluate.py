#!/usr/bin/env python3
"""
06_evaluate.py — Evaluate best GAT model on test set.

Inference:
    1. Encode span text → h_k (frozen Qwen, already in span_embeddings_test.pt)
    2. Run GAT forward → refined node embeddings Z
    3. Score each span against all 19 error nodes: p_hat[k, i] = sigmoid(h_k^T M z_i)
    4. Threshold at 0.5: pred_span[k] = {i : p_hat[k,i] > 0.5, i < 19}

Trace-level aggregation (mean-pool):
    p_hat_trace[t, i] = mean(p_hat[k, i] for k in spans of trace t)
    pred_trace[t]     = {i : p_hat_trace[t, i] > 0.5}

Metric:
    Category F1 = sklearn f1_score(y_true, y_pred, average='weighted')
    Over 19 error type labels (indices 0-18). Correct node (19) excluded.

Outputs:
    graph/outputs/eval_results.json    — per-class F1 + overall Cat. F1
    graph/outputs/confusion_matrix.png — (optional, requires matplotlib)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
)

BENCH_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR       = BENCH_DIR / "graph" / "data"
OUTPUT_DIR     = BENCH_DIR / "graph" / "outputs"
MODEL_DIR      = BENCH_DIR / "graph" / "models"
DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CORRECT_IDX = 19

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import model classes directly from 05_train_gat.py
# ---------------------------------------------------------------------------
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gat_module", Path(__file__).parent / "05_train_gat.py"
)
_gat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gat_mod)
GATModel             = _gat_mod.GATModel
build_adj_matrix     = _gat_mod.build_adj_matrix
build_flat_span_list = _gat_mod.build_flat_span_list


@torch.no_grad()
def run_inference(
    model: "GATModel",
    spans: list[dict],
    x: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
        y_true: (T, 19) multi-label ground truth per trace
        y_pred: (T, 19) predicted labels per trace
        trace_results: dict of per-trace prediction details
    """
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))  # (20, hidden_dim)

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    y_true_list, y_pred_list = [], []
    trace_results = {}

    for tid, trace_spans in sorted(by_trace.items()):
        embs = torch.stack([sp["emb"] for sp in trace_spans]).to(device)  # (K, feat_dim)
        scores = model.score_spans(embs, Z)                                # (K, 20)
        probs  = torch.sigmoid(scores).cpu()                               # (K, 20)

        # Trace-level max-pool: a trace has error type i if any span exceeds threshold.
        # More robust than mean-pool when most spans are "Correct" (77.7%).
        trace_max = probs[:, :CORRECT_IDX].max(dim=0).values.numpy()      # (19,)
        pred = (trace_max > threshold).astype(int)                         # (19,)

        # Ground truth: union of error labels across trace spans
        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in trace_spans:
            for i, v in enumerate(sp["label_vec"][:CORRECT_IDX].tolist()):
                if v > 0:
                    gt[i] = 1

        y_true_list.append(gt)
        y_pred_list.append(pred)
        trace_results[tid] = {
            "pred_indices": pred.nonzero()[0].tolist(),
            "true_indices": gt.nonzero()[0].tolist(),
            "trace_max":    trace_max.tolist(),
        }

    return np.array(y_true_list), np.array(y_pred_list), trace_results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate best GAT on test set")
    ap.add_argument("--model_path", default=str(MODEL_DIR / "best_model.pt"))
    ap.add_argument("--split",      default="test", choices=["val", "test"])
    ap.add_argument("--gpu",        type=int, default=0)
    ap.add_argument("--threshold",  type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    gi    = torch.load(GRAPH_INPUT, weights_only=False)
    x     = gi["x"].float()
    adj   = build_adj_matrix(gi["edge_index"], gi["edge_weight"], gi["n_nodes"]).to(device)
    x     = x.to(device)
    node_names = gi["node_names"]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    feat_dim = ckpt.get("feat_dim", 4096)
    # Use the val-tuned threshold unless overridden on the command line
    threshold = ckpt.get("val_threshold", args.threshold)
    log.info("Using threshold=%.2f (from val tuning)", threshold)

    model = GATModel(
        feat_dim=feat_dim,
        hidden_dim=saved_args["hidden_dim"],
        n_heads=saved_args["n_heads"],
        n_nodes=gi["n_nodes"],
        dropout=0.0,   # no dropout at eval
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded model from epoch %d (val F1=%.4f)", ckpt["epoch"], ckpt["val_f1"])

    # ------------------------------------------------------------------
    # Load test spans
    # ------------------------------------------------------------------
    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())
    emb_dict = torch.load(DATA_DIR / f"span_embeddings_{args.split}.pt", weights_only=True)
    test_spans = build_flat_span_list(
        DATASET_FILE, args.split, emb_dict, label_map, gi["n_nodes"]
    )
    log.info("Loaded %d %s spans", len(test_spans), args.split)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    y_true, y_pred, trace_results = run_inference(
        model, test_spans, x, adj, device, threshold
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    error_names = node_names[:CORRECT_IDX]   # 19 error type names

    cat_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cat_f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    cat_f1_micro    = float(f1_score(y_true, y_pred, average="micro",    zero_division=0))

    report_str = classification_report(
        y_true, y_pred,
        target_names=error_names,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"Evaluation on {args.split} set")
    print(f"{'='*60}")
    print(report_str)
    print(f"Category F1 (weighted): {cat_f1_weighted:.4f}")
    print(f"Category F1 (macro):    {cat_f1_macro:.4f}")
    print(f"Category F1 (micro):    {cat_f1_micro:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Per-class F1 from classification_report
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    per_class = {}
    for i, name in enumerate(error_names):
        per_class[name] = {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(sup[i]),
        }

    results = {
        "split":              args.split,
        "n_traces":           len(trace_results),
        "threshold":          threshold,   # actual val-tuned threshold, not CLI default
        "cat_f1_weighted":    cat_f1_weighted,
        "cat_f1_macro":       cat_f1_macro,
        "cat_f1_micro":       cat_f1_micro,
        "per_class":          per_class,
        "trace_predictions":  trace_results,
    }

    out_path = OUTPUT_DIR / f"eval_results_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out_path)

    # ------------------------------------------------------------------
    # Confusion matrix (optional)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cms = multilabel_confusion_matrix(y_true, y_pred)  # (19, 2, 2)
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        for i, (name, cm) in enumerate(zip(error_names, cms)):
            ax = axes[i // 5, i % 5]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name[:25], fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
            ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(cm[row, col]),
                            ha="center", va="center", fontsize=9,
                            color="white" if cm[row, col] > cm.max() / 2 else "black")
        # Hide last axis if 19 < 4*5
        axes[3, 4].axis("off")
        plt.tight_layout()
        cm_path = OUTPUT_DIR / f"confusion_matrix_{args.split}.png"
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Saved confusion matrix → %s", cm_path)
    except Exception as e:
        log.warning("Could not save confusion matrix: %s", e)


if __name__ == "__main__":
    main()
