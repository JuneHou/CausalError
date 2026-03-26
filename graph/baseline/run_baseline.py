#!/usr/bin/env python3
"""
baseline/run_baseline.py — No-graph ablation baseline for the GAT error-type pipeline.

Ablation design (controls for the golden-graph contribution):
  ✓  Same error-type embeddings      (same prototypes.pt, same span_embeddings_*.pt)
  ✓  Same span-to-error-type scorer  (bilinear + cosine / τ)
  ✓  Same threshold tuning            sweep [0.10..0.50] on val
  ✓  Same trace-level max-pool
  ✓  Same weighted / macro / micro F1
  ✗  No edges
  ✗  No GAT message passing           Z = ELU(Proj(x))  — one linear layer

  --lambda_graph 0   (default):  L = L_span only
  --lambda_graph >0:             L = L_span + λ·L_graph
                                 Tests graph-structure regularization without message passing.

Model:
    x  (20, feat_dim)  ──Proj──► Z (20, D)        D = hidden_dim = 256
    span h_k  ──Proj──► h_k' (D)
    score(h_k, z_i) = h_k'^T · M · z_i  +  cos(h_k', z_i) / τ

Inputs (shared with main pipeline — no re-encoding needed):
    graph/data/graph_input.pt           prototype node features x  (edges ignored)
    graph/data/span_embeddings_*.pt     pre-encoded span embeddings
    graph/data/span_dataset.jsonl       span texts + labels + splits
    graph/data/label_to_node_idx.json   label → node index map

Outputs:
    graph/baseline/models/best_model.pt
    graph/baseline/outputs/eval_results_val.json
    graph/baseline/outputs/eval_results_test.json

Usage:
    # From trail-benchmark/:
    python graph/baseline/run_baseline.py              # train + evaluate
    python graph/baseline/run_baseline.py --eval_only  # evaluate saved model
"""

import argparse
import importlib.util
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPH_DIR   = Path(__file__).resolve().parent.parent   # trail-benchmark/graph/
BENCH_DIR   = GRAPH_DIR.parent / "benchmarking"        # trail-benchmark/benchmarking/

DATA_DIR    = GRAPH_DIR / "data"
BASELINE_DIR = Path(__file__).resolve().parent          # trail-benchmark/graph/baseline/
MODEL_DIR   = BASELINE_DIR / "models"
OUTPUT_DIR  = BASELINE_DIR / "outputs"

DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CORRECT_IDX = 19

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import shared helpers from the main GAT pipeline (avoid duplication)
# ---------------------------------------------------------------------------
# build_flat_span_list — loads span embeddings and constructs per-span label vectors
# evaluate_cat_f1      — val threshold sweep; works with any model that exposes
#                        encode_graph(x, adj) and score_spans(embs, Z)
# build_adj_matrix     — used to load adj (passed to evaluate_cat_f1 but ignored here)

def _load_gat_module():
    spec = importlib.util.spec_from_file_location(
        "gat_module", GRAPH_DIR / "05_train_gat.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_module", GRAPH_DIR / "06_evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_gat  = _load_gat_module()
_eval = _load_eval_module()

build_flat_span_list = _gat.build_flat_span_list
build_adj_matrix     = _gat.build_adj_matrix
evaluate_cat_f1      = _gat.evaluate_cat_f1
run_inference        = _eval.run_inference
compute_metrics      = _eval.compute_metrics
print_metrics        = _eval.print_metrics


# ---------------------------------------------------------------------------
# No-graph baseline model
# ---------------------------------------------------------------------------

class NoGraphBaseline(nn.Module):
    """
    Ablation: replaces GAT(x, adj) with a single linear projection.
    Exposes the same encode_graph / score_spans API as GATModel so that the
    shared evaluation helpers (evaluate_cat_f1, run_inference) work unchanged.
    """

    def __init__(
        self,
        feat_dim: int = 4096,
        hidden_dim: int = 256,
        n_nodes: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj           = nn.Linear(feat_dim, hidden_dim)
        self.bilinear       = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.graph_bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.log_temp       = nn.Parameter(torch.tensor(np.log(0.07)))
        self.node_dropout   = nn.Dropout(dropout)
        self.span_dropout   = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.bilinear.unsqueeze(0))
        nn.init.xavier_uniform_(self.graph_bilinear.unsqueeze(0))

    # ------------------------------------------------------------------
    # Graph encoding — no message passing, adj is accepted but ignored
    # ------------------------------------------------------------------

    def encode_graph(
        self,
        x: torch.Tensor,           # (N, feat_dim) prototype node features
        adj: torch.Tensor = None,  # ignored — present only for API compatibility
    ) -> torch.Tensor:
        """Z = ELU(Proj(x))  — one linear layer, no propagation."""
        return F.elu(self.node_dropout(self.proj(x)))   # (N, hidden_dim)

    # ------------------------------------------------------------------
    # Span scorer — identical to GATModel.score_spans
    # ------------------------------------------------------------------

    def score_spans(
        self,
        span_embs: torch.Tensor,   # (B, feat_dim)
        Z: torch.Tensor,           # (N, hidden_dim)
    ) -> torch.Tensor:
        h = self.span_dropout(F.elu(self.proj(span_embs)))   # (B, hidden_dim)
        bilinear_scores = h @ self.bilinear @ Z.T             # (B, N)
        h_norm = F.normalize(h, p=2, dim=1)
        z_norm = F.normalize(Z, p=2, dim=1)
        tau    = self.log_temp.exp().clamp(min=0.01, max=1.0)
        cos_scores = (h_norm @ z_norm.T) / tau                # (B, N)
        return bilinear_scores + cos_scores                    # (B, N)

    # ------------------------------------------------------------------
    # forward — matches GATModel signature (adj accepted, not used)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        span_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Z           = self.encode_graph(x, adj)
        span_scores = self.score_spans(span_embs, Z)
        return span_scores, Z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load prototype features (x) — edges are loaded but ignored in training
    # ------------------------------------------------------------------
    if not GRAPH_INPUT.exists():
        raise FileNotFoundError(f"{GRAPH_INPUT} — run 04_build_graph_input.py first")

    gi         = torch.load(GRAPH_INPUT, weights_only=False)
    x          = gi["x"].float().to(device)          # (20, feat_dim)
    n_nodes    = gi["n_nodes"]
    node_names = gi["node_names"]
    feat_dim   = x.shape[1]

    # Build adj for API compatibility with shared eval helpers (ignored by baseline)
    adj = build_adj_matrix(gi["edge_index"], gi["edge_weight"], n_nodes).to(device)

    # Gold adjacency for L_graph (19×19 error nodes only, binarized)
    A_gold = (adj[:CORRECT_IDX, :CORRECT_IDX] > 0).float()

    log.info("Prototypes loaded: x=%s, n_nodes=%d  (message passing disabled)",
             tuple(x.shape), n_nodes)
    if args.lambda_graph > 0:
        log.info("L_graph enabled: lambda_graph=%.3f  (graph-structure regularizer, no propagation)",
                 args.lambda_graph)

    # ------------------------------------------------------------------
    # Load label map and span data
    # ------------------------------------------------------------------
    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())

    log.info("Loading span embeddings...")
    train_embs = torch.load(DATA_DIR / "span_embeddings_train.pt", weights_only=True)
    val_embs   = torch.load(DATA_DIR / "span_embeddings_val.pt",   weights_only=True)

    train_spans = build_flat_span_list(DATASET_FILE, "train", train_embs, label_map, n_nodes)
    val_spans   = build_flat_span_list(DATASET_FILE, "val",   val_embs,   label_map, n_nodes)
    log.info("Train spans: %d  Val spans: %d", len(train_spans), len(val_spans))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = NoGraphBaseline(
        feat_dim   = feat_dim,
        hidden_dim = args.hidden_dim,
        n_nodes    = n_nodes,
        dropout    = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("NoGraphBaseline params: %d", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Pos-weight to counteract class imbalance (~77% Correct spans)
    n_annot = sum(1 for sp in train_spans if not sp["is_correct"])
    n_corr  = len(train_spans) - n_annot
    pos_weight_val = n_corr / max(n_annot, 1)
    pos_weights = torch.ones(n_nodes, device=device)
    pos_weights[:CORRECT_IDX] = pos_weight_val
    bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    bce          = nn.BCEWithLogitsLoss()   # unweighted, for L_graph

    best_f1, best_thr, best_epoch = 0.0, 0.5, 0
    patience_counter = 0

    loss_desc = f"L = L_span + {args.lambda_graph}·L_graph" if args.lambda_graph > 0 else "L = L_span"
    log.info("Training for up to %d epochs (patience=%d, %s)...",
             args.epochs, args.patience, loss_desc)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_spans)

        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(train_spans), args.batch_size):
            batch  = train_spans[start: start + args.batch_size]
            embs   = torch.stack([sp["emb"]       for sp in batch]).to(device)
            labels = torch.stack([sp["label_vec"] for sp in batch]).to(device)

            optimizer.zero_grad()

            Z           = model.encode_graph(x)             # (N, D)  — no adj
            span_scores = model.score_spans(embs, Z)        # (B, N)

            loss = bce_weighted(span_scores, labels)        # L_span

            if args.lambda_graph > 0:
                A_hat = torch.sigmoid(
                    Z[:CORRECT_IDX] @ model.graph_bilinear @ Z[:CORRECT_IDX].T
                )
                loss = loss + args.lambda_graph * bce(A_hat, A_gold)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Val evaluation — reuse shared evaluate_cat_f1 (adj passed but ignored)
        val_f1, val_thr = evaluate_cat_f1(model, val_spans, x, adj, device)

        if val_f1 > best_f1:
            best_f1, best_thr, best_epoch = val_f1, val_thr, epoch
            patience_counter = 0
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "val_f1":        val_f1,
                "val_threshold": val_thr,
                "args":          vars(args),
                "node_names":    node_names,
                "feat_dim":      feat_dim,
                "model_type":    "NoGraphBaseline",
            }, MODEL_DIR / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch <= 3:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_F1=%.4f(thr=%.2f)  best=%.4f@%d",
                epoch, args.epochs, avg_loss, val_f1, val_thr, best_f1, best_epoch,
            )

        if patience_counter >= args.patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training done. Best val F1=%.4f (thr=%.2f) at epoch %d",
             best_f1, best_thr, best_epoch)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model checkpoint
    # ------------------------------------------------------------------
    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} — run training first")

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved     = ckpt["args"]
    feat_dim  = ckpt.get("feat_dim", 4096)
    threshold = ckpt.get("val_threshold", 0.5)
    log.info("Loaded checkpoint from epoch %d (val F1=%.4f, threshold=%.2f)",
             ckpt["epoch"], ckpt["val_f1"], threshold)

    model = NoGraphBaseline(
        feat_dim   = feat_dim,
        hidden_dim = saved["hidden_dim"],
        n_nodes    = 20,
        dropout    = 0.0,   # no dropout at eval
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ------------------------------------------------------------------
    # Load graph input (x + adj for API compat; edges unused)
    # ------------------------------------------------------------------
    gi         = torch.load(GRAPH_INPUT, weights_only=False)
    x          = gi["x"].float().to(device)
    n_nodes    = gi["n_nodes"]
    node_names = gi["node_names"]
    adj        = build_adj_matrix(gi["edge_index"], gi["edge_weight"], n_nodes).to(device)

    # ------------------------------------------------------------------
    # Load label map
    # ------------------------------------------------------------------
    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())
    error_names = node_names[:CORRECT_IDX]   # 19 error-type names

    # ------------------------------------------------------------------
    # Evaluate on both val and test splits
    # ------------------------------------------------------------------
    for split in ("val", "test"):
        emb_dict = torch.load(DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True)
        spans    = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)
        log.info("Evaluating on %s (%d spans, threshold=%.2f)...", split, len(spans), threshold)

        # run_inference from 06_evaluate.py — uses encode_graph + score_spans API
        y_true, y_pred, trace_results = run_inference(model, spans, x, adj, device, threshold)

        metrics = compute_metrics(y_true, y_pred, error_names)

        print(f"\n{'='*60}")
        print(f"Baseline evaluation — {split} set")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=error_names, zero_division=0))
        print_metrics(metrics)

        results = {
            "model":             "NoGraphBaseline",
            "split":             split,
            "n_traces":          len(trace_results),
            "threshold":         threshold,
            **{k: v for k, v in metrics.items() if k != "per_class"},
            "per_class":         metrics["per_class"],
            "trace_predictions": trace_results,
        }

        out_path = OUTPUT_DIR / f"eval_results_{split}.json"
        out_path.write_text(json.dumps(results, indent=2))
        log.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="No-graph ablation baseline: Z=Proj(x), L=L_span, same scorer+eval as GAT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training args (match 05_train_gat.py defaults so comparison is fair)
    ap.add_argument("--hidden_dim",   type=int,   default=256)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--patience",     type=int,   default=10)
    ap.add_argument("--lambda_graph", type=float, default=0.0,
                    help="Weight for graph-structure regularizer L_graph (0 = disabled)")
    ap.add_argument("--gpu",          type=int,   default=0)
    ap.add_argument("--seed",         type=int,   default=42)
    # Evaluation-only mode
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; load best_model.pt and evaluate")
    args = ap.parse_args()

    if not args.eval_only:
        train(args)

    evaluate(args)


if __name__ == "__main__":
    main()
