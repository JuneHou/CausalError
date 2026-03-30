#!/usr/bin/env python3
"""
05_train_gat.py — Train GAT + bilinear link prediction for error classification.

Architecture:
    Input prototype features: (20, 4096)
    Linear projection:        (20, D)        D = 256 (hidden_dim)
    GATLayer 1 (K=4 heads):   (20, D)
    GATLayer 2 (K=4 heads):   (20, D)
    → Refined node embeddings Z (20, D)

Per-span link prediction:
    Span embedding h_k (4096-dim, projected to D) scored against each node:
    score(h_k, z_i) = h_k^T · M · z_i   (M: DxD bilinear matrix)
    p_hat[k, i]     = sigmoid(score)     for i in 0..19

Loss:
    L = L_span + λ_graph · L_graph
    L_span  = BCE(p_hat[k, 0:20], y[k, 0:20])   (span-level multi-label)
    L_graph = BCE(A_hat, A_gold)                  (graph structure reconstruction, 19×19)

Training:
    Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
    Batch size: 32 spans
    Early stopping on val Category F1 (patience=10)
    Best model saved to graph/models/best_model.pt
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

BENCH_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR      = BENCH_DIR / "graph" / "data"
OUTPUT_DIR    = BENCH_DIR / "graph" / "outputs"
MODEL_DIR     = BENCH_DIR / "graph" / "models"
DATASET_FILE  = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT   = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CORRECT_IDX = 19   # index of the "Correct" node


# ---------------------------------------------------------------------------
# Dense GAT layer (no torch_geometric dependency)
# ---------------------------------------------------------------------------

class DenseGATLayer(nn.Module):
    """
    Graph Attention layer for a small fully-dense graph (N nodes).
    Uses the standard GAT attention mechanism with edge-weight bias.

    For N=20 nodes a dense N×N approach is efficient.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim))
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(
        self,
        h: torch.Tensor,          # (N, in_dim)
        adj_bias: torch.Tensor,   # (N, N) edge weights as attention bias (0 = absent)
    ) -> torch.Tensor:
        N = h.shape[0]
        Wh = self.W(h).view(N, self.n_heads, self.head_dim)   # (N, K, head_dim)

        # Attention coefficients: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.n_heads, self.head_dim)  # (N, N, K, d)
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.n_heads, self.head_dim)  # (N, N, K, d)
        cat  = torch.cat([Wh_i, Wh_j], dim=-1)                            # (N, N, K, 2d)
        e    = self.leaky((cat * self.a).sum(dim=-1))                      # (N, N, K)

        # Add edge-weight bias (broadcast over heads); mask absent edges with -inf
        mask = (adj_bias == 0).unsqueeze(-1)                   # (N, N, 1) bool
        bias = adj_bias.unsqueeze(-1)                          # (N, N, 1)
        e = e + bias                                           # (N, N, K)
        e = e.masked_fill(mask, float("-inf"))

        # Prevent NaN for isolated nodes (all -inf row): force self-loop attention
        # by replacing fully-masked rows with a self-attention mask
        all_masked = mask.all(dim=1, keepdim=True)             # (N, 1, 1) bool
        # Create self-loop mask: only keep diagonal
        self_loop = torch.eye(N, device=e.device, dtype=torch.bool).unsqueeze(-1)
        # For isolated nodes, unmask the self-loop
        e = e.masked_fill(all_masked & self_loop, 0.0)

        alpha = F.softmax(e, dim=1)                            # (N, N, K)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = (alpha.unsqueeze(-1) * Wh_j).sum(dim=1)         # (N, K, head_dim)
        if self.concat:
            out = out.reshape(N, self.n_heads * self.head_dim)
        else:
            out = out.mean(dim=1)                              # (N, head_dim)
        return F.elu(out)


# ---------------------------------------------------------------------------
# Full GAT model
# ---------------------------------------------------------------------------

class GATModel(nn.Module):
    """
    GAT model for error-type node embedding + bilinear span-to-node scoring.
    """

    def __init__(
        self,
        feat_dim: int = 4096,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_nodes: int = 20,
        dropout: float = 0.3,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        assert n_layers in (1, 2), "n_layers must be 1 or 2"
        self.n_layers = n_layers
        self.proj = nn.Linear(feat_dim, hidden_dim)                # span + node projection

        # With 1 layer: concat=False → output stays (N, hidden_dim)
        # With 2 layers: concat=True on gat1 → (N, n_heads*hidden_dim), then gat2 → (N, hidden_dim)
        self.gat1 = DenseGATLayer(hidden_dim, hidden_dim, n_heads, dropout,
                                   concat=(n_layers == 2))
        if n_layers == 2:
            self.gat2 = DenseGATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout, concat=False)

        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.graph_bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear.unsqueeze(0))
        nn.init.xavier_uniform_(self.graph_bilinear.unsqueeze(0))

        # Learnable temperature for cosine-based scoring (log scale for stability)
        # score = cos_sim(h_k, z_i) / τ;  τ = exp(log_temp), init τ=0.07 (CLIP-style)
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.07)))

        self.node_dropout = nn.Dropout(dropout)
        self.span_dropout = nn.Dropout(dropout)

    def encode_graph(
        self,
        x: torch.Tensor,          # (N, feat_dim) prototype node features
        adj: torch.Tensor,         # (N, N) adjacency with edge weights
    ) -> torch.Tensor:
        """Return refined node embeddings Z: (N, hidden_dim)."""
        h = F.elu(self.proj(x))                # (N, hidden_dim)
        h = self.node_dropout(h)
        h = self.gat1(h, adj)                  # (N, hidden_dim) if 1-layer; (N, hidden_dim*K) if 2-layer
        if self.n_layers == 2:
            h = self.gat2(h, adj)              # (N, hidden_dim)
        return h

    def score_spans(
        self,
        span_embs: torch.Tensor,   # (B, feat_dim)
        Z: torch.Tensor,           # (N, hidden_dim)
    ) -> torch.Tensor:
        """
        Score spans against nodes using both bilinear and cosine-similarity components.
        Final score = bilinear(h, z) + cos_sim(h_norm, z_norm) / τ
        The cosine term with learned temperature provides calibrated probabilities.
        """
        h = self.span_dropout(F.elu(self.proj(span_embs)))   # (B, hidden_dim)
        # Bilinear term
        bilinear_scores = h @ self.bilinear @ Z.T             # (B, N)
        # Cosine similarity term with temperature
        h_norm = F.normalize(h, p=2, dim=1)                   # (B, D)
        z_norm = F.normalize(Z, p=2, dim=1)                   # (N, D)
        tau = self.log_temp.exp().clamp(min=0.01, max=1.0)
        cos_scores = (h_norm @ z_norm.T) / tau                # (B, N)
        return bilinear_scores + cos_scores

    def forward(
        self,
        x: torch.Tensor,          # (N, feat_dim) node prototypes
        adj: torch.Tensor,         # (N, N) adjacency
        span_embs: torch.Tensor,   # (B, feat_dim) span embeddings
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Z = self.encode_graph(x, adj)             # (N, hidden_dim)
        span_scores = self.score_spans(span_embs, Z)   # (B, N)
        return span_scores, Z


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_span_embeddings(split: str) -> dict[str, dict[str, torch.Tensor]]:
    """Load {trace_id: {span_id: tensor(D,)}} from split .pt file."""
    p = DATA_DIR / f"span_embeddings_{split}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run 03_encode_spans.py first")
    return torch.load(p, weights_only=True)


def build_flat_span_list(
    dataset_file: Path,
    split: str,
    emb_dict: dict,
    label_map: dict[str, int],
    n_nodes: int,
) -> list[dict]:
    """
    Return flat list of {emb, label_vec, trace_id, span_id} for a split.
    label_vec: binary tensor(n_nodes,) — 1 for each assigned label index.
    """
    result = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            tid = rec["trace_id"]
            for sp in rec["spans"]:
                sid = sp["span_id"]
                emb = (emb_dict.get(tid) or {}).get(sid)
                if emb is None:
                    continue
                label_vec = torch.zeros(n_nodes, dtype=torch.float32)
                if sp["is_correct"]:
                    label_vec[CORRECT_IDX] = 1.0
                else:
                    for cat in sp["labels"]:
                        if cat in label_map:
                            label_vec[label_map[cat]] = 1.0
                result.append({
                    "emb":      emb.float(),
                    "label_vec": label_vec,
                    "trace_id": tid,
                    "span_id":  sid,
                    "is_correct": sp["is_correct"],
                    "labels":   sp["labels"],
                })
    return result


def build_adj_matrix(
    edge_index: torch.Tensor,    # (2, E)
    edge_weight: torch.Tensor,   # (E,)
    n_nodes: int,
) -> torch.Tensor:
    """Dense N×N adjacency matrix (only error nodes 0..18 are connected)."""
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    src, dst = edge_index[0], edge_index[1]
    adj[src, dst] = edge_weight
    return adj


def build_controlled_adj(
    adj_golden: torch.Tensor,   # (N, N) from build_adj_matrix
    adj_type: str,              # "golden" | "self_loop" | "random"
    n_nodes: int,
    correct_idx: int,           # index of the Correct node (19)
    rand_seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (adj, A_gold) for ablation control conditions.

    adj    — (N, N) adjacency used for GAT message passing
    A_gold — (correct_idx, correct_idx) binary target for L_graph

    Correct node (index correct_idx) is always isolated (no cross-edges).

    golden   : standard CAPRI/Suppes correlation graph
    self_loop: I_N — each node attends only to itself; isolates whether
               neighbor propagation is the damage vs. the GAT block itself
    random   : density-matched random directed graph over the 19 error nodes;
               tests whether the golden graph provides structure beyond chance
    """
    n_err = correct_idx   # 19 error nodes

    if adj_type == "golden":
        A_gold = (adj_golden[:n_err, :n_err] > 0).float()
        return adj_golden, A_gold

    elif adj_type == "self_loop":
        adj    = torch.eye(n_nodes, dtype=torch.float32)
        # No edges between distinct error nodes → L_graph pushes all cross-pairs apart
        A_gold = torch.zeros(n_err, n_err, dtype=torch.float32)
        return adj, A_gold

    elif adj_type == "random":
        # Match directed-edge density of golden graph (error nodes, no self-loops)
        golden_err   = (adj_golden[:n_err, :n_err] > 0).float()
        n_possible   = n_err * (n_err - 1)
        diag_edges   = int(golden_err.diagonal().sum().item())
        n_edges      = int(golden_err.sum().item()) - diag_edges
        density      = n_edges / max(n_possible, 1)

        rng = torch.Generator()
        rng.manual_seed(rand_seed)
        rand_edges = (torch.rand(n_err, n_err, generator=rng) < density).float()
        rand_edges.fill_diagonal_(0.0)      # no self-loops; DenseGATLayer handles isolation

        adj        = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        adj[:n_err, :n_err] = rand_edges   # Correct node stays isolated
        A_gold     = rand_edges            # L_graph reconstructs this random structure
        return adj, A_gold

    else:
        raise ValueError(f"Unknown adj_type {adj_type!r} — choose: golden, self_loop, random")


# ---------------------------------------------------------------------------
# Category F1 evaluation (trace-level max-pool, 19 error types only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cat_f1(
    model: GATModel,
    spans: list[dict],
    x: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Returns (f1, threshold) — evaluates at fixed threshold=0.5.
    Threshold is kept as a parameter for API compatibility but always fixed at 0.5.
    """
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    trace_maxes = []
    y_true_list = []

    for tid, trace_spans in by_trace.items():
        embs = torch.stack([sp["emb"] for sp in trace_spans]).to(device)
        scores = model.score_spans(embs, Z)                       # (K, 20)
        probs  = torch.sigmoid(scores)                            # (K, 20)
        trace_max = probs[:, :CORRECT_IDX].max(dim=0).values.cpu().numpy()  # (19,)
        trace_maxes.append(trace_max)

        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in trace_spans:
            if not sp["is_correct"]:
                for cat_idx in [i for i, lv in enumerate(sp["label_vec"][:CORRECT_IDX].tolist()) if lv > 0]:
                    gt[cat_idx] = 1
        y_true_list.append(gt)

    if not trace_maxes:
        return 0.0, 0.5

    y_true    = np.array(y_true_list)   # (T, 19)
    all_maxes = np.array(trace_maxes)   # (T, 19)

    y_pred = (all_maxes > 0.5).astype(int)
    f1     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return f1, 0.5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    if not GRAPH_INPUT.exists():
        raise FileNotFoundError(f"{GRAPH_INPUT} not found — run 04_build_graph_input.py first")

    gi = torch.load(GRAPH_INPUT, weights_only=False)
    x            = gi["x"].float()                    # (20, 4096)
    edge_index   = gi["edge_index"]                   # (2, E)
    edge_weight  = gi["edge_weight"]                  # (E,)
    n_nodes      = gi["n_nodes"]
    node_names   = gi["node_names"]
    feat_dim     = x.shape[1]

    adj_golden = build_adj_matrix(edge_index, edge_weight, n_nodes)
    adj, A_gold = build_controlled_adj(
        adj_golden, args.adj_type, n_nodes, CORRECT_IDX, rand_seed=args.seed + 1
    )
    adj    = adj.to(device)
    A_gold = A_gold.to(device)
    x      = x.to(device)

    log.info("adj_type=%s  |  edges in adj: %d  |  edges in A_gold: %d",
             args.adj_type, int((adj > 0).sum().item()), int(A_gold.sum().item()))

    # ------------------------------------------------------------------
    # Load label map and span data
    # ------------------------------------------------------------------
    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())

    log.info("Loading span embeddings...")
    train_embs = load_span_embeddings("train")
    val_embs   = load_span_embeddings("val")

    train_spans = build_flat_span_list(DATASET_FILE, "train", train_embs, label_map, n_nodes)
    val_spans   = build_flat_span_list(DATASET_FILE, "val",   val_embs,   label_map, n_nodes)
    log.info("Train spans: %d  Val spans: %d", len(train_spans), len(val_spans))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GATModel(
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_nodes=n_nodes,
        dropout=args.dropout,
        n_layers=args.n_layers,
    ).to(device)
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    # Positive weight for class imbalance (correct spans ~77%)
    # Annotated / Correct ratio per class is unbalanced; use pos_weight for BCE
    n_annot = sum(1 for sp in train_spans if not sp["is_correct"])
    n_corr  = len(train_spans) - n_annot
    pos_weight_val = n_corr / max(n_annot, 1)
    pos_weights = torch.ones(n_nodes, device=device)
    pos_weights[:CORRECT_IDX] = pos_weight_val    # upweight error nodes

    bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    best_f1    = -1.0
    best_thr   = 0.5
    best_epoch = 0
    patience_counter = 0

    log.info("Starting training for %d epochs (patience=%d)...", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_spans)

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(train_spans), args.batch_size):
            batch = train_spans[start: start + args.batch_size]
            embs   = torch.stack([sp["emb"] for sp in batch]).to(device)       # (B, D)
            labels = torch.stack([sp["label_vec"] for sp in batch]).to(device)  # (B, 20)

            optimizer.zero_grad()

            # Forward
            Z = model.encode_graph(x, adj)
            span_scores = model.score_spans(embs, Z)              # (B, 20)

            # Span-level loss
            L_span = bce_weighted(span_scores, labels)

            # Graph structure reconstruction loss (19×19)
            A_hat = torch.sigmoid(
                Z[:CORRECT_IDX] @ model.graph_bilinear @ Z[:CORRECT_IDX].T
            )
            L_graph = bce(A_hat, A_gold)

            loss = L_span + args.lambda_graph * L_graph
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation — sweep thresholds [0.05..0.5] and pick best
        val_f1, val_thr = evaluate_cat_f1(model, val_spans, x, adj, device)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_thr   = val_thr
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_f1":       val_f1,
                "val_threshold": val_thr,
                "args":         vars(args),
                "node_names":   node_names,
                "feat_dim":     feat_dim,
            }, MODEL_DIR / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch <= 3:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_F1=%.4f(thr=%.2f)  best=%.4f@%d",
                epoch, args.epochs, avg_loss, val_f1, val_thr, best_f1, best_epoch,
            )

        if patience_counter >= args.patience:
            log.info("Early stopping at epoch %d (no val F1 improvement for %d epochs)",
                     epoch, args.patience)
            break

    log.info("Training done. Best val Cat.F1=%.4f (thr=%.2f) at epoch %d",
             best_f1, best_thr, best_epoch)
    log.info("Best model saved → %s", MODEL_DIR / "best_model.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train GAT for span-level error classification")
    ap.add_argument("--hidden_dim",   type=int,   default=256)
    ap.add_argument("--n_heads",      type=int,   default=4)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--patience",     type=int,   default=10)
    ap.add_argument("--lambda_graph", type=float, default=0.1)
    ap.add_argument("--n_layers",     type=int,   default=2, choices=[1, 2],
                    help="Number of GAT layers (1 = single-hop, 2 = two-hop; default: 2)")
    ap.add_argument("--adj_type",     default="golden",
                    choices=["golden", "self_loop", "random"],
                    help="Adjacency for message passing: golden (CAPRI graph), "
                         "self_loop (I_N — no neighbor mixing), "
                         "random (density-matched random graph)")
    ap.add_argument("--gpu",          type=int,   default=0)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--split_tag",    default="",
                    help="Tag suffix for data/model/output dirs (e.g. 'train' → data_train/, "
                         "models_train/, outputs_train/). Empty = default dirs.")
    args = ap.parse_args()

    # Apply split_tag: only changes MODEL_DIR and OUTPUT_DIR (not DATA_DIR).
    # Training data is always from the default GAIA data dir.
    # Use split_tag to name ablation runs, e.g. --split_tag self_loop saves to
    # graph/models_self_loop/ and graph/outputs_self_loop/.
    if args.split_tag:
        tag = f"_{args.split_tag}"
        global OUTPUT_DIR, MODEL_DIR
        OUTPUT_DIR = BENCH_DIR / "graph" / f"outputs{tag}"
        MODEL_DIR  = BENCH_DIR / "graph" / f"models{tag}"

    train(args)


if __name__ == "__main__":
    main()
