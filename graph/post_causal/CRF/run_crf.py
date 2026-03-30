#!/usr/bin/env python3
"""
CRF (pairwise Ising decoder) post-hoc inference for TRAIL error type prediction.

Literature basis:
    Ising model / pairwise MRF for multi-label decoding; same family as
    GSPEN (Wick et al., 2015) and SEAL (Tu & Gimpel, 2018), which validate
    that structured energy-based decoding over label graphs improves multi-label
    classification beyond independent per-label thresholds.

    This implementation starts from the simplest viable version:
    greedy coordinate ascent on a pairwise Ising energy, with the Suppes graph
    restricted to a sparse validated/high-support/top-k edge set.

───────────────────────────────────────────────────────────────────────────────
MODEL
───────────────────────────────────────────────────────────────────────────────

Energy (score to maximise):

    Score(y | x) = Σ_i  u_i(x) · y_i  +  λ · Σ_{(i,j)∈E}  w_ij · y_i · y_j

    u_i(x) = baseline logit for label i   (frozen, no calibration in v1)
    w_ij   = edge weight from Suppes graph (fixed, not learned in v1)
    λ      = global pairwise scale         (tuned on val)
    E      = sparse edge set from the Suppes graph (three candidates)

Inference:

    ŷ = argmax_{y∈{0,1}^19} Score(y | x)

    Option A: greedy coordinate ascent (default)
        Initialize y from baseline logit thresholding.
        For each label i, flip y_i if it increases Score.
        Sweep until convergence or max_sweeps.

    Option B: mean-field
        Relax y_i ∈ [0,1] and iterate:
            q_i ← σ(u_i + λ · Σ_{(j,i)∈E} w_ji·q_j + λ · Σ_{(i,k)∈E} w_ik·q_k)
        Threshold final q_i.

───────────────────────────────────────────────────────────────────────────────
EDGE SETS
───────────────────────────────────────────────────────────────────────────────

Three candidate edge sets (loaded from graph/outputs/edge_list.csv):

    A  validated   : only the 11 manually validated causal edges (is_causal=True)
                     avg indegree ≈ 0.6 — most labels are unconstrained
                     these are the most trustworthy edges; good baseline

    B  high_support: edges with precedence_n >= 5 AND weight >= 0.10
                     proxy for bootstrap stability (no bootstrap freq file available)
                     avg indegree ≈ 1.9 — moderate sparsity
                     filtered by co-occurrence count and combined Suppes strength

    C  topk        : top-k incoming edges per label by combined score
                     combined_score_ij = 0.5·weight + 0.3·precedence + 0.2·pr_delta_norm
                     k=2 by default → avg indegree = 2.0
                     forces uniform sparsity, avoids hubs

CDN result showed that dense conditioning (avg indegree 8–12) is not selective enough.
These edge sets enforce indegree ≤ 3, which is the critical design difference.

───────────────────────────────────────────────────────────────────────────────
Usage (from trail-benchmark/):
    python graph/post_causal/CRF/run_crf.py
    python graph/post_causal/CRF/run_crf.py --decoder mean_field
    python graph/post_causal/CRF/run_crf.py --edge_set validated
    python graph/post_causal/CRF/run_crf.py --topk 3 --lam 0.2 --tau_init 0.3
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import importlib.util
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CRF_DIR      = Path(__file__).resolve().parent           # graph/post_causal/CRF/
POST_DIR     = CRF_DIR.parent                            # graph/post_causal/
GRAPH_DIR    = POST_DIR.parent                           # graph/
BASELINE_DIR = GRAPH_DIR / "baseline"
DATA_DIR     = GRAPH_DIR / "data"
OUTPUT_DIR   = CRF_DIR / "outputs"

# Reuse baseline cache from CDN if available, otherwise extract fresh
CDN_CACHE_DIR  = POST_DIR / "CDN" / "outputs"
EDGE_LIST_CSV  = GRAPH_DIR / "outputs" / "edge_list.csv"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CORRECT_IDX = 19
N_LABELS    = 19

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
# Numerics
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# Baseline logit extraction (with CDN-cache reuse)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_trace_logits(model, spans, x, adj, device):
    """Max-pool span logits to trace level. Returns (logits, y_true, trace_ids)."""
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))

    by_trace = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    logit_list, label_list, tid_list = [], [], []
    for tid in sorted(by_trace.keys()):
        ts = by_trace[tid]
        embs   = torch.stack([sp["emb"] for sp in ts]).to(device)
        scores = model.score_spans(embs, Z)                      # (K, 20)
        logit_list.append(scores[:, :CORRECT_IDX].max(dim=0).values.cpu().numpy())

        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in ts:
            for idx, v in enumerate(sp["label_vec"][:CORRECT_IDX].tolist()):
                if v > 0:
                    gt[idx] = 1
        label_list.append(gt)
        tid_list.append(tid)

    return (
        np.array(logit_list, dtype=np.float32),
        np.array(label_list, dtype=int),
        tid_list,
    )


def load_baseline_outputs(model, gi, adj, device, label_map, force: bool = False) -> dict:
    """
    Load trace-level baseline logits for all splits.

    Prefers CDN cache (graph/post_causal/CDN/outputs/baseline_outputs_{split}.npz)
    to avoid redundant baseline forward passes. Falls back to fresh extraction
    if CDN cache is absent or force=True.

    Cache written to CRF outputs dir if fresh extraction is needed.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x       = gi["x"].float().to(device)
    n_nodes = gi["n_nodes"]
    splits  = {}

    for split in ("train", "val", "test"):
        # Prefer CDN cache, then CRF cache, then extract
        cdn_cache = CDN_CACHE_DIR / f"baseline_outputs_{split}.npz"
        crf_cache = OUTPUT_DIR   / f"baseline_outputs_{split}.npz"

        cache = cdn_cache if (cdn_cache.exists() and not force) else crf_cache

        if cache.exists() and not force:
            log.info("Loading baseline outputs (%s) from %s", split, cache.parent.name)
            d = np.load(cache, allow_pickle=True)
            splits[split] = {
                "logits":    d["logits"],
                "y_true":    d["y_true"],
                "trace_ids": d["trace_ids"].tolist(),
            }
        else:
            log.info("Extracting baseline outputs for split=%s ...", split)
            emb_dict = torch.load(
                DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True
            )
            spans = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)
            logits, y_true, trace_ids = _get_trace_logits(model, spans, x, adj, device)
            np.savez(
                crf_cache,
                logits=logits, y_true=y_true,
                trace_ids=np.array(trace_ids, dtype=object),
            )
            log.info("Saved → %s  (%d traces)", crf_cache.name, len(logits))
            splits[split] = {"logits": logits, "y_true": y_true, "trace_ids": trace_ids}

    return splits


# ---------------------------------------------------------------------------
# Edge set construction
# ---------------------------------------------------------------------------

def build_edge_sets(edge_list_csv: Path, topk: int = 2) -> dict[str, list[tuple]]:
    """
    Build the three candidate sparse edge sets from edge_list.csv.

    Each edge is represented as (src_id, dst_id, weight) — weight is the
    value used as w_ij in the CRF pairwise term.

    Edge sets:
        validated   : is_causal=True (11 edges), w_ij = 1.0 (causal overrides)
        high_support: precedence_n >= 5 AND weight >= 0.10
                      proxy for bootstrap stability; avg indegree ≈ 1.9
        topk        : top-k incoming edges per label by combined score
                      combined = 0.5·weight + 0.3·precedence + 0.2·pr_delta_norm
                      avg indegree = topk = 2 (by default)
    """
    rows = list(csv.DictReader(open(edge_list_csv)))

    # Ignore the self-duplicate: edge_list has 156 rows (155 unique + 1 duplicate)
    seen = set()
    unique_rows = []
    for r in rows:
        key = (r["src_id"], r["dst_id"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    def to_edge(r):
        return (int(r["src_id"]), int(r["dst_id"]), float(r["weight"]))

    # Edge set A: validated causal edges only
    validated = [to_edge(r) for r in unique_rows if r["is_causal"] == "True"]

    # Edge set B: high-support edges (precedence_n >= 5, weight >= 0.10)
    high_support = [
        to_edge(r) for r in unique_rows
        if int(r["precedence_n"]) >= 5 and float(r["weight"]) >= 0.10
    ]

    # Edge set C: top-k per destination by combined score
    for r in unique_rows:
        r["_score"] = (
            0.5 * float(r["weight"])
            + 0.3 * float(r["precedence"])
            + 0.2 * float(r["pr_delta_norm"])
        )
    by_dst: dict[str, list] = defaultdict(list)
    for r in unique_rows:
        by_dst[r["dst_id"]].append(r)

    topk_edges = []
    for _, ers in by_dst.items():
        top = sorted(ers, key=lambda x: -x["_score"])[:topk]
        topk_edges.extend(to_edge(r) for r in top)

    sets = {
        "validated":    validated,
        "high_support": high_support,
        "topk":         topk_edges,
    }
    for name, edges in sets.items():
        avg_indegree = len(edges) / N_LABELS
        log.info(
            "Edge set %-14s : %3d edges  avg indegree=%.2f",
            name, len(edges), avg_indegree,
        )
    return sets


# ---------------------------------------------------------------------------
# Decoder A: greedy coordinate ascent
# ---------------------------------------------------------------------------

def decode_greedy(
    unary: np.ndarray,          # (N, L)  baseline logits
    edges: list[tuple],         # [(i, j, w), ...]
    lam: float,
    tau_init: float,
    max_sweeps: int = 30,
) -> np.ndarray:
    """
    Greedy coordinate ascent on the Ising energy.

    For each label i, the gain of setting y_i=1 vs y_i=0 is:
        gain_i = u_i  +  λ · Σ_{j: (i,j)∈E} w_ij·y_j
                       +  λ · Σ_{j: (j,i)∈E} w_ji·y_j

    y_i = 1  iff  gain_i > 0,  else  y_i = 0.

    Sweeps until no label flips or max_sweeps reached.

    Returns binary predictions (N, L).
    """
    _, L = unary.shape

    # Build neighbour lists: nbrs[i] = list of (j, w) for all edges incident to i
    # Both (i,j) and (j,i) edges contribute to i's gain
    nbrs: list[list[tuple]] = [[] for _ in range(L)]
    for src, dst, w in edges:
        nbrs[src].append((dst, w))
        nbrs[dst].append((src, w))

    y = (unary >= tau_init).astype(np.float32)   # (N, L) binary init

    for _ in range(max_sweeps):
        changed = False
        for i in range(L):
            # gain of setting y_i = 1 vs 0
            neighbor_support = sum(w * y[:, j] for j, w in nbrs[i]) if nbrs[i] else 0.0
            gain = unary[:, i] + lam * neighbor_support    # (N,)
            new_yi = (gain > 0).astype(np.float32)         # (N,)
            if not np.array_equal(new_yi, y[:, i]):
                y[:, i] = new_yi
                changed = True
        if not changed:
            break

    return y.astype(int)


# ---------------------------------------------------------------------------
# Decoder B: mean-field
# ---------------------------------------------------------------------------

def decode_mean_field(
    unary: np.ndarray,          # (N, L)
    edges: list[tuple],         # [(i, j, w), ...]
    lam: float,
    tau_init: float,
    threshold: float,
    max_sweeps: int = 30,
) -> np.ndarray:
    """
    Mean-field variational inference on the pairwise CRF.

    Relaxes y_i ∈ {0,1} to q_i ∈ [0,1]:
        q_i ← σ(u_i + λ · Σ_{(j,i)∈E} w_ji·q_j + λ · Σ_{(i,k)∈E} w_ik·q_k)

    Uses both incoming and outgoing edges symmetrically (same as greedy).
    Iterates until convergence or max_sweeps.

    Returns binary predictions (N, L) after thresholding final q.
    """
    _, L = unary.shape

    nbrs: list[list[tuple]] = [[] for _ in range(L)]
    for src, dst, w in edges:
        nbrs[src].append((dst, w))
        nbrs[dst].append((src, w))

    q = _sigmoid(unary) * (1 if tau_init is None else 1.0)
    q = (_sigmoid(unary) >= tau_init).astype(np.float32)   # binary init

    for _ in range(max_sweeps):
        q_new = q.copy()
        for i in range(L):
            msg = sum(w * q[:, j] for j, w in nbrs[i]) if nbrs[i] else 0.0
            q_new[:, i] = _sigmoid(unary[:, i] + lam * msg)
        if np.allclose(q_new, q, atol=1e-5):
            break
        q = q_new

    return (q >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Val sweep
# ---------------------------------------------------------------------------

def sweep_val(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    edges: list[tuple],
    decoder: str,
    lam_grid: list[float],
    tau_grid: list[float],
    thr_grid: list[float] | None,
) -> tuple[float, float, float, float]:
    """
    Grid search over (λ, τ_init, [threshold]) on val.

    For greedy: threshold is implicit (y is already binary after decoding);
                τ_init acts as both init and final threshold.
    For mean_field: τ_init initialises q; threshold binarises final q.

    Returns (best_lam, best_tau, best_thr, best_f1).
    """
    best_f1, best_lam, best_tau, best_thr = 0.0, lam_grid[0], tau_grid[0], 0.5

    for lam in lam_grid:
        for tau in tau_grid:
            if decoder == "greedy":
                y_pred = decode_greedy(logits_val, edges, lam=lam, tau_init=tau)
                f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
                if f1 > best_f1:
                    best_f1, best_lam, best_tau, best_thr = f1, lam, tau, tau
            else:  # mean_field
                for thr in (thr_grid or [0.5]):
                    y_pred = decode_mean_field(
                        logits_val, edges, lam=lam, tau_init=tau, threshold=thr
                    )
                    f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
                    if f1 > best_f1:
                        best_f1, best_lam, best_tau, best_thr = f1, lam, tau, thr

    log.info(
        "Val best: λ=%.2f, τ_init=%.2f, thr=%.2f → weighted F1=%.4f",
        best_lam, best_tau, best_thr, best_f1,
    )
    return best_lam, best_tau, best_thr, best_f1


# ---------------------------------------------------------------------------
# Span-level location and joint accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_span_location_metrics(
    model,
    gi: dict,
    adj: torch.Tensor,
    device: torch.device,
    split: str,
    label_map: dict,
    crf_trace_pred: np.ndarray,    # (T, 19) CRF binary predictions
    threshold: float,
    trace_ids: list[str],
) -> tuple[float, float]:
    """
    Span-level metrics using CRF trace predictions as a hard mask on baseline span probs.

    A span is flagged as error if:
        ∃ label i : crf_trace_pred[t, i] == 1 AND baseline_span_prob[k, i] > threshold
    """
    model.eval()
    x       = gi["x"].float().to(device)
    n_nodes = gi["n_nodes"]
    Z       = model.encode_graph(x, adj)

    emb_dict = torch.load(DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True)
    spans    = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)

    by_trace: dict[str, list] = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    tid_to_idx = {tid: idx for idx, tid in enumerate(trace_ids)}

    loc_acc_list, joint_acc_list = [], []

    for tid in sorted(by_trace.keys()):
        ts = by_trace[tid]
        if tid not in tid_to_idx:
            continue
        t_idx    = tid_to_idx[tid]
        crf_mask = crf_trace_pred[t_idx].astype(np.float32)   # (19,)

        embs       = torch.stack([sp["emb"] for sp in ts]).to(device)
        scores     = model.score_spans(embs, Z)
        span_probs = torch.sigmoid(scores).cpu().numpy()[:, :CORRECT_IDX]   # (K, 19)

        masked = span_probs * crf_mask[np.newaxis, :]    # (K, 19)

        pred_error    = masked.max(axis=1) > threshold
        gt_span_ids   = {sp["span_id"] for sp in ts if not sp["is_correct"]}
        pred_span_ids = {sp["span_id"] for sp, e in zip(ts, pred_error) if e}

        loc_acc = (
            len(gt_span_ids & pred_span_ids) / len(gt_span_ids)
            if gt_span_ids else (1.0 if not pred_span_ids else 0.0)
        )
        loc_acc_list.append(loc_acc)

        gt_pairs, pred_pairs = set(), set()
        for k, sp in enumerate(ts):
            sid = sp["span_id"]
            if not sp["is_correct"]:
                for cat in sp["labels"]:
                    gt_pairs.add((sid, cat))
            for cat, cat_idx in label_map.items():
                if cat_idx < CORRECT_IDX and masked[k, cat_idx] > threshold:
                    pred_pairs.add((sid, cat))

        joint_acc = (
            len(gt_pairs & pred_pairs) / len(gt_pairs)
            if gt_pairs else (1.0 if not pred_span_ids else 0.0)
        )
        joint_acc_list.append(joint_acc)

    return float(np.mean(loc_acc_list)), float(np.mean(joint_acc_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pairwise Ising CRF post-hoc decoder for TRAIL error classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--edge_set", default="all",
        choices=["validated", "high_support", "topk", "all"],
        help="Suppes edge set to use. 'all' runs all three and reports best.",
    )
    ap.add_argument(
        "--decoder", default="greedy",
        choices=["greedy", "mean_field"],
        help="Decoding algorithm: greedy coordinate ascent or mean-field.",
    )
    ap.add_argument(
        "--topk", type=int, default=2,
        help="k for top-k-per-destination edge set (ignored unless edge_set=topk or all).",
    )
    ap.add_argument(
        "--lam", type=float, default=None,
        help="Fixed λ (pairwise scale). Default: grid-search on val.",
    )
    ap.add_argument(
        "--tau_init", type=float, default=None,
        help="Fixed τ_init (init threshold). Default: grid-search on val.",
    )
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="Final threshold (mean_field only). Default: grid-search on val.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--force_extract", action="store_true",
        help="Re-extract baseline outputs, ignoring any existing cache.",
    )
    ap.add_argument("--split_tag", default="",
                    help="Tag suffix for data/model/output dirs (e.g. '712' → data_712/, "
                         "models_712/, outputs_712/). Empty = default dirs.")
    args = ap.parse_args()

    # Apply split_tag to module-level path variables
    if args.split_tag:
        tag = f"_{args.split_tag}"
        global DATA_DIR, OUTPUT_DIR, DATASET_FILE, GRAPH_INPUT, LABEL_MAP_FILE
        DATA_DIR       = GRAPH_DIR / f"data{tag}"
        OUTPUT_DIR     = CRF_DIR / f"outputs{tag}"
        DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
        GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
        LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph input + baseline model
    # ------------------------------------------------------------------
    gi         = torch.load(GRAPH_INPUT, weights_only=False)
    n_nodes    = gi["n_nodes"]
    adj        = build_adj_matrix(gi["edge_index"], gi["edge_weight"], n_nodes).to(device)
    node_names = gi["node_names"]
    error_names = node_names[:CORRECT_IDX]

    models_subdir = f"models_{args.split_tag}" if args.split_tag else "models"
    ckpt_path = BASELINE_DIR / models_subdir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} — run graph/baseline/run_baseline.py first")

    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_dim = ckpt.get("feat_dim", 4096)
    model    = NoGraphBaseline(
        feat_dim=feat_dim, hidden_dim=ckpt["args"]["hidden_dim"],
        n_nodes=n_nodes, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Baseline model: epoch=%d  val_F1=%.4f", ckpt["epoch"], ckpt["val_f1"])

    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())

    # ------------------------------------------------------------------
    # Load (or extract) baseline logits
    # ------------------------------------------------------------------
    splits     = load_baseline_outputs(model, gi, adj, device, label_map, force=args.force_extract)
    X_train    = splits["train"]["logits"]
    X_val      = splits["val"]["logits"]
    X_test     = splits["test"]["logits"]
    Y_val      = splits["val"]["y_true"]
    Y_test     = splits["test"]["y_true"]
    tids_val   = splits["val"]["trace_ids"]
    tids_test  = splits["test"]["trace_ids"]

    log.info(
        "Splits: train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # ------------------------------------------------------------------
    # Build edge sets
    # ------------------------------------------------------------------
    all_edge_sets = build_edge_sets(EDGE_LIST_CSV, topk=args.topk)

    # Determine which edge sets to run
    if args.edge_set == "all":
        run_sets = list(all_edge_sets.keys())
    else:
        run_sets = [args.edge_set]

    # ------------------------------------------------------------------
    # Val sweep grids
    # ------------------------------------------------------------------
    lam_grid = (
        [args.lam] if args.lam is not None
        else [0.05, 0.1, 0.2, 0.3, 0.5]
    )
    tau_grid = (
        [args.tau_init] if args.tau_init is not None
        else [0.2, 0.25, 0.3, 0.35, 0.5]
    )
    thr_grid = (
        [args.threshold] if args.threshold is not None
        else [round(t, 2) for t in np.arange(0.2, 0.55, 0.05)]
    )

    BASELINE_FIXED_THR = 0.5
    best_overall: dict = {}   # track best config across edge sets
    all_results: list[dict] = []

    for eset_name in run_sets:
        edges = all_edge_sets[eset_name]

        log.info(
            "\n[%s | %s] sweeping λ∈%s × τ_init∈%s on val ...",
            eset_name, args.decoder, lam_grid, tau_grid,
        )
        best_lam, best_tau, best_thr, _ = sweep_val(
            X_val, Y_val, edges,
            decoder=args.decoder,
            lam_grid=lam_grid,
            tau_grid=tau_grid,
            thr_grid=thr_grid if args.decoder == "mean_field" else None,
        )

        # ------ Evaluate on val + test ------
        for split, X, Y_true, trace_ids in [
            ("val",  X_val,  Y_val,  tids_val),
            ("test", X_test, Y_test, tids_test),
        ]:
            if args.decoder == "greedy":
                y_pred = decode_greedy(X, edges, lam=best_lam, tau_init=best_tau)
            else:
                y_pred = decode_mean_field(
                    X, edges, lam=best_lam, tau_init=best_tau, threshold=best_thr
                )

            # Baseline at fixed 0.5 for comparison
            y_pred_base  = (_sigmoid(X) >= BASELINE_FIXED_THR).astype(int)
            base_metrics = compute_metrics(Y_true, y_pred_base, error_names)

            metrics  = compute_metrics(Y_true, y_pred, error_names)
            delta_f1 = metrics["f1_weighted"] - base_metrics["f1_weighted"]

            # Span-level metrics (use best_thr as span probability threshold)
            span_thr = best_thr if args.decoder == "mean_field" else best_tau
            loc_acc, joint_acc = compute_span_location_metrics(
                model, gi, adj, device,
                split=split, label_map=label_map,
                crf_trace_pred=y_pred,
                threshold=span_thr,
                trace_ids=trace_ids,
            )

            if split == "test":
                print(f"\n{'='*65}")
                print(
                    f"CRF ({args.decoder}, edge_set={eset_name}) — test set  "
                    f"[λ={best_lam:.2f}, τ_init={best_tau:.2f}, thr={best_thr:.2f}]"
                )
                print(f"{'='*65}")
                print(classification_report(
                    Y_true, y_pred, target_names=error_names, zero_division=0
                ))
                print_metrics(metrics, header="--- CRF ---")
                print_metrics(
                    base_metrics,
                    header=f"\n--- Baseline (thr={BASELINE_FIXED_THR:.2f}, no CRF) ---",
                )
                print(f"\nΔ F1 weighted vs baseline (thr=0.50): {delta_f1:+.4f}")
                print(f"Location Accuracy (Loc. Acc.):        {loc_acc:.4f}")
                print(f"Joint Accuracy    (Joint Acc.):       {joint_acc:.4f}")

            tag = f"crf_{args.decoder}_{eset_name}"
            result = {
                "model":                f"CRF_{args.decoder}_{eset_name}",
                "split":                split,
                "decoder":              args.decoder,
                "edge_set":             eset_name,
                "n_edges":              len(edges),
                "lam":                  best_lam,
                "tau_init":             best_tau,
                "threshold":            best_thr,
                "topk":                 args.topk,
                "n_traces":             int(len(X)),
                **{k: v for k, v in metrics.items() if k != "per_class"},
                "location_accuracy":    loc_acc,
                "joint_accuracy":       joint_acc,
                "per_class":            metrics["per_class"],
                "baseline_f1_weighted": base_metrics["f1_weighted"],
                "delta_f1_weighted":    delta_f1,
            }
            out_path = OUTPUT_DIR / f"eval_results_{tag}_{split}.json"
            out_path.write_text(json.dumps(result, indent=2))
            log.info("Saved → %s", out_path)
            all_results.append(result)

            if split == "test":
                if (not best_overall
                        or metrics["f1_weighted"] > best_overall["f1_weighted"]):
                    best_overall = {
                        "edge_set": eset_name,
                        "lam":      best_lam,
                        "tau_init": best_tau,
                        "thr":      best_thr,
                        **{k: metrics[k] for k in
                           ("f1_weighted", "f1_macro", "f1_micro", "precision", "recall")},
                        "delta_f1_weighted": delta_f1,
                    }

    # ------------------------------------------------------------------
    # Summary table across all edge sets
    # ------------------------------------------------------------------
    test_results = [r for r in all_results if r["split"] == "test"]
    if len(test_results) > 1:
        print(f"\n{'='*90}")
        print(f"Summary — test set  (decoder={args.decoder})")
        print(f"{'='*90}")
        hdr = f"{'Edge set':<15} {'n_edges':>7} {'λ':>5} {'τ':>5} {'W-F1':>7} {'Ma-F1':>7} {'Mi-F1':>7}  {'Δ vs base':>10}"
        print(hdr)
        print("-" * 90)
        baseline_f1 = test_results[0]["baseline_f1_weighted"]
        for r in test_results:
            print(
                f"  {r['edge_set']:<13} {r['n_edges']:>7} {r['lam']:>5.2f} "
                f"{r['tau_init']:>5.2f} {r['f1_weighted']:>7.4f} "
                f"{r['f1_macro']:>7.4f} {r['f1_micro']:>7.4f} "
                f"  {r['delta_f1_weighted']:>+10.4f}"
            )
        print("-" * 90)
        print(f"  {'Baseline (thr=0.5)':<13} {'—':>7} {'—':>5} {'0.50':>5} "
              f"{baseline_f1:>7.4f} {'—':>7} {'—':>7}  {'+0.0000':>10}")

        if best_overall:
            print(f"\nBest config: edge_set={best_overall['edge_set']}  "
                  f"λ={best_overall['lam']:.2f}  τ_init={best_overall['tau_init']:.2f}  "
                  f"weighted F1={best_overall['f1_weighted']:.4f}  "
                  f"Δ={best_overall['delta_f1_weighted']:+.4f}")


if __name__ == "__main__":
    main()
