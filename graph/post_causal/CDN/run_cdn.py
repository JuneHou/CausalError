#!/usr/bin/env python3
"""
CDN (Conditional Dependency Network) post-hoc inference for TRAIL error type prediction.

Reference:
    Heckerman D., Chickering D.M., Meek C., Rounthwaite R., Kadie C. (2000).
    Dependency Networks for Inference, Collaborative Filtering, and Data Visualization.
    Journal of Machine Learning Research, 1, 49–75.
    https://dl.acm.org/doi/abs/10.5555/2283516.2283613

───────────────────────────────────────────────────────────────────────────────
ALGORITHM
───────────────────────────────────────────────────────────────────────────────

Phase A — Training (graph-restricted CDN)
──────────────────────────────────────────
For each error label i (0..18):

    neighbors[i] = parents(i) ∪ children(i) in the Suppes graph
                   (or parents-only / children-only via --neighborhood flag)

    X_i = concat(
        baseline_logits_train,              # (N_train, 19)  frozen baseline outputs
        gold_labels_train[:, neighbors[i]]  # (N_train, d_i)  teacher-forced neighbors
    )
    y_i = gold_labels_train[:, i]

    model_i = LogisticRegression(class_weight='balanced').fit(X_i, y_i)

Key choices:
  - baseline_logits = max-pooled pre-sigmoid scores (not probabilities), because logistic
    regression applies its own linear transformation; logits carry more signal.
  - Teacher forcing: neighbors are gold labels at training time (standard CDN protocol).
  - class_weight='balanced': handles the severe class imbalance (~few positives per type).
  - Zero-positive labels (some types have 0 positive train traces): trivial classifier,
    always predicts 0.

Phase B — Inference (deterministic mean-field or stochastic Gibbs)
───────────────────────────────────────────────────────────────────
At test time, true neighbor labels are unknown. Use iterative updates:

    Initialize: y^(0) = (sigmoid(baseline_logits) >= 0.5)   ← strong prior from baseline

    For each sweep s = 1 .. n_sweeps:
        For each label i:
            p_i = model_i.predict_proba(concat(logits, y[:, neighbors[i]]))[1]

            Deterministic (mean-field):
                y_i ← p_i          (soft update; threshold at end of all sweeps)

            Stochastic (Gibbs):
                y_i ~ Bernoulli(p_i)

    Deterministic output: final soft probabilities → threshold at val-tuned thr
    Gibbs output: average binary samples after burn-in → threshold at val-tuned thr

Hyperparameter sweep on val:
  - n_sweeps ∈ {1, 3, 5, 10, 20}
  - threshold ∈ [0.05, 0.50]  (step 0.05)
  Best (n_sweeps, threshold) selected by weighted F1 on val.

───────────────────────────────────────────────────────────────────────────────
Usage (from trail-benchmark/):
    python graph/post_causal/CDN/run_cdn.py
    python graph/post_causal/CDN/run_cdn.py --inference gibbs --n_sweeps 20 --burn_in 5
    python graph/post_causal/CDN/run_cdn.py --neighborhood parents
    python graph/post_causal/CDN/run_cdn.py --C 0.1
    python graph/post_causal/CDN/run_cdn.py --force_extract   # re-run baseline extraction
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import importlib.util
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CDN_DIR      = Path(__file__).resolve().parent           # graph/post_causal/CDN/
POST_DIR     = CDN_DIR.parent                            # graph/post_causal/
GRAPH_DIR    = POST_DIR.parent                           # graph/
BASELINE_DIR = GRAPH_DIR / "baseline"
DATA_DIR     = GRAPH_DIR / "data"
OUTPUT_DIR   = CDN_DIR / "outputs"

DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CORRECT_IDX = 19
N_LABELS    = 19

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import shared helpers (same import pattern as run_causal_inference.py)
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
# Numerics helper
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# Step 1: Extract baseline trace-level logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_trace_logits(
    model,
    spans: list[dict],
    x: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run baseline model and max-pool span-level logits to trace level.

    Returns:
        logits   : (T, 19)  max-pooled pre-sigmoid scores per trace
        y_true   : (T, 19)  ground-truth multi-label binary matrix
        trace_ids: list[str] ordered trace IDs
    """
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))

    by_trace: dict[str, list] = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    logit_list, label_list, tid_list = [], [], []

    for tid in sorted(by_trace.keys()):
        trace_spans = by_trace[tid]
        embs   = torch.stack([sp["emb"] for sp in trace_spans]).to(device)
        scores = model.score_spans(embs, Z)                          # (K, 20)
        # Max-pool logits over spans — same aggregation as the baseline
        trace_max = scores[:, :CORRECT_IDX].max(dim=0).values        # (19,)
        logit_list.append(trace_max.cpu().numpy())

        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in trace_spans:
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


def extract_all_splits(
    model,
    gi: dict,
    adj: torch.Tensor,
    device: torch.device,
    label_map: dict,
    force: bool = False,
) -> dict[str, dict]:
    """
    Extract and cache trace-level baseline logits for train / val / test.

    Cache files: OUTPUT_DIR/baseline_outputs_{split}.npz
    Set force=True to ignore existing cache and re-extract.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x       = gi["x"].float().to(device)
    n_nodes = gi["n_nodes"]
    splits  = {}

    for split in ("train", "val", "test"):
        cache = OUTPUT_DIR / f"baseline_outputs_{split}.npz"

        if cache.exists() and not force:
            log.info("Loading cached baseline outputs (%s): %s", split, cache.name)
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
                cache,
                logits=logits,
                y_true=y_true,
                trace_ids=np.array(trace_ids, dtype=object),
            )
            log.info(
                "Saved baseline outputs (%s): %d traces  → %s",
                split, len(logits), cache.name,
            )
            splits[split] = {"logits": logits, "y_true": y_true, "trace_ids": trace_ids}

    return splits


# ---------------------------------------------------------------------------
# Step 2: Build Suppes neighborhood sets
# ---------------------------------------------------------------------------

def build_neighborhoods(
    gi: dict,
    neighborhood: str = "both",
) -> dict[int, list[int]]:
    """
    Build the conditioning set for each label from the Suppes graph.

    Edge direction convention (from build_adj_matrix):
        adj[src, dst] > 0  ⟺  edge src → dst  ⟺  src causally precedes dst

    Args:
        neighborhood:
            "parents"  — condition only on causal predecessors (sources pointing to i)
            "children" — condition only on causal successors (targets pointed to by i)
            "both"     — parents ∪ children  (default; faithful to CDN dependency model)

    Returns:
        neighbors: {label_index: sorted list of neighbor indices}
    """
    adj_full = build_adj_matrix(gi["edge_index"], gi["edge_weight"], gi["n_nodes"])
    A = adj_full[:CORRECT_IDX, :CORRECT_IDX].numpy()   # (19, 19)

    parents  = {i: [] for i in range(N_LABELS)}
    children = {i: [] for i in range(N_LABELS)}

    for src in range(N_LABELS):
        for dst in range(N_LABELS):
            if A[src, dst] > 0:
                parents[dst].append(src)
                children[src].append(dst)

    if neighborhood == "parents":
        nbrs = {i: sorted(set(parents[i]))                    for i in range(N_LABELS)}
    elif neighborhood == "children":
        nbrs = {i: sorted(set(children[i]))                   for i in range(N_LABELS)}
    else:  # "both"
        nbrs = {i: sorted(set(parents[i]) | set(children[i])) for i in range(N_LABELS)}

    n_with_nbrs = sum(1 for v in nbrs.values() if len(v) > 0)
    log.info(
        "Neighborhoods (%s): %d / %d labels have Suppes neighbors",
        neighborhood, n_with_nbrs, N_LABELS,
    )
    for i, ns in nbrs.items():
        if ns:
            log.debug("  label %d ← neighbors %s", i, ns)

    return nbrs


# ---------------------------------------------------------------------------
# Step 3: Train CDN classifiers (graph-restricted)
# ---------------------------------------------------------------------------

def _build_cdn_features(
    logits: np.ndarray,    # (N, 19) baseline logits
    y_nbr: np.ndarray,     # (N, 19) current neighbor label estimates
    nbr: list[int],        # indices of neighbors to append
) -> np.ndarray:
    """Concatenate baseline logits with the selected neighbor columns."""
    if not nbr:
        return logits
    return np.hstack([logits, y_nbr[:, nbr]])


def train_cdn(
    logits_train: np.ndarray,          # (N, 19)
    y_train: np.ndarray,               # (N, 19)
    neighbors: dict[int, list[int]],
    C: float = 1.0,
) -> list:
    """
    Train one logistic regression classifier per label.

    Training uses teacher-forced gold neighbor labels (CDN protocol).
    Labels with zero positive training examples get a None classifier
    (always predicts 0 at inference).

    Returns:
        classifiers: list of 19 entries, each a fitted LogisticRegression or None.
    """
    classifiers = []
    n_pos_total, n_zero, n_no_nbr = 0, 0, 0

    for i in range(N_LABELS):
        y_i   = y_train[:, i]
        n_pos = int(y_i.sum())

        if n_pos == 0:
            # Trivial: no positive examples in training data
            classifiers.append(None)
            n_zero += 1
            log.debug("Label %d: zero positives — trivial classifier", i)
            continue

        nbr = neighbors[i]
        if not nbr:
            n_no_nbr += 1

        X_i = _build_cdn_features(logits_train, y_train, nbr)

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            C=C,
            random_state=42,
        )
        clf.fit(X_i, y_i)
        classifiers.append(clf)
        n_pos_total += 1

    log.info(
        "CDN training: %d classifiers trained  |  %d zero-positive  |  %d with no neighbors",
        n_pos_total, n_zero, n_no_nbr,
    )
    return classifiers


# ---------------------------------------------------------------------------
# Step 4a: Deterministic mean-field inference
# ---------------------------------------------------------------------------

def cdn_inference_deterministic(
    logits_test: np.ndarray,              # (N, 19)
    classifiers: list,
    neighbors: dict[int, list[int]],
    n_sweeps: int = 10,
    init_threshold: float = 0.5,
) -> np.ndarray:
    """
    Deterministic mean-field CDN inference.

    Each sweep updates every label's soft probability using the current
    probability estimates of its neighbors (not hard binary values).
    This is a mean-field variational approximation to the CDN posterior.

    Returns:
        marginals: (N, 19) final soft probability estimates
    """
    # Initialize from baseline probabilities thresholded at init_threshold
    y = (_sigmoid(logits_test) >= init_threshold).astype(np.float32)   # (N, 19)

    for _ in range(n_sweeps):
        y_new = y.copy()
        for i in range(N_LABELS):
            if classifiers[i] is None:
                y_new[:, i] = 0.0
                continue
            X_i = _build_cdn_features(logits_test, y, neighbors[i])
            y_new[:, i] = classifiers[i].predict_proba(X_i)[:, 1]  # (N,)
        y = y_new

    return y   # (N, 19) soft probabilities — threshold at the end


# ---------------------------------------------------------------------------
# Step 4b: Stochastic Gibbs sampling inference
# ---------------------------------------------------------------------------

def cdn_inference_gibbs(
    logits_test: np.ndarray,              # (N, 19)
    classifiers: list,
    neighbors: dict[int, list[int]],
    n_sweeps: int = 20,
    burn_in: int = 5,
    init_threshold: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Stochastic Gibbs sampling CDN inference (original CDN paper protocol).

    Each label is sampled from Bernoulli(p_i) given the current binary
    state of its neighbors. Post-burn-in binary samples are averaged to
    estimate marginal probabilities.

    Returns:
        marginals: (N, 19) empirical marginal probabilities from Gibbs chain
    """
    rng = np.random.default_rng(seed)
    N   = logits_test.shape[0]
    y   = (_sigmoid(logits_test) >= init_threshold).astype(np.float32)   # (N, 19) binary init

    samples: list[np.ndarray] = []

    for sweep in range(n_sweeps):
        for i in range(N_LABELS):
            if classifiers[i] is None:
                y[:, i] = 0.0
                continue
            X_i = _build_cdn_features(logits_test, y, neighbors[i])
            p_i = classifiers[i].predict_proba(X_i)[:, 1]             # (N,)
            y[:, i] = (rng.random(N) < p_i).astype(np.float32)        # Bernoulli sample

        if sweep >= burn_in:
            samples.append(y.copy())

    if not samples:
        log.warning("No post-burn-in samples collected (n_sweeps=%d, burn_in=%d). "
                    "Returning final y.", n_sweeps, burn_in)
        return y

    return np.mean(samples, axis=0)   # (N, 19) marginal probabilities


# ---------------------------------------------------------------------------
# Step 4c: Run inference with the chosen mode
# ---------------------------------------------------------------------------

def run_inference(
    logits: np.ndarray,
    classifiers: list,
    neighbors: dict[int, list[int]],
    inference: str,
    n_sweeps: int,
    burn_in: int,
) -> np.ndarray:
    """Dispatch to deterministic or Gibbs inference. Returns (N, 19) marginals."""
    if inference == "gibbs":
        return cdn_inference_gibbs(
            logits, classifiers, neighbors,
            n_sweeps=n_sweeps, burn_in=burn_in,
        )
    return cdn_inference_deterministic(logits, classifiers, neighbors, n_sweeps=n_sweeps)


# ---------------------------------------------------------------------------
# Step 5: Hyperparameter sweep on val
# ---------------------------------------------------------------------------

def sweep_hyperparams(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    classifiers: list,
    neighbors: dict[int, list[int]],
    inference: str,
    n_sweeps_grid: list[int],
    thresholds: list[float],
    burn_in: int,
) -> tuple[int, float, float]:
    """
    Grid search over n_sweeps × threshold on the val split.

    Returns:
        best_n_sweeps: int
        best_thr:      float
        best_f1:       float  (val weighted F1)
    """
    best_f1, best_n_sweeps, best_thr = 0.0, n_sweeps_grid[0], thresholds[0]

    for n_sw in n_sweeps_grid:
        marginals = run_inference(
            logits_val, classifiers, neighbors,
            inference=inference, n_sweeps=n_sw, burn_in=burn_in,
        )
        for thr in thresholds:
            y_pred = (marginals >= thr).astype(int)
            f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            if f1 > best_f1:
                best_f1      = f1
                best_n_sweeps = n_sw
                best_thr     = thr

    log.info(
        "Val sweep best: n_sweeps=%d, thr=%.2f → weighted F1=%.4f",
        best_n_sweeps, best_thr, best_f1,
    )
    return best_n_sweeps, best_thr, best_f1


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
    cdn_trace_pred: np.ndarray,   # (T, 19) CDN binary trace predictions
    threshold: float,
    trace_ids: list[str],
) -> tuple[float, float]:
    """
    Compute span-level Location Accuracy and Joint Accuracy after CDN refinement.

    The CDN operates at trace level. For span-level metrics we apply CDN's binary
    trace prediction as a hard mask on baseline span probabilities:

        span is predicted as error ⟺
            ∃ label i : cdn_trace_pred[t, i] == 1
                        AND baseline_span_prob[k, i] > threshold

    This is analogous to the causal gate's multiplicative mask, but uses a hard
    binary gate (CDN predicts presence/absence, not a soft weight).

    Returns:
        loc_acc   : average fraction of GT error spans correctly identified
        joint_acc : average fraction of GT (span_id, category) pairs correctly predicted
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
        trace_spans = by_trace[tid]
        if tid not in tid_to_idx:
            continue
        t_idx       = tid_to_idx[tid]
        cdn_mask    = cdn_trace_pred[t_idx].astype(np.float32)   # (19,) binary

        embs       = torch.stack([sp["emb"] for sp in trace_spans]).to(device)
        scores     = model.score_spans(embs, Z)
        span_probs = torch.sigmoid(scores).cpu().numpy()[:, :CORRECT_IDX]   # (K, 19)

        # Apply CDN mask: zero out error types not predicted by CDN
        masked = span_probs * cdn_mask[np.newaxis, :]   # (K, 19)

        # A span is predicted as error if any masked probability exceeds threshold
        pred_error    = masked.max(axis=1) > threshold            # (K,) bool
        gt_span_ids   = {sp["span_id"] for sp in trace_spans if not sp["is_correct"]}
        pred_span_ids = {
            sp["span_id"] for sp, is_err in zip(trace_spans, pred_error) if is_err
        }

        loc_acc = (
            len(gt_span_ids & pred_span_ids) / len(gt_span_ids)
            if gt_span_ids else (1.0 if not pred_span_ids else 0.0)
        )
        loc_acc_list.append(loc_acc)

        gt_pairs, pred_pairs = set(), set()
        for k, sp in enumerate(trace_spans):
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
        description="Graph-restricted CDN post-hoc inference for TRAIL error classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--inference", default="deterministic",
        choices=["deterministic", "gibbs"],
        help="Inference mode: deterministic mean-field or stochastic Gibbs sampling",
    )
    ap.add_argument(
        "--neighborhood", default="both",
        choices=["parents", "children", "both"],
        help="Suppes graph conditioning set for each label",
    )
    ap.add_argument(
        "--n_sweeps", type=int, default=None,
        help="Fixed number of inference sweeps. Default: grid-search on val.",
    )
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed classification threshold. Default: grid-search on val.",
    )
    ap.add_argument(
        "--burn_in", type=int, default=5,
        help="Burn-in sweeps for Gibbs (ignored for deterministic).",
    )
    ap.add_argument(
        "--C", type=float, default=1.0,
        help="Logistic regression inverse regularization strength.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--force_extract", action="store_true",
        help="Re-extract baseline outputs even if cache files exist.",
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
        OUTPUT_DIR     = CDN_DIR / f"outputs{tag}"
        DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
        GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
        LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph input
    # ------------------------------------------------------------------
    gi         = torch.load(GRAPH_INPUT, weights_only=False)
    n_nodes    = gi["n_nodes"]
    adj        = build_adj_matrix(gi["edge_index"], gi["edge_weight"], n_nodes).to(device)
    node_names = gi["node_names"]
    error_names = node_names[:CORRECT_IDX]

    # ------------------------------------------------------------------
    # Load baseline model
    # ------------------------------------------------------------------
    models_subdir = f"models_{args.split_tag}" if args.split_tag else "models"
    ckpt_path = BASELINE_DIR / models_subdir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} — run graph/baseline/run_baseline.py first"
        )

    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_dim = ckpt.get("feat_dim", 4096)
    model    = NoGraphBaseline(
        feat_dim   = feat_dim,
        hidden_dim = ckpt["args"]["hidden_dim"],
        n_nodes    = n_nodes,
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(
        "Baseline model loaded: epoch=%d  val_F1=%.4f",
        ckpt["epoch"], ckpt["val_f1"],
    )

    label_map: dict[str, int] = json.loads(LABEL_MAP_FILE.read_text())

    # ------------------------------------------------------------------
    # Step 1: Extract (or load cached) trace-level baseline logits
    # ------------------------------------------------------------------
    splits = extract_all_splits(
        model, gi, adj, device, label_map, force=args.force_extract
    )

    X_train, Y_train = splits["train"]["logits"], splits["train"]["y_true"]
    X_val,   Y_val   = splits["val"]["logits"],   splits["val"]["y_true"]
    X_test,  Y_test  = splits["test"]["logits"],  splits["test"]["y_true"]
    trace_ids_val    = splits["val"]["trace_ids"]
    trace_ids_test   = splits["test"]["trace_ids"]

    log.info(
        "Splits: train=%d traces (%d pos)  val=%d  test=%d",
        len(X_train), int(Y_train.sum()), len(X_val), len(X_test),
    )

    # ------------------------------------------------------------------
    # Step 2: Build Suppes neighborhood sets
    # ------------------------------------------------------------------
    neighbors = build_neighborhoods(gi, neighborhood=args.neighborhood)

    # ------------------------------------------------------------------
    # Step 3: Train CDN classifiers (with caching)
    # ------------------------------------------------------------------
    model_tag  = f"{args.neighborhood}_C{args.C}"
    cdn_cache  = OUTPUT_DIR / f"cdn_models_{model_tag}.pkl"

    if cdn_cache.exists():
        log.info("Loading cached CDN models: %s", cdn_cache.name)
        with open(cdn_cache, "rb") as fh:
            classifiers = pickle.load(fh)
    else:
        classifiers = train_cdn(X_train, Y_train, neighbors, C=args.C)
        with open(cdn_cache, "wb") as fh:
            pickle.dump(classifiers, fh)
        log.info("CDN models saved → %s", cdn_cache.name)

    # ------------------------------------------------------------------
    # Step 4: Tune n_sweeps and threshold on val
    # ------------------------------------------------------------------
    n_sweeps_grid = (
        [args.n_sweeps] if args.n_sweeps is not None
        else [1, 3, 5, 10, 20]
    )
    thresholds = (
        [args.threshold] if args.threshold is not None
        else [round(t, 2) for t in np.arange(0.05, 0.55, 0.05)]
    )

    log.info(
        "Val sweep: inference=%s, n_sweeps∈%s, thresholds∈[%.2f..%.2f]",
        args.inference, n_sweeps_grid, thresholds[0], thresholds[-1],
    )
    best_n_sweeps, best_thr, best_val_f1 = sweep_hyperparams(
        X_val, Y_val, classifiers, neighbors,
        inference=args.inference,
        n_sweeps_grid=n_sweeps_grid,
        thresholds=thresholds,
        burn_in=args.burn_in,
    )

    # ------------------------------------------------------------------
    # Step 5: Evaluate on val and test
    # ------------------------------------------------------------------
    BASELINE_FIXED_THR = 0.5   # fixed threshold for baseline comparison (same as causal gate)

    all_results = {}

    for split, X, Y_true, trace_ids in [
        ("val",  X_val,  Y_val,  trace_ids_val),
        ("test", X_test, Y_test, trace_ids_test),
    ]:
        # CDN predictions
        marginals = run_inference(
            X, classifiers, neighbors,
            inference=args.inference,
            n_sweeps=best_n_sweeps,
            burn_in=args.burn_in,
        )
        y_pred = (marginals >= best_thr).astype(int)

        # Baseline at fixed 0.5 for comparison
        y_pred_base  = (_sigmoid(X) >= BASELINE_FIXED_THR).astype(int)
        base_metrics = compute_metrics(Y_true, y_pred_base, error_names)

        metrics  = compute_metrics(Y_true, y_pred, error_names)
        delta_f1 = metrics["f1_weighted"] - base_metrics["f1_weighted"]

        # Span-level location + joint accuracy
        loc_acc, joint_acc = compute_span_location_metrics(
            model, gi, adj, device,
            split=split,
            label_map=label_map,
            cdn_trace_pred=y_pred,
            threshold=best_thr,
            trace_ids=trace_ids,
        )

        # ------ Print ------
        print(f"\n{'='*65}")
        print(
            f"CDN ({args.inference}, nbr={args.neighborhood}) — {split} set  "
            f"[n_sweeps={best_n_sweeps}, thr={best_thr:.2f}]"
        )
        print(f"{'='*65}")
        print(
            classification_report(Y_true, y_pred, target_names=error_names, zero_division=0)
        )
        print_metrics(metrics, header="--- CDN ---")
        print_metrics(
            base_metrics,
            header=f"\n--- Baseline (thr={BASELINE_FIXED_THR:.2f}, no CDN) ---",
        )
        print(f"\nΔ F1 weighted vs baseline (thr=0.50): {delta_f1:+.4f}")
        print(f"Location Accuracy (Loc. Acc.):        {loc_acc:.4f}")
        print(f"Joint Accuracy    (Joint Acc.):       {joint_acc:.4f}")

        # ------ Save ------
        tag = f"cdn_{args.inference}_{args.neighborhood}"
        result = {
            "model":                f"CDN_{args.inference}_{args.neighborhood}",
            "split":                split,
            "inference":            args.inference,
            "neighborhood":         args.neighborhood,
            "n_sweeps":             best_n_sweeps,
            "burn_in":              args.burn_in if args.inference == "gibbs" else None,
            "threshold":            best_thr,
            "C":                    args.C,
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
        all_results[split] = result

    # Summary table
    print(f"\n{'='*65}")
    print("Summary")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'Val':>10} {'Test':>10}  {'Baseline (test, thr=0.5)':>24}")
    for m in ("f1_weighted", "f1_macro", "f1_micro", "precision", "recall"):
        v  = all_results["val"].get(m, float("nan"))
        t  = all_results["test"].get(m, float("nan"))
        b  = all_results["test"].get("baseline_f1_weighted", float("nan")) if m == "f1_weighted" else float("nan")
        b_str = f"{b:.4f}" if m == "f1_weighted" else "—"
        print(f"  {m:<23} {v:>10.4f} {t:>10.4f}  {b_str:>24}")


if __name__ == "__main__":
    main()
