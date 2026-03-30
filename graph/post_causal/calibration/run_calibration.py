#!/usr/bin/env python3
"""
Per-label threshold calibration for TRAIL multi-label error classification.

Motivation
──────────
All post-hoc methods (causal gate, CDN, CRF) use a single global threshold
applied identically to all 19 error types. But the baseline over-predicts on
common error types and under-predicts on rare ones. A per-label threshold
τ_i* lets each label independently control the precision-recall trade-off.

Relation to "Calibrate and Rerank" (Pillai et al., ECML 2020)
──────────────────────────────────────────────────────────────
The calibration stage of the paper adjusts per-class decision boundaries using
a small validation set. This script implements the calibration stage directly:
no reranking, no learned scorer — just find the threshold per label that
maximises F1_i on val.

Two strategies are implemented and compared:

  Strategy A — per-label F1 optimisation
    For each label i:
        τ_i* = argmax_{τ ∈ grid} F1_i(y_val[:, i], predict(X_val[:, i], τ))
    Labels with zero val support fall back to the global best threshold.

  Strategy B — global greedy coordinate ascent on weighted F1
    Initialise all τ_i = global_best_τ.
    Repeat until convergence:
        For each label i:
            τ_i ← argmax_{τ ∈ grid} weighted_F1(y_val, predict(X_val, τ))
            where all τ_j (j ≠ i) are held fixed.
    This directly optimises the target metric rather than per-label surrogates.

Usage (from trail-benchmark/):
    python graph/post_causal/calibration/run_calibration.py
    python graph/post_causal/calibration/run_calibration.py --strategy A
    python graph/post_causal/calibration/run_calibration.py --strategy B
    python graph/post_causal/calibration/run_calibration.py --strategy both
"""

import argparse
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
CAL_DIR      = Path(__file__).resolve().parent           # graph/post_causal/calibration/
POST_DIR     = CAL_DIR.parent                            # graph/post_causal/
GRAPH_DIR    = POST_DIR.parent                           # graph/
BASELINE_DIR = GRAPH_DIR / "baseline"
DATA_DIR     = GRAPH_DIR / "data"
OUTPUT_DIR   = CAL_DIR / "outputs"

DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

CDN_CACHE_DIR = POST_DIR / "CDN" / "outputs"

CORRECT_IDX = 19
N_LABELS    = 19

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
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

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# Load cached baseline logits (prefer CDN cache, fall back to extracting fresh)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_logits(model, spans, x, adj, device):
    """Max-pool span logits to trace level."""
    model.eval()
    Z = model.encode_graph(x.to(device), adj.to(device))
    by_trace = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    logit_list, label_list, tid_list = [], [], []
    for tid in sorted(by_trace.keys()):
        ts   = by_trace[tid]
        embs = torch.stack([sp["emb"] for sp in ts]).to(device)
        scores = model.score_spans(embs, Z)
        logit_list.append(scores[:, :CORRECT_IDX].max(dim=0).values.cpu().numpy())
        gt = np.zeros(CORRECT_IDX, dtype=int)
        for sp in ts:
            for idx, v in enumerate(sp["label_vec"][:CORRECT_IDX].tolist()):
                if v > 0:
                    gt[idx] = 1
        label_list.append(gt)
        tid_list.append(tid)

    return np.array(logit_list, dtype=np.float32), np.array(label_list, dtype=int), tid_list


def load_baseline_outputs(model, gi, adj, device, label_map) -> dict:
    """Load logits from CDN cache if present, else extract fresh."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, n_nodes = gi["x"].float().to(device), gi["n_nodes"]
    splits = {}
    for split in ("val", "test"):
        cdn = CDN_CACHE_DIR / f"baseline_outputs_{split}.npz"
        cal = OUTPUT_DIR   / f"baseline_outputs_{split}.npz"
        cache = cdn if cdn.exists() else cal

        if cache.exists():
            log.info("Loading baseline outputs (%s) from %s", split, cache.parent.name)
            d = np.load(cache, allow_pickle=True)
            splits[split] = {
                "logits":    d["logits"],
                "y_true":    d["y_true"],
                "trace_ids": d["trace_ids"].tolist(),
            }
        else:
            log.info("Extracting baseline outputs for split=%s ...", split)
            emb_dict = torch.load(DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True)
            spans    = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)
            logits, y_true, tids = _extract_logits(model, spans, x, adj, device)
            np.savez(cal, logits=logits, y_true=y_true,
                     trace_ids=np.array(tids, dtype=object))
            splits[split] = {"logits": logits, "y_true": y_true, "trace_ids": tids}
    return splits


# ---------------------------------------------------------------------------
# Strategy A — per-label F1 optimisation
# ---------------------------------------------------------------------------

def calibrate_per_label(
    X_val: np.ndarray,       # (N_val, 19) baseline logits
    Y_val: np.ndarray,       # (N_val, 19) gold labels
    thr_grid: np.ndarray,    # thresholds to search
    fallback_thr: float,     # used when val support == 0
) -> tuple[np.ndarray, dict]:
    """
    Find τ_i* = argmax_{τ} F1_i(val) for each label independently.

    Returns:
        thresholds : (19,) array of per-label thresholds
        report     : dict with per-label details (support, chosen τ, val F1)
    """
    probs_val = _sigmoid(X_val)   # (N_val, 19)
    thresholds = np.full(N_LABELS, fallback_thr)
    report = {}

    for i in range(N_LABELS):
        support = int(Y_val[:, i].sum())
        if support == 0:
            thresholds[i] = fallback_thr
            report[i] = {"support": 0, "threshold": fallback_thr,
                         "val_f1": 0.0, "note": "zero-support → fallback"}
            continue

        best_f1, best_thr = 0.0, fallback_thr
        for thr in thr_grid:
            y_pred_i = (probs_val[:, i] >= thr).astype(int)
            f1 = float(f1_score(Y_val[:, i], y_pred_i, zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        thresholds[i] = best_thr
        report[i] = {"support": support, "threshold": float(best_thr),
                     "val_f1": round(best_f1, 4)}

    return thresholds, report


# ---------------------------------------------------------------------------
# Strategy B — global greedy coordinate ascent on weighted F1
# ---------------------------------------------------------------------------

def calibrate_global_greedy(
    X_val: np.ndarray,
    Y_val: np.ndarray,
    thr_grid: np.ndarray,
    init_thr: float,
    max_rounds: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """
    Coordinate ascent: for each label i in turn, pick τ_i that maximises
    global weighted F1 on val while holding all other τ_j fixed.
    Repeat until the full threshold vector stops changing (or max_rounds exceeded).

    Zero-support labels on val are pinned to init_thr to prevent the ascent
    from setting them to τ=0.05 (predict-everything) — weighted F1 on val
    ignores those labels since their val weight is 0, but on test they have
    support and the low threshold causes false positives.

    Returns:
        thresholds   : (19,) final per-label thresholds
        f1_history   : weighted F1 after each full round
    """
    probs_val    = _sigmoid(X_val)
    thresholds   = np.full(N_LABELS, init_thr)
    zero_support = (Y_val.sum(axis=0) == 0)   # (19,) bool — pin these labels
    f1_history   = []

    for rnd in range(max_rounds):
        prev = thresholds.copy()

        for i in range(N_LABELS):
            if zero_support[i]:
                thresholds[i] = init_thr   # keep pinned
                continue
            best_f1  = -1.0
            best_thr = thresholds[i]
            for thr in thr_grid:
                thresholds[i] = thr
                y_pred = (probs_val >= thresholds).astype(int)
                f1 = float(f1_score(Y_val, y_pred, average="weighted", zero_division=0))
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            thresholds[i] = best_thr

        round_f1 = float(f1_score(
            Y_val, (probs_val >= thresholds).astype(int),
            average="weighted", zero_division=0,
        ))
        f1_history.append(round_f1)
        changed = not np.allclose(thresholds, prev)
        log.info("  Round %d: weighted F1=%.4f  changed=%s", rnd + 1, round_f1, changed)
        if not changed:
            break

    return thresholds, f1_history


# ---------------------------------------------------------------------------
# Span-level location and joint accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_span_location_metrics(
    model, gi, adj, device, split, label_map,
    pred_trace: np.ndarray, threshold_per_label: np.ndarray, trace_ids: list,
) -> tuple[float, float]:
    """
    Span-level metrics using per-label thresholds.

    A span is flagged if baseline_span_prob[k, i] > threshold_per_label[i]
    for at least one label i.
    """
    model.eval()
    x, n_nodes = gi["x"].float().to(device), gi["n_nodes"]
    Z = model.encode_graph(x, adj)

    emb_dict = torch.load(DATA_DIR / f"span_embeddings_{split}.pt", weights_only=True)
    spans    = build_flat_span_list(DATASET_FILE, split, emb_dict, label_map, n_nodes)

    by_trace: dict[str, list] = defaultdict(list)
    for sp in spans:
        by_trace[sp["trace_id"]].append(sp)

    tid_to_idx = {tid: idx for idx, tid in enumerate(trace_ids)}
    loc_list, joint_list = [], []

    for tid in sorted(by_trace.keys()):
        ts = by_trace[tid]
        if tid not in tid_to_idx:
            continue
        t_idx = tid_to_idx[tid]
        # Use per-label thresholds to gate span probabilities
        cal_mask = pred_trace[t_idx].astype(np.float32)   # (19,) binary CRF-style mask

        embs       = torch.stack([sp["emb"] for sp in ts]).to(device)
        scores     = model.score_spans(embs, Z)
        span_probs = torch.sigmoid(scores).cpu().numpy()[:, :CORRECT_IDX]  # (K, 19)

        # Apply per-label threshold directly to span probs (no trace-level mask)
        span_pred = (span_probs >= threshold_per_label[np.newaxis, :])     # (K, 19) bool
        pred_error = span_pred.any(axis=1)                                 # (K,) bool

        gt_span_ids   = {sp["span_id"] for sp in ts if not sp["is_correct"]}
        pred_span_ids = {sp["span_id"] for sp, e in zip(ts, pred_error) if e}

        loc = (
            len(gt_span_ids & pred_span_ids) / len(gt_span_ids)
            if gt_span_ids else (1.0 if not pred_span_ids else 0.0)
        )
        loc_list.append(loc)

        gt_pairs, pred_pairs = set(), set()
        for k, sp in enumerate(ts):
            sid = sp["span_id"]
            if not sp["is_correct"]:
                for cat in sp["labels"]:
                    gt_pairs.add((sid, cat))
            for cat, cat_idx in label_map.items():
                if cat_idx < CORRECT_IDX and span_probs[k, cat_idx] >= threshold_per_label[cat_idx]:
                    pred_pairs.add((sid, cat))

        joint = (
            len(gt_pairs & pred_pairs) / len(gt_pairs)
            if gt_pairs else (1.0 if not pred_span_ids else 0.0)
        )
        joint_list.append(joint)

    return float(np.mean(loc_list)), float(np.mean(joint_list))


# ---------------------------------------------------------------------------
# Evaluate one threshold vector
# ---------------------------------------------------------------------------

def evaluate_thresholds(
    name: str,
    thresholds: np.ndarray,            # (19,)
    X_val: np.ndarray, Y_val: np.ndarray, tids_val: list,
    X_test: np.ndarray, Y_test: np.ndarray, tids_test: list,
    error_names: list,
    model, gi, adj, device, label_map,
    global_baseline_thr: float,
) -> dict:
    """Run metrics for both val and test; print test table; return results dict."""
    probs_val  = _sigmoid(X_val)
    probs_test = _sigmoid(X_test)

    y_pred_val  = (probs_val  >= thresholds).astype(int)
    y_pred_test = (probs_test >= thresholds).astype(int)

    # Baseline at global threshold for delta comparison
    y_base_test  = (_sigmoid(X_test) >= global_baseline_thr).astype(int)
    base_metrics = compute_metrics(Y_test, y_base_test, error_names)

    val_metrics  = compute_metrics(Y_val,  y_pred_val,  error_names)
    test_metrics = compute_metrics(Y_test, y_pred_test, error_names)
    delta_f1     = test_metrics["f1_weighted"] - base_metrics["f1_weighted"]

    loc_acc, joint_acc = compute_span_location_metrics(
        model, gi, adj, device, "test", label_map,
        pred_trace=y_pred_test,
        threshold_per_label=thresholds,
        trace_ids=tids_test,
    )

    print(f"\n{'='*65}")
    print(f"Per-label calibration ({name}) — test set")
    print(f"{'='*65}")
    print(classification_report(Y_test, y_pred_test, target_names=error_names, zero_division=0))
    print_metrics(test_metrics, header=f"--- {name} ---")
    print_metrics(
        base_metrics,
        header=f"\n--- Baseline (global thr={global_baseline_thr:.2f}) ---",
    )
    print(f"\nΔ F1 weighted vs baseline: {delta_f1:+.4f}")
    print(f"Location Accuracy:        {loc_acc:.4f}")
    print(f"Joint Accuracy:           {joint_acc:.4f}")

    return {
        "strategy":             name,
        "val_f1_weighted":      val_metrics["f1_weighted"],
        "test_f1_weighted":     test_metrics["f1_weighted"],
        "test_f1_macro":        test_metrics["f1_macro"],
        "test_f1_micro":        test_metrics["f1_micro"],
        "test_precision":       test_metrics["precision"],
        "test_recall":          test_metrics["recall"],
        "location_accuracy":    loc_acc,
        "joint_accuracy":       joint_acc,
        "delta_f1_weighted":    delta_f1,
        "thresholds":           thresholds.tolist(),
        "per_class":            test_metrics["per_class"],
        "baseline_f1_weighted": base_metrics["f1_weighted"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-label threshold calibration for TRAIL error classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--strategy", default="both", choices=["A", "B", "both"],
        help="A=per-label F1, B=global greedy coord ascent, both=run both",
    )
    ap.add_argument(
        "--thr_step", type=float, default=0.05,
        help="Threshold grid step size.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split_tag", default="",
                    help="Tag suffix for data/model/output dirs (e.g. '712' → data_712/, "
                         "models_712/, outputs_712/). Empty = default dirs.")
    args = ap.parse_args()

    # Apply split_tag to module-level path variables
    if args.split_tag:
        tag = f"_{args.split_tag}"
        global DATA_DIR, OUTPUT_DIR, CDN_CACHE_DIR
        global DATASET_FILE, GRAPH_INPUT, LABEL_MAP_FILE
        DATA_DIR       = GRAPH_DIR / f"data{tag}"
        OUTPUT_DIR     = CAL_DIR / f"outputs{tag}"
        CDN_CACHE_DIR  = POST_DIR / "CDN" / f"outputs{tag}"
        DATASET_FILE   = DATA_DIR / "span_dataset.jsonl"
        GRAPH_INPUT    = DATA_DIR / "graph_input.pt"
        LABEL_MAP_FILE = DATA_DIR / "label_to_node_idx.json"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph + baseline model
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
    # Load logits
    # ------------------------------------------------------------------
    splits   = load_baseline_outputs(model, gi, adj, device, label_map)
    X_val    = splits["val"]["logits"]
    Y_val    = splits["val"]["y_true"]
    tids_val = splits["val"]["trace_ids"]
    X_test   = splits["test"]["logits"]
    Y_test   = splits["test"]["y_true"]
    tids_test = splits["test"]["trace_ids"]

    # ------------------------------------------------------------------
    # Global best threshold on val (baseline to beat)
    # ------------------------------------------------------------------
    thr_grid = np.arange(0.05, 0.96, args.thr_step)
    probs_val = _sigmoid(X_val)

    global_best_f1, global_best_thr = 0.0, 0.5
    for thr in thr_grid:
        f1 = float(f1_score(Y_val, (probs_val >= thr).astype(int),
                            average="weighted", zero_division=0))
        if f1 > global_best_f1:
            global_best_f1, global_best_thr = f1, thr

    log.info(
        "Global threshold baseline: thr=%.2f → val_F1=%.4f",
        global_best_thr, global_best_f1,
    )

    # ------------------------------------------------------------------
    # Print val label support and max baseline probability
    # ------------------------------------------------------------------
    print("\nVal label statistics:")
    print(f"  {'Label':<40} {'support':>8}  {'max_p':>7}  {'zero?':>6}")
    for i, name in enumerate(error_names):
        sup = int(Y_val[:, i].sum())
        mx  = float(probs_val[:, i].max())
        print(f"  {name:<40} {sup:>8}  {mx:>7.3f}  {'✗' if sup == 0 else '':>6}")

    # ------------------------------------------------------------------
    # Run selected strategies
    # ------------------------------------------------------------------
    run_A = args.strategy in ("A", "both")
    run_B = args.strategy in ("B", "both")
    all_results = {}

    if run_A:
        log.info("\nStrategy A: per-label F1 optimisation on val ...")
        thr_A, report_A = calibrate_per_label(
            X_val, Y_val, thr_grid, fallback_thr=global_best_thr,
        )
        print("\nStrategy A — per-label thresholds:")
        print(f"  {'Label':<40} {'support':>8}  {'τ_i':>6}  {'val_F1':>8}  {'note'}")
        for i, name in enumerate(error_names):
            r = report_A[i]
            print(f"  {name:<40} {r['support']:>8}  {r['threshold']:>6.2f}  "
                  f"{r['val_f1']:>8.4f}  {r.get('note','')}")

        res_A = evaluate_thresholds(
            "Strategy A (per-label F1)",
            thr_A, X_val, Y_val, tids_val, X_test, Y_test, tids_test,
            error_names, model, gi, adj, device, label_map, global_best_thr,
        )
        out = OUTPUT_DIR / "eval_results_strategy_A.json"
        out.write_text(json.dumps(res_A, indent=2))
        log.info("Saved → %s", out.name)
        all_results["A"] = res_A

    if run_B:
        log.info("\nStrategy B: global greedy coordinate ascent on val ...")
        thr_B, f1_hist = calibrate_global_greedy(
            X_val, Y_val, thr_grid, init_thr=global_best_thr,
        )
        print(f"\nStrategy B — coordinate ascent converged in {len(f1_hist)} rounds")
        print(f"  F1 history: {[round(f, 4) for f in f1_hist]}")
        print("\nStrategy B — per-label thresholds:")
        print(f"  {'Label':<40} {'τ_i':>6}  {'changed from global?':>22}")
        for i, name in enumerate(error_names):
            changed = "yes" if abs(thr_B[i] - global_best_thr) > 1e-6 else "—"
            print(f"  {name:<40} {thr_B[i]:>6.2f}  {changed:>22}")

        res_B = evaluate_thresholds(
            "Strategy B (global greedy)",
            thr_B, X_val, Y_val, tids_val, X_test, Y_test, tids_test,
            error_names, model, gi, adj, device, label_map, global_best_thr,
        )
        out = OUTPUT_DIR / "eval_results_strategy_B.json"
        out.write_text(json.dumps(res_B, indent=2))
        log.info("Saved → %s", out.name)
        all_results["B"] = res_B

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("Summary — test set")
    print(f"{'='*75}")
    print(f"  {'Method':<35} {'W-F1':>7} {'Ma-F1':>7} {'Mi-F1':>7} {'Prec':>7} {'Rec':>7}  {'Δ':>7}")
    print(f"  {'-'*73}")

    # Baseline row
    y_base = (_sigmoid(X_test) >= global_best_thr).astype(int)
    bm     = compute_metrics(Y_test, y_base, error_names)
    print(f"  {'Baseline (global thr='+str(round(global_best_thr,2))+')':<35} "
          f"{bm['f1_weighted']:>7.4f} {bm['f1_macro']:>7.4f} {bm['f1_micro']:>7.4f} "
          f"{bm['precision']:>7.4f} {bm['recall']:>7.4f}  {'0.0000':>7}")

    for key, res in all_results.items():
        print(
            f"  {res['strategy']:<35} "
            f"{res['test_f1_weighted']:>7.4f} {res['test_f1_macro']:>7.4f} "
            f"{res['test_f1_micro']:>7.4f} {res['test_precision']:>7.4f} "
            f"{res['test_recall']:>7.4f}  {res['delta_f1_weighted']:>+7.4f}"
        )


if __name__ == "__main__":
    main()
