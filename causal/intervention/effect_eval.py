#!/usr/bin/env python3
"""
Causal-effect evaluation (B1–B3, C).

Since we cannot re-execute the agent environment, this module uses an
annotation-based proxy to estimate prevention effects:

  Δ(A→B) = E[1(B present downstream of A in baseline)
              − 1(B present downstream of A in patched)]

Because we cannot rerun, the "patched" proxy is:
  • Assume the patch eliminates error A at its location.
  • Count how often B appeared AFTER A in the baseline trace.
  • This gives the UPPER BOUND on the prevention effect
    (= "if A is removed, B can no longer be triggered by A").

Additionally:
  • Presence-drop matrix  : P(B follows A) per (A, B) pair
  • First-occurrence shift: E[T_B^patched − T_B^baseline] (span-index difference)
  • Edge validation       : edge A→B is "validated" if Δ(A→B) > threshold
  • Placebo comparison    : compare against a family-shuffled null distribution

Outputs (written to out_dir/):
  effect_edges.json       – full result dict
  (also prints summary to stdout)

Usage (from benchmarking/):
    python causal/intervention/effect_eval.py \\
        --patch_log       outputs/interventions/patch_log.jsonl \\
        --annotations_dir processed_annotations_gaia \\
        --stage1_edges    causal/graph/...   # optional \\
        --out             outputs/interventions/effect_edges.json
"""
from __future__ import annotations

import os
import sys
_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarking"))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> List[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_rerun_to_patch_record(rec: dict) -> dict:
    """Convert rerun_log.jsonl line to the shape effect_eval expects (patch_log shape)."""
    error_id = rec.get("error_id", "")
    # error_id format: trace_id|span_id|category|index
    parts = error_id.split("|")
    error_type = rec.get("error_type", "") or (parts[2] if len(parts) > 2 else "")
    return {
        "trace_id": rec.get("trace_id", ""),
        "error_id": error_id,
        "operator_family": rec.get("operator_family", ""),
        "success": rec.get("success", False),
        "instantiated_spec": {"error_type": error_type},
        "location": rec.get("location", ""),
        "output_path": rec.get("output_path", ""),
    }


def _load_annotations(ann_dir: str) -> Dict[str, List[dict]]:
    """
    Load all per-trace annotation JSON files → {trace_id: [error_record, ...]}.
    Each error record: {span_id, category, span_order_index, impact, evidence}.
    """
    result: Dict[str, List[dict]] = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        tid = os.path.splitext(fname)[0]
        try:
            with open(os.path.join(ann_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        errors = []
        for i, e in enumerate(data.get("errors") or []):
            errors.append({
                "span_id":   e.get("location") or "",
                "category":  (e.get("category") or "").strip(),
                "index":     i,        # order of appearance = proxy for temporal order
                "impact":    e.get("impact", ""),
                "evidence":  e.get("evidence", ""),
            })
        result[tid] = errors
    return result


def _load_stage1_edges(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Load a causal graph file (AIC/BIC) if provided.
    Expected format: {A: [B1, B2, ...]} or {edges: [{source, target}, ...]}.
    Returns {source_type: [target_type, ...]}.
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "edges" in raw:
        result: Dict[str, List[str]] = defaultdict(list)
        for edge in raw["edges"]:
            s, t = edge.get("source", ""), edge.get("target", "")
            if s and t:
                result[s].append(t)
        return dict(result)
    if isinstance(raw, dict):
        return {k: list(v) for k, v in raw.items() if isinstance(v, list)}
    return {}


# ---------------------------------------------------------------------------
# Core metrics (B3)
# ---------------------------------------------------------------------------


def _get_categories_with_index(errors: List[dict]) -> List[Tuple[str, int]]:
    """Return (category, index) pairs sorted by appearance order."""
    return sorted([(e["category"], e["index"]) for e in errors], key=lambda x: x[1])


def _presence_drop(
    baseline_errors: List[Tuple[str, int]],
    patched_type: str,
) -> Dict[str, int]:
    """
    For each error type B ≠ A:
    Return 1 if B appears AFTER the first occurrence of A in baseline
    (this is what a successful A patch could prevent), else 0.
    """
    result: Dict[str, int] = {}
    a_indices = [idx for cat, idx in baseline_errors if cat == patched_type]
    if not a_indices:
        return result
    first_a = min(a_indices)
    for cat, idx in baseline_errors:
        if cat != patched_type and idx > first_a:
            result[cat] = 1
    return result


def _timing_shift(
    baseline_errors: List[Tuple[str, int]],
    patched_type: str,
) -> Dict[str, float]:
    """
    For each downstream B, return the expected shift in first occurrence:
    Δt(A→B) = first_occurrence(B) − first_occurrence(A)
    Positive = B occurs later (good; suggests A causes B).
    """
    result: Dict[str, float] = {}
    a_indices = [idx for cat, idx in baseline_errors if cat == patched_type]
    if not a_indices:
        return result
    first_a = min(a_indices)
    # For each B appearing after A, compute the gap
    seen: Dict[str, int] = {}
    for cat, idx in baseline_errors:
        if cat != patched_type and idx > first_a and cat not in seen:
            seen[cat] = idx
    for cat, b_idx in seen.items():
        result[cat] = b_idx - first_a
    return result


# ---------------------------------------------------------------------------
# Placebo comparison
# ---------------------------------------------------------------------------


def _placebo_family(family: str, all_families: List[str]) -> Optional[str]:
    """Return a different family to use as placebo (null intervention)."""
    others = [f for f in all_families if f != family]
    return others[0] if others else None


# ---------------------------------------------------------------------------
# Main compute_effects
# ---------------------------------------------------------------------------


def compute_effects(
    patch_log_path: str,
    annotations_dir: str,
    out_path: str,
    stage1_edges_path: Optional[str] = None,
    validation_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Compute Δ(A→B) prevention-effect scores and edge-validation indicators.

    Returns a result dict and writes it to out_path.
    """
    # Load patch log (or rerun_log from single-intervention runs)
    patch_log = _load_jsonl(patch_log_path)
    if patch_log and os.path.basename(patch_log_path or "").startswith("rerun_log"):
        patch_log = [_normalize_rerun_to_patch_record(r) for r in patch_log]
        print(f"[effect_eval] Normalized rerun log: {len(patch_log)} records")
    elif not patch_log and patch_log_path:
        rerun_path = os.path.join(os.path.dirname(patch_log_path), "rerun_log.jsonl")
        if os.path.isfile(rerun_path):
            raw = _load_jsonl(rerun_path)
            patch_log = [_normalize_rerun_to_patch_record(r) for r in raw]
            print(f"[effect_eval] Using rerun log: {rerun_path} ({len(patch_log)} records)")
    if not patch_log:
        print(f"[effect_eval] Patch/rerun log not found or empty: {patch_log_path}")
        return {}
    baseline_ann = _load_annotations(annotations_dir)
    stage1_edges = _load_stage1_edges(stage1_edges_path)

    print(
        f"[effect_eval] {len(patch_log)} patch records | "
        f"{len(baseline_ann)} annotated traces"
    )

    # ----------------------------------------------------------------
    # Aggregate per (patched_type, downstream_B): cumulative Δ stats
    # ----------------------------------------------------------------

    # effect_counts[A][B] = number of traces where B appears downstream of A
    effect_counts:    Dict[str, Dict[str, int]]   = defaultdict(lambda: defaultdict(int))
    timing_sums:      Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    timing_counts:    Dict[str, Dict[str, int]]   = defaultdict(lambda: defaultdict(int))
    # coverage: how many patches targeted each error type A
    coverage:         Dict[str, int] = defaultdict(int)
    success_count:    Dict[str, int] = defaultdict(int)
    family_per_type:  Dict[str, str] = {}

    for rec in patch_log:
        if not rec.get("success"):
            continue

        tid    = rec["trace_id"]
        family = rec.get("operator_family", "")
        inst   = rec.get("instantiated_spec") or {}
        patched_type = inst.get("error_type", "")
        if not patched_type:
            parts = rec.get("error_id", "").split("|")
            patched_type = parts[2] if len(parts) > 2 else ""
        if not patched_type:
            continue

        coverage[patched_type] += 1
        success_count[patched_type] += 1
        family_per_type[patched_type] = family

        baseline = baseline_ann.get(tid, [])
        cats_with_idx = _get_categories_with_index(baseline)

        # presence drop
        drop = _presence_drop(cats_with_idx, patched_type)
        for b_type, present in drop.items():
            effect_counts[patched_type][b_type] += present

        # timing shift
        shift = _timing_shift(cats_with_idx, patched_type)
        for b_type, delta_t in shift.items():
            timing_sums[patched_type][b_type]   += delta_t
            timing_counts[patched_type][b_type] += 1

    # ----------------------------------------------------------------
    # Normalise to Δ scores
    # ----------------------------------------------------------------

    presence_drop_matrix:  Dict[str, Dict[str, float]] = {}
    timing_shift_matrix:   Dict[str, Dict[str, float]] = {}
    edge_validation:       Dict[str, Dict[str, Any]]   = {}

    all_a_types = list(effect_counts.keys())
    for a_type in all_a_types:
        n = max(coverage[a_type], 1)
        presence_drop_matrix[a_type] = {
            b: cnt / n
            for b, cnt in effect_counts[a_type].items()
            if cnt > 0
        }
        timing_shift_matrix[a_type] = {
            b: timing_sums[a_type][b] / timing_counts[a_type][b]
            for b in timing_counts[a_type]
        }
        edge_validation[a_type] = {}
        for b, cnt in effect_counts[a_type].items():
            delta = cnt / n
            # Check if this edge also appears in the stage-1 causal graph
            in_stage1 = b in stage1_edges.get(a_type, [])
            edge_validation[a_type][b] = {
                "delta":            round(delta, 4),
                "n_downstream":     cnt,
                "n_patches":        n,
                "avg_timing_shift": round(
                    timing_sums[a_type].get(b, 0.0) / max(timing_counts[a_type].get(b, 1), 1),
                    2,
                ),
                "in_stage1_graph":  in_stage1,
                "validated":        delta >= validation_threshold,
            }

    # ----------------------------------------------------------------
    # Top-K effects per A type
    # ----------------------------------------------------------------

    top_effects: Dict[str, List[Any]] = {}
    for a_type, b_dict in presence_drop_matrix.items():
        top_effects[a_type] = sorted(b_dict.items(), key=lambda x: -x[1])[:5]

    # ----------------------------------------------------------------
    # Stage-1 graph reweighting
    # ----------------------------------------------------------------

    reweighted_edges: Dict[str, Any] = {}
    for a_type, targets in stage1_edges.items():
        reweighted_edges[a_type] = {}
        for b in targets:
            delta = presence_drop_matrix.get(a_type, {}).get(b, 0.0)
            reweighted_edges[a_type][b] = {
                "original_edge": True,
                "delta": round(delta, 4),
                "validated": delta >= validation_threshold,
            }

    # ----------------------------------------------------------------
    # Summary stats
    # ----------------------------------------------------------------

    total_patches  = len(patch_log)
    total_success  = sum(1 for p in patch_log if p.get("success"))

    result: Dict[str, Any] = {
        "total_patches":        total_patches,
        "total_successful":     total_success,
        "patch_success_rate":   round(total_success / max(total_patches, 1), 4),
        "coverage_by_error_type": dict(coverage),
        "family_per_error_type":  family_per_type,
        "presence_drop_matrix":  presence_drop_matrix,
        "timing_shift_matrix":   timing_shift_matrix,
        "edge_validation":       edge_validation,
        "top_effects":           top_effects,
        "reweighted_stage1_edges": reweighted_edges,
        "validation_threshold":  validation_threshold,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n=== Effect Evaluation ===")
    print(f"  Patches total    : {total_patches}")
    print(f"  Patches succeeded: {total_success}  ({result['patch_success_rate']:.1%})")
    print("\nTop prevention effects Δ(A→B):")
    for a_type, effects in sorted(top_effects.items()):
        if not effects:
            continue
        print(f"  {a_type}:")
        for b, delta in effects[:3]:
            timing = timing_shift_matrix.get(a_type, {}).get(b, None)
            t_str = f"  avg_gap={timing:.1f} turns" if timing is not None else ""
            validated = edge_validation.get(a_type, {}).get(b, {}).get("validated", False)
            mark = "[✓]" if validated else "[ ]"
            print(f"    {mark} → {b}: Δ={delta:.3f}{t_str}")

    if reweighted_edges:
        print("\nStage-1 edge reweighting:")
        for a_type, b_dict in reweighted_edges.items():
            for b, info in b_dict.items():
                mark = "[✓]" if info["validated"] else "[ ]"
                print(f"  {mark} {a_type} → {b}: Δ={info['delta']:.3f}")

    print(f"\nWrote: {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Δ(A→B) prevention-effect scores from patch log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patch_log", default="outputs/interventions/patch_log.jsonl",
        help="Path to patch_log.jsonl produced by intervene.py.",
    )
    parser.add_argument(
        "--annotations_dir", default="processed_annotations_gaia",
        help="Directory with per-trace annotation JSON files.",
    )
    parser.add_argument(
        "--stage1_edges", default=None,
        help="Optional: path to AIC/BIC causal graph JSON for edge reweighting.",
    )
    parser.add_argument(
        "--out", default="outputs/interventions/effect_edges.json",
        help="Output path for the effect edges JSON.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Δ threshold above which an edge is considered 'validated'.",
    )
    args = parser.parse_args()

    compute_effects(
        patch_log_path      = args.patch_log,
        annotations_dir     = args.annotations_dir,
        out_path            = args.out,
        stage1_edges_path   = args.stage1_edges,
        validation_threshold = args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
