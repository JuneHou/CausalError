#!/usr/bin/env python3
"""
Split TRAIL traces into train / val / test by trace_id.

Rules:
  - Never split by span; always by trace.
  - Every error type must be covered in train.
  - Error types with ≤ 2 supporting traces are force-locked into train.
  - Remaining traces: 80 / 10 / 10 stratified shuffle.
  - After split: repair any coverage gaps by pulling one trace
    per missing error type from val/test into train.
  - Strict disjointness assertions.

Outputs (written to --out_dir):
  train_trace_ids.json    list[str]
  val_trace_ids.json
  test_trace_ids.json
  split_stats.json        coverage / count summary

Usage (from benchmarking/):
    python graph/splits/make_trail_split.py \
        --gaia_dir   processed_annotations_gaia \
        --swe_dir    processed_annotations_swe_bench \
        --out_dir    graph/splits/outputs \
        --seed       42
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_annotations(gaia_dir: str, swe_dir: str) -> Dict[str, List[str]]:
    """
    Returns trace_to_errors: {trace_id: [category, ...]} in annotation order.
    Annotations order = temporal order in the trace (earliest error first).
    GAIA trace_id = filename stem; SWE bench trace_id = 'trace_id' field.
    """
    trace_to_errors: Dict[str, List[str]] = {}

    for ann_dir, use_filename_as_id in [(gaia_dir, True), (swe_dir, False)]:
        files = sorted(f for f in os.listdir(ann_dir)
                       if f.endswith(".json") and not f.startswith("old_"))
        for fname in files:
            path = os.path.join(ann_dir, fname)
            with open(path, encoding="utf-8") as f:
                ann = json.load(f)
            if use_filename_as_id:
                tid = os.path.splitext(fname)[0]
            else:
                tid = ann.get("trace_id", os.path.splitext(fname)[0])
            errors = [e["category"].strip() for e in ann.get("errors", [])]
            if errors:
                trace_to_errors[tid] = errors

    return trace_to_errors


def derive_root(errors: List[str]) -> str:
    """
    Root error = first error in annotation order (earliest temporal occurrence).
    Override this function if you have explicit root labels.
    """
    return errors[0]


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def make_split(
    trace_to_errors: Dict[str, List[str]],
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    rare_threshold: int = 2,
) -> tuple[List[str], List[str], List[str]]:

    all_traces = sorted(trace_to_errors.keys())  # sorted for reproducibility

    # 1. Build coverage maps
    error_to_traces: Dict[str, List[str]] = defaultdict(list)
    for tid, errors in trace_to_errors.items():
        for cat in set(errors):  # unique per trace
            error_to_traces[cat].append(tid)

    all_error_types = set(error_to_traces.keys())

    # 2. Force-lock rare classes into train
    locked_train: Set[str] = set()
    for err, tids in error_to_traces.items():
        if len(tids) <= rare_threshold:
            locked_train.update(tids)

    # 3. Split remaining traces 80/10/10
    remaining = sorted(set(all_traces) - locked_train)
    rng = random.Random(seed)
    rng.shuffle(remaining)

    n = len(remaining)
    n_val  = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    n_train = n - n_val - n_test

    split_train = set(remaining[:n_train])
    split_val   = set(remaining[n_train:n_train + n_val])
    split_test  = set(remaining[n_train + n_val:])

    # 4. Add locked traces back into train
    train_ids = split_train | locked_train
    val_ids   = split_val
    test_ids  = split_test

    # 5. Repair coverage: ensure every error type is in train
    def covered_by(id_set: Set[str]) -> Set[str]:
        cats: Set[str] = set()
        for tid in id_set:
            cats.update(trace_to_errors[tid])
        return cats

    for err in sorted(all_error_types):
        if err not in covered_by(train_ids):
            # Pull one supporting trace from val or test
            candidates = sorted(error_to_traces[err])
            moved = False
            for pool, name in [(val_ids, "val"), (test_ids, "test")]:
                for tid in candidates:
                    if tid in pool:
                        pool.discard(tid)
                        train_ids.add(tid)
                        print(f"  [repair] moved {tid[:12]}... from {name} → train  (missing: {err})")
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                print(f"  [warn] cannot repair coverage for '{err}' — no supporting trace in val/test")

    # 6. Assert strict disjointness
    assert train_ids.isdisjoint(val_ids),  "train ∩ val non-empty!"
    assert train_ids.isdisjoint(test_ids), "train ∩ test non-empty!"
    assert val_ids.isdisjoint(test_ids),   "val ∩ test non-empty!"

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def split_stats(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    trace_to_errors: Dict[str, List[str]],
) -> dict:
    def coverage(ids: List[str]) -> Set[str]:
        cats: Set[str] = set()
        for tid in ids:
            cats.update(trace_to_errors.get(tid, []))
        return cats

    def root_dist(ids: List[str]) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for tid in ids:
            errs = trace_to_errors.get(tid, [])
            if errs:
                d[derive_root(errs)] += 1
        return dict(d)

    all_types = sorted(coverage(train_ids) | coverage(val_ids) | coverage(test_ids))

    per_type: Dict[str, dict] = {}
    error_to_traces: Dict[str, List[str]] = defaultdict(list)
    for tid, errors in trace_to_errors.items():
        for cat in set(errors):
            error_to_traces[cat].append(tid)

    for cat in all_types:
        tids = set(error_to_traces[cat])
        per_type[cat] = {
            "total_traces": len(tids),
            "train": len(tids & set(train_ids)),
            "val":   len(tids & set(val_ids)),
            "test":  len(tids & set(test_ids)),
        }

    return {
        "n_train": len(train_ids),
        "n_val":   len(val_ids),
        "n_test":  len(test_ids),
        "n_total": len(train_ids) + len(val_ids) + len(test_ids),
        "train_coverage": sorted(coverage(train_ids)),
        "val_coverage":   sorted(coverage(val_ids)),
        "test_coverage":  sorted(coverage(test_ids)),
        "per_error_type": per_type,
        "root_dist_train": root_dist(train_ids),
        "root_dist_val":   root_dist(val_ids),
        "root_dist_test":  root_dist(test_ids),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split TRAIL traces into train/val/test by trace_id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gaia_dir", default="processed_annotations_gaia")
    ap.add_argument("--swe_dir",  default="processed_annotations_swe_bench")
    ap.add_argument("--out_dir",  default="graph/splits/outputs")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac",type=float, default=0.10)
    ap.add_argument("--rare_threshold", type=int, default=2,
                    help="Error types with ≤ this many traces are force-locked into train.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading annotations...")
    trace_to_errors = load_annotations(args.gaia_dir, args.swe_dir)
    print(f"  {len(trace_to_errors)} traces, "
          f"{len(set(c for v in trace_to_errors.values() for c in v))} unique error types")

    print(f"\nSplitting (seed={args.seed}, val={args.val_frac:.0%}, test={args.test_frac:.0%})...")
    train_ids, val_ids, test_ids = make_split(
        trace_to_errors,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        rare_threshold=args.rare_threshold,
    )

    # Save splits
    def save(ids: List[str], name: str) -> None:
        path = os.path.join(args.out_dir, f"{name}_trace_ids.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        print(f"  Saved {name}: {len(ids)} traces → {path}")

    save(train_ids, "train")
    save(val_ids,   "val")
    save(test_ids,  "test")

    # Also save the full trace_to_errors and root labels for downstream use
    root_labels = {tid: derive_root(errs) for tid, errs in trace_to_errors.items()}
    root_path = os.path.join(args.out_dir, "root_labels.json")
    with open(root_path, "w", encoding="utf-8") as f:
        json.dump(root_labels, f, indent=2)
    print(f"  Saved root labels → {root_path}")

    # Stats
    stats = split_stats(train_ids, val_ids, test_ids, trace_to_errors)
    stats_path = os.path.join(args.out_dir, "split_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nSplit summary:")
    print(f"  train: {stats['n_train']}  val: {stats['n_val']}  test: {stats['n_test']}")
    print(f"  train coverage: {len(stats['train_coverage'])}/19 error types")
    print(f"  val   coverage: {len(stats['val_coverage'])}/19 error types")
    print(f"  test  coverage: {len(stats['test_coverage'])}/19 error types")
    print(f"\nPer error type (train/val/test):")
    for cat, s in sorted(stats["per_error_type"].items()):
        rare = " [LOCKED]" if s["total_traces"] <= args.rare_threshold else ""
        print(f"  {cat:<40} total={s['total_traces']:3d}  "
              f"train={s['train']:3d}  val={s['val']:2d}  test={s['test']:2d}{rare}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
