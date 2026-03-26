#!/usr/bin/env python3
"""
01_make_splits.py — Split traces into train / val / test.

Ratio: 80 / 10 / 10  (trace-level)
Rules:
  1. Every error type must appear at least once in train.
  2. Types with ≤ 2 traces → ALL those traces are forced to train.
  3. Types with 3–5 traces → at least 1 trace per type is anchored to train
     before the random assignment of remaining traces.
  4. Remaining traces are split randomly to reach the ~80/10/10 target.

Input:
  processed_annotations_gaia/     *.json   (excluding old_* files)
  processed_annotations_swe_bench/*.json   (excluding old_* files)

Output (in graph/splits/):
  train_trace_ids.json
  val_trace_ids.json
  test_trace_ids.json
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

ANNOTATION_DIRS = [
    "processed_annotations_gaia",
    "processed_annotations_swe_bench",
]

# Thresholds for forcing traces into train
FORCE_ALL_THRESHOLD = 2    # types with ≤ this many traces: all go to train
ANCHOR_THRESHOLD = 5       # types with ≤ this many traces: at least 1 anchored to train

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test gets the remainder

SEED = 42

OUTPUT_DIR = Path("graph/splits")


# ---------------------------------------------------------------------------
# Load annotations
# ---------------------------------------------------------------------------

def load_annotations(base_dir: str) -> dict[str, set[str]]:
    """Return {trace_id: set_of_error_categories} for all non-old annotation files."""
    trace_to_errors: dict[str, set[str]] = {}
    for ann_dir in ANNOTATION_DIRS:
        path = Path(base_dir) / ann_dir
        if not path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {path}")
        for fpath in sorted(path.glob("*.json")):
            if fpath.stem.startswith("old_"):
                continue
            data = json.loads(fpath.read_text())
            # Some files may be missing trace_id; fall back to filename stem
            trace_id = data.get("trace_id") or fpath.stem
            cats = {
                e["category"]
                for e in data.get("errors", [])
                if e.get("category")
            }
            trace_to_errors[trace_id] = cats
    return trace_to_errors


# ---------------------------------------------------------------------------
# Build error-type → trace-list index
# ---------------------------------------------------------------------------

def build_type_index(trace_to_errors: dict[str, set[str]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for trace_id, cats in trace_to_errors.items():
        for c in cats:
            index[c].append(trace_id)
    return dict(index)


# ---------------------------------------------------------------------------
# Greedy anchor selection
# ---------------------------------------------------------------------------

def select_anchors(
    type_to_traces: dict[str, list[str]],
    trace_to_errors: dict[str, set[str]],
    force_all_threshold: int,
    anchor_threshold: int,
) -> set[str]:
    """
    Return the set of trace IDs that must be placed in train.

    Strategy:
    - Types with count <= force_all_threshold: every trace is forced.
    - Types with count <= anchor_threshold (and > force_all_threshold):
        greedily pick the trace that covers the most un-covered types first,
        until the type has at least 1 forced trace.
    """
    forced: set[str] = set()

    # Phase 1 — force ALL traces for tiny types
    for et, traces in type_to_traces.items():
        if len(traces) <= force_all_threshold:
            for t in traces:
                forced.add(t)
            print(f"  [force-all]  {et}: {len(traces)} traces → all forced to train")

    # Phase 2 — anchor at least 1 trace for small types
    # Sort types by rarity (fewest traces first) so we assign anchors to the
    # rarest types first.
    small_types = [
        et for et, traces in type_to_traces.items()
        if force_all_threshold < len(traces) <= anchor_threshold
    ]
    small_types.sort(key=lambda et: len(type_to_traces[et]))

    for et in small_types:
        traces = type_to_traces[et]
        # Check if this type is already covered by a forced trace
        if any(t in forced for t in traces):
            print(f"  [anchor-ok]  {et}: {len(traces)} traces → already covered by forced set")
            continue
        # Greedy: pick the trace that has the most error types (maximises coverage)
        best = max(traces, key=lambda t: len(trace_to_errors[t]))
        forced.add(best)
        print(f"  [anchor]     {et}: {len(traces)} traces → anchored {best[:12]}...")

    return forced


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def make_splits(base_dir: str = ".") -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    trace_to_errors = load_annotations(base_dir)
    type_to_traces = build_type_index(trace_to_errors)

    all_trace_ids = sorted(trace_to_errors.keys())
    n_total = len(all_trace_ids)
    n_train_target = round(n_total * TRAIN_RATIO)
    n_val_target = round(n_total * VAL_RATIO)
    n_test_target = n_total - n_train_target - n_val_target

    print(f"\nTotal traces: {n_total}")
    print(f"Target split: train={n_train_target}, val={n_val_target}, test={n_test_target}")
    print(f"\nApplying stratification rules (seed={SEED})...")

    # Determine forced-train traces
    forced_train = select_anchors(
        type_to_traces, trace_to_errors,
        FORCE_ALL_THRESHOLD, ANCHOR_THRESHOLD
    )
    print(f"\nForced-train set: {len(forced_train)} traces")

    # Remaining traces to be randomly split
    remaining = [t for t in all_trace_ids if t not in forced_train]
    random.shuffle(remaining)

    # How many more traces do we need in train beyond the forced set?
    n_more_train = max(0, n_train_target - len(forced_train))
    # Val and test share the leftover
    n_remaining_val = n_val_target
    n_remaining_test = len(remaining) - n_more_train - n_remaining_val

    if n_remaining_test < 0:
        # Forced set is larger than target train; take fewer from remaining for train
        n_more_train = max(0, len(remaining) - n_val_target - n_val_target)
        n_remaining_val = (len(remaining) - n_more_train) // 2
        n_remaining_test = len(remaining) - n_more_train - n_remaining_val

    train_ids = sorted(forced_train) + remaining[:n_more_train]
    val_ids = remaining[n_more_train: n_more_train + n_remaining_val]
    test_ids = remaining[n_more_train + n_remaining_val:]

    # Verify: every error type must be represented in train
    train_coverage = set()
    for t in train_ids:
        train_coverage |= trace_to_errors[t]

    missing = set(type_to_traces.keys()) - train_coverage
    if missing:
        # Rescue: move one val/test trace per missing type into train
        for et in sorted(missing):
            rescued = False
            for candidate_pool in [val_ids, test_ids]:
                for t in list(candidate_pool):
                    if et in trace_to_errors[t]:
                        candidate_pool.remove(t)
                        train_ids.append(t)
                        train_coverage |= trace_to_errors[t]
                        print(f"  [rescue] {et}: moved {t[:12]}... from "
                              f"{'val' if candidate_pool is val_ids else 'test'} → train")
                        rescued = True
                        break
                if rescued:
                    break
            if not rescued:
                print(f"  [WARNING] Could not rescue type: {et}")

    # Final check
    final_missing = set(type_to_traces.keys()) - {
        c for t in train_ids for c in trace_to_errors[t]
    }
    assert not final_missing, f"Train set missing types: {final_missing}"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total, "Split sizes don't sum to total"
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/val overlap"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/test overlap"

    # Print summary
    print(f"\n{'='*60}")
    print(f"Final split")
    print(f"{'='*60}")
    print(f"  Train: {len(train_ids):4d} traces")
    print(f"  Val:   {len(val_ids):4d} traces")
    print(f"  Test:  {len(test_ids):4d} traces")
    print(f"  Total: {len(train_ids)+len(val_ids)+len(test_ids):4d} traces")

    # Per-type coverage
    print(f"\nError type coverage:")
    print(f"  {'Type':<40} {'Total':>5}  {'Train':>5}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*40}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for et in sorted(type_to_traces.keys()):
        n_tr = sum(1 for t in train_ids if et in trace_to_errors[t])
        n_va = sum(1 for t in val_ids if et in trace_to_errors[t])
        n_te = sum(1 for t in test_ids if et in trace_to_errors[t])
        n_to = len(type_to_traces[et])
        warn = "  <-- 0 in val+test" if (n_va + n_te) == 0 else ""
        print(f"  {et:<40} {n_to:>5}  {n_tr:>5}  {n_va:>5}  {n_te:>5}{warn}")

    # Save
    out_train = OUTPUT_DIR / "train_trace_ids.json"
    out_val = OUTPUT_DIR / "val_trace_ids.json"
    out_test = OUTPUT_DIR / "test_trace_ids.json"

    out_train.write_text(json.dumps(sorted(train_ids), indent=2))
    out_val.write_text(json.dumps(sorted(val_ids), indent=2))
    out_test.write_text(json.dumps(sorted(test_ids), indent=2))

    print(f"\nSaved:")
    print(f"  {out_train}")
    print(f"  {out_val}")
    print(f"  {out_test}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Split TRAIL traces into train/val/test")
    ap.add_argument("--base_dir", default=".",
                    help="Repo root (directory containing processed_annotations_*/)")
    args = ap.parse_args()
    make_splits(args.base_dir)
