#!/usr/bin/env python3
"""
01_make_splits.py — Split traces into train / val / test.

Ratio: ~80 / 10 / 10  (trace-level)
Rules:
  1. Every error type must appear at least once in BOTH train and test.
  2. Types with ≤ RARE_THRESHOLD traces (default 4):
       - 1 trace forced to TEST  (ensures test coverage for all labels)
       - remaining traces forced to TRAIN
     This replaces the old "force all to train" rule that left rare types
     with 0 support in test, making weighted F1 meaningless for those classes.
  3. Types with RARE_THRESHOLD < count ≤ ANCHOR_THRESHOLD (default 8):
       - at least 1 trace anchored to train before random assignment.
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
import random
from collections import defaultdict
from pathlib import Path

_GRAPH_DIR = Path(__file__).resolve().parent          # trail-benchmark/graph/
_BENCH_DIR = _GRAPH_DIR.parent / "benchmarking"      # trail-benchmark/benchmarking/

ANNOTATION_DIRS = [
    "processed_annotations_gaia",
    # "processed_annotations_swe_bench",  # excluded: train/eval GAIA only for now
]

# Types with ≤ RARE_THRESHOLD total traces → 1 forced to test, rest forced to train.
# Setting to 4 captures: Service Errors (2), Timeout Issues (2),
# Tool Definition Issues (3), Resource Exhaustion (3).
RARE_THRESHOLD   = 4

# Types with RARE_THRESHOLD < count ≤ ANCHOR_THRESHOLD → at least 1 anchored to train.
ANCHOR_THRESHOLD = 8

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

SEED = 42

OUTPUT_DIR = _GRAPH_DIR / "splits"


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
# Select forced-train and forced-test sets
# ---------------------------------------------------------------------------

def select_forced(
    type_to_traces: dict[str, list[str]],
    trace_to_errors: dict[str, set[str]],
    rare_threshold: int,
    anchor_threshold: int,
) -> tuple[set[str], set[str]]:
    """
    Return (forced_train, forced_test).

    For rare types (count ≤ rare_threshold):
      - 1 trace forced to TEST  (the one with fewest *other* error types,
        so the test example is as pure/specific as possible)
      - all remaining traces for that type forced to TRAIN

    For anchor types (rare_threshold < count ≤ anchor_threshold):
      - greedy: anchor 1 trace to train (highest multi-type coverage)

    A single-trace type (count == 1) is only forced to train — we cannot
    hold out test without losing train coverage.
    """
    forced_train: set[str] = set()
    forced_test:  set[str] = set()

    # ---- Phase 1: rare types (1 to test, rest to train) ----
    rare_types = [et for et, tr in type_to_traces.items() if len(tr) <= rare_threshold]
    rare_types.sort(key=lambda et: len(type_to_traces[et]))   # rarest first

    for et in rare_types:
        traces = type_to_traces[et]

        if len(traces) == 1:
            # Only 1 trace — can't split; force to train
            forced_train.add(traces[0])
            print(f"  [single]    {et}: 1 trace → forced to train only")
            continue

        # Pick the test trace: the one already in forced_test (covers another rare type)
        # if possible; otherwise the one with the fewest *other* error categories
        # (most specific to this error type → cleanest test example).
        already_test = [t for t in traces if t in forced_test]
        if already_test:
            # Re-use a trace already marked for test (no new test slot needed)
            test_trace = already_test[0]
            print(f"  [rare-test] {et}: {len(traces)} traces → test={test_trace[:12]}... (shared)")
        else:
            # Pick the trace with fewest *other* error categories (most specific)
            test_trace = min(traces, key=lambda t: len(trace_to_errors[t]))
            forced_test.add(test_trace)
            print(f"  [rare-test] {et}: {len(traces)} traces → 1 to test, {len(traces)-1} to train")

        for t in traces:
            if t != test_trace:
                forced_train.add(t)

    # ---- Phase 2: anchor types (at least 1 to train) ----
    anchor_types = [
        et for et, tr in type_to_traces.items()
        if rare_threshold < len(tr) <= anchor_threshold
    ]
    anchor_types.sort(key=lambda et: len(type_to_traces[et]))

    for et in anchor_types:
        traces = type_to_traces[et]
        if any(t in forced_train for t in traces):
            print(f"  [anchor-ok] {et}: {len(traces)} traces → already covered in train")
            continue
        # Greedy: pick trace with most error types (maximises train coverage)
        best = max(traces, key=lambda t: len(trace_to_errors[t]))
        forced_train.add(best)
        print(f"  [anchor]    {et}: {len(traces)} traces → anchored {best[:12]}... to train")

    # Sanity: a trace can't be in both
    overlap = forced_train & forced_test
    assert not overlap, f"Trace(s) in both forced_train and forced_test: {overlap}"

    return forced_train, forced_test


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def make_splits(base_dir: str = ".") -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    trace_to_errors = load_annotations(base_dir)
    type_to_traces  = build_type_index(trace_to_errors)

    all_trace_ids  = sorted(trace_to_errors.keys())
    n_total        = len(all_trace_ids)
    n_train_target = round(n_total * TRAIN_RATIO)
    n_val_target   = round(n_total * VAL_RATIO)
    n_test_target  = n_total - n_train_target - n_val_target

    print(f"\nTotal traces: {n_total}")
    print(f"Target split: train={n_train_target}, val={n_val_target}, test={n_test_target}")
    print(f"\nApplying stratification rules (seed={SEED})...")

    forced_train, forced_test = select_forced(
        type_to_traces, trace_to_errors, RARE_THRESHOLD, ANCHOR_THRESHOLD
    )
    print(f"\nForced-train: {len(forced_train)} traces  |  Forced-test: {len(forced_test)} traces")

    # Remaining traces (not forced to either set) → random split
    remaining = [t for t in all_trace_ids if t not in forced_train and t not in forced_test]
    random.shuffle(remaining)

    # How many more train traces needed beyond forced set?
    n_more_train = max(0, n_train_target - len(forced_train))
    # How many more test traces beyond forced_test?
    n_more_test  = max(0, n_test_target - len(forced_test))
    # Val gets the rest of remaining after allocating to train and test
    n_more_val   = len(remaining) - n_more_train - n_more_test

    if n_more_val < 0:
        # Forced sets are large — reduce val proportionally
        n_more_val   = n_val_target
        n_more_test  = max(0, len(remaining) - n_more_train - n_more_val)

    train_ids = sorted(forced_train) + remaining[:n_more_train]
    val_ids   = remaining[n_more_train: n_more_train + n_more_val]
    test_ids  = sorted(forced_test)  + remaining[n_more_train + n_more_val:]

    # Verify: every error type must appear in both train and test
    train_coverage = {c for t in train_ids for c in trace_to_errors[t]}
    test_coverage  = {c for t in test_ids  for c in trace_to_errors[t]}

    missing_train = set(type_to_traces.keys()) - train_coverage
    missing_test  = set(type_to_traces.keys()) - test_coverage

    if missing_train:
        for et in sorted(missing_train):
            for t in list(val_ids) + list(test_ids):
                if et in trace_to_errors[t]:
                    if t in val_ids:
                        val_ids.remove(t)
                    else:
                        test_ids.remove(t)
                    train_ids.append(t)
                    train_coverage |= trace_to_errors[t]
                    print(f"  [rescue-train] {et}: moved {t[:12]}... → train")
                    break

    if missing_test:
        for et in sorted(missing_test):
            # Prefer moving from val (costs fewer training examples); fall back to train
            for t in list(val_ids) + list(train_ids):
                if et in trace_to_errors[t]:
                    if t in val_ids:
                        val_ids.remove(t)
                    else:
                        train_ids.remove(t)
                    test_ids.append(t)
                    test_coverage |= trace_to_errors[t]
                    print(f"  [rescue-test]  {et}: moved {t[:12]}... → test")
                    break
            else:
                print(f"  [WARNING] {et}: no test trace available (only {len(type_to_traces[et])} total)")

    # Final assertions
    assert not (set(type_to_traces.keys()) - {c for t in train_ids for c in trace_to_errors[t]}), \
        "Train still missing some error types"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total, "Split sizes don't sum"
    assert not (set(train_ids) & set(val_ids)),  "Train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "Train/test overlap"
    assert not (set(val_ids)   & set(test_ids)), "Val/test overlap"

    # Summary
    print(f"\n{'='*60}")
    print(f"Final split")
    print(f"{'='*60}")
    print(f"  Train: {len(train_ids):4d} traces")
    print(f"  Val:   {len(val_ids):4d} traces")
    print(f"  Test:  {len(test_ids):4d} traces")
    print(f"  Total: {len(train_ids)+len(val_ids)+len(test_ids):4d} traces")

    print(f"\nError type coverage:")
    print(f"  {'Type':<40} {'Total':>5}  {'Train':>5}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*40}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for et in sorted(type_to_traces.keys()):
        n_tr = sum(1 for t in train_ids if et in trace_to_errors[t])
        n_va = sum(1 for t in val_ids   if et in trace_to_errors[t])
        n_te = sum(1 for t in test_ids  if et in trace_to_errors[t])
        n_to = len(type_to_traces[et])
        warn = "  *** 0 in test" if n_te == 0 else ""
        print(f"  {et:<40} {n_to:>5}  {n_tr:>5}  {n_va:>5}  {n_te:>5}{warn}")

    # Save
    out_train = OUTPUT_DIR / "train_trace_ids.json"
    out_val   = OUTPUT_DIR / "val_trace_ids.json"
    out_test  = OUTPUT_DIR / "test_trace_ids.json"

    out_train.write_text(json.dumps(sorted(train_ids), indent=2))
    out_val.write_text(  json.dumps(sorted(val_ids),   indent=2))
    out_test.write_text( json.dumps(sorted(test_ids),  indent=2))

    print(f"\nSaved:")
    print(f"  {out_train}")
    print(f"  {out_val}")
    print(f"  {out_test}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Split TRAIL traces into train/val/test")
    ap.add_argument("--base_dir", default=str(_BENCH_DIR),
                    help="Directory containing processed_annotations_*/ (default: benchmarking/)")
    args = ap.parse_args()
    make_splits(args.base_dir)
