"""
Task #59: Build stratified 80/20 train/test split per benchmark.

Output per benchmark: assignments/{benchmark}.json mapping trace_id -> one of
{"train", "test", "pinned"}. Pinned = rare categories that stay in training to
avoid empty cells (still feed graph construction).

Also writes:
  - onsets_train/{benchmark}.jsonl  (training-side onset rows for graph build)
  - For TRAIL: each row already keyed by trace_id, deduped.
  - For MAST: each unique trace_id may appear in multiple onset rows (model
    instances). We assign by unique trace_id so all instances share a side.
    The training onset file includes ALL records whose trace_id is on the
    training side.

Usage:
    python build_folds.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    ASSIGN_DIR, BENCHMARKS, MAST_ONSETS, MIN_PER_CAT, ONSETS_DIR, SEED, TRAIL_ONSETS,
)


def load_onsets(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def primary_category(rec: dict) -> str:
    counts = rec.get("count") or rec.get("present") or {}
    if not counts or max(counts.values()) == 0:
        return "_no_error"
    max_val = max(counts.values())
    return sorted(k for k, v in counts.items() if v == max_val)[0]


def dedup_by_trace_id(records: list[dict]) -> list[dict]:
    """Keep one representative record per unique trace_id (first occurrence)."""
    seen = set()
    out = []
    for rec in records:
        tid = rec["trace_id"]
        if tid in seen:
            continue
        seen.add(tid)
        out.append(rec)
    return out


def stratified_holdout(trace_ids: list, primaries: list, test_frac: float, seed: int):
    """Stratified 80/20. Returns {trace_id: 'train'|'test'|'pinned'}."""
    cat_counts = Counter(primaries)
    pinned = {c for c, n in cat_counts.items() if n < MIN_PER_CAT}

    assignment = {}
    rotated_ids, rotated_cats = [], []
    for tid, cat in zip(trace_ids, primaries):
        if cat in pinned:
            assignment[tid] = "pinned"
        else:
            rotated_ids.append(tid)
            rotated_cats.append(cat)

    if rotated_ids:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(rotated_ids)), rotated_cats))
        for i in train_idx:
            assignment[rotated_ids[i]] = "train"
        for i in test_idx:
            assignment[rotated_ids[i]] = "test"

    return assignment, pinned, cat_counts


def process_benchmark(name: str, cfg: dict, source_records: list[dict]):
    if cfg["split_filter"]:
        records = [r for r in source_records if r.get("split") == cfg["split_filter"]]
    else:
        records = source_records

    # Dedup for assignment (MAST has multiple rows per trace_id; TRAIL is already unique).
    unique = dedup_by_trace_id(records)
    trace_ids = [r["trace_id"] for r in unique]
    primaries = [primary_category(r) for r in unique]

    assignment, pinned, cat_counts = stratified_holdout(
        trace_ids, primaries, cfg["test_frac"], SEED
    )

    # Write assignment json
    ASSIGN_DIR.mkdir(parents=True, exist_ok=True)
    n_train = sum(1 for v in assignment.values() if v in ("train", "pinned"))
    n_test = sum(1 for v in assignment.values() if v == "test")
    out = {
        "benchmark": name,
        "test_frac": cfg["test_frac"],
        "n_unique_traces": len(trace_ids),
        "n_train_unique": n_train,
        "n_test_unique": n_test,
        "n_records_total": len(records),
        "n_pinned_unique": sum(1 for v in assignment.values() if v == "pinned"),
        "pinned_categories": sorted(pinned),
        "category_counts": dict(cat_counts),
        "assignment": {str(tid): v for tid, v in assignment.items()},
    }
    with open(ASSIGN_DIR / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)

    # Write training-side onset file (all records whose trace_id is train or pinned).
    ONSETS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = ONSETS_DIR / f"{name}.jsonl"
    n_train_records = 0
    with open(train_path, "w") as f:
        for rec in records:
            if assignment.get(rec["trace_id"]) in ("train", "pinned"):
                f.write(json.dumps(rec) + "\n")
                n_train_records += 1

    print(f"[{name}] unique={len(trace_ids)}  records={len(records)}  "
          f"train_unique={n_train}  test_unique={n_test}  "
          f"pinned={out['n_pinned_unique']} ({len(pinned)} cats)  "
          f"train_records_for_graph={n_train_records}")
    if pinned:
        print(f"    pinned cats (<{MIN_PER_CAT} traces): {sorted(pinned)}")


def main():
    trail = load_onsets(TRAIL_ONSETS)
    mast = load_onsets(MAST_ONSETS)

    for name, cfg in BENCHMARKS.items():
        src = trail if cfg["source"] == "trail" else mast
        process_benchmark(name, cfg, src)

    print(f"\nAssignments -> {ASSIGN_DIR}")
    print(f"Training onsets -> {ONSETS_DIR}")


if __name__ == "__main__":
    main()
