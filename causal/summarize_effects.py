#!/usr/bin/env python3
"""
Merge b_effect.jsonl files from one or more run directories and produce a
combined per-edge summary table.

Usage (from benchmarking/):
    # Single run
    python summarize_effects.py --runs outputs/full_run_gpt4o

    # Merge incremental on top of base (incremental wins on duplicate error_id×edge)
    python summarize_effects.py \
        --runs outputs/full_run_gpt4o outputs/full_run_incremental \
        --out outputs/combined_effect_summary.csv

Output columns (printed + optional CSV):
    edge_a, edge_b, n_total, n_valid (b_present_baseline=True),
    n_disappeared, n_weakened, n_not_observable, n_unchanged,
    n_emerged, n_strengthened,
    delta, delta_valid_only, validated

delta formula:
    disappeared=-1, weakened=-0.5, unchanged=0, not_observable=0,
    emerged=+1, strengthened=+1, delayed=0, earlier=0
    delta = sum(scores) / n_total
    delta_valid_only = same but restricted to b_present_baseline=True cases
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Effect score mapping
# ---------------------------------------------------------------------------

EFFECT_SCORES: Dict[str, float] = {
    "disappeared":   -1.0,
    "weakened":      -0.5,
    "delayed":        0.0,
    "earlier":        0.0,
    "unchanged":      0.0,
    "not_observable": 0.0,
    "emerged":       +1.0,
    "strengthened":  +1.0,
}

# Negative delta means intervention suppressed B (causal prevention confirmed).
VALIDATED_THRESHOLD = -0.3   # delta ≤ this → "validated"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_postcheck_failures(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "postcheck_failures.jsonl")
    return _load_jsonl(path)


# ---------------------------------------------------------------------------
# Merge logic  (later run dirs win on duplicate error_id × edge)
# ---------------------------------------------------------------------------

def _merge_key(r: dict) -> Tuple[str, str, str, str]:
    """
    Stable merge key: (trace_id, span_id, edge_a.strip(), edge_b.strip()).

    Using trace_id + span_id (second field of error_id) instead of full error_id
    avoids spurious duplicates when the category part of error_id was renamed
    (e.g., ' Incorrect Problem Identification' → 'Incorrect Problem Identification').
    edge_a/edge_b are also stripped to survive the same rename.
    """
    tid = r.get("trace_id", "")
    eid = r.get("error_id", "")
    parts = eid.split("|")
    span_id = parts[1] if len(parts) > 1 else eid  # second segment is the span/location id
    edge = r.get("edge", {})
    return (tid, span_id, edge.get("a", "").strip(), edge.get("b", "").strip())


def merge_b_effects(run_dirs: List[str]) -> List[dict]:
    """
    Load b_effect.jsonl from each run_dir in order.
    Last writer wins on key = (trace_id, span_id, edge_a, edge_b).
    Category renames in error_id are handled by using only the span_id segment.
    """
    merged: Dict[Tuple[str, str, str, str], dict] = {}
    for d in run_dirs:
        path = os.path.join(d, "b_effect.jsonl")
        records = _load_jsonl(path)
        if not records:
            print(f"  [warn] no b_effect.jsonl in {d}", file=sys.stderr)
            continue
        for r in records:
            merged[_merge_key(r)] = r
        print(f"  Loaded {len(records):4d} records from {d}  (total merged so far: {len(merged)})")
    return list(merged.values())


# ---------------------------------------------------------------------------
# Aggregate per edge
# ---------------------------------------------------------------------------

def aggregate(records: List[dict], threshold: float = VALIDATED_THRESHOLD) -> List[dict]:
    """
    Aggregate merged records into per-(edge_a, edge_b) stats.
    Returns list of row dicts, sorted by edge_a then edge_b.

    Δ formula (matches effect_aggregator.py):
        Δ = mean(target_present_after) - mean(b_present_baseline)
    Negative Δ means do(A=0) reduced downstream B (causal prevention confirmed).

    delta_valid_only uses only records where b_present_baseline=True,
    giving the cleaner estimate restricted to co-occurrence cases.
    """
    label_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # For Δ: collect b_present_baseline and target_present_after per edge
    baseline_lists: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    rerun_lists:    Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    valid_baseline: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    valid_rerun:    Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        edge = r.get("edge", {})
        a = edge.get("a", "").strip()
        b = edge.get("b", "").strip()
        label = r.get("effect_label", "not_observable")
        b_present = bool(r.get("b_present_baseline", False))
        b_after   = bool(r.get("target_present_after", False))

        label_counts[a][b][label] += 1
        baseline_lists[a][b].append(b_present)
        rerun_lists[a][b].append(b_after)
        if b_present:
            valid_baseline[a][b].append(b_present)
            valid_rerun[a][b].append(b_after)

    rows = []
    for a in sorted(label_counts):
        for b in sorted(label_counts[a]):
            c = label_counts[a][b]
            bl = baseline_lists[a][b]
            rr = rerun_lists[a][b]
            n = len(bl)
            n_valid = len(valid_baseline[a][b])

            if n > 0:
                delta = sum(rr) / n - sum(bl) / n
            else:
                delta = None

            if n_valid > 0:
                delta_valid = sum(valid_rerun[a][b]) / n_valid - sum(valid_baseline[a][b]) / n_valid
            else:
                delta_valid = None

            validated = (delta is not None and delta < -abs(threshold))

            rows.append({
                "edge_a":           a,
                "edge_b":           b,
                "n_total":          n,
                "n_valid":          n_valid,
                "n_disappeared":    c.get("disappeared", 0),
                "n_weakened":       c.get("weakened", 0),
                "n_not_observable": c.get("not_observable", 0),
                "n_unchanged":      c.get("unchanged", 0),
                "n_emerged":        c.get("emerged", 0),
                "n_strengthened":   c.get("strengthened", 0),
                "delta":            round(delta, 4) if delta is not None else None,
                "delta_valid_only": round(delta_valid, 4) if delta_valid is not None else None,
                "validated":        validated,
            })
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(rows: List[dict], postcheck_failures: List[dict] = ()) -> None:
    # Edge effect table
    col_w = 50
    print()
    print("  Δ = mean(target_present_after) - mean(b_present_baseline)  [negative = causal prevention]")
    print(f"{'Edge':<{col_w}}  {'n':>5}  {'n_valid':>7}  {'delta':>7}  {'delta(valid)':>12}  validated")
    print("-" * (col_w + 50))
    for r in rows:
        edge_str = f"{r['edge_a']} -> {r['edge_b']}"
        delta_str     = f"{r['delta']:+.3f}" if r["delta"] is not None else "  N/A"
        dv_str        = f"{r['delta_valid_only']:+.3f}" if r["delta_valid_only"] is not None else "  N/A"
        val_str       = "YES" if r["validated"] else "no"
        print(f"{edge_str:<{col_w}}  {r['n_total']:>5}  {r['n_valid']:>7}  {delta_str:>7}  {dv_str:>12}  {val_str}")

    # Label breakdown
    print()
    print("Label breakdown per edge:")
    label_cols = ["disappeared", "weakened", "unchanged", "not_observable", "emerged", "strengthened"]
    header = f"{'Edge':<{col_w}}" + "".join(f"  {l[:4]:>4}" for l in label_cols)
    print(header)
    print("-" * (col_w + len(label_cols) * 6))
    for r in rows:
        edge_str = f"{r['edge_a']} -> {r['edge_b']}"
        vals = "".join(f"  {r['n_' + l]:>4}" for l in label_cols)
        print(f"{edge_str:<{col_w}}{vals}")

    # Patch failures summary
    if postcheck_failures:
        print()
        print(f"Patch failures ({len(postcheck_failures)} cases):")
        for pf in postcheck_failures:
            eid = pf.get("error_id", "?")
            tmpl = pf.get("template_used", "?")
            reasons = pf.get("postcheck_failures", [])
            print(f"  [{tmpl}] {eid}")
            for r in reasons:
                print(f"    → {r}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge b_effect.jsonl from multiple runs and produce a summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more run output directories (processed left-to-right; last wins on duplicates).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--threshold", type=float, default=VALIDATED_THRESHOLD,
        help="Delta threshold (≤) for marking an edge as validated (causal prevention).",
    )
    args = parser.parse_args()

    threshold = args.threshold

    print(f"\nMerging b_effect from {len(args.runs)} run(s):")
    records = merge_b_effects(args.runs)
    print(f"  → {len(records)} unique (error_id × edge) records after merge")

    # Collect all postcheck failures from all runs
    all_failures: List[dict] = []
    for d in args.runs:
        all_failures.extend(_load_postcheck_failures(d))

    rows = aggregate(records, threshold=threshold)
    print_table(rows, postcheck_failures=all_failures)

    if args.out:
        write_csv(rows, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
