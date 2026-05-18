#!/usr/bin/env python3
"""
Prepare QC subset for manual re-labeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_sheet_gpt52_labeled.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    parser.add_argument("--n_qc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = load_csv(args.labeled_csv)
    if not rows:
        raise SystemExit("Labeled CSV is empty.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r.get("mechanism_bucket", "UNLABELED")].append(r)

    selected = []
    # First pass: one per bucket if possible.
    for bucket, rws in sorted(by_bucket.items(), key=lambda x: len(x[1]), reverse=True):
        if len(selected) >= args.n_qc:
            break
        if rws:
            selected.append(rng.choice(rws))

    # Fill with random from remainder.
    selected_ids = {r.get("trace_id", "") for r in selected}
    remaining = [r for r in rows if r.get("trace_id", "") not in selected_ids]
    rng.shuffle(remaining)
    while len(selected) < args.n_qc and remaining:
        selected.append(remaining.pop())

    # Produce blank manual copy.
    manual_rows = []
    for r in selected:
        m = dict(r)
        m["mechanism_bucket"] = ""
        m["pattern_tags"] = ""
        m["pattern_characteristics"] = ""
        m["corr_edge_role"] = ""
        m["impact_severity"] = ""
        m["confidence"] = ""
        m["evidence_note"] = ""
        m["annotator"] = "human_qc_pending"
        manual_rows.append(m)

    qc_ref = args.out_dir / "trail_qc_subset_reference_gpt52.csv"
    qc_blank = args.out_dir / "trail_qc_subset_manual_blank.csv"
    qc_manifest = args.out_dir / "trail_qc_subset_manifest.json"
    write_csv(qc_ref, selected)
    write_csv(qc_blank, manual_rows)

    manifest = {
        "n_rows_total": len(rows),
        "n_qc": len(selected),
        "bucket_counts_qc": dict(Counter(r.get("mechanism_bucket", "UNLABELED") for r in selected)),
        "outputs": {
            "qc_reference_gpt52": str(qc_ref),
            "qc_manual_blank": str(qc_blank),
        },
    }
    with qc_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
