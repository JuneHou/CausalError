#!/usr/bin/env python3
"""
Adjudicate GPT and human labels into final locked sheet.

Policy:
- If human adjudication file has non-empty value for a field, use human.
- Otherwise keep GPT value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ADJ_FIELDS = [
    "mechanism_bucket",
    "pattern_tags",
    "pattern_characteristics",
    "corr_edge_role",
    "impact_severity",
    "confidence",
    "evidence_note",
    "annotator",
]


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
        "--gpt_main_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_sheet_gpt52_labeled.csv"),
    )
    parser.add_argument(
        "--human_override_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_qc_subset_manual_filled.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    args = parser.parse_args()

    gpt_rows = load_csv(args.gpt_main_csv)
    if not gpt_rows:
        raise SystemExit("GPT main labeled sheet is empty.")
    human_rows = load_csv(args.human_override_csv) if args.human_override_csv.exists() else []
    hmap = {r.get("trace_id", ""): r for r in human_rows if r.get("trace_id", "")}

    merged = []
    n_overrides = 0
    n_rows_touched = 0

    for row in gpt_rows:
        tid = row.get("trace_id", "")
        h = hmap.get(tid)
        if not h:
            merged.append(row)
            continue
        m = dict(row)
        touched = False
        for f in ADJ_FIELDS:
            hv = h.get(f, "").strip()
            if hv:
                if m.get(f, "") != hv:
                    n_overrides += 1
                    touched = True
                m[f] = hv
        if touched:
            n_rows_touched += 1
        merged.append(m)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "trail_audit_sheet_final_locked.csv"
    write_csv(out_csv, merged)

    manifest = {
        "gpt_rows": len(gpt_rows),
        "human_rows": len(human_rows),
        "overrides_applied": n_overrides,
        "rows_touched": n_rows_touched,
        "output_csv": str(out_csv),
    }
    with (args.out_dir / "trail_adjudication_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
