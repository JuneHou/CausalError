#!/usr/bin/env python3
"""
Compute QC agreement between GPT labels and manual relabeling subset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


FIELDS = [
    "mechanism_bucket",
    "corr_edge_role",
    "impact_severity",
]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {r.get("trace_id", ""): r for r in rows if r.get("trace_id", "")}


def pairwise_agreement(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def cohen_kappa(a: List[str], b: List[str]) -> float:
    # Simple nominal kappa implementation.
    n = len(a)
    if n == 0:
        return 0.0
    po = pairwise_agreement(a, b)
    ca = Counter(a)
    cb = Counter(b)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt_qc_reference_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_qc_subset_reference_gpt52.csv"),
    )
    parser.add_argument(
        "--human_qc_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_qc_subset_manual_filled.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    args = parser.parse_args()

    gpt = as_map(load_csv(args.gpt_qc_reference_csv))
    human = as_map(load_csv(args.human_qc_csv))
    common_ids = sorted(set(gpt) & set(human))
    if not common_ids:
        raise SystemExit("No overlapping trace_id between GPT and human QC files.")

    summary_rows = []
    disagreements = []
    for field in FIELDS:
        a = [gpt[t].get(field, "").strip() for t in common_ids]
        b = [human[t].get(field, "").strip() for t in common_ids]
        acc = pairwise_agreement(a, b)
        kappa = cohen_kappa(a, b)
        summary_rows.append(
            {
                "field": field,
                "n": len(common_ids),
                "agreement": f"{acc:.4f}",
                "kappa": f"{kappa:.4f}",
            }
        )
        for tid, av, bv in zip(common_ids, a, b):
            if av != bv:
                disagreements.append(
                    {
                        "trace_id": tid,
                        "field": field,
                        "gpt_value": av,
                        "human_value": bv,
                    }
                )

    overall = {
        "n_qc_common": len(common_ids),
        "fields": FIELDS,
        "mean_agreement": round(sum(float(r["agreement"]) for r in summary_rows) / len(summary_rows), 4),
        "mean_kappa": round(sum(float(r["kappa"]) for r in summary_rows) / len(summary_rows), 4),
        "outputs": {
            "qc_agreement_csv": str(args.out_dir / "trail_qc_agreement_summary.csv"),
            "qc_disagreements_csv": str(args.out_dir / "trail_qc_disagreements.csv"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "trail_qc_agreement_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    if disagreements:
        with (args.out_dir / "trail_qc_disagreements.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)

    with (args.out_dir / "trail_qc_agreement_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
