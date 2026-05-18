#!/usr/bin/env python3
"""
Summarize annotated Workstream-C TRAIL audit sheet into paper-ready artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from annotation_schema import BENCHMARK_TAXONOMY

SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_sheet_final_locked.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.audit_csv)
    if not rows:
        raise SystemExit("Audit sheet is empty.")

    annotated = [r for r in rows if r.get("mechanism_bucket", "").strip()]
    if not annotated:
        raise SystemExit("No annotated rows found. Fill mechanism_bucket first.")

    total = len(annotated)
    bucket_counter = Counter(r.get("mechanism_bucket", "").strip() or "UNLABELED" for r in annotated)
    split_counter = Counter(r.get("split", "").strip() or "UNKNOWN" for r in annotated)
    severity_counter = Counter((r.get("impact_severity", "").strip() or "unspecified").lower() for r in annotated)
    corr_role_counter = Counter((r.get("corr_edge_role", "").strip() or "unspecified").lower() for r in annotated)
    pattern_counter = Counter()
    taxonomy_leaf_counter = Counter()
    taxonomy_l1_counter = Counter()
    taxonomy_l2_counter = Counter()
    leaf_to_l1 = {}
    leaf_to_l2 = {}
    for l1, l2 in BENCHMARK_TAXONOMY.items():
        for l2_name, leaves in l2.items():
            for leaf in leaves:
                leaf_to_l1[leaf] = l1
                leaf_to_l2[leaf] = l2_name
    for row in annotated:
        tags = [x.strip() for x in (row.get("pattern_tags", "") or "").split("|")]
        for tag in tags:
            if tag:
                pattern_counter[tag] += 1
        for field in ("gold_leaf_categories", "baseline_leaf_categories", "causal_leaf_categories", "corr_leaf_categories"):
            for leaf in [x.strip() for x in (row.get(field, "") or "").split("|") if x.strip()]:
                taxonomy_leaf_counter[(field, leaf)] += 1
                taxonomy_l1_counter[(field, leaf_to_l1.get(leaf, "UNKNOWN"))] += 1
                taxonomy_l2_counter[(field, leaf_to_l2.get(leaf, "UNKNOWN"))] += 1

    bucket_rows = []
    for bucket, count in bucket_counter.most_common():
        bucket_rows.append({"bucket": bucket, "count": count, "percent": f"{pct(count, total):.2f}"})

    split_rows = []
    for split, count in sorted(split_counter.items()):
        split_rows.append({"split": split, "count": count, "percent": f"{pct(count, total):.2f}"})

    bucket_by_split = defaultdict(Counter)
    for row in annotated:
        bucket_by_split[row.get("split", "").strip() or "UNKNOWN"][row.get("mechanism_bucket", "").strip() or "UNLABELED"] += 1

    split_bucket_rows = []
    for split, bcounts in sorted(bucket_by_split.items()):
        denom = sum(bcounts.values())
        for bucket, count in bcounts.most_common():
            split_bucket_rows.append(
                {
                    "split": split,
                    "bucket": bucket,
                    "count": count,
                    "percent_within_split": f"{pct(count, denom):.2f}",
                }
            )

    severity_weighted_failure = 0
    n_corr_added_gain = 0
    n_corr_induced_harm = 0
    n_causal_backed_gain = 0
    n_high_regression = 0
    representatives: Dict[str, Dict] = {}

    for row in annotated:
        primary = (row.get("mechanism_bucket", "").strip() or "UNLABELED").lower()
        sev = (row.get("impact_severity", "").strip() or "unspecified").lower()
        sev_w = SEVERITY_WEIGHT.get(sev, 0)

        if primary == "corr-added-gain":
            n_corr_added_gain += 1
        if primary == "causal-backed-gain":
            n_causal_backed_gain += 1
        if primary == "corr-induced-harm":
            n_corr_induced_harm += 1
            severity_weighted_failure += sev_w
            if sev == "high":
                n_high_regression += 1

        bucket_key = row.get("mechanism_bucket", "").strip() or "UNLABELED"
        if bucket_key not in representatives:
            representatives[bucket_key] = {
                "bucket": bucket_key,
                "trace_id": row.get("trace_id", ""),
                "sample_id": row.get("sample_id", ""),
                "split": row.get("split", ""),
                "model": row.get("model", ""),
                "pattern_tags": row.get("pattern_tags", ""),
                "pattern_characteristics": row.get("pattern_characteristics", ""),
                "corr_edge_role": row.get("corr_edge_role", ""),
                "impact_severity": row.get("impact_severity", ""),
                "evidence_note": row.get("evidence_note", ""),
            }

    representative_rows = list(representatives.values())
    representative_rows.sort(key=lambda r: r["bucket"])

    net_corr_effect = n_corr_added_gain - n_corr_induced_harm
    summary = {
        "n_total_annotated": total,
        "mechanism_bucket_counts": bucket_counter,
        "split_counts": split_counter,
        "severity_counts": severity_counter,
        "corr_edge_role_counts": corr_role_counter,
        "top_pattern_tags": pattern_counter.most_common(20),
        "severity_weighted_failure": severity_weighted_failure,
        "n_causal_backed_gain": n_causal_backed_gain,
        "n_corr_added_gain": n_corr_added_gain,
        "n_corr_induced_harm": n_corr_induced_harm,
        "n_high_severity_regressions": n_high_regression,
        "net_corr_effect_added_gain_minus_harm": net_corr_effect,
        "outputs": {
            "bucket_summary_csv": str(args.out_dir / "trail_mechanism_bucket_summary.csv"),
            "split_summary_csv": str(args.out_dir / "trail_split_summary.csv"),
            "split_bucket_summary_csv": str(args.out_dir / "trail_split_mechanism_bucket_summary.csv"),
            "pattern_tag_summary_csv": str(args.out_dir / "trail_pattern_tag_summary.csv"),
            "taxonomy_leaf_summary_csv": str(args.out_dir / "trail_taxonomy_leaf_summary.csv"),
            "taxonomy_l1_summary_csv": str(args.out_dir / "trail_taxonomy_l1_summary.csv"),
            "taxonomy_l2_summary_csv": str(args.out_dir / "trail_taxonomy_l2_summary.csv"),
            "representative_cases_csv": str(args.out_dir / "trail_representative_cases.csv"),
        },
        "notes": [
            "Primary label is mechanism_bucket (not a generic failure bucket).",
            "Rows with missing mechanism_bucket are excluded from summary.",
        ],
    }

    pattern_rows = [{"pattern_tag": tag, "count": c, "percent": f"{pct(c, total):.2f}"} for tag, c in pattern_counter.most_common()]
    tax_leaf_rows = [
        {"field": field, "leaf_category": leaf, "count": c}
        for (field, leaf), c in taxonomy_leaf_counter.most_common()
    ]
    tax_l1_rows = [
        {"field": field, "taxonomy_l1": l1, "count": c}
        for (field, l1), c in taxonomy_l1_counter.most_common()
    ]
    tax_l2_rows = [
        {"field": field, "taxonomy_l2": l2, "count": c}
        for (field, l2), c in taxonomy_l2_counter.most_common()
    ]

    write_csv(args.out_dir / "trail_mechanism_bucket_summary.csv", bucket_rows)
    write_csv(args.out_dir / "trail_split_summary.csv", split_rows)
    write_csv(args.out_dir / "trail_split_mechanism_bucket_summary.csv", split_bucket_rows)
    write_csv(args.out_dir / "trail_pattern_tag_summary.csv", pattern_rows)
    write_csv(args.out_dir / "trail_taxonomy_leaf_summary.csv", tax_leaf_rows)
    write_csv(args.out_dir / "trail_taxonomy_l1_summary.csv", tax_l1_rows)
    write_csv(args.out_dir / "trail_taxonomy_l2_summary.csv", tax_l2_rows)
    write_csv(args.out_dir / "trail_representative_cases.csv", representative_rows)

    with (args.out_dir / "trail_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda x: dict(x))

    print(json.dumps(summary, indent=2, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
