#!/usr/bin/env python3
"""
Select Workstream-B case-study candidates from existing TRAIL outputs only.

This script compares three variants for each (model, split, trace_id):
1) baseline
2) +GI causal-only (span-index)
3) +GI corr-union at tau=0.35 (span-index)

Outputs:
- summary CSV with per-instance scores/deltas
- top working candidates
- top non-working candidates
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ALL_CATEGORIES: List[str] = [
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Incorrect Memory Usage",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
]

# Keep defaults aligned with current TRAIL table scope.
OPEN_SOURCE_MODELS: List[str] = [
    "openai-gpt-oss-120b",
    "openai-gpt-oss-20b",
    "google-gemma-3-27b-it",
    "mistralai-Mistral-Small-3.1-24B-Instruct-2503",
    "Tongyi-Zhiwen-QwenLong-L1-32B",
]


@dataclass
class MetricRow:
    wf1: float
    loc: float
    joint: float


def normalize_category(category: str) -> str:
    if not category:
        return ""
    cat = category.lower().strip()
    cat_ns = cat.replace(" ", "")
    for std in ALL_CATEGORIES:
        if cat == std.lower() or cat_ns == std.lower().replace(" ", ""):
            return std
    for std in ALL_CATEGORIES:
        if cat_ns in std.lower().replace(" ", ""):
            return std
    return category.strip()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(gold: Dict, pred: Dict) -> MetricRow:
    gt_errors = gold.get("errors", [])
    pr_errors = pred.get("errors", [])

    gt_cats = [normalize_category(e.get("category", "")) for e in gt_errors if e.get("category", "")]
    pr_cats = [normalize_category(e.get("category", "")) for e in pr_errors if e.get("category", "")]

    gt_locs = [e.get("location", "") for e in gt_errors]
    pr_locs = [e.get("location", "") for e in pr_errors]

    gt_pairs = {(gt_locs[i], gt_cats[i]) for i in range(min(len(gt_locs), len(gt_cats)))}
    pr_pairs = {(pr_locs[i], pr_cats[i]) for i in range(min(len(pr_locs), len(pr_cats)))}

    gt_set = set(gt_cats)
    pr_set = set(pr_cats)
    tp = len(gt_set & pr_set)
    fp = len(pr_set - gt_set)
    fn = len(gt_set - pr_set)

    if 2 * tp + fp + fn == 0:
        wf1 = 0.0
    else:
        wf1 = (2.0 * tp) / (2.0 * tp + fp + fn)

    loc_common = set(gt_locs) & set(pr_locs)
    loc = len(loc_common) / len(set(gt_locs)) if gt_locs else 0.0
    joint = len(gt_pairs & pr_pairs) / len(gt_pairs) if gt_pairs else 0.0
    return MetricRow(wf1=wf1, loc=loc, joint=joint)


def discover_trace_ids(folder: Path) -> List[str]:
    ids = []
    for p in folder.glob("*.json"):
        if p.name.startswith("_meta_"):
            continue
        ids.append(p.stem)
    return ids


def build_rows(repo_root: Path, models: List[str], splits: List[str]) -> Tuple[List[Dict], Dict[str, int]]:
    rows: List[Dict] = []
    stats = {
        "missing_gold": 0,
        "json_decode_error": 0,
        "other_read_error": 0,
        "rows_emitted": 0,
    }
    gt_dir_map = {
        "GAIA_dedup": repo_root / "benchmarking/processed_annotations_gaia",
        "SWE_Bench_dedup": repo_root / "benchmarking/processed_annotations_swe_bench",
    }

    for model in models:
        for split in splits:
            baseline_dir = (
                repo_root
                / "benchmarking/outputs/zero_shot/compressed"
                / f"outputs_{model}-{split}"
            )
            causal_dir = (
                repo_root
                / "benchmarking/outputs/zero_shot/compressed"
                / f"outputs_{model}-{split}-graph_inject_causal_only_span_index"
            )
            corr_dir = (
                repo_root
                / "benchmarking/outputs_thres/t0.35"
                / f"outputs_{model}-{split}-graph_inject_causal_corr0.35_span_index"
            )
            gt_dir = gt_dir_map[split]

            if not (baseline_dir.exists() and causal_dir.exists() and corr_dir.exists() and gt_dir.exists()):
                continue

            trace_ids = sorted(
                set(discover_trace_ids(baseline_dir))
                & set(discover_trace_ids(causal_dir))
                & set(discover_trace_ids(corr_dir))
            )

            for trace_id in trace_ids:
                gold_path = gt_dir / f"{trace_id}.json"
                if not gold_path.exists():
                    stats["missing_gold"] += 1
                    continue

                try:
                    gold = load_json(gold_path)
                    baseline = load_json(baseline_dir / f"{trace_id}.json")
                    causal = load_json(causal_dir / f"{trace_id}.json")
                    corr = load_json(corr_dir / f"{trace_id}.json")
                except json.JSONDecodeError:
                    stats["json_decode_error"] += 1
                    continue
                except Exception:
                    stats["other_read_error"] += 1
                    continue

                m_base = compute_metrics(gold, baseline)
                m_causal = compute_metrics(gold, causal)
                m_corr = compute_metrics(gold, corr)

                # Balanced score proxy for ranking.
                score_base = m_base.wf1 + m_base.loc + m_base.joint
                score_causal = m_causal.wf1 + m_causal.loc + m_causal.joint
                score_corr = m_corr.wf1 + m_corr.loc + m_corr.joint

                corr_vs_causal = score_corr - score_causal
                corr_vs_base = score_corr - score_base

                d_wf1_vs_causal = m_corr.wf1 - m_causal.wf1
                d_loc_vs_causal = m_corr.loc - m_causal.loc
                d_joint_vs_causal = m_corr.joint - m_causal.joint
                d_wf1_vs_base = m_corr.wf1 - m_base.wf1
                d_loc_vs_base = m_corr.loc - m_base.loc
                d_joint_vs_base = m_corr.joint - m_base.joint

                row = {
                    "model": model,
                    "split": split,
                    "trace_id": trace_id,
                    "base_wf1": m_base.wf1,
                    "base_loc": m_base.loc,
                    "base_joint": m_base.joint,
                    "causal_wf1": m_causal.wf1,
                    "causal_loc": m_causal.loc,
                    "causal_joint": m_causal.joint,
                    "corr_wf1": m_corr.wf1,
                    "corr_loc": m_corr.loc,
                    "corr_joint": m_corr.joint,
                    "score_base": score_base,
                    "score_causal": score_causal,
                    "score_corr": score_corr,
                    "delta_corr_vs_causal": corr_vs_causal,
                    "delta_corr_vs_base": corr_vs_base,
                    "delta_wf1_vs_causal": d_wf1_vs_causal,
                    "delta_loc_vs_causal": d_loc_vs_causal,
                    "delta_joint_vs_causal": d_joint_vs_causal,
                    "delta_wf1_vs_base": d_wf1_vs_base,
                    "delta_loc_vs_base": d_loc_vs_base,
                    "delta_joint_vs_base": d_joint_vs_base,
                }

                if corr_vs_causal > 0 and corr_vs_base > 0:
                    row["bucket"] = "working"
                elif corr_vs_causal < 0:
                    row["bucket"] = "not_working"
                else:
                    row["bucket"] = "neutral"

                rows.append(row)
                stats["rows_emitted"] += 1

    return rows, stats


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_top(rows: List[Dict], bucket: str, k: int) -> List[Dict]:
    filtered = [r for r in rows if r["bucket"] == bucket]
    if bucket == "working":
        filtered.sort(key=lambda r: (r["delta_corr_vs_causal"], r["delta_corr_vs_base"]), reverse=True)
    else:
        filtered.sort(key=lambda r: r["delta_corr_vs_causal"])
    return filtered[:k]


def choose_top_metric(rows: List[Dict], metric: str, bucket: str, k: int) -> List[Dict]:
    key_vs_causal = f"delta_{metric}_vs_causal"
    key_vs_base = f"delta_{metric}_vs_base"
    if bucket == "working":
        filtered = [r for r in rows if r[key_vs_causal] > 0 and r[key_vs_base] > 0]
        filtered.sort(key=lambda r: (r[key_vs_causal], r[key_vs_base]), reverse=True)
    else:
        filtered = [r for r in rows if r[key_vs_causal] < 0]
        filtered.sort(key=lambda r: r[key_vs_causal])
    return filtered[:k]


def build_model_priority(rows: List[Dict]) -> List[Dict]:
    # Per-model summary so we can pick models that best support the paper goal.
    by_model: Dict[str, List[Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out: List[Dict] = []
    for model, rws in by_model.items():
        n = len(rws)
        def mean(key: str) -> float:
            return sum(r[key] for r in rws) / n if n else 0.0
        def rate_pos(key: str) -> float:
            return sum(1 for r in rws if r[key] > 0) / n if n else 0.0
        out.append(
            {
                "model": model,
                "n_instances": n,
                "gaia_n": sum(1 for r in rws if r["split"] == "GAIA_dedup"),
                "swe_n": sum(1 for r in rws if r["split"] == "SWE_Bench_dedup"),
                "mean_delta_wf1_vs_causal": mean("delta_wf1_vs_causal"),
                "mean_delta_loc_vs_causal": mean("delta_loc_vs_causal"),
                "mean_delta_joint_vs_causal": mean("delta_joint_vs_causal"),
                "mean_delta_wf1_vs_base": mean("delta_wf1_vs_base"),
                "mean_delta_loc_vs_base": mean("delta_loc_vs_base"),
                "mean_delta_joint_vs_base": mean("delta_joint_vs_base"),
                "wf1_win_rate_vs_causal": rate_pos("delta_wf1_vs_causal"),
                "loc_win_rate_vs_causal": rate_pos("delta_loc_vs_causal"),
                "joint_win_rate_vs_causal": rate_pos("delta_joint_vs_causal"),
                "balanced_mean_delta_vs_causal": mean("delta_corr_vs_causal"),
                "balanced_win_rate_vs_causal": rate_pos("delta_corr_vs_causal"),
            }
        )
    # Sort by balanced mean as a default overall priority view.
    out.sort(key=lambda x: x["balanced_mean_delta_vs_causal"], reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_b_case_study/results"),
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--models",
        nargs="*",
        default=OPEN_SOURCE_MODELS,
        help="Model folder-name tokens used in outputs paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["GAIA_dedup", "SWE_Bench_dedup"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = build_rows(args.repo_root, args.models, args.splits)
    if not rows:
        raise SystemExit("No comparable rows found. Check paths and available outputs.")

    summary_path = args.out_dir / "instance_comparison_summary.csv"
    working_path = args.out_dir / "top_working_candidates.csv"
    not_working_path = args.out_dir / "top_not_working_candidates.csv"
    model_priority_path = args.out_dir / "model_priority_by_metric.csv"

    write_csv(rows, summary_path)
    write_csv(choose_top(rows, "working", args.top_k), working_path)
    write_csv(choose_top(rows, "not_working", args.top_k), not_working_path)
    write_csv(build_model_priority(rows), model_priority_path)

    for metric in ("wf1", "loc", "joint"):
        write_csv(
            choose_top_metric(rows, metric=metric, bucket="working", k=args.top_k),
            args.out_dir / f"top_working_candidates_{metric}.csv",
        )
        write_csv(
            choose_top_metric(rows, metric=metric, bucket="not_working", k=args.top_k),
            args.out_dir / f"top_not_working_candidates_{metric}.csv",
        )

    manifest = {
        "rows_total": len(rows),
        "rows_working": sum(1 for r in rows if r["bucket"] == "working"),
        "rows_not_working": sum(1 for r in rows if r["bucket"] == "not_working"),
        "rows_neutral": sum(1 for r in rows if r["bucket"] == "neutral"),
        "read_stats": stats,
        "models": args.models,
        "splits": args.splits,
        "outputs": {
            "summary": str(summary_path),
            "working": str(working_path),
            "not_working": str(not_working_path),
            "model_priority": str(model_priority_path),
            "working_wf1": str(args.out_dir / "top_working_candidates_wf1.csv"),
            "working_loc": str(args.out_dir / "top_working_candidates_loc.csv"),
            "working_joint": str(args.out_dir / "top_working_candidates_joint.csv"),
            "not_working_wf1": str(args.out_dir / "top_not_working_candidates_wf1.csv"),
            "not_working_loc": str(args.out_dir / "top_not_working_candidates_loc.csv"),
            "not_working_joint": str(args.out_dir / "top_not_working_candidates_joint.csv"),
        },
        "notes": [
            "Selection uses existing outputs only (no new experiments).",
            "Scores are per-instance proxies mirroring W-F1, location, and joint notions.",
            "Final paper examples should still be manually validated from top candidates.",
        ],
    }
    with (args.out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
