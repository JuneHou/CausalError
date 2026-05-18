#!/usr/bin/env python3
"""
Build Workstream-C TRAIL audit sheet from existing outputs only.

Inputs:
- Workstream-B per-instance comparison CSV
- Existing TRAIL prediction/gold JSON files

Outputs:
- trail_audit_sheet.csv (manual annotation sheet with blank label fields)
- trail_audit_candidates.json (selected row metadata)
- trail_sampling_manifest.json (sampling + coverage stats)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from annotation_schema import (
    extract_leaf_categories,
    normalize_category,
    write_schema_files,
)

def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def normalize_row_types(rows: List[Dict[str, str]]) -> List[Dict]:
    float_fields = [
        "base_wf1",
        "base_loc",
        "base_joint",
        "causal_wf1",
        "causal_loc",
        "causal_joint",
        "corr_wf1",
        "corr_loc",
        "corr_joint",
        "score_base",
        "score_causal",
        "score_corr",
        "delta_corr_vs_causal",
        "delta_corr_vs_base",
        "delta_wf1_vs_causal",
        "delta_loc_vs_causal",
        "delta_joint_vs_causal",
        "delta_wf1_vs_base",
        "delta_loc_vs_base",
        "delta_joint_vs_base",
    ]
    out: List[Dict] = []
    for row in rows:
        nr = dict(row)
        for field in float_fields:
            nr[field] = to_float(row.get(field, "0"))
        out.append(nr)
    return out


def row_sort_key(row: Dict, bucket: str) -> Tuple[float, float, float]:
    if bucket == "working":
        return (
            row["delta_corr_vs_causal"],
            row["delta_corr_vs_base"],
            row["delta_joint_vs_causal"],
        )
    if bucket == "not_working":
        # More negative = stronger regression evidence.
        return (
            -row["delta_corr_vs_causal"],
            -row["delta_loc_vs_causal"],
            -row["delta_joint_vs_causal"],
        )
    # For neutral rows, prefer near-zero change.
    return (
        -(abs(row["delta_corr_vs_causal"]) + abs(row["delta_corr_vs_base"])),
        -abs(row["delta_loc_vs_causal"]),
        -abs(row["delta_joint_vs_causal"]),
    )


def pop_first_with_model(rows: List[Dict], allowed_models: Sequence[str]) -> Optional[Dict]:
    for i, row in enumerate(rows):
        if row["model"] in allowed_models:
            return rows.pop(i)
    return None


def pop_first(rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None
    return rows.pop(0)


def choose_bucket_rows(
    all_rows: List[Dict],
    bucket: str,
    n_target: int,
    rng: random.Random,
) -> List[Dict]:
    pool = [r for r in all_rows if r.get("bucket") == bucket]
    pool.sort(key=lambda r: row_sort_key(r, bucket), reverse=True)
    if not pool or n_target <= 0:
        return []

    by_split: Dict[str, List[Dict]] = defaultdict(list)
    for row in pool:
        by_split[row["split"]].append(row)

    split_order = sorted(by_split.keys(), key=lambda s: len(by_split[s]), reverse=True)
    for split in split_order:
        rng.shuffle(by_split[split][: min(3, len(by_split[split]))])

    selected: List[Dict] = []
    selected_models = set()
    selected_by_split = Counter()

    # Seed with one per split when possible.
    for split in split_order:
        if len(selected) >= n_target:
            break
        candidate = pop_first(by_split[split])
        if candidate is None:
            continue
        selected.append(candidate)
        selected_models.add(candidate["model"])
        selected_by_split[candidate["split"]] += 1

    # Add rows with unseen models first for diversity.
    remaining_models = sorted({r["model"] for r in pool} - selected_models)
    if remaining_models and len(selected) < n_target:
        for model in remaining_models:
            if len(selected) >= n_target:
                break
            for split in split_order:
                candidate = pop_first_with_model(by_split[split], [model])
                if candidate is not None:
                    selected.append(candidate)
                    selected_models.add(candidate["model"])
                    selected_by_split[candidate["split"]] += 1
                    break

    # Fill remainder by favoring currently underrepresented splits.
    while len(selected) < n_target:
        available_splits = [s for s in split_order if by_split[s]]
        if not available_splits:
            break
        available_splits.sort(key=lambda s: selected_by_split[s])
        split = available_splits[0]
        candidate = pop_first(by_split[split])
        if candidate is None:
            break
        selected.append(candidate)
        selected_models.add(candidate["model"])
        selected_by_split[candidate["split"]] += 1

    return selected


def classify_contrast_stratum(row: Dict) -> str:
    eps = 1e-12
    score_causal = float(row.get("score_causal", 0.0))
    score_corr = float(row.get("score_corr", 0.0))
    dcb = score_corr - score_causal

    if dcb > eps:
        return "corr_strong"
    if dcb < -eps:
        return "causal_strong"
    if abs(dcb) <= eps:
        return "all_tie"
    return "mixed_other"


def select_by_strata(rows: List[Dict], quotas: Dict[str, int], rng: random.Random) -> List[Dict]:
    by_stratum: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        s = classify_contrast_stratum(r)
        by_stratum[s].append(r)

    selected: List[Dict] = []
    used_ids = set()
    for stratum, target in quotas.items():
        pool = by_stratum.get(stratum, [])
        # Rank stronger evidence first within each stratum.
        if stratum == "causal_strong":
            pool.sort(key=lambda r: float(r.get("delta_corr_vs_causal", 0.0)))
        elif stratum == "corr_strong":
            pool.sort(key=lambda r: abs(float(r.get("delta_corr_vs_causal", 0.0))), reverse=True)
        else:
            pool.sort(key=lambda r: abs(float(r.get("delta_corr_vs_causal", 0.0))))
        take = 0
        for r in pool:
            tid = r["trace_id"]
            if tid in used_ids:
                continue
            selected.append(r)
            used_ids.add(tid)
            take += 1
            if take >= target:
                break

    return selected


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def variant_path(repo_root: Path, model: str, split: str, trace_id: str, variant: str) -> Path:
    if variant == "baseline":
        return (
            repo_root
            / "benchmarking/outputs/zero_shot/compressed"
            / f"outputs_{model}-{split}"
            / f"{trace_id}.json"
        )
    if variant == "causal":
        return (
            repo_root
            / "benchmarking/outputs/zero_shot/compressed"
            / f"outputs_{model}-{split}-graph_inject_causal_only_span_index"
            / f"{trace_id}.json"
        )
    if variant == "corr":
        return (
            repo_root
            / "benchmarking/outputs_thres/t0.35"
            / f"outputs_{model}-{split}-graph_inject_causal_corr0.35_span_index"
            / f"{trace_id}.json"
        )
    if variant == "gold":
        gt_dir = "processed_annotations_gaia" if split == "GAIA_dedup" else "processed_annotations_swe_bench"
        return repo_root / "benchmarking" / gt_dir / f"{trace_id}.json"
    raise ValueError(f"Unknown variant: {variant}")


def raw_trace_path(repo_root: Path, split: str, trace_id: str) -> Path:
    split_dir = "GAIA_dedup" if split == "GAIA_dedup" else "SWE_Bench_dedup"
    return repo_root / "benchmarking" / "data" / split_dir / f"{trace_id}.json"


def _walk_for_prompt(obj: object) -> Optional[str]:
    if isinstance(obj, dict):
        for key in ("question", "task", "issue"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.replace("\n", " ").replace("\r", " ").strip()[:260]
        for value in obj.values():
            found = _walk_for_prompt(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _walk_for_prompt(item)
            if found:
                return found
    elif isinstance(obj, str):
        text = obj.replace("\n", " ").replace("\r", " ").strip()
        if "New task:" in text:
            snippet = text.split("New task:", 1)[1].strip()
            if "Here is the task:" in snippet:
                snippet = snippet.split("Here is the task:", 1)[1].strip()
            return snippet[:260]
        if "<issue>" in text:
            return text.split("<issue>", 1)[1].strip()[:260]
        if "Here is the task:" in text:
            return text.split("Here is the task:", 1)[1].strip()[:260]
    return None


def extract_question_snippet(repo_root: Path, split: str, trace_id: str, *objs: object) -> str:
    # 1) try all provided structured objects
    for obj in objs:
        snippet = _walk_for_prompt(obj)
        if snippet:
            return snippet

    # 2) parse from raw trace json where task often lives in span_attributes.input.value
    try:
        trace_obj = load_json(raw_trace_path(repo_root, split, trace_id))
        snippet = _walk_for_prompt(trace_obj)
        if snippet:
            return snippet
    except Exception:
        pass

    # 3) fallback to first meaningful line from any discovered string, but clip aggressively
    for obj in objs:
        if isinstance(obj, dict):
            text = json.dumps(obj, ensure_ascii=False)
            if "You have one question to answer." in text:
                return "You have one question to answer. (full prompt omitted)"
    return "(snippet unavailable)"


def format_errors(obj: Dict) -> str:
    errors = obj.get("errors", [])
    if not isinstance(errors, list) or not errors:
        return "-"
    out = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        category = str(err.get("category", "")).strip() or "UNK"
        location = str(err.get("location", "")).strip() or "UNK"
        out.append(f"{category}@{location}")
    return " | ".join(out) if out else "-"


def format_error_evidence(obj: Dict, max_items: int = 3, max_chars: int = 380) -> str:
    errors = obj.get("errors", [])
    if not isinstance(errors, list) or not errors:
        return "-"
    parts = []
    for err in errors[:max_items]:
        if not isinstance(err, dict):
            continue
        cat = str(err.get("category", "")).strip() or "UNK"
        ev = str(err.get("evidence", "")).replace("\n", " ").replace("\r", " ").strip()
        desc = str(err.get("description", "")).replace("\n", " ").replace("\r", " ").strip()
        # Keep concise but informative.
        ev = ev[:120] + ("..." if len(ev) > 120 else "")
        desc = desc[:120] + ("..." if len(desc) > 120 else "")
        parts.append(f"{cat}: ev={ev}; desc={desc}")
    out = " | ".join(parts) if parts else "-"
    return out[:max_chars] + ("..." if len(out) > max_chars else "")


def taxonomy_overlap(gold: Dict, pred: Dict) -> Dict[str, object]:
    gold_cats = extract_leaf_categories(gold.get("errors", []))
    pred_cats = extract_leaf_categories(pred.get("errors", []))
    gset = set(gold_cats)
    pset = set(pred_cats)
    return {
        "gold_cats": gold_cats,
        "pred_cats": pred_cats,
        "tp": sorted(gset & pset),
        "fp": sorted(pset - gset),
        "fn": sorted(gset - pset),
    }


def enrich_selected(repo_root: Path, selected: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    enriched: List[Dict] = []
    stats = Counter()
    for idx, row in enumerate(selected, start=1):
        model = row["model"]
        split = row["split"]
        trace_id = row["trace_id"]

        try:
            gold = load_json(variant_path(repo_root, model, split, trace_id, "gold"))
            baseline = load_json(variant_path(repo_root, model, split, trace_id, "baseline"))
            causal = load_json(variant_path(repo_root, model, split, trace_id, "causal"))
            corr = load_json(variant_path(repo_root, model, split, trace_id, "corr"))
        except json.JSONDecodeError:
            stats["json_decode_error"] += 1
            continue
        except Exception:
            stats["read_error"] += 1
            continue

        snippet = extract_question_snippet(repo_root, split, trace_id, gold, baseline, causal, corr)
        # Rule-based mechanism bucket (deterministic; performance-gain driven).
        eps = 1e-12
        d_corr_vs_causal = float(row.get("delta_corr_vs_causal", 0.0))
        score_base = float(row.get("score_base", 0.0))
        score_causal = float(row.get("score_causal", 0.0))
        score_corr = float(row.get("score_corr", 0.0))
        causal_gain_vs_base = score_causal - score_base
        corr_gain_vs_base = score_corr - score_base

        if d_corr_vs_causal > eps:
            rule_bucket = "corr-added-gain"
            rule_reason = "corr score > causal score"
        elif d_corr_vs_causal < -eps:
            rule_bucket = "corr-induced-harm"
            rule_reason = "corr score < causal score"
        else:
            if causal_gain_vs_base > eps and corr_gain_vs_base > eps:
                rule_bucket = "causal-backed-gain"
                rule_reason = "corr ~= causal and both beat baseline"
            elif abs(causal_gain_vs_base) <= eps and abs(corr_gain_vs_base) <= eps:
                rule_bucket = "shared-failure"
                rule_reason = "corr ~= causal ~= baseline"
            else:
                rule_bucket = "causal-preserving-neutral"
                rule_reason = "corr ~= causal with mixed baseline relation"

        base_tax = taxonomy_overlap(gold, baseline)
        causal_tax = taxonomy_overlap(gold, causal)
        corr_tax = taxonomy_overlap(gold, corr)

        enriched.append(
            {
                "sample_id": idx,
                "trace_id": trace_id,
                "split": split,
                "model": model,
                "bucket_from_metrics": row["bucket"],
                "rule_based_mechanism_bucket": rule_bucket,
                "rule_based_bucket_reason": rule_reason,
                "suggested_mechanism_bucket": rule_bucket,
                "delta_corr_vs_causal": row["delta_corr_vs_causal"],
                "delta_corr_vs_base": row["delta_corr_vs_base"],
                "delta_wf1_vs_causal": row["delta_wf1_vs_causal"],
                "delta_loc_vs_causal": row["delta_loc_vs_causal"],
                "delta_joint_vs_causal": row["delta_joint_vs_causal"],
                "delta_wf1_vs_base": row["delta_wf1_vs_base"],
                "delta_loc_vs_base": row["delta_loc_vs_base"],
                "delta_joint_vs_base": row["delta_joint_vs_base"],
                "gold_errors": format_errors(gold),
                "baseline_errors": format_errors(baseline),
                "causal_errors": format_errors(causal),
                "corr_errors": format_errors(corr),
                "gold_evidence_excerpt": format_error_evidence(gold),
                "baseline_evidence_excerpt": format_error_evidence(baseline),
                "causal_evidence_excerpt": format_error_evidence(causal),
                "corr_evidence_excerpt": format_error_evidence(corr),
                "gold_leaf_categories": "|".join(base_tax["gold_cats"]),
                "baseline_leaf_categories": "|".join(base_tax["pred_cats"]),
                "causal_leaf_categories": "|".join(causal_tax["pred_cats"]),
                "corr_leaf_categories": "|".join(corr_tax["pred_cats"]),
                "baseline_tp_leaf": "|".join(base_tax["tp"]),
                "baseline_fp_leaf": "|".join(base_tax["fp"]),
                "baseline_fn_leaf": "|".join(base_tax["fn"]),
                "causal_tp_leaf": "|".join(causal_tax["tp"]),
                "causal_fp_leaf": "|".join(causal_tax["fp"]),
                "causal_fn_leaf": "|".join(causal_tax["fn"]),
                "corr_tp_leaf": "|".join(corr_tax["tp"]),
                "corr_fp_leaf": "|".join(corr_tax["fp"]),
                "corr_fn_leaf": "|".join(corr_tax["fn"]),
                "question_snippet": snippet,
                # Annotation fields (left blank on purpose).
                "mechanism_bucket": "",
                "pattern_tags": "",
                "pattern_characteristics": "",
                "corr_edge_role": "",
                "impact_severity": "",
                "confidence": "",
                "evidence_note": "",
                "annotator": "",
            }
        )
    return enriched, dict(stats)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_diversity(rows: Iterable[Dict]) -> int:
    return len({r["model"] for r in rows})


def split_counts(rows: Iterable[Dict]) -> Dict[str, int]:
    c = Counter()
    for row in rows:
        c[row["split"]] += 1
    return dict(c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark"),
    )
    parser.add_argument(
        "--input_summary_csv",
        type=Path,
        default=Path(
            "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_b_case_study/results/instance_comparison_summary.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_working", type=int, default=22)
    parser.add_argument("--n_not_working", type=int, default=10)
    parser.add_argument("--n_neutral", type=int, default=6)
    parser.add_argument("--min_models", type=int, default=3)
    parser.add_argument("--sampling_mode", choices=["legacy", "contrast"], default="contrast")
    parser.add_argument("--contrast_n_per_group", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows_raw = load_csv_rows(args.input_summary_csv)
    rows = normalize_row_types(rows_raw)

    if args.sampling_mode == "legacy":
        selected = []
        selected.extend(choose_bucket_rows(rows, "working", args.n_working, rng))
        selected.extend(choose_bucket_rows(rows, "not_working", args.n_not_working, rng))
        selected.extend(choose_bucket_rows(rows, "neutral", args.n_neutral, rng))
    else:
        # Simplified contrast-stratified quotas for causal-vs-corr narrative.
        quotas = {
            "causal_strong": args.contrast_n_per_group,
            "corr_strong": args.contrast_n_per_group,
            "all_tie": args.contrast_n_per_group,
        }
        selected = select_by_strata(rows, quotas=quotas, rng=rng)

    # Keep deterministic ordering by evidence strength for easier annotation flow.
    selected.sort(
        key=lambda r: (
            0 if r["bucket"] == "working" else 1 if r["bucket"] == "not_working" else 2,
            -abs(r["delta_corr_vs_causal"]),
            r["model"],
            r["split"],
        )
    )

    enriched, read_stats = enrich_selected(args.repo_root, selected)

    audit_csv = args.out_dir / "trail_audit_sheet.csv"
    candidates_json = args.out_dir / "trail_audit_candidates.json"
    manifest_json = args.out_dir / "trail_sampling_manifest.json"
    schema_paths = write_schema_files(args.out_dir)
    write_csv(audit_csv, enriched)
    with candidates_json.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    manifest = {
        "inputs": {
            "input_summary_csv": str(args.input_summary_csv),
            "seed": args.seed,
        },
        "targets": {
            "n_working": args.n_working,
            "n_not_working": args.n_not_working,
            "n_neutral": args.n_neutral,
            "n_total": (
                args.n_working + args.n_not_working + args.n_neutral
                if args.sampling_mode == "legacy"
                else 3 * args.contrast_n_per_group
            ),
            "sampling_mode": args.sampling_mode,
            "contrast_n_per_group": args.contrast_n_per_group,
        },
        "selected_counts": {
            "n_selected_raw": len(selected),
            "n_selected_enriched": len(enriched),
            "by_bucket": dict(Counter(r["bucket_from_metrics"] for r in enriched)),
            "by_split": split_counts(enriched),
            "model_diversity": model_diversity(enriched),
            "meets_min_models": model_diversity(enriched) >= args.min_models,
            "contrast_strata_counts": dict(Counter(classify_contrast_stratum(r) for r in selected)),
        },
        "read_stats": read_stats,
        "outputs": {
            "trail_audit_sheet_csv": str(audit_csv),
            "trail_audit_candidates_json": str(candidates_json),
            **schema_paths,
        },
        "notes": [
            "Uses existing outputs only; no new experiments.",
            "Audit labels are intentionally left blank for manual annotation.",
            "Use mechanism_bucket as the primary label: causal-backed-gain / corr-added-gain / causal-preserving-neutral / corr-induced-harm / shared-failure.",
            "If model diversity is below threshold, rerun with different seed or bucket sizes.",
        ],
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
