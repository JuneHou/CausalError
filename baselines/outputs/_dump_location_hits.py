"""
Per-trace location + category breakdown for a prediction output dir.

Mirrors the metric computations in benchmarking/eval/calculate_scores.py:

  - location_accuracy = |gt_loc ∩ pred_loc| / |gt_loc|
  - joint_accuracy    = |gt_(loc,cat) ∩ pred_(loc,cat)| / |gt_(loc,cat)|
  - per-trace TP/FP/FN over the 21 leaf-category multi-label set,
    matching how calculate_scores.py aggregates per-category F1.

Category strings are normalized using the same rules as calculate_scores.py
(exact match, then substring match of the prediction inside a leaf name).
Predictions that fail to normalize to a known leaf — e.g. dotted taxonomy
paths like "Planning and Coordination Errors.Context Management.Resource Abuse"
— are surfaced in `unmapped_pred_categories` so generation-level issues
are visible.

Output JSON is written next to the script (baselines/outputs/) regardless
of where the prediction dir lives, named after the prediction dir.

Usage:
    python _dump_location_hits.py PRED_DIR
Split (gaia/swe_bench) is inferred from the dir name, matching
calculate_scores.py.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

ALL_CATEGORIES = [
    "Language-only", "Tool-related", "Poor Information Retrieval", "Incorrect Memory Usage",
    "Tool Output Misinterpretation", "Incorrect Problem Identification", "Tool Selection Errors",
    "Formatting Errors", "Instruction Non-compliance", "Tool Definition Issues",
    "Environment Setup Errors", "Rate Limiting", "Authentication Errors", "Service Errors",
    "Resource Not Found", "Resource Exhaustion", "Timeout Issues", "Context Handling Failures",
    "Resource Abuse", "Goal Deviation", "Task Orchestration",
]


def normalize_category(category: str) -> str:
    """Same logic as benchmarking/eval/calculate_scores.py::normalize_category."""
    if not category:
        return ""
    category = category.lower().strip()
    category_no_spaces = category.replace(" ", "")
    for std in ALL_CATEGORIES:
        if category == std.lower() or category_no_spaces == std.lower().replace(" ", ""):
            return std
    for std in ALL_CATEGORIES:
        if category_no_spaces in std.lower().replace(" ", ""):
            return std
    return category


def extract_json_from_text(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    s = m.group(0)
    while len(s) > 2:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s[:-1]
    raise ValueError("could not parse JSON")


def locations_of(obj) -> list[str]:
    return [e.get("location", "") for e in obj.get("errors", [])]


def loc_cat_pairs(obj) -> list[tuple[str, str, str]]:
    """Return list of (location, raw_category, normalized_category) per error."""
    out = []
    for e in obj.get("errors", []):
        loc = e.get("location", "")
        raw = e.get("category", "") or ""
        out.append((loc, raw, normalize_category(raw)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path,
                        help="Prediction output folder directory.")
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    name_lower = pred_dir.name.lower()
    if "swe_bench" in name_lower or "swe-bench" in name_lower or "swe bench" in name_lower:
        gt_dir = REPO / "benchmarking" / "processed_annotations_swe_bench"
    else:
        gt_dir = REPO / "benchmarking" / "processed_annotations_gaia"
    out_json = HERE / f"{pred_dir.name}-location_hits.json"

    rows = []
    for gt_path in sorted(glob.glob(str(gt_dir / "*.json"))):
        trace_id = os.path.splitext(os.path.basename(gt_path))[0]
        if trace_id.startswith("_meta_"):
            continue
        with open(gt_path) as f:
            gt = json.load(f)

        gt_pairs_full = loc_cat_pairs(gt)
        gt_locs = set(p[0] for p in gt_pairs_full)
        gt_joint = set((p[0], p[2]) for p in gt_pairs_full)
        gt_cats = set(p[2] for p in gt_pairs_full if p[2] in ALL_CATEGORIES)

        pred_path = pred_dir / f"{trace_id}.json"
        pred_pairs_full: list[tuple[str, str, str]] = []
        status = "ok"
        if not pred_path.exists():
            status = "missing_prediction"
        else:
            try:
                with open(pred_path) as f:
                    pred = extract_json_from_text(f.read())
                pred_pairs_full = loc_cat_pairs(pred)
            except Exception as e:
                status = f"parse_error: {e}"

        pred_locs = set(p[0] for p in pred_pairs_full)
        pred_joint = set((p[0], p[2]) for p in pred_pairs_full)
        pred_cats = set(p[2] for p in pred_pairs_full if p[2] in ALL_CATEGORIES)
        unmapped = sorted({p[1] for p in pred_pairs_full
                           if p[1] and p[2] not in ALL_CATEGORIES})

        loc_hit = gt_locs & pred_locs
        loc_acc = len(loc_hit) / len(gt_locs) if gt_locs else 0.0

        joint_hit = gt_joint & pred_joint
        joint_acc = len(joint_hit) / len(gt_joint) if gt_joint else 0.0

        cat_tp = gt_cats & pred_cats
        cat_fp = pred_cats - gt_cats
        cat_fn = gt_cats - pred_cats
        tp, fp, fn = len(cat_tp), len(cat_fp), len(cat_fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_pred_errors = []
        for loc, raw, norm in pred_pairs_full:
            if norm not in ALL_CATEGORIES:
                cat_status = "unmapped_category"
            elif (loc, norm) in gt_joint:
                cat_status = "TP_joint"
            elif norm in gt_cats:
                cat_status = "TP_category_only"
            else:
                cat_status = "FP"
            per_pred_errors.append({
                "location": loc,
                "raw_category": raw,
                "normalized_category": norm,
                "location_hit": loc in gt_locs,
                "joint_hit": (loc, norm) in gt_joint,
                "status": cat_status,
            })

        rows.append({
            "trace_id": trace_id,
            "status": status,
            "n_gt_locations": len(gt_locs),
            "n_pred_locations": len(pred_locs),
            "n_correct_locations": len(loc_hit),
            "location_accuracy": loc_acc,
            "joint_accuracy": joint_acc,
            "n_joint_hits": len(joint_hit),
            "cat_tp": tp,
            "cat_fp": fp,
            "cat_fn": fn,
            "cat_precision": precision,
            "cat_recall": recall,
            "cat_f1": f1,
            "gt_locations": sorted(gt_locs),
            "pred_locations": sorted(pred_locs),
            "correct_locations": sorted(loc_hit),
            "gt_categories": sorted(gt_cats),
            "pred_categories": sorted(pred_cats),
            "unmapped_pred_categories": unmapped,
            "joint_hits": sorted([f"{loc} | {cat}" for loc, cat in joint_hit]),
            "pred_errors": per_pred_errors,
        })

    rows.sort(key=lambda r: (-r["location_accuracy"], -r["joint_accuracy"],
                              -r["cat_f1"], r["trace_id"]))

    full = [r for r in rows if r["location_accuracy"] == 1.0 and r["n_gt_locations"] > 0]
    partial = [r for r in rows if 0 < r["location_accuracy"] < 1.0]
    missed = [r for r in rows if r["location_accuracy"] == 0.0]
    avg_loc = sum(r["location_accuracy"] for r in rows) / len(rows) if rows else 0.0
    avg_joint = sum(r["joint_accuracy"] for r in rows) / len(rows) if rows else 0.0
    avg_f1 = sum(r["cat_f1"] for r in rows) / len(rows) if rows else 0.0
    n_unmapped = sum(1 for r in rows if r["unmapped_pred_categories"])

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    with open(out_json, "w") as f:
        json.dump({
            "pred_dir": _rel(pred_dir),
            "gt_dir": _rel(gt_dir),
            "n_traces": len(rows),
            "average_location_accuracy": avg_loc,
            "average_joint_accuracy": avg_joint,
            "average_cat_f1_per_trace": avg_f1,
            "n_full_loc_match": len(full),
            "n_partial_loc_match": len(partial),
            "n_missed_loc": len(missed),
            "n_traces_with_unmapped_categories": n_unmapped,
            "traces": rows,
        }, f, indent=2)

    print(f"wrote {out_json}")
    print(f"avg loc acc   = {avg_loc:.4f}  (full={len(full)}, partial={len(partial)}, missed={len(missed)})")
    print(f"avg joint acc = {avg_joint:.4f}")
    print(f"avg per-trace cat F1 = {avg_f1:.4f}")
    print(f"traces with unmapped predicted categories: {n_unmapped}")


if __name__ == "__main__":
    main()
