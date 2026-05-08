"""One-off: rewrite QwenLong W2 prediction categories to canonical TRAIL leaf names.

QwenLong-L1-32B emits hierarchical category paths with mixed separators:
  - "Information Processing -> Tool Output Misinterpretation"
  - "System Execution Errors > Configuration > Tool Definition Issues"
  - "Configuration/Tool Definition Issues"
  - "System Execution Errors.Configuration.Environment Setup Errors"

calculate_scores.py:normalize_category does (a) exact then (b) `predicted ⊆ canonical`
substring match. Both fail on these because the predicted contains separators that don't
appear in canonical leaf names. Result: joint accuracy = 0 even when leaf is correct.

This script splits each predicted category on common hierarchy separators, takes the
last token, and matches it (case/space insensitive) against the 19 TRAIL leaf names.
On match, it overwrites with the canonical leaf string. On no match, it leaves the
category as-is so downstream scoring still penalizes genuinely unmappable predictions.

Writes the rewritten files to a sibling directory `<input>_leafnorm/`. Does NOT modify
the input directory or calculate_scores.py.

Usage:
    python _normalize_qwenlong_w2_categories.py \\
        --input  baselines/outputs/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-SWE_Bench_dedup-who_and_when_w2

Then re-score:
    python benchmarking/eval/calculate_scores.py \\
        --pred_dir baselines/outputs/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-SWE_Bench_dedup-who_and_when_w2_leafnorm
"""

import argparse
import json
import os
import re
import shutil
import sys

TAXONOMY_LEAF_CATEGORIES = [
    "Language-only", "Tool-related", "Poor Information Retrieval",
    "Tool Output Misinterpretation", "Incorrect Problem Identification",
    "Tool Selection Errors", "Formatting Errors", "Instruction Non-compliance",
    "Tool Definition Issues", "Environment Setup Errors", "Rate Limiting",
    "Authentication Errors", "Service Errors", "Resource Not Found",
    "Resource Exhaustion", "Timeout Issues", "Context Handling Failures",
    "Resource Abuse", "Goal Deviation", "Task Orchestration",
]

_LEAF_LOOKUP = {c.lower().replace(" ", ""): c for c in TAXONOMY_LEAF_CATEGORIES}

_SEP = re.compile(r"\s*(?:->|>|/|::|\\|:|\(|\))\s*| - |\.(?=[A-Z])")


def normalize_to_leaf(cat: str) -> tuple:
    """Return (canonical_leaf, hit) — hit=True if mapped to a canonical TRAIL leaf."""
    if not cat:
        return cat, False
    parts = [p.strip() for p in _SEP.split(cat) if p and p.strip()]
    if not parts:
        return cat, False
    leaf = parts[-1]
    key = leaf.lower().replace(" ", "")
    if key in _LEAF_LOOKUP:
        return _LEAF_LOOKUP[key], True
    return cat, False


def process_dir(input_dir: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    stats = {"files": 0, "errors": 0, "rewrites": 0, "unchanged": 0, "unmapped": 0}
    unmapped_samples = []

    for fname in files:
        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)

        for err in data.get("errors", []):
            stats["errors"] += 1
            original = err.get("category", "")
            new_cat, hit = normalize_to_leaf(original)
            if hit and new_cat != original:
                err["category"] = new_cat
                stats["rewrites"] += 1
            elif hit:
                stats["unchanged"] += 1
            else:
                stats["unmapped"] += 1
                if len(unmapped_samples) < 10:
                    unmapped_samples.append(original)

        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        stats["files"] += 1

    if unmapped_samples:
        stats["unmapped_samples"] = unmapped_samples
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input prediction directory")
    ap.add_argument("--output", default=None,
                    help="Output directory (default: <input>_leafnorm)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting an existing output directory")
    args = ap.parse_args()

    input_dir = args.input.rstrip("/")
    output_dir = args.output or f"{input_dir}_leafnorm"

    if os.path.exists(output_dir):
        if not args.overwrite:
            print(f"refusing to overwrite existing {output_dir}; pass --overwrite", file=sys.stderr)
            sys.exit(2)
        shutil.rmtree(output_dir)

    stats = process_dir(input_dir, output_dir)

    print(f"input  : {input_dir}")
    print(f"output : {output_dir}")
    print(f"files processed : {stats['files']}")
    print(f"errors total    : {stats['errors']}")
    print(f"  rewritten     : {stats['rewrites']}  (mapped to canonical leaf, was different)")
    print(f"  unchanged     : {stats['unchanged']} (already canonical leaf)")
    print(f"  unmapped      : {stats['unmapped']}  (no canonical leaf found, left as-is)")
    if "unmapped_samples" in stats:
        print("unmapped sample categories:")
        for s in stats["unmapped_samples"]:
            print(f"  - {s!r}")


if __name__ == "__main__":
    main()
