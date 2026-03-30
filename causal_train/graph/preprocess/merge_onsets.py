"""
merge_onsets.py — Merge two or more onset JSONL files into one.

Used to combine GAIA-train onsets and SWE-bench-train onsets before
running the Suppes / CAPRI pipeline on the joint training set.

Checks for duplicate trace IDs and warns if found.

Usage (from trail-benchmark/):
    python causal_train/graph/preprocess/merge_onsets.py \
        --inputs  benchmarking/data/trail_derived/onsets_gaia_train.jsonl \
                  benchmarking/data/trail_derived/onsets_swe_train.jsonl \
        --out_path benchmarking/data/trail_derived/onsets_combined_train.jsonl
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Merge multiple onset JSONL files")
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="Two or more onset JSONL files to merge (e.g. onsets_gaia_train.jsonl onsets_swe_train.jsonl)",
    )
    ap.add_argument(
        "--out_path", required=True,
        help="Output merged JSONL path",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    seen_ids: set[str] = set()
    all_records: list[dict] = []

    for path in args.inputs:
        n_before = len(all_records)
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                tid = rec.get("trace_id", "")
                if tid in seen_ids:
                    print(f"  WARNING: duplicate trace_id '{tid}' in {path} — skipping")
                    continue
                seen_ids.add(tid)
                all_records.append(rec)
        print(f"  {path}: +{len(all_records) - n_before} traces")

    with open(args.out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_records)} traces -> {args.out_path}")


if __name__ == "__main__":
    main()
