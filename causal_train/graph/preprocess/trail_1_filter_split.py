"""
Step 1: Build trace + annotation pairs (TRAIL by split: GAIA or SWE Bench).

Outputs a jsonl where each line is:
  {trace_id, split, trace_path, annotation_path, n_errors, error_locations}

error_locations = list of span_ids that appear in annotations (for coverage check in Step 2).

Usage (run from benchmarking/):
  python causal_explore/preprocess/trail_1_filter_split.py --data_dir data --out_dir data/trail_filtered
  python causal_explore/preprocess/trail_1_filter_split.py --split GAIA --out_path data/trail_filtered/gaia.jsonl
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Build TRAIL trace + annotation pairs with metadata")
    ap.add_argument("--data_dir", default="data", help="Directory containing GAIA/ and SWE Bench/ trace dirs")
    ap.add_argument("--annotation_dir", default=None, help="If set, annotation dirs (processed_annotations_*) are under this path; else under cwd")
    ap.add_argument("--out_dir", default="data/trail_filtered", help="Output directory for filtered jsonl")
    ap.add_argument("--split", choices=["GAIA", "SWE Bench", "both"], default="both",
                    help="Which split(s) to output")
    ap.add_argument("--out_path", default=None,
                    help="If set, single output file (overrides out_dir for one split)")
    ap.add_argument("--include_ids", default=None,
                    help="Path to a JSON file containing a list of trace IDs to include. "
                         "If not set, all traces with annotations are included.")
    args = ap.parse_args()

    # Load allowed trace IDs if provided
    include_ids = None
    if args.include_ids:
        with open(args.include_ids, "r") as f:
            include_ids = set(json.load(f))
        print(f"Restricting to {len(include_ids)} trace IDs from {args.include_ids}")

    os.makedirs(args.out_dir, exist_ok=True)

    splits_config = [
        ("GAIA", "GAIA", "processed_annotations_gaia"),
        ("SWE Bench", "SWE Bench", "processed_annotations_swe_bench"),
    ]

    for split_name, trace_subdir, ann_subdir in splits_config:
        if args.split != "both" and args.split != split_name:
            continue

        trace_dir = os.path.join(args.data_dir, trace_subdir)
        ann_dir = os.path.join(args.annotation_dir, ann_subdir) if args.annotation_dir else ann_subdir
        if not os.path.isdir(trace_dir):
            print(f"Skipping {split_name}: trace dir not found: {trace_dir}")
            continue
        if not os.path.isdir(ann_dir):
            print(f"Skipping {split_name}: annotation dir not found: {ann_dir}")
            continue

        trace_files = [f for f in os.listdir(trace_dir) if f.endswith(".json")]
        records = []
        n_skipped_ids = 0
        for f in trace_files:
            trace_id = os.path.splitext(f)[0]
            if include_ids is not None and trace_id not in include_ids:
                n_skipped_ids += 1
                continue
            trace_path = os.path.join(trace_dir, f)
            annotation_path = os.path.join(ann_dir, f)
            if not os.path.isfile(annotation_path):
                continue
            n_errors = 0
            error_locations = []
            try:
                with open(annotation_path, "r") as af:
                    ann = json.load(af)
                errors = ann.get("errors") or []
                n_errors = len(errors)
                error_locations = sorted(set((e.get("location") or "").strip() for e in errors if (e.get("location") or "").strip()))
            except (json.JSONDecodeError, OSError):
                pass
            records.append({
                "trace_id": trace_id,
                "split": split_name,
                "trace_path": trace_path,
                "annotation_path": annotation_path,
                "n_errors": n_errors,
                "error_locations": error_locations,
            })

        if args.out_path and args.split == split_name:
            out_path = args.out_path
        else:
            suffix = "gaia" if split_name == "GAIA" else "swe_bench"
            out_path = os.path.join(args.out_dir, f"{suffix}.jsonl")

        with open(out_path, "w") as of:
            for r in records:
                of.write(json.dumps(r, ensure_ascii=False) + "\n")

        skipped_msg = f" ({n_skipped_ids} excluded by --include_ids)" if include_ids is not None else ""
        print(f"✓ {split_name}: {len(records)} traces -> {out_path}{skipped_msg}")

    print("Done.")


if __name__ == "__main__":
    main()
