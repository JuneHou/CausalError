"""
Step 3: Build onset table from annotations + span order (candidate timeline).

present[A], onset[A] (min rank per category), count[A], and optional ties (categories sharing same span).
REM_MAST-compatible.

Usage (run from benchmarking/):
  python causal_explore/preprocess/trail_3_build_onsets.py --filtered_path data/trail_filtered/gaia.jsonl --span_order_path data/trail_span_order/gaia.jsonl --out_path data/trail_derived/onsets_gaia.jsonl
"""

import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description="Build onset table from TRAIL annotations + span order")
    ap.add_argument("--filtered_path", required=True, help="jsonl from trail_1")
    ap.add_argument("--span_order_path", required=True, help="jsonl from trail_2 (trace_id, span_rank)")
    ap.add_argument("--out_path", required=True, help="Output onsets jsonl")
    ap.add_argument("--include_ties", action="store_true", default=True, help="Include ties (categories sharing same span)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    # Load filtered list (trace_id -> annotation_path, split)
    with open(args.filtered_path, "r") as f:
        filtered = {json.loads(line)["trace_id"]: json.loads(line) for line in f}

    # Load span order (trace_id -> span_rank)
    span_orders = {}
    with open(args.span_order_path, "r") as f:
        for line in f:
            r = json.loads(line)
            span_orders[r["trace_id"]] = r["span_rank"]

    # Collect all categories and per-trace (category, span_id) pairs
    all_categories = set()
    trace_errors = {}  # trace_id -> [(category, span_id), ...]
    for trace_id, rec in filtered.items():
        ann_path = rec["annotation_path"]
        if not os.path.isfile(ann_path):
            continue
        try:
            with open(ann_path, "r") as af:
                ann = json.load(af)
        except json.JSONDecodeError as err:
            print(f"Skip {trace_id}: invalid JSON in {ann_path} ({err})")
            continue
        errors = ann.get("errors") or []
        pairs = []
        for e in errors:
            cat = (e.get("category") or "").strip()
            loc = (e.get("location") or "").strip()
            if cat and loc:
                all_categories.add(cat)
                pairs.append((cat, loc))
        trace_errors[trace_id] = pairs

    all_categories = sorted(all_categories)

    with open(args.out_path, "w") as of:
        for trace_id, rec in filtered.items():
            if trace_id not in trace_errors:
                continue
            split = rec.get("split", "GAIA")
            span_rank = span_orders.get(trace_id) or {}
            pairs = trace_errors[trace_id]

            # First occurrence of each category -> earliest rank; count per category
            category_ranks = defaultdict(list)
            category_count = defaultdict(int)
            for cat, span_id in pairs:
                category_count[cat] += 1
                r = span_rank.get(span_id)
                if r is not None:
                    category_ranks[cat].append(r)
            onset = {cat: min(ranks) for cat, ranks in category_ranks.items() if ranks}

            # Present: 1 if category appears in this trace (in onset), else 0
            present_cats = set(onset.keys())
            present = {cat: 1 if cat in present_cats else 0 for cat in all_categories}

            # count: occurrences per category (only for categories present in trace)
            count = {cat: category_count.get(cat, 0) for cat in all_categories}

            # ties: pairs of distinct categories that share at least one span_id
            span_to_cats = defaultdict(set)
            for cat, span_id in pairs:
                if span_rank.get(span_id) is not None:
                    span_to_cats[span_id].add(cat)
            ties = []
            seen_pair = set()
            for cats in span_to_cats.values():
                if len(cats) < 2:
                    continue
                for a in cats:
                    for b in cats:
                        if a < b and (a, b) not in seen_pair:
                            seen_pair.add((a, b))
                            ties.append([a, b])

            out_row = {
                "trace_id": trace_id,
                "split": split,
                "present": present,
                "onset": onset,
                "count": count,
            }
            if args.include_ties and ties:
                out_row["ties"] = ties
            of.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"✓ Wrote onsets for {len(filtered)} traces -> {args.out_path}")
    print(f"  Categories: {len(all_categories)}")


if __name__ == "__main__":
    main()
