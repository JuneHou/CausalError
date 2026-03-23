"""
Step 2: Build candidate-span timeline (Option 2A: OpenInference LLM/TOOL/CHAIN).

Candidate spans = those with span_attributes["openinference.span.kind"] in {"LLM","TOOL","CHAIN"}.
Order by timestamp, tie-break by span_id. Output span_rank and missing_annotated_span_ids.
When any annotated span_id is missing from candidates, fallback = all-spans ranking for that trace.

Usage (run from benchmarking/):
  python causal_explore/preprocess/trail_2_build_span_order.py --filtered_path data/trail_filtered/gaia.jsonl --out_path data/trail_span_order/gaia.jsonl
"""

import argparse
import json
import os
from datetime import datetime

CANDIDATE_KINDS = {"LLM", "TOOL", "CHAIN"}


def parse_ts(ts):
    """Parse ISO timestamp; return sortable value. Missing/invalid -> epoch."""
    if not ts:
        return datetime.min.isoformat()
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        return datetime.min.isoformat()


def _is_candidate_span(span):
    """True if span is an OpenInference action span (LLM, TOOL, or CHAIN)."""
    attrs = span.get("span_attributes") or span.get("attributes")
    if not isinstance(attrs, dict):
        return False
    kind = attrs.get("openinference.span.kind")
    return kind in CANDIDATE_KINDS


def collect_spans_from_trace(trace, candidates_only=True):
    """
    Collect spans from trace. If candidates_only=True, only spans with
    openinference.span.kind in {LLM, TOOL, CHAIN}. Otherwise all spans with span_id and timestamp.
    Returns list of {"span_id", "timestamp"}.
    """
    out = []
    seen = set()
    trace_id = trace.get("trace_id") if isinstance(trace, dict) else None

    def visit(span):
        if not isinstance(span, dict):
            return
        sid = span.get("span_id")
        ts = span.get("timestamp")
        if sid is None or ts is None:
            pass
        else:
            if candidates_only and not _is_candidate_span(span):
                pass
            else:
                key = (trace_id, sid)
                if key not in seen:
                    seen.add(key)
                    out.append({"span_id": sid, "timestamp": ts})
        children = span.get("child_spans")
        if isinstance(children, list):
            for child in children:
                visit(child)

    for root in trace.get("spans") or []:
        visit(root)

    return out


def build_span_rank(spans):
    """Sort by timestamp, tie-break by span_id; return {span_id: rank}."""
    if not spans:
        return {}
    spans = sorted(spans, key=lambda x: (parse_ts(x["timestamp"]), x["span_id"]))
    return {s["span_id"]: i for i, s in enumerate(spans)}


def main():
    ap = argparse.ArgumentParser(description="Build candidate-span timeline with coverage report")
    ap.add_argument("--filtered_path", required=True, help="Path to jsonl from trail_1 (e.g. data/trail_filtered/gaia.jsonl)")
    ap.add_argument("--out_path", required=True, help="Output jsonl path (e.g. data/trail_span_order/gaia.jsonl)")
    ap.add_argument("--fallback_all_spans", action="store_true", default=True,
                    help="When annotated span_ids are missing from candidates, use all-spans ranking for that trace (default: True)")
    args = ap.parse_args()

    if os.path.isdir(args.filtered_path):
        raise SystemExit(
            f"filtered_path must be a .jsonl file, not a directory: {args.filtered_path}\n"
            "Example: --filtered_path data/trail_filtered/gaia.jsonl --out_path data/trail_span_order/gaia.jsonl"
        )
    if os.path.isdir(args.out_path):
        raise SystemExit(f"out_path must be a file path, not a directory: {args.out_path}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    with open(args.filtered_path, "r") as f:
        filtered = [json.loads(line) for line in f]

    total_candidates = 0
    traces_with_missing = 0
    with open(args.out_path, "w") as of:
        for rec in filtered:
            trace_id = rec["trace_id"]
            trace_path = rec["trace_path"]
            split = rec.get("split", "GAIA")
            error_locations = set(rec.get("error_locations") or [])

            if not os.path.isfile(trace_path):
                print(f"Skip {trace_id}: file not found {trace_path}")
                continue

            with open(trace_path, "r") as tf:
                trace = json.load(tf)

            candidate_spans = collect_spans_from_trace(trace, candidates_only=True)
            span_rank = build_span_rank(candidate_spans)
            missing = [sid for sid in error_locations if sid not in span_rank]

            if missing and args.fallback_all_spans:
                all_spans = collect_spans_from_trace(trace, candidates_only=False)
                span_rank = build_span_rank(all_spans)
                missing = [sid for sid in error_locations if sid not in span_rank]

            if missing:
                traces_with_missing += 1

            total_candidates += len(span_rank)
            of.write(json.dumps({
                "trace_id": trace_id,
                "split": split,
                "span_rank": span_rank,
                "missing_annotated_span_ids": missing,
            }, ensure_ascii=False) + "\n")

    print(f"✓ Processed {len(filtered)} traces -> {args.out_path}")
    print(f"  Total span ranks (candidates or fallback all-spans): {total_candidates}")
    if traces_with_missing:
        print(f"  Traces with still-missing annotated span_ids after fallback: {traces_with_missing}")


if __name__ == "__main__":
    main()
