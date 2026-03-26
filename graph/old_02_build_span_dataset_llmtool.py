#!/usr/bin/env python3
"""
02_build_span_dataset.py — Extract LLM/TOOL spans and map error annotations.

Encoding units are the LLM and TOOL action spans — the same level as annotation
location span_ids. These spans carry actual input.value / output.value.

For each trace:
  1. Load trace JSON + annotation JSON.
  2. DFS-collect all spans with openinference.span.kind = LLM or TOOL.
  3. Remove any LLM/TOOL span whose subtree contains another LLM/TOOL span
     (keep only innermost). This handles rare nesting (≈8 cases in the dataset)
     where a wrapper LLM/TOOL span encloses a child LLM/TOOL span; the child
     is the actual execution unit referenced by annotations.
  4. Sort surviving spans by start_time; assign step_index 1..K.
  5. Labels: annotation location = span_id → direct lookup (no parent-walk).
     Spans with no annotation mapping → Correct (labels=[]).

Output: graph/data/span_dataset.jsonl — one JSON record per trace.

Record schema:
  {
    "trace_id": str,
    "split":    "train"|"val"|"test",
    "dataset":  "GAIA"|"SWE-bench",
    "spans": [
      {
        "span_id":    str,
        "step_index": int,       // 1-based, start-time order
        "span_name":  str,
        "text":       str,       // formatted content of this action span
        "labels":     [str, ...], // error categories; [] = Correct
        "is_correct": bool
      }, ...\
    ],
    "n_spans":          int,
    "n_annotated_spans":int,
    "n_skipped_spans":  int      // LLM/TOOL spans with missing required fields
  }
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

GRAPH_DIR = Path(__file__).resolve().parent
BENCH_DIR = GRAPH_DIR.parent / "benchmarking"
sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import _span_name  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_DIR = GRAPH_DIR / "splits"
OUTPUT_DIR = GRAPH_DIR / "data"
GAIA_TRACE_DIR = BENCH_DIR / "data" / "GAIA"
SWE_TRACE_DIR  = BENCH_DIR / "data" / "SWE Bench"
GAIA_ANN_DIR   = BENCH_DIR / "processed_annotations_gaia"
SWE_ANN_DIR    = BENCH_DIR / "processed_annotations_swe_bench"
OUTPUT_FILE    = OUTPUT_DIR / "span_dataset.jsonl"

ACTION_KINDS = frozenset({"LLM", "TOOL"})

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_trace_file(trace_id: str) -> tuple[Optional[Path], Optional[str]]:
    for d, name in [(GAIA_TRACE_DIR, "GAIA"), (SWE_TRACE_DIR, "SWE-bench")]:
        p = d / f"{trace_id}.json"
        if p.exists():
            return p, name
    return None, None


def _find_annotation_file(trace_id: str) -> Optional[Path]:
    for d in (GAIA_ANN_DIR, SWE_ANN_DIR):
        p = d / f"{trace_id}.json"
        if p.exists():
            return p
    return None


def _span_kind(span: dict) -> str:
    return (span.get("span_attributes") or {}).get("openinference.span.kind", "")


def _parse_timestamp(span: dict) -> str:
    from datetime import datetime
    ts = span.get("start_time") or span.get("timestamp") or ""
    if not ts:
        return datetime.min.isoformat()
    s = str(ts).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        return datetime.min.isoformat()


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)


def _log_text(logs: list) -> str:
    """Text summary from log entries (function.name / function.output / severity)."""
    parts = []
    for entry in logs:
        body = entry.get("body") or {}
        if not isinstance(body, dict):
            continue
        fn_name = body.get("function.name") or ""
        fn_out  = body.get("function.output") or ""
        if not fn_name and not fn_out:
            continue
        lines = []
        sev = entry.get("severity_text") or ""
        if sev:
            lines.append(f"severity: {sev}")
        if fn_name:
            lines.append(f"function: {fn_name}")
        if fn_out:
            lines.append(f"output: {_to_str(fn_out)}")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Collect all LLM/TOOL spans (DFS), tracking which have LLM/TOOL descendants
# ---------------------------------------------------------------------------

def collect_action_spans_with_nesting(
    span: dict,
    result: list,
    has_action_descendant: set,
) -> bool:
    """
    DFS over span tree. Collects all LLM/TOOL spans into `result`.
    Returns True if this span or any descendant is a LLM/TOOL span.
    Populates has_action_descendant with span_ids of LLM/TOOL spans
    that have at least one LLM/TOOL descendant (these are "outer" wrappers).
    """
    kind = _span_kind(span)
    sid  = span.get("span_id")

    child_has_action = False
    for child in span.get("child_spans") or []:
        child_has_action |= collect_action_spans_with_nesting(
            child, result, has_action_descendant
        )

    is_action = kind in ACTION_KINDS
    if is_action:
        result.append(span)
        if child_has_action and sid:
            has_action_descendant.add(sid)

    return is_action or child_has_action


def get_innermost_action_spans(trace_data: dict) -> list[dict]:
    """
    Return all LLM/TOOL spans in the trace, excluding any whose subtree
    contains another LLM/TOOL span (keep only innermost/most-specific).
    Sorted by start_time.
    """
    all_action: list[dict] = []
    has_action_descendant: set = set()

    for root in trace_data.get("spans") or []:
        collect_action_spans_with_nesting(root, all_action, has_action_descendant)

    # Keep only innermost (those NOT in has_action_descendant)
    innermost = [s for s in all_action if s.get("span_id") not in has_action_descendant]

    # Sort by start_time
    innermost.sort(key=lambda s: (_parse_timestamp(s), s.get("span_id", "")))
    return innermost


# ---------------------------------------------------------------------------
# Build text for one action span (LLM or TOOL)
# ---------------------------------------------------------------------------

def _action_span_text(span: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (text, None) on success or (None, reason) if required fields missing.
    """
    attrs  = span.get("span_attributes") or {}
    kind   = _span_kind(span)
    name   = _span_name(span)
    in_val = attrs.get("input.value")
    out_val = attrs.get("output.value")

    if in_val is None:
        return None, f"missing input.value ({name!r})"
    if kind == "LLM" and out_val is None:
        return None, f"missing output.value on LLM span ({name!r})"

    sections = [f"[SPAN] {name}"]

    if kind == "TOOL":
        tool_name = attrs.get("tool.name") or name
        tool_desc = attrs.get("tool.description") or ""
        header = f"[TOOL] {tool_name}"
        if tool_desc:
            header += f" — {tool_desc}"
        sections.append(header)

    sections.append(f"[INPUT]\n{_to_str(in_val)}")
    if out_val is not None:
        sections.append(f"[OUTPUT]\n{_to_str(out_val)}")

    lt = _log_text(span.get("logs") or [])
    if lt:
        sections.append(f"[LOGS]\n{lt}")

    return "\n\n".join(sections), None


# ---------------------------------------------------------------------------
# Process one trace
# ---------------------------------------------------------------------------

def process_trace(trace_id: str, split: str) -> Optional[dict]:
    trace_path, dataset = _find_trace_file(trace_id)
    if trace_path is None:
        log.error("Trace file not found: %s", trace_id)
        return None
    ann_path = _find_annotation_file(trace_id)
    if ann_path is None:
        log.error("Annotation file not found: %s", trace_id)
        return None

    trace_data = _load_json(trace_path)
    trace_data["trace_id"] = trace_id
    ann_data = _load_json(ann_path)

    # Build annotation lookup: span_id → set of error categories
    ann_by_span: dict[str, set[str]] = defaultdict(set)
    ann_total = 0
    for err in ann_data.get("errors") or []:
        loc = err.get("location")
        cat = err.get("category")
        if not loc or not cat:
            continue
        ann_by_span[loc].add(cat)
        ann_total += 1

    # Collect innermost LLM/TOOL spans sorted by time
    action_spans = get_innermost_action_spans(trace_data)
    if not action_spans:
        log.warning("No LLM/TOOL spans in trace %s — skipping", trace_id)
        return None

    # Build per-span records
    span_records = []
    n_skipped = 0
    ann_matched = 0

    for i, span in enumerate(action_spans):
        sid = span.get("span_id")
        text, err = _action_span_text(span)
        if text is None:
            log.error(
                "Skipping span (trace=%s idx=%d %r): %s",
                trace_id, i + 1, _span_name(span), err,
            )
            n_skipped += 1
            continue

        labels = sorted(ann_by_span.get(sid, set()))
        if labels:
            ann_matched += len(ann_by_span[sid])

        span_records.append({
            "span_id":    sid,
            "step_index": len(span_records) + 1,
            "span_name":  _span_name(span),
            "text":       text,
            "labels":     labels,
            "is_correct": len(labels) == 0,
        })

    if not span_records:
        log.error("All spans skipped for trace %s", trace_id)
        return None

    if ann_total > 0 and ann_matched == 0:
        log.warning("trace %s: 0/%d annotations matched to any span", trace_id, ann_total)

    return {
        "trace_id":           trace_id,
        "split":              split,
        "dataset":            dataset,
        "spans":              span_records,
        "n_spans":            len(span_records),
        "n_annotated_spans":  sum(1 for s in span_records if not s["is_correct"]),
        "n_skipped_spans":    n_skipped,
        "_ann_matched":       ann_matched,
        "_ann_total":         ann_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, str] = {}
    for name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{name}_trace_ids.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run 01_make_splits.py first")
        for tid in _load_json(p):
            splits[tid] = name

    log.info("Processing %d traces (train=%d val=%d test=%d)",
             len(splits),
             sum(1 for v in splits.values() if v == "train"),
             sum(1 for v in splits.values() if v == "val"),
             sum(1 for v in splits.values() if v == "test"))

    records   = []
    n_failed  = 0
    ann_matched_total = 0
    ann_total_total   = 0

    for trace_id in sorted(splits):
        r = process_trace(trace_id, splits[trace_id])
        if r is None:
            n_failed += 1
            continue
        ann_matched_total += r.pop("_ann_matched")
        ann_total_total   += r.pop("_ann_total")
        records.append(r)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total_spans     = sum(r["n_spans"] for r in records)
    total_annotated = sum(r["n_annotated_spans"] for r in records)
    total_correct   = total_spans - total_annotated
    total_skipped   = sum(r["n_skipped_spans"] for r in records)

    by_split: dict[str, dict] = defaultdict(lambda: {"traces": 0, "spans": 0, "annotated": 0})
    for r in records:
        s = r["split"]
        by_split[s]["traces"]    += 1
        by_split[s]["spans"]     += r["n_spans"]
        by_split[s]["annotated"] += r["n_annotated_spans"]

    print(f"\n{'='*60}")
    print("Span dataset summary")
    print(f"{'='*60}")
    print(f"  Traces processed:      {len(records)}")
    print(f"  Traces failed:         {n_failed}")
    print(f"  Total action spans:    {total_spans}  "
          f"(avg {total_spans/max(len(records),1):.1f}/trace)")
    print(f"    Annotated (error):   {total_annotated}  "
          f"({100*total_annotated/max(total_spans,1):.1f}%)")
    print(f"    Correct:             {total_correct}  "
          f"({100*total_correct/max(total_spans,1):.1f}%)")
    print(f"  Spans skipped:         {total_skipped}  (missing required fields)")
    print(f"  Annotation coverage:   {ann_matched_total}/{ann_total_total}  "
          f"({100*ann_matched_total/max(ann_total_total,1):.1f}%)")
    print()
    print(f"  {'Split':<8}  {'Traces':>7}  {'Spans':>7}  {'Avg/trace':>9}  {'Annotated':>9}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")
    for sn in ("train", "val", "test"):
        d = by_split[sn]
        avg = d["spans"] / max(d["traces"], 1)
        print(f"  {sn:<8}  {d['traces']:>7}  {d['spans']:>7}  {avg:>9.1f}  {d['annotated']:>9}")

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
