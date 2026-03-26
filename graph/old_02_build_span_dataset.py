#!/usr/bin/env python3
"""
02_build_span_dataset.py — Extract step-level spans and map error annotations.

The paper (Table 5) counts at the "step" level: CodeAgent.run, ToolCallingAgent.run,
and "Step N" spans (~977 spans total for GAIA, 8.28 avg/trace). These are the
encoding units. Their TEXT is built from the LLM/TOOL action spans nested inside them
(which carry the actual input.value / output.value). Annotations reference those
inner action spans; span_level_parser's parent-walk maps them up to the step span.

For each trace:
  1. Load trace JSON + annotation JSON.
  2. Use span_level_parser to extract step spans (CodeAgent.run, ToolCallingAgent.run,
     Step N) in start-time order.
  3. For each step span, collect all LLM and TOOL action spans anywhere inside it.
     Build the step's text by concatenating their content:
       LLM span:  [SPAN] / [INPUT] / [OUTPUT]           — both fields required
       TOOL span: [SPAN] / [TOOL] / [INPUT] / [OUTPUT?] — input required; output optional
     Logs (function.name / function.output / severity) appended when present.
     If a step span has no action children with required fields → skip + error.
  4. Map annotation location (inner span_id) → step span via parent-walk.
  5. Labels: step span → union of error categories from annotations that map to it.
     Steps with no annotation mapping → Correct (labels=[]).

Output: graph/data/span_dataset.jsonl — one JSON record per trace.

Record schema:
  {
    "trace_id": str,
    "split":    "train"|"val"|"test",
    "dataset":  "GAIA"|"SWE-bench",
    "spans": [
      {
        "span_id":    str,
        "step_index": int,       // 1-based, matches span_level_parser ordering
        "span_name":  str,
        "text":       str,       // concatenated text of action children
        "labels":     [str, ...], // error categories; [] = Correct
        "is_correct": bool
      }, ...
    ],
    "n_spans":          int,
    "n_annotated_spans":int,
    "n_skipped_spans":  int      // step spans with no usable action children
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
from span_level_parser import (  # noqa: E402
    parse_trace_to_step_level,
    map_annotation_to_step,
    _span_name,
)

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
CONTAINER_KINDS = frozenset({"AGENT", "CHAIN"})  # step-span kinds — stop recursion here

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
# Collect action spans under a step span (any depth)
# ---------------------------------------------------------------------------

def collect_action_spans(span: dict) -> list[dict]:
    """
    DFS over child_spans; collect every LLM or TOOL span that belongs
    DIRECTLY to this step span — not to any nested step span inside it.

    Recursion stops when a child is itself a step-span container (AGENT or CHAIN
    kind). Those nested step spans are independent encoding units in the dataset
    and will be processed separately. This prevents content overlap between a
    parent step span (e.g. CodeAgent.run) and its child step spans (Step N).
    """
    result = []
    for child in span.get("child_spans") or []:
        kind = _span_kind(child)
        if kind in ACTION_KINDS:
            result.append(child)
        elif kind in CONTAINER_KINDS:
            # Nested step span — do NOT recurse; it will be encoded separately
            pass
        else:
            # Intermediate non-step, non-action span (e.g. unnamed wrapper) — recurse
            result.extend(collect_action_spans(child))
    return result


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
# Build the full text for a step span from its action children
# ---------------------------------------------------------------------------

def build_step_text(
    step_span: dict,
    trace_id: str,
    step_index: int,
) -> tuple[Optional[str], int]:
    """
    Collect all action spans inside step_span and concatenate their texts.

    Returns:
        (text, n_skipped_actions)  — text=None if no usable action children
    """
    action_spans = collect_action_spans(step_span)
    if not action_spans:
        return None, 0

    text_parts = []
    n_skipped = 0
    for i, asp in enumerate(action_spans):
        t, err = _action_span_text(asp)
        if t is None:
            log.error(
                "Skipping action span (trace=%s step=%d action=%d): %s",
                trace_id, step_index, i + 1, err,
            )
            n_skipped += 1
        else:
            text_parts.append(t)

    if not text_parts:
        return None, n_skipped

    # Separate multiple action spans with a divider
    full_text = "\n\n" + ("=" * 40) + "\n\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
    return full_text, n_skipped


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

    # Parse step spans
    parsed = parse_trace_to_step_level(trace_data)
    step_spans = parsed.get("step_spans") or []
    if not step_spans:
        log.warning("No step spans in trace %s — skipping", trace_id)
        return None

    # Map annotation location → step span_id using parent-walk
    # step_span_id → set of error categories
    step_to_labels: dict[str, set[str]] = defaultdict(set)
    ann_matched = 0
    ann_total = 0

    for err in ann_data.get("errors") or []:
        loc = err.get("location")
        cat = err.get("category")
        if not loc or not cat:
            continue
        ann_total += 1
        mapping = map_annotation_to_step(parsed, loc)
        if mapping is None:
            log.debug("Annotation loc=%s in trace %s: no step ancestor found", loc, trace_id)
            continue
        step_to_labels[mapping["step_span_id"]].add(cat)
        ann_matched += 1

    if ann_total > 0 and ann_matched == 0:
        log.warning("trace %s: 0/%d annotations mapped to step spans", trace_id, ann_total)

    # Build per-step records
    span_records = []
    n_skipped_steps = 0
    total_skipped_actions = 0

    for ss in step_spans:
        sp         = ss["span"]
        step_index = ss["step_index"]
        span_id    = sp.get("span_id")

        text, n_sk = build_step_text(sp, trace_id, step_index)
        total_skipped_actions += n_sk

        if text is None:
            log.error(
                "No usable action children for step %d (%r) in trace %s — skipping step",
                step_index, _span_name(sp), trace_id,
            )
            n_skipped_steps += 1
            continue

        labels = sorted(step_to_labels.get(span_id, set()))
        span_records.append({
            "span_id":    span_id,
            "step_index": step_index,
            "span_name":  _span_name(sp),
            "text":       text,
            "labels":     labels,
            "is_correct": len(labels) == 0,
        })

    if not span_records:
        log.error("All steps skipped for trace %s", trace_id)
        return None

    return {
        "trace_id":           trace_id,
        "split":              split,
        "dataset":            dataset,
        "spans":              span_records,
        "n_spans":            len(span_records),
        "n_annotated_spans":  sum(1 for s in span_records if not s["is_correct"]),
        "n_skipped_spans":    n_skipped_steps,
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
    print(f"  Total step spans:      {total_spans}  "
          f"(avg {total_spans/max(len(records),1):.1f}/trace)")
    print(f"    Annotated (error):   {total_annotated}  "
          f"({100*total_annotated/max(total_spans,1):.1f}%)")
    print(f"    Correct:             {total_correct}  "
          f"({100*total_correct/max(total_spans,1):.1f}%)")
    print(f"  Steps skipped:         {total_skipped}  (no usable action children)")
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
