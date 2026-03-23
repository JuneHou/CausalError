#!/usr/bin/env python3
"""
Single-intervention re-run pipeline (counterfactual do(A_i ≈ 0)).

We do NOT truncate the trace after the patch — the saved counterfactual is the full trace
with only the error span replaced (so causal evaluation over later errors/sequence is valid).
Truncation is only for patch-validity checks; see trace_replay.truncate_trace_after_span.

Modes (--mode):
  patch_only: One folder (gaia_output_dir). One readable JSON per intervention: trace_id,
    steps, intervention_annotation, raw_trace. No model calls.
  rerun: One folder only (gaia_rerun_dir). One JSON per intervention containing
    trace_id, intervention_span_id, intervention_annotation, messages_prefix,
    observation_tape, rerun_transcript. No patch-only output.

Output format: Each saved JSON is human-readable: trace_id, intervention_span_id, steps
(execution-order list with kind, span_id, content, is_patched), intervention_annotation
(the error record for this intervention), and raw_trace (full OTEL for downstream use).

Usage (from benchmarking/):
  python causal/intervention/rerun_intervention.py \\
      --trace_dir       data/GAIA \\
      --gaia_output_dir data/GAIA_interventions \\
      --mode patch_only
  python causal/intervention/rerun_intervention.py \\
      --mode rerun --gaia_rerun_dir data/GAIA_interventions_rerun
"""
from __future__ import annotations

import os
import sys
_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import re
from typing import Any, Dict, List, Optional

from trail_io import TraceObj, load_trail_trace, iter_trail_traces, get_expanded_snippet
from patch_apply import PatchSpec, load_patch_specs, apply_patch, instantiate_spec
from trace_replay import (
    get_run_config_from_trace,
    get_llm_input_messages_for_span,
    get_ordered_steps,
    build_readable_steps,
    get_patched_span_content,
    clone_trace_and_patch_span,
    format_tool_observation_message,
)

from causal.intervention.intervene import route_error_to_family, ERROR_TYPE_TO_FAMILY, _normalize


# ---------------------------------------------------------------------------
# Sanitize filename from error_id / trace_id
# ---------------------------------------------------------------------------

def _sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:64]


# ---------------------------------------------------------------------------
# Single-intervention run: do(A_i ≈ 0)
# ---------------------------------------------------------------------------

def run_single_intervention(
    trace_obj: TraceObj,
    error_instance: dict,
    patch_spec: PatchSpec,
    patched_text: str,
    window: int,
    gaia_output_dir: str,
    trace_raw: dict,
    mode: str = "patch_only",
    gaia_rerun_dir: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one counterfactual: do(A_i ≈ 0). Do NOT truncate — full trace with patched span
    so causal evaluation (later errors, sequence) is possible.

    mode:
      patch_only: clone trace, replace error span with patched_text, save to gaia_output_dir
        (static patch artifact; no model call).
      rerun: same as above plus build prefix (get_llm_input_messages_for_span), observation
        tape, run replay driver, save transcript to gaia_rerun_dir.
    """
    trace_id = trace_obj.trace_id
    location = error_instance.get("annotated_span_id") or error_instance.get("location") or ""
    if not location:
        return {
            "trace_id": trace_id,
            "error_id": error_instance.get("error_id", ""),
            "location": "",
            "output_path": "",
            "success": False,
            "reason": "missing_location",
        }

    if not trace_raw:
        trace_raw = getattr(trace_obj, "raw_trace", None) or {}
    if not trace_raw:
        return {
            "trace_id": trace_id,
            "error_id": error_instance.get("error_id", ""),
            "location": location,
            "output_path": "",
            "success": False,
            "reason": "no_raw_trace",
        }

    # Patch input (original content at the intervention span) vs patch output (patched_text).
    # This is the key information needed to validate that the intervention actually fixes A_i.
    original_span_output = get_patched_span_content(trace_raw, location)

    # Clone and patch the error span only — do not truncate (keeps full trace for causal eval)
    counterfactual_trace = clone_trace_and_patch_span(trace_raw, location, patched_text)
    counterfactual_trace["intervention_span_id"] = location

    error_ix = error_instance.get("error_index", 0)
    safe_id = _sanitize_for_filename(error_instance.get("error_id", str(error_ix)))
    intervention_annotation = dict(error_instance)

    # Base payload: one intervention per JSON file (regardless of mode).
    base_payload: Dict[str, Any] = {
        "trace_id": trace_id,
        "error_index": error_ix,
        "error_id": error_instance.get("error_id", ""),
        "operator_family": patch_spec.operator_family,
        "intervention_span_id": location,
        "intervention_annotation": intervention_annotation,
        "patch": {
            "original_span_output": original_span_output,
            "patched_span_output": patched_text,
        },
    }

    out_path = ""
    if mode != "rerun":
        # patch_only output: include a readable step list + full counterfactual trace for inspection.
        steps_readable = build_readable_steps(counterfactual_trace, intervention_span_id=location)
        base_payload["steps"] = steps_readable
        base_payload["raw_trace"] = counterfactual_trace

        out_name = f"{trace_id}_do_{error_ix}_{safe_id}.json"
        out_path = os.path.join(gaia_output_dir, out_name)
        os.makedirs(gaia_output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(base_payload, f, indent=2, ensure_ascii=False)

    result: Dict[str, Any] = {
        "trace_id": trace_id,
        "error_id": error_instance.get("error_id", ""),
        "location": location,
        "operator_family": patch_spec.operator_family,
        "output_path": out_path or None,
        "success": True,
        "reason": "saved",
        "intervention_span_id": location,
    }

    if mode == "rerun" and gaia_rerun_dir:
        rerun_result = _run_rerun_suffix(
            trace_raw=trace_raw,
            trace_id=trace_id,
            location=location,
            patched_text=patched_text,
            error_ix=error_ix,
            safe_id=safe_id,
            gaia_rerun_dir=gaia_rerun_dir,
            model_override=model_override,
            base_payload=base_payload,
        )
        result["rerun_paths"] = rerun_result.get("paths", {})
        if not rerun_result.get("success", True):
            result["rerun_error"] = rerun_result.get("error", "")

    return result


def _run_rerun_suffix(
    trace_raw: dict,
    trace_id: str,
    location: str,
    patched_text: str,
    error_ix: int,
    safe_id: str,
    gaia_rerun_dir: str,
    model_override: Optional[str] = None,
    base_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build prefix, observation tape, run replay driver; write one consolidated JSON per intervention.
    The output JSON contains everything needed to inspect the intervention:
      - patch input/output (base_payload["patch"])
      - intervention annotation (base_payload["intervention_annotation"])
      - messages_prefix, observation_tape, suffix_schedule, rerun_transcript
    """
    gaia_rerun_dir = os.path.abspath(gaia_rerun_dir)
    try:
        run_config = get_run_config_from_trace(trace_raw)
    except ValueError as e:
        return {"success": False, "error": str(e), "paths": {}}

    prefix = get_llm_input_messages_for_span(trace_raw, location)
    messages = prefix + [{"role": "assistant", "content": patched_text}]

    steps_after = get_ordered_steps(trace_raw)
    found = False
    tape: List[Dict[str, Any]] = []
    original_suffix_events: List[Dict[str, Any]] = []
    for s in steps_after:
        if s.get("span_id") == location:
            found = True
            continue
        if not found:
            continue
        kind = s.get("kind")
        sid = s.get("span_id")
        if kind == "tool":
            obs = format_tool_observation_message(s)
            tape.append(obs)
            original_suffix_events.append({"kind": "tool", "span_id": sid, "content": obs.get("content", "")})
        elif kind == "llm":
            original_suffix_events.append({"kind": "llm", "span_id": sid, "content": get_patched_span_content(trace_raw, sid) or ""})

    base = f"{trace_id}_do_{error_ix}_{safe_id}"
    out_path = os.path.join(gaia_rerun_dir, f"{base}_rerun.json")
    os.makedirs(gaia_rerun_dir, exist_ok=True)
    tape_serializable = [{"role": "user", "content": t.get("content", "")} for t in tape]

    # Replay driver: follow the original post-intervention step schedule.
    # For each original LLM step after `location`, call the model once.
    # For each original TOOL step after `location`, inject the recorded observation.
    try:
        from replication import run_one_llm_step
    except ImportError:
        payload = dict(base_payload or {})
        payload.update(
            {
                "run_config": run_config,
                "messages_prefix": messages,
                # Keep only two after-intervention views: original vs rerun (counterfactual).
                "after_intervention": {
                    "original": original_suffix_events,
                    "rerun": [{"error": "litellm not available; no suffix re-generated"}],
                },
            }
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return {"success": False, "error": "litellm not available", "paths": {"rerun": out_path}}

    model_id = model_override or run_config["model_id"]
    temperature = int(run_config.get("temperature", 0))
    transcript_lines: List[Dict[str, Any]] = []
    transcript_turn = 0
    transcript_lines.append({"turn": transcript_turn, "role": "assistant", "content": patched_text, "source": "patched"})
    transcript_turn += 1

    tape_idx = 0
    # Follow the ORIGINAL suffix shape (kind/span_id) and generate a rerun suffix.
    for evt in original_suffix_events:
        kind = evt.get("kind")
        sid = evt.get("span_id")
        if kind == "tool":
            # Inject observation corresponding to this tool step (in original order).
            if tape_idx < len(tape_serializable):
                obs_msg = tape_serializable[tape_idx]
                tape_idx += 1
            else:
                obs_msg = {"role": "user", "content": "Observation (tool unknown): (missing from tape)"}
            messages.append(obs_msg)
            transcript_lines.append({"turn": transcript_turn, "role": "user", "content": obs_msg.get("content", ""), "source": "tape", "span_id": sid})
            transcript_turn += 1
            continue

        # LLM step: call model once.
        try:
            out = run_one_llm_step(messages, model=model_id, temperature=temperature)
        except Exception as e:
            transcript_lines.append({"turn": transcript_turn, "error": str(e)})
            break
        messages.append({"role": "assistant", "content": out})
        transcript_lines.append({"turn": transcript_turn, "role": "assistant", "content": out, "source": "model", "span_id": sid})
        transcript_turn += 1

    payload = dict(base_payload or {})
    payload.update(
        {
            "run_config": {**run_config, "model_id": model_id, "temperature": temperature},
            "messages_prefix": messages,
            # Keep only two after-intervention views:
            # - original: what happened after the intervention in the original trace (LLM outputs + tool obs)
            # - rerun: what happened after the intervention in the counterfactual run (LLM regenerated + tool obs replayed)
            # Note: the patched turn itself is in patch.patched_span_output and as transcript_lines[0].
            "after_intervention": {
                "original": original_suffix_events,
                "rerun": transcript_lines[1:],  # exclude the patched turn
            },
        }
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {"success": True, "paths": {"rerun": out_path}}


# ---------------------------------------------------------------------------
# Full pipeline: for each trace, for each error → one counterfactual run
# ---------------------------------------------------------------------------

def run_all_single_interventions(
    trace_dir: str,
    annotations_dir: str,
    patch_specs_dir: str,
    out_dir: str,
    gaia_output_dir: str,
    trace_ids: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
    window: int = 0,
    mode: str = "patch_only",
    gaia_rerun_dir: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each trace with annotations, for each error A_i:
      - Route error to operator family, apply patch to get patched_text.
      - Build counterfactual trace do(A_i ≈ 0) and save in GAIA format under gaia_output_dir.

    Writes:
      - out_dir/rerun_log.jsonl — one line per intervention (trace_id, error_id, output_path, success).
      - gaia_output_dir/<trace_id>_do_<i>_<safe_id>.json — counterfactual traces in GAIA format.

    Returns stats dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gaia_output_dir, exist_ok=True)
    rerun_log_path = os.path.join(out_dir, "rerun_log.jsonl")

    patch_specs: Dict[str, PatchSpec] = load_patch_specs(patch_specs_dir)
    trace_pairs = list(iter_trail_traces(trace_dir, annotations_dir, trace_ids, max_traces))

    stats: Dict[str, Any] = {
        "traces_processed": 0,
        "interventions_attempted": 0,
        "interventions_saved": 0,
        "skipped_no_family": 0,
        "skipped_no_location": 0,
        "by_family": {},
    }

    with open(rerun_log_path, "w", encoding="utf-8") as log_f:
        for trace_path, ann_path in trace_pairs:
            try:
                trace_obj = load_trail_trace(trace_path, ann_path)
            except Exception as exc:
                print(f"  [WARN] load failed {trace_path}: {exc}", file=sys.stderr)
                continue

            stats["traces_processed"] += 1
            trace_raw = getattr(trace_obj, "raw_trace", None) or {}
            if not trace_raw and trace_path:
                try:
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace_raw = json.load(f)
                except Exception:
                    trace_raw = {}

            for error_ix, err in enumerate(trace_obj.errors):
                raw_type = err.get("error_type") or err.get("category") or ""
                family = route_error_to_family(raw_type)
                err["error_index"] = error_ix

                if not family:
                    stats["skipped_no_family"] += 1
                    continue
                if family not in patch_specs:
                    stats["skipped_no_family"] += 1
                    continue

                loc = err.get("annotated_span_id") or err.get("location") or ""
                if not loc:
                    stats["skipped_no_location"] += 1
                    continue

                spec = patch_specs[family]
                stats["interventions_attempted"] += 1
                stats["by_family"][family] = stats["by_family"].get(family, 0) + 1

                # Apply patch to get patched_text (same as intervene.py)
                snippet = get_expanded_snippet(trace_obj, loc, window=window)
                inst = instantiate_spec(spec, err, snippet, trace_obj)
                patched_text = inst.get("patched_text", snippet)

                record = run_single_intervention(
                    trace_obj,
                    err,
                    spec,
                    patched_text,
                    window,
                    gaia_output_dir,
                    trace_raw,
                    mode=mode,
                    gaia_rerun_dir=gaia_rerun_dir,
                    model_override=model_override,
                )
                if record.get("success"):
                    stats["interventions_saved"] += 1
                    path = record.get("output_path") or (record.get("rerun_paths") or {}).get("rerun", "")
                    print(f"  [OK]   {record['trace_id']} do({record['error_id'][:24]}...) -> {path}")
                else:
                    print(f"  [FAIL] {record['trace_id']} {record.get('reason', '')}")

                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== Rerun intervention summary ===")
    print(f"  Traces processed       : {stats['traces_processed']}")
    print(f"  Interventions attempted: {stats['interventions_attempted']}")
    print(f"  Interventions saved    : {stats['interventions_saved']}")
    print(f"  Skipped (no family)     : {stats['skipped_no_family']}")
    print(f"  Skipped (no location)   : {stats['skipped_no_location']}")
    print(f"  Rerun log              : {rerun_log_path}")
    print(f"  GAIA-format outputs    : {gaia_output_dir}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-intervention re-runs: do(A_i≈0) per error, save GAIA-format traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trace_dir", default="data/GAIA", help="Directory of trace JSON files.")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia", help="Per-trace annotation JSONs.")
    parser.add_argument("--patch_specs_dir", default="data/patches", help="Operator-family patch specs.")
    parser.add_argument("--out_dir", default="outputs/interventions", help="Where to write rerun_log.jsonl.")
    parser.add_argument(
        "--gaia_output_dir",
        default="data/GAIA_interventions",
        help="Output directory for counterfactual traces in GAIA format (same format as data/GAIA).",
    )
    parser.add_argument("--trace_ids", nargs="*", help="Optional list of trace IDs.")
    parser.add_argument("--max_traces", type=int, default=None, help="Cap number of traces.")
    parser.add_argument("--window", type=int, default=0, help="Snippet expansion for patch (0 = error span only).")
    parser.add_argument(
        "--mode",
        choices=["patch_only", "rerun"],
        default="patch_only",
        help="patch_only: one folder (gaia_output_dir), one JSON per intervention. rerun: one folder only (gaia_rerun_dir), one JSON per intervention (no patch output).",
    )
    parser.add_argument(
        "--gaia_rerun_dir",
        default="data/GAIA_interventions_rerun",
        help="For --mode rerun: single output directory; one *_rerun.json per intervention (messages_prefix, observation_tape, rerun_transcript inside).",
    )
    parser.add_argument(
        "--model",
        default="o3-mini",
        help="Model to use for --mode rerun (LLM calls for suffix generation). Default: o3-mini. Ignored when mode is patch_only.",
    )
    args = parser.parse_args()

    run_all_single_interventions(
        trace_dir=args.trace_dir,
        annotations_dir=args.annotations_dir,
        patch_specs_dir=args.patch_specs_dir,
        out_dir=args.out_dir,
        gaia_output_dir=args.gaia_output_dir,
        trace_ids=args.trace_ids,
        max_traces=args.max_traces,
        window=args.window,
        mode=args.mode,
        gaia_rerun_dir=args.gaia_rerun_dir if args.mode == "rerun" else None,
        model_override=args.model if args.mode == "rerun" else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
