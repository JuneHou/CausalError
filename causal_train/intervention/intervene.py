#!/usr/bin/env python3
"""Intervention runner — main execution loop (A4, C1, C2).

Modes:
  - Static patching (default): apply patch specs to snippets, write patch_log.jsonl.
  - Re-run (--rerun): for each error A_i create single-intervention run do(A_i≈0),
    save counterfactual traces in GAIA format for per-edge effect attribution.

For each TRAIL trace with annotated errors:
  1. Route each error's category → operator family (A4).
  2. Load the local snippet at the error's span location (A3).
  3. Instantiate + apply the patch spec (A5, D).
  4. Validate the patch (A6).
  5. Save patch_log.jsonl, patched_traces.jsonl, intervention_stats.json.
  6. If --rerun: also run single-intervention counterfactuals and save to --gaia_output_dir.

Usage (from benchmarking/):
    # Static only
    python causal/intervention/intervene.py --trace_dir data/GAIA --annotations_dir processed_annotations_gaia ...
    # With re-run (counterfactual traces in GAIA format)
    python causal/intervention/intervene.py --trace_dir data/GAIA ... --rerun --gaia_output_dir data/GAIA_interventions

Then run causal/intervention/effect_eval.py to compute Δ(A→B) scores (from patch_log or rerun_log).
"""
from __future__ import annotations

import argparse
import os
import sys
_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarking"))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import traceback
from collections import defaultdict
from typing import Dict, List, Optional

from trail_io import TraceObj, load_trail_trace, iter_trail_traces
from patch_apply import PatchSpec, PatchRecord, load_patch_specs, apply_patch

# ---------------------------------------------------------------------------
# Error-type → operator-family routing table (A4)
# ---------------------------------------------------------------------------
# Keys are normalised; also cover common raw variants.
# Keep top-1 per type as instructed; add FALLBACK for unmapped types.

ERROR_TYPE_TO_FAMILY: Dict[str, str] = {
    # Resource overuse
    "Resource Abuse":                       "BUDGET_GUARD_STOP_CONDITION",
    # Retrieval quality
    "Poor Information Retrieval":           "RETRIEVAL_REQUERY",
    # Output / argument formatting
    "Formatting Errors":                    "TOOL_SCHEMA_REPAIR",
    "Formatting Error":                     "TOOL_SCHEMA_REPAIR",
    # Wrong tool chosen
    "Tool Selection Errors":                "TOOL_SELECTION_SWAP",
    "Tool Selection Error":                 "TOOL_SELECTION_SWAP",
    # Context / state loss
    "Context Handling Failures":            "CONTEXT_STATE_CARRYOVER",
    "Context Handling Failure":             "CONTEXT_STATE_CARRYOVER",
    # Tool-call related (execution / interpretation)
    "Tool-Related":                         "OUTPUT_INTERPRETATION_VERIFY",
    "Tool-related":                         "OUTPUT_INTERPRETATION_VERIFY",
    "Tool Related":                         "OUTPUT_INTERPRETATION_VERIFY",
    # Goal / instruction compliance
    "Goal Deviation":                       "GOAL_CONSTRAINT_CHECK",
    "Language-Only":                        "GOAL_CONSTRAINT_CHECK",
    "Language-only":                        "GOAL_CONSTRAINT_CHECK",
    "Instruction Non-compliance":           "GOAL_CONSTRAINT_CHECK",
    "Incorrect Problem Identification":     "GOAL_CONSTRAINT_CHECK",
    # Orchestration: describe instead of execute
    "Task Orchestration":                   "EXECUTE_INSTEAD_OF_DESCRIBE",
    # Auth / budget errors (treat as resource guard)
    "Authentication Errors":               "BUDGET_GUARD_STOP_CONDITION",
}

# Normalisation map (mirrors action_primitive_library._ERROR_NORM_MAP)
_NORM_MAP: Dict[str, str] = {
    "Context Handling Failure":  "Context Handling Failures",
    "Formatting Error":          "Formatting Errors",
    "Tool Selection Error":      "Tool Selection Errors",
    "Tool related":              "Tool-Related",
    "Tool-related":              "Tool-Related",
    "Language-only":             "Language-Only",
}


def _normalize(etype: str) -> str:
    s = etype.strip().title()
    return _NORM_MAP.get(s, _NORM_MAP.get(etype.strip(), etype.strip()))


def route_error_to_family(error_type: str) -> Optional[str]:
    """Return operator family string for an error type, or None if unknown."""
    norm = _normalize(error_type)
    return ERROR_TYPE_TO_FAMILY.get(norm) or ERROR_TYPE_TO_FAMILY.get(error_type.strip())


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _record_to_dict(rec: PatchRecord, snippet_max: int = 2000) -> dict:
    """Serialise a PatchRecord to a JSON-safe dict (truncating long texts)."""
    return {
        "trace_id":         rec.trace_id,
        "error_id":         rec.error_id,
        "operator_family":  rec.operator_family,
        "location":         rec.location,
        "original_text":    rec.original_text[:snippet_max],
        "patched_text":     rec.patched_text[:snippet_max],
        "diff_lines":       rec.diff_lines,
        "instantiated_spec": {
            k: (v[:300] if isinstance(v, str) else v)
            for k, v in rec.instantiated_spec.items()
            if k != "local_snippet"
        },
        "validation":  rec.validation,
        "success":     rec.success,
    }


# ---------------------------------------------------------------------------
# Main intervention loop (C2)
# ---------------------------------------------------------------------------


def run_interventions(
    trace_dir: str,
    annotations_dir: str,
    patch_specs_dir: str,
    out_dir: str,
    trace_ids: Optional[List[str]] = None,
    max_traces: Optional[int] = None,
    window: int = 0,
    rerun: bool = False,
    gaia_output_dir: Optional[str] = None,
) -> dict:
    """
    Loop over traces, apply patches for every annotated error, write outputs.

    Outputs written to out_dir/:
      patch_log.jsonl          – one PatchRecord per line
      patched_traces.jsonl     – one entry per trace (with all its patches)
      intervention_stats.json  – summary counts
    If rerun=True: also run single-intervention counterfactuals and write
      rerun_log.jsonl and gaia_output_dir/<trace_id>_do_<i>.json (GAIA format).
    """
    os.makedirs(out_dir, exist_ok=True)
    if rerun and gaia_output_dir:
        os.makedirs(gaia_output_dir, exist_ok=True)
    patch_log_path      = os.path.join(out_dir, "patch_log.jsonl")
    patched_traces_path = os.path.join(out_dir, "patched_traces.jsonl")
    stats_path          = os.path.join(out_dir, "intervention_stats.json")

    # Load all patch specs
    patch_specs: Dict[str, PatchSpec] = load_patch_specs(patch_specs_dir)
    print(f"[intervene] Loaded {len(patch_specs)} patch specs: {sorted(patch_specs)}")

    # Discover traces
    trace_pairs = list(iter_trail_traces(trace_dir, annotations_dir, trace_ids, max_traces))
    print(f"[intervene] Processing {len(trace_pairs)} traces (window={window})")

    stats: Dict = {
        "traces":               0,
        "errors_seen":          0,
        "patches_attempted":    0,
        "patches_succeeded":    0,
        "patches_failed":       0,
        "skipped_no_family":    0,
        "skipped_no_location":  0,
        "by_family":            defaultdict(int),
        "by_family_success":    defaultdict(int),
        "by_error_type":        defaultdict(int),
    }

    with (
        open(patch_log_path, "w", encoding="utf-8") as log_f,
        open(patched_traces_path, "w", encoding="utf-8") as pt_f,
    ):
        for trace_path, ann_path in trace_pairs:
            try:
                trace_obj = load_trail_trace(trace_path, ann_path)
            except Exception as exc:
                print(f"  [WARN] load failed {trace_path}: {exc}", file=sys.stderr)
                continue

            stats["traces"] += 1
            trace_patches: List[dict] = []

            for err in trace_obj.errors:
                stats["errors_seen"] += 1
                raw_type = err.get("error_type") or err.get("category") or ""
                family = route_error_to_family(raw_type)
                stats["by_error_type"][_normalize(raw_type)] += 1

                if not family:
                    stats["skipped_no_family"] += 1
                    print(f"  [SKIP] no routing for error_type='{raw_type}'")
                    continue
                if family not in patch_specs:
                    stats["skipped_no_family"] += 1
                    print(f"  [SKIP] family '{family}' not in patch_specs (missing JSON?)")
                    continue

                loc = err.get("annotated_span_id") or err.get("location") or ""
                if not loc:
                    stats["skipped_no_location"] += 1
                    print(f"  [SKIP] no location in error {err.get('error_id','?')}")
                    continue

                spec = patch_specs[family]
                stats["patches_attempted"] += 1

                try:
                    record = apply_patch(trace_obj, err, spec, window=window)
                except Exception as exc:
                    print(
                        f"  [WARN] apply_patch raised for {err.get('error_id','?')}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    stats["patches_failed"] += 1
                    continue

                stats["by_family"][family] += 1
                if record.success:
                    stats["patches_succeeded"] += 1
                    stats["by_family_success"][family] += 1
                    print(
                        f"  [OK]   {family:35s} @ {loc[:16]}  diff={record.diff_lines}"
                    )
                else:
                    stats["patches_failed"] += 1
                    print(
                        f"  [FAIL] {family:35s} @ {loc[:16]}  "
                        f"reasons={record.validation['reasons']}"
                    )

                entry = _record_to_dict(record)
                log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                trace_patches.append(entry)

            if trace_patches:
                pt_entry = {
                    "trace_id":    trace_obj.trace_id,
                    "num_patches": len(trace_patches),
                    "patches":     trace_patches,
                }
                pt_f.write(json.dumps(pt_entry, ensure_ascii=False) + "\n")

    # Optional: single-intervention re-runs (do(A_i≈0)) and save GAIA-format traces
    if rerun and gaia_output_dir:
        try:
            from rerun_intervention import run_all_single_interventions
            run_all_single_interventions(
                trace_dir=trace_dir,
                annotations_dir=annotations_dir,
                patch_specs_dir=patch_specs_dir,
                out_dir=out_dir,
                gaia_output_dir=gaia_output_dir,
                trace_ids=trace_ids,
                max_traces=max_traces,
                window=window,
            )
        except Exception as e:
            print(f"  [WARN] Rerun intervention failed: {e}", file=sys.stderr)

    # Write stats
    stats["by_family"]         = dict(stats["by_family"])
    stats["by_family_success"] = dict(stats["by_family_success"])
    stats["by_error_type"]     = dict(stats["by_error_type"])
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Summary print
    print("\n=== Intervention summary ===")
    print(f"  Traces processed : {stats['traces']}")
    print(f"  Errors seen      : {stats['errors_seen']}")
    print(f"  Patches attempted: {stats['patches_attempted']}")
    print(f"  Patches succeeded: {stats['patches_succeeded']}")
    print(f"  Patches failed   : {stats['patches_failed']}")
    print(f"  Skipped (no fam) : {stats['skipped_no_family']}")
    print(f"  Skipped (no loc) : {stats['skipped_no_location']}")
    print(f"  By family        : {stats['by_family']}")
    print(f"Outputs → {out_dir}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRAIL intervention runner: apply operator-family patches to annotated errors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trace_dir", default="data/GAIA",
        help="Directory containing per-trace JSON files.",
    )
    parser.add_argument(
        "--annotations_dir", default="processed_annotations_gaia",
        help="Directory containing per-trace annotation JSON files.",
    )
    parser.add_argument(
        "--patch_specs_dir", default="data/patches",
        help="Directory containing operator-family patch spec JSON files.",
    )
    parser.add_argument(
        "--out_dir", default="outputs/interventions",
        help="Output directory for patch_log.jsonl, patched_traces.jsonl, stats.json.",
    )
    parser.add_argument(
        "--trace_ids", nargs="*", metavar="TRACE_ID",
        help="Explicit list of trace IDs to process (default: all matched).",
    )
    parser.add_argument(
        "--max_traces", type=int, default=None,
        help="Cap number of traces processed (useful for quick testing).",
    )
    parser.add_argument(
        "--window", type=int, default=0,
        help="Snippet expansion window: 0=just the error span, 1=+1 sibling, etc.",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Also run single-intervention counterfactuals and save GAIA-format traces.",
    )
    parser.add_argument(
        "--gaia_output_dir", default="data/GAIA_interventions",
        help="Output directory for counterfactual traces (GAIA format). Used when --rerun.",
    )
    args = parser.parse_args()

    run_interventions(
        trace_dir       = args.trace_dir,
        annotations_dir = args.annotations_dir,
        patch_specs_dir = args.patch_specs_dir,
        out_dir         = args.out_dir,
        trace_ids       = args.trace_ids,
        max_traces      = args.max_traces,
        window          = args.window,
        rerun           = args.rerun,
        gaia_output_dir = args.gaia_output_dir if args.rerun else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
