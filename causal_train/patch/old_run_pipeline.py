#!/usr/bin/env python3
"""
Step 9: CLI entry point — runs the full causal intervention pipeline end-to-end.

Data flow:
  Step 0: filter_traces  → eligible_traces.json
  Step 1: case_builder   → a_instances.jsonl + edge_pairs.jsonl
  Step 2: patch_generator→ patch_results.jsonl, postcheck_failures.jsonl
  Step 3: rerun_harness  → rerun_results.jsonl
  Step 4: judge_a        → a_resolved.jsonl
  Step 5: judge_b        → b_effect.jsonl      (fan-out: one call per edge_pair)
  Step 6: aggregator     → effect_edges.json

Key invariant: Steps 2–4 are one record per A-instance (keyed by error_id).
Step 5 fans out to EdgePairs, reusing the shared rerun result per error_id.

Usage (from benchmarking/):
    python causal/patch/run_pipeline.py \\
        --trace_dir        data/GAIA \\
        --annotations_dir  processed_annotations_gaia \\
        --causal_graph     data/trail_causal_outputs_AIC/capri_graph.json \\
        --out_dir          outputs/interventions \\
        --model            openai/gpt-4o \\
        --max_steps_after 12

Skip individual steps with --skip_* flags if intermediate files already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Optional

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarking"))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_BENCH, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import filter_traces as _ft
import case_builder as _cb
import patch_generator as _pg
import rerun_harness as _rh
import judge_a_resolved as _ja
import judge_b_effect as _jb
import effect_aggregator as _ea


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if hasattr(r, "__dataclass_fields__"):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step0_filter_traces(args) -> str:
    # Save alongside the causal graph so it's shared across all pipeline runs
    out = os.path.join(os.path.dirname(os.path.abspath(args.causal_graph)),
                       "eligible_traces.json")
    print("\n[Step 0] Filtering eligible traces...")
    result = _ft.filter_traces(
        args.annotations_dir, args.causal_graph,
        min_errors=args.min_errors, strict=args.strict_filter,
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  {result['n_eligible']} / {result['n_total']} traces eligible "
          f"(min_errors={args.min_errors}, strict={args.strict_filter})")
    from collections import Counter
    ec: Counter = Counter()
    for t in result["eligible"]:
        for e in t["covered_edges"]:
            ec[f"{e['a']} -> {e['b']}"] += 1
    for edge, cnt in sorted(ec.items()):
        print(f"    {cnt:3d}  {edge}")
    print(f"  → {out}")
    return out


def step1_build_cases(args, eligible_path: str):
    a_out = _path(args.out_dir, "a_instances.jsonl")
    e_out = _path(args.out_dir, "edge_pairs.jsonl")
    print("\n[Step 1] Building AInstanceRecords + EdgePairs...")
    eligible_ids = None
    if os.path.isfile(eligible_path):
        with open(eligible_path, "r", encoding="utf-8") as f:
            et = json.load(f)
        eligible_ids = [t["trace_id"] for t in et.get("eligible", [])]
    a_instances, edge_pairs = _cb.build_cases(
        args.trace_dir, args.annotations_dir,
        args.causal_graph, args.patch_library,
        eligible_trace_ids=eligible_ids,
        max_traces=args.max_traces,
    )
    conflicts_path = _path(args.out_dir, "intervention_location_conflicts.jsonl")
    a_instances = _cb.dedup_by_intervention_location(a_instances, conflicts_path=conflicts_path)
    _write_jsonl(a_out, a_instances)
    _write_jsonl(e_out, edge_pairs)
    print(f"  → {len(a_instances)} A-instances → {a_out}")
    print(f"  → {len(edge_pairs)} edge pairs  → {e_out}")
    return a_out, e_out


def step2_generate_patches(args, a_instances_path: str) -> str:
    out = _path(args.out_dir, "patch_results.jsonl")
    failures_out = _path(args.out_dir, "postcheck_failures.jsonl")
    print("\n[Step 2-4] Generating patches (one per A-instance)...")

    with open(args.patch_library, "r", encoding="utf-8") as f:
        patch_library = json.load(f)
    a_instances = _load_jsonl(a_instances_path)

    results = []
    failures = []
    for ai in a_instances:
        result = _pg.generate_patch(ai, patch_library,
                                    model=args.model,
                                    max_retries=args.max_retries)
        if result.postcheck_passed:
            status = "OK"
        else:
            err = result.postcheck_failures[0][:100] if result.postcheck_failures else "?"
            status = f"FAIL: {err}"
        results.append(result)
        print(f"  [{status}] {result.trace_id[:8]} err={result.error_id[-20:]} "
              f"attempts={result.attempts}")
        if not result.postcheck_passed:
            failures.append(result)

    _write_jsonl(out, results)
    _write_jsonl(failures_out, failures)
    n_ok = sum(1 for r in results if r.postcheck_passed)
    print(f"  → {n_ok}/{len(results)} patches passed postcheck → {out}")
    return out


def step3_rerun(args, patch_results_path: str) -> str:
    out = _path(args.out_dir, "rerun_results.jsonl")
    rerun_model = getattr(args, "rerun_model", args.model)
    print(f"\n[Step 5] Rerun (model={rerun_model}, max_steps_after={args.max_steps_after})...")

    patch_results = _load_jsonl(patch_results_path)
    to_rerun = [p for p in patch_results if p.get("postcheck_passed")]
    print(f"  Rerunning {len(to_rerun)} / {len(patch_results)} (postcheck passed)")

    results = []
    for pr in to_rerun:
        rr = _rh.run_rerun(
            pr, args.trace_dir, args.annotations_dir,
            model=rerun_model,
            max_steps_after=args.max_steps_after,
        )
        results.append(rr)
        n_new = len(rr.rerun_suffix_spans)
        print(f"  [{rr.rerun_status}] {rr.trace_id[:8]} "
              f"err={rr.error_id[-20:]} new_spans={n_new}")

    _write_jsonl(out, results)
    from collections import Counter
    status_counts = Counter(rr.rerun_status for rr in results)
    print(f"  → {len(results)} rerun results → {out}")
    for s, n in sorted(status_counts.items()):
        print(f"    {s}: {n}")
    return out


def step4_judge_a(args, rerun_path: str, patch_path: str, a_instances_path: str) -> str:
    out = _path(args.out_dir, "a_resolved.jsonl")
    print("\n[Step 6] Judge 1 — A-resolved (one per A-instance)...")

    rerun_results = _load_jsonl(rerun_path)
    patch_results = _load_jsonl(patch_path)
    a_instances = _load_jsonl(a_instances_path)

    # Index by (trace_id, error_id)
    patch_idx = {(p["trace_id"], p.get("error_id", "")): p for p in patch_results}
    instance_idx = {(a["trace_id"], a["error_id"]): a for a in a_instances}

    verdicts = []
    for rr in rerun_results:
        if not rr.get("rerun_success"):
            continue
        key = (rr["trace_id"], rr.get("error_id", ""))
        pr = patch_idx.get(key)
        ai = instance_idx.get(key)
        if not pr or not ai:
            continue
        verdict = _ja.judge_a_resolved(rr, pr, ai, model=args.model)
        verdicts.append(verdict)
        status = "RESOLVED" if verdict.resolved else "UNRESOLVED"
        print(f"  [{status}] {verdict.trace_id[:8]} err={verdict.error_id[-20:]} "
              f"conf={verdict.confidence:.2f}")

    _write_jsonl(out, verdicts)
    n_res = sum(1 for v in verdicts if v.resolved)
    print(f"  → {n_res}/{len(verdicts)} resolved → {out}")
    return out


def step5_judge_b(args, rerun_path: str, a_resolved_path: str, edge_pairs_path: str) -> str:
    out = _path(args.out_dir, "b_effect.jsonl")
    print("\n[Step 7] Judge 2 — B-effect (fan-out: one call per EdgePair)...")

    rerun_results = _load_jsonl(rerun_path)
    a_verdicts = _load_jsonl(a_resolved_path)
    edge_pairs = _load_jsonl(edge_pairs_path)

    # Resolved set: (trace_id, error_id) — A-instance level
    resolved_keys = {
        (v["trace_id"], v.get("error_id", ""))
        for v in a_verdicts if v.get("resolved")
    }

    # Rerun index: (trace_id, error_id) → rerun_result
    rerun_idx = {
        (rr["trace_id"], rr.get("error_id", "")): rr
        for rr in rerun_results if rr.get("rerun_success")
    }

    from collections import Counter
    label_counts: Counter = Counter()
    verdicts = []
    n_skipped = 0
    for ep in edge_pairs:
        key = (ep["trace_id"], ep.get("error_id", ""))
        if key not in resolved_keys:
            n_skipped += 1
            continue
        rr = rerun_idx.get(key)
        if not rr:
            n_skipped += 1
            continue
        verdict = _jb.judge_b_effect(rr, ep, model=args.model)
        verdicts.append(verdict)
        label_counts[verdict.effect_label] += 1
        print(f"  {verdict.trace_id[:8]} {verdict.edge} "
              f"→ {verdict.effect_label} [{verdict.confidence}]")

    _write_jsonl(out, verdicts)
    print(f"  → {len(verdicts)} B-effect verdicts (skipped={n_skipped}) → {out}")
    print(f"  Effect distribution: {dict(label_counts)}")
    return out


def step6_aggregate(args) -> str:
    out = _path(args.out_dir, "effect_edges.json")
    print("\n[Step 8] Aggregating Δ(A→B)...")
    result = _ea.aggregate(
        _path(args.out_dir, "b_effect.jsonl"),
        _path(args.out_dir, "a_resolved.jsonl"),
        _path(args.out_dir, "patch_results.jsonl"),
        args.causal_graph,
        threshold=args.threshold,
        min_n=args.min_n,
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  {'Edge':<52} {'n':>4}  {'Δ':>7}  {'validated'}")
    print("  " + "-" * 75)
    for edge_key, info in result["edges"].items():
        n = info["n_valid_interventions"]
        delta = info["delta"]
        delta_str = f"{delta:+.3f}" if delta is not None else "  N/A "
        val = "YES" if info["validated"] else "no"
        print(f"  {edge_key:<52} {n:>4}  {delta_str}  {val}")
    pl = result["placebo"]
    print(f"\n  Placebo null: mean={pl['null_delta_mean']:.4f} "
          f"std={pl['null_delta_std']:.4f}")
    print(f"  → {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full causal intervention pipeline: Steps 0-8."
    )
    parser.add_argument("--trace_dir", default="data/GAIA")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia")
    parser.add_argument("--causal_graph",
                        default="data/trail_causal_outputs_AIC/capri_graph.json")
    parser.add_argument("--patch_library",
                        default="causal/patch/patch_library.json")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o",
                        help="LLM for patch generation and judges (Steps 2-5)")
    parser.add_argument("--rerun_model", default="openai/o3-mini",
                        help="LLM for counterfactual rerun continuation (Step 3); "
                             "should match the model that generated the original traces")
    parser.add_argument("--max_traces", type=int, default=None)
    parser.add_argument("--eligible_file", default=None,
                        help="Path to a pre-sampled eligible_traces JSON file "
                             "(e.g. from sample_coverage.py). If given, step 0 is "
                             "skipped automatically and this file is used instead.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--min_errors", type=int, default=2,
                        help="Min total errors per trace for eligibility (Step 0)")
    parser.add_argument("--strict_filter", action="store_true",
                        help="Step 0: also require at least one B-type error after an A-type error")
    parser.add_argument("--max_steps_after", type=int, default=12,
                        help="Max LLM turns to generate after t_A in the rerun (Step 5)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Δ threshold for edge validation")
    parser.add_argument("--min_n", type=int, default=1,
                        help="Min valid interventions for edge validation")
    # Skip flags for resuming partial runs
    parser.add_argument("--skip_filter", action="store_true")
    parser.add_argument("--skip_cases", action="store_true")
    parser.add_argument("--skip_patches", action="store_true")
    parser.add_argument("--skip_rerun", action="store_true")
    parser.add_argument("--skip_judge_a", action="store_true")
    parser.add_argument("--skip_judge_b", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- API connectivity test ---
    print("\n[API Test] Sending hello to model...")
    try:
        from patch_generator_llm import _call_llm
        reply = _call_llm(
            "You are a helpful assistant.",
            "Say exactly: API OK",
            model=args.model,
            max_tokens=10,
        )
        print(f"  Response: {reply!r}")
        print("  API connection: OK")
    except Exception as e:
        print(f"  API connection FAILED: {e}")
        print("  Fix the API key / model before continuing.")
        return 1

    # --eligible_file bypasses step 0
    if args.eligible_file:
        if not os.path.isfile(args.eligible_file):
            print(f"ERROR: --eligible_file not found: {args.eligible_file}")
            return 1
        eligible_path = args.eligible_file
        args.skip_filter = True
        with open(eligible_path) as _f:
            _n = len(json.load(_f).get("eligible", []))
        print(f"[Step 0] Skipped — using provided --eligible_file: {eligible_path} ({_n} traces)")
    else:
        eligible_path = os.path.join(os.path.dirname(os.path.abspath(args.causal_graph)),
                                     "eligible_traces.json")

    a_instances_path = _path(args.out_dir, "a_instances.jsonl")
    edge_pairs_path = _path(args.out_dir, "edge_pairs.jsonl")
    patches_path = _path(args.out_dir, "patch_results.jsonl")
    rerun_path = _path(args.out_dir, "rerun_results.jsonl")
    a_path = _path(args.out_dir, "a_resolved.jsonl")

    if not args.skip_filter:
        step0_filter_traces(args)
    else:
        if not args.eligible_file:
            print(f"[Step 0] Skipped — using {eligible_path}")

    if not args.skip_cases:
        step1_build_cases(args, eligible_path)
    else:
        print(f"[Step 1] Skipped — using {a_instances_path}, {edge_pairs_path}")

    if not args.skip_patches:
        step2_generate_patches(args, a_instances_path)
    else:
        print(f"[Step 2-4] Skipped — using {patches_path}")

    if not args.skip_rerun:
        step3_rerun(args, patches_path)
    else:
        print(f"[Step 5] Skipped — using {rerun_path}")

    if not args.skip_judge_a:
        step4_judge_a(args, rerun_path, patches_path, a_instances_path)
    else:
        print(f"[Step 6] Skipped — using {a_path}")

    if not args.skip_judge_b:
        step5_judge_b(args, rerun_path, a_path, edge_pairs_path)
    else:
        print("[Step 7] Skipped")

    step6_aggregate(args)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
