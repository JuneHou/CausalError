#!/usr/bin/env python3
"""
Replicate full GAIA traces by re-running every LLM call in each trace with the same
prompt and settings (o3-mini, temperature=0).

- One data point in GAIA = one trace file = one full agent run (many LLM steps).
- A trace = the full execution log: all LLM calls, tool use, agent steps in order.
- Replication = replay each LLM step in the trace with the exact same input messages
  and temp=0, then compare outputs to the original trace (and to TRAIL annotations).
  Note: This replays only LLM calls (no tool execution). To replicate the full agent
  run including tools you would need the original agent code (e.g. OpenDeepResearch).

1. Randomly samples 3-5 GAIA trace files (data points) from data/GAIA.
2. Extracts the full trace config: every LLM step in order (span_id, model, messages,
   original_output from the trace).
3. For each step, calls the model with the same messages and temperature=0; collects
   replicated output.
4. Saves per-trace: replicated outputs for all steps, plus comparison to original
   so you can check if the error trace replicates.

Usage (from benchmarking/):
  python replication.py
  python replication.py --data_dir data --output_dir replication_outputs --n_sample 5
  python replication.py --trace_ids id1,id2 --data_dir data --output_dir replication_outputs
  python replication.py --max_steps 5   # cap steps per trace (for quick runs)
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from extract_run_config import extract_run_config

try:
    import litellm
    from litellm import completion, RateLimitError
except ImportError:
    litellm = None


def _gaia_trace_paths(data_dir: str) -> List[str]:
    gaia_dir = os.path.join(data_dir, "GAIA")
    if not os.path.isdir(gaia_dir):
        return []
    return [
        os.path.join(gaia_dir, f)
        for f in os.listdir(gaia_dir)
        if f.endswith(".json")
    ]


def sample_gaia_traces(
    data_dir: str,
    n_sample: int = 5,
    trace_ids: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Return list of paths to GAIA trace JSON files.
    If trace_ids is given, use those (ids with or without .json); otherwise random sample of n_sample.
    """
    paths = _gaia_trace_paths(data_dir)
    if not paths:
        return []

    if trace_ids is not None:
        by_basename = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        out = []
        for tid in trace_ids:
            tid_clean = tid.replace(".json", "")
            if tid_clean in by_basename:
                out.append(by_basename[tid_clean])
        return out

    if seed is not None:
        random.seed(seed)
    n = min(n_sample, len(paths))
    return random.sample(paths, n)


# Trace (OpenInference/smolagents) uses roles tool-call / tool-response that OpenAI rejects
# unless preceded by assistant with tool_calls (trace doesn't store that structure).
# "tool" role requires a preceding assistant message with "tool_calls"; trace doesn't store that.
# Use "user" for tool-call/tool-response so content is unchanged and API accepts the request.
_TRACE_ROLE_TO_API: Dict[str, str] = {
    "tool-call": "user",
    "tool-response": "user",
}


def _messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert trace message roles to API-valid roles. Content from trace only (no injection).
    tool-call and tool-response are sent as 'user' so we never send 'tool' without tool_calls."""
    out = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        api_role = _TRACE_ROLE_TO_API.get(role, role)
        content = m.get("content")
        if content is None:
            content = ""
        out.append({"role": api_role, "content": content})
    return out


def run_one_llm_step(
    messages: List[Dict[str, str]],
    model: str,
    temperature: int = 0,
    model_override: Optional[str] = None,
) -> str:
    """
    Run a single LLM call with the given messages and model. Uses temperature=0 for replication.
    Trace roles (e.g. tool-call, tool-response) are mapped to API-valid roles; content is unchanged.
    """
    if litellm is None:
        raise RuntimeError("litellm is required: pip install litellm")

    use_model = model_override or model
    if use_model == "o3-mini":
        use_model = "openai/o3-mini"

    api_messages = _messages_for_api(messages)

    params = {
        "messages": api_messages,
        "model": use_model,
        "temperature": temperature,
        "max_completion_tokens": 8192,
        "drop_params": True,
    }
    if "o3" in use_model or "o1" in use_model or "o4" in use_model:
        params["reasoning_effort"] = "high"

    try:
        response = completion(**params)
        return (response.choices or [{}])[0].message.get("content", "") or ""
    except RateLimitError as e:
        print(f"  Rate limit: {e}. Sleeping 60s...")
        time.sleep(60)
        return completion(**params).choices[0].message.get("content", "") or ""


def main():
    parser = argparse.ArgumentParser(
        description="Replicate GAIA traces with o3-mini and temperature=0"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing GAIA/ subdir with trace JSONs",
    )
    parser.add_argument(
        "--output_dir",
        default="replication_outputs",
        help="Where to write extracted configs and replication responses",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        help="Number of GAIA traces to randomly sample (default 5)",
    )
    parser.add_argument(
        "--trace_ids",
        type=str,
        default=None,
        help="Comma-separated trace IDs to use instead of random sample (e.g. fcdcb46c7df316b571138b53bd3c822a)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override (e.g. openai/o3-mini). If not set, model from each trace step is used.",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=0,
        help="Temperature for replication (default 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only extract configs and print; do not call LLM",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max LLM steps per trace to replicate (default: all). Use for quick runs.",
    )
    args = parser.parse_args()

    trace_ids_list = None
    if args.trace_ids:
        trace_ids_list = [s.strip() for s in args.trace_ids.split(",") if s.strip()]

    paths = sample_gaia_traces(
        args.data_dir,
        n_sample=args.n_sample,
        trace_ids=trace_ids_list,
        seed=args.seed,
    )
    if not paths:
        print(f"No GAIA traces found under {args.data_dir}/GAIA/")
        return

    print(f"Using {len(paths)} trace(s) (each = one data point = full trace):")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for trace_path in paths:
        trace_id = os.path.splitext(os.path.basename(trace_path))[0]
        print(f"\n--- {trace_id} ---")
        try:
            config = extract_run_config(trace_path)
        except Exception as e:
            print(f"  Extract error: {e}")
            results.append({"trace_id": trace_id, "error": str(e)})
            continue

        llm_steps = config.get("llm_steps") or []
        if not llm_steps:
            print(f"  No LLM steps found in trace (not GAIA or empty). Skipping.")
            results.append({"trace_id": trace_id, "error": "no_llm_steps"})
            continue

        n_total = len(llm_steps)
        steps_to_run = llm_steps
        if args.max_steps is not None and args.max_steps < n_total:
            steps_to_run = llm_steps[: args.max_steps]
            print(f"  Replicating {len(steps_to_run)} of {n_total} LLM steps (--max_steps={args.max_steps})")
        else:
            print(f"  Replicating all {n_total} LLM steps in this trace.")

        # Save full extracted config (includes all llm_steps)
        config_path = os.path.join(args.output_dir, f"{trace_id}_config.json")
        # Don't write huge messages in config JSON; keep llm_steps for replay but save compact
        config_export = {k: v for k, v in config.items() if k != "llm_steps"}
        config_export["num_llm_steps"] = n_total
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_export, f, indent=2)
        print(f"  Saved config to {config_path}")

        if args.dry_run:
            results.append({
                "trace_id": trace_id,
                "config_path": config_path,
                "num_llm_steps": n_total,
            })
            continue

        # Replay each LLM step with same messages and temp=0
        replicated_steps: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps_to_run):
            span_id = step.get("span_id", "")
            messages = step.get("messages", [])
            original_output = step.get("original_output", "")
            model = step.get("model")
            if not model:
                raise ValueError(f"Trace {trace_id} step {idx}: no model in trace (llm.model_name missing)")
            print(f"  Step {idx + 1}/{len(steps_to_run)} (span_id={span_id[:8]}...)")
            try:
                replicated_output = run_one_llm_step(
                    messages,
                    model=model,
                    temperature=args.temperature,
                    model_override=args.model,
                )
            except Exception as e:
                print(f"    LLM error: {e}")
                replicated_steps.append({
                    "step_index": idx,
                    "span_id": span_id,
                    "original_output": original_output[:500] + "..." if len(original_output) > 500 else original_output,
                    "replicated_output": "",
                    "error": str(e),
                })
                continue
            replicated_steps.append({
                "step_index": idx,
                "span_id": span_id,
                "original_output": original_output,
                "replicated_output": replicated_output,
            })

        # Save full replicated trace for this data point
        replicated_path = os.path.join(args.output_dir, f"{trace_id}_replicated_trace.json")
        with open(replicated_path, "w", encoding="utf-8") as f:
            json.dump({
                "trace_id": trace_id,
                "num_steps": len(replicated_steps),
                "steps": replicated_steps,
            }, f, indent=2)
        print(f"  Full replicated trace saved to {replicated_path}")

        # Also write a short comparison summary (first 200 chars of each output per step)
        summary_steps = []
        for s in replicated_steps:
            summary_steps.append({
                "step_index": s["step_index"],
                "span_id": s.get("span_id", ""),
                "original_preview": (s.get("original_output") or "")[:500],
                "replicated_preview": (s.get("replicated_output") or s.get("error") or "")[:500],
            })
        comparison_path = os.path.join(args.output_dir, f"{trace_id}_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump({"trace_id": trace_id, "steps": summary_steps}, f, indent=2)

        results.append({
            "trace_id": trace_id,
            "config_path": config_path,
            "replicated_trace_path": replicated_path,
            "comparison_path": comparison_path,
            "num_steps_replicated": len(replicated_steps),
            "num_steps_total": n_total,
        })

    summary_path = os.path.join(args.output_dir, "replication_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
