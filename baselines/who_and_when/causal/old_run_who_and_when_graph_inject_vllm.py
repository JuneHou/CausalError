"""
baselines/who_and_when/causal/run_who_and_when_graph_inject_vllm.py

+GI+SI (two-pass dynamic graph injection) applied to Who&When W1/W2.

Pipeline per trace:
  Pass 1 : Standard W1 (one call) or W2 (one call per span) over the trace.
           No graph guidance in Pass 1 — collect raw categories detected.
  Propagate: detected categories × graph edges → filtered subgraph.
  Pass 2 : Single trace-level targeted call (W1-style) injected with
           Pass 1 summary + filtered subgraph, asking only for NEW errors.
  Merge  : Pass 1 + Pass 2 errors deduped on (category, span_id).

Why a single Pass 2 call (not per-span Pass 2 on W2):
  Two-pass × per-span on W2 doubles the W2 call cost (2N calls/trace),
  which is already 9× a single-pass baseline. A trace-level Pass 2
  preserves the N+1 cost profile and matches the structural choice
  documented in paper/baseline_who_and_when.tex.

Output naming:
    outputs_{model}-{split}-who_and_when_{w1|w2}_graph_inject_{tag}{_span_index}/
where tag is `causal_only` or `causal_corr<corr>` or `suppes_t<threshold>`
or `random{n}_seed{seed}` (matching the main +GI+SI eval).

Usage (from baselines/who_and_when/causal/, GPUs 1,2,6,7):

  OUTPUT=/data/wang/junh/githubs/trail-benchmark/baselines/outputs

  # +GI+SI on W1 (Pass 1 = 1 call; Pass 2 = 1 call; ~2× baseline cost)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_graph_inject_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w1 --causal_only --span_index \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 131072 --output_dir $OUTPUT

  # +GI+SI on W2 (Pass 1 = N calls; Pass 2 = 1 call; ~(N+1)× baseline cost)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_graph_inject_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w2 --causal_only --span_index \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 32768 --output_dir $OUTPUT
"""

import os
import re
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

WAW_DIR   = Path(__file__).resolve().parent.parent
REPO      = WAW_DIR.parent.parent
BENCH_DIR = REPO / "benchmarking"
sys.path.insert(0, str(WAW_DIR))
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR / "eval"))

from run_who_and_when_vllm import (
    apply_chat_template,
    extract_span_ids,
    extract_task_description,
    format_trace_for_prompt,
    get_ordered_step_spans,
    parse_json_output,
    run_w1,
    run_w2,
)
from run_eval_graph_inject_vllm import (
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
    GRAPH_INJECT_TEMPLATE,
    TAXONOMY_BLOCK as MAIN_TAXONOMY_BLOCK,
    build_span_index,
    format_graph_guidance,
    load_graph_edges,
    propagate_confidence,
    validate_locations,
)


def _build_pass2_prompt(
    trace_str: str,
    pass1_errors: List[dict],
    filtered_edges: List[Tuple[str, str, float]],
    span_index: str = "",
) -> str:
    """Reuse the main eval's Pass-2 template for consistency.
    The template is task-agnostic (works on any error list as Pass-1 summary)."""
    pass1_summary = (
        "\n".join(f"  - {e['category']} at span {e.get('location', '?')}" for e in pass1_errors)
        if pass1_errors else "  (none)"
    )
    graph_text = "\n".join(
        f'  "{src}" → "{dst}"  [weight: {w:.2f}]' for src, dst, w in filtered_edges
    ) if filtered_edges else "  (no relevant edges)"
    span_index_block = (span_index + "\n\n") if span_index else ""
    return GRAPH_INJECT_TEMPLATE.format(
        taxonomy_block   = MAIN_TAXONOMY_BLOCK,
        pass1_summary    = pass1_summary,
        graph_text       = graph_text,
        span_index_block = span_index_block,
        trace            = trace_str,
    )


def main():
    parser = argparse.ArgumentParser(
        description="W1/W2 + two-pass graph injection (+GI+SI) via vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str,   default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--data_dir",               type=str,   default=str(BENCH_DIR / "data"))
    parser.add_argument("--output_dir",             type=str,   default=str(REPO / "baselines" / "outputs"))
    parser.add_argument("--split",                  type=str,   default="GAIA_dedup")
    parser.add_argument("--variant",                type=str,   default="w1", choices=["w1", "w2"])
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--causal_only",            action="store_true")
    parser.add_argument("--corr_threshold",         type=float, default=1.0,
                        help="Include causal + Suppes edges with geomean >= this. Set to e.g. 0.20 for the corr graph. Ignored if --causal_only.")
    parser.add_argument("--edge_threshold",         type=float, default=0.20,
                        help="Pure-Suppes threshold when neither --causal_only nor --corr_threshold<1 is set.")
    parser.add_argument("--random_edges",           action="store_true")
    parser.add_argument("--random_seed",            type=int,   default=42)
    parser.add_argument("--random_n",               type=int,   default=12)
    parser.add_argument("--propagation_threshold",  type=float, default=0.10)
    parser.add_argument("--span_index",             action="store_true", default=False)
    parser.add_argument("--validate_span_id",       action="store_true", default=True)
    parser.add_argument("--no_validate_span_id",    dest="validate_span_id", action="store_false")
    parser.add_argument("--causal_graph",           type=str,   default=None)
    parser.add_argument("--suppes_graph",           type=str,   default=None)
    args = parser.parse_args()

    is_reasoning_model = bool(re.search(r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)", args.model, re.IGNORECASE))
    if is_reasoning_model and args.max_new_tokens <= 8000:
        print(f"[INFO] Reasoning model detected ({args.model}); bumping max_new_tokens 8000 → 24000")
        args.max_new_tokens = 24000

    causal_graph_path = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH

    edges = load_graph_edges(
        threshold       = args.edge_threshold,
        causal_only     = args.causal_only,
        corr_threshold  = args.corr_threshold,
        causal_graph    = causal_graph_path,
        suppes_graph    = suppes_graph_path,
        random_edges    = args.random_edges,
        random_seed     = args.random_seed,
        random_n        = args.random_n,
    )
    if args.random_edges:
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
    elif args.causal_only:
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
    else:
        graph_tag = f"suppes_t{args.edge_threshold}"
    print(f"Loaded {len(edges)} edges ({graph_tag})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 10:
        print(f"    ... and {len(edges)-10} more")

    span_tag  = "_span_index" if args.span_index else ""
    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}_graph_inject_{graph_tag}{span_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"Found {len(file_paths)} traces in {data_dir}")
    print(f"Variant: {args.variant.upper()} +GI+SI ({graph_tag}) | Output → {out_dir}\n")

    is_mistral = "Mistral" in args.model or "mistral" in args.model
    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **({"fix_mistral_regex": True} if is_mistral else {}),
    )
    print(f"Loading model {args.model} ...")
    llm = LLM(
        model                  = args.model,
        tensor_parallel_size   = args.tensor_parallel_size,
        trust_remote_code      = True,
        max_model_len          = args.max_model_len,
        dtype                  = "bfloat16",
        gpu_memory_utilization = args.gpu_memory_utilization,
        enforce_eager          = args.enforce_eager,
    )

    skipped = 0
    p2_triggered_total = 0
    runner_p1 = run_w1 if args.variant == "w1" else run_w2

    for fp in tqdm(file_paths, desc=f"who_and_when_{args.variant}+GI+SI"):
        trace_id  = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")
        if os.path.exists(out_file):
            continue
        with open(fp) as f:
            trace_str = f.read()

        meta = {
            "trace_id": trace_id,
            "variant": args.variant,
            "graph": graph_tag,
            "pass1_detected": [],
            "pass2_triggered": False,
            "pass2_filtered_edges": 0,
            "pass2_new_errors": 0,
            "pass2_parse_failed": False,
            "p2_dropped": 0,
        }

        # ----- Pass 1 — vanilla W1 / W2, no graph in prompt -----
        p1_output, p1_meta = runner_p1(
            trace_str, llm, tokenizer,
            args.max_model_len, args.max_new_tokens,
        )
        meta["pass1_meta"] = p1_meta
        if p1_output is None:
            output = {"errors": [], "scores": [], "_error": p1_meta.get("error", "pass1_failed")}
            with open(out_file, "w") as f:  json.dump(output, f, indent=2)
            with open(meta_file, "w") as f: json.dump(meta, f, indent=2)
            skipped += 1
            continue
        p1_errors = p1_output.get("errors", [])
        p1_scores = p1_output.get("scores", [])
        detected_cats = list({e["category"] for e in p1_errors})
        meta["pass1_detected"] = detected_cats

        # ----- Pass 2 — trace-level targeted call with filtered subgraph -----
        p2_errors: List[dict] = []
        if detected_cats and edges:
            filtered_edges = propagate_confidence(detected_cats, edges, args.propagation_threshold)
            meta["pass2_filtered_edges"] = len(filtered_edges)
            if filtered_edges:
                meta["pass2_triggered"] = True
                p2_triggered_total += 1
                span_index_text = build_span_index(trace_str) if args.span_index else ""
                p2_user_text   = _build_pass2_prompt(trace_str, p1_errors, filtered_edges, span_index=span_index_text)
                p2_prompt_text = apply_chat_template(tokenizer, p2_user_text)

                tok_len_p2 = len(tokenizer.encode(p2_prompt_text, add_special_tokens=False))
                _avail_p2  = args.max_model_len - tok_len_p2
                if _avail_p2 < 2048:
                    meta["pass2_error"] = "context_overflow"
                else:
                    sp2 = SamplingParams(temperature=0.0, max_tokens=min(args.max_new_tokens, _avail_p2))
                    try:
                        p2_raw = llm.generate([p2_prompt_text], sp2)[0].outputs[0].text
                        p2_parsed = parse_json_output(p2_raw)
                        if p2_parsed is None:
                            meta["pass2_parse_failed"] = True
                        else:
                            p2_errors = p2_parsed.get("errors", [])
                            if args.validate_span_id:
                                valid_ids = extract_span_ids(trace_str)
                                p2_errors, p2_dropped = validate_locations(p2_errors, valid_ids)
                                meta["p2_dropped"] = p2_dropped
                            p1_cats = {e["category"] for e in p1_errors}
                            p2_errors = [e for e in p2_errors if e.get("category") not in p1_cats]
                            meta["pass2_new_errors"] = len(p2_errors)
                    except Exception as e:
                        meta["pass2_error"] = str(e)

        merged = p1_errors + p2_errors
        output = {"errors": merged, "scores": p1_scores}
        with open(out_file, "w") as f:  json.dump(output, f, indent=2)
        with open(meta_file, "w") as f: json.dump(meta, f, indent=2)

    n_done = len(file_paths) - skipped
    print(f"\nDone. {n_done} processed, {skipped} skipped.")
    print(f"Pass 2 triggered for {p2_triggered_total} traces.")
    print(f"Score with (from benchmarking/):")
    print(f"  python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
