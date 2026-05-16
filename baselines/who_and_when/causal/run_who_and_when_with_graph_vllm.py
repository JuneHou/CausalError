"""
baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py

+CG (one-pass causal graph in prompt) applied to Who&When W1/W2 prompts.

Mirrors benchmarking/eval/run_eval_with_graph_vllm.py but uses the W1
(all-at-once) / W2 (step-by-step) prompt skeletons from
baselines/who_and_when/run_who_and_when_vllm.py instead of the TRAIL prompt.
The graph guidance block is injected once per call:
  W1: once in the single full-trace prompt.
  W2: once in every per-step prompt (so each span call sees the graph).

Output naming:
    outputs_{model}-{split}-who_and_when_{w1|w2}_graph_{tag}{_span_index}/
where tag is `causal_only` or `t<edge_threshold>`.

Usage (from baselines/who_and_when/causal/, GPUs 1,2,6,7):

  OUTPUT=/data/wang/junh/githubs/trail-benchmark/baselines/outputs

  # +CG on W1 (single call per trace)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_with_graph_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w1 --causal_only \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 131072 --output_dir $OUTPUT

  # +CG on W2 (one call per span; graph appears in every call)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_with_graph_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w2 --causal_only \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 32768 --output_dir $OUTPUT

  # Score
  python /data/wang/junh/githubs/trail-benchmark/benchmarking/eval/calculate_scores.py \\
      --results_dir $OUTPUT
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

# Make sibling helpers + main-eval graph utilities importable.
WAW_DIR   = Path(__file__).resolve().parent.parent
REPO      = WAW_DIR.parent.parent
BENCH_DIR = REPO / "benchmarking"
sys.path.insert(0, str(WAW_DIR))
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR / "eval"))

from run_who_and_when_vllm import (
    TAXONOMY_BLOCK,
    apply_chat_template,
    extract_span_ids,
    extract_task_description,
    format_trace_for_prompt,
    get_ordered_step_spans,
    parse_json_output,
    run_scores_call,
)
from run_eval_with_graph_vllm import (
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
    format_graph_guidance,
)
# load_graph_edges is imported from the +GI eval because that copy carries the
# full --corr_threshold / --random_edges surface that we added to this runner's
# flags; the +CG eval's copy still uses the legacy (causal_only, threshold)
# signature. format_graph_guidance stays here — it's +CG-specific (one-pass
# prompt block), independent of which loader produced the edges.
from run_eval_graph_inject_vllm import build_span_index, load_graph_edges


# ---------------------------------------------------------------------------
# W1 / W2 prompt templates with a graph-guidance slot
# ---------------------------------------------------------------------------
#
# Identical to the templates in run_who_and_when_vllm.py except the
# `{graph_guidance_block}` slot is inserted between the taxonomy and the
# task instructions. Keeping the templates here lets us evolve the
# graph-injected variant without touching the no-graph runner.

W1_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with analyzing an AI agent execution trace when solving a real-world problem.
The problem is: {task_description}

{taxonomy_block}

{graph_guidance_block}\
{span_index_block}\
Identify which error categories from the taxonomy above are present in the trace, at which span, and explain the reason for each error.

Here's the trace:

{trace}

Based on this trace, please predict the following:
1. For each error category from the 19 categories above that is present in the trace, the exact category name. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors"). Zero, one, or multiple categories may be present. If no categories are clearly present, return an empty list.
2. The span_id where each predicted error first occurs (use the exact hex string copied from the trace).
3. A brief evidence quote and one-sentence reason for each predicted error.
4. The impact severity for each predicted error: HIGH, MEDIUM, or LOW.

Please answer in strictly valid JSON with no markdown, no explanation, no preamble:

{{
    "errors": [
        {{
            "category": "<exact category name from the taxonomy>",
            "location": "<exact span_id hex string>",
            "evidence": "<brief quote from the trace>",
            "description": "<one-sentence description>",
            "impact": "<HIGH, MEDIUM, or LOW>"
        }}
    ],
    "scores": []
}}

If no errors are present: {{"errors": [], "scores": []}}\
"""

W2_STEP_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with evaluating the correctness of each step in an ongoing AI agent execution aimed at solving a real-world problem. The problem being addressed is: {task_description}.

{taxonomy_block}

{graph_guidance_block}\
Here is the execution history up to the current step:
{cumulative_spans}

The most recent step (Step {step_num}, span_id "{span_id}") was: {step_name}.

Your task is to determine whether this most recent step (Step {step_num}) contains any errors from the taxonomy above that could hinder the problem-solving process or lead to an incorrect solution. Zero, one, or multiple error categories from the taxonomy may be present in this single step. Please respond with the list of error categories present (or an empty list if none) and a clear explanation for your judgment.

Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.

For each error you find, also rate its impact severity as HIGH, MEDIUM, or LOW.

Respond ONLY in strictly valid JSON:
{{
  "step_id": {step_num},
  "span_id": "{span_id}",
  "has_error": true/false,
  "errors": [
    {{"category": "<exact category name from the taxonomy>", "evidence": "<brief quote>", "description": "<one sentence>", "impact": "<HIGH, MEDIUM, or LOW>"}}
  ]
}}
If no errors: {{"step_id": {step_num}, "span_id": "{span_id}", "has_error": false, "errors": []}}\
"""


def _graph_block(graph_guidance: str) -> str:
    return (graph_guidance + "\n") if graph_guidance else ""


# ---------------------------------------------------------------------------
# Variant W1 + graph
# ---------------------------------------------------------------------------

def run_w1_with_graph(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    graph_guidance: str,
    span_index: str = "",
) -> Tuple[Optional[dict], dict]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)
    trace_text    = format_trace_for_prompt(ordered_spans)

    span_index_block = (span_index + "\n\n") if span_index else ""
    user_text = W1_PROMPT_TEMPLATE.format(
        taxonomy_block       = TAXONOMY_BLOCK,
        graph_guidance_block = _graph_block(graph_guidance),
        span_index_block     = span_index_block,
        task_description     = task_desc,
        trace                = trace_text,
    )
    prompt_text = apply_chat_template(tokenizer, user_text)

    tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    if tok_len + 8192 > max_model_len:
        return None, {"error": "context_overflow", "tok_len": tok_len}

    avail = max_model_len - tok_len
    sp = SamplingParams(temperature=0.0, max_tokens=min(max_new_tokens, avail))
    try:
        raw = llm.generate([prompt_text], sp)[0].outputs[0].text
    except Exception as e:
        return None, {"error": str(e)}

    parsed = parse_json_output(raw)
    if parsed is None:
        return None, {"error": "json_parse_failed", "raw": raw[:500]}

    errors = parsed.get("errors", [])
    errors = [e for e in errors if (e.get("location") or "").strip() in valid_ids]

    scores_budget = min(2048, max_new_tokens)
    scores, scores_meta = run_scores_call(
        trace_text, llm, tokenizer, max_model_len, scores_max_tokens=scores_budget,
    )
    return (
        {"errors": errors, "scores": scores},
        {"tok_len": tok_len, "n_raw_errors": len(parsed.get("errors", [])), **scores_meta},
    )


# ---------------------------------------------------------------------------
# Variant W2 + graph
# ---------------------------------------------------------------------------

def run_w2_with_graph(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    graph_guidance: str,
) -> Tuple[Optional[dict], dict]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)

    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

    meta = {"calls": 0, "error": None}
    per_step_budget = min(4096, max(512, max_new_tokens // 8)) if max_new_tokens else 512
    full_trace_text = format_trace_for_prompt(ordered_spans)

    all_errors: List[dict] = []
    seen_pairs: set = set()
    cumulative_text = ""

    graph_block = _graph_block(graph_guidance)

    for i, entry in enumerate(ordered_spans):
        step_num = i + 1
        step_name = entry["name"]
        span_id   = entry["span_id"]
        span_content = entry["content"]

        cumulative_text += f"\n--- Step {step_num}: {step_name} (span_id: \"{span_id}\") ---\n{span_content}\n"

        user_text = W2_STEP_PROMPT_TEMPLATE.format(
            taxonomy_block       = TAXONOMY_BLOCK,
            graph_guidance_block = graph_block,
            task_description     = task_desc,
            step_num             = step_num,
            step_name            = step_name,
            span_id              = span_id,
            cumulative_spans     = cumulative_text.strip(),
        )
        prompt_text = apply_chat_template(tokenizer, user_text)

        tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if tok_len + per_step_budget > max_model_len:
            meta["error"] = f"context_overflow_at_step_{step_num}"
            break

        sp = SamplingParams(temperature=0.0, max_tokens=per_step_budget)
        meta["calls"] += 1
        try:
            raw = llm.generate([prompt_text], sp)[0].outputs[0].text.strip()
        except Exception as e:
            meta["error"] = str(e)
            break

        parsed = parse_json_output(raw)
        if parsed is None or not parsed.get("has_error"):
            continue

        for err in parsed.get("errors", []):
            category = (err.get("category") or "").strip()
            evidence = err.get("evidence", "")
            description = err.get("description", f"Error at step {step_num} ({step_name}).")
            impact = err.get("impact", "")
            pair_key = (category, span_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if span_id in valid_ids:
                all_errors.append({
                    "category": category,
                    "location": span_id,
                    "evidence": evidence,
                    "description": description,
                    "impact": impact,
                })

    scores_budget = min(2048, max_new_tokens) if max_new_tokens else 512
    scores, scores_meta = run_scores_call(
        full_trace_text, llm, tokenizer, max_model_len, scores_max_tokens=scores_budget,
    )
    meta["calls"] += 1
    meta.update(scores_meta)
    return {"errors": all_errors, "scores": scores}, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="W1/W2 + causal graph (one-pass +CG) via vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str,   default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--data_dir",               type=str,   default=str(BENCH_DIR / "data"))
    parser.add_argument("--output_dir",             type=str,   default=str(REPO / "baselines" / "who_and_when" / "causal" / "outputs"))
    parser.add_argument("--split",                  type=str,   default="GAIA_dedup")
    parser.add_argument("--variant",                type=str,   default="w1", choices=["w1", "w2"])
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--causal_only",            action="store_true", default=False,
                        help="Use only the intervention-validated causal edges.")
    parser.add_argument("--corr_threshold",         type=float, default=1.0,
                        help="Causal-union threshold: include intervention-validated causal edges UNION Suppes edges with geomean sqrt(precedence*PR_delta) >= this. Set e.g. 0.35 for the corr graph. Ignored if --causal_only.")
    parser.add_argument("--edge_threshold",         type=float, default=0.20,
                        help="Pure-Suppes threshold (no causal union) using the same geomean sqrt(precedence*PR_delta) score. Only used when neither --causal_only nor --corr_threshold<1 is set.")
    parser.add_argument("--random_edges",           action="store_true",
                        help="Sample N random non-Suppes edges as a null-graph control.")
    parser.add_argument("--random_seed",            type=int,   default=42)
    parser.add_argument("--random_n",               type=int,   default=12)
    parser.add_argument("--causal_graph",           type=str,   default=None)
    parser.add_argument("--suppes_graph",           type=str,   default=None)
    parser.add_argument("--span_index",             action="store_true", default=False,
                        help="Inject a span-id index listing into W1 prompts (W2 omits it by design).")
    args = parser.parse_args()

    # Reasoning models need bigger output budget (same heuristic as the no-graph runner).
    is_reasoning_model = bool(re.search(r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)", args.model, re.IGNORECASE))
    if is_reasoning_model and args.max_new_tokens <= 8000:
        print(f"[INFO] Reasoning model detected ({args.model}); bumping max_new_tokens 8000 → 24000")
        args.max_new_tokens = 24000

    causal_graph = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        causal_only    = args.causal_only,
        threshold      = args.edge_threshold,
        corr_threshold = args.corr_threshold,
        causal_graph   = causal_graph,
        suppes_graph   = suppes_graph,
        random_edges   = args.random_edges,
        random_seed    = args.random_seed,
        random_n       = args.random_n,
    )
    if args.random_edges:
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
    elif args.causal_only:
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
    else:
        graph_tag = f"t{args.edge_threshold}"
    graph_guidance = format_graph_guidance(edges, causal_only=args.causal_only)
    print(f"Loaded {len(edges)} edges ({graph_tag})")
    span_tag  = "_span_index" if args.span_index else ""
    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}_graph_{graph_tag}{span_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"Found {len(file_paths)} traces in {data_dir}")
    print(f"Variant: {args.variant.upper()} +CG ({graph_tag}) | Output → {out_dir}\n")

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

    runner = run_w1_with_graph if args.variant == "w1" else run_w2_with_graph
    skipped = 0
    for fp in tqdm(file_paths, desc=f"who_and_when_{args.variant}+graph"):
        trace_id  = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")
        if os.path.exists(out_file):
            continue
        with open(fp) as f:
            trace_str = f.read()
        # W1 injects span_index if requested; W2 omits it by design (same as +GI runner).
        span_index_text = build_span_index(trace_str) if (args.span_index and args.variant == "w1") else ""
        if args.variant == "w1":
            output, meta = runner(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens, graph_guidance, span_index_text,
            )
        else:
            output, meta = runner(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens, graph_guidance,
            )
        meta["trace_id"] = trace_id
        meta["variant"]  = args.variant
        meta["graph"]    = graph_tag
        if output is None:
            output = {"errors": [], "scores": [], "_error": meta.get("error", "unknown")}
            skipped += 1
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    n_done = len(file_paths) - skipped
    print(f"\nDone. {n_done} processed, {skipped} failed/skipped.")
    print(f"Score with (from benchmarking/):")
    print(f"  python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
