"""
baselines/who_and_when/causal/run_who_and_when_causal_vllm.py

Who&When W2 / W3 with TRAIL causal-graph injection.

Two new variants, both reusing the graph artifacts already used by
benchmarking/eval/run_eval_graph_inject_vllm.py:

  w2_graph — Step-by-step (run_w2) + STATEFUL graph hint per step.
             At step i, propagate confidence from categories detected in
             steps 1..i-1 through the causal/Suppes graph; prepend the
             filtered subgraph and the running detected-set to the per-step
             prompt. Uses the precedence-filtered graph the way it was
             designed: A precedes B → if A was just seen, look for B now.

  w3_graph — Per-label bisection, but only for labels selected by the graph.
             1. Run a holistic Pass-1 (W1-shape) to detect source categories.
             2. Propagate over edges → target labels above
                propagation_threshold.
             3. Run _bisect_label only for those target labels (typically
                5–8 labels instead of all 19). Pass-1 errors are merged with
                the bisection findings.

Outputs:
    baselines/who_and_when/causal/outputs/
        outputs_{model_tag}-{split}-who_and_when_{variant}_{graph_tag}/

Run commands (from anywhere; paths are absolute-resolved):

  CAUSAL_DIR=/data/wang/junh/githubs/trail-benchmark/baselines/who_and_when/causal

  # W2 + graph (stateful propagation across steps)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python $CAUSAL_DIR/run_who_and_when_causal_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w2_graph \\
      --causal_only \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 32768

  # W3 + graph (label prefilter via Pass-1 + propagation)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python $CAUSAL_DIR/run_who_and_when_causal_vllm.py \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --split GAIA_dedup --variant w3_graph \\
      --causal_only \\
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \\
      --max_model_len 32768
"""

import json
import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Path setup ------------------------------------------------------------
THIS_DIR  = Path(__file__).resolve().parent           # .../who_and_when/causal
WW_DIR    = THIS_DIR.parent                           # .../who_and_when
ROOT_DIR  = WW_DIR.parent.parent                      # .../trail-benchmark
BENCH_DIR = ROOT_DIR / "benchmarking"
EVAL_DIR  = BENCH_DIR / "eval"

for p in (BENCH_DIR, WW_DIR, EVAL_DIR):
    sys.path.insert(0, str(p))

# --- Reused helpers --------------------------------------------------------
from run_who_and_when_vllm import (
    TAXONOMY_BLOCK,
    TAXONOMY_LEAF_CATEGORIES,
    LABEL_DEFINITIONS,
    W1_PROMPT_TEMPLATE,
    W3_BISECT_PER_LABEL_TEMPLATE,
    W3_LEAF_PROMPT_TEMPLATE,
    build_span_index,
    get_ordered_step_spans,
    extract_task_description,
    extract_span_ids,
    parse_json_output,
    apply_chat_template,
    format_trace_for_prompt,
    _closest_category,
    run_scores_call,
)

from run_eval_graph_inject_vllm import (
    load_graph_edges,
    propagate_confidence,
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
)


# ---------------------------------------------------------------------------
# Per-step W2 graph-hint template
# ---------------------------------------------------------------------------

W2_GRAPH_STEP_PROMPT_TEMPLATE = """\
You are an expert evaluator of AI agent execution traces.

{taxonomy_block}

{graph_hint_block}You are examining an agent execution trace step by step.

Here is the execution so far, up to and including the CURRENT step (Step {step_num}):

{cumulative_spans}

Your task: determine whether the CURRENT STEP (Step {step_num}: {step_name}) contains
any errors from the taxonomy above. This is a multi-label task — a single step may have
zero, one, or multiple error types. Predict an error only if the current step itself
provides direct evidence.

Valid category names (use EXACTLY as written — do not paraphrase):
Language-only, Tool-related, Poor Information Retrieval, Tool Output Misinterpretation,
Incorrect Problem Identification, Tool Selection Errors, Formatting Errors,
Instruction Non-compliance, Tool Definition Issues, Environment Setup Errors,
Rate Limiting, Authentication Errors, Service Errors, Resource Not Found,
Resource Exhaustion, Timeout Issues, Context Handling Failures,
Resource Abuse, Goal Deviation, Task Orchestration

Output strictly valid JSON:
{{
  "step_id": {step_num},
  "span_id": "{span_id}",
  "has_error": true/false,
  "errors": [
    {{"category": "<exact category name>", "evidence": "<brief quote>", "description": "<one sentence>"}}
  ]
}}
If no errors: {{"step_id": {step_num}, "span_id": "{span_id}", "has_error": false, "errors": []}}\
"""


def _stateful_graph_hint(
    detected_so_far: List[str],
    filtered_edges: List[Tuple[str, str, float]],
    causal_only: bool,
) -> str:
    """Format the per-step hint for W2_GRAPH. Empty if nothing is detected yet
    or no edges are propagated."""
    if not detected_so_far or not filtered_edges:
        return ""
    header = (
        "# Causal Graph Hint (intervention-validated)"
        if causal_only
        else "# Causal Graph Hint (observational, precedence-filtered)"
    )
    detected_str = ", ".join(sorted(set(detected_so_far)))
    target_set = sorted({dst for _, dst, _ in filtered_edges})
    lines = [
        header,
        f"In earlier steps you have already detected: {detected_str}.",
        "Statistical analysis of agent traces shows the following error types are",
        "likely to follow what was already seen. Pay particular attention to",
        f"whether the CURRENT step contains any of: {', '.join(target_set)}.",
        "",
        "Format: [Source Error] → [Consequent Error]  (effect: X.XX)",
        "",
    ]
    for src, dst, w in filtered_edges:
        lines.append(f"  {src} → {dst}  (effect: {w:.2f})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# W2 + graph (stateful)
# ---------------------------------------------------------------------------

def run_w2_graph(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
    causal_only: bool,
    max_spans: int = 20,
) -> Tuple[Optional[dict], dict]:
    """Step-by-step W2 with cumulative-detected → propagated graph hint per step."""
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc = extract_task_description(trace_str)  # noqa: F841 — kept for parity / future use
    valid_span_ids = extract_span_ids(trace_str)

    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

    ordered_spans = ordered_spans[:max_spans]
    full_trace_text = format_trace_for_prompt(ordered_spans)

    meta: Dict = {
        "calls": 0,
        "error": None,
        "n_steps_scanned": 0,
        "per_step_filtered_edges": [],
        "per_step_new_detections": [],
        "cumulative_detected": [],
    }

    all_errors: List[dict] = []
    seen_pairs: set = set()
    detected_so_far: List[str] = []
    cumulative_text = ""

    for i, entry in enumerate(ordered_spans):
        step_num = i + 1
        step_name = entry["name"]
        span_id = entry["span_id"]
        span_content = entry["content"]

        cumulative_text += (
            f"\n--- Step {step_num}: {step_name} (span_id: \"{span_id}\") ---\n"
            f"{span_content}\n"
        )

        # Propagate over already-detected categories → graph hint for this step
        filtered_edges = (
            propagate_confidence(detected_so_far, edges, propagation_threshold)
            if detected_so_far else []
        )
        graph_hint = _stateful_graph_hint(detected_so_far, filtered_edges, causal_only)
        graph_hint_block = (graph_hint + "\n") if graph_hint else ""
        meta["per_step_filtered_edges"].append(len(filtered_edges))

        user_text = W2_GRAPH_STEP_PROMPT_TEMPLATE.format(
            taxonomy_block=TAXONOMY_BLOCK,
            graph_hint_block=graph_hint_block,
            step_num=step_num,
            step_name=step_name,
            span_id=span_id,
            cumulative_spans=cumulative_text.strip(),
        )
        prompt_text = apply_chat_template(tokenizer, user_text)

        tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if tok_len + 1024 > max_model_len:
            meta["error"] = f"context_overflow_at_step_{step_num}"
            break

        sp = SamplingParams(temperature=0.0, max_tokens=512)
        meta["calls"] += 1
        try:
            raw = llm.generate([prompt_text], sp)[0].outputs[0].text.strip()
        except Exception as e:
            meta["error"] = str(e)
            break

        meta["n_steps_scanned"] = step_num
        parsed = parse_json_output(raw)
        if parsed is None or not parsed.get("has_error"):
            meta["per_step_new_detections"].append([])
            continue

        new_cats: List[str] = []
        for err in parsed.get("errors", []):
            category = _closest_category(err.get("category", ""))
            evidence = err.get("evidence", "")
            description = err.get("description", f"Error at step {step_num} ({step_name}).")
            pair_key = (category, span_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if span_id in valid_span_ids:
                all_errors.append({
                    "category": category,
                    "location": span_id,
                    "evidence": evidence,
                    "description": description,
                    "impact": "HIGH",
                })
                new_cats.append(category)

        meta["per_step_new_detections"].append(new_cats)
        # Update running set BEFORE next step so propagation reflects what we know now
        for c in new_cats:
            if c not in detected_so_far:
                detected_so_far.append(c)

    meta["cumulative_detected"] = detected_so_far

    # Trailing rubric-scores call (Option A) — same pattern as run_w2
    scores, scores_meta = run_scores_call(full_trace_text, llm, tokenizer, max_model_len)
    meta["calls"] += 1
    meta.update(scores_meta)

    return {"errors": all_errors, "scores": scores}, meta


# ---------------------------------------------------------------------------
# W3 + graph (label prefilter)
# ---------------------------------------------------------------------------

def _bisect_label(
    label: str,
    spans: List[dict],
    global_offset: int,
    *,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    task_desc: str,
    valid_span_ids: Dict[str, str],
    meta: Dict,
) -> List[str]:
    """Recursively bisect a single label over an ordered span list. Returns
    the list of span_ids where the label is positively located. Mirrors the
    closure inside run_w3 in run_who_and_when_vllm.py but lifted to module
    scope for reuse."""

    def _format_window(spans_slice: List[dict], offset: int) -> str:
        lines = []
        for j, entry in enumerate(spans_slice):
            snum = offset + j + 1
            lines.append(
                f"--- Step {snum}: {entry['name']} (span_id: \"{entry['span_id']}\") ---"
            )
            lines.append(entry["content"][:600])
        return "\n".join(lines)

    if not spans:
        return []

    if len(spans) == 1:
        entry = spans[0]
        step_num = global_offset + 1
        user_text = W3_LEAF_PROMPT_TEMPLATE.format(
            task_description=task_desc,
            label=label,
            label_definition=LABEL_DEFINITIONS.get(label, ""),
            step_num=step_num,
            span_id=entry["span_id"],
            step_name=entry["name"],
            span_content=entry["content"][:2000],
        )
        prompt = apply_chat_template(tokenizer, user_text)
        tok_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tok_len + 128 > max_model_len:
            return []
        sp = SamplingParams(temperature=0.0, max_tokens=128)
        meta["calls"] += 1
        try:
            raw = llm.generate([prompt], sp)[0].outputs[0].text
        except Exception as e:
            meta["error"] = str(e)
            return []
        parsed = parse_json_output(raw)
        if parsed and parsed.get("present") and entry["span_id"] in valid_span_ids:
            return [entry["span_id"]]
        return []

    mid = len(spans) // 2
    lower = spans[:mid]
    upper = spans[mid:]
    low_num = global_offset + 1
    high_num = global_offset + len(spans)
    mid_num = global_offset + mid
    mid_num_plus1 = global_offset + mid + 1

    window_text = _format_window(spans, global_offset)
    user_text = W3_BISECT_PER_LABEL_TEMPLATE.format(
        task_description=task_desc,
        label=label,
        label_definition=LABEL_DEFINITIONS.get(label, ""),
        low_num=low_num,
        high_num=high_num,
        span_window=window_text,
        mid_num=mid_num,
        mid_num_plus1=mid_num_plus1,
    )
    prompt = apply_chat_template(tokenizer, user_text)
    tok_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tok_len + 256 > max_model_len:
        return []
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    meta["calls"] += 1
    try:
        raw = llm.generate([prompt], sp)[0].outputs[0].text
    except Exception as e:
        meta["error"] = str(e)
        return []
    parsed = parse_json_output(raw)
    if parsed is None:
        return []

    locations: List[str] = []
    if parsed.get("lower_half_present"):
        locations += _bisect_label(
            label, lower, global_offset,
            llm=llm, tokenizer=tokenizer, max_model_len=max_model_len,
            task_desc=task_desc, valid_span_ids=valid_span_ids, meta=meta,
        )
    if parsed.get("upper_half_present"):
        locations += _bisect_label(
            label, upper, global_offset + mid,
            llm=llm, tokenizer=tokenizer, max_model_len=max_model_len,
            task_desc=task_desc, valid_span_ids=valid_span_ids, meta=meta,
        )
    return locations


def _run_pass1_holistic(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    use_span_index: bool = True,
) -> Tuple[List[dict], List[dict], Dict]:
    """Run the W1-shaped holistic pass once. Returns (errors, scores, meta)."""
    ordered_spans = get_ordered_step_spans(trace_str)
    span_index = build_span_index(trace_str) if use_span_index else ""
    span_index_block = (span_index + "\n\n") if span_index else ""
    valid_span_ids = extract_span_ids(trace_str)
    trace_text = format_trace_for_prompt(ordered_spans)

    user_text = W1_PROMPT_TEMPLATE.format(
        taxonomy_block=TAXONOMY_BLOCK,
        span_index_block=span_index_block,
        trace=trace_text,
    )
    ASST_PREFIX = '{"errors":'
    prompt_text = apply_chat_template(tokenizer, user_text, assistant_prefix=ASST_PREFIX)

    tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    if tok_len + 8192 > max_model_len:
        return [], [], {"pass1_error": "context_overflow", "pass1_tok_len": tok_len}

    avail = max_model_len - tok_len
    sp = SamplingParams(temperature=0.0, max_tokens=min(max_new_tokens, avail))
    try:
        raw = ASST_PREFIX + llm.generate([prompt_text], sp)[0].outputs[0].text
    except Exception as e:
        return [], [], {"pass1_error": str(e)}

    parsed = parse_json_output(raw)
    if parsed is None:
        return [], [], {"pass1_error": "json_parse_failed", "pass1_raw": raw[:500]}

    errors = [
        e for e in parsed.get("errors", [])
        if (e.get("location") or "").strip() in valid_span_ids
    ]

    scores, scores_meta = run_scores_call(trace_text, llm, tokenizer, max_model_len)
    return errors, scores, {"pass1_tok_len": tok_len, **scores_meta}


def run_w3_graph(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
    include_pass1_cats_in_bisect: bool = False,
) -> Tuple[Optional[dict], dict]:
    """W3 with graph-driven label prefilter.

    Pipeline:
      1. Holistic Pass-1 → detected source categories (and their span_ids).
      2. propagate_confidence(detected, edges, threshold) → target labels.
      3. _bisect_label only for those target labels (optionally also re-bisect
         the source categories themselves, off by default — Pass-1 already
         returned a span for each).
      4. Merge Pass-1 errors with the bisection findings; dedup by
         (category, span_id).
    """
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc = extract_task_description(trace_str)
    valid_span_ids = extract_span_ids(trace_str)

    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

    meta: Dict = {
        "calls": 0,
        "error": None,
        "pass1_detected": [],
        "pass1_n_errors": 0,
        "n_filtered_edges": 0,
        "target_labels": [],
        "labels_bisected": [],
        "labels_found_via_bisect": [],
    }

    # --- Step 1: holistic Pass-1 ----------------------------------------
    p1_errors, p1_scores, p1_meta = _run_pass1_holistic(
        trace_str, llm, tokenizer, max_model_len, max_new_tokens,
    )
    meta.update(p1_meta)
    # Pass-1 made 1 generation call + 1 scores call (run_scores_call adds one)
    meta["calls"] += 2
    detected_cats = list({e["category"] for e in p1_errors})
    meta["pass1_detected"] = detected_cats
    meta["pass1_n_errors"] = len(p1_errors)

    # --- Step 2: propagate ----------------------------------------------
    filtered_edges = (
        propagate_confidence(detected_cats, edges, propagation_threshold)
        if detected_cats and edges else []
    )
    meta["n_filtered_edges"] = len(filtered_edges)
    target_labels = sorted({dst for _, dst, _ in filtered_edges})
    meta["target_labels"] = target_labels

    # Optionally also bisect the labels Pass-1 itself flagged
    bisect_labels = list(target_labels)
    if include_pass1_cats_in_bisect:
        for c in detected_cats:
            if c not in bisect_labels:
                bisect_labels.append(c)
    meta["labels_bisected"] = bisect_labels

    # --- Step 3: bisect only the selected labels ------------------------
    bisect_errors: List[dict] = []
    for label in bisect_labels:
        if label not in TAXONOMY_LEAF_CATEGORIES:
            continue
        span_ids = _bisect_label(
            label, ordered_spans, 0,
            llm=llm, tokenizer=tokenizer, max_model_len=max_model_len,
            task_desc=task_desc, valid_span_ids=valid_span_ids, meta=meta,
        )
        if span_ids:
            meta["labels_found_via_bisect"].append(label)
        for sid in span_ids:
            bisect_errors.append({
                "category": label,
                "location": sid,
                "evidence": "",
                "description": f"{label} detected at span {sid} via graph-targeted bisection.",
                "impact": "HIGH",
            })

    # --- Step 4: merge --------------------------------------------------
    merged: List[dict] = list(p1_errors)
    seen_pairs: set = {(e["category"], e.get("location", "")) for e in merged}
    for be in bisect_errors:
        key = (be["category"], be["location"])
        if key not in seen_pairs:
            merged.append(be)
            seen_pairs.add(key)

    return {"errors": merged, "scores": p1_scores}, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_graph_tag(args) -> str:
    if args.causal_only:
        return "causal_only"
    if args.corr_threshold < 1.0:
        return f"causal_corr{args.corr_threshold}"
    return f"suppes_t{args.edge_threshold}"


def main():
    parser = argparse.ArgumentParser(
        description="Who&When W2/W3 with TRAIL causal graph injection (vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str,   default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--data_dir",               type=str,   default=str(BENCH_DIR / "data"))
    parser.add_argument("--output_dir",             type=str,   default=str(THIS_DIR / "outputs"))
    parser.add_argument("--split",                  type=str,   default="GAIA_dedup")
    parser.add_argument("--variant",                type=str,   default="w2_graph",
                        choices=["w2_graph", "w3_graph"])
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--w2_max_spans",           type=int,   default=20)

    # Graph args (mirror run_eval_graph_inject_vllm.py)
    parser.add_argument("--causal_only",            action="store_true",
                        help="Use only the CAPRI-AIC validated causal edges.")
    parser.add_argument("--corr_threshold",         type=float, default=1.0,
                        help="Geomean threshold for adding Suppes edges. "
                             "Set to e.g. 0.20 for the extended graph. "
                             "Ignored if --causal_only.")
    parser.add_argument("--edge_threshold",         type=float, default=0.20,
                        help="Min geomean for observational edges (only when "
                             "neither --causal_only nor --corr_threshold<1.0).")
    parser.add_argument("--propagation_threshold",  type=float, default=0.10,
                        help="Min boosted score to keep a propagated target.")
    parser.add_argument("--causal_graph",           type=str,   default=None)
    parser.add_argument("--suppes_graph",           type=str,   default=None)
    parser.add_argument("--w3_include_pass1_cats",  action="store_true",
                        help="Also re-bisect the categories Pass-1 detected "
                             "(off by default — Pass-1 already returned a span).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    causal_graph_path = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH

    print(f"Loading graph (causal_only={args.causal_only}, corr_threshold={args.corr_threshold}) ...")
    edges = load_graph_edges(
        threshold=args.edge_threshold,
        causal_only=args.causal_only,
        corr_threshold=args.corr_threshold,
        causal_graph=causal_graph_path,
        suppes_graph=suppes_graph_path,
    )
    graph_tag = _build_graph_tag(args)
    print(f"  {len(edges)} edges loaded ({graph_tag})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 10:
        print(f"    ... and {len(edges)-10} more")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}_{graph_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"\nFound {len(file_paths)} traces in {data_dir}")
    print(f"Variant: {args.variant.upper()} | Output → {out_dir}\n")

    # ------------------------------------------------------------------
    # Tokenizer + model
    # ------------------------------------------------------------------
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
    for fp in tqdm(file_paths, desc=f"who_and_when_{args.variant}"):
        trace_id = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")

        if os.path.exists(out_file):
            continue

        with open(fp) as f:
            trace_str = f.read()

        if args.variant == "w2_graph":
            output, meta = run_w2_graph(
                trace_str, llm, tokenizer, args.max_model_len,
                edges=edges,
                propagation_threshold=args.propagation_threshold,
                causal_only=args.causal_only,
                max_spans=args.w2_max_spans,
            )
        else:  # w3_graph
            output, meta = run_w3_graph(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens,
                edges=edges,
                propagation_threshold=args.propagation_threshold,
                include_pass1_cats_in_bisect=args.w3_include_pass1_cats,
            )

        meta["trace_id"] = trace_id
        meta["variant"] = args.variant
        meta["graph_tag"] = graph_tag

        if output is None:
            output = {"errors": [], "scores": [], "_error": meta.get("error", "unknown")}
            skipped += 1

        n_errors = len(output.get("errors", []))
        print(f"  {trace_id}: {n_errors} error(s) predicted  [calls={meta.get('calls')}]")

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
