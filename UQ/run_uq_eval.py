"""
UQ/run_uq_eval.py — Architecture A: open-source white-box UQ with causal graph propagation.

Two-pass pipeline:
  Pass 1  : QwenLong-L1-32B generates error predictions + token logprobs.
            Per-category confidence = mean exp(logprob) of the category-name tokens,
            i.e. the geometric-mean token probability for that string.
  Propagate: boosted_score(B) = Σ_{A→B} conf(A) × edge_weight(A→B)
  Pass 2  : Targeted re-verification for categories where boosted_score exceeds
            --propagation_threshold but were not detected in Pass 1.
  Merge   : Deduplicate Pass 1 + Pass 2 → final JSON compatible with calculate_scores.py.

Usage (from trail-benchmark/):
    CUDA_VISIBLE_DEVICES=4,7 python UQ/run_uq_eval.py --split GAIA
    CUDA_VISIBLE_DEVICES=4,7 python UQ/run_uq_eval.py --split GAIA --propagation_threshold 0.15 --edge_threshold 0.10

Output:
    benchmarking/outputs/uq/outputs_QwenLong-L1-32B-{split}-uq_causal/
    (one JSON file per trace, scored with benchmarking/eval/calculate_scores.py)
"""

import os
import re
import json
import glob
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
BENCH_DIR   = REPO_ROOT / "benchmarking"
GRAPH_INPUT = REPO_ROOT / "graph" / "data" / "graph_input.pt"
MODEL_ID    = "Tongyi-Zhiwen/QwenLong-L1-32B"

TAXONOMY_CATEGORIES = [
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
]

# ---------------------------------------------------------------------------
# Suppes causal graph
# ---------------------------------------------------------------------------

def load_suppes_edges(
    threshold: float,
    causal_only: bool = False,
    graph_input: Path = GRAPH_INPUT,
) -> List[Tuple[str, str, float]]:
    """Return list of (src_name, dst_name, weight) edges from graph_input.pt."""
    gi             = torch.load(graph_input, weights_only=False)
    node_names     = gi["node_names"]
    edge_index     = gi["edge_index"]
    edge_weight    = gi["edge_weight"]
    edge_is_causal = gi.get("edge_is_causal")
    correct_idx    = node_names.index("Correct") if "Correct" in node_names else len(node_names) - 1

    edges = []
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        w   = edge_weight[i].item()
        if src == correct_idx or dst == correct_idx:
            continue
        if causal_only:
            is_causal = (edge_is_causal[i].item() == 1.0) if edge_is_causal is not None else (w == 1.0)
            if not is_causal:
                continue
        else:
            if w < threshold:
                continue
        edges.append((node_names[src], node_names[dst], w))

    edges.sort(key=lambda x: -x[2])
    return edges

# ---------------------------------------------------------------------------
# Prompts  (identical taxonomy block as run_eval.py)
# ---------------------------------------------------------------------------

TAXONOMY_BLOCK = """Follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

# Taxonomy
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   ├── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description. For example, web search tool was defined as a calculator tool)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
├── Planning and Coordination Errors
│    ├── Context Management
│    │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
│    │   └── Resource Abuse (Called the tool excessively due to memory issues)
│    └── Task Management
│        ├── Goal Deviation (The system deviated from the task or the subtask)
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)"""

OUTPUT_TEMPLATE = """\
Template for output:

{
    "errors": [
        {
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]",
            "location": "[INSERT THE span_id VALUE HERE]", # Must be the exact span_id hex string from the trace JSON (e.g. '77fb7128d6f04862'), NOT the span name
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]"
        },
        ...
    ],
    "scores": [
        {
            "reliability_score": 3,
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]",
            "security_score": 5,
            "security_reasoning": "[INSERT DETAILED REASONING HERE]",
            "instruction_adherence_score": 4,
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]",
            "plan_opt_score": 3,
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]",
            "overall": 3.75
        }
    ]
}

If the trace has no errors output {"errors": [], "scores": [{...}]}.
- Output strictly valid JSON; no markdown, no extra text.
- Only use final leaf categories from the taxonomy.
- For location, use the exact span_id hex value from the trace JSON (e.g. '77fb7128d6f04862'), not the span name.
- For "Resource Abuse" mark the last instance; for all others mark the first instance."""


def build_pass1_prompt(trace_str: str) -> str:
    return (
        TAXONOMY_BLOCK
        + "\n\n"
        + "- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.\n"
        + "- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy.\n"
        + "- You must provide the output strictly in JSON format as shown below (do not wrap in markdown, output only JSON).\n\n"
        + OUTPUT_TEMPLATE
        + "\n\nThe data to analyze is as follows:\n\n"
        + trace_str
    )


def build_pass2_prompt(trace_str: str, pass1_errors: List[dict], to_verify: List[str]) -> str:
    """Targeted prompt asking the model to specifically check for categories flagged by
    causal propagation that were not detected in Pass 1."""
    detected_summary = (
        "\n".join(
            f"  - {e['category']} at span {e.get('location','?')} (impact: {e.get('impact','?')})"
            for e in pass1_errors
        )
        if pass1_errors
        else "  (none)"
    )
    verify_list = "\n".join(f"  - {c}" for c in to_verify)

    header = (
        TAXONOMY_BLOCK
        + "\n\n"
        + "You are performing a TARGETED second-pass verification of an LLM agent trace.\n\n"
        + "In a first-pass analysis the following errors were already detected:\n"
        + detected_summary
        + "\n\n"
        + "Based on causal relationships between error types, the following error categories\n"
        + "are likely present but may have been missed. For EACH category listed below,\n"
        + "determine whether it is present in the trace and, if so, provide the span_id location:\n"
        + verify_list
        + "\n\n"
        + "- Output ONLY the newly found errors (do not repeat errors from Pass 1).\n"
        + "- If none of the listed categories are present, output an empty errors list.\n"
        + "- Do NOT include scores in this pass.\n"
        + "- Output strictly valid JSON: {\"errors\": [...]}\n\n"
        + "The data to analyze is as follows:\n\n"
        + trace_str
    )
    return header

# ---------------------------------------------------------------------------
# Confidence extraction from vLLM logprobs
# ---------------------------------------------------------------------------

def extract_category_confidence(
    output_text: str,
    token_logprobs: List[Dict],   # list of {token_id: Logprob} dicts, one per generated token
    detected_categories: List[str],
) -> Dict[str, float]:
    """
    For each detected category name, locate the substring in output_text,
    map to token positions via cumulative decoded_token lengths, and return
    mean exp(logprob) of those tokens as the confidence in [0,1].

    Falls back to a low default (0.05) if a category string cannot be located.
    """
    # Build parallel list of (decoded_token_text, logprob_value)
    tokens: List[Tuple[str, float]] = []
    for lp_dict in token_logprobs:
        if not lp_dict:
            continue
        # lp_dict maps token_id -> Logprob; the sampled token is the one with rank=1 or
        # simply the only/first entry when logprobs=1
        lp_obj = next(iter(lp_dict.values()))
        decoded = lp_obj.decoded_token or ""
        tokens.append((decoded, lp_obj.logprob))

    # Reconstruct text from tokens and build char->token index mapping
    reconstructed = ""
    char_to_tok: List[int] = []
    for tok_idx, (tok_text, _) in enumerate(tokens):
        for _ in tok_text:
            char_to_tok.append(tok_idx)
        reconstructed += tok_text

    conf: Dict[str, float] = {}
    for cat in detected_categories:
        # Search for the category string as a JSON value: "category": "CAT"
        # We look for the literal category name surrounded by quotes in the output
        pattern = re.escape(cat)
        match = re.search(pattern, reconstructed)
        if not match or not char_to_tok:
            conf[cat] = 0.05  # low default when not found
            continue

        start_char = match.start()
        end_char   = match.end()

        # Map char range to token indices
        tok_start = char_to_tok[start_char] if start_char < len(char_to_tok) else None
        tok_end   = char_to_tok[end_char - 1] if (end_char - 1) < len(char_to_tok) else None

        if tok_start is None or tok_end is None:
            conf[cat] = 0.05
            continue

        lp_values = [tokens[i][1] for i in range(tok_start, tok_end + 1)]
        mean_lp   = sum(lp_values) / len(lp_values)
        conf[cat] = math.exp(mean_lp)   # in (0, 1]

    return conf

# ---------------------------------------------------------------------------
# Causal graph propagation
# ---------------------------------------------------------------------------

def propagate_confidence(
    conf: Dict[str, float],
    edges: List[Tuple[str, str, float]],
    all_categories: List[str],
) -> Dict[str, float]:
    """
    boosted_score(B) = Σ_{A→B} conf(A) × edge_weight(A→B)
    Only A with conf > 0 contribute. Undetected categories have conf=0 unless
    they appear as B in a propagation edge.
    """
    boosted: Dict[str, float] = {c: 0.0 for c in all_categories}
    for src, dst, w in edges:
        src_conf = conf.get(src, 0.0)
        if src_conf > 0:
            boosted[dst] = boosted.get(dst, 0.0) + src_conf * w
    return boosted

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Optional[dict]:
    """Attempt to parse JSON from the model output, handling common formatting issues."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON object via regex
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def extract_span_ids(trace_str: str) -> Dict[str, str]:
    """Return {span_id: span_name} for all spans in the trace JSON."""
    try:
        trace = json.loads(trace_str)
    except json.JSONDecodeError:
        return {}
    result = {}
    def walk(spans):
        for s in spans:
            sid = s.get("span_id")
            if sid:
                result[sid] = s.get("span_name", "")
            walk(s.get("child_spans", []))
    walk(trace.get("spans", []))
    return result


def validate_and_repair_locations(
    errors: List[dict],
    valid_span_ids: Dict[str, str],
    repair: bool,
) -> Tuple[List[dict], dict]:
    """
    Gate: check every error location against the set of valid span_ids.
    - If location is a valid span_id: keep as-is.
    - If repair=True: try fuzzy match against span_names → substitute span_id.
    - Otherwise: drop the error.

    Returns (cleaned_errors, diagnostics_dict).
    """
    hex_re = re.compile(r'^[0-9a-f]{8,}$')
    cleaned, repaired, dropped = [], 0, 0
    name_to_id = {v: k for k, v in valid_span_ids.items()} if repair else {}

    for e in errors:
        loc = (e.get("location") or "").strip()
        if loc in valid_span_ids:
            cleaned.append(e)
        elif repair and loc in name_to_id:
            e = dict(e)
            e["location"] = name_to_id[loc]
            cleaned.append(e)
            repaired += 1
        elif repair:
            # partial match: check if loc is a suffix of any span_name
            match = next((sid for sname, sid in name_to_id.items()
                          if loc and (loc in sname or sname.endswith(loc))), None)
            if match:
                e = dict(e)
                e["location"] = match
                cleaned.append(e)
                repaired += 1
            else:
                dropped += 1
        else:
            dropped += 1

    diagnostics = {
        "span_id_valid_count":   len(cleaned) - repaired,
        "span_id_invalid_count": repaired + dropped,
        "repaired_count":        repaired,
        "dropped_count":         dropped,
    }
    return cleaned, diagnostics


def normalize_to_leaf(category: str) -> str:
    """Strip taxonomy path prefix, keeping only the leaf name.

    The model sometimes outputs full paths like
    'Reasoning Errors/Hallucinations/Language-only' instead of just
    'Language-only'.  We take the last '/'-delimited segment so that
    confidence lookup and causal propagation (which use short names) work
    correctly, and so calculate_scores.py can normalize the category.
    """
    return category.split("/")[-1].strip()


def load_trace(file_path: str) -> str:
    """Load a trace JSON file and return its string representation for the prompt."""
    with open(file_path) as f:
        return f.read()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    llm: LLM,
    tokenizer: AutoTokenizer,
    trace_str: str,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
    max_new_tokens: int,
    validate_span_id: bool = True,
    repair_location: bool = False,
) -> dict:
    """Run two-pass UQ pipeline on a single trace. Returns final JSON dict."""

    sp_with_lp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        logprobs=1,         # return logprob of the sampled token at each position
        stop=None,
    )
    sp_no_lp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    def format_prompt(user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def prompt_token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # --- Pass 1 ---
    p1_text     = format_prompt(build_pass1_prompt(trace_str))
    p1_tok_len  = prompt_token_len(p1_text)
    print(f"  [Pass 1] prompt tokens: {p1_tok_len:,}  (model limit: {sp_with_lp.max_tokens} output, {131072} context)")

    p1_out  = llm.generate([p1_text], sp_with_lp)[0].outputs[0]
    print(f"  [Pass 1] response tokens: {len(p1_out.token_ids):,}  finish_reason: {p1_out.finish_reason}")

    p1_parsed = parse_json_output(p1_out.text)
    if p1_parsed is None:
        print(f"  [Pass 1] JSON parse FAILED. Raw response (first 500 chars):\n    {p1_out.text[:500]!r}")
        return {
            "errors": [],
            "scores": [{"reliability_score": 0, "reliability_reasoning": "parse error",
                        "security_score": 5, "security_reasoning": "",
                        "instruction_adherence_score": 0, "instruction_adherence_reasoning": "",
                        "plan_opt_score": 0, "plan_opt_reasoning": "", "overall": 0}],
            "_uq_meta": {
                "pass1_parse_error": True,
                "pass1_raw_response": p1_out.text[:2000],
                "pass1_prompt_tokens": p1_tok_len,
                "pass1_response_tokens": len(p1_out.token_ids),
                "pass1_finish_reason": p1_out.finish_reason,
            },
        }

    p1_errors = p1_parsed.get("errors", [])
    # Normalize full taxonomy paths to leaf names in-place
    for e in p1_errors:
        if e.get("category"):
            e["category"] = normalize_to_leaf(e["category"])

    # Span-ID gate
    location_diag = {"span_id_valid_count": len(p1_errors), "span_id_invalid_count": 0,
                     "repaired_count": 0, "dropped_count": 0}
    if validate_span_id:
        valid_span_ids = extract_span_ids(trace_str)
        p1_errors, location_diag = validate_and_repair_locations(
            p1_errors, valid_span_ids, repair=repair_location
        )
        if location_diag["dropped_count"] or location_diag["repaired_count"]:
            print(f"  [Gate] valid: {location_diag['span_id_valid_count']}  "
                  f"repaired: {location_diag['repaired_count']}  "
                  f"dropped: {location_diag['dropped_count']}")

    detected_cats = list({e["category"] for e in p1_errors if e.get("category")})

    # --- Confidence extraction ---
    conf = extract_category_confidence(
        output_text   = p1_out.text,
        token_logprobs = p1_out.logprobs or [],
        detected_categories = detected_cats,
    )

    # --- Causal propagation ---
    boosted = propagate_confidence(conf, edges, TAXONOMY_CATEGORIES)

    # --- Identify categories to verify in Pass 2 ---
    detected_set = set(detected_cats)
    to_verify = [
        cat for cat in TAXONOMY_CATEGORIES
        if cat not in detected_set and boosted.get(cat, 0.0) > propagation_threshold
    ]

    print(f"  [Pass 1] detected {len(p1_errors)} errors: {detected_cats}")
    print(f"  [Conf]   {', '.join(f'{k}: {v:.3f}' for k, v in conf.items()) or '(none)'}")

    # Store UQ metadata for analysis
    uq_meta = {
        "valid_json":            True,
        "location_diagnostics":  location_diag,
        "pass1_detected":        detected_cats,
        "pass1_prompt_tokens":   p1_tok_len,
        "pass1_response_tokens": len(p1_out.token_ids),
        "pass1_finish_reason":   p1_out.finish_reason,
        "pass1_raw_response":    p1_out.text,
        "confidence":            {k: round(v, 4) for k, v in conf.items()},
        "boosted_scores":        {k: round(v, 4) for k, v in boosted.items() if v > 0},
        "pass2_verify":          to_verify,
        "propagation_threshold": propagation_threshold,
    }

    p2_errors: List[dict] = []
    if to_verify:
        # --- Pass 2 ---
        p2_text    = format_prompt(build_pass2_prompt(trace_str, p1_errors, to_verify))
        p2_tok_len = prompt_token_len(p2_text)
        print(f"  [Pass 2] triggered for: {to_verify}")
        print(f"  [Pass 2] prompt tokens: {p2_tok_len:,}")
        p2_out  = llm.generate([p2_text], sp_no_lp)[0].outputs[0]
        print(f"  [Pass 2] response tokens: {len(p2_out.token_ids):,}  finish_reason: {p2_out.finish_reason}")
        p2_parsed = parse_json_output(p2_out.text)
        if p2_parsed is None:
            print(f"  [Pass 2] JSON parse FAILED. Raw response (first 500 chars):\n    {p2_out.text[:500]!r}")
        if p2_parsed:
            p2_errors = p2_parsed.get("errors", [])
        uq_meta["pass2_prompt_tokens"]   = p2_tok_len
        uq_meta["pass2_response_tokens"] = len(p2_out.token_ids)
        uq_meta["pass2_finish_reason"]   = p2_out.finish_reason
        uq_meta["pass2_raw_response"]    = p2_out.text

    # --- Merge ---
    merged_errors = list(p1_errors)
    p1_key_set = {(e.get("category", ""), e.get("location", "")) for e in p1_errors}
    for e in p2_errors:
        key = (e.get("category", ""), e.get("location", ""))
        if key not in p1_key_set:
            merged_errors.append(e)
            p1_key_set.add(key)

    final = {
        "errors": merged_errors,
        "scores": p1_parsed.get("scores", []),
        "_uq_meta": uq_meta,
    }
    return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Architecture A: white-box UQ + causal propagation with QwenLong-L1-32B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split",                  type=str,   default="GAIA",
                        help="Dataset split directory name under benchmarking/data/")
    parser.add_argument("--data_dir",               type=str,   default=None,
                        help="Override data directory (default: benchmarking/data/{split})")
    parser.add_argument("--output_dir",             type=str,   default=None,
                        help="Override output directory")
    parser.add_argument("--model",                  type=str,   default=MODEL_ID)
    parser.add_argument("--tensor_parallel_size",     type=int,   default=4)
    parser.add_argument("--max_model_len",            type=int,   default=131072,
                        help="Maximum sequence length (must not exceed model's max_position_embeddings=131072)")
    parser.add_argument("--gpu_memory_utilization",   type=float, default=0.75,
                        help="Fraction of GPU memory vLLM may use (lower if other jobs are running)")
    parser.add_argument("--enforce_eager",            action="store_true", default=True,
                        help="Disable CUDA graph capture (avoids OOM during warmup, ~20%% slower decode)")
    parser.add_argument("--no_enforce_eager",         dest="enforce_eager", action="store_false",
                        help="Re-enable CUDA graph capture (faster decode, needs more free memory)")
    parser.add_argument("--validate_span_id",         action="store_true", default=True,
                        help="Drop errors whose location is not a valid trace span_id")
    parser.add_argument("--repair_location",          action="store_true", default=False,
                        help="Before dropping, try fuzzy span-name → span_id repair")
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--propagation_threshold",  type=float, default=0.10,
                        help="boosted_score threshold above which Pass 2 is triggered")
    parser.add_argument("--edge_threshold",         type=float, default=0.10,
                        help="Minimum Suppes edge weight to include in propagation")
    parser.add_argument("--causal_only",            action="store_true",
                        help="Use only the ~11 fully validated causal edges")
    parser.add_argument("--graph_input",            type=str,   default=None)
    args = parser.parse_args()

    # --- Paths ---
    data_dir = Path(args.data_dir) if args.data_dir else BENCH_DIR / "data" / args.split
    model_tag = args.model.split("/")[-1]
    graph_tag = "causal_only" if args.causal_only else f"suppes_t{args.edge_threshold}"
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "UQ" / "outputs" / f"outputs_{model_tag}-{args.split}-uq_{graph_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Suppes edges ---
    graph_input_path = Path(args.graph_input) if args.graph_input else GRAPH_INPUT
    print(f"Loading Suppes graph from {graph_input_path} ...")
    edges = load_suppes_edges(
        threshold   = args.edge_threshold,
        causal_only = args.causal_only,
        graph_input = graph_input_path,
    )
    print(f"  {len(edges)} edges loaded")
    for src, dst, w in edges[:5]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 5:
        print(f"    ... and {len(edges) - 5} more")

    # --- Load tokenizer ---
    print(f"\nLoading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Load vLLM model ---
    print(f"Loading model {args.model} with tensor_parallel_size={args.tensor_parallel_size} ...")
    llm = LLM(
        model                 = args.model,
        tensor_parallel_size  = args.tensor_parallel_size,
        trust_remote_code     = True,
        max_model_len         = args.max_model_len,
        dtype                 = "bfloat16",
        gpu_memory_utilization= args.gpu_memory_utilization,
        enforce_eager         = args.enforce_eager,
    )
    print("Model loaded.\n")

    # --- Process traces ---
    file_paths = sorted(glob.glob(str(data_dir / "*.json")))
    print(f"Processing {len(file_paths)} traces from {data_dir}")
    print(f"Output → {out_dir}\n")

    skipped = 0
    for fp in tqdm(file_paths):
        out_file = out_dir / Path(fp).name
        if out_file.exists():
            continue  # resume

        trace_str = load_trace(fp)
        try:
            result = run_pipeline(
                llm                  = llm,
                tokenizer            = tokenizer,
                trace_str            = trace_str,
                edges                = edges,
                propagation_threshold= args.propagation_threshold,
                max_new_tokens       = args.max_new_tokens,
                validate_span_id     = args.validate_span_id,
                repair_location      = args.repair_location,
            )
        except Exception as e:
            print(f"\nError on {fp}: {e}")
            result = {
                "errors": [],
                "scores": [],
                "_uq_meta": {"error": str(e)},
            }
            skipped += 1

        # calculate_scores.py reads JSON files; write result without _uq_meta key
        # but save full result with meta for analysis
        public_result = {k: v for k, v in result.items() if k != "_uq_meta"}
        with open(out_file, "w") as f:
            json.dump(public_result, f)

        # Save meta separately for analysis
        meta_file = out_dir / ("_meta_" + Path(fp).name)
        with open(meta_file, "w") as f:
            json.dump(result.get("_uq_meta", {}), f, indent=2)

    print(f"\nDone. {len(file_paths) - skipped} processed, {skipped} errors.")
    print(f"Score with (from benchmarking/):")
    print(f"  python eval/calculate_scores.py --results_dir {out_dir.parent}")


if __name__ == "__main__":
    main()
