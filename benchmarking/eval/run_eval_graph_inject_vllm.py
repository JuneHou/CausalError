"""
eval/run_eval_graph_inject_vllm.py — Two-pass graph-inject evaluation via vLLM.

Mirrors run_eval_graph_inject.py exactly, but uses a local vLLM model instead of
the litellm API.  Designed for Mistral-Small-3.1-24B and similar local models.

Pipeline per trace:
  Pass 1 : Holistic error detection (same prompt as run_eval_with_graph_vllm.py).
  Propagate: hard-binary confidence × edge weights → filtered subgraph.
  Pass 2 : Targeted re-check with filtered causal subgraph injected.
            Only fires when Pass 1 detected at least one graph source category.
  Merge  : Deduplicate Pass 1 + Pass 2 errors → JSON compatible with calculate_scores.py.

Usage (from benchmarking/):
    CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_graph_inject_vllm.py \\
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
        --data_dir data --split GAIA_dedup \\
        --causal_only --corr_threshold 0.2 --span_index \\
        --tensor_parallel_size 2 --output_dir outputs/zero_shot2

Outputs:
    outputs/zero_shot2/outputs_{model_tag}-{split}-graph_inject_{graph_tag}{span_tag}/
    and can be scored with the standard calculate_scores.py.
"""

import math
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

BENCH_DIR            = Path(__file__).resolve().parent.parent
CAUSAL_DIR           = BENCH_DIR / "data" / "trail_causal_outputs_full_gaia_swe_AIC"
DEFAULT_CAUSAL_GRAPH = BENCH_DIR / "outputs" / "interventions_full_gaia_swe_merged" / "effect_edges.json"
DEFAULT_SUPPES_GRAPH = CAUSAL_DIR / "suppes_graph.json"

sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name


# ---------------------------------------------------------------------------
# Taxonomy block (shared across both passes)
# ---------------------------------------------------------------------------

TAXONOMY_BLOCK = """\
Follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

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
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)\
"""

TAXONOMY_CATEGORIES = [
    "Language-only", "Tool-related",
    "Poor Information Retrieval", "Tool Output Misinterpretation",
    "Incorrect Problem Identification", "Tool Selection Errors",
    "Formatting Errors", "Instruction Non-compliance",
    "Tool Definition Issues", "Environment Setup Errors",
    "Rate Limiting", "Authentication Errors", "Service Errors", "Resource Not Found",
    "Resource Exhaustion", "Timeout Issues",
    "Context Handling Failures", "Resource Abuse",
    "Goal Deviation", "Task Orchestration",
]

PASS1_PROMPT_TEMPLATE = """\
{taxonomy_block}
{graph_guidance_block}\
- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

Template for output:

{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]",
            "location": "[INSERT THE span_id VALUE HERE]",
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]"
        }},
        ...
    ],
    "scores": [
        {{
            "reliability_score": 3,
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]",
            "security_score": 5,
            "security_reasoning": "[INSERT DETAILED REASONING HERE]",
            "instruction_adherence_score": 4,
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]",
            "plan_opt_score": 3,
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]",
            "overall": 3.75
        }}
    ]
}}

Example output:

{{
    "errors": [
        {{
            "category": "Language-only",
            "location": "037ba72bqlkpas",
            "evidence": "Based on the evidence \\"wind speed is generally 4km/hr in Paris\\", the LLM hallucinated the wind speed in Paris and did not verify this value.",
            "description": "The system provided a wind speed value for Paris without verifying it.",
            "impact": "HIGH"
        }}
    ],
    "scores": [
        {{
            "reliability_score": 1,
            "reliability_reasoning": "The system failed to provide accurate information.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "The system did not follow instructions to verify all information.",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "The system did not incorporate the use of search tool effectively.",
            "overall": 2.5
        }}
    ]
}}

If the trace has no errors, output {{"errors": [], "scores": [{{"reliability_score": 5, ...}}]}}.

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
- The location field must be the exact span_id hex string from the trace JSON, NOT the span name.

{span_index_block}The data to analyze:

{trace}\
"""

GRAPH_INJECT_TEMPLATE = """\
{taxonomy_block}

You are performing a TARGETED SECOND-PASS analysis of an LLM agent trace.

PASS 1 RESULTS — The following errors were already detected:
{pass1_summary}

CAUSAL GRAPH CONTEXT — Statistical analysis of agent traces shows these error
type relationships (source → target [edge weight]):
{graph_text}

Based on the causal graph above, look specifically for the TARGET error types
listed — they are statistically likely given what was detected in Pass 1.

{span_index_block}\
INSTRUCTIONS:
- Output ONLY errors not already found in Pass 1.
- If no additional errors are present, output {{"errors": []}}.
- Do NOT include scores.
- Output strictly valid JSON with no markdown formatting, no explanation, no preamble: {{"errors": [...]}}
- Use the same schema: category, location (exact span_id hex), evidence, description, impact.
- The location field must be the exact span_id hex string from the trace JSON, NOT the span name.
- The category field must be a FINAL LEAF subcategory from the taxonomy (e.g. "Resource Abuse", "Goal Deviation") — NOT a parent category.

The trace to analyze:

{trace}\
"""


# ---------------------------------------------------------------------------
# Graph loading + propagation (same logic as run_eval_graph_inject.py)
# ---------------------------------------------------------------------------

def _parse_causal_graph(path: Path, suppes_by_key: dict) -> List[Tuple[str, str, float]]:
    with open(path) as f:
        data = json.load(f)
    raw = data["edges"]
    if isinstance(raw, dict):
        return [
            (v["a"], v["b"], abs(v["delta"]))
            for v in raw.values()
            if v.get("validated", False)
        ]
    else:
        return [(e["a"], e["b"], suppes_by_key.get((e["a"], e["b"]), 1.0)) for e in raw]


def load_graph_edges(
    threshold: float = 0.10,
    causal_only: bool = False,
    corr_threshold: float = 1.0,
    causal_graph: Path = DEFAULT_CAUSAL_GRAPH,
    suppes_graph: Path = DEFAULT_SUPPES_GRAPH,
    random_edges: bool = False,
    random_seed: int = 42,
    random_n: int = 12,
) -> List[Tuple[str, str, float]]:
    if not suppes_graph.exists():
        raise FileNotFoundError(f"{suppes_graph} not found")
    with open(suppes_graph) as f:
        sg = json.load(f)
    suppes_by_key = {(e["a"], e["b"]): e["pr_delta"] for e in sg["edges"]}
    suppes_edges = sg["edges"]

    if random_edges:
        import random as _rnd
        suppes_keys = {(e["a"], e["b"]) for e in suppes_edges}
        nodes = sorted(TAXONOMY_CATEGORIES)
        candidate = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in suppes_keys]
        rnd = _rnd.Random(random_seed)
        sampled = rnd.sample(candidate, min(random_n, len(candidate)))
        return [(a, b, 1.0) for a, b in sampled]

    if causal_only:
        if not causal_graph.exists():
            raise FileNotFoundError(f"{causal_graph} not found")
        edges = _parse_causal_graph(causal_graph, suppes_by_key)
    elif corr_threshold < 1.0:
        causal_keys: set = set()
        if causal_graph.exists():
            causal_keys = {(e[0], e[1]) for e in _parse_causal_graph(causal_graph, suppes_by_key)}
        edges = []
        for e in suppes_edges:
            a, b = e["a"], e["b"]
            score = math.sqrt(e["precedence"] * e["pr_delta"])
            if (a, b) in causal_keys or score >= corr_threshold:
                edges.append((a, b, score))
    else:
        edges = []
        for e in suppes_edges:
            score = math.sqrt(e["precedence"] * e["pr_delta"])
            if score >= threshold:
                edges.append((e["a"], e["b"], score))

    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(edges: List[Tuple[str, str, float]],
                           causal_only: bool = False,
                           random_edges: bool = False) -> str:
    if not edges:
        return ""
    if random_edges:
        lines = [
            "# Random Error Pattern Baseline (uncalibrated)",
            "The following edges are sampled uniformly at random from directed category pairs",
            "outside the Suppes-screened graph. They carry no probabilistic interpretation and",
            "serve as a control for graph-structure ablations.",
            "When you identify an error of type A in the trace, consider also checking for error type B.",
            "",
            "Format: [Source Error] → [Consequent Error]",
            "",
        ]
        for src, dst, _ in edges:
            lines.append(f"  {src} → {dst}")
    elif causal_only:
        lines = [
            "# Causal Error Patterns (intervention-validated)",
            "The following edges were validated via counterfactual patching experiments.",
            "When you identify an error of type A in the trace, actively look for errors of type B,",
            "as removing A causally reduces B's occurrence rate.",
            "Higher values indicate stronger causal effect.",
            "",
            "Format: [Source Error] → [Consequent Error]  (causal effect: X.XX)",
            "",
        ]
        for src, dst, w in edges:
            lines.append(f"  {src} → {dst}  (causal effect: {w:.2f})")
    else:
        lines = [
            "# Correlated Error Patterns (observational, precedence-filtered)",
            "The following error pairs consistently co-occur with A preceding B across agent traces.",
            "Score = geometric mean of precedence P(A precedes B | both occur) and probability-raising delta P(B|A)−P(B|¬A).",
            "When you identify an error of type A in the trace, consider also checking for error type B.",
            "Higher values indicate stronger observational association.",
            "",
            "Format: [Source Error] → [Consequent Error]  (observational score: X.XX)",
            "",
        ]
        for src, dst, w in edges:
            lines.append(f"  {src} → {dst}  (observational score: {w:.2f})")
    lines.append("")
    return "\n".join(lines)


def propagate_confidence(
    detected_cats: List[str],
    edges: List[Tuple[str, str, float]],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    detected_set = set(detected_cats)
    boosted: Dict[str, float] = {}
    for src, dst, w in edges:
        if src in detected_set:
            boosted[dst] = boosted.get(dst, 0.0) + w
    return [
        (src, dst, w) for src, dst, w in edges
        if src in detected_set
        and dst not in detected_set
        and boosted.get(dst, 0.0) > threshold
    ]


# ---------------------------------------------------------------------------
# JSON parsing + span validation
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def extract_span_ids(trace_str: str) -> Dict[str, str]:
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


def validate_locations(
    errors: List[dict],
    valid_span_ids: Dict[str, str],
) -> Tuple[List[dict], int]:
    cleaned, dropped = [], 0
    for e in errors:
        loc = (e.get("location") or "").strip()
        if loc in valid_span_ids:
            cleaned.append(e)
        else:
            dropped += 1
    return cleaned, dropped


# ---------------------------------------------------------------------------
# Span index
# ---------------------------------------------------------------------------

def build_span_index(trace_str: str) -> str:
    try:
        trace_data = json.loads(trace_str)
    except Exception:
        return ""
    parsed = parse_trace_to_step_level(trace_data)
    step_spans = parsed.get("step_spans", [])
    if not step_spans:
        return ""
    lines = ["Span index for this trace (use these exact span_id hex values for the location field):"]
    seen = set()
    for entry in step_spans:
        span = entry["span"]
        sid = span.get("span_id")
        sname = _span_name(span)
        if sid and sid not in seen:
            seen.add(sid)
            lines.append(f'  span_id "{sid}"  ({sname})')
        for child in span.get("child_spans") or []:
            csid = child.get("span_id")
            csname = _span_name(child)
            if csid and csid not in seen:
                seen.add(csid)
                lines.append(f'    span_id "{csid}"  ({csname})')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_pass1_prompt(trace_str: str, span_index: str = "", graph_guidance: str = "") -> str:
    span_index_block = (span_index + "\n\n") if span_index else ""
    graph_guidance_block = (graph_guidance + "\n") if graph_guidance else ""
    return PASS1_PROMPT_TEMPLATE.format(
        taxonomy_block=TAXONOMY_BLOCK,
        graph_guidance_block=graph_guidance_block,
        span_index_block=span_index_block,
        trace=trace_str,
    )


def build_graph_inject_prompt(
    trace_str: str,
    pass1_errors: List[dict],
    filtered_edges: List[Tuple[str, str, float]],
    span_index: str = "",
) -> str:
    pass1_summary = (
        "\n".join(f"  - {e['category']} at span {e.get('location', '?')}" for e in pass1_errors)
        if pass1_errors else "  (none)"
    )
    graph_text = "\n".join(
        f'  "{src}" → "{dst}"  [weight: {w:.2f}]' for src, dst, w in filtered_edges
    ) if filtered_edges else "  (no relevant edges)"
    span_index_block = (span_index + "\n\n") if span_index else ""
    return GRAPH_INJECT_TEMPLATE.format(
        taxonomy_block=TAXONOMY_BLOCK,
        pass1_summary=pass1_summary,
        graph_text=graph_text,
        span_index_block=span_index_block,
        trace=trace_str,
    )


# ---------------------------------------------------------------------------
# vLLM inference helper
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, user_text: str) -> str:
    if tokenizer.chat_template is None:
        bos = tokenizer.bos_token or "<s>"
        return f"{bos}[INST] {user_text} [/INST]"
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass graph-inject eval via vLLM (Mistral / local models)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str,   default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--data_dir",               type=str,   default="data")
    parser.add_argument("--output_dir",             type=str,   default="outputs/zero_shot")
    parser.add_argument("--split",                  type=str,   default="GAIA_dedup")
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--temperature",            type=float, default=0.0,
                        help="Decoding temperature. >0 enables stochastic sampling; "
                             "re-invoke the script multiple times to collect i.i.d. samples.")
    parser.add_argument("--seed",                   type=int,   default=0,
                        help="Per-request sampling seed; pass distinct values across "
                             "invocations to obtain i.i.d. samples at temperature>0.")
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--causal_only",            action="store_true",
                        help="Use only the 12 intervention-validated causal edges")
    parser.add_argument("--corr_threshold",         type=float, default=1.0,
                        help="Include causal + Suppes edges with geomean sqrt(precedence*PR_delta) >= this. "
                             "Set to 0.20/0.25/0.35 for the threshold sweep. Ignored if --causal_only.")
    parser.add_argument("--edge_threshold",         type=float, default=0.20,
                        help="Min geomean score sqrt(precedence*PR_delta) for observational edges (non-causal-only mode).")
    parser.add_argument("--random_edges",           action="store_true",
                        help="Random-12 baseline: sample edges from full taxonomy minus Suppes graph.")
    parser.add_argument("--random_seed",            type=int,   default=42,
                        help="Seed for --random_edges sampling.")
    parser.add_argument("--random_n",               type=int,   default=12,
                        help="Number of random edges to sample (default 12, matches causal-only).")
    parser.add_argument("--propagation_threshold",  type=float, default=0.10,
                        help="Min boosted score to trigger Pass 2 for a target category.")
    parser.add_argument("--span_index",             action="store_true", default=False,
                        help="Prepend compact span_id index to each prompt.")
    parser.add_argument("--validate_span_id",       action="store_true", default=True,
                        help="Drop errors whose location is not a valid span_id.")
    parser.add_argument("--no_validate_span_id",    dest="validate_span_id", action="store_false")
    parser.add_argument("--causal_graph",           type=str,   default=None)
    parser.add_argument("--suppes_graph",           type=str,   default=None)
    args = parser.parse_args()

    causal_graph_path = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    print(f"Loading graph edges (causal_only={args.causal_only}, corr_threshold={args.corr_threshold}, "
          f"random_edges={args.random_edges}) ...")
    edges = load_graph_edges(
        threshold=args.edge_threshold,
        causal_only=args.causal_only,
        corr_threshold=args.corr_threshold,
        causal_graph=causal_graph_path,
        suppes_graph=suppes_graph_path,
        random_edges=args.random_edges,
        random_seed=args.random_seed,
        random_n=args.random_n,
    )
    if args.random_edges:
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
        print(f"  {len(edges)} random edges (seed={args.random_seed}, non-Suppes)")
    elif args.causal_only:
        graph_tag = "causal_only"
        print(f"  {len(edges)} edges (causal_only)")
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
        print(f"  {len(edges)} edges (causal + corr geomean>={args.corr_threshold})")
    else:
        graph_tag = f"suppes_t{args.edge_threshold}"
        print(f"  {len(edges)} edges (geomean>={args.edge_threshold})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 10:
        print(f"    ... and {len(edges)-10} more")

    graph_guidance = format_graph_guidance(
        edges, causal_only=args.causal_only, random_edges=args.random_edges,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_tag  = args.model.replace("/", "-")
    span_tag   = "_span_index" if args.span_index else ""
    temp_tag   = f"_t{args.temperature}" if args.temperature > 0 else ""
    seed_tag   = f"_s{args.seed}" if args.seed != 0 else ""
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-graph_inject_{graph_tag}{span_tag}{temp_tag}{seed_tag}",
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
    print(f"Output → {out_dir}\n")

    # ------------------------------------------------------------------
    # Load tokenizer + model
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
    sp_p1 = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, seed=args.seed)
    sp_p2 = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, seed=args.seed)

    # ------------------------------------------------------------------
    # Two-pass loop (sequential — Pass 2 depends on Pass 1 per trace)
    # ------------------------------------------------------------------
    p2_triggered_total = 0
    skipped = 0

    for fp in tqdm(file_paths):
        trace_id = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")

        if os.path.exists(out_file):
            continue

        with open(fp) as f:
            trace_str = f.read()

        span_index     = build_span_index(trace_str) if args.span_index else ""
        valid_span_ids = extract_span_ids(trace_str) if args.validate_span_id else {}

        meta = {
            "trace_id": trace_id,
            "pass1_detected": [],
            "pass2_triggered": False,
            "pass2_filtered_edges": 0,
            "pass2_new_errors": 0,
            "pass2_parse_failed": False,
            "p1_dropped": 0,
            "p2_dropped": 0,
        }

        # --------------------------------------------------------------
        # Pass 1 — holistic error detection
        # --------------------------------------------------------------
        p1_user_text  = build_pass1_prompt(trace_str, span_index=span_index, graph_guidance=graph_guidance)
        p1_prompt_text = apply_chat_template(tokenizer, p1_user_text)

        tok_len = len(tokenizer.encode(p1_prompt_text, add_special_tokens=False))
        _min_output = 2048
        if tok_len + _min_output > args.max_model_len:
            print(f"\n  [Pass 1] skipping {trace_id}: prompt too long ({tok_len:,} tokens)")
            with open(out_file, "w") as f:
                json.dump({"errors": [], "scores": [], "_error": "context_overflow"}, f)
            with open(meta_file, "w") as f:
                json.dump({**meta, "error": "context_overflow"}, f, indent=2)
            skipped += 1
            continue

        _avail = args.max_model_len - tok_len
        _sp1 = SamplingParams(temperature=args.temperature, max_tokens=min(args.max_new_tokens, _avail), seed=args.seed)
        try:
            p1_raw = llm.generate([p1_prompt_text], _sp1)[0].outputs[0].text
        except Exception as e:
            print(f"\n  [Pass 1] error for {trace_id}: {e}")
            with open(out_file, "w") as f:
                json.dump({"errors": [], "scores": [], "_error": str(e)}, f)
            with open(meta_file, "w") as f:
                json.dump({**meta, "error": str(e)}, f, indent=2)
            skipped += 1
            continue

        p1_parsed = parse_json_output(p1_raw)
        if p1_parsed is None:
            print(f"\n  [Pass 1] JSON parse FAILED for {trace_id}")
            with open(out_file, "w") as f:
                f.write(p1_raw or "")
            with open(meta_file, "w") as f:
                json.dump({**meta, "p1_parse_failed": True}, f, indent=2)
            continue

        p1_errors = p1_parsed.get("errors", [])
        p1_scores = p1_parsed.get("scores", [])

        if args.validate_span_id and valid_span_ids:
            p1_errors, p1_dropped = validate_locations(p1_errors, valid_span_ids)
            meta["p1_dropped"] = p1_dropped

        detected_cats = list({e["category"] for e in p1_errors})
        meta["pass1_detected"] = detected_cats
        print(f"  [Pass 1] {trace_id}: {len(p1_errors)} errors, cats: {detected_cats}")

        # --------------------------------------------------------------
        # Pass 2 — graph inject
        # --------------------------------------------------------------
        p2_errors: List[dict] = []
        if detected_cats and edges:
            filtered_edges = propagate_confidence(detected_cats, edges, args.propagation_threshold)
            meta["pass2_filtered_edges"] = len(filtered_edges)

            if filtered_edges:
                meta["pass2_triggered"] = True
                p2_triggered_total += 1
                print(f"  [graph_inject] {len(filtered_edges)} filtered edges → Pass 2")

                p2_user_text   = build_graph_inject_prompt(
                    trace_str, p1_errors, filtered_edges, span_index=span_index
                )
                p2_prompt_text = apply_chat_template(tokenizer, p2_user_text)

                tok_len_p2 = len(tokenizer.encode(p2_prompt_text, add_special_tokens=False))
                _avail_p2  = args.max_model_len - tok_len_p2
                if _avail_p2 < _min_output:
                    print(f"  [graph_inject] skipping Pass 2 for {trace_id}: prompt too long")
                    meta["pass2_error"] = "context_overflow"
                else:
                    _sp2 = SamplingParams(temperature=args.temperature, max_tokens=min(args.max_new_tokens, _avail_p2), seed=args.seed)
                    try:
                        p2_raw    = llm.generate([p2_prompt_text], _sp2)[0].outputs[0].text
                        p2_parsed = parse_json_output(p2_raw)
                        if p2_parsed is None:
                            print(f"  [graph_inject] JSON parse FAILED for {trace_id}")
                            meta["pass2_parse_failed"] = True
                            debug_file = os.path.join(out_dir, f"_debug_p2_{trace_id}.txt")
                            with open(debug_file, "w") as f:
                                f.write(p2_raw or "")
                        else:
                            p2_errors = p2_parsed.get("errors", [])
                            if args.validate_span_id and valid_span_ids:
                                p2_errors, p2_dropped = validate_locations(p2_errors, valid_span_ids)
                                meta["p2_dropped"] = p2_dropped
                            p1_cats = {e["category"] for e in p1_errors}
                            p2_errors = [e for e in p2_errors if e["category"] not in p1_cats]
                            meta["pass2_new_errors"] = len(p2_errors)
                            print(f"  [graph_inject] found {len(p2_errors)} new errors")
                    except Exception as e:
                        print(f"  [graph_inject] error for {trace_id}: {e}")
                        meta["pass2_error"] = str(e)
            else:
                print(f"  [graph_inject] no relevant edges for {trace_id} — skipping Pass 2")

        # --------------------------------------------------------------
        # Merge and write
        # --------------------------------------------------------------
        merged_errors = p1_errors + p2_errors
        output = {"errors": merged_errors, "scores": p1_scores}

        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    n_done = len(file_paths) - skipped
    print(f"\nDone. {n_done} processed, {skipped} skipped (context overflow / error).")
    print(f"Pass 2 triggered for {p2_triggered_total} traces.")
    print(f"Score with:\n  python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
