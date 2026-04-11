"""
eval/run_eval_graph_inject.py — Two-pass graph-inject evaluation via litellm API.

Implements the graph_inject design (Exp 4) for API-based models (Gemini, GPT-4o, etc.)
that do not require a local vLLM server or HuggingFace tokenizer.

Pipeline per trace:
  Pass 1  : Holistic error detection (same prompt as run_eval_with_graph.py).
  Propagate: hard-binary confidence (detected=1.0) × edge weights → filtered subgraph.
  Pass 2  : Single holistic re-check with the filtered causal subgraph injected.
            Only fires when Pass 1 detected at least one graph source category.
  Merge   : Deduplicate Pass 1 + Pass 2 errors → JSON compatible with calculate_scores.py.

Key differences vs run_eval_with_graph.py (single-pass):
  - Two API calls per triggered trace (one per untriggered trace).
  - Pass 1 uses the same graph guidance as run_eval_with_graph.py (global hint in prompt).
  - Pass 2 uses targeted subgraph injection: only edges where src ∈ detected AND dst ∉ detected.

Key differences vs UQ/run_uq_eval.py:
  - Uses litellm (no vLLM, no AutoTokenizer).
  - Supports any model accessible via litellm (Gemini, GPT-4o, Claude, etc.).
  - Hard-binary confidence only (no logprob extraction).

Usage (from benchmarking/):
    python eval/run_eval_graph_inject.py --split GAIA --model gemini/gemini-2.5-flash
    python eval/run_eval_graph_inject.py --split GAIA --model gemini/gemini-2.5-flash \\
        --causal_only --span_index --max_workers 5

Outputs:
    outputs/zero_shot/outputs_{model_tag}-{split}-graph_inject_{graph_tag}{span_tag}/
    and can be scored with the standard calculate_scores.py.
"""

import os
import re
import sys
import glob
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR            = Path(__file__).resolve().parent.parent
CAUSAL_DIR           = BENCH_DIR / "data" / "trail_causal_outputs_full_gaia_swe_AIC"
DEFAULT_CAUSAL_GRAPH = CAUSAL_DIR / "capri_graph.json"
DEFAULT_SUPPES_GRAPH = CAUSAL_DIR / "suppes_graph.json"

sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name


# ---------------------------------------------------------------------------
# Taxonomy
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
    "Language-only", "Tool-related", "Poor Information Retrieval",
    "Tool Output Misinterpretation", "Incorrect Problem Identification",
    "Tool Selection Errors", "Formatting Errors", "Instruction Non-compliance",
    "Tool Definition Issues", "Environment Setup Errors", "Rate Limiting",
    "Authentication Errors", "Service Errors", "Resource Not Found",
    "Resource Exhaustion", "Timeout Issues", "Context Handling Failures",
    "Resource Abuse", "Goal Deviation", "Task Orchestration",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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
- The category field must be a FINAL LEAF subcategory from the taxonomy (e.g. "Resource Abuse", "Goal Deviation") — NOT a parent category like "Planning and Coordination Errors" or "Reasoning Errors".

The trace to analyze:

{trace}\
"""


def format_graph_guidance(edges: List[Tuple[str, str, float]]) -> str:
    """Format graph edges as a guidance block for Pass 1 (same as run_eval_with_graph.py)."""
    if not edges:
        return ""
    lines = [
        "# Causal Error Patterns (data-driven, from prior trace analysis)",
        "The following causal relationships between error types have been statistically observed.",
        "When you identify an error of type A in the trace, actively look for errors of type B",
        "in subsequent spans, as B has been found to causally follow A.",
        "Higher strength values indicate stronger causal association.",
        "",
        "Format: [Source Error] → [Consequent Error]  (strength: X.XX)",
        "",
    ]
    for src, dst, w in edges:
        lines.append(f"  {src} → {dst}  (strength: {w:.2f})")
    lines.append("")
    return "\n".join(lines)


def build_pass1_prompt(trace_str: str, span_index: str = "",
                       graph_guidance: str = "") -> str:
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
        "\n".join(
            f"  - {e['category']} at span {e.get('location', '?')}"
            for e in pass1_errors
        )
        if pass1_errors else "  (none)"
    )
    graph_text = "\n".join(
        f'  "{src}" → "{dst}"  [weight: {w:.2f}]'
        for src, dst, w in filtered_edges
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
# Suppes graph loading + propagation
# ---------------------------------------------------------------------------

def load_graph_edges(
    threshold: float = 0.10,
    causal_only: bool = False,
    corr_threshold: float = 1.0,
    causal_graph: Path = DEFAULT_CAUSAL_GRAPH,
    suppes_graph: Path = DEFAULT_SUPPES_GRAPH,
) -> List[Tuple[str, str, float]]:
    """
    Load edges from plain JSON files — no torch or embeddings needed.

    Selection modes:
      causal_only=True         — 13 CAPRI-AIC edges from capri_graph.json (weight=pr_delta)
      corr_threshold < 1.0     — causal edges + Suppes edges with pr_delta >= corr_threshold
                                 (recommended: corr_threshold=0.20)
      default                  — all Suppes edges with pr_delta >= threshold
    """
    # Load Suppes edges for pr_delta lookup and threshold filtering
    if not suppes_graph.exists():
        raise FileNotFoundError(f"{suppes_graph} not found")
    with open(suppes_graph) as f:
        sg = json.load(f)
    suppes_by_key = {(e["a"], e["b"]): e["pr_delta"] for e in sg["edges"]}

    if causal_only:
        if not causal_graph.exists():
            raise FileNotFoundError(f"{causal_graph} not found")
        with open(causal_graph) as f:
            cg = json.load(f)
        edges = [(e["a"], e["b"], suppes_by_key.get((e["a"], e["b"]), 1.0))
                 for e in cg["edges"]]
    elif corr_threshold < 1.0:
        # Causal edges (all) + correlation Suppes edges above corr_threshold
        causal_keys: set = set()
        if causal_graph.exists():
            with open(causal_graph) as f:
                cg = json.load(f)
            causal_keys = {(e["a"], e["b"]) for e in cg["edges"]}
        edges = []
        for (a, b), w in suppes_by_key.items():
            if (a, b) in causal_keys or w >= corr_threshold:
                edges.append((a, b, w))
    else:
        edges = [(a, b, w) for (a, b), w in suppes_by_key.items() if w >= threshold]

    edges.sort(key=lambda x: -x[2])
    return edges


def propagate_confidence(
    detected_cats: List[str],
    edges: List[Tuple[str, str, float]],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    """
    Hard-binary confidence: detected=1.0, undetected=0.0.
    Returns filtered edges: src ∈ detected_cats, dst ∉ detected_cats,
    boosted_score(dst) > threshold.

    boosted_score(dst) = Σ_{src→dst} 1.0 × edge_weight
    """
    detected_set = set(detected_cats)
    boosted: Dict[str, float] = {}
    for src, dst, w in edges:
        if src in detected_set:
            boosted[dst] = boosted.get(dst, 0.0) + w

    # Return the actual edges that point to above-threshold undetected targets
    filtered = [
        (src, dst, w)
        for src, dst, w in edges
        if src in detected_set
        and dst not in detected_set
        and boosted.get(dst, 0.0) > threshold
    ]
    return filtered


# ---------------------------------------------------------------------------
# JSON parsing + span validation
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Optional[dict]:
    """Parse JSON from model output, stripping markdown fences and thinking blocks."""
    text = text.strip()
    # Strip thinking/reasoning blocks (Gemini 2.5, Claude extended thinking)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Strip all markdown fences (anywhere in the text)
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
    """Return {span_id: span_name} for every span in the trace JSON."""
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
    """Drop errors whose location is not a valid span_id in the trace."""
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
# LiteLLM calls
# ---------------------------------------------------------------------------

def _call_litellm(prompt: str, model: str, max_completion_tokens: int = 8000,
                  use_reasoning: bool = True) -> str:
    """Single litellm call with retry logic. Used for both Pass 1 and Pass 2.

    Pass 1 uses use_reasoning=True (thinking enabled, reasoning_effort='high').
    Pass 2 uses use_reasoning=False (no thinking, temperature=0.0) to avoid
    wasting token budget on reasoning when we only need concise JSON output.
    """
    messages = [{"role": "user", "content": prompt}]
    is_reasoning_model = (
        "o1" in model or "o3" in model or "o4" in model
        or "anthropic" in model or "gemini-2.5" in model
    )
    if is_reasoning_model and use_reasoning:
        # Reasoning models: no temperature/top_p; explicit reasoning_effort.
        # This matches run_eval_with_graph.py's call_litellm() exactly.
        params = {
            "messages": messages, "model": model,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": "high", "drop_params": True,
        }
    else:
        # Non-reasoning path: standard temperature=0 call.
        # Used for Pass 2 (simpler task; avoid burning token budget on thinking)
        # and for non-reasoning models.
        # For Gemini 2.5, reasoning_effort="none" → thinkingBudget=0 (thinking disabled).
        # Without this, Gemini 2.5 Flash defaults to ~8192 thinking tokens, consuming
        # nearly the entire max_completion_tokens budget and truncating the JSON output.
        re_effort = "none" if ("gemini-2.5" in model) else None
        params = {
            "messages": messages, "model": model,
            "temperature": 0.0, "top_p": 1,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": re_effort, "drop_params": True,
        }
    for attempt in range(3):
        try:
            response = completion(**params)
            time.sleep(6)  # free tier: 10 RPM = 1 req/6s minimum
            return response.choices[0].message["content"]
        except RateLimitError:
            print(f"  Rate limit (attempt {attempt+1}/3): sleeping 60s for RPM window reset...")
            time.sleep(60)
    raise RuntimeError(f"Exceeded 3 retries due to rate limiting for model {model}")


# ---------------------------------------------------------------------------
# Per-file two-pass pipeline
# ---------------------------------------------------------------------------

def process_file(
    file_path: str,
    output_dir: str,
    model: str,
    edges: List[Tuple[str, str, float]],
    graph_guidance: str,
    propagation_threshold: float,
    use_span_index: bool,
    validate_span_id: bool,
) -> str:
    trace_id   = os.path.splitext(os.path.basename(file_path))[0]
    out_file   = os.path.join(output_dir, f"{trace_id}.json")
    meta_file  = os.path.join(output_dir, f"_meta_{trace_id}.json")

    if os.path.exists(out_file):
        return file_path  # already done

    with open(file_path) as f:
        trace_str = f.read()

    span_index   = build_span_index(trace_str) if use_span_index else ""
    valid_span_ids = extract_span_ids(trace_str) if validate_span_id else {}

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

    # ------------------------------------------------------------------
    # Pass 1 — holistic error detection
    # ------------------------------------------------------------------
    p1_prompt = build_pass1_prompt(trace_str, span_index=span_index, graph_guidance=graph_guidance)
    try:
        p1_raw = _call_litellm(p1_prompt, model)
    except ContextWindowExceededError as e:
        print(f"  [Pass 1] context overflow for {trace_id}: {e}")
        with open(out_file, "w") as f:
            json.dump({"errors": [], "scores": [], "_error": "context_overflow"}, f)
        with open(meta_file, "w") as f:
            json.dump({**meta, "error": "context_overflow"}, f, indent=2)
        return file_path
    except Exception as e:
        print(f"  [Pass 1] error for {trace_id}: {e}")
        with open(out_file, "w") as f:
            json.dump({"errors": [], "scores": [], "_error": str(e)}, f)
        with open(meta_file, "w") as f:
            json.dump({**meta, "error": str(e)}, f, indent=2)
        return file_path

    p1_parsed = parse_json_output(p1_raw)
    if p1_parsed is None:
        print(f"  [Pass 1] JSON parse FAILED for {trace_id}")
        with open(out_file, "w") as f:
            f.write(p1_raw or "")
        with open(meta_file, "w") as f:
            json.dump({**meta, "p1_parse_failed": True}, f, indent=2)
        return file_path

    p1_errors = p1_parsed.get("errors", [])
    p1_scores = p1_parsed.get("scores", [])

    # Validate Pass 1 span_ids
    if validate_span_id and valid_span_ids:
        p1_errors, p1_dropped = validate_locations(p1_errors, valid_span_ids)
        meta["p1_dropped"] = p1_dropped

    detected_cats = list({e["category"] for e in p1_errors})
    meta["pass1_detected"] = detected_cats
    print(f"  [Pass 1] {trace_id}: {len(p1_errors)} errors, cats: {detected_cats}")

    # ------------------------------------------------------------------
    # Pass 2 — graph_inject
    # ------------------------------------------------------------------
    p2_errors: List[dict] = []
    if detected_cats and edges:
        filtered_edges = propagate_confidence(detected_cats, edges, propagation_threshold)
        meta["pass2_filtered_edges"] = len(filtered_edges)

        if filtered_edges:
            meta["pass2_triggered"] = True
            print(f"  [graph_inject] {len(filtered_edges)} filtered edges → Pass 2 call")

            p2_prompt = build_graph_inject_prompt(
                trace_str=trace_str,
                pass1_errors=p1_errors,
                filtered_edges=filtered_edges,
                span_index=span_index,
            )
            try:
                p2_raw = _call_litellm(p2_prompt, model, max_completion_tokens=8000,
                                       use_reasoning=False)
                p2_parsed = parse_json_output(p2_raw)
                if p2_parsed is None:
                    print(f"  [graph_inject] JSON parse FAILED for {trace_id}")
                    meta["pass2_parse_failed"] = True
                    # Save raw response for debugging
                    debug_file = os.path.join(output_dir, f"_debug_p2_{trace_id}.txt")
                    with open(debug_file, "w") as f:
                        f.write(p2_raw or "")
                else:
                    p2_errors = p2_parsed.get("errors", [])
                    if validate_span_id and valid_span_ids:
                        p2_errors, p2_dropped = validate_locations(p2_errors, valid_span_ids)
                        meta["p2_dropped"] = p2_dropped
                    # Deduplicate: drop any P2 category already in P1
                    p1_cats = {e["category"] for e in p1_errors}
                    p2_errors = [e for e in p2_errors if e["category"] not in p1_cats]
                    meta["pass2_new_errors"] = len(p2_errors)
                    print(f"  [graph_inject] found {len(p2_errors)} new errors")
            except Exception as e:
                print(f"  [graph_inject] error for {trace_id}: {e}")
                meta["pass2_error"] = str(e)
        else:
            print(f"  [graph_inject] no relevant edges for {trace_id} — skipping Pass 2")

    # ------------------------------------------------------------------
    # Merge and write
    # ------------------------------------------------------------------
    merged_errors = p1_errors + p2_errors
    output = {"errors": merged_errors, "scores": p1_scores}

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return file_path


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    directory: str,
    output_dir: str,
    model: str,
    edges: List[Tuple[str, str, float]],
    graph_guidance: str,
    propagation_threshold: float,
    max_workers: int,
    use_span_index: bool,
    validate_span_id: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    file_paths = sorted(glob.glob(f"{directory}/*.json"))
    print(f"\nProcessing {len(file_paths)} traces from {directory}")
    print(f"Output → {output_dir}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file, fp, output_dir, model, edges, graph_guidance,
                propagation_threshold, use_span_index, validate_span_id,
            )
            for fp in file_paths
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(file_paths)):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-pass graph-inject eval via litellm (Gemini / GPT-4o / etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",          type=str,   default="gemini/gemini-2.5-flash")
    parser.add_argument("--split",          type=str,   default="GAIA",
                        help="Dataset split: GAIA or SWE Bench")
    parser.add_argument("--data_dir",       type=str,   default="data")
    parser.add_argument("--output_dir",     type=str,   default="outputs/zero_shot")
    parser.add_argument("--max_workers",    type=int,   default=5)
    parser.add_argument("--causal_only",    action="store_true",
                        help="Use only the 13 CAPRI-AIC validated causal edges")
    parser.add_argument("--corr_threshold", type=float, default=1.0,
                        help="Include causal + Suppes edges with pr_delta >= this. "
                             "Set to 0.20 for extended graph. Ignored if --causal_only.")
    parser.add_argument("--edge_threshold", type=float, default=0.10,
                        help="Min pr_delta for default (non-causal-only) Suppes mode.")
    parser.add_argument("--propagation_threshold", type=float, default=0.10,
                        help="Min boosted score to trigger Pass 2 for a target category.")
    parser.add_argument("--span_index",     action="store_true", default=False,
                        help="Prepend compact span_id → name index to each prompt.")
    parser.add_argument("--validate_span_id", action="store_true", default=True,
                        help="Drop errors whose location is not a valid span_id in the trace.")
    parser.add_argument("--no_validate_span_id", dest="validate_span_id", action="store_false")
    parser.add_argument("--causal_graph",   type=str,   default=None,
                        help=f"Path to capri_graph.json (default: {DEFAULT_CAUSAL_GRAPH})")
    parser.add_argument("--suppes_graph",   type=str,   default=None,
                        help=f"Path to suppes_graph.json (default: {DEFAULT_SUPPES_GRAPH})")
    args = parser.parse_args()

    causal_graph_path = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    print(f"Loading graph edges (causal_only={args.causal_only}, corr_threshold={args.corr_threshold}) ...")
    edges = load_graph_edges(
        threshold=args.edge_threshold,
        causal_only=args.causal_only,
        corr_threshold=args.corr_threshold,
        causal_graph=causal_graph_path,
        suppes_graph=suppes_graph_path,
    )
    if args.causal_only:
        graph_tag = "causal_only"
        print(f"  {len(edges)} edges (causal_only)")
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
        print(f"  {len(edges)} edges (causal + corr pr_delta>={args.corr_threshold})")
    else:
        graph_tag = f"suppes_t{args.edge_threshold}"
        print(f"  {len(edges)} edges (pr_delta>={args.edge_threshold})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 10:
        print(f"    ... and {len(edges)-10} more")

    # Build graph guidance for Pass 1 (same as run_eval_with_graph.py)
    graph_guidance = format_graph_guidance(edges)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_tag  = args.model.replace("/", "-")
    span_tag   = "_span_index" if args.span_index else ""
    output_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-graph_inject_{graph_tag}{span_tag}",
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    data_dir = os.path.join(args.data_dir, args.split)
    run_eval(
        directory             = data_dir,
        output_dir            = output_dir,
        model                 = args.model,
        edges                 = edges,
        graph_guidance        = graph_guidance,
        propagation_threshold = args.propagation_threshold,
        max_workers           = args.max_workers,
        use_span_index        = args.span_index,
        validate_span_id      = args.validate_span_id,
    )
    print(f"\nDone. Score with:\n  python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
