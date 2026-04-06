"""
longcontext/run_streaming_eval.py — StreamingLLM (Attention Sinks) baseline for long-context traces.

Implements the Attention Sinks method (Xiao et al. 2023, arXiv:2309.17453) as a comparison
against ReAttention.  Instead of a full KV cache, the model sees only:
  - first `num_sink_tokens` tokens  (attention sinks — stabilize softmax)
  - last `window_length - num_sink_tokens` tokens  (sliding window)

Everything in between is discarded.  This is accomplished by truncating the tokenized prompt
before generation — functionally equivalent to what a SinkCache would contain after processing
the full sequence token-by-token.

Uses standard AutoModelForCausalLM (no custom attention code).

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python longcontext/run_streaming_eval.py \\
      --split GAIA \\
      --causal_only \\
      --window_length 32768 \\
      --num_sink_tokens 4

  # Only traces that overflow QwenLong's 128K:
  CUDA_VISIBLE_DEVICES=0,1 python longcontext/run_streaming_eval.py \\
      --split GAIA --causal_only --min_tokens 131073 --window_length 32768
"""

import os
import sys
import re
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarking"
LC_DIR    = Path(__file__).resolve().parent

sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(LC_DIR))

from span_level_parser import parse_trace_to_step_level, _span_name
GRAPH_INPUT = REPO_ROOT / "graph" / "data" / "graph_input.pt"
MODEL_ID    = "Tongyi-Zhiwen/QwenLong-L1-32B"

# ---------------------------------------------------------------------------
# Taxonomy & prompts  (identical to run_longcontext_eval.py)
# ---------------------------------------------------------------------------
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

GRAPH_PROBE_TEMPLATE_IMPLICIT = """\
{taxonomy_block}

You are performing a targeted verification for a specific error type in an LLM agent trace.

Context: A causally related error was already detected in this trace. Based on statistical
causal relationships between error types, a "{category}" error is likely also present.

Your task: Determine whether a "{category}" error exists in the trace below.

{span_index_block}\
- Output ONLY valid JSON. No markdown, no extra text.
- If the error IS present:
  {{"present": true, "location": "<span_id_hex>", "evidence": "...", "description": "...", "impact": "HIGH|MEDIUM|LOW"}}
- If the error is NOT present:
  {{"present": false}}

The trace:

{trace}"""

GRAPH_PROBE_TEMPLATE_EXPLICIT = """\
{taxonomy_block}

You are performing a targeted verification for a specific error type in an LLM agent trace.

Causal graph analysis: "{source_category}" was detected in this trace (at span {source_span}).
Statistical analysis of agent execution traces shows "{source_category}" causally precedes
"{category}" in a bootstrap-validated causal graph (Suppes criterion, edge weight={weight:.2f}).
Based on this causal relationship, a "{category}" error is likely also present in this trace.

Your task: Determine whether a "{category}" error exists in the trace below.

{span_index_block}\
- Output ONLY valid JSON. No markdown, no extra text.
- If the error IS present:
  {{"present": true, "location": "<span_id_hex>", "evidence": "...", "description": "...", "impact": "HIGH|MEDIUM|LOW"}}
- If the error is NOT present:
  {{"present": false}}

The trace:

{trace}"""

# ---------------------------------------------------------------------------
# Suppes causal graph
# ---------------------------------------------------------------------------

def load_suppes_edges(
    threshold: float,
    causal_only: bool = False,
    corr_threshold: float = 1.0,
    graph_input: Path = GRAPH_INPUT,
) -> List[Tuple[str, str, float]]:
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
        is_causal = (edge_is_causal[i].item() == 1.0) if edge_is_causal is not None else (w == 1.0)
        if causal_only:
            if not is_causal:
                continue
        elif corr_threshold < 1.0:
            if not is_causal and w < corr_threshold:
                continue
        else:
            if w < threshold:
                continue
        edges.append((node_names[src], node_names[dst], w))

    edges.sort(key=lambda x: -x[2])
    return edges

# ---------------------------------------------------------------------------
# Prompt builders
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


def build_pass1_prompt(trace_str: str, span_index: str = "") -> str:
    span_block = (span_index + "\n\n") if span_index else ""
    return (
        TAXONOMY_BLOCK
        + "\n\n"
        + "- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.\n"
        + "- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy.\n"
        + "- You must provide the output strictly in JSON format as shown below (do not wrap in markdown, output only JSON).\n\n"
        + OUTPUT_TEMPLATE
        + "\n\nThe data to analyze is as follows:\n\n"
        + span_block
        + trace_str
    )


def build_graph_probe_prompt(
    category: str,
    trace_str: str,
    span_index: str = "",
    explicit_causal_encoding: bool = False,
    source_category: str = "",
    source_span: str = "",
    weight: float = 0.0,
) -> str:
    span_index_block = (span_index + "\n\n") if span_index else ""
    if explicit_causal_encoding and source_category:
        return GRAPH_PROBE_TEMPLATE_EXPLICIT.format(
            taxonomy_block=TAXONOMY_BLOCK,
            category=category,
            source_category=source_category,
            source_span=source_span or "unknown",
            weight=weight,
            span_index_block=span_index_block,
            trace=trace_str,
        )
    return GRAPH_PROBE_TEMPLATE_IMPLICIT.format(
        taxonomy_block=TAXONOMY_BLOCK,
        category=category,
        span_index_block=span_index_block,
        trace=trace_str,
    )

# ---------------------------------------------------------------------------
# JSON / span helpers
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
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


def validate_and_repair_locations(
    errors: List[dict],
    valid_span_ids: Dict[str, str],
    repair: bool,
) -> Tuple[List[dict], dict]:
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
    return category.split("/")[-1].strip()


def load_trace(file_path: str) -> str:
    with open(file_path) as f:
        return f.read()

# ---------------------------------------------------------------------------
# Causal propagation
# ---------------------------------------------------------------------------

def propagate_confidence(
    conf: Dict[str, float],
    edges: List[Tuple[str, str, float]],
    all_categories: List[str],
) -> Dict[str, float]:
    boosted: Dict[str, float] = {c: 0.0 for c in all_categories}
    for src, dst, w in edges:
        src_conf = conf.get(src, 0.0)
        if src_conf > 0:
            boosted[dst] = boosted.get(dst, 0.0) + src_conf * w
    return boosted

# ---------------------------------------------------------------------------
# ProbeTarget
# ---------------------------------------------------------------------------
from typing import NamedTuple

class ProbeTarget(NamedTuple):
    target:          str
    source_category: str   = ""
    source_span:     str   = ""
    weight:          float = 0.0

# ---------------------------------------------------------------------------
# StreamingLLM inference
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer: AutoTokenizer, user_text: str) -> torch.Tensor:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def truncate_to_streaming_window(
    input_ids: torch.Tensor,
    num_sink_tokens: int,
    window_length: int,
) -> torch.Tensor:
    """Keep first num_sink_tokens (attention sinks) + last (window_length - num_sink_tokens) tokens.

    If the sequence fits within window_length already, return unchanged.
    The sink tokens anchor softmax stability; the tail window holds recent context.
    """
    seq_len = input_ids.shape[-1]
    if seq_len <= window_length:
        return input_ids
    tail_size = window_length - num_sink_tokens
    sink = input_ids[:, :num_sink_tokens]
    tail = input_ids[:, -tail_size:]
    truncated = torch.cat([sink, tail], dim=-1)
    dropped = seq_len - window_length
    print(f"    [StreamingLLM] {seq_len:,} → {truncated.shape[-1]:,} tokens "
          f"(dropped {dropped:,} middle tokens, kept {num_sink_tokens} sinks + {tail_size} tail)",
          flush=True)
    return truncated


@torch.no_grad()
def hf_generate(
    model,
    tokenizer: AutoTokenizer,
    user_text: str,
    max_new_tokens: int,
    device: torch.device,
    num_sink_tokens: int,
    window_length: int,
) -> str:
    input_ids = apply_chat_template(tokenizer, user_text).to(device)
    seq_len = input_ids.shape[-1]
    print(f"    input tokens: {seq_len:,}", flush=True)

    input_ids = truncate_to_streaming_window(input_ids, num_sink_tokens, window_length)

    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    outputs = model.generate(
        input_ids,
        generation_config=gen_config,
        use_cache=True,
    )
    generated_ids = outputs[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    output tokens: {len(generated_ids):,}", flush=True)
    torch.cuda.empty_cache()
    return output_text

# ---------------------------------------------------------------------------
# Graph probing
# ---------------------------------------------------------------------------

def run_graph_probing_hf(
    model,
    tokenizer: AutoTokenizer,
    trace_str: str,
    to_probe: List[ProbeTarget],
    span_index: str,
    max_new_tokens: int,
    device: torch.device,
    num_sink_tokens: int,
    window_length: int,
    explicit_causal_encoding: bool = False,
) -> Tuple[List[dict], dict]:
    if not to_probe:
        return [], {}

    new_errors: List[dict] = []
    probe_meta: dict = {}

    for pt in to_probe:
        cat = pt.target
        prompt = build_graph_probe_prompt(
            category=cat,
            trace_str=trace_str,
            span_index=span_index,
            explicit_causal_encoding=explicit_causal_encoding,
            source_category=pt.source_category,
            source_span=pt.source_span,
            weight=pt.weight,
        )
        raw = hf_generate(model, tokenizer, prompt,
                          max_new_tokens=512, device=device,
                          num_sink_tokens=num_sink_tokens, window_length=window_length)
        parsed = parse_json_output(raw)
        if parsed is None:
            print(f"    [probe] {cat}: parse failed")
            probe_meta[cat] = {"present": None, "parse_failed": True}
            continue
        present = parsed.get("present", False)
        probe_meta[cat] = {"present": present, "raw": raw[:300]}
        if present:
            loc = parsed.get("location", "")
            if loc:
                new_errors.append({
                    "category":    cat,
                    "location":    loc,
                    "evidence":    parsed.get("evidence", ""),
                    "description": parsed.get("description", ""),
                    "impact":      parsed.get("impact", "MEDIUM"),
                })
                print(f"    [probe] {cat}: FOUND at {loc}")
            else:
                print(f"    [probe] {cat}: present=true but no location")
        else:
            print(f"    [probe] {cat}: not present")

    return new_errors, probe_meta

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model,
    tokenizer: AutoTokenizer,
    trace_str: str,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
    max_new_tokens: int,
    device: torch.device,
    num_sink_tokens: int,
    window_length: int,
    validate_span_id: bool = True,
    repair_location: bool = False,
    span_index: str = "",
    explicit_causal_encoding: bool = False,
) -> dict:
    # --- Pass 1 ---
    print(f"  [Pass 1]", flush=True)
    p1_prompt = build_pass1_prompt(trace_str, span_index=span_index)
    p1_text   = hf_generate(model, tokenizer, p1_prompt,
                             max_new_tokens=max_new_tokens, device=device,
                             num_sink_tokens=num_sink_tokens, window_length=window_length)

    p1_tok_len = len(tokenizer.encode(p1_prompt, add_special_tokens=False))
    p1_parsed  = parse_json_output(p1_text)
    if p1_parsed is None:
        print(f"  [Pass 1] JSON parse FAILED. Raw (first 500):\n    {p1_text[:500]!r}")
        return {
            "errors": [],
            "scores": [{"reliability_score": 0, "reliability_reasoning": "parse error",
                        "security_score": 5, "security_reasoning": "",
                        "instruction_adherence_score": 0, "instruction_adherence_reasoning": "",
                        "plan_opt_score": 0, "plan_opt_reasoning": "", "overall": 0}],
            "_uq_meta": {
                "pass1_parse_error": True,
                "pass1_raw_response": p1_text[:2000],
                "pass1_prompt_tokens": p1_tok_len,
            },
        }

    p1_errors = p1_parsed.get("errors", [])
    for e in p1_errors:
        if e.get("category"):
            e["category"] = normalize_to_leaf(e["category"])

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
    print(f"  [Pass 1] detected {len(p1_errors)} errors: {detected_cats}")

    conf    = {cat: 1.0 for cat in detected_cats}
    boosted = propagate_confidence(conf, edges, TAXONOMY_CATEGORIES)

    detected_set   = set(detected_cats)
    to_verify_cats = [
        cat for cat in TAXONOMY_CATEGORIES
        if cat not in detected_set and boosted.get(cat, 0.0) > propagation_threshold
    ]

    to_verify: List[ProbeTarget] = []
    for target in to_verify_cats:
        candidates = [(src, w) for src, dst, w in edges if dst == target and conf.get(src, 0.0) > 0]
        if candidates:
            best_src, best_w = max(candidates, key=lambda x: conf.get(x[0], 0.0) * x[1])
            src_span = next(
                (e.get("location", "") for e in p1_errors if e.get("category") == best_src), ""
            )
        else:
            best_src, best_w, src_span = "", 0.0, ""
        to_verify.append(ProbeTarget(target, best_src, src_span, best_w))

    print(f"  [Conf]   {', '.join(f'{k}: {v:.3f}' for k, v in conf.items()) or '(none)'}")

    uq_meta = {
        "valid_json":            True,
        "location_diagnostics":  location_diag,
        "pass1_detected":        detected_cats,
        "pass1_prompt_tokens":   p1_tok_len,
        "pass1_raw_response":    p1_text,
        "confidence":            {k: round(v, 4) for k, v in conf.items()},
        "boosted_scores":        {k: round(v, 4) for k, v in boosted.items() if v > 0},
        "to_verify":             [pt.target for pt in to_verify],
        "propagation_threshold": propagation_threshold,
        "mode":                  "streaming_graph_probe",
        "window_length":         window_length,
        "num_sink_tokens":       num_sink_tokens,
    }

    p2_errors: List[dict] = []
    if to_verify:
        print(f"  [probe] Graph-triggered categories: {[pt.target for pt in to_verify]}")
        p2_errors, probe_meta = run_graph_probing_hf(
            model=model, tokenizer=tokenizer,
            trace_str=trace_str, to_probe=to_verify,
            span_index=span_index, max_new_tokens=max_new_tokens,
            device=device, num_sink_tokens=num_sink_tokens, window_length=window_length,
            explicit_causal_encoding=explicit_causal_encoding,
        )
        uq_meta["graph_probe_meta"] = probe_meta

    if p2_errors and validate_span_id:
        valid_span_ids = extract_span_ids(trace_str)
        p2_errors, p2_loc_diag = validate_and_repair_locations(
            p2_errors, valid_span_ids, repair=repair_location
        )
        if p2_loc_diag["dropped_count"]:
            print(f"  [Gate-p2] dropped {p2_loc_diag['dropped_count']} invalid span_ids")
        uq_meta["p2_location_diagnostics"] = p2_loc_diag

    merged_errors = list(p1_errors)
    p1_key_set = {(e.get("category", ""), e.get("location", "")) for e in p1_errors}
    for e in p2_errors:
        key = (e.get("category", ""), e.get("location", ""))
        if key not in p1_key_set:
            merged_errors.append(e)
            p1_key_set.add(key)

    return {
        "errors": merged_errors,
        "scores": p1_parsed.get("scores", []),
        "_uq_meta": uq_meta,
    }

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, attn_impl: str):
    """Load model with standard HuggingFace AutoModelForCausalLM (no custom attention)."""
    print(f"  Loading {model_id} with device_map='auto', dtype=float16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype          = torch.float16,
        device_map           = "auto",
        trust_remote_code    = True,
        attn_implementation  = attn_impl,
    )
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False

    device = next(model.parameters()).device
    print(f"  Model loaded. Primary device: {device}")
    return model, device

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StreamingLLM (Attention Sinks) baseline for long-context trace analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split",                    type=str,   default="GAIA")
    parser.add_argument("--data_dir",                 type=str,   default=None)
    parser.add_argument("--output_dir",               type=str,   default=None)
    parser.add_argument("--model",                    type=str,   default=MODEL_ID)
    parser.add_argument("--window_length",            type=int,   default=32768,
                        help="Total token window: num_sink_tokens + sliding window tail. "
                             "Must fit in GPU memory for the chosen model.")
    parser.add_argument("--num_sink_tokens",          type=int,   default=4,
                        help="Number of initial tokens kept as attention sinks (paper recommends 4).")
    parser.add_argument("--attn_implementation",      type=str,   default="flash_attention_2",
                        choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--max_new_tokens",           type=int,   default=8000)
    parser.add_argument("--min_tokens",               type=int,   default=0,
                        help="Skip traces shorter than this (set to 131073 for overflow-only).")
    parser.add_argument("--max_tokens",               type=int,   default=0,
                        help="Skip traces longer than this (0 = no limit).")
    parser.add_argument("--validate_span_id",         action="store_true", default=True)
    parser.add_argument("--repair_location",          action="store_true", default=False)
    parser.add_argument("--span_index",               action="store_true", default=False)
    parser.add_argument("--propagation_threshold",    type=float, default=0.10)
    parser.add_argument("--edge_threshold",           type=float, default=0.10)
    parser.add_argument("--causal_only",              action="store_true")
    parser.add_argument("--corr_threshold",           type=float, default=1.0)
    parser.add_argument("--explicit_causal_encoding", action="store_true", default=False)
    parser.add_argument("--graph_input",              type=str,   default=None)
    args = parser.parse_args()

    # --- Paths ---
    data_dir  = Path(args.data_dir) if args.data_dir else BENCH_DIR / "data" / args.split
    model_tag = args.model.split("/")[-1]
    if args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
    elif args.causal_only:
        graph_tag = "causal_only"
    else:
        graph_tag = f"suppes_t{args.edge_threshold}"
    window_tag = f"_sink{args.num_sink_tokens}_win{args.window_length}"
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else LC_DIR / "outputs" / f"outputs_{model_tag}-{args.split}-streaming_{graph_tag}{window_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load causal graph ---
    graph_input_path = Path(args.graph_input) if args.graph_input else GRAPH_INPUT
    print(f"Loading Suppes graph from {graph_input_path} ...")
    edges = load_suppes_edges(
        threshold      = args.edge_threshold,
        causal_only    = args.causal_only and args.corr_threshold >= 1.0,
        corr_threshold = args.corr_threshold,
        graph_input    = graph_input_path,
    )
    print(f"  {len(edges)} edges loaded")
    for src, dst, w in edges[:5]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 5:
        print(f"    ... and {len(edges) - 5} more")

    # --- Load tokenizer + model ---
    print(f"\nLoading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\nLoading StreamingLLM model ...")
    print(f"  window_length={args.window_length:,}  num_sink_tokens={args.num_sink_tokens}")
    model, device = load_model(model_id=args.model, attn_impl=args.attn_implementation)
    print("Model loaded.\n")

    # --- Process traces ---
    file_paths = sorted(glob.glob(str(data_dir / "*.json")))
    print(f"Processing {len(file_paths)} traces from {data_dir}")
    print(f"Output → {out_dir}")
    if args.min_tokens > 0:
        print(f"Skipping traces with prompt token count < {args.min_tokens:,}")
    if args.max_tokens > 0:
        print(f"Skipping traces with prompt token count > {args.max_tokens:,}")
    print()

    skipped_short = 0
    skipped_error = 0
    for fp in tqdm(file_paths):
        out_file  = out_dir / Path(fp).name
        meta_file = out_dir / ("_meta_" + Path(fp).name)
        if out_file.exists():
            continue

        trace_str = load_trace(fp)
        span_idx  = build_span_index(trace_str) if args.span_index else ""

        if args.min_tokens > 0 or args.max_tokens > 0:
            prompt_text = build_pass1_prompt(trace_str, span_index=span_idx)
            prompt_toks = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if args.min_tokens > 0 and prompt_toks < args.min_tokens:
                print(f"  Skipping {Path(fp).name} ({prompt_toks:,} tokens < {args.min_tokens:,})")
                skipped_short += 1
                continue
            if args.max_tokens > 0 and prompt_toks > args.max_tokens:
                print(f"  Skipping {Path(fp).name} ({prompt_toks:,} tokens > {args.max_tokens:,})")
                skipped_short += 1
                continue

        print(f"\n{'='*60}\n{Path(fp).name}", flush=True)
        try:
            result = run_pipeline(
                model                    = model,
                tokenizer                = tokenizer,
                trace_str                = trace_str,
                edges                    = edges,
                propagation_threshold    = args.propagation_threshold,
                max_new_tokens           = args.max_new_tokens,
                device                   = device,
                num_sink_tokens          = args.num_sink_tokens,
                window_length            = args.window_length,
                validate_span_id         = args.validate_span_id,
                repair_location          = args.repair_location,
                span_index               = span_idx,
                explicit_causal_encoding = args.explicit_causal_encoding,
            )
        except Exception as e:
            import traceback
            print(f"\nError on {fp}:\n{traceback.format_exc()}")
            result = {
                "errors": [],
                "scores": [],
                "_uq_meta": {"error": str(e)},
            }
            skipped_error += 1

        public_result = {k: v for k, v in result.items() if k != "_uq_meta"}
        with open(out_file, "w") as f:
            json.dump(public_result, f)
        with open(meta_file, "w") as f:
            json.dump(result.get("_uq_meta", {}), f, indent=2)

    print(f"\nDone. skipped_short={skipped_short}  errors={skipped_error}")


if __name__ == "__main__":
    main()
