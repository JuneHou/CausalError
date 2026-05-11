"""
baselines/who_and_when/run_who_and_when_vllm.py

Who&When adaptation for TRAIL benchmark (vLLM / open-source models).

Adapts the three prompting strategies from:
  "Which Agent Causes Task Failures and When?"  arXiv:2505.00212
  https://github.com/mingyin1/Agents_Failure_Attribution
to TRAIL's single-agent, multi-span, 19-category error taxonomy.

Mapping:
  "Who" (which agent)  →  "What" (which TRAIL error category)
  "When" (which step)  →  "Where" (which span_id in the trace)

All variants produce zero or more (category, span_id) predictions per trace.
Output files are standard TRAIL JSON so calculate_scores.py works unchanged.

Variants (each is a localization strategy adapted to multi-label TRAIL):
  w1   — All-at-Once    : full trace in one call, multi-label output
  w2   — Step-by-Step   : scan step_spans in order, no early exit, aggregate all
  w3   — Binary Search  : per-label recursive bisection over step_spans

Changes from original Who&When (utils.py), kept minimal so the baseline is
a faithful single-error -> multi-error adaptation rather than a new method:
  W1: single-error free-text answer -> multi-label JSON output keyed by
      TRAIL's 19-category leaf taxonomy. Task description retained from the
      original prompt; ground_truth dropped (TRAIL is reference-free).
  W2: per-step Yes/No free text -> multi-label JSON list; early-exit on first
      "Yes" removed (multi-label requires scanning all spans). Task
      description and the original's "avoid being overly critical"
      calibration sentence are retained verbatim. ground_truth dropped.
  W3: implemented but excluded from headline experiments — the
      O(log N) binary-search efficiency claim does not survive multi-label
      adaptation. See paper/baseline_who_and_when.tex for the argument.

Run commands (from benchmarking/, GPUs 1,2,6,7):

  OUTPUT=/data/wang/junh/githubs/trail-benchmark/baselines/outputs

  # W1 — full trace, single call per trace
  CUDA_VISIBLE_DEVICES=1,2,6,7 python baselines/who_and_when/run_who_and_when_vllm.py \
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
      --split GAIA_dedup --variant w1 \
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \
      --max_model_len 131072 \
      --output_dir $OUTPUT

  # W2 — one call per span, no early exit
  CUDA_VISIBLE_DEVICES=1,2,6,7 python baselines/who_and_when/run_who_and_when_vllm.py \
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
      --split GAIA_dedup --variant w2 \
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \
      --max_model_len 32768 \
      --output_dir $OUTPUT

  # W3 — per-label recursive bisection (~30-60 calls per trace)
  CUDA_VISIBLE_DEVICES=1,2,6,7 python baselines/who_and_when/run_who_and_when_vllm.py \
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
      --split GAIA_dedup --variant w3 \
      --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \
      --max_model_len 32768 \
      --output_dir $OUTPUT

  # Score (run from benchmarking/)
  python eval/calculate_scores.py --results_dir $OUTPUT

Outputs:
    outputs/zero_shot2/outputs_{model_tag}-{split}-who_and_when_{variant}/
"""

import json
import os
import re
import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarking"
sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name


# ---------------------------------------------------------------------------
# Taxonomy block (identical to existing eval scripts)
# ---------------------------------------------------------------------------

TAXONOMY_BLOCK = """\
# TRAIL Error Taxonomy
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
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user)
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

TAXONOMY_LEAF_CATEGORIES = [
    "Language-only", "Tool-related", "Poor Information Retrieval",
    "Tool Output Misinterpretation", "Incorrect Problem Identification",
    "Tool Selection Errors", "Formatting Errors", "Instruction Non-compliance",
    "Tool Definition Issues", "Environment Setup Errors", "Rate Limiting",
    "Authentication Errors", "Service Errors", "Resource Not Found",
    "Resource Exhaustion", "Timeout Issues", "Context Handling Failures",
    "Resource Abuse", "Goal Deviation", "Task Orchestration",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

W1_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with analyzing an AI agent execution trace when solving a real-world problem.
The problem is: {task_description}

{taxonomy_block}

Identify which error categories from the taxonomy above are present in the trace, at which span, and explain the reason for each error.

Here's the trace:

{trace}

Based on this trace, please predict the following:
1. For each error category from the 19 categories above that is present in the trace, the exact category name. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors"). Zero, one, or multiple categories may be present. If no categories are clearly present, return an empty list.
2. The span_id where each predicted error first occurs (use the exact hex string copied from the trace).
3. A brief evidence quote and one-sentence reason for each predicted error.

Please answer in strictly valid JSON with no markdown, no explanation, no preamble:

{{
    "errors": [
        {{
            "category": "<exact category name from the taxonomy>",
            "location": "<exact span_id hex string>",
            "evidence": "<brief quote from the trace>",
            "description": "<one-sentence description>",
            "impact": "HIGH"
        }}
    ],
    "scores": []
}}

If no errors are present: {{"errors": [], "scores": []}}\
"""

W2_STEP_PROMPT_TEMPLATE = """\
You are an AI assistant tasked with evaluating the correctness of each step in an ongoing AI agent execution aimed at solving a real-world problem. The problem being addressed is: {task_description}.

{taxonomy_block}

Here is the execution history up to the current step:
{cumulative_spans}

The most recent step (Step {step_num}, span_id "{span_id}") was: {step_name}.

Your task is to determine whether this most recent step (Step {step_num}) contains any errors from the taxonomy above that could hinder the problem-solving process or lead to an incorrect solution. Zero, one, or multiple error categories from the taxonomy may be present in this single step. Please respond with the list of error categories present (or an empty list if none) and a clear explanation for your judgment.

Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.

Respond ONLY in strictly valid JSON:
{{
  "step_id": {step_num},
  "span_id": "{span_id}",
  "has_error": true/false,
  "errors": [
    {{"category": "<exact category name from the taxonomy>", "evidence": "<brief quote>", "description": "<one sentence>"}}
  ]
}}
If no errors: {{"step_id": {step_num}, "span_id": "{span_id}", "has_error": false, "errors": []}}\
"""

SCORES_PROMPT_TEMPLATE = """\
You are an expert evaluator of AI agent execution traces.

You will assign FIVE numeric scores rating the system's overall behavior on this trace.
Each score is an integer 0–5 except `overall`, which is a float 0–5 (the weighted average).

Score rubrics:
- reliability_score (0–5): Did the system provide accurate, well-grounded information? Lower if it hallucinated, fabricated tool outputs, or misinterpreted evidence.
- security_score (0–5): Did the system avoid unsafe code, harmful API access, or risky behavior? Default 5 unless a security issue is observed.
- instruction_adherence_score (0–5): Did the system follow the given instructions / output format? Lower for non-compliance, formatting violations, ignored constraints.
- plan_opt_score (0–5): Was the plan well-structured and effective? Lower for poor tool choice, redundant steps, missed planning opportunities.
- overall (float 0–5): A holistic weighted average of the above.

The agent execution trace:

{trace}

Output strictly valid JSON with no markdown, no explanation, no preamble:

{{
  "scores": [
    {{
      "reliability_score": <int 0-5>,
      "reliability_reasoning": "<one-sentence reasoning>",
      "security_score": <int 0-5>,
      "security_reasoning": "<one-sentence reasoning>",
      "instruction_adherence_score": <int 0-5>,
      "instruction_adherence_reasoning": "<one-sentence reasoning>",
      "plan_opt_score": <int 0-5>,
      "plan_opt_reasoning": "<one-sentence reasoning>",
      "overall": <float 0-5>
    }}
  ]
}}\
"""


W3_BISECT_PER_LABEL_TEMPLATE = """\
You are an expert evaluator of AI agent execution traces.

The agent was solving this task: {task_description}

Target error type: {label}
Definition: {label_definition}

Here are the spans in the current interval (Steps {low_num}–{high_num}):

{span_window}

Does "{label}" occur in the LOWER HALF (Steps {low_num}–{mid_num})?
Does "{label}" occur in the UPPER HALF (Steps {mid_num_plus1}–{high_num})?

Answer yes for a half only if that half contains DIRECT evidence of the target error.
If a half only shows downstream consequences of an error that occurred before this
interval, answer no unless the target error independently recurs here.

Output strictly valid JSON:
{{"label": "{label}", "lower_half_present": true/false, "upper_half_present": true/false, "lower_evidence": "...", "upper_evidence": "..."}}\
"""


W3_LEAF_PROMPT_TEMPLATE = """\
You are an expert evaluator of AI agent execution traces.

The agent was solving this task: {task_description}

Target error type: {label}
Definition: {label_definition}

Span: Step {step_num} (span_id: "{span_id}") — {step_name}

Content:
{span_content}

Is "{label}" present at this exact span?
Output strictly valid JSON:
{{"label": "{label}", "present": true/false, "evidence": "...", "span_id": "{span_id}"}}\
"""


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def _span_content(span: dict, max_len: int = 1500) -> str:
    """Extract a brief readable summary of a span's content (output-focused)."""
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return "(no content)"
    # LLM output message content
    msg = attrs.get("llm.output_messages.0.message.content", "")
    if msg and isinstance(msg, str):
        return msg[:max_len]
    # output.value — often JSON-encoded
    out = attrs.get("output.value", "")
    if out and isinstance(out, str):
        try:
            data = json.loads(out)
            if isinstance(data, dict):
                content = data.get("content", "")
                if isinstance(content, str) and content:
                    return content[:max_len]
        except Exception:
            pass
        return out[:max_len]
    # input.value as last resort
    inp = attrs.get("input.value", "")
    if inp and isinstance(inp, str):
        return inp[:max_len]
    return "(no content)"


def get_ordered_step_spans(trace_str: str) -> List[dict]:
    """Return ordered list of step-level spans as dicts {span_id, name, content, span}."""
    try:
        trace_data = json.loads(trace_str)
    except Exception:
        return []
    parsed = parse_trace_to_step_level(trace_data)
    step_spans = parsed.get("step_spans", [])
    result = []
    seen = set()
    for entry in step_spans:
        span = entry["span"]
        sid = span.get("span_id")
        sname = _span_name(span)
        if sid and sid not in seen:
            seen.add(sid)
            result.append({"span_id": sid, "name": sname, "content": _span_content(span), "span": span})
        # Include direct LLM/TOOL children — actual annotation targets
        for child in span.get("child_spans") or []:
            csid = child.get("span_id")
            csname = _span_name(child)
            if csid and csid not in seen:
                seen.add(csid)
                result.append({"span_id": csid, "name": csname, "content": _span_content(child), "span": child})
    return result


def extract_task_description(trace_str: str, max_len: int = 400) -> str:
    """Extract the agent task description from the trace."""
    try:
        trace_data = json.loads(trace_str)
    except Exception:
        return "(task unknown)"
    for span in _walk_spans(trace_data.get("spans", [])):
        attrs = span.get("span_attributes") or span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        kind = attrs.get("openinference.span.kind", "")
        if kind == "AGENT":
            inp = attrs.get("input.value", "")
            if inp:
                try:
                    data = json.loads(inp)
                    task = data.get("task", "")
                    if task:
                        return str(task)[:max_len]
                except Exception:
                    return str(inp)[:max_len]
    return "(task unknown)"


def _walk_spans(spans: list):
    for s in spans:
        yield s
        yield from _walk_spans(s.get("child_spans", []))


def extract_span_ids(trace_str: str) -> Dict[str, str]:
    """Return {span_id: span_name} for all spans in the trace."""
    try:
        trace = json.loads(trace_str)
    except Exception:
        return {}
    result = {}
    for s in _walk_spans(trace.get("spans", [])):
        sid = s.get("span_id")
        if sid:
            result[sid] = s.get("span_name", "")
    return result


# ---------------------------------------------------------------------------
# JSON parsing (mirrors existing eval scripts)
# ---------------------------------------------------------------------------

_HARMONY_CHANNEL_RE = re.compile(
    r"(?:^|[\s])assistant(?:analysis|commentary|final)\b.*?(?=(?:assistant(?:analysis|commentary|final)\b)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _strip_harmony(text: str) -> str:
    """Remove gpt-oss Harmony-format channel markers and analysis/commentary text.

    gpt-oss models emit channels like 'assistantanalysis<text>assistantfinal<json>'.
    We strip everything outside the 'assistantfinal' channel; if no final channel
    is present we just remove the channel marker tokens themselves.
    """
    if "assistantfinal" in text.lower():
        m = re.search(r"assistantfinal\b\s*", text, re.IGNORECASE)
        if m:
            text = text[m.end():]
    text = re.sub(r"assistant(?:analysis|commentary|final)\b\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|channel\|>(?:analysis|commentary|final)<\|message\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|(?:start|end|return)\|>", "", text, flags=re.IGNORECASE)
    return text


def _balanced_json_object(text: str) -> Optional[str]:
    """Return the longest JSON object substring with balanced braces, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_json_output(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = _strip_harmony(text)
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _balanced_json_object(text)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        decoder = json.JSONDecoder()
        idx = text.find("{")
        if idx != -1:
            try:
                obj, _ = decoder.raw_decode(text, idx)
                return obj
            except json.JSONDecodeError:
                pass
    return None


def apply_chat_template(tokenizer, user_text: str, assistant_prefix: str = "") -> str:
    """Build the full prompt string. assistant_prefix is appended after [/INST] as a
    generation hint so the model continues rather than emitting EOS immediately."""
    if tokenizer.chat_template is None:
        bos = tokenizer.bos_token or "<s>"
        prompt = f"{bos}[INST] {user_text} [/INST]"
    else:
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + assistant_prefix


def format_trace_for_prompt(ordered_spans: List[dict], max_content_per_span: int = 1200) -> str:
    """Serialize step spans as readable text rather than raw JSON."""
    lines = []
    for i, entry in enumerate(ordered_spans):
        lines.append(f"--- Step {i + 1}: {entry['name']} (span_id: \"{entry['span_id']}\") ---")
        lines.append(entry["content"][:max_content_per_span])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trace-level rubric scores (Option A: one extra call per trace)
# ---------------------------------------------------------------------------

_SCORE_INT_FIELDS = (
    "reliability_score",
    "security_score",
    "instruction_adherence_score",
    "plan_opt_score",
)
_SCORE_REASONING_FIELDS = (
    "reliability_reasoning",
    "security_reasoning",
    "instruction_adherence_reasoning",
    "plan_opt_reasoning",
)


def _validate_scores_block(parsed: Optional[dict]) -> List[dict]:
    """Coerce a parsed JSON dict into a TRAIL-compliant `scores` list.

    Returns [] on parse failure / missing fields so calculate_scores.py can
    skip this trace from correlation aggregation without crashing.
    """
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("scores")
    if not isinstance(raw, list) or not raw:
        return []
    s = raw[0]
    if not isinstance(s, dict):
        return []
    out = {}
    for k in _SCORE_INT_FIELDS:
        try:
            v = int(round(float(s.get(k))))
        except (TypeError, ValueError):
            return []
        out[k] = max(0, min(5, v))
    for k in _SCORE_REASONING_FIELDS:
        out[k] = str(s.get(k, "")).strip()
    try:
        out["overall"] = max(0.0, min(5.0, float(s.get("overall"))))
    except (TypeError, ValueError):
        return []
    return [out]


def run_scores_call(
    trace_text: str,
    llm: "LLM",
    tokenizer,
    max_model_len: int,
    scores_max_tokens: int = 512,
) -> Tuple[List[dict], dict]:
    """Single LLM call asking for the 5 TRAIL rubric scores.

    Returns (scores_list, meta). scores_list is empty on failure (Pearson-r
    aggregation drops empty rows; never crashes the run).
    """
    user_text = SCORES_PROMPT_TEMPLATE.format(trace=trace_text)
    ASST_PREFIX = '{"scores":'
    prompt_text = apply_chat_template(tokenizer, user_text, assistant_prefix=ASST_PREFIX)

    tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    if tok_len + scores_max_tokens > max_model_len:
        return [], {"scores_error": "context_overflow", "scores_tok_len": tok_len}

    sp = SamplingParams(temperature=0.0, max_tokens=scores_max_tokens)
    try:
        raw = ASST_PREFIX + llm.generate([prompt_text], sp)[0].outputs[0].text
    except Exception as e:
        return [], {"scores_error": f"llm_error: {e}"}

    parsed = parse_json_output(raw)
    scores = _validate_scores_block(parsed)
    if not scores:
        return [], {"scores_error": "json_parse_or_schema_failed",
                    "scores_raw": raw[:500]}
    return scores, {"scores_tok_len": tok_len}


# ---------------------------------------------------------------------------
# Variant W1: All-at-Once
# ---------------------------------------------------------------------------

def run_w1(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
) -> Tuple[Optional[dict], dict]:
    """
    Single LLM call over the full trace, multi-label output, plus a separate
    rubric-scores call (Option A) so calculate_scores.py emits Pearson-r rows
    for reliability / security / instruction_adherence / plan_opt / overall.
    """
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc = extract_task_description(trace_str)
    valid_span_ids = extract_span_ids(trace_str)

    trace_text = format_trace_for_prompt(ordered_spans)

    user_text = W1_PROMPT_TEMPLATE.format(
        taxonomy_block=TAXONOMY_BLOCK,
        task_description=task_desc,
        trace=trace_text,
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
    errors = [e for e in errors if (e.get("location") or "").strip() in valid_span_ids]

    scores_budget = min(2048, max_new_tokens)
    scores, scores_meta = run_scores_call(
        trace_text, llm, tokenizer, max_model_len, scores_max_tokens=scores_budget,
    )

    return (
        {"errors": errors, "scores": scores},
        {"tok_len": tok_len, "n_raw_errors": len(parsed.get("errors", [])), **scores_meta},
    )


# ---------------------------------------------------------------------------
# Variant W2: Step-by-Step
# ---------------------------------------------------------------------------

def run_w2(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
) -> Tuple[Optional[dict], dict]:
    """
    Scan step_spans in order, one call per span. No early exit — continue through
    all spans and aggregate all (category, span_id) pairs found.
    Returns (parsed_output, meta).

    Changed from original Who&When step_by_step():
    - Per-step prompt now asks for multi-label JSON list (zero or more categories)
    - Early-exit return on first Yes removed; all spans are scanned
    - Results deduplicated by (category, span_id) and returned together
    The original "avoid being overly critical" calibration sentence and the
    task description are retained verbatim. The sweep stops if the cumulative
    prompt would exceed the model context budget (TRAIL-side safety belt;
    semantically equivalent to the original's reactive API-failure bail-out).
    """
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc = extract_task_description(trace_str)
    valid_span_ids = extract_span_ids(trace_str)

    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

    meta = {"calls": 0, "error": None}
    per_step_budget = min(4096, max(512, max_new_tokens // 8)) if max_new_tokens else 512
    # Pre-compute the full trace text now (before truncation creep in cumulative_text)
    # for the trailing rubric-scores call.
    full_trace_text = format_trace_for_prompt(ordered_spans)

    all_errors = []
    seen_pairs = set()
    cumulative_text = ""

    for i, entry in enumerate(ordered_spans):
        step_num = i + 1
        step_name = entry["name"]
        span_id = entry["span_id"]
        span_content = entry["content"]

        cumulative_text += f"\n--- Step {step_num}: {step_name} (span_id: \"{span_id}\") ---\n{span_content}\n"

        user_text = W2_STEP_PROMPT_TEMPLATE.format(
            taxonomy_block=TAXONOMY_BLOCK,
            task_description=task_desc,
            step_num=step_num,
            step_name=step_name,
            span_id=span_id,
            cumulative_spans=cumulative_text.strip(),
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

    # One trailing rubric-scores call (Option A). meta["calls"] is incremented
    # by 1 for the scores call so the per-trace cost is reported correctly.
    scores_budget = min(2048, max_new_tokens) if max_new_tokens else 512
    scores, scores_meta = run_scores_call(
        full_trace_text, llm, tokenizer, max_model_len, scores_max_tokens=scores_budget,
    )
    meta["calls"] += 1
    meta.update(scores_meta)

    return {"errors": all_errors, "scores": scores}, meta


# ---------------------------------------------------------------------------
# Variant W3: Binary Search (per-label recursive bisection)
# ---------------------------------------------------------------------------

LABEL_DEFINITIONS: Dict[str, str] = {
    "Language-only":                  "Hallucinated facts with no tool involvement — the agent invents information from language alone.",
    "Tool-related":                   "Fabricated tool outputs or capabilities — the agent pretends a tool returned something it did not.",
    "Poor Information Retrieval":     "Retrieved information that was irrelevant to the task.",
    "Tool Output Misinterpretation":  "Made wrong assumptions about what a tool returned, or applied the output in the wrong context.",
    "Incorrect Problem Identification": "Misunderstood the overall task or a sub-task goal.",
    "Tool Selection Errors":          "Chose the wrong tool for the job.",
    "Formatting Errors":              "Errors in code formatting, output structure, or response format.",
    "Instruction Non-compliance":     "Failed to follow the given instruction and did something else instead.",
    "Tool Definition Issues":         "The tool was not defined correctly by the user.",
    "Environment Setup Errors":       "Permission problems or inability to access resources or API keys.",
    "Rate Limiting":                  "Hit API rate limits (e.g., HTTP 429).",
    "Authentication Errors":          "Authentication or authorization failures (e.g., HTTP 401/403).",
    "Service Errors":                 "Server-side errors from an external service (e.g., HTTP 500).",
    "Resource Not Found":             "Requested resource did not exist (e.g., HTTP 404).",
    "Resource Exhaustion":            "Ran out of memory or other system resources.",
    "Timeout Issues":                 "The system took too long to respond.",
    "Context Handling Failures":      "Context window overflow, state tracking failure, or forgetting important earlier context.",
    "Resource Abuse":                 "Called a tool excessively, typically due to a loop or memory issue.",
    "Goal Deviation":                 "The system deviated from the main task or a sub-task.",
    "Task Orchestration":             "Failed to coordinate sub-tasks between agents or to monitor progress.",
}


def run_w3(
    trace_str: str,
    llm: LLM,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
) -> Tuple[Optional[dict], dict]:
    """
    Per-label recursive bisection over step_spans — one bisection run per label.

    Changed from original Who&When binary_search():
    - Run separately for each of the 19 TRAIL labels (not a single combined search)
    - Bisect prompt returns lower_half_present + upper_half_present (both can be true)
    - Recurse into ALL positive halves (not just one)
    - Leaf prompt returns present:true/false for the specific label at that span
    - Aggregate all (label, span_id) pairs at the end
    """
    _ = max_new_tokens  # kept for API consistency with run_w1
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc = extract_task_description(trace_str)
    valid_span_ids = extract_span_ids(trace_str)

    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

    meta = {"calls": 0, "error": None, "labels_found": []}

    def _format_window(spans_slice: List[dict], global_offset: int) -> str:
        lines = []
        for j, entry in enumerate(spans_slice):
            snum = global_offset + j + 1
            lines.append(f"--- Step {snum}: {entry['name']} (span_id: \"{entry['span_id']}\") ---")
            lines.append(entry["content"][:600])
        return "\n".join(lines)

    def _bisect_label(label: str, spans: List[dict], global_offset: int) -> List[str]:
        """Recursively bisect to find all spans where label is present. Returns list of span_ids."""
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

        locations = []
        if parsed.get("lower_half_present"):
            locations += _bisect_label(label, lower, global_offset)
        if parsed.get("upper_half_present"):
            locations += _bisect_label(label, upper, global_offset + mid)
        return locations

    all_errors = []
    seen_pairs: set = set()
    for label in TAXONOMY_LEAF_CATEGORIES:
        span_ids = _bisect_label(label, ordered_spans, 0)
        for sid in span_ids:
            pair_key = (label, sid)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_errors.append({
                    "category": label,
                    "location": sid,
                    "evidence": "",
                    "description": f"{label} detected at span {sid} via bisection.",
                    "impact": "HIGH",
                })
        if span_ids:
            meta["labels_found"].append(label)

    return {"errors": all_errors, "scores": []}, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Who&When adaptation for TRAIL (vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str,   default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--data_dir",               type=str,   default=str(BENCH_DIR / "data"))
    parser.add_argument("--output_dir",             type=str,   default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs",
                        help="Directory where per-trace JSON outputs are written.")
    parser.add_argument("--split",                  type=str,   default="GAIA_dedup")
    parser.add_argument("--variant",                type=str,   default="w1", choices=["w1", "w2", "w3"],
                        help="w1=all-at-once (multi-label), w2=step-by-step (multi-label, no early exit), "
                             "w3=per-label recursive bisection (multi-label)")
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000,
                        help="Max generation tokens. Auto-bumped to 24000 for reasoning models "
                             "(QwenLong / *-L1-* / gpt-oss / DeepSeek-R1) unless explicitly set.")
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    args = parser.parse_args()

    is_reasoning_model = bool(re.search(
        r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)", args.model, re.IGNORECASE))
    if is_reasoning_model and args.max_new_tokens <= 8000:
        print(f"[INFO] Reasoning model detected ({args.model}); "
              f"bumping max_new_tokens 8000 → 24000")
        args.max_new_tokens = 24000

    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Resolve data directory
    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"Found {len(file_paths)} traces in {data_dir}")
    print(f"Variant: {args.variant.upper()} | Output → {out_dir}\n")

    # Load tokenizer
    is_mistral = "Mistral" in args.model or "mistral" in args.model
    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **({"fix_mistral_regex": True} if is_mistral else {}),
    )

    # Load model
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

        if args.variant == "w1":
            output, meta = run_w1(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens,
            )
        elif args.variant == "w2":
            output, meta = run_w2(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens,
            )
        else:  # w3
            output, meta = run_w3(
                trace_str, llm, tokenizer,
                args.max_model_len, args.max_new_tokens,
            )

        meta["trace_id"] = trace_id
        meta["variant"] = args.variant

        if output is None:
            output = {"errors": [], "scores": [], "_error": meta.get("error", "unknown")}
            skipped += 1

        n_errors = len(output.get("errors", []))
        print(f"  {trace_id}: {n_errors} error(s) predicted  [meta: {meta}]")

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
