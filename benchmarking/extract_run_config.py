#!/usr/bin/env python3
"""
Extract run configuration from TRAIL trace JSON files for replication.

Supports:
- GAIA: OpenDeepResearch-style traces; extracts task question, model, and
  first LLM call messages (system + user) from LiteLLMModel.__call__ spans.
- SWE Bench: CodeAct-style traces; extracts system prompt, input prompt, model
  from the first LLM span.

Usage (from benchmarking/):
  python extract_run_config.py --trace_file "data/GAIA/<trace_id>.json"
  python extract_run_config.py --trace_file "data/GAIA/<trace_id>.json" --out_dir extracted_configs
  python extract_run_config.py --trace_file "data/SWE Bench/<trace_id>.json" --out_dir extracted_configs
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple



def _is_gaia_trace(trace_data: Dict[str, Any]) -> bool:
    """Heuristic: GAIA traces have 'GAIA-Samples' in service_name or nested child_spans with answer_single_question."""
    for span in trace_data.get("spans", []):
        if "GAIA" in str(span.get("service_name", "")):
            return True
        if _span_contains_gaia(span):
            return True
    return False


def _span_contains_gaia(span: Dict) -> bool:
    if "answer_single_question" in str(span.get("span_name", "")):
        return True
    for child in span.get("child_spans", []):
        if _span_contains_gaia(child):
            return True
    return False


def _collect_question_from_gaia_logs(span: Dict) -> Optional[str]:
    """Extract 'question' from logs[].body['function.arguments']['example']['question'] or function.output."""
    for log in span.get("logs", []):
        body = log.get("body") or {}
        args = body.get("function.arguments") or {}
        if isinstance(args, dict):
            example = args.get("example")
            if isinstance(example, dict) and "question" in example:
                return example["question"]
        out = body.get("function.output")
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            if "question" in out[0]:
                return out[0]["question"]
    return None


def _find_question_in_spans(spans: List[Dict]) -> Optional[str]:
    for span in spans:
        q = _collect_question_from_gaia_logs(span)
        if q:
            return q
        q = _find_question_in_spans(span.get("child_spans", []))
        if q:
            return q
    return None


def _get_llm_messages_from_span(span: Dict) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    """
    If span is LiteLLMModel.__call__, extract model name and messages from span_attributes.
    Returns (model_name, messages) where messages is list of {"role": ..., "content": ...}.
    """
    if span.get("span_name") != "LiteLLMModel.__call__":
        return None
    attrs = span.get("span_attributes") or {}
    model = attrs.get("llm.model_name")
    if not model:
        return None
    messages = []
    i = 0
    while True:
        content_key = f"llm.input_messages.{i}.message.content"
        role_key = f"llm.input_messages.{i}.message.role"
        if content_key not in attrs:
            break
        content = attrs[content_key]
        role = attrs.get(role_key, "user")
        messages.append({"role": role, "content": content})
        i += 1
    if not messages:
        return None
    return (model, messages)


def _first_llm_span_and_config(spans: List[Dict]) -> Optional[Tuple[Dict, str, List[Dict]]]:
    """Depth-first find first LiteLLMModel.__call__ span; return (span, model, messages)."""
    for span in spans:
        res = _get_llm_messages_from_span(span)
        if res is not None:
            model, messages = res
            return (span, model, messages)
        child_res = _first_llm_span_and_config(span.get("child_spans", []))
        if child_res is not None:
            return child_res
    return None


def _get_llm_step_from_span(span: Dict) -> Optional[Dict[str, Any]]:
    """
    If span is LiteLLMModel.__call__, return a step dict: span_id, model, messages, original_output.
    Used for full-trace replication (replay every LLM call in order).
    """
    if span.get("span_name") != "LiteLLMModel.__call__":
        return None
    attrs = span.get("span_attributes") or {}
    model = attrs.get("llm.model_name")
    if not model:
        return None
    messages = []
    i = 0
    while True:
        content_key = f"llm.input_messages.{i}.message.content"
        role_key = f"llm.input_messages.{i}.message.role"
        if content_key not in attrs:
            break
        content = attrs[content_key]
        role = attrs.get(role_key, "user")
        messages.append({"role": role, "content": content})
        i += 1
    if not messages:
        return None
    output_content = attrs.get("llm.output_messages.0.message.content", "")
    return {
        "span_id": span.get("span_id", ""),
        "model": model,
        "messages": messages,
        "original_output": output_content,
    }


def collect_all_llm_steps(spans: List[Dict], order: List[Dict[str, Any]]) -> None:
    """
    Depth-first walk spans and append every LiteLLMModel.__call__ step to order.
    Preserves execution order of the trace (DFS = order of execution in a single thread).
    """
    for span in spans:
        step = _get_llm_step_from_span(span)
        if step is not None:
            order.append(step)
        collect_all_llm_steps(span.get("child_spans", []), order)


def extract_gaia(trace_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """Extract run config from a GAIA trace (first LLM step only; for backward compat)."""
    full = extract_gaia_full_trace(trace_data, trace_id)
    # Expose first-step fields for callers that only want first call
    first_messages = (full.get("llm_steps") or [{}])[0].get("messages", []) if full.get("llm_steps") else []
    first_system_prompt = ""
    first_user_prompt = ""
    for m in first_messages:
        if m.get("role") == "system":
            first_system_prompt = m.get("content") or ""
        elif m.get("role") == "user":
            first_user_prompt = m.get("content") or ""
            break
    full["system_prompt"] = first_system_prompt
    full["input_prompt"] = first_user_prompt or full.get("input_prompt", "")
    full["first_llm_messages"] = first_messages
    return full


def extract_gaia_full_trace(trace_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    Extract full run config from a GAIA trace: one data point = one trace = all LLM steps in order.
    Returns config with llm_steps: [ { span_id, model, messages, original_output }, ... ].
    All fields are taken only from the trace; no synthetic or default values are injected.
    Raises ValueError if required data is missing from the trace.
    """
    spans = trace_data.get("spans", [])
    question = _find_question_in_spans(spans)
    llm_steps: List[Dict[str, Any]] = []
    collect_all_llm_steps(spans, llm_steps)

    if not llm_steps:
        raise ValueError(f"GAIA trace {trace_id}: no LLM steps (LiteLLMModel.__call__) found in trace")

    first_step = llm_steps[0]
    model_id = first_step.get("model")
    if not model_id:
        raise ValueError(f"GAIA trace {trace_id}: first LLM step has no model (llm.model_name missing in trace)")

    first_user_prompt = ""
    for m in first_step.get("messages", []):
        if m.get("role") == "user":
            first_user_prompt = m.get("content") or ""
            break

    return {
        "trace_id": trace_id,
        "split": "GAIA",
        "model_id": model_id,
        "question": question,
        "input_prompt": first_user_prompt,
        "llm_steps": llm_steps,
    }


def extract_swe_bench(trace_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """Extract run config from a SWE Bench trace (first LLM span). Only uses data from the trace; raises if missing."""
    spans = trace_data.get("spans", [])
    first_llm = _first_llm_span_and_config(spans)
    if not first_llm:
        raise ValueError(f"SWE Bench trace {trace_id}: no LLM span (LiteLLMModel.__call__) found in trace")
    _span, model_id, first_messages = first_llm
    if not model_id:
        raise ValueError(f"SWE Bench trace {trace_id}: LLM span has no model (llm.model_name missing in trace)")
    system_prompt = ""
    input_prompt = ""
    for m in first_messages:
        if m.get("role") == "system":
            system_prompt = m.get("content") or ""
        elif m.get("role") == "user":
            input_prompt = m.get("content") or ""
            break
    return {
        "trace_id": trace_id,
        "split": "SWE Bench",
        "model_id": model_id,
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
        "first_llm_messages": first_messages,
    }


def extract_run_config(trace_file: str) -> Dict[str, Any]:
    """Load trace JSON and return extracted run config (GAIA or SWE Bench)."""
    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    trace_id = trace_data.get("trace_id") or os.path.splitext(os.path.basename(trace_file))[0]
    if _is_gaia_trace(trace_data):
        return extract_gaia(trace_data, trace_id)
    return extract_swe_bench(trace_data, trace_id)


def main():
    parser = argparse.ArgumentParser(description="Extract run config from TRAIL trace JSON")
    parser.add_argument("--trace_file", required=True, help="Path to trace JSON file")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="If set, write config JSON and prompt .txt files here",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.trace_file):
        raise SystemExit(f"Trace file not found: {args.trace_file}")

    config = extract_run_config(args.trace_file)
    trace_id = config["trace_id"]

    print(json.dumps(config, indent=2))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.join(args.out_dir, trace_id)
        with open(base + "_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        if config.get("system_prompt"):
            with open(base + "_system_prompt.txt", "w", encoding="utf-8") as f:
                f.write(config["system_prompt"])
        if config.get("input_prompt"):
            with open(base + "_input_prompt.txt", "w", encoding="utf-8") as f:
                f.write(config["input_prompt"])
        print(f"Wrote {base}_config.json and prompt .txt files to {args.out_dir}")


if __name__ == "__main__":
    main()
