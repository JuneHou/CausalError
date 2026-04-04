"""
UQ/rerank_spans.py — Exp 3C: local-window span re-ranking (offline post-processing).

Reads an existing UQ output directory (e.g. from Exp 2C) and re-ranks the location
field of each error entry using pointwise LLM scoring over top-K span candidates.
No full trace re-submission: each scoring call uses only a local window (~500-1500 tokens)
comprising span content, parent step, and preceding sibling context.

Rationale:
  Abdallah et al. (EMNLP 2025) show LLM rerankers generalise poorly to novel/unseen
  queries under distribution shift — our execution spans are exactly this case, so we
  use pointwise (not listwise) scoring.  Ou et al. (EMNLP 2025, AgentDiagnose) validate
  that pointwise LLM scoring of individual agent steps achieves r=0.78 with humans.
  LLM4Rerank (ACM Web 2025) shows including execution-flow context (parent / sibling)
  outperforms presenting span content in isolation.

Usage (from trail-benchmark/):
    conda activate /data/wang/junh/envs/causal
    python UQ/rerank_spans.py \\
        --input_dir  UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe \\
        --trace_dir  benchmarking/data/GAIA \\
        --output_dir UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_2C_reranked \\
        --top_k 5
"""

import os
import sys
import re
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT  = Path(__file__).resolve().parent.parent
BENCH_DIR  = REPO_ROOT / "benchmarking"
sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name

MODEL_ID = "Tongyi-Zhiwen/QwenLong-L1-32B"

# One-line definitions for each taxonomy leaf, used in the scoring prompt.
CATEGORY_DEFINITIONS = {
    "Language-only":                  "model fabricates text without tool evidence (hallucination)",
    "Tool-related":                   "model fabricates or misrepresents tool outputs or capabilities",
    "Poor Information Retrieval":     "retrieved information irrelevant or insufficient for the task",
    "Tool Output Misinterpretation":  "model makes incorrect assumptions about or misuses tool output",
    "Incorrect Problem Identification":"model misunderstood the overall or local task goal",
    "Tool Selection Errors":          "model used the wrong tool for the subtask",
    "Formatting Errors":              "errors in code formatting, output structure, or required format",
    "Instruction Non-compliance":     "model failed to perform the required task and did something else",
    "Tool Definition Issues":         "tool was defined incorrectly or inconsistently with its description",
    "Environment Setup Errors":       "permission problems or inability to access required resources/API keys",
    "Rate Limiting":                  "API rate-limit error (e.g. HTTP 429)",
    "Authentication Errors":          "authentication failure (e.g. HTTP 401/403)",
    "Service Errors":                 "server-side error (e.g. HTTP 500)",
    "Resource Not Found":             "requested resource missing (e.g. HTTP 404)",
    "Resource Exhaustion":            "memory overflow or other resource depletion",
    "Timeout Issues":                 "system took too long to respond",
    "Context Handling Failures":      "context window overflow or loss of important prior context",
    "Resource Abuse":                 "tool called excessively due to memory or loop issues",
    "Goal Deviation":                 "system deviated from the assigned task or subtask",
    "Task Orchestration":             "failures in subtask coordination, progress monitoring, or agent delegation",
}

RERANK_PROMPT_TEMPLATE = """\
You are verifying the exact location of an error in an LLM agent execution trace.

Error type   : {category}
Definition   : {definition}
Evidence     : {evidence}

Below are {n_candidates} candidate spans from the trace. For each, rate how likely it is
the span where this specific error occurred.

{candidate_block}
Output a JSON array with one entry per candidate, in the same order:
[{{"span_id": "<id>", "score": <1-5>}}, ...]
Scores: 1=very unlikely  2=unlikely  3=uncertain  4=likely  5=very likely
Output ONLY the JSON array. No markdown, no extra text."""


def _span_content(span: dict, max_chars: int = 300) -> str:
    """Extract meaningful content from a span's attributes or logs."""
    attrs = span.get("span_attributes", {})
    # Prefer output over input (output reflects what the model/tool produced)
    for key in ("output.value", "input.value"):
        val = attrs.get(key)
        if val:
            return str(val)[:max_chars]
    logs = span.get("logs", [])
    if logs:
        body = logs[0].get("body", "")
        if body:
            return str(body)[:max_chars]
    return ""


def _find_span(spans: list, target_id: str) -> Optional[dict]:
    for s in spans:
        if s.get("span_id") == target_id:
            return s
        found = _find_span(s.get("child_spans", []), target_id)
        if found:
            return found
    return None


def _parent_name(spans: list, target_id: str, parent_name: str = "") -> str:
    """Return the span_name of the direct parent of target_id."""
    for s in spans:
        for child in s.get("child_spans", []):
            if child.get("span_id") == target_id:
                return _span_name(s)
        found = _parent_name(s.get("child_spans", []), target_id, _span_name(s))
        if found:
            return found
    return ""


def build_span_candidates(
    trace_data: dict,
    span_index_entries: List[dict],
    evidence: str,
    top_k: int,
) -> List[dict]:
    """
    Select top_k candidate spans by simple token overlap between evidence and span content.
    Falls back to all indexed spans if fewer than top_k have any overlap.
    Returns list of {span_id, span_name, parent_name, content, sibling_preview}.
    """
    root_spans = trace_data.get("spans", [])

    # Score each indexed span by word overlap with evidence
    evidence_words = set(re.findall(r"\w+", evidence.lower()))
    scored = []
    for entry in span_index_entries:
        span = entry["span"]
        sid  = span.get("span_id", "")
        content = _span_content(span)
        span_words = set(re.findall(r"\w+", content.lower()))
        overlap = len(evidence_words & span_words)
        scored.append((overlap, sid, span, content))

    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k] if len(scored) >= top_k else scored

    candidates = []
    seen_ids = set()
    for _, sid, span, content in top:
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        parent = _parent_name(root_spans, sid)

        # Preceding sibling: find the child that comes just before this span
        # under its parent (by position in child_spans list)
        sibling_preview = ""
        parent_span = _find_span(root_spans, span.get("parent_span_id", ""))
        if parent_span:
            siblings = parent_span.get("child_spans", [])
            for i, sib in enumerate(siblings):
                if sib.get("span_id") == sid and i > 0:
                    sibling_preview = _span_content(siblings[i - 1], max_chars=100)
                    break

        candidates.append({
            "span_id":        sid,
            "span_name":      _span_name(span),
            "parent_name":    parent,
            "content":        content,
            "sibling_preview": sibling_preview,
        })

    return candidates


def build_rerank_prompt(
    category: str,
    evidence: str,
    candidates: List[dict],
) -> str:
    definition = CATEGORY_DEFINITIONS.get(category, "")
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"[{i}] span_id \"{c['span_id']}\" ({c['span_name']})\n"
            f"    Execution position: {c['parent_name']} → {c['span_name']}\n"
            + (f"    Preceding context: {c['sibling_preview']}\n" if c["sibling_preview"] else "")
            + f"    Content: {c['content']}"
        )
    candidate_block = "\n\n".join(lines)
    return RERANK_PROMPT_TEMPLATE.format(
        category=category,
        definition=definition,
        evidence=evidence,
        n_candidates=len(candidates),
        candidate_block=candidate_block,
    )


def parse_scores(text: str, candidates: List[dict]) -> Dict[str, int]:
    """Parse the JSON array of {span_id, score} from the model output."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Extract JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return {}
    try:
        items = json.loads(m.group())
        if isinstance(items, list):
            return {item["span_id"]: int(item.get("score", 1)) for item in items if "span_id" in item}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return {}


def rerank_errors(
    llm: LLM,
    tokenizer: AutoTokenizer,
    errors: List[dict],
    trace_data: dict,
    span_index_entries: List[dict],
    top_k: int,
    min_score: int,
    max_new_tokens: int,
    valid_span_ids: set,
) -> Tuple[List[dict], dict]:
    """
    For each error entry, run pointwise span re-ranking and update its location.
    Returns (updated_errors, rerank_meta).
    """
    if not errors or not span_index_entries:
        return errors, {}

    # Build prompts for all errors in one batch
    per_error_candidates: List[List[dict]] = []
    prompts: List[str] = []

    for err in errors:
        candidates = build_span_candidates(
            trace_data=trace_data,
            span_index_entries=span_index_entries,
            evidence=err.get("evidence", ""),
            top_k=top_k,
        )
        per_error_candidates.append(candidates)
        if candidates:
            prompt_text = build_rerank_prompt(
                category=err.get("category", ""),
                evidence=err.get("evidence", ""),
                candidates=candidates,
            )
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(formatted)
        else:
            prompts.append(None)

    # Filter out None prompts for batch generation
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]

    if not valid_prompts:
        return errors, {}

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(valid_prompts, sp)

    rerank_meta = {}
    output_iter = iter(outputs)
    updated_errors = list(errors)

    for i, err in enumerate(updated_errors):
        candidates = per_error_candidates[i]
        if not candidates or i not in valid_indices:
            continue

        out = next(output_iter)
        raw = out.outputs[0].text
        scores = parse_scores(raw, candidates)

        if not scores:
            rerank_meta[i] = {"parse_failed": True, "original": err.get("location")}
            continue

        # Pick best-scoring span that passes the span_id gate
        best_sid   = err.get("location", "")
        best_score = 0
        for cand in candidates:
            sid   = cand["span_id"]
            score = scores.get(sid, 0)
            if score > best_score and sid in valid_span_ids:
                best_score = score
                best_sid   = sid

        original = err.get("location", "")
        if best_score >= min_score and best_sid != original:
            updated_errors[i] = dict(err)
            updated_errors[i]["location"] = best_sid
            rerank_meta[i] = {
                "original": original,
                "reranked": best_sid,
                "score":    best_score,
                "changed":  True,
            }
            print(f"    [{err.get('category','')}] {original} → {best_sid} (score={best_score})")
        else:
            rerank_meta[i] = {
                "original": original,
                "reranked": best_sid,
                "score":    best_score,
                "changed":  False,
            }

    return updated_errors, rerank_meta


def extract_all_span_ids(trace_data: dict) -> set:
    """Return set of all span_ids present in the trace."""
    result = set()
    def walk(spans):
        for s in spans:
            sid = s.get("span_id")
            if sid:
                result.add(sid)
            walk(s.get("child_spans", []))
    walk(trace_data.get("spans", []))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 3C: offline local-window span re-ranking for UQ outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",   required=True,
                        help="Directory of existing UQ output JSONs (e.g. Exp 2C outputs)")
    parser.add_argument("--trace_dir",   required=True,
                        help="Directory of original trace JSONs (benchmarking/data/GAIA)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write re-ranked output JSONs")
    parser.add_argument("--model",       type=str,   default=MODEL_ID)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--enforce_eager", action="store_true", default=True)
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Scoring calls are short; 512 is ample for a JSON array")
    parser.add_argument("--top_k",       type=int,   default=5,
                        help="Number of span candidates to score per error entry")
    parser.add_argument("--min_score",   type=int,   default=3,
                        help="Minimum score (1-5) to accept re-ranked location; "
                             "if all candidates score below this, retain original")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    trace_dir  = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_model_len

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
    print("Model loaded.\n")

    uq_files = sorted(glob.glob(str(input_dir / "*.json")))
    print(f"Re-ranking {len(uq_files)} files from {input_dir}")
    print(f"Output → {output_dir}\n")

    changed_total = 0
    for uq_fp in tqdm(uq_files):
        fname     = Path(uq_fp).name
        out_file  = output_dir / fname
        if out_file.exists():
            continue

        trace_fp = trace_dir / fname
        if not trace_fp.exists():
            print(f"  [skip] trace not found: {trace_fp}")
            continue

        with open(uq_fp)    as f: uq_data    = json.load(f)
        with open(trace_fp) as f: trace_data = json.load(f)

        errors = uq_data.get("errors", [])
        if not errors:
            with open(out_file, "w") as f:
                json.dump(uq_data, f)
            continue

        # Build span index for this trace (same logic as run_uq_eval.py)
        trace_str = json.dumps(trace_data)
        parsed    = parse_trace_to_step_level(trace_data)
        span_index_entries = parsed.get("step_spans", [])

        valid_span_ids = extract_all_span_ids(trace_data)

        print(f"\n{fname}: {len(errors)} errors, {len(span_index_entries)} indexed spans")
        updated_errors, rerank_meta = rerank_errors(
            llm=llm,
            tokenizer=tokenizer,
            errors=errors,
            trace_data=trace_data,
            span_index_entries=span_index_entries,
            top_k=args.top_k,
            min_score=args.min_score,
            max_new_tokens=args.max_new_tokens,
            valid_span_ids=valid_span_ids,
        )

        n_changed = sum(1 for m in rerank_meta.values() if m.get("changed"))
        changed_total += n_changed

        result = dict(uq_data)
        result["errors"] = updated_errors
        result.setdefault("_uq_meta", {})["rerank_meta"] = rerank_meta
        with open(out_file, "w") as f:
            json.dump(result, f)

    print(f"\nDone. Location changed in {changed_total} error entries total.")


if __name__ == "__main__":
    main()
