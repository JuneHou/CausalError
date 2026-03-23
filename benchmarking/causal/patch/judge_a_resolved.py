#!/usr/bin/env python3
"""
Step 6: Judge 1 — A-resolved (treatment validity).

Verifies that the do(A=0) patch actually eliminated the source error A
in the patched span. Cases where resolved=False are excluded from
Δ(A→B) estimation and tracked in patch_failure_rate by category.

One judgment per A-instance (not per edge). The verdict is shared by all
EdgePairs that reference the same error_id.

Judge 1 compares the patch_payload against the original local_snippet and
error evidence, and (when available) also examines the rerun suffix around t_A.

Input:  rerun_results.jsonl, patch_results.jsonl, a_instances.jsonl
Output: a_resolved.jsonl  (one record per A-instance)
Each line: {trace_id, error_id, a_location, resolved, confidence,
            evidence_excerpt, rerun_status}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from patch_generator_llm import _call_llm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_A_SYSTEM = """You are verifying whether a source error of type A has been eliminated by a patch.

PRIMARY TASK: Compare ORIGINAL_SPAN with PATCHED_SPAN and determine whether the specific labeled
error A (defined by SOURCE_ERROR_TYPE, ERROR_DESCRIPTION, ERROR_EVIDENCE) is no longer present
in the patched version.

IMPORTANT RULES:
1. Base your verdict on the ORIGINAL_SPAN vs PATCHED_SPAN comparison first.
2. RERUN_SUFFIX is supplementary context only. Do NOT use rerun failures to override a clear
   patch-level fix. If the patched span itself resolves error A, mark resolved=true even if
   the rerun encountered downstream difficulties or repeated errors in OTHER spans.
3. Focus solely on error A. Do not penalize for downstream errors (type B) that the patch
   was not designed to fix.

Return ONLY JSON:
{
  "resolved": true,
  "confidence": 0.0,
  "evidence_excerpt": "string"
}"""


# For replace_span_output: check the patched content for error absence
JUDGE_A_USER_OUTPUT_TEMPLATE = """\
SOURCE_ERROR_TYPE: {A}
ERROR_DESCRIPTION: {ERROR_DESCRIPTION}
ERROR_EVIDENCE: {ERROR_EVIDENCE}

ORIGINAL_SPAN (content that was replaced):
<<<
{ORIGINAL_SNIPPET}
>>>

PATCHED_SPAN (replacement content):
<<<
{PATCH_PAYLOAD}
>>>

{RERUN_CONTEXT_BLOCK}

Task: Has error A been eliminated in the PATCHED_SPAN compared to ORIGINAL_SPAN?
Focus on whether the specific error criterion is met in the patched content itself.
Respond with JSON only: {{"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}}\
"""


# For replace_span_input: the patch modifies the INPUT (message history) to the LLM, not its output.
# The error was observed in the LLM's output behavior, not in the input text.
# Check whether the patched input removed or corrected the error-causing context.
JUDGE_A_USER_INPUT_TEMPLATE = """\
SOURCE_ERROR_TYPE: {A}
ERROR_DESCRIPTION: {ERROR_DESCRIPTION}
ERROR_EVIDENCE: {ERROR_EVIDENCE}

NOTE: This patch modifies the INPUT (message history context) seen by the LLM, not the LLM's
output. The error A was caused by the context in the original input. The intervention removes
or corrects that error-causing context so the LLM has better information going forward.

ORIGINAL_INPUT (message history before patching):
<<<
{ORIGINAL_SNIPPET}
>>>

PATCHED_INPUT (message history after patching):
<<<
{PATCH_PAYLOAD}
>>>

{RERUN_CONTEXT_BLOCK}

Task: Has the error-causing context for error A been removed or corrected in the PATCHED_INPUT?
Do NOT require the patched input to contain the correct output behavior — it is an input, not
an output. Check whether the specific pattern described in ERROR_EVIDENCE/ERROR_DESCRIPTION
is absent or corrected in the patched context compared to the original.
Respond with JSON only: {{"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}}\
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class AResolvedVerdict:
    trace_id: str
    error_id: str
    a_location: str
    resolved: bool
    confidence: float
    evidence_excerpt: str
    rerun_status: str   # live_rerun_success | rerun_missing_suffix
    model_used: str


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_a_resolved(
    rerun_result: dict,
    patch_result: dict,
    a_instance_record: dict,
    model: str = "openai/gpt-4o",
) -> AResolvedVerdict:
    """
    Run Judge 1 for one A-instance. Returns AResolvedVerdict.

    rerun_result      : from rerun_harness (has rerun_suffix_spans, rerun_status)
    patch_result      : from patch_generator (has patch_payload, slot_values)
    a_instance_record : AInstanceRecord dict (has a_instance, local_snippet, patch_side)
    """
    a_instance = a_instance_record.get("a_instance", {})
    a_cat = a_instance.get("category", "")
    patch_side = a_instance_record.get("patch_side", "replace_span_output")

    # Rerun suffix context (first 3 spans after t_A, trimmed).
    # Included as supplementary only — primary judgment from ORIGINAL vs PATCHED comparison.
    rerun_status = rerun_result.get("rerun_status", "rerun_missing_suffix")
    rerun_spans = rerun_result.get("rerun_suffix_spans") or []
    if rerun_status == "live_rerun_success" and rerun_spans:
        rerun_text = "\n---\n".join(str(s)[:800] for s in rerun_spans[:3])
        rerun_block = (
            "RERUN_SUFFIX (supplementary context — first 3 spans after t_A in counterfactual run):\n"
            "<<<\n" + rerun_text + "\n>>>"
        )
    else:
        rerun_block = f"RERUN_SUFFIX: ({rerun_status} — counterfactual trace not available)"

    # Select template based on patch_side (Fix C)
    template = (
        JUDGE_A_USER_INPUT_TEMPLATE
        if patch_side == "replace_span_input"
        else JUDGE_A_USER_OUTPUT_TEMPLATE
    )

    user_msg = template.format(
        A=a_cat,
        ERROR_DESCRIPTION=(a_instance.get("description") or "")[:800],
        ERROR_EVIDENCE=(a_instance.get("evidence") or "")[:800],
        ORIGINAL_SNIPPET=(a_instance_record.get("local_snippet") or "")[:2000],
        PATCH_PAYLOAD=(patch_result.get("patch_payload") or "")[:2000],
        RERUN_CONTEXT_BLOCK=rerun_block,
    )

    resolved = False
    confidence = 0.0
    evidence_excerpt = ""

    try:
        raw = _call_llm(JUDGE_A_SYSTEM, user_msg, model=model, max_tokens=256)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()
        parsed = json.loads(raw)
        resolved = bool(parsed.get("resolved", False))
        confidence = float(parsed.get("confidence", 0.0))
        evidence_excerpt = str(parsed.get("evidence_excerpt", ""))[:400]
    except Exception as e:
        evidence_excerpt = f"judge_error: {e}"

    return AResolvedVerdict(
        trace_id=rerun_result["trace_id"],
        error_id=patch_result.get("error_id", ""),
        a_location=rerun_result.get("a_location", ""),
        resolved=resolved,
        confidence=confidence,
        evidence_excerpt=evidence_excerpt,
        rerun_status=rerun_status,
        model_used=model,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge 1: verify do(A=0) treatment validity."
    )
    parser.add_argument("--rerun_results",
                        default="outputs/interventions/rerun_results.jsonl")
    parser.add_argument("--patch_results",
                        default="outputs/interventions/patch_results.jsonl")
    parser.add_argument("--cases",
                        default="outputs/interventions/a_instances.jsonl")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o")
    args = parser.parse_args()

    def _load_jsonl(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    rerun_results = _load_jsonl(args.rerun_results)
    patch_results = _load_jsonl(args.patch_results)
    a_instances = _load_jsonl(args.cases)

    # Index by (trace_id, error_id)
    patch_idx = {(p["trace_id"], p.get("error_id", "")): p for p in patch_results}
    instance_idx = {(a["trace_id"], a["error_id"]): a for a in a_instances}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "a_resolved.jsonl")

    n_resolved = n_unresolved = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rr in rerun_results:
            if not rr.get("rerun_success"):
                continue
            key = (rr["trace_id"], rr.get("error_id", ""))
            pr = patch_idx.get(key)
            ai = instance_idx.get(key)
            if not pr or not ai:
                continue

            verdict = judge_a_resolved(rr, pr, ai, model=args.model)
            f.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
            status = "RESOLVED" if verdict.resolved else "UNRESOLVED"
            print(f"  [{status}] {verdict.trace_id[:8]} err={verdict.error_id[-20:]} "
                  f"conf={verdict.confidence:.2f}")
            if verdict.resolved:
                n_resolved += 1
            else:
                n_unresolved += 1

    print(f"\nWrote {out_path}. Resolved={n_resolved} Unresolved={n_unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
