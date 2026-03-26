#!/usr/bin/env python3
"""
Step 7: Judge 2 — B-effect label (outcome evaluation).

Uses the recommended judge prompt to determine what happened to downstream
error type B in the rerun suffix vs the baseline.

Fan-out: one LLM call per EdgePair where Judge 1 confirmed resolved=True.
All EdgePairs sharing the same error_id reuse the same RerunResult.

B taxonomy definitions (from TRAIL appendix) are embedded here as the
B_DEFINITIONS dict, used as TARGET_ERROR_DEFINITION in the judge prompt.

Input:  rerun_results.jsonl  (per A-instance)
        a_resolved.jsonl     (per A-instance)
        edge_pairs.jsonl     (per A-instance × B-type)
Output: b_effect.jsonl       (per A-instance × B-type, resolved only)
Each line: {trace_id, error_id, edge, effect_label, target_present_after,
            original_onset_ref, rerun_onset_ref, confidence, evidence,
            b_present_baseline, rerun_status}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarking"))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from patch_generator_llm import _call_llm


# ---------------------------------------------------------------------------
# B taxonomy definitions (TRAIL appendix, for B types in the graph)
# ---------------------------------------------------------------------------

B_DEFINITIONS = {
    "Context Handling Failures": (
        "Window overflow / state tracking / forgetting important context. "
        "The agent lost track of key prior information, failed to carry over "
        "required state, or exceeded its context window leading to incorrect behavior."
    ),
    "Incorrect Problem Identification": (
        "Misunderstood the overall task or the local task. "
        "The agent addressed the wrong problem, misread the question, or targeted "
        "a different entity/goal than what was requested."
    ),
    "Resource Abuse": (
        "Excessive tool calling due to memory issues / repeated unnecessary use of resources. "
        "The agent issued redundant or repeated calls to the same tool/resource "
        "without making progress, often looping without a stopping condition."
    ),
    "Tool Output Misinterpretation": (
        "Made assumptions about tool output or used it in the wrong context. "
        "The agent misread or hallucinated the content of a tool's return value, "
        "or applied tool output to a different context than intended."
    ),
    "Authentication Errors": (
        "Auth / permission failures like 401/403. "
        "The agent encountered or triggered authentication or authorization errors "
        "when accessing resources, tools, or APIs."
    ),
    "Tool-related": (
        "Fabricated tool outputs or tool capabilities. "
        "The agent invented results from a tool call that did not actually occur, "
        "or assumed a tool had capabilities it does not have."
    ),
    "Goal Deviation": (
        "Deviated from the task or current subtask. "
        "The agent stopped pursuing the assigned goal and began working on a "
        "different objective, or gave up on the task prematurely."
    ),
    "Language-only": (
        "Ungrounded language-only hallucination. "
        "The agent made factual claims not supported by any tool output, "
        "retrieved document, or given context — pure language-model hallucination."
    ),
    "Task Orchestration": (
        "Subtask coordination and progress monitoring failures. "
        "The agent failed to sequence subtasks correctly, skipped required "
        "intermediate steps, or lost track of which subtasks had been completed."
    ),
}


# ---------------------------------------------------------------------------
# Judge prompt (from recommended prompts — used verbatim)
# ---------------------------------------------------------------------------

JUDGE_B_SYSTEM = """You are evaluating the downstream effect of a do(A=0) intervention on error type B.

The source error A was locally patched at one labeled span.
The rerun trace suffix shows the counterfactual execution after the intervention.

You must judge ONLY the downstream error type B.

Effect labels:
- disappeared    : B was present in baseline, absent in rerun (intervention removed B)
- delayed        : B was present in baseline, appears later in rerun
- unchanged      : B was present in baseline, appears at similar position in rerun
- earlier        : B was present in baseline, appears earlier in rerun
- weakened       : B was present in baseline, present in rerun but less severe
- strengthened   : B was present in baseline, present in rerun and more severe
- emerged        : B was ABSENT in baseline, but NOW PRESENT in rerun (intervention introduced B)
- not_observable : B was absent in baseline and absent in rerun; effect cannot be assessed

Return ONLY JSON."""

JUDGE_B_USER_TEMPLATE = """\
SOURCE_ERROR_TYPE: {A}
TARGET_ERROR_TYPE: {B}

TARGET_ERROR_DEFINITION:
{B_TAXONOMY_DEFINITION_OR_INSTANCE_DESCRIPTION}

ORIGINAL_TRACE_SUFFIX:
<<<
{ORIGINAL_SUFFIX}
>>>

ORIGINAL_ONSET_REF:
{ORIGINAL_B_ONSET}

RERUN_TRACE_SUFFIX_AFTER_DO_A_0:
<<<
{RERUN_SUFFIX}
>>>

Task:
Judge how B changed after the do(A=0) intervention.

Required output schema:
{{
  "source_error_type": "string",
  "target_error_type": "string",
  "effect_label": "disappeared|delayed|unchanged|earlier|weakened|strengthened|emerged|not_observable",
  "target_present_after": true,
  "original_onset_ref": "string|null",
  "rerun_onset_ref": "string|null",
  "confidence": "high|medium|low",
  "evidence": "string"
}}\
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BEffectVerdict:
    trace_id: str
    error_id: str
    edge: dict
    effect_label: str
    target_present_after: bool
    original_onset_ref: Optional[str]
    rerun_onset_ref: Optional[str]
    confidence: str
    evidence: str
    b_present_baseline: bool
    rerun_status: str   # live_rerun_success | rerun_missing_suffix
    model_used: str


VALID_EFFECT_LABELS = {
    "disappeared", "delayed", "unchanged", "earlier",
    "weakened", "strengthened", "emerged", "not_observable",
}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_b_effect(
    rerun_result: dict,
    edge_pair: dict,
    model: str = "openai/gpt-4o",
) -> BEffectVerdict:
    """
    Run Judge 2 for one EdgePair. Returns BEffectVerdict.

    rerun_result : from rerun_harness (shared across all EdgePairs with same error_id)
    edge_pair    : EdgePair dict (has edge, b_present_baseline, b_onset_baseline)
    """
    a_cat = edge_pair["edge"]["a"]
    b_cat = edge_pair["edge"]["b"]
    b_definition = B_DEFINITIONS.get(b_cat, f"TRAIL category: {b_cat}")

    # Original suffix from baseline trace
    original_suffix_spans = rerun_result.get("original_suffix_spans") or []
    original_suffix = "\n---\n".join(str(s)[:600] for s in original_suffix_spans[:8])
    if not original_suffix:
        original_suffix = "(no suffix spans extracted)"

    # Baseline onset reference
    b_onset_baseline = edge_pair.get("b_onset_baseline", -1)
    b_present_baseline = edge_pair.get("b_present_baseline", False)
    if b_present_baseline and b_onset_baseline >= 0:
        original_onset_ref = f"annotation index {b_onset_baseline}"
    else:
        original_onset_ref = "not present in baseline"

    # Rerun suffix — real counterfactual LLM outputs from t_A onward
    rerun_status = rerun_result.get("rerun_status", "rerun_missing_suffix")
    rerun_suffix_spans = rerun_result.get("rerun_suffix_spans") or []

    if rerun_status == "live_rerun_success" and rerun_suffix_spans:
        rerun_suffix = "\n---\n".join(str(s)[:600] for s in rerun_suffix_spans[:8])
    else:
        rerun_suffix = (
            "(rerun_missing_suffix: LLM rerun did not produce a counterfactual trace. "
            "Effect on B cannot be assessed from trace evidence.)"
        )

    user_msg = JUDGE_B_USER_TEMPLATE.format(
        A=a_cat,
        B=b_cat,
        B_TAXONOMY_DEFINITION_OR_INSTANCE_DESCRIPTION=b_definition,
        ORIGINAL_SUFFIX=original_suffix[:3000],
        ORIGINAL_B_ONSET=original_onset_ref,
        RERUN_SUFFIX=rerun_suffix[:3000],
    )

    effect_label = "not_observable"
    target_present_after = False
    rerun_onset_ref = None
    confidence = "low"
    evidence = ""

    try:
        raw = _call_llm(JUDGE_B_SYSTEM, user_msg, model=model, max_tokens=512)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()
        parsed = json.loads(raw)
        label = str(parsed.get("effect_label", "not_observable")).lower().strip()
        if label not in VALID_EFFECT_LABELS:
            label = "not_observable"
        effect_label = label
        target_present_after = bool(parsed.get("target_present_after", False))
        rerun_onset_ref = parsed.get("rerun_onset_ref") or None
        confidence = str(parsed.get("confidence", "low"))
        evidence = str(parsed.get("evidence", ""))[:600]
    except Exception as e:
        evidence = f"judge_error: {e}"

    return BEffectVerdict(
        trace_id=rerun_result["trace_id"],
        error_id=rerun_result.get("error_id", ""),
        edge=edge_pair.get("edge", {}),
        effect_label=effect_label,
        target_present_after=target_present_after,
        original_onset_ref=original_onset_ref,
        rerun_onset_ref=rerun_onset_ref,
        confidence=confidence,
        evidence=evidence,
        b_present_baseline=b_present_baseline,
        rerun_status=rerun_status,
        model_used=model,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge 2: evaluate downstream B-effect after do(A=0)."
    )
    parser.add_argument("--rerun_results",
                        default="outputs/interventions/rerun_results.jsonl")
    parser.add_argument("--a_resolved",
                        default="outputs/interventions/a_resolved.jsonl")
    parser.add_argument("--edge_pairs",
                        default="outputs/interventions/edge_pairs.jsonl")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o")
    args = parser.parse_args()

    def _load_jsonl(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    rerun_results = _load_jsonl(args.rerun_results)
    a_resolved = _load_jsonl(args.a_resolved)
    edge_pairs = _load_jsonl(args.edge_pairs)

    # Resolved set keyed by (trace_id, error_id)
    resolved_keys = {
        (v["trace_id"], v.get("error_id", ""))
        for v in a_resolved if v.get("resolved")
    }

    # Rerun index keyed by (trace_id, error_id)
    rerun_idx = {
        (rr["trace_id"], rr.get("error_id", "")): rr
        for rr in rerun_results if rr.get("rerun_success")
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "b_effect.jsonl")

    from collections import Counter
    label_counts: Counter = Counter()
    n_skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in edge_pairs:
            key = (ep["trace_id"], ep.get("error_id", ""))
            if key not in resolved_keys:
                n_skipped += 1
                continue
            rr = rerun_idx.get(key)
            if not rr:
                n_skipped += 1
                continue

            verdict = judge_b_effect(rr, ep, model=args.model)
            f.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
            label_counts[verdict.effect_label] += 1
            print(f"  {verdict.trace_id[:8]} {verdict.edge} "
                  f"→ {verdict.effect_label} [{verdict.confidence}]")

    print(f"\nWrote {out_path}. Skipped (unresolved A)={n_skipped}")
    print("Effect label distribution:", dict(label_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
