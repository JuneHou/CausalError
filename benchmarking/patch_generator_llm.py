#!/usr/bin/env python3
"""
LLM-based patch creation pipeline (per-error, no operator-family mapping).

Flow:
  1. Build patch input: error annotation (type, evidence, description) + problematic span + minimal local context.
  2. Mechanism diagnosis: ask LLM to identify proximal mechanism and repair target (input/context | local_decision_policy | output_surface).
  3. Single-error patch generation: ask LLM to produce the smallest repair that fixes only this error in this span.
  4. Output: patched span text + rerun request (patch_log.jsonl, rerun_requests.jsonl).
  Patches are applied and the trace is rerun by separate code after this step.

Uses litellm (same as run_eval.py). Set OPENAI_API_KEY or use model that fits your litellm config.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional: litellm for LLM calls
try:
    from litellm import completion
    from litellm import RateLimitError, ContextWindowExceededError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

from trail_io import TraceObj, load_trail_trace, get_expanded_snippet, get_span_io

# Default token limits for prompts (avoid overflow)
MAX_PROBLEMATIC_SPAN_CHARS = 6000
MAX_CONTEXT_CHARS = 3000
MAX_EVIDENCE_CHARS = 1500
MAX_DESCRIPTION_CHARS = 1500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PatchInput:
    """Structured input to the patch generator (error + span + context + exact I/O)."""
    trace_id: str
    error_id: str
    location: str  # span_id
    error_type: str
    evidence: str
    description: str
    impact: str
    problematic_span: str   # text of the span where error occurs
    local_context: str     # minimal context (e.g. previous/next span or parent summary)
    tools_available: List[str]
    # Exact input/output for this span (from TraceObj.input_by_location / output_by_location)
    exact_input: dict   # { "kind", "messages", "input_value", "input_value_raw" }
    exact_output: dict  # { "output_text", "output_value_raw" }


@dataclass
class MechanismDiagnosis:
    """Output of mechanism diagnosis step."""
    target: str  # "input_context" | "local_decision_policy" | "output_surface"
    mechanism: str  # one sentence describing proximal local failure mechanism
    reasoning: str


@dataclass
class PatchResult:
    """Result of one LLM-based patch generation."""
    trace_id: str
    error_id: str
    location: str
    error_type: str
    diagnosis: MechanismDiagnosis
    original_span_text: str
    patched_span_text: str
    success: bool
    error_message: Optional[str] = None
    # For rerun harness: what to substitute at this span
    rerun_request: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 1. Build patch input
# ---------------------------------------------------------------------------


def build_patch_input(
    trace_obj: TraceObj,
    error_instance: dict,
    context_window: int = 1,
) -> PatchInput:
    """
    Build the structured input for the patch generator: error annotation plus
    problematic span and minimal local context.
    """
    location = error_instance.get("annotated_span_id") or error_instance.get("location") or ""
    problematic = get_expanded_snippet(trace_obj, location, window=0)
    context = get_expanded_snippet(trace_obj, location, window=context_window)
    # If context is same as problematic (window=0), local_context can be empty or sibling summary
    if context.strip() == problematic.strip():
        local_context = ""
    else:
        # Exclude the problematic part from "context" to get only the neighbourhood
        local_context = context.replace(problematic, "").strip()[:MAX_CONTEXT_CHARS]

    span_io = get_span_io(trace_obj, location)
    exact_input = span_io.get("input") or {"kind": "", "messages": [], "input_value": "", "input_value_raw": ""}
    exact_output = span_io.get("output") or {"output_text": "", "output_value_raw": ""}

    return PatchInput(
        trace_id=trace_obj.trace_id,
        error_id=error_instance.get("error_id", ""),
        location=location,
        error_type=error_instance.get("error_type") or error_instance.get("category") or "",
        evidence=(error_instance.get("evidence") or "")[:MAX_EVIDENCE_CHARS],
        description=(error_instance.get("description") or "")[:MAX_DESCRIPTION_CHARS],
        impact=error_instance.get("impact", ""),
        problematic_span=problematic[:MAX_PROBLEMATIC_SPAN_CHARS],
        local_context=local_context,
        tools_available=list(trace_obj.tools_available or [])[:50],
        exact_input=exact_input,
        exact_output=exact_output,
    )


# ---------------------------------------------------------------------------
# 2. Mechanism diagnosis (LLM)
# ---------------------------------------------------------------------------


DIAGNOSIS_SYSTEM = """You are an expert at debugging agent traces. Your task is to identify the proximal mechanism of one labeled error and decide where the minimal repair should target.

Important constraints:
- Analyze ONLY the specified error in the labeled span.
- Do not try to fix or reason about other possible errors.
- Choose the smallest intervention that would plausibly prevent this error.

Repair targets (choose exactly one):
- input_context: The error is caused by missing, incorrect, or insufficient context available to the span (e.g. missing requirement, omitted tool result detail, missing carryover state). Repair by adding or correcting the local context before rerun.
- local_decision_policy: The error is caused by the agent's local decision at this step (e.g. wrong tool choice, skipped verification, describing instead of acting, missing guard/check). Repair by injecting a short local rule/checklist immediately before rerunning the span.
- output_surface: The error is purely a surface or formatting issue in the emitted span text (e.g. missing tag, malformed JSON, wrong wrapper, exact format mismatch). Repair by editing the output text only.

Respond in this exact format (no extra text):
MECHANISM: <one sentence describing the proximal local failure mechanism>
TARGET: <input_context|local_decision_policy|output_surface>
REASONING: <1-3 sentences explaining why this is the minimal repair target>"""


def _call_llm(
    system: str,
    user_content: str,
    model: str = "openai/gpt-4o",
    max_tokens: int = 128000,
    temperature: float = 0.0,
) -> str:
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm is not installed. pip install litellm")
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "o1" in model or "o3" in model or "o4" in model:
        params["reasoning_effort"] = "medium"
    try:
        response = completion(**params)
        return (response.choices[0].message.content or "").strip()
    except RateLimitError:
        import time
        time.sleep(30)
        return (completion(**params).choices[0].message.content or "").strip()


def diagnose_mechanism(
    patch_input: PatchInput,
    model: str = "openai/gpt-4o",
) -> MechanismDiagnosis:
    """
    Ask the LLM to identify the proximal mechanism and the repair target
    (input_context | local_decision_policy | output_surface).
    """
    # Format exact input for the prompt (messages or input_value)
    exact_in = patch_input.exact_input or {}
    input_block = ""
    if exact_in.get("messages"):
        input_block = "Exact input messages into this span:\n" + "\n".join(
            f"  [{m.get('role', '')}]: {str(m.get('content', ''))[:1500]}"
            for m in exact_in["messages"]
        )
    elif exact_in.get("input_value"):
        input_block = f"Exact input to this span:\n  {exact_in['input_value'][:2000]}"
    else:
        input_block = "Exact input: (not recovered)"

    exact_out = patch_input.exact_output or {}
    output_block = exact_out.get("output_text") or exact_out.get("output_value_raw") or "(not recovered)"
    if len(output_block) > 2500:
        output_block = output_block[:2500] + "\n..."

    user = f"""Error type: {patch_input.error_type}
Evidence: {patch_input.evidence}
Description: {patch_input.description}

{input_block}

Exact output of this span:
---
{output_block}
---

Problematic span (same as above, for reference):
---
{patch_input.problematic_span[:3000]}
---

Local context (neighbouring content, if any):
---
{patch_input.local_context or "(none)"}
---

Which single repair target should a minimal fix use? Reply with MECHANISM:, TARGET:, and REASONING:."""

    raw = _call_llm(DIAGNOSIS_SYSTEM, user, model=model, max_tokens=512)
    target = "output_surface"  # default
    mechanism = ""
    reasoning = raw
    m_target = re.search(r"TARGET\s*:\s*(\w+)", raw, re.IGNORECASE)
    if m_target:
        t = m_target.group(1).lower().replace("-", "_")
        if t in ("input_context", "local_decision_policy", "output_surface"):
            target = t
    m_mech = re.search(r"MECHANISM\s*:\s*(.+?)(?=\nTARGET:|\nREASONING:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m_mech:
        mechanism = m_mech.group(1).strip()
    m_reason = re.search(r"REASONING\s*:\s*(.+?)(?=\n[A-Z]|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m_reason:
        reasoning = m_reason.group(1).strip()
    return MechanismDiagnosis(target=target, mechanism=mechanism, reasoning=reasoning)


# ---------------------------------------------------------------------------
# 3. Single-error patch generation (LLM) — target-specific prompts
# ---------------------------------------------------------------------------


PATCH_GENERATION_SYSTEM_OUTPUT = """You are generating a directly applicable replacement patch for one labeled error in an agent trace.

Task:
- Fix ONLY the specified error.
- Produce the smallest possible change.
- Preserve all unrelated content exactly.
- The patch must be directly usable as a literal replacement for the labeled span output.
- Do not put explanations inside the patch text.

Output format:
Respond in exact JSON with no extra text:
{
  "reason": "<one short sentence explaining why this replacement fixes the labeled error>",
  "patch": "<the full replacement text for the labeled span output>"
}"""

PATCH_GENERATION_SYSTEM_INPUT = """You are generating a directly applicable replacement patch for one labeled error in an agent trace.

Task:
- Fix ONLY the specified error by replacing the content of the local input-context patch slot used before rerunning the labeled span.
- The patch must be directly usable as a literal replacement for that input-context patch slot.
- Do not rewrite the original span output.
- Do not put explanations inside the patch text.

Output format:
Respond in exact JSON with no extra text:
{
  "reason": "<one short sentence explaining why this replacement fixes the labeled error>",
  "patch": "<the exact replacement text for the input-context patch slot>"
}"""

PATCH_GENERATION_SYSTEM_POLICY = """You are generating a directly applicable replacement patch for one labeled error in an agent trace.

Task:
- Fix ONLY the specified error by replacing the content of the local decision-policy patch slot used before rerunning the labeled span.
- The patch must be the literal rule/checklist text itself, written as direct imperative instructions.
- The patch must be directly usable as a literal replacement for that policy patch slot.
- Do not put explanations inside the patch text.

Output format:
Respond in exact JSON with no extra text:
{
  "reason": "<one short sentence explaining why this replacement fixes the labeled error>",
  "patch": "<the exact replacement text for the local decision-policy patch slot>"
}"""


def generate_single_error_patch(
    patch_input: PatchInput,
    diagnosis: MechanismDiagnosis,
    model: str = "openai/gpt-4o",
) -> Tuple[str, bool, str]:
    """
    Ask the LLM to generate the minimal repair. Branches on diagnosis.target.
    Response must be JSON: {"reason": "...", "patch": "..."}. Returns (patch_payload, success, reason).
    """
    if diagnosis.target == "output_surface":
        system = PATCH_GENERATION_SYSTEM_OUTPUT
        user = f"""Error type: {patch_input.error_type}
Evidence: {patch_input.evidence}
Description: {patch_input.description}

Target: {diagnosis.target}
Mechanism: {diagnosis.mechanism}

Original span (reference only):
---
{patch_input.problematic_span}
---

Generate a replacement for the labeled span output. Respond with JSON only. The "patch" field must contain the full replacement text for the span."""
    elif diagnosis.target == "input_context":
        system = PATCH_GENERATION_SYSTEM_INPUT
        user = f"""Error type: {patch_input.error_type}
Evidence: {patch_input.evidence}
Description: {patch_input.description}

Target: {diagnosis.target}
Mechanism: {diagnosis.mechanism}

Labeled span (reference only; do not rewrite it):
---
{patch_input.problematic_span}
---

Existing local context:
---
{patch_input.local_context or "(none)"}
---

Generate a replacement for the input-context patch slot used before rerunning this span. Respond with JSON only. The "patch" field must contain only the literal context snippet."""
    else:
        # local_decision_policy
        system = PATCH_GENERATION_SYSTEM_POLICY
        user = f"""Error type: {patch_input.error_type}
Evidence: {patch_input.evidence}
Description: {patch_input.description}

Target: {diagnosis.target}
Mechanism: {diagnosis.mechanism}

Labeled span (reference only):
---
{patch_input.problematic_span}
---

Generate a replacement for the local decision-policy patch slot used immediately before rerunning this span. Respond with JSON only. The "patch" field must contain only the literal rule/checklist text."""

    raw = _call_llm(
        system,
        user,
        model=model,
        max_tokens=min(4096, 512 + len(patch_input.problematic_span) // 2),
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

    reason = ""
    patched = ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            patched = (parsed.get("patch") or "").strip()
            reason = (parsed.get("reason") or "").strip()
    except (json.JSONDecodeError, TypeError):
        # Fallback: treat whole response as patch (no reason)
        patched = raw
    if not patched and raw:
        patched = raw

    if diagnosis.target == "output_surface":
        success = bool(patched and patched != patch_input.problematic_span.strip())
    else:
        success = bool(patched)
    return patched, success, reason


# ---------------------------------------------------------------------------
# 4. Apply patch (string surgery) + rerun request
# ---------------------------------------------------------------------------


def build_rerun_request(
    trace_id: str,
    location: str,
    error_id: str,
    error_type: str,
    original_span_text: str,
    patch_payload: str,
    diagnosis: MechanismDiagnosis,
    patch_reason: str = "",
) -> Dict[str, Any]:
    """
    Build a rerun request for an external harness. apply_mode and instruction
    depend on diagnosis.target; patch_payload is either replacement span text
    (output_surface) or context/policy snippet to insert (input_context, local_decision_policy).
    patch_reason is the short explanation from the LLM JSON response (reason field).
    """
    if diagnosis.target == "output_surface":
        apply_mode = "replace_span_output"
        instruction = "Replace the content of the intervention span with patch_payload, then regenerate the suffix from that point."
    elif diagnosis.target == "input_context":
        apply_mode = "prepend_context_before_rerun"
        instruction = "Insert patch_payload as additional local context immediately before rerunning the intervention span, then regenerate the suffix."
    else:
        apply_mode = "inject_local_policy_before_rerun"
        instruction = "Insert patch_payload as a local decision rule immediately before rerunning the intervention span, then regenerate the suffix."

    out = {
        "trace_id": trace_id,
        "intervention_span_id": location,
        "error_id": error_id,
        "error_type": error_type,
        "diagnosis_target": diagnosis.target,
        "diagnosis_reasoning": diagnosis.reasoning,
        "apply_mode": apply_mode,
        "original_span_text": original_span_text[:2000],
        "patch_payload": patch_payload[:8000],
        "instruction": instruction,
    }
    if patch_reason:
        out["patch_reason"] = patch_reason[:500]
    return out


# ---------------------------------------------------------------------------
# 5. Evaluation: compare downstream B
# ---------------------------------------------------------------------------


def compare_baseline_rerun_errors(
    baseline_errors: List[dict],
    rerun_errors: List[dict],
) -> Dict[str, Any]:
    """
    Compare error presence between baseline and rerun (e.g. after counterfactual rerun).
    Use after rerun: load baseline annotation errors and rerun annotation errors for the same trace.

    Returns: {
      "baseline_by_type": { "A": count, ... },
      "rerun_by_type": { "A": count, ... },
      "delta_presence": { "A": baseline_count - rerun_count, ... },
      "summary": str
    }
    """
    from collections import Counter
    c_base = Counter(e.get("category", "") for e in baseline_errors)
    c_rerun = Counter(e.get("category", "") for e in rerun_errors)
    all_cats = set(c_base) | set(c_rerun)
    delta_presence = {b: c_base.get(b, 0) - c_rerun.get(b, 0) for b in all_cats}
    summary = (
        f"Baseline: {dict(c_base)}. Rerun: {dict(c_rerun)}. "
        f"Delta (baseline - rerun): {delta_presence}"
    )
    return {
        "baseline_by_type": dict(c_base),
        "rerun_by_type": dict(c_rerun),
        "delta_presence": delta_presence,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Full pipeline for one error
# ---------------------------------------------------------------------------


def run_llm_patch_pipeline(
    trace_obj: TraceObj,
    error_instance: dict,
    model: str = "openai/gpt-4o",
    context_window: int = 1,
) -> PatchResult:
    """
    Run the full pipeline for one error: build input → diagnose → generate patch.
    """
    location = error_instance.get("annotated_span_id") or error_instance.get("location") or ""
    if not location:
        return PatchResult(
            trace_id=trace_obj.trace_id,
            error_id=error_instance.get("error_id", ""),
            location="",
            error_type=error_instance.get("category", ""),
            diagnosis=MechanismDiagnosis(target="", mechanism="", reasoning="no location"),
            original_span_text="",
            patched_span_text="",
            success=False,
            error_message="missing location",
        )

    try:
        patch_input = build_patch_input(trace_obj, error_instance, context_window=context_window)
    except Exception as e:
        return PatchResult(
            trace_id=trace_obj.trace_id,
            error_id=error_instance.get("error_id", ""),
            location=location,
            error_type=error_instance.get("category", ""),
            diagnosis=MechanismDiagnosis(target="", mechanism="", reasoning=""),
            original_span_text="",
            patched_span_text="",
            success=False,
            error_message=str(e),
        )

    try:
        diagnosis = diagnose_mechanism(patch_input, model=model)
    except Exception as e:
        return PatchResult(
            trace_id=trace_obj.trace_id,
            error_id=patch_input.error_id,
            location=location,
            error_type=patch_input.error_type,
            diagnosis=MechanismDiagnosis(target="", mechanism="", reasoning=""),
            original_span_text=patch_input.problematic_span,
            patched_span_text="",
            success=False,
            error_message=f"diagnosis failed: {e}",
        )

    try:
        patched_text, gen_ok, patch_reason = generate_single_error_patch(patch_input, diagnosis, model=model)
    except Exception as e:
        return PatchResult(
            trace_id=trace_obj.trace_id,
            error_id=patch_input.error_id,
            location=location,
            error_type=patch_input.error_type,
            diagnosis=diagnosis,
            original_span_text=patch_input.problematic_span,
            patched_span_text="",
            success=False,
            error_message=f"patch generation failed: {e}",
        )

    if not gen_ok:
        return PatchResult(
            trace_id=trace_obj.trace_id,
            error_id=patch_input.error_id,
            location=location,
            error_type=patch_input.error_type,
            diagnosis=diagnosis,
            original_span_text=patch_input.problematic_span,
            patched_span_text=patched_text or "",
            success=False,
            error_message="patch unchanged or empty",
        )

    rerun_request = build_rerun_request(
        trace_obj.trace_id,
        location,
        patch_input.error_id,
        patch_input.error_type,
        patch_input.problematic_span,
        patch_payload=patched_text,
        diagnosis=diagnosis,
        patch_reason=patch_reason,
    )

    return PatchResult(
        trace_id=trace_obj.trace_id,
        error_id=patch_input.error_id,
        location=location,
        error_type=patch_input.error_type,
        diagnosis=diagnosis,
        original_span_text=patch_input.problematic_span,
        patched_span_text=patched_text,
        success=True,
        rerun_request=rerun_request,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM-based per-error patch generation: diagnose mechanism, generate minimal patch, output rerun request.",
    )
    parser.add_argument("--trace_dir", default="data/GAIA", help="Directory of trace JSONs")
    parser.add_argument("--annotations_dir", default="processed_annotations_gaia", help="Directory of annotation JSONs")
    parser.add_argument("--out_dir", default="outputs/llm_patches", help="Output directory for patch_log.jsonl and rerun_requests.jsonl")
    parser.add_argument("--trace_ids", nargs="*", help="Optional list of trace IDs; default: all with annotations")
    parser.add_argument("--max_traces", type=int, default=None, help="Cap number of traces")
    parser.add_argument("--max_errors_per_trace", type=int, default=None, help="Cap errors per trace (default: all)")
    parser.add_argument("--model", default="openai/gpt-4o", help="Model for litellm (e.g. openai/gpt-4o)")
    parser.add_argument("--context_window", type=int, default=1, help="Context window for local context (0=span only)")
    args = parser.parse_args()

    if not LITELLM_AVAILABLE:
        print("litellm is not installed. pip install litellm", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "patch_log.jsonl")
    rerun_path = os.path.join(args.out_dir, "rerun_requests.jsonl")

    from trail_io import iter_trail_traces
    pairs = list(iter_trail_traces(args.trace_dir, args.annotations_dir, args.trace_ids, args.max_traces))
    n_success = 0
    n_fail = 0

    with open(log_path, "w", encoding="utf-8") as log_f, open(rerun_path, "w", encoding="utf-8") as rerun_f:
        for trace_path, ann_path in pairs:
            try:
                trace_obj = load_trail_trace(trace_path, ann_path)
            except Exception as e:
                print(f"[WARN] load failed {trace_path}: {e}", file=sys.stderr)
                continue
            errors = trace_obj.errors
            if args.max_errors_per_trace is not None:
                errors = errors[: args.max_errors_per_trace]
            for err in errors:
                result = run_llm_patch_pipeline(
                    trace_obj,
                    err,
                    model=args.model,
                    context_window=args.context_window,
                )
                rec = {
                    "trace_id": result.trace_id,
                    "error_id": result.error_id,
                    "location": result.location,
                    "error_type": result.error_type,
                    "diagnosis_target": result.diagnosis.target,
                    "diagnosis_reasoning": result.diagnosis.reasoning[:500],
                    "success": result.success,
                    "error_message": result.error_message,
                }
                if result.success and result.rerun_request and result.rerun_request.get("patch_reason"):
                    rec["patch_reason"] = result.rerun_request["patch_reason"]
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if result.success and result.rerun_request:
                    rerun_f.write(json.dumps(result.rerun_request, ensure_ascii=False) + "\n")
                    n_success += 1
                else:
                    n_fail += 1
                status = "OK" if result.success else "FAIL"
                print(f"  [{status}] {result.trace_id} {result.location} {result.error_type} -> {result.diagnosis.target}")

    print(f"\nWrote {log_path}, {rerun_path}. Success: {n_success}, Failed: {n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
