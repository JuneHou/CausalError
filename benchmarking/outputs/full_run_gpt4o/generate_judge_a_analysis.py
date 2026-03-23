#!/usr/bin/env python3
"""
Generate analysis file for the 18 unresolved Judge-A cases.

For each unresolved case, reconstructs the exact input given to Judge-A
and appends the annotation (description + evidence) from the processed
annotations for easier error analysis.
"""
import json
import os
import sys

BASE_DIR = "/data/wang/junh/githubs/trail-benchmark/benchmarking"
OUT_DIR = os.path.join(BASE_DIR, "outputs/full_run_gpt4o")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "processed_annotations_gaia")

# ---------------------------------------------------------------------------
# Prompt templates (mirroring judge_a_resolved.py exactly)
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
Respond with JSON only: {{"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}}"""

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
Respond with JSON only: {{"resolved": bool, "confidence": float 0-1, "evidence_excerpt": "string"}}"""


def build_rerun_block(rerun_result: dict) -> str:
    rerun_status = rerun_result.get("rerun_status", "rerun_missing_suffix")
    rerun_spans = rerun_result.get("rerun_suffix_spans") or []
    if rerun_status == "live_rerun_success" and rerun_spans:
        rerun_text = "\n---\n".join(str(s)[:800] for s in rerun_spans[:3])
        return (
            "RERUN_SUFFIX (supplementary context — first 3 spans after t_A in counterfactual run):\n"
            "<<<\n" + rerun_text + "\n>>>"
        )
    return f"RERUN_SUFFIX: ({rerun_status} — counterfactual trace not available)"


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_annotation(trace_id: str, span_id: str):
    ann_path = os.path.join(ANNOTATIONS_DIR, f"{trace_id}.json")
    if not os.path.exists(ann_path):
        return None
    data = json.load(open(ann_path))
    for err in data.get("errors", []):
        if err.get("location") == span_id:
            return err
    return None


def main():
    # Load all data
    a_resolved = load_jsonl(os.path.join(OUT_DIR, "a_resolved.jsonl"))
    a_instances = {r["error_id"]: r for r in load_jsonl(os.path.join(OUT_DIR, "a_instances.jsonl"))}
    patch_results = {r["error_id"]: r for r in load_jsonl(os.path.join(OUT_DIR, "patch_results.jsonl"))}
    rerun_results = {r["error_id"]: r for r in load_jsonl(os.path.join(OUT_DIR, "rerun_results.jsonl"))}

    unresolved = [r for r in a_resolved if not r.get("resolved", True)]
    print(f"Total a_resolved: {len(a_resolved)}, Unresolved: {len(unresolved)}")

    output = []
    for verdict in unresolved:
        error_id = verdict["error_id"]
        trace_id = verdict["trace_id"]
        a_location = verdict["a_location"]

        a_inst = a_instances.get(error_id, {})
        patch = patch_results.get(error_id, {})
        rerun = rerun_results.get(error_id, {})

        a_instance = a_inst.get("a_instance", {})
        a_cat = a_instance.get("category", "")
        patch_side = a_inst.get("patch_side", "replace_span_output")

        rerun_block = build_rerun_block(rerun)

        template = (
            JUDGE_A_USER_INPUT_TEMPLATE
            if patch_side == "replace_span_input"
            else JUDGE_A_USER_OUTPUT_TEMPLATE
        )

        user_msg = template.format(
            A=a_cat,
            ERROR_DESCRIPTION=(a_instance.get("description") or "")[:800],
            ERROR_EVIDENCE=(a_instance.get("evidence") or "")[:800],
            ORIGINAL_SNIPPET=(a_inst.get("local_snippet") or "")[:2000],
            PATCH_PAYLOAD=(patch.get("patch_payload") or "")[:2000],
            RERUN_CONTEXT_BLOCK=rerun_block,
        )

        # Look up annotation from processed_annotations_gaia using annotated_location
        # (intervention_location may differ from annotated_location)
        annotated_location = a_inst.get("annotated_location", a_location)
        annotation = load_annotation(trace_id, annotated_location)
        # Fallback: try intervention span if annotated_location didn't match
        if annotation is None and annotated_location != a_location:
            annotation = load_annotation(trace_id, a_location)

        entry = {
            "error_id": error_id,
            "trace_id": trace_id,
            "a_location": a_location,
            "annotated_location": annotated_location,
            "a_category": a_cat,
            "patch_side": patch_side,
            "judge_verdict": {
                "resolved": verdict["resolved"],
                "confidence": verdict["confidence"],
                "evidence_excerpt": verdict.get("evidence_excerpt", ""),
                "rerun_status": verdict.get("rerun_status", ""),
            },
            "judge_a_input": {
                "system": JUDGE_A_SYSTEM,
                "user": user_msg,
            },
            "local_snippet": a_inst.get("local_snippet", ""),
            "patch_payload": patch.get("patch_payload", ""),
            "postcheck_passed": patch.get("postcheck_passed", None),
            "postcheck_failures": patch.get("postcheck_failures", []),
            "annotation": annotation,
        }
        output.append(entry)

    out_path = os.path.join(OUT_DIR, "judge_a_unresolved_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} unresolved cases to {out_path}")

    # Summary
    print("\n--- Summary ---")
    for i, e in enumerate(output):
        print(f"[{i+1:2d}] {e['error_id']}")
        print(f"     category={e['a_category']}, confidence={e['judge_verdict']['confidence']}")
        print(f"     postcheck_passed={e['postcheck_passed']}, failures={e['postcheck_failures']}")
        ann = e.get("annotation")
        if ann:
            print(f"     annotation.description (first 100): {ann.get('description','')[:100]}")
        else:
            print(f"     annotation: NOT FOUND in processed_annotations_gaia")
        print()


if __name__ == "__main__":
    main()
