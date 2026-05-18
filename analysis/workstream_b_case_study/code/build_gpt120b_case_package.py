#!/usr/bin/env python3
"""
Build a paper-ready Workstream B case package for selected GPT-oss-120B IDs.

Inputs:
- results/gpt120b_top4_shortlist.csv
- predictions from baseline / causal-only / corr0.35 directories
- gold annotations from processed annotations
- trace prompt snippets from benchmark data files
- corr metadata (_meta_*.json)

Outputs:
- results/gpt120b_case_package.json
- results/gpt120b_case_package.md
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO = Path("/data/wang/junh/githubs/trail-benchmark")
RESULTS = REPO / "analysis/workstream_b_case_study/results"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_n_categories(payload: Dict, n: int = 6) -> List[str]:
    return [e.get("category", "") for e in payload.get("errors", [])[:n]]


def scores(payload: Dict) -> Dict:
    s = (payload.get("scores") or [{}])[0]
    return {
        "reliability": s.get("reliability_score"),
        "instruction_adherence": s.get("instruction_adherence_score"),
        "plan_opt": s.get("plan_opt_score"),
        "overall": s.get("overall"),
    }


def _extract_from_input_value(raw: str) -> Optional[str]:
    # raw often looks like {"task": "..."} but may contain escaped JSON.
    candidates = [raw]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("task"), str):
            return parsed["task"][:260].replace("\n", " ").replace("\r", " ").strip()
    except Exception:
        pass
    # Try stripping outer braces escaped as text.
    if '\\"task\\"' in raw or '"task"' in raw:
        candidates.append(raw.replace('\\"', '"'))
    for c in candidates:
        m = re.search(r'"task"\s*:\s*"(.+?)"', c, flags=re.S)
        if m:
            return m.group(1)[:260].replace('\\"', '"').replace("\\n", " ").replace("\\r", " ").strip()
    return None


def _walk_for_prompt(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        # Direct question/task keys.
        for k in ("question", "task", "issue"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v[:260].replace("\n", " ").replace("\r", " ").strip()
        # Span-level input.value often holds escaped JSON with task.
        span_attrs = obj.get("span_attributes")
        if isinstance(span_attrs, dict):
            iv = span_attrs.get("input.value")
            if isinstance(iv, str):
                extracted = _extract_from_input_value(iv)
                if extracted:
                    return extracted
        # Many SWE traces store task text in generic content fields.
        for k, v in obj.items():
            if isinstance(v, str):
                vv = v.replace("\\n", " ").replace("\\r", " ").strip()
                if "New task:" in vv:
                    snippet = vv.split("New task:", 1)[1].strip()
                    return snippet[:260]
                if "<issue>" in vv:
                    snippet = vv.split("<issue>", 1)[1].strip()
                    return snippet[:260]
                if "You will be provided with a partial code base" in vv:
                    return vv[:260]

        for v in obj.values():
            got = _walk_for_prompt(v)
            if got:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _walk_for_prompt(it)
            if got:
                return got
    return None


def extract_question_snippet(trace_json_text: str) -> str:
    # Best effort structured traversal first.
    try:
        obj = json.loads(trace_json_text)
        got = _walk_for_prompt(obj)
        if got:
            return got
    except Exception:
        pass

    # Try GAIA-style question first.
    m = re.search(r'"question"\s*:\s*"(.+?)"', trace_json_text, flags=re.S)
    if m:
        q = m.group(1)
        q = q.replace('\\"', '"').replace("\\n", " ").replace("\\r", " ")
        return q[:260].strip()

    # SWE-style input task field in spans.
    m2 = re.search(r'"input\.value"\s*:\s*"(?:\\\{)?\\?"task\\?"\s*:\s*\\?"(.+?)\\?(?:\\\})?"', trace_json_text, flags=re.S)
    if m2:
        q = m2.group(1)
        q = q.replace('\\"', '"').replace("\\n", " ").replace("\\r", " ")
        return q[:260].strip()

    # Generic final fallback for raw text with SWE-style prompt blocks.
    m3 = re.search(r'New task:\\n(.+?)\\n\\n<issue>', trace_json_text, flags=re.S)
    if m3:
        q = m3.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\r", " ")
        return q[:260].strip()
    m4 = re.search(r'<issue>\\n(.+?)\\n\\n</issue>', trace_json_text, flags=re.S)
    if m4:
        q = m4.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\r", " ")
        return q[:260].strip()

    return "(question snippet unavailable)"


def extract_true_answer(trace_json_text: str) -> Optional[str]:
    m = re.search(r'"true_answer"\s*:\s*"(.+?)"', trace_json_text, flags=re.S)
    if m:
        return m.group(1).replace('\\"', '"').strip()
    return None


def locate_files(split: str, trace_id: str) -> Dict[str, Path]:
    split_gt = "processed_annotations_gaia" if split == "GAIA_dedup" else "processed_annotations_swe_bench"
    split_data = "GAIA" if split == "GAIA_dedup" else "SWE_Bench_dedup"
    return {
        "gold": REPO / "benchmarking" / split_gt / f"{trace_id}.json",
        "baseline": REPO / "benchmarking/outputs/zero_shot2" / f"outputs_openai-gpt-oss-120b-{split}" / f"{trace_id}.json",
        "causal": REPO / "benchmarking/outputs/zero_shot2" / f"outputs_openai-gpt-oss-120b-{split}-graph_inject_causal_only_span_index" / f"{trace_id}.json",
        "corr": REPO / "benchmarking/outputs_thres/t0.35" / f"outputs_gpt-oss-120b-{split}-graph_inject_causal_corr0.35_span_index" / f"{trace_id}.json",
        "corr_meta": REPO / "benchmarking/outputs_thres/t0.35" / f"outputs_gpt-oss-120b-{split}-graph_inject_causal_corr0.35_span_index" / f"_meta_{trace_id}.json",
        "trace_data": REPO / "benchmarking/data" / split_data / f"{trace_id}.json",
    }


def build_case(row: Dict) -> Dict:
    split = row["split"]
    tid = row["trace_id"]
    p = locate_files(split, tid)

    gold = load_json(p["gold"])
    base = load_json(p["baseline"])
    causal = load_json(p["causal"])
    corr = load_json(p["corr"])
    meta = load_json(p["corr_meta"])
    trace_text = p["trace_data"].read_text(encoding="utf-8", errors="ignore")

    case = {
        "case_type": row["case_type"],
        "model": row["model"],
        "split": split,
        "trace_id": tid,
        "judge_score": float(row["judge_score"]),
        "deltas": {
            "delta_corr_vs_causal": float(row["delta_corr_vs_causal"]),
            "delta_corr_vs_base": float(row["delta_corr_vs_base"]),
            "delta_wf1_vs_causal": float(row["delta_wf1_vs_causal"]),
            "delta_loc_vs_causal": float(row["delta_loc_vs_causal"]),
            "delta_joint_vs_causal": float(row["delta_joint_vs_causal"]),
        },
        "prompt_snippet": extract_question_snippet(trace_text),
        "true_answer": extract_true_answer(trace_text),
        "gold_categories": first_n_categories(gold, n=8),
        "baseline_categories": first_n_categories(base, n=8),
        "causal_categories": first_n_categories(causal, n=8),
        "corr_categories": first_n_categories(corr, n=8),
        "score_snapshot": {
            "gold": scores(gold),
            "baseline": scores(base),
            "causal": scores(causal),
            "corr": scores(corr),
        },
        "corr_meta": {
            "graph": meta.get("graph"),
            "pass1_detected": meta.get("pass1_detected", []),
            "pass2_triggered": meta.get("pass2_triggered"),
            "pass2_filtered_edges": meta.get("pass2_filtered_edges"),
            "pass2_new_errors": meta.get("pass2_new_errors"),
        },
    }

    # Draft one-line takeaway for writing acceleration.
    d = case["deltas"]
    if case["case_type"] == "working":
        case["draft_takeaway"] = (
            f"Corr-union improves over causal-only (ΔW-F1 {d['delta_wf1_vs_causal']:+.3f}, "
            f"ΔLoc {d['delta_loc_vs_causal']:+.3f}, ΔJoint {d['delta_joint_vs_causal']:+.3f}) "
            "while preserving explicit graph-trigger traceability."
        )
    else:
        case["draft_takeaway"] = (
            f"Corr-union regresses vs causal-only (ΔW-F1 {d['delta_wf1_vs_causal']:+.3f}, "
            f"ΔLoc {d['delta_loc_vs_causal']:+.3f}, ΔJoint {d['delta_joint_vs_causal']:+.3f}); "
            "use as bounded failure evidence."
        )
    return case


def to_markdown(cases: List[Dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# GPT-120B Workstream B Case Package")
    lines.append("")
    lines.append("Selected cases (2 working + 2 not-working) with traceable judge logic and reusable writing snippets.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("- Source shortlist: `gpt120b_top4_shortlist.csv` (locked 4 IDs).")
    lines.append("- For each case, gather artifacts from existing outputs only:")
    lines.append("  - gold annotation, baseline prediction, causal-only prediction, corr-union prediction, corr metadata, and trace data.")
    lines.append("- Normalize and summarize into one structured package for writing.")
    lines.append("- Emit both machine-readable JSON and reviewer-facing Markdown.")
    lines.append("")
    lines.append("## Filters and Selection Logic (already applied before this package)")
    lines.append("")
    lines.append("- Model fixed to `GPT-oss-120B`.")
    lines.append("- Case composition fixed to `2 working + 2 not_working`.")
    lines.append("- Working eligibility:")
    lines.append("  - `Δ(corr-causal) > 0`, `Δ(corr-baseline) > 0`, `ΔW-F1 > 0`, `ΔLoc > 0`.")
    lines.append("- Joint handling:")
    lines.append("  - prefer `ΔJoint > 0`; if insufficient candidates, relax joint only.")
    lines.append("- User-approved hard include:")
    lines.append("  - SWE working case `c104d0e28f4f8dddeea1dd90b4138e5a`.")
    lines.append("- Not-working eligibility:")
    lines.append("  - `Δ(corr-causal) < 0` with at least one negative metric delta.")
    lines.append("")
    lines.append("## Output Schema (What each case includes)")
    lines.append("")
    lines.append("- case type (`working` / `not_working`)")
    lines.append("- split and trace ID")
    lines.append("- delta metrics (`ΔW-F1`, `ΔLoc`, `ΔJoint`, plus corr-vs-causal/base)")
    lines.append("- prompt/question snippet (when extractable)")
    lines.append("- gold / baseline / causal-only / corr category heads")
    lines.append("- score snapshot (reliability, instruction adherence, plan optimization, overall)")
    lines.append("- corr metadata:")
    lines.append("  - `pass1_detected`")
    lines.append("  - `pass2_triggered`")
    lines.append("  - `pass2_filtered_edges`")
    lines.append("  - `pass2_new_errors`")
    lines.append("- draft takeaway sentence")
    lines.append("- writing slots for:")
    lines.append("  - activated-edge evidence")
    lines.append("  - reviewer-convincing rationale")
    lines.append("")
    for i, c in enumerate(cases, start=1):
        lines.append(f"## Case {i}: `{c['trace_id']}` ({c['case_type']}, {c['split']})")
        lines.append("")
        lines.append(f"- **Prompt snippet**: {c['prompt_snippet']}")
        if c.get("true_answer"):
            lines.append(f"- **Reference answer**: `{c['true_answer']}`")
        lines.append(f"- **Delta vs causal-only**: W-F1 `{c['deltas']['delta_wf1_vs_causal']:+.3f}`, Loc `{c['deltas']['delta_loc_vs_causal']:+.3f}`, Joint `{c['deltas']['delta_joint_vs_causal']:+.3f}`")
        lines.append(f"- **Pass-1 detected**: {', '.join(c['corr_meta']['pass1_detected']) if c['corr_meta']['pass1_detected'] else '(none)'}")
        lines.append(f"- **Pass-2 filtered edges**: `{c['corr_meta']['pass2_filtered_edges']}`; **new errors**: `{c['corr_meta']['pass2_new_errors']}`")
        lines.append(f"- **Gold categories (head)**: {', '.join(c['gold_categories'][:5]) if c['gold_categories'] else '(none)'}")
        lines.append(f"- **Baseline categories (head)**: {', '.join(c['baseline_categories'][:5]) if c['baseline_categories'] else '(none)'}")
        lines.append(f"- **Causal-only categories (head)**: {', '.join(c['causal_categories'][:5]) if c['causal_categories'] else '(none)'}")
        lines.append(f"- **Corr-union categories (head)**: {', '.join(c['corr_categories'][:5]) if c['corr_categories'] else '(none)'}")
        lines.append(f"- **Draft takeaway**: {c['draft_takeaway']}")
        lines.append("- **Writing slots**:")
        lines.append("  - Activated-edge evidence: `<fill from pass2 / prompt trace>`")
        lines.append("  - Why this is convincing for reviewers: `<1-2 lines>`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    shortlist_path = RESULTS / "gpt120b_top4_shortlist.csv"
    rows = list(csv.DictReader(shortlist_path.open("r", encoding="utf-8")))
    cases = [build_case(r) for r in rows]

    out_json = RESULTS / "gpt120b_case_package.json"
    out_md = RESULTS / "gpt120b_case_package.md"
    out_json.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    to_markdown(cases, out_md)

    print(json.dumps({
        "shortlist_source": str(shortlist_path),
        "out_json": str(out_json),
        "out_md": str(out_md),
        "case_count": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()
