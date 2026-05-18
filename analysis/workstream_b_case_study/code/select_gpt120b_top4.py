#!/usr/bin/env python3
"""
Pick a paper-ready top-4 case shortlist for GPT-oss-120B with explicit judge logic.

Input:
- results/gpt120b_instance_comparison.csv

Output:
- results/gpt120b_top4_shortlist.csv
- results/gpt120b_top4_shortlist.json

Design goal:
- maximize reviewer-convincing evidence while remaining transparent:
  2 working + 2 not-working, with split diversity when possible.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f_in:
        for r in csv.DictReader(f_in):
            # materialize numeric fields used in logic
            for k in (
                "delta_corr_vs_causal",
                "delta_corr_vs_base",
                "delta_wf1_vs_causal",
                "delta_loc_vs_causal",
                "delta_joint_vs_causal",
            ):
                r[k] = f(r[k])
            rows.append(r)
    return rows


def working_score(r: Dict) -> float:
    # prioritize joint + wf1 gains, keep loc in mix, and reward gain over baseline
    metric_mix = (
        0.50 * r["delta_joint_vs_causal"]
        + 0.35 * r["delta_wf1_vs_causal"]
        + 0.15 * r["delta_loc_vs_causal"]
    )
    base_bonus = 0.25 * max(0.0, r["delta_corr_vs_base"])
    split_bonus = 0.05 if r["split"] == "SWE_Bench_dedup" else 0.0
    return metric_mix + base_bonus + split_bonus


def not_working_score(r: Dict) -> float:
    # favor clear, high-impact regressions (especially joint/loc damage)
    reg = abs(min(0.0, r["delta_corr_vs_causal"]))
    joint = abs(min(0.0, r["delta_joint_vs_causal"]))
    loc = abs(min(0.0, r["delta_loc_vs_causal"]))
    wf1 = abs(min(0.0, r["delta_wf1_vs_causal"]))
    split_bonus = 0.03 if r["split"] == "SWE_Bench_dedup" else 0.0
    return reg + 0.6 * joint + 0.3 * loc + 0.2 * wf1 + split_bonus


def is_working_eligible(r: Dict) -> bool:
    return (
        r["bucket"] == "working"
        and r["delta_corr_vs_causal"] > 0
        and r["delta_corr_vs_base"] > 0
        and r["delta_wf1_vs_causal"] > 0
        and r["delta_loc_vs_causal"] > 0
    )


def is_not_working_eligible(r: Dict) -> bool:
    return (
        r["bucket"] == "not_working"
        and r["delta_corr_vs_causal"] < 0
        and (
            r["delta_joint_vs_causal"] < 0
            or r["delta_loc_vs_causal"] < 0
            or r["delta_wf1_vs_causal"] < 0
        )
    )


def choose_with_split_diversity(cands: List[Dict], n: int, score_key: str, forced_trace_ids: List[str] | None = None) -> List[Dict]:
    # Prefer one SWE item when available for generalization evidence.
    ordered = sorted(cands, key=lambda r: r[score_key], reverse=True)
    out: List[Dict] = []
    used = set()
    forced_trace_ids = forced_trace_ids or []

    # Hard include forced ids first (if present in candidate set).
    by_id = {r["trace_id"]: r for r in ordered}
    for tid in forced_trace_ids:
        r = by_id.get(tid)
        if r and r["trace_id"] not in used and len(out) < n:
            out.append(r)
            used.add(r["trace_id"])

    swe = [r for r in ordered if r["split"] == "SWE_Bench_dedup"]
    gaia = [r for r in ordered if r["split"] == "GAIA_dedup"]

    if swe:
        out.append(swe[0])
        used.add(swe[0]["trace_id"])
    if gaia and len(out) < n:
        for r in gaia:
            if r["trace_id"] not in used:
                out.append(r)
                used.add(r["trace_id"])
                break

    for r in ordered:
        if len(out) >= n:
            break
        if r["trace_id"] in used:
            continue
        out.append(r)
        used.add(r["trace_id"])
    return out[:n]


def main() -> None:
    base = Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_b_case_study/results")
    src = base / "gpt120b_instance_comparison.csv"
    rows = load_rows(src)

    w = [r for r in rows if is_working_eligible(r)]
    nw = [r for r in rows if is_not_working_eligible(r)]

    for r in w:
        r["judge_score"] = working_score(r)
        r["case_type"] = "working"
    for r in nw:
        r["judge_score"] = not_working_score(r)
        r["case_type"] = "not_working"

    # Strict base eligibility (already encoded in w): positive F1 and Loc.
    forced_working_ids = ["c104d0e28f4f8dddeea1dd90b4138e5a"]  # user-approved SWE working case
    by_id = {r["trace_id"]: r for r in w}
    selected_working: List[Dict] = []
    if forced_working_ids[0] in by_id:
        selected_working.append(by_id[forced_working_ids[0]])

    # Then prefer joint-positive for remaining working slots.
    w_joint = [r for r in w if r["delta_joint_vs_causal"] > 0 and r["trace_id"] not in {x["trace_id"] for x in selected_working}]
    needed = 2 - len(selected_working)
    add_joint = choose_with_split_diversity(w_joint, n=needed, score_key="judge_score") if needed > 0 else []
    selected_working.extend(add_joint)

    # Fallback: if still not enough, relax joint-only preference.
    joint_fallback_used = False
    if len(selected_working) < 2:
        joint_fallback_used = True
        used = {x["trace_id"] for x in selected_working}
        remaining = [r for r in w if r["trace_id"] not in used]
        selected_working.extend(choose_with_split_diversity(remaining, n=2 - len(selected_working), score_key="judge_score"))
    selected_not_working = choose_with_split_diversity(nw, n=2, score_key="judge_score")
    selected = selected_working + selected_not_working

    # stable order: working first, then not_working; inside by score desc
    selected = sorted(
        selected,
        key=lambda r: (0 if r["case_type"] == "working" else 1, -r["judge_score"]),
    )

    out_csv = base / "gpt120b_top4_shortlist.csv"
    fields = [
        "case_type",
        "model",
        "split",
        "trace_id",
        "judge_score",
        "delta_corr_vs_causal",
        "delta_corr_vs_base",
        "delta_wf1_vs_causal",
        "delta_loc_vs_causal",
        "delta_joint_vs_causal",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for r in selected:
            writer.writerow({k: r.get(k, "") for k in fields})

    judge_logic = {
        "goal": "Select 2 working + 2 not-working cases that maximize convincing evidence while preserving transparency.",
        "hard_constraints": [
            "Exactly 2 working and 2 not_working cases.",
            "Prefer split diversity: include SWE and GAIA when possible in each case type.",
            "Working cases must improve corr over both causal-only and baseline.",
            "Not-working cases must be genuine corr regressions vs causal-only.",
        ],
        "working_eligibility": "bucket=working AND delta_corr_vs_causal>0 AND delta_corr_vs_base>0 AND delta_wf1_vs_causal>0 AND delta_loc_vs_causal>0.",
        "working_joint_preference": "Prefer delta_joint_vs_causal>0. If strict set cannot fill two working slots, fallback relaxes this joint constraint only.",
        "forced_include": "Working set hard-includes trace_id c104d0e28f4f8dddeea1dd90b4138e5a (SWE) per user confirmation.",
        "not_working_eligibility": "bucket=not_working AND delta_corr_vs_causal<0 AND at least one metric delta negative.",
        "working_score": "0.50*delta_joint_vs_causal + 0.35*delta_wf1_vs_causal + 0.15*delta_loc_vs_causal + 0.25*max(0,delta_corr_vs_base) + SWE bonus 0.05",
        "not_working_score": "|min(0,delta_corr_vs_causal)| + 0.6*|min(0,delta_joint_vs_causal)| + 0.3*|min(0,delta_loc_vs_causal)| + 0.2*|min(0,delta_wf1_vs_causal)| + SWE bonus 0.03",
    }

    out_json = base / "gpt120b_top4_shortlist.json"
    payload = {
        "source_file": str(src),
        "selected_count": len(selected),
        "selected": selected,
        "judge_logic": judge_logic,
        "coverage": {
            "eligible_working": len(w),
            "eligible_working_joint_positive": len(w_joint),
            "eligible_not_working": len(nw),
            "selected_working": len(selected_working),
            "selected_not_working": len(selected_not_working),
            "joint_fallback_used": joint_fallback_used,
        },
    }
    with out_json.open("w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=2)

    print(json.dumps({
        "out_csv": str(out_csv),
        "out_json": str(out_json),
        "selected_count": len(selected),
        "eligible_working": len(w),
        "eligible_not_working": len(nw),
    }, indent=2))


if __name__ == "__main__":
    main()
