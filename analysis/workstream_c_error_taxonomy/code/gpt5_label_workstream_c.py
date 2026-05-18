#!/usr/bin/env python3
"""
Label Workstream-C audit sheet with GPT-5.2 (default) using frozen two-layer schema.

Modes:
- pilot: label first N rows into trail_audit_sheet_pilot_labeled.csv
- main: label all rows into trail_audit_sheet_gpt52_labeled.csv

Prompt profile is fixed to compact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from annotation_schema import (
    CONFIDENCE_LABELS,
    CORR_EDGE_ROLE_LABELS,
    MECHANISM_BUCKETS,
    PATTERN_TAGS,
    SEVERITY_LABELS,
)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_tag_list(tags: List[str]) -> List[str]:
    allowed = set(PATTERN_TAGS)
    out = []
    for t in tags:
        tt = t.strip()
        if tt in allowed and tt not in out:
            out.append(tt)
    return out[:4]


def coerce_label(value: str, allowed: List[str], fallback: str) -> str:
    v = (value or "").strip()
    return v if v in allowed else fallback


def build_prompt_compact(row: Dict[str, str], label_mechanism_bucket: bool) -> str:
    # Compact prompt for context-window safety and latency stability.
    keys = (
        '"mechanism_bucket":"","pattern_tags":[],"pattern_characteristics":"","corr_edge_role":"","impact_severity":"","confidence":"","evidence_note":""'
        if label_mechanism_bucket
        else '"pattern_tags":[],"pattern_characteristics":"","corr_edge_role":"","impact_severity":"","confidence":"","evidence_note":""'
    )
    mechanism_label_line = f"mechanism_bucket={MECHANISM_BUCKETS}\n" if label_mechanism_bucket else ""
    return f"""Annotate one TRAIL row. Return JSON only:
{{{keys}}}

Allowed labels:
{mechanism_label_line}pattern_tags(max 3)={PATTERN_TAGS}
corr_edge_role={CORR_EDGE_ROLE_LABELS}
impact_severity={SEVERITY_LABELS}
confidence={CONFIDENCE_LABELS}

Rules:
- Use only allowed labels.
- Keep pattern_characteristics <= 20 words.
- Keep evidence_note <= 35 words.
- Prefer concise, high-precision labeling.

Row:
id={row.get("sample_id","")} trace={row.get("trace_id","")} split={row.get("split","")}
deltas: corr_vs_causal={row.get("delta_corr_vs_causal","")}, wf1_vs_causal={row.get("delta_wf1_vs_causal","")}, loc_vs_causal={row.get("delta_loc_vs_causal","")}, joint_vs_causal={row.get("delta_joint_vs_causal","")}
gold={row.get("gold_leaf_categories","")}
base_tp={row.get("baseline_tp_leaf","")} base_fp={row.get("baseline_fp_leaf","")} base_fn={row.get("baseline_fn_leaf","")}
causal_tp={row.get("causal_tp_leaf","")} causal_fp={row.get("causal_fp_leaf","")} causal_fn={row.get("causal_fn_leaf","")}
corr_tp={row.get("corr_tp_leaf","")} corr_fp={row.get("corr_fp_leaf","")} corr_fn={row.get("corr_fn_leaf","")}
task={row.get("question_snippet","")}
"""


def call_openai_responses(model: str, prompt: str) -> Dict:
    # Lazy import so the rest of the pipeline runs without API deps.
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    text = getattr(resp, "output_text", None) or ""
    if not text:
        # Fallback for SDK variants.
        text = json.dumps(resp.model_dump()) if hasattr(resp, "model_dump") else "{}"
    try:
        return json.loads(text)
    except Exception:
        # Try coarse extraction.
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def apply_label(row: Dict[str, str], label: Dict[str, object], annotator: str) -> Dict[str, str]:
    out = dict(row)
    rule_bucket = row.get("rule_based_mechanism_bucket", row.get("suggested_mechanism_bucket", "causal-preserving-neutral"))
    # Always enforce deterministic mechanism bucket from rule-based assignment.
    out["mechanism_bucket"] = coerce_label(str(rule_bucket), MECHANISM_BUCKETS, "causal-preserving-neutral")
    out["pattern_tags"] = "|".join(sanitize_tag_list(label.get("pattern_tags", []) if isinstance(label.get("pattern_tags"), list) else []))
    out["pattern_characteristics"] = str(label.get("pattern_characteristics", "")).strip()[:220]
    out["corr_edge_role"] = coerce_label(str(label.get("corr_edge_role", "")), CORR_EDGE_ROLE_LABELS, "unknown")
    out["impact_severity"] = coerce_label(str(label.get("impact_severity", "")), SEVERITY_LABELS, "medium")
    out["confidence"] = coerce_label(str(label.get("confidence", "")), CONFIDENCE_LABELS, "medium")
    out["evidence_note"] = str(label.get("evidence_note", "")).strip()[:280]
    out["annotator"] = annotator
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_sheet.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    parser.add_argument("--pilot_n", type=int, default=10)
    parser.add_argument("--sleep_s", type=float, default=0.2)
    parser.add_argument("--label_mechanism_bucket", action="store_true", help="If set, ask model to emit mechanism_bucket; otherwise mechanism_bucket stays rule-based only.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input_csv)
    if not rows:
        raise SystemExit("Input audit sheet is empty.")

    selected = rows[: args.pilot_n] if args.mode == "pilot" else rows
    out_rows = []
    logs = []
    t0 = time.time()

    for i, row in enumerate(selected, start=1):
        prompt = build_prompt_compact(row, label_mechanism_bucket=args.label_mechanism_bucket)
        status = "ok"
        label = {}
        err = ""
        try:
            label = call_openai_responses(args.model, prompt)
            out_rows.append(apply_label(row, label, annotator=f"{args.model}:{args.mode}"))
        except Exception as e:  # noqa: BLE001
            status = "error"
            err = str(e)
            # Keep row for manual follow-up.
            fallback = dict(row)
            fallback["annotator"] = f"{args.model}:{args.mode}:failed"
            fallback["evidence_note"] = f"LABELING_ERROR: {err[:220]}"
            out_rows.append(fallback)

        logs.append(
            {
                "index": i,
                "sample_id": row.get("sample_id", ""),
                "trace_id": row.get("trace_id", ""),
                "status": status,
                "error": err,
            }
        )
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    out_csv = args.out_dir / ("trail_audit_sheet_pilot_labeled.csv" if args.mode == "pilot" else "trail_audit_sheet_gpt52_labeled.csv")
    log_json = args.out_dir / ("trail_gpt5_pilot_log.json" if args.mode == "pilot" else "trail_gpt5_main_log.json")
    run_json = args.out_dir / ("trail_gpt5_pilot_run_manifest.json" if args.mode == "pilot" else "trail_gpt5_main_run_manifest.json")
    write_rows(out_csv, out_rows)
    with log_json.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    manifest = {
        "mode": args.mode,
        "model": args.model,
        "input_csv": str(args.input_csv),
        "output_csv": str(out_csv),
        "log_json": str(log_json),
        "rows_input": len(rows),
        "rows_labeled": len(out_rows),
        "prompt_profile": "compact-fixed",
        "ok": sum(1 for x in logs if x["status"] == "ok"),
        "error": sum(1 for x in logs if x["status"] == "error"),
        "elapsed_s": round(time.time() - t0, 3),
        "notes": [
            "Uses frozen two-layer schema in annotation_schema.py.",
            "mechanism_bucket is rule-based deterministic unless --label_mechanism_bucket is explicitly enabled.",
            "Run pilot first, review, then run main.",
        ],
    }
    with run_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
