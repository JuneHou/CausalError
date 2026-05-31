"""
Per-trace yes/no label breakdown for MAST holdout predictions.

Mirrors MAST eval/calculate_scores_yesno.py:
  - GT from holdout test.jsonl (mast_annotation, rec_id = line index 0000, 0001, ...)
  - 13 binary MAST modes per trace
  - weighted F1 / macro F1 via sklearn on (n_traces, 13) matrices

Unlike TRAIL dump_location_hits.py, MAST yes/no has no predicted locations.
Each trace record includes GT human errors (with step locations) from the
annotation file for qualitative cross-check.

Usage:
    python dump_yesno_hits.py PRED_DIR
    python dump_yesno_hits.py PRED_DIR --annotation /path/to/test.jsonl

PRED_DIR may be either:
  - a folder of 0000.json, 0001.json, ... (baseline), or
  - a parent folder (e.g. .../edge/) containing one run subdir with those files.

Output: {PRED_DIR}-yesno_hits.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
DEFAULT_ANNOTATION = HERE / "data" / "test_traces" / "mast" / "test.jsonl"

MAST_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.6",
    "3.1", "3.2", "3.3",
]

MAST_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "Weak Verification",
    "3.3": "No or Incorrect Verification",
}

# MAST yes/no prompt format: "1.1 Disobey Task Specification"
MAST_SHORT_NAMES = {m: f"{m} {MAST_NAMES[m]}" for m in MAST_MODES}


def modes_to_short_names(modes: list[str]) -> list[str]:
    return [MAST_SHORT_NAMES[m] for m in modes if m in MAST_SHORT_NAMES]


_REC_JSON = re.compile(r"^\d{4}\.json$")


def resolve_pred_dir(pred_dir: Path) -> tuple[Path, str]:
    """
    Find the directory that actually holds per-rec_id prediction JSON files.

    Edge runs store predictions under a nested subdir (e.g.
    edge/mistralai-...-yesno-graph-inject-t0.35/*.json); baseline often
    points at that leaf dir directly.
    """
    rec_jsons = [f for f in pred_dir.glob("*.json") if _REC_JSON.match(f.name)]
    if rec_jsons:
        return pred_dir, "direct"

    candidates: list[tuple[Path, int]] = []
    for sub in sorted(pred_dir.iterdir()):
        if not sub.is_dir():
            continue
        n = sum(1 for f in sub.glob("*.json") if _REC_JSON.match(f.name))
        if n:
            candidates.append((sub, n))

    if not candidates:
        return pred_dir, "no_rec_json_found"
    if len(candidates) == 1:
        sub, n = candidates[0]
        return sub, f"subdir:{sub.name} ({n} files)"
    best = max(candidates, key=lambda x: x[1])
    names = ", ".join(s.name for s, _ in candidates)
    return best[0], f"subdir:{best[0].name} ({best[1]} files; auto-picked among [{names}])"


def load_annotation(annotation_path: Path) -> tuple[dict[str, dict], dict[str, list[int]]]:
    """Load per-rec_id metadata and mast_annotation vectors (same as calculate_scores_yesno)."""
    meta: dict[str, dict] = {}
    gt: dict[str, list[int]] = {}
    with open(annotation_path) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rec_id = f"{idx:04d}"
            gt[rec_id] = [int(r["mast_annotation"].get(m, 0)) for m in MAST_MODES]
            meta[rec_id] = {
                "trace_id": r.get("trace_id"),
                "mas_name": r.get("mas_name"),
                "gt_positive_modes": [
                    m for m in MAST_MODES if int(r["mast_annotation"].get(m, 0))
                ],
                "gt_human_errors": [
                    _human_error_entry(e)
                    for e in r.get("errors", [])
                ],
            }
    return meta, gt


def _mode_from_category(category: str) -> str:
    m = re.match(r"^(\d+\.\d+)", category or "")
    return m.group(1) if m else ""


def _human_error_entry(e: dict) -> dict:
    mode = _mode_from_category(e.get("category", ""))
    return {
        "mode": mode,
        "short_name": MAST_SHORT_NAMES.get(mode, e.get("category", "")),
        "category": e.get("category", ""),
        "location": e.get("location", ""),
        "description": (e.get("description") or "")[:200],
    }


def _label_status(gt: int, pred: int) -> str:
    if gt == 1 and pred == 1:
        return "TP"
    if gt == 0 and pred == 0:
        return "TN"
    if gt == 0 and pred == 1:
        return "FP"
    return "FN"


def analyze(pred_dir: Path, annotation_path: Path) -> dict:
    pred_dir, resolution = resolve_pred_dir(pred_dir)
    meta, gt = load_annotation(annotation_path)
    repo = HERE.parent.parent

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo))
        except ValueError:
            return str(p)

    rows = []
    y_true_rows: list[list[int]] = []
    y_pred_rows: list[list[int]] = []

    for rec_id in sorted(gt.keys()):
        gt_vec = gt[rec_id]
        info = meta[rec_id]
        pred_path = pred_dir / f"{rec_id}.json"

        pred_vec: list[int] | None = None
        pred_positive: list[str] = []
        status = "ok"
        trace_id = info.get("trace_id")

        if not pred_path.exists():
            status = "missing_prediction"
            pred_vec = [0] * len(MAST_MODES)
        else:
            try:
                with open(pred_path) as f:
                    pred = json.load(f)
                predictions = pred.get("predictions", {})
                pred_vec = [int(predictions.get(m, 0)) for m in MAST_MODES]
                pred_positive = [m for m, v in zip(MAST_MODES, pred_vec) if v]
                trace_id = pred.get("trace_id", trace_id)
            except Exception as e:
                status = f"parse_error: {e}"
                pred_vec = [0] * len(MAST_MODES)

        assert pred_vec is not None
        y_true_rows.append(gt_vec)
        y_pred_rows.append(pred_vec)

        labels = []
        tp = fp = fn = tn = 0
        for i, mode in enumerate(MAST_MODES):
            g, p = gt_vec[i], pred_vec[i]
            st = _label_status(g, p)
            labels.append({
                "mode": mode,
                "name": MAST_NAMES[mode],
                "short_name": MAST_SHORT_NAMES[mode],
                "gt": g,
                "pred": p,
                "status": st,
            })
            if st == "TP":
                tp += 1
            elif st == "FP":
                fp += 1
            elif st == "FN":
                fn += 1
            else:
                tn += 1

        n_labels = len(MAST_MODES)
        label_accuracy = (tp + tn) / n_labels
        exact_match = pred_vec == gt_vec

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        gt_pos = set(info["gt_positive_modes"])
        pred_pos = set(pred_positive)
        mode_tp = sorted(gt_pos & pred_pos)
        mode_fp = sorted(pred_pos - gt_pos)
        mode_fn = sorted(gt_pos - pred_pos)

        # Modes flagged in human errors but missed by yes/no (FN on positive GT)
        human_error_modes = sorted({
            e["mode"] for e in info["gt_human_errors"] if e["mode"] in MAST_MODES
        })

        rows.append({
            "rec_id": rec_id,
            "trace_id": trace_id,
            "status": status,
            "n_labels": n_labels,
            "n_tp": tp,
            "n_fp": fp,
            "n_fn": fn,
            "n_tn": tn,
            "label_accuracy": label_accuracy,
            "exact_match": exact_match,
            "micro_f1": micro_f1,
            "gt_positive_modes": info["gt_positive_modes"],
            "pred_positive_modes": pred_positive,
            "gt_positive_short": modes_to_short_names(info["gt_positive_modes"]),
            "pred_positive_short": modes_to_short_names(pred_positive),
            "mode_tp": mode_tp,
            "mode_fp": mode_fp,
            "mode_fn": mode_fn,
            "mode_tp_short": modes_to_short_names(mode_tp),
            "mode_fp_short": modes_to_short_names(mode_fp),
            "mode_fn_short": modes_to_short_names(mode_fn),
            "gt_human_error_modes": human_error_modes,
            "gt_human_error_short": modes_to_short_names(human_error_modes),
            "gt_human_errors": info["gt_human_errors"],
            "labels": labels,
        })

    rows.sort(key=lambda r: (-r["label_accuracy"], -r["micro_f1"], r["rec_id"]))

    y_true = np.array(y_true_rows)
    y_pred = np.array(y_pred_rows)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    exact = [r for r in rows if r["exact_match"]]
    high = [r for r in rows if not r["exact_match"] and r["label_accuracy"] >= 0.85]
    mid = [r for r in rows if 0.5 <= r["label_accuracy"] < 0.85]
    low = [r for r in rows if r["label_accuracy"] < 0.5]

    return {
        "pred_dir": _rel(pred_dir.resolve()),
        "pred_dir_resolution": resolution,
        "annotation": _rel(annotation_path.resolve()),
        "mode_short_names": MAST_SHORT_NAMES,
        "n_traces": len(rows),
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "average_label_accuracy": float(np.mean([r["label_accuracy"] for r in rows])) if rows else 0.0,
        "average_micro_f1_per_trace": float(np.mean([r["micro_f1"] for r in rows])) if rows else 0.0,
        "n_exact_match": len(exact),
        "n_high_accuracy": len(high),
        "n_mid_accuracy": len(mid),
        "n_low_accuracy": len(low),
        "n_missing_prediction": sum(1 for r in rows if r["status"] == "missing_prediction"),
        "traces": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pred_dir", type=Path,
                        help="Directory of per-rec_id prediction JSON files (0000.json, ...)")
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION,
                        help=f"Holdout MAST annotation jsonl (default: {DEFAULT_ANNOTATION})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON (default: {pred_dir}-yesno_hits.json)")
    args = parser.parse_args()

    pred_dir_arg = args.pred_dir.resolve()
    annotation_path = args.annotation.resolve()
    if not pred_dir_arg.is_dir():
        sys.exit(f"pred_dir is not a directory: {pred_dir_arg}")
    if not annotation_path.is_file():
        sys.exit(f"annotation file not found: {annotation_path}")

    out_json = args.out or Path(f"{pred_dir_arg}-yesno_hits.json")
    result = analyze(pred_dir_arg, annotation_path)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {out_json}")
    if result.get("pred_dir_resolution", "").startswith("subdir:"):
        print(f"predictions resolved from: {result['pred_dir']} ({result['pred_dir_resolution']})")
    elif result.get("pred_dir_resolution") == "no_rec_json_found":
        print("warning: no 0000.json-style prediction files found under pred_dir")
    print(f"weighted F1   = {result['weighted_f1']:.4f}")
    print(f"macro F1      = {result['macro_f1']:.4f}")
    print(f"avg label acc = {result['average_label_accuracy']:.4f}")
    print(f"exact match   = {result['n_exact_match']}/{result['n_traces']} traces")
    print(f"high (acc≥85%)= {result['n_high_accuracy']}, mid= {result['n_mid_accuracy']}, "
          f"low (<50%)= {result['n_low_accuracy']}")


if __name__ == "__main__":
    main()
