"""
Per-trace location-accuracy breakdown for a prediction output dir.

Mirrors the location-accuracy computation in
benchmarking/eval/calculate_scores.py: for each ground-truth trace,

    loc_acc = |set(gt_locations) ∩ set(pred_locations)| / |set(gt_locations)|

A trace is "fully correct" if loc_acc == 1.0, "partially correct" if 0 < loc_acc < 1,
and "missed" if loc_acc == 0 (or prediction file is missing/unparseable).

Output files are written next to the script (baselines/outputs/) regardless
of where the prediction dir lives, named after the prediction dir.

Usage:
    python _dump_location_hits.py [PRED_DIR]
PRED_DIR defaults to the Mistral W2 baseline. Split (gaia/swe_bench) is
inferred from the dir name, matching calculate_scores.py.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_PRED = HERE / "outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-GAIA-who_and_when_w2"

PRED_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_PRED
_name_lower = PRED_DIR.name.lower()
if "swe_bench" in _name_lower or "swe-bench" in _name_lower or "swe bench" in _name_lower:
    GT_DIR = REPO / "benchmarking" / "processed_annotations_swe_bench"
else:
    GT_DIR = REPO / "benchmarking" / "processed_annotations_gaia"
OUT_JSON = HERE / f"{PRED_DIR.name}-location_hits.json"
OUT_TXT = HERE / f"{PRED_DIR.name}-location_hits.txt"


def extract_json_from_text(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    s = m.group(0)
    while len(s) > 2:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s[:-1]
    raise ValueError("could not parse JSON")


def locations_of(obj) -> list[str]:
    return [e.get("location", "") for e in obj.get("errors", [])]


def main() -> None:
    rows = []
    for gt_path in sorted(glob.glob(str(GT_DIR / "*.json"))):
        trace_id = os.path.splitext(os.path.basename(gt_path))[0]
        if trace_id.startswith("_meta_"):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        gt_locs = set(locations_of(gt))

        pred_path = PRED_DIR / f"{trace_id}.json"
        pred_locs: set[str] = set()
        status = "ok"
        if not pred_path.exists():
            status = "missing_prediction"
        else:
            try:
                with open(pred_path) as f:
                    pred = extract_json_from_text(f.read())
                pred_locs = set(locations_of(pred))
            except Exception as e:
                status = f"parse_error: {e}"

        if gt_locs:
            hit = gt_locs & pred_locs
            loc_acc = len(hit) / len(gt_locs)
        else:
            hit = set()
            loc_acc = 0.0

        rows.append({
            "trace_id": trace_id,
            "status": status,
            "n_gt_locations": len(gt_locs),
            "n_pred_locations": len(pred_locs),
            "n_correct_locations": len(hit),
            "location_accuracy": loc_acc,
            "gt_locations": sorted(gt_locs),
            "pred_locations": sorted(pred_locs),
            "correct_locations": sorted(hit),
        })

    rows.sort(key=lambda r: (-r["location_accuracy"], -r["n_correct_locations"], r["trace_id"]))

    full = [r for r in rows if r["location_accuracy"] == 1.0 and r["n_gt_locations"] > 0]
    partial = [r for r in rows if 0 < r["location_accuracy"] < 1.0]
    missed = [r for r in rows if r["location_accuracy"] == 0.0]
    avg = sum(r["location_accuracy"] for r in rows) / len(rows) if rows else 0.0

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "pred_dir": _rel(PRED_DIR),
            "gt_dir": _rel(GT_DIR),
            "n_traces": len(rows),
            "average_location_accuracy": avg,
            "n_full_match": len(full),
            "n_partial_match": len(partial),
            "n_missed": len(missed),
            "traces": rows,
        }, f, indent=2)

    with open(OUT_TXT, "w") as f:
        f.write(f"Pred dir: {PRED_DIR.name}\n")
        f.write(f"GT dir:   {_rel(GT_DIR)}\n")
        f.write(f"Traces:   {len(rows)}\n")
        f.write(f"Avg location accuracy: {avg:.4f}\n")
        f.write(f"Full-match traces:    {len(full)}\n")
        f.write(f"Partial-match traces: {len(partial)}\n")
        f.write(f"Missed traces:        {len(missed)}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"FULL MATCH (loc_acc == 1.0)  —  {len(full)} traces\n")
        f.write("=" * 80 + "\n")
        for r in full:
            f.write(f"{r['trace_id']}  gt={r['n_gt_locations']}  pred={r['n_pred_locations']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"PARTIAL MATCH (0 < loc_acc < 1)  —  {len(partial)} traces\n")
        f.write("=" * 80 + "\n")
        for r in partial:
            f.write(
                f"{r['trace_id']}  acc={r['location_accuracy']:.3f}  "
                f"hit={r['n_correct_locations']}/{r['n_gt_locations']}  "
                f"pred={r['n_pred_locations']}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"MISSED (loc_acc == 0)  —  {len(missed)} traces\n")
        f.write("=" * 80 + "\n")
        for r in missed:
            tag = "" if r["status"] == "ok" else f"  [{r['status']}]"
            f.write(
                f"{r['trace_id']}  gt={r['n_gt_locations']}  pred={r['n_pred_locations']}{tag}\n"
            )

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TXT}")
    print(f"avg loc acc = {avg:.4f}  (full={len(full)}, partial={len(partial)}, missed={len(missed)})")


if __name__ == "__main__":
    main()
