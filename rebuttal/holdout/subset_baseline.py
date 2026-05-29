"""
Task #62: Subset existing full-corpus baseline predictions to the held-out
test set, per (benchmark, backbone).

Baseline (Stage-1) does not consult the graph, so its full-corpus predictions
match what we would get on a held-out re-run. We just copy the per-trace
prediction files (TRAIL) or filter the predictions jsonl (MAST) so the scorer
runs on exactly the same set as the +EDGE outputs.

Usage:
    python subset_baseline.py
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from config import BACKBONES, BENCHMARKS, ASSIGN_DIR, PREDICTIONS_DIR


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def load_json_loose(path: Path):
    """Read a prediction file, stripping ```json``` markdown fences if present."""
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("```"):
        m = _JSON_FENCE_RE.search(text)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_assignment(name: str) -> dict:
    with open(ASSIGN_DIR / f"{name}.json") as f:
        return json.load(f)


def copy_trail_predictions(src_dirs: list[Path], dst_dir: Path, held_ids: set[str]):
    """Copy {trace_id}.json from any of src_dirs into dst_dir for held traces."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copy, n_missing = 0, 0
    for tid in held_ids:
        copied = False
        for src in src_dirs:
            f = src / f"{tid}.json"
            if f.exists():
                shutil.copy2(f, dst_dir / f"{tid}.json")
                n_copy += 1
                copied = True
                break
        if not copied:
            n_missing += 1
    return n_copy, n_missing


def subset_mast_predictions(src_dir: Path, dst_dir: Path, held_ids: set[str]):
    """MAST predictions: one JSON per record. Filter records whose trace_id is held."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    held_int = {int(t) for t in held_ids}
    n_copy, n_missing = 0, 0
    if not src_dir.exists():
        return 0, 0
    # MAST baseline dirs typically contain per-record JSON files; copy any whose
    # filename starts with a held trace_id or whose content's trace_id matches.
    # Walk source files; trace_id may live in JSON content (gpt-oss) or be inferable
    # from filename ordering (mistral baseline files are wrapped in ```json``` fences,
    # named 0000.json, 0001.json etc. = position in annotation_ag2_filtered.jsonl).
    from config import MAST_ANNOTATION_FILE
    index_to_trace_id = {}
    with open(MAST_ANNOTATION_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                index_to_trace_id[i] = json.loads(line).get("trace_id")

    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.suffix != ".json":
            continue
        rec = load_json_loose(src)
        tid = None
        if isinstance(rec, dict):
            tid = rec.get("trace_id") or rec.get("rec_id")
        if tid is None:
            # fall back to filename index (mistral baseline style)
            stem = src.stem
            try:
                tid = index_to_trace_id.get(int(stem))
            except ValueError:
                pass
        try:
            tid = int(tid)
        except (ValueError, TypeError):
            continue
        if tid in held_int:
            # Strip ```json``` fences during copy so downstream scorer's json.load works.
            if isinstance(rec, dict):
                with open(dst_dir / src.name, "w") as fout:
                    json.dump(rec, fout)
            else:
                shutil.copy2(src, dst_dir / src.name)
            n_copy += 1
    return n_copy, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    ap.add_argument("--backbone", choices=list(BACKBONES.keys()) + ["all"], default="all")
    args = ap.parse_args()

    benches = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    backs = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]

    for name in benches:
        info = load_assignment(name)
        held_ids = {tid for tid, v in info["assignment"].items() if v == "test"}
        for bb in backs:
            bb_cfg = BACKBONES[bb]
            dst = PREDICTIONS_DIR / name / bb / "baseline"
            if BENCHMARKS[name]["source"] == "trail":
                srcs = bb_cfg["trail_baseline_dirs"]
                missing = [str(s) for s in srcs if not s.exists()]
                if missing:
                    print(f"  ! {bb}/{name}: missing baseline dirs: {missing}")
                    print(f"    edit BACKBONES in config.py if paths are wrong")
                n_copy, n_miss = copy_trail_predictions(srcs, dst, held_ids)
                print(f"  {name}/{bb}: copied {n_copy}/{len(held_ids)}"
                      + (f"  [WARN missing={n_miss}]" if n_miss else ""))
            else:
                src = bb_cfg["mast_baseline_dir"]
                if not src.exists():
                    print(f"  ! {bb}/{name}: missing baseline dir: {src}")
                    print(f"    edit BACKBONES in config.py if path is wrong")
                    continue
                n_copy, _ = subset_mast_predictions(src, dst, held_ids)
                print(f"  {name}/{bb}: copied {n_copy} records "
                      f"({len(held_ids)} held-out trace_ids)")


if __name__ == "__main__":
    main()
