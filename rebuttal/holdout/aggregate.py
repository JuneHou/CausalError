"""
Task #63: Score per-benchmark per-backbone predictions and emit the rebuttal
table.

For each (benchmark, backbone, variant in {baseline, edge}):
  - Runs the appropriate scorer (TRAIL or MAST) on the held-out prediction dir.
  - Parses weighted F1, Loc, Joint from the scorer's stdout.

Also rescores the FULL-CORPUS baseline + EDGE predictions on the SAME combined
trace set to produce the "full-corpus comparison" cell (this is the main
paper's numbers, just pooled across GAIA + SWE for TRAIL so it matches the
combined held-out scope).

Output:
  results/per_cell_metrics.csv
  results/rebuttal_holdout_table.tex

Usage:
    python aggregate.py
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from config import (
    BACKBONES, BENCHMARKS, MAST_ANNOTATION_FILE, MAST_ROOT, MAST_SCORER,
    PREDICTIONS_DIR, RESULTS_DIR, TRAIL_GAIA_TRACE_DIR, TRAIL_ROOT, TRAIL_SCORER,
    TRAIL_SWE_TRACE_DIR,
)


METRIC_PATTERNS = {
    "weighted_f1": [r"weighted[_ -]?f1\s*[:=]\s*([\d.]+)", r"weighted F1\s*[:=]\s*([\d.]+)"],
    "loc": [r"location[_ ]?accuracy\s*[:=]\s*([\d.]+)", r"\bLoc\b\s*[:=]\s*([\d.]+)"],
    "joint": [r"joint[_ ]?accuracy\s*[:=]\s*([\d.]+)", r"\bJoint\b\s*[:=]\s*([\d.]+)"],
}


def parse_metrics(stdout: str) -> dict:
    out = {}
    for key, patterns in METRIC_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, stdout, re.IGNORECASE)
            if m:
                out[key] = float(m.group(1))
                break
    return out


def _source_split(trace_id: str) -> str:
    """Return 'gaia' or 'swe_bench' based on which raw data dir owns the trace."""
    if (TRAIL_GAIA_TRACE_DIR / f"{trace_id}.json").exists():
        return "gaia"
    if (TRAIL_SWE_TRACE_DIR / f"{trace_id}.json").exists():
        return "swe_bench"
    return "unknown"


def _import_trail_scorer():
    """Import the TRAIL scorer's main() so we can call it on subset GT directly."""
    sys.path.insert(0, str(TRAIL_ROOT / "benchmarking" / "eval"))
    import calculate_scores as _trail
    return _trail.main


def score_trail(pred_dir: Path) -> dict:
    """Per-split: stage subset GT + preds, call scorer's main(), pool by N."""
    trail_main = _import_trail_scorer()
    bench_dir = TRAIL_ROOT / "benchmarking"

    per_split_metrics: dict[str, dict] = {}
    per_split_n: dict[str, int] = {}

    for f in pred_dir.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        split = _source_split(f.stem)
        if split == "unknown":
            continue
        per_split_n[split] = per_split_n.get(split, 0) + 1

    for split in per_split_n:
        gt_src = bench_dir / f"processed_annotations_{split}"
        if not gt_src.exists():
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"trail_score_{split}_"))
        gt_sub = tmp / "gt"
        pred_sub = tmp / "pred"
        gt_sub.mkdir()
        pred_sub.mkdir()
        for f in pred_dir.iterdir():
            if not f.is_file() or f.suffix != ".json":
                continue
            if _source_split(f.stem) != split:
                continue
            gt_file = gt_src / f.name
            if not gt_file.exists():
                continue
            shutil.copy2(f, pred_sub / f.name)
            shutil.copy2(gt_file, gt_sub / f.name)
        cwd = Path.cwd()
        try:
            os.chdir(bench_dir)  # scorer uses cwd-relative paths internally
            res = trail_main(ground_truth_dir=str(gt_sub), generated_dir=str(pred_sub))
        finally:
            os.chdir(cwd)
        if not res:
            continue
        per_split_metrics[split] = {
            "weighted_f1": res.get("weighted_f1"),
            "loc": res.get("location_accuracy"),
            "joint": res.get("joint_accuracy"),
        }

    if not per_split_metrics:
        return {}

    pooled = {}
    for metric in ("weighted_f1", "loc", "joint"):
        num, den = 0.0, 0
        for split, m in per_split_metrics.items():
            v = m.get(metric)
            if v is not None:
                num += v * per_split_n[split]
                den += per_split_n[split]
        if den:
            pooled[metric] = num / den
    return pooled


def score_mast(pred_dir: Path) -> dict:
    cmd = [
        sys.executable, str(MAST_SCORER),
        "--annotation", str(MAST_ANNOTATION_FILE),
        "--pred_dir", str(pred_dir),
    ]
    r = subprocess.run(cmd, cwd=MAST_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! MAST scorer failed: {r.stderr[:200]}")
        return {}
    return parse_metrics(r.stdout)


def pool_trail_full_corpus(backbone_key: str, variant: str) -> Path:
    """For the full-corpus reference number, pool baseline (or +EDGE) predictions
    from GAIA + SWE dirs into a temp dir and let TRAIL scorer chew on them."""
    bb = BACKBONES[backbone_key]
    if variant == "baseline":
        srcs = bb["trail_baseline_dirs"]
    else:
        # +EDGE full-corpus dirs follow the naming pattern from main paper runs.
        # User can override here if needed.
        srcs = []
        for split in ["GAIA_dedup", "SWE_Bench_dedup"]:
            candidate = (TRAIL_ROOT / "benchmarking" / "outputs" / "outputs_thres" /
                         "t0.35" / f"outputs_{bb['hf_name'].replace('/', '-')}-{split}")
            if candidate.exists():
                srcs.append(candidate)
    tmp = Path(tempfile.mkdtemp(prefix=f"trail_full_{backbone_key}_{variant}_"))
    n = 0
    for src in srcs:
        if not src.exists():
            continue
        for f in src.iterdir():
            if f.is_file() and f.suffix == ".json":
                shutil.copy2(f, tmp / f.name)
                n += 1
    return tmp if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_full_corpus", action="store_true",
                    help="Skip rescoring main-paper full-corpus numbers")
    args = ap.parse_args()

    rows = []  # dicts with benchmark/backbone/variant/scope/metrics
    for name in BENCHMARKS:
        cfg = BENCHMARKS[name]
        scorer = score_mast if cfg["source"] == "mast" else score_trail
        for bb in BACKBONES:
            for variant in ("baseline", "edge"):
                # held-out cell
                pred_dir = PREDICTIONS_DIR / name / bb / variant
                if pred_dir.exists() and any(pred_dir.iterdir()):
                    print(f"  scoring held-out {name}/{bb}/{variant}")
                    m = scorer(pred_dir)
                    rows.append({"benchmark": name, "backbone": bb,
                                 "variant": variant, "scope": "holdout", **m})
                # full-corpus reference cell (TRAIL needs pooling; MAST is one dir)
                if args.skip_full_corpus:
                    continue
                if cfg["source"] == "trail":
                    pooled = pool_trail_full_corpus(bb, variant)
                    if pooled:
                        print(f"  scoring full-corpus {name}/{bb}/{variant}")
                        m = score_trail(pooled)
                        rows.append({"benchmark": name, "backbone": bb,
                                     "variant": variant, "scope": "full", **m})
                else:
                    src = (BACKBONES[bb]["mast_baseline_dir"] if variant == "baseline"
                           else None)
                    if src and src.exists():
                        print(f"  scoring full-corpus {name}/{bb}/{variant}")
                        m = score_mast(src)
                        rows.append({"benchmark": name, "backbone": bb,
                                     "variant": variant, "scope": "full", **m})

    if not rows:
        print("No predictions found. Run earlier tasks first.")
        return

    # Write CSV
    csv_path = RESULTS_DIR / "per_cell_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "backbone", "variant", "scope",
              "weighted_f1", "loc", "joint"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {csv_path}")

    # Build LaTeX table
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["benchmark"], r["backbone"])][(r["scope"], r["variant"])] = r

    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l cc c cc c}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Full corpus}} & "
        r"\multicolumn{3}{c}{\textbf{Held-out 20\%}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"\textbf{Benchmark} & \textbf{Backbone} & "
        r"Baseline & +EDGE & $\Delta$ & "
        r"Baseline & +EDGE & $\Delta$ \\",
        r"\midrule",
    ]
    for name in BENCHMARKS:
        for bb in BACKBONES:
            c = by_cell.get((name, bb), {})
            def f1(scope, var):
                return c.get((scope, var), {}).get("weighted_f1")
            fb, fe = f1("full", "baseline"), f1("full", "edge")
            hb, he = f1("holdout", "baseline"), f1("holdout", "edge")
            fmt = lambda v: f"{v:.2f}" if v is not None else "--"
            dfull = f"{fe - fb:+.2f}" if fb is not None and fe is not None else "--"
            dhold = f"{he - hb:+.2f}" if hb is not None and he is not None else "--"
            lines.append(
                f"{name.upper()} & {bb} & {fmt(fb)} & {fmt(fe)} & {dfull} & "
                f"{fmt(hb)} & {fmt(he)} & {dhold} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Held-out 80/20 weighted F1. The graph $\mathcal{G}_\tau$ is rebuilt "
        r"from the 80\% training-side traces only (corr-union at $\tau{=}0.35$, no "
        r"validated $\mathcal{G}_V$ edges) and evaluated on the held-out 20\%. "
        r"Full-corpus numbers are rescored from the main-paper runs on the same combined trace set.}",
        r"\label{tab:rebuttal_holdout}",
        r"\end{table}",
    ]
    tex_path = RESULTS_DIR / "rebuttal_holdout_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
