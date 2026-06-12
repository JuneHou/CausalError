"""
R3 aggregator — score the 60 temp=0.7 prediction dirs against full-corpus GT,
plus the existing temp=0 anchor dirs, and emit per-cell mean ± std + Delta.

For each (backbone, benchmark, variant):
  - score temp=0.7_sample{1,2,3} prediction dirs
  - score the existing temp=0 Table-1 prediction dir as anchor
  - compute mean ± std across the 3 samples and the delta vs baseline

TRAIL: per-split scoring (GAIA / SWE), pool by N. MAST: single annotation file.

Usage:
    python aggregate.py                       # score everything that exists
    python aggregate.py --skip_anchor         # only score temp=0.7 reruns
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path

from config import (
    BACKBONES, BENCHMARKS, TEMPERATURE, N_SAMPLES,
    PREDICTIONS_DIR, RESULTS_DIR, TRAIL_ROOT, MAST_ROOT,
    TRAIL_SCORER, MAST_SCORER, MAST_ANNOTATION,
    MAST_EDGE_THRESHOLD,
)


# ----------------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------------

def _import_trail_scorer():
    sys.path.insert(0, str(TRAIL_ROOT / "benchmarking" / "eval"))
    import calculate_scores as _t
    return _t.main


def score_trail(pred_dir: Path) -> dict:
    """pred_dir is expected to contain per-split subdirs (GAIA_dedup, SWE_Bench_dedup)
    where each split holds {trace_id}.json prediction files. Returns per-split metrics
    {split_name: {weighted_f1, loc, joint}} — GAIA and SWE kept SEPARATE to match the
    main results table. (Pooling GAIA+SWE is the held-out experiment's convention, not
    the randomized-variance table.)"""
    trail_main = _import_trail_scorer()
    bench_dir = TRAIL_ROOT / "benchmarking"
    per_split_metrics: dict[str, dict] = {}
    per_split_n: dict[str, int] = {}

    for split_name in ("GAIA_dedup", "SWE_Bench_dedup"):
        # The eval scripts write into pred_dir/outputs_<model_tag>-<split>[-suffix]/
        # (run_r3.py also pre-creates an empty pred_dir/<split>/ placeholder). Find
        # the outputs_*<split>* subdir that actually contains json files.
        candidates = [
            d for d in pred_dir.glob(f"outputs_*{split_name}*")
            if d.is_dir() and any(f for f in d.glob("*.json") if not f.name.startswith("_"))
        ]
        if not candidates:
            # Fall back to the bare split dir if someone wrote files there directly.
            bare = pred_dir / split_name
            if bare.exists() and any(f for f in bare.glob("*.json") if not f.name.startswith("_")):
                split_dir = bare
            else:
                continue
        else:
            split_dir = candidates[0]
        gt_key = "gaia" if "GAIA" in split_name else "swe_bench"
        gt_src = bench_dir / f"processed_annotations_{gt_key}"
        if not gt_src.exists():
            print(f"  [trail] no GT dir at {gt_src}; skipping {split_name}")
            continue
        # Count pred files
        pred_files = [f for f in split_dir.glob("*.json") if not f.name.startswith("_")]
        if not pred_files:
            continue
        per_split_n[split_name] = len(pred_files)
        # Stage GT + pred subset into a temp dir, then invoke scorer's main()
        tmp = Path(tempfile.mkdtemp(prefix=f"trail_score_{split_name}_"))
        gt_sub = tmp / "gt"; pred_sub = tmp / "pred"
        gt_sub.mkdir(); pred_sub.mkdir()
        for f in pred_files:
            gt_file = gt_src / f.name
            if not gt_file.exists():
                continue
            shutil.copy2(f, pred_sub / f.name)
            shutil.copy2(gt_file, gt_sub / f.name)
        cwd = Path.cwd()
        try:
            os.chdir(bench_dir)
            res = trail_main(ground_truth_dir=str(gt_sub), generated_dir=str(pred_sub))
        finally:
            os.chdir(cwd)
        if not res:
            continue
        per_split_metrics[split_name] = {
            "weighted_f1": res.get("weighted_f1"),
            "loc":         res.get("location_accuracy"),
            "joint":       res.get("joint_accuracy"),
        }
    # Return per-split metrics (GAIA / SWE kept separate to match the main results
    # table). per_split_n is retained above for completeness checks but no longer
    # pooled — the randomized-variance table reports each split as its own cell.
    return per_split_metrics


def score_mast(pred_dir: Path) -> dict:
    """Resolve a single layer of nesting if the eval script wrote
    <pred_dir>/<model_tag>-yesno-...-{baseline,graph-inject-tau}/{rec_id}.json."""
    target = pred_dir
    rec_files = [f for f in target.glob("*.json") if f.name[:4].isdigit()]
    if not rec_files:
        subdirs = [d for d in target.iterdir()
                   if d.is_dir() and list(d.glob("[0-9][0-9][0-9][0-9].json"))]
        # A superseded edge run (e.g. the wrong-tau t0.35) can sit beside the
        # correct one, and iterdir() order is arbitrary -- prefer the subdir
        # tagged with the configured MAST edge threshold so we score t0.5, not
        # whichever happens to come first. Baseline dirs carry no such tag and
        # fall through to the single available subdir.
        edge_tag = f"graph-inject-t{MAST_EDGE_THRESHOLD:g}"
        preferred = [d for d in subdirs if edge_tag in d.name]
        chosen = preferred or subdirs
        if chosen:
            target = chosen[0]
    # Scorer writes a -metrics.json next to pred_dir; parse it directly.
    metrics_file = Path(f"{target}-metrics.json")
    # On a local checkout the CausalMAST scorer is absent, but the -metrics.json
    # produced during the run is committed alongside the predictions — reuse it.
    if MAST_SCORER.exists():
        cmd = [
            sys.executable, str(MAST_SCORER),
            "--annotation", str(MAST_ANNOTATION),
            "--pred_dir", str(target),
        ]
        r = subprocess.run(cmd, cwd=MAST_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [mast] scorer failed on {target}: {r.stderr[:200]}")
            if not metrics_file.exists():
                return {}
    elif not metrics_file.exists():
        print(f"  [mast] no scorer and no precomputed metrics for {target}")
        return {}
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        m = json.load(f)
    return {
        "weighted_f1": m.get("weighted_f1"),
        "macro_f1":    m.get("macro_f1"),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _trail_sample_complete(pred_dir: Path) -> bool:
    """Sample is complete iff both GAIA (117) and SWE (31) splits have full trace files."""
    EXPECTED = {"GAIA_dedup": 117, "SWE_Bench_dedup": 31}
    for split, n_expected in EXPECTED.items():
        cands = [
            d for d in pred_dir.glob(f"outputs_*{split}*")
            if d.is_dir() and any(f for f in d.glob("*.json") if not f.name.startswith("_"))
        ]
        if not cands:
            return False
        n_have = len([f for f in cands[0].glob("*.json") if not f.name.startswith("_")])
        if n_have < n_expected:
            return False
    return True


def _mast_sample_complete(pred_dir: Path) -> bool:
    """Sample is complete iff the MAST subdir has all 393 trace files."""
    EXPECTED = 393
    cands = [d for d in pred_dir.iterdir() if d.is_dir()] if pred_dir.exists() else []
    for sub in cands:
        n = len(list(sub.glob("[0-9][0-9][0-9][0-9].json")))
        if n >= EXPECTED:
            return True
    # Top-level fallback
    return len(list(pred_dir.glob("[0-9][0-9][0-9][0-9].json"))) >= EXPECTED


def _cell_complete(benchmark: str, bb_key: str, variant: str) -> bool:
    """All 3 samples present AND fully populated."""
    checker = _trail_sample_complete if benchmark == "trail" else _mast_sample_complete
    for sample_idx in range(1, N_SAMPLES + 1):
        pred_dir = PREDICTIONS_DIR / benchmark / bb_key / variant / f"temp{TEMPERATURE}_sample{sample_idx}"
        if not pred_dir.exists() or not checker(pred_dir):
            return False
    return True


# Reject a "mean" row whose 3 samples are byte-identical (σ ≈ 0 from
# unpatched seed) — that's not a variance estimate.
SIGMA_FLOOR = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_anchor", action="store_true",
                    help="Skip rescoring the existing temp=0 Table-1 prediction dirs.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Backbone keys to skip entirely (e.g. --exclude gemma-3-27b "
                         "for reruns still in flight).")
    args = ap.parse_args()

    bb_keys = [k for k in BACKBONES if k not in args.exclude]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []  # (bb, bench, variant, reason)

    # Report cells: TRAIL is split into GAIA / SWE to match the main results table;
    # MAST stays a single corpus.
    TRAIL_SPLIT_LABEL = {"GAIA_dedup": "trail-GAIA", "SWE_Bench_dedup": "trail-SWE"}

    for bb_key in bb_keys:
        for benchmark in BENCHMARKS:
            for variant in ("baseline", "edge"):
                if not _cell_complete(benchmark, bb_key, variant):
                    skipped.append((bb_key, benchmark, variant, "incomplete: missing samples or split"))
                    continue
                # Per-sample weighted-F1, keyed by report cell. TRAIL yields one
                # entry per split; MAST a single "mast" entry.
                cell_f1s = defaultdict(list)
                for sample_idx in range(1, N_SAMPLES + 1):
                    pred_dir = PREDICTIONS_DIR / benchmark / bb_key / variant / f"temp{TEMPERATURE}_sample{sample_idx}"
                    if benchmark == "trail":
                        split_metrics = score_trail(pred_dir)   # {split_name: {weighted_f1, loc, joint}}
                        for split_name, m in split_metrics.items():
                            label = TRAIL_SPLIT_LABEL.get(split_name, f"trail-{split_name}")
                            rows.append({
                                "backbone": bb_key, "benchmark": label, "variant": variant,
                                "scope": f"temp{TEMPERATURE}_sample{sample_idx}", **m,
                            })
                            if m.get("weighted_f1") is not None:
                                cell_f1s[label].append(m["weighted_f1"])
                    else:
                        m = score_mast(pred_dir)
                        if not m:
                            continue
                        rows.append({
                            "backbone": bb_key, "benchmark": "mast", "variant": variant,
                            "scope": f"temp{TEMPERATURE}_sample{sample_idx}", **m,
                        })
                        if m.get("weighted_f1") is not None:
                            cell_f1s["mast"].append(m["weighted_f1"])
                # mean±σ per report cell
                for label, f1s in cell_f1s.items():
                    if len(f1s) < N_SAMPLES:
                        skipped.append((bb_key, label, variant,
                                        f"scorer produced {len(f1s)}/{N_SAMPLES} samples"))
                        continue
                    sd = statistics.stdev(f1s)
                    if sd < SIGMA_FLOOR:
                        skipped.append((bb_key, label, variant,
                                        f"sigma={sd:.2e} below floor (seed didn't vary)"))
                        continue
                    rows.append({
                        "backbone": bb_key, "benchmark": label, "variant": variant,
                        "scope":      f"temp{TEMPERATURE}_mean",
                        "weighted_f1": statistics.mean(f1s),
                        "weighted_f1_std": sd,
                        "n_samples":  len(f1s),
                    })

    csv_path = RESULTS_DIR / "r3_per_cell_metrics.csv"
    fields = ["backbone", "benchmark", "variant", "scope",
              "weighted_f1", "weighted_f1_std", "n_samples", "loc", "joint", "macro_f1"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {csv_path}")

    # ------- Build a compact LaTeX table: per (bb, bench) report Delta-mean and std -------
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["backbone"], r["benchmark"])][(r["variant"], r["scope"])] = r

    lines = [
        r"\begin{table}[!tbp]",
        r"\centering \footnotesize",
        r"\begin{tabular}{l l c c c}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Benchmark} & "
        r"Base $\mu\pm\sigma$ & +EDGE $\mu\pm\sigma$ & $\Delta$ \\",
        r"\midrule",
    ]
    emitted = 0
    REPORT_CELLS = ["trail-GAIA", "trail-SWE", "mast"]
    for bb_key in bb_keys:
        for benchmark in REPORT_CELLS:
            c = by_cell.get((bb_key, benchmark), {})
            mb = c.get(("baseline", f"temp{TEMPERATURE}_mean"))
            me = c.get(("edge",     f"temp{TEMPERATURE}_mean"))
            # Both baseline AND edge must have passed the completeness + sigma check
            if mb is None or me is None:
                continue
            base_mu, base_sd = mb["weighted_f1"], mb["weighted_f1_std"]
            edge_mu, edge_sd = me["weighted_f1"], me["weighted_f1_std"]
            d = edge_mu - base_mu
            lines.append(
                f"{bb_key} & {benchmark.upper()} & "
                f"{base_mu*100:.1f}$\\pm${base_sd*100:.1f} & "
                f"{edge_mu*100:.1f}$\\pm${edge_sd*100:.1f} & "
                f"{d*100:+.1f} \\\\"
            )
            emitted += 1
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{R3 temperature robustness. Weighted F1 (\%) at "
        f"temperature {TEMPERATURE} averaged across {N_SAMPLES} i.i.d. samples.}}",
        r"\label{tab:r3_temperature}",
        r"\end{table}",
    ]
    tex_path = RESULTS_DIR / "r3_temperature_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {tex_path}  ({emitted} cells emitted)")

    if skipped:
        print(f"\nSkipped {len(skipped)} (backbone, benchmark, variant) cells:")
        for bb, bm, va, reason in skipped:
            print(f"  - {bb}/{bm}/{va}: {reason}")


if __name__ == "__main__":
    main()
