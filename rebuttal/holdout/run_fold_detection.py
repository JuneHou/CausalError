"""
Task #61: +EDGE Stage-2 detection runs on the held-out test set per benchmark.

For each (benchmark, backbone):
  - TRAIL: symlinks the held-out trace JSONs (from GAIA_dedup or
    SWE_Bench_dedup, looked up by trace_id) into a single test dir, then
    invokes run_eval_graph_inject.py with --data_dir + --split pointing to it,
    plus --suppes_graph + --causal_graph from the per-benchmark holdout graph.
  - MAST: writes a filtered annotation jsonl containing only records whose
    trace_id is on the held-out side, then invokes the MAST eval script.

Baseline (Stage-1) does not use the graph; see subset_baseline.py for the
matched baseline-prediction subset.

Usage:
    python run_fold_detection.py
    python run_fold_detection.py --benchmark trail --backbone mistral-24b
    python run_fold_detection.py --dry_run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import (
    BACKBONES, BENCHMARKS, CORR_THRESHOLD, ASSIGN_DIR, TEST_TRACE_DIR,
    GRAPHS_DIR, MAST_ANNOTATION_FILE, MAST_EVAL_SCRIPT_API, MAST_EVAL_SCRIPT_VLLM,
    MAST_ROOT, PREDICTIONS_DIR, TRAIL_EVAL_SCRIPT_API, TRAIL_EVAL_SCRIPT_VLLM,
    TRAIL_GAIA_TRACE_DIR, TRAIL_ROOT, TRAIL_SWE_TRACE_DIR,
)


def load_assignment(name: str) -> dict:
    with open(ASSIGN_DIR / f"{name}.json") as f:
        return json.load(f)


def stage_trail_test_traces(held_ids: set[str]) -> Path:
    """Symlink TRAIL held-out trace JSONs into a single dir; return parent data_dir."""
    parent = TEST_TRACE_DIR / "trail"
    test_dir = parent / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for f in test_dir.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()
    n_found, n_missing = 0, 0
    for tid in held_ids:
        # Try GAIA first, then SWE.
        src = TRAIL_GAIA_TRACE_DIR / f"{tid}.json"
        if not src.exists():
            src = TRAIL_SWE_TRACE_DIR / f"{tid}.json"
        if not src.exists():
            n_missing += 1
            print(f"  ! missing trace file for {tid}")
            continue
        (test_dir / f"{tid}.json").symlink_to(src)
        n_found += 1
    print(f"  staged {n_found} test traces ({n_missing} missing)")
    return parent  # eval --data_dir; --split = "test"


def stage_mast_test_jsonl(held_ids: set[str]) -> Path:
    parent = TEST_TRACE_DIR / "mast"
    parent.mkdir(parents=True, exist_ok=True)
    out_path = parent / "test.jsonl"
    held_int = {int(t) for t in held_ids}
    n_records = 0
    with open(MAST_ANNOTATION_FILE) as fin, open(out_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("trace_id") in held_int:
                fout.write(line)
                n_records += 1
    print(f"  staged {n_records} MAST test records ({len(held_ids)} unique trace_ids)")
    return out_path


def run_trail(backbone_key: str, suppes_path: Path, causal_path: Path,
              data_dir: Path, output_dir: Path, dry_run: bool) -> bool:
    bb = BACKBONES[backbone_key]
    if bb["backend"] == "vllm":
        cmd = [
            sys.executable, str(TRAIL_EVAL_SCRIPT_VLLM),
            "--model", bb["hf_name"],
            "--split", "test",
            "--data_dir", str(data_dir),
            "--corr_threshold", str(CORR_THRESHOLD),
            "--span_index",
            "--output_dir", str(output_dir),
            "--suppes_graph", str(suppes_path),
            "--causal_graph", str(causal_path),
            "--tensor_parallel_size", str(bb["tensor_parallel_size"]),
            "--gpu_memory_utilization", str(bb["gpu_memory_utilization"]),
        ]
        label = "TRAIL [vllm]"
    else:
        cmd = [
            sys.executable, str(TRAIL_EVAL_SCRIPT_API),
            "--model", bb["hf_name"],
            "--split", "test",
            "--data_dir", str(data_dir),
            "--corr_threshold", str(CORR_THRESHOLD),
            "--span_index",
            "--output_dir", str(output_dir),
            "--suppes_graph", str(suppes_path),
            "--causal_graph", str(causal_path),
        ]
        label = "TRAIL [arc]"
    print(f"  {label} >", " ".join(cmd))
    if dry_run:
        return True
    return subprocess.run(cmd, cwd=TRAIL_ROOT / "benchmarking").returncode == 0


def run_mast(backbone_key: str, suppes_path: Path, effect_path: Path,
             input_path: Path, output_dir: Path, dry_run: bool) -> bool:
    bb = BACKBONES[backbone_key]
    script = MAST_EVAL_SCRIPT_VLLM if bb["backend"] == "vllm" else MAST_EVAL_SCRIPT_API
    cmd = [
        sys.executable, str(script),
        "--model", bb["hf_name"],
        "--input", str(input_path),
        "--output_dir", str(output_dir),
        "--edge_threshold", str(CORR_THRESHOLD),
        "--suppes_graph", str(suppes_path),
        "--effect_edges", str(effect_path),
    ]
    if bb["backend"] == "vllm":
        if bb.get("tensor_parallel_size"):
            cmd += ["--tp", str(bb["tensor_parallel_size"])]
        if bb.get("gpu_memory_utilization"):
            cmd += ["--gpu_memory_utilization", str(bb["gpu_memory_utilization"])]
    # ARC scripts honor an internal rate-limiter (30 req/min) and take no concurrency flag.
    label = "MAST  [vllm]" if bb["backend"] == "vllm" else "MAST  [arc]"
    print(f"  {label} >", " ".join(cmd))
    if dry_run:
        return True
    return subprocess.run(cmd, cwd=MAST_ROOT).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    ap.add_argument("--backbone", choices=list(BACKBONES.keys()) + ["all"], default="all")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    benches = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    backs = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]

    for name in benches:
        cfg = BENCHMARKS[name]
        info = load_assignment(name)
        held_ids = {tid for tid, v in info["assignment"].items() if v == "test"}
        if not held_ids:
            print(f"  ! {name}: empty test set, skipping")
            continue

        suppes_path = GRAPHS_DIR / name / "suppes_graph.json"
        effect_path = GRAPHS_DIR / name / "effect_edges.json"
        if not suppes_path.exists():
            print(f"  ! missing {suppes_path}, run build_fold_graphs.py first")
            continue

        print(f"\n=== {name} | n_test_unique={len(held_ids)} ===")
        if cfg["source"] == "trail":
            data_dir = stage_trail_test_traces(held_ids)
        else:
            input_path = stage_mast_test_jsonl(held_ids)

        for bb in backs:
            out_dir = PREDICTIONS_DIR / name / bb / "edge"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n-- {name} | {bb} | EDGE Stage-2 --")
            if cfg["source"] == "trail":
                ok = run_trail(bb, suppes_path, effect_path, data_dir, out_dir, args.dry_run)
            else:
                ok = run_mast(bb, suppes_path, effect_path, input_path, out_dir, args.dry_run)
            if not ok:
                print(f"  ! detection failed for {name}/{bb}")


if __name__ == "__main__":
    main()
