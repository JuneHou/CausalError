"""
R3 temperature-robustness dispatcher.

For each (backbone, benchmark, variant, sample_idx in 1..N_SAMPLES):
  - vllm backbones -> emit an sbatch command (one job per cell; the job loops
    N_SAMPLES samples internally to amortize the model load).
  - api backbones  -> launch the Python eval directly as a background process
    against the ARC endpoint.

This script does NOT submit anything by default; it prints the commands so the
user can review and then pipe to `sbatch`/`bash` (or pass --execute).

Usage:
    python run_r3.py                        # print all dispatched commands
    python run_r3.py --backbone mistral-24b # one backbone
    python run_r3.py --execute              # actually submit / launch
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import (
    BACKBONES, BENCHMARKS, TEMPERATURE, N_SAMPLES, CORR_THRESHOLD,
    R3_ROOT, TRAIL_ROOT, MAST_ROOT, PREDICTIONS_DIR,
    TRAIL_BASELINE_SCRIPT_VLLM, TRAIL_BASELINE_SCRIPT_API,
    TRAIL_EDGE_SCRIPT_VLLM, TRAIL_EDGE_SCRIPT_ARC,
    MAST_BASELINE_SCRIPT_VLLM, MAST_BASELINE_SCRIPT_API,
    MAST_EDGE_SCRIPT_VLLM, MAST_EDGE_SCRIPT_ARC,
    output_dir,
)

SBATCH_TEMPLATE = R3_ROOT / "sbatch" / "r3_template.sbatch"


def _trail_vllm_cmd(bb, variant: str, sample_idx: int, split: str) -> list[str]:
    """Build a vLLM TRAIL eval command for one (backbone, variant, split, sample)."""
    out = output_dir("trail", _bb_key(bb), variant, sample_idx, split=split)
    out.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        str(TRAIL_EDGE_SCRIPT_VLLM if variant == "edge" else TRAIL_BASELINE_SCRIPT_VLLM),
        "--model", bb["hf_name"],
        "--split", split,
        "--data_dir", str(TRAIL_ROOT / "benchmarking" / "data"),
        "--output_dir", str(out.parent),     # script appends a model_tag subdir
        "--temperature", str(TEMPERATURE),
        "--tensor_parallel_size", str(bb["tensor_parallel_size"]),
        "--gpu_memory_utilization", str(bb["gpu_memory_utilization"]),
        "--max_model_len", str(bb["max_model_len_trail"]),
    ]
    if variant == "edge":
        base += [
            "--corr_threshold", str(CORR_THRESHOLD),
            "--span_index",
            "--suppes_graph", BENCHMARKS["trail"]["suppes_graph"],
            "--causal_graph", BENCHMARKS["trail"]["effect_edges"],
        ]
    return base


def _trail_arc_cmd(bb, variant: str, sample_idx: int, split: str) -> list[str]:
    out = output_dir("trail", _bb_key(bb), variant, sample_idx, split=split)
    out.mkdir(parents=True, exist_ok=True)
    if variant == "baseline":
        # TRAIL baseline via litellm (run_eval.py). Caller must configure
        # OPENAI_API_BASE + OPENAI_API_KEY (or equivalent) to route gpt-oss-120b
        # to the ARC endpoint.
        return [
            sys.executable,
            str(TRAIL_BASELINE_SCRIPT_API),
            "--model", bb["hf_name"],
            "--split", split,
            "--data_dir", str(TRAIL_ROOT / "benchmarking" / "data"),
            "--output_dir", str(out.parent),
            "--temperature", str(TEMPERATURE),
        ]
    return [
        sys.executable,
        str(TRAIL_EDGE_SCRIPT_ARC),
        "--model", bb["hf_name"],
        "--model_tag", bb["model_tag"],
        "--split", split,
        "--data_dir", str(TRAIL_ROOT / "benchmarking" / "data"),
        "--output_dir", str(out.parent),
        "--temperature", str(TEMPERATURE),
        "--corr_threshold", str(CORR_THRESHOLD),
        "--span_index",
        "--suppes_graph", BENCHMARKS["trail"]["suppes_graph"],
        "--causal_graph", BENCHMARKS["trail"]["effect_edges"],
    ]


def _mast_vllm_cmd(bb, variant: str, sample_idx: int) -> list[str]:
    out = output_dir("mast", _bb_key(bb), variant, sample_idx)
    out.mkdir(parents=True, exist_ok=True)
    if variant == "edge":
        return [
            sys.executable, str(MAST_EDGE_SCRIPT_VLLM),
            "--model", bb["hf_name"],
            "--model_tag", bb["model_tag"],
            "--input", str(BENCHMARKS["mast"]["input"]) if False else BENCHMARKS["mast"]["input"],
            "--output_dir", str(out),
            "--temperature", str(TEMPERATURE),
            "--edge_threshold", str(CORR_THRESHOLD),
            "--suppes_graph", BENCHMARKS["mast"]["suppes_graph"],
            "--effect_edges", BENCHMARKS["mast"]["effect_edges"],
            "--max_model_len", str(bb["max_model_len_mast"]),
            "--gpu_memory_utilization", str(bb["gpu_memory_utilization"]),
            "--tp", str(bb["tensor_parallel_size"]),
        ]
    return [
        sys.executable, str(MAST_BASELINE_SCRIPT_VLLM),
        "--model", bb["hf_name"],
        "--model_tag", bb["model_tag"],
        "--input", BENCHMARKS["mast"]["input"],
        "--output_dir", str(out),
        "--temperature", str(TEMPERATURE),
        "--max_model_len", str(bb["max_model_len_mast"]),
        "--gpu_memory_utilization", str(bb["gpu_memory_utilization"]),
        "--tp", str(bb["tensor_parallel_size"]),
    ]


def _mast_arc_cmd(bb, variant: str, sample_idx: int) -> list[str]:
    out = output_dir("mast", _bb_key(bb), variant, sample_idx)
    out.mkdir(parents=True, exist_ok=True)
    if variant == "edge":
        return [
            sys.executable, str(MAST_EDGE_SCRIPT_ARC),
            "--model", bb["hf_name"],
            "--model_tag", bb["model_tag"],
            "--input", BENCHMARKS["mast"]["input"],
            "--output_dir", str(out),
            "--temperature", str(TEMPERATURE),
            "--edge_threshold", str(CORR_THRESHOLD),
            "--suppes_graph", BENCHMARKS["mast"]["suppes_graph"],
            "--effect_edges", BENCHMARKS["mast"]["effect_edges"],
        ]
    return [
        sys.executable, str(MAST_BASELINE_SCRIPT_API),
        "--model", bb["hf_name"],
        "--model_tag", bb["model_tag"],
        "--input", BENCHMARKS["mast"]["input"],
        "--output_dir", str(out),
        "--temperature", str(TEMPERATURE),
    ]


def _bb_key(bb_cfg: dict) -> str:
    for k, v in BACKBONES.items():
        if v is bb_cfg:
            return k
    raise KeyError("backbone config not in BACKBONES")


def all_cells(only_backbone: str | None = None, only_benchmark: str | None = None):
    """Yield (bb_key, bb_cfg, benchmark, variant, sample_idx) for every cell."""
    bb_keys = [only_backbone] if only_backbone else list(BACKBONES.keys())
    benches = [only_benchmark] if only_benchmark else list(BENCHMARKS.keys())
    for bb_key in bb_keys:
        bb = BACKBONES[bb_key]
        for benchmark in benches:
            for variant in ("baseline", "edge"):
                for sample_idx in range(1, N_SAMPLES + 1):
                    yield bb_key, bb, benchmark, variant, sample_idx


def build_commands(bb_key, bb, benchmark, variant, sample_idx) -> list[list[str]]:
    """Return list of subprocess commands for one cell (TRAIL = 2 splits, MAST = 1)."""
    if benchmark == "trail":
        cmds = []
        for split in BENCHMARKS["trail"]["splits"]:
            cmd_fn = _trail_arc_cmd if bb["backend"] == "api" else _trail_vllm_cmd
            cmds.append(cmd_fn(bb, variant, sample_idx, split))
        return cmds
    if benchmark == "mast":
        cmd_fn = _mast_arc_cmd if bb["backend"] == "api" else _mast_vllm_cmd
        return [cmd_fn(bb, variant, sample_idx)]
    raise ValueError(benchmark)


def emit_sbatch(bb_key, bb, benchmark, variant, sample_idx, cmds) -> str:
    """Wrap commands into an sbatch invocation via the parametric template."""
    job_name = f"r3-{bb_key}-{benchmark}-{variant}-s{sample_idx}"
    out_log  = R3_ROOT / "sbatch" / f"slurm.{job_name}.%j.out"
    err_log  = R3_ROOT / "sbatch" / f"slurm.{job_name}.%j.err"
    cmd_text = "\n".join(" ".join(c) for c in cmds)
    env = (
        f"JOB_NAME={job_name} "
        f"CMDS=$'{cmd_text}' "
        f"sbatch --job-name={job_name} -o {out_log} -e {err_log} {SBATCH_TEMPLATE}"
    )
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES.keys()) + ["all"], default="all")
    ap.add_argument("--benchmark", choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    ap.add_argument("--execute", action="store_true",
                    help="Actually run sbatch / launch background; default just prints.")
    ap.add_argument("--local", action="store_true",
                    help="For vllm backbones, emit raw python commands instead of sbatch "
                         "wrappers (use when running on a local machine, not Slurm). "
                         "Caller is responsible for CUDA_VISIBLE_DEVICES.")
    args = ap.parse_args()

    only_bb = None if args.backbone == "all" else args.backbone
    only_bm = None if args.benchmark == "all" else args.benchmark

    for bb_key, bb, benchmark, variant, sample_idx in all_cells(only_bb, only_bm):
        cmds = build_commands(bb_key, bb, benchmark, variant, sample_idx)
        if bb["backend"] == "vllm" and not args.local:
            line = emit_sbatch(bb_key, bb, benchmark, variant, sample_idx, cmds)
            print(line)
            if args.execute:
                subprocess.run(line, shell=True, check=False)
        else:
            # Local vllm or ARC: emit raw python commands. ARC ones get
            # backgrounded with & so they can run in parallel; local vllm
            # commands run serially in the caller's shell (model loads once
            # per process but vLLM batches all prompts in that process).
            suffix = " &" if bb["backend"] == "api" else ""
            for cmd in cmds:
                line = " ".join(cmd) + suffix
                print(line)
                if args.execute:
                    if bb["backend"] == "api":
                        subprocess.Popen(cmd)
                    else:
                        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
