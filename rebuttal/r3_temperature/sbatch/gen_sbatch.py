#!/usr/bin/env python3
"""
Generate the per-job R3 sbatch files. One job per (vllm backbone x benchmark x
variant) -> 4 backbones x {trail,mast} x {baseline,edge} = 16 jobs, plus
submit_all.sh. Self-contained, holdout-style:

  - in-process vLLM (NOT the older `vllm serve` + litellm-client pattern)
  - --account=slmreasoning --partition=a100_normal_q --gres=gpu:4 --mem=256G
  - mandatory cache redirects off $HOME (home quota is tiny -> ENOSPC otherwise)
  - server paths; correct cwd per benchmark (TRAIL -> benchmarking, MAST -> CausalMAST)
  - baseline and edge are SEPARATE jobs (edge is the long pass; keeping it on its
    own job means it gets a full wall-time budget instead of sharing with baseline).
  - each job loops sample (x split for TRAIL); one model load per python
    invocation, re-invoked per i.i.d. sample. --seed "$sample" makes the samples
    reproducible-yet-distinct (without it, vLLM can return identical draws -> sigma=0).

Covers the backbones that run in-process via sbatch on this server, including
gpt-oss-120b (4xA100, tp=4, /common/data/models copy). mistral-24b is omitted:
it was run on the local box without sbatch.

Re-run this script after editing backbone params to regenerate all files.
"""
from pathlib import Path

REPO_DIR = "/projects/slmreasoning/junh/causal-error/CausalError"
MAST_DIR = "/projects/slmreasoning/junh/causal-error/CausalMAST"
ENV_PATH = "/projects/slmreasoning/junh/envs/causal"
SBATCH_DIR = Path(__file__).resolve().parent

TEMPERATURE = "0.7"
N_SAMPLES = 3
# Per-benchmark corr-union optimum (main-results table): TRAIL tau=0.35, MAST tau=0.50.
# These are NOT the same value -- a single shared threshold is the bug that sent MAST
# edge runs to tau=0.35. Threshold is applied at RUNTIME (filters suppes_graph edges by
# geomean score), so no graph rebuild is needed; only the edge variant consults it.
#
# MAST edge MUST use eval/full_run_eval_graph_inject.py --corr_threshold (the script
# that produced the main-table CASCADE cells): UNION (Suppes geomean >= tau) ∪
# (11 validated causal edges) = 25 edges at tau=0.50. The sibling
# run_eval_graph_inject.py --edge_threshold is a PURE 15-edge cut — a different
# method; substituting it is the bug behind the discarded 06-05/06-12 MAST edge runs.
CORR_THRESHOLD = "0.35"       # TRAIL --corr_threshold
MAST_EDGE_THRESHOLD = "0.5"   # MAST  --corr_threshold (full_run_eval_graph_inject.py)
# MAST max_model_len is per-backbone ("mlm" in BACKBONES), matching the values the
# main-table runs used (MAST/eval/run_threshold_sweep.sh + run_eval_yesno_vllm.py
# default): 108000 for mistral/gemma/gpt-oss, 128000 for qwenlong. MAST traces max
# out near ~8k tokens, so this only sizes the KV cache — but keep protocol identical.

# Per-job wall-time. Sized from observed runtimes so SLURM can backfill into smaller
# gaps and start sooner (a 12h request waits for a 12h hole). MAST combined baseline+edge
# ran <=2.88h (qwenlong) at tau=0.35; edge-only at tau=0.50 is lighter, so 4h is ~1.4x
# headroom. TRAIL stays 12h (edge sweeps GAIA+SWE x 3 samples, genuinely long).
WALLTIME = {
    ("mast", "baseline"): "02:00:00",
    ("mast", "edge"):     "04:00:00",
    ("trail", "baseline"): "12:00:00",
    ("trail", "edge"):     "12:00:00",
}
DEFAULT_WALLTIME = "12:00:00"

# TRAIL graph artifacts (full-corpus tau=0.35, no rebuild)
TRAIL_SUPPES = f"{REPO_DIR}/benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json"
TRAIL_EFFECT = f"{REPO_DIR}/benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json"
# MAST graph artifacts
MAST_SUPPES = f"{MAST_DIR}/causal_graph/outputs/suppes_graph.json"
MAST_EFFECT = f"{MAST_DIR}/causal_graph/outputs/interventions/effect_edges.json"
MAST_INPUT = f"{MAST_DIR}/data/annotation/annotation_ag2_filtered.jsonl"

R3_DIR = f"{REPO_DIR}/rebuttal/r3_temperature"
PRED = f"{R3_DIR}/data/predictions"

# Per-backbone constants (mirror config.py:BACKBONES).
# hf_cache=True  -> load weights from HF cache (no /common/data/models copy).
BACKBONES = {
    "gpt-oss-120b": {
        "model_path": "/common/data/models/openai--gpt-oss-120b",
        "model_tag":  "openai-gpt-oss-120b",
        "mlt": "131072", "mlm": "108000", "hf_cache": False,
    },
    "gpt-oss-20b": {
        "model_path": "/common/data/models/openai--gpt-oss-20b",
        "model_tag":  "openai-gpt-oss-20b",
        "mlt": "131072", "mlm": "108000", "hf_cache": False,
    },
    "gemma-3-27b": {
        "model_path": "/common/data/models/google--gemma-3-27b-it",
        "model_tag":  "openai-gemma-3-27b-it",
        "mlt": "65536", "mlm": "108000", "hf_cache": False,
    },
    "qwenlong-32b": {
        "model_path": "Tongyi-Zhiwen/QwenLong-L1-32B",
        "model_tag":  "Tongyi-Zhiwen-QwenLong-L1-32B",
        "mlt": "131072", "mlm": "128000", "hf_cache": True,
    },
}

HEADER = """#!/bin/bash
#SBATCH --job-name=r3-{bb}-{bench}-{variant}
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time={walltime}
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

module load Miniconda3

ENV_PATH="{env}"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="{repo}"
MAST_DIR="{mast}"
R3_DIR="$REPO_DIR/rebuttal/r3_temperature"
MODEL_PATH="{model_path}"
MODEL_TAG="{model_tag}"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Redirect caches off $HOME (home quota is small; jobs fail with ENOSPC otherwise)
CACHE_ROOT="$R3_DIR/results/.cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
{hf_override}"""

HF_OVERRIDE = """# QwenLong has no /common/data/models copy; load from the populated HF cache.
export HF_HOME="/projects/slmreasoning/junh/.cache/huggingface"
"""

TRAIL_EDGE_BODY = """
# run_eval_*_vllm.py resolve data/output paths relative to cwd
cd "$REPO_DIR/benchmarking"

variant=edge
for split in GAIA_dedup SWE_Bench_dedup; do
  for sample in $(seq 1 {n_samples}); do
    OUT="$R3_DIR/data/predictions/trail/{bb}/$variant/temp{temp}_sample$sample"
    mkdir -p "$OUT"
    echo "=== TRAIL $variant $split sample$sample -> $OUT ==="
    "$PYTHON" -u eval/run_eval_graph_inject_vllm.py \\
      --model "$MODEL_PATH" \\
      --split "$split" \\
      --data_dir data \\
      --output_dir "$OUT" \\
      --temperature {temp} \\
      --seed "$sample" \\
      --tensor_parallel_size 4 \\
      --gpu_memory_utilization 0.75 \\
      --max_model_len {mlt} \\
      --corr_threshold {corr} \\
      --span_index \\
      --suppes_graph "{trail_suppes}" \\
      --causal_graph "{trail_effect}"
  done
done

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
"""

TRAIL_BASELINE_BODY = """
# run_eval_*_vllm.py resolve data/output paths relative to cwd
cd "$REPO_DIR/benchmarking"

variant=baseline
for split in GAIA_dedup SWE_Bench_dedup; do
  for sample in $(seq 1 {n_samples}); do
    OUT="$R3_DIR/data/predictions/trail/{bb}/$variant/temp{temp}_sample$sample"
    mkdir -p "$OUT"
    echo "=== TRAIL $variant $split sample$sample -> $OUT ==="
    "$PYTHON" -u eval/run_eval_vllm.py \\
      --model "$MODEL_PATH" \\
      --split "$split" \\
      --data_dir data \\
      --output_dir "$OUT" \\
      --temperature {temp} \\
      --seed "$sample" \\
      --tensor_parallel_size 4 \\
      --gpu_memory_utilization 0.75 \\
      --max_model_len {mlt}
  done
done

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
"""

MAST_EDGE_BODY = """
# MAST eval scripts resolve relative paths from the CausalMAST repo root
cd "$MAST_DIR"

variant=edge
for sample in $(seq 1 {n_samples}); do
  OUT="$R3_DIR/data/predictions/mast/{bb}/$variant/temp{temp}_sample$sample"
  mkdir -p "$OUT"
  echo "=== MAST $variant sample$sample -> $OUT ==="
  "$PYTHON" -u eval/full_run_eval_graph_inject.py \\
    --model "$MODEL_PATH" \\
    --model_tag "$MODEL_TAG" \\
    --input "{mast_input}" \\
    --output_dir "$OUT" \\
    --temperature {temp} \\
    --seed "$sample" \\
    --tp 4 \\
    --gpu_memory_utilization 0.75 \\
    --max_model_len {mlm} \\
    --corr_threshold {corr} \\
    --suppes_graph "{mast_suppes}" \\
    --effect_edges "{mast_effect}"
done

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
"""

MAST_BASELINE_BODY = """
# MAST eval scripts resolve relative paths from the CausalMAST repo root
cd "$MAST_DIR"

variant=baseline
for sample in $(seq 1 {n_samples}); do
  OUT="$R3_DIR/data/predictions/mast/{bb}/$variant/temp{temp}_sample$sample"
  mkdir -p "$OUT"
  echo "=== MAST $variant sample$sample -> $OUT ==="
  "$PYTHON" -u eval/run_eval_yesno_vllm.py \\
    --model "$MODEL_PATH" \\
    --model_tag "$MODEL_TAG" \\
    --input "{mast_input}" \\
    --output_dir "$OUT" \\
    --temperature {temp} \\
    --seed "$sample" \\
    --tp 4 \\
    --gpu_memory_utilization 0.75 \\
    --max_model_len {mlm}
done

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
"""

BODIES = {
    ("trail", "baseline"): TRAIL_BASELINE_BODY,
    ("trail", "edge"):     TRAIL_EDGE_BODY,
    ("mast", "baseline"):  MAST_BASELINE_BODY,
    ("mast", "edge"):      MAST_EDGE_BODY,
}


def render(bb, cfg, bench, variant):
    header = HEADER.format(
        bb=bb, bench=bench, variant=variant, env=ENV_PATH, repo=REPO_DIR, mast=MAST_DIR,
        model_path=cfg["model_path"], model_tag=cfg["model_tag"],
        walltime=WALLTIME.get((bench, variant), DEFAULT_WALLTIME),
        hf_override=(HF_OVERRIDE if cfg["hf_cache"] else ""),
    )
    if bench == "trail":
        body = BODIES[(bench, variant)].format(
            bb=bb, n_samples=N_SAMPLES, temp=TEMPERATURE, mlt=cfg["mlt"],
            corr=CORR_THRESHOLD, trail_suppes=TRAIL_SUPPES, trail_effect=TRAIL_EFFECT,
        )
    else:
        body = BODIES[(bench, variant)].format(
            bb=bb, n_samples=N_SAMPLES, temp=TEMPERATURE, mlm=cfg["mlm"],
            corr=MAST_EDGE_THRESHOLD, mast_input=MAST_INPUT,
            mast_suppes=MAST_SUPPES, mast_effect=MAST_EFFECT,
        )
    return header + body


def main():
    files = []
    for bb, cfg in BACKBONES.items():
        for bench in ("trail", "mast"):
            for variant in ("baseline", "edge"):
                path = SBATCH_DIR / f"r3_{bb}_{bench}_{variant}.sbatch"
                path.write_text(render(bb, cfg, bench, variant))
                files.append(path.name)
                print(f"wrote {path}")

    submit = ["#!/bin/bash",
              "# Queue all in-process-vLLM R3 jobs (mistral-24b ran locally without sbatch).",
              "# baseline and edge are separate jobs; submit only what you need to (re)run.",
              "set -euo pipefail",
              'cd "$(dirname "$0")"', ""]
    submit += [f"sbatch {f}" for f in files]
    submit_path = SBATCH_DIR / "submit_all.sh"
    submit_path.write_text("\n".join(submit) + "\n")
    submit_path.chmod(0o755)
    print(f"wrote {submit_path}")


if __name__ == "__main__":
    main()
