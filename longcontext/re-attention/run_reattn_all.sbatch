#!/bin/bash
#SBATCH --job-name=reattn-all
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

# ReAttention long-context eval — full GAIA dataset (all 117 traces).
# For traces within 128K the model runs normally; for overflow traces
# YaRN RoPE scaling (factor=8.0) extends effective context to ~1M tokens.
# Uses device_map="auto" across 4 A100 GPUs (no tensor_parallel_size).

set -euo pipefail

module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/causal"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="/projects/slmreasoning/junh/causal-error/CausalError"

# HuggingFace model cache — set to a path with sufficient space (~65GB for QwenLong-32B)
export HF_HOME="/projects/slmreasoning/junh/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V
"$PYTHON" -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_count:', torch.cuda.device_count())"

# Pre-flight: check ReAttention + pipeline dependencies (no vllm needed)
"$PYTHON" -c "
import importlib.util, sys
mods = ['torch', 'transformers', 'einops', 'triton', 'tqdm']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print('missing:', missing if missing else 'none')
sys.exit(1 if missing else 0)
"

# Verify re_attention package is importable from longcontext/
"$PYTHON" -c "
import sys
sys.path.insert(0, 'longcontext')
from re_attention.cache_utils_v0921 import ReAttentionConfig
from re_attention.re_attention_qwen2 import Qwen2ForCausalLM
print('re_attention imports OK')
"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Process all 117 traces (no --min_tokens filter).
# --causal_only: 11 bootstrap-validated Suppes causal edges
# --validate_span_id: drop errors with invalid span hex locations
# --rope_scaling_factor 8.0: YaRN extends 128K → ~1M effective context
# --graph_probe (default True in script): hard-binary conf + targeted per-category probing
"$PYTHON" -u longcontext/run_longcontext_eval.py \
    --split GAIA \
    --causal_only \
    --validate_span_id \
    --rope_scaling_factor 8.0 \
    "$@"

EC=$?
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: $EC"
exit $EC
