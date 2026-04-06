#!/bin/bash
#SBATCH --job-name=uq-eval-corr02-span-index
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/causal/"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="/projects/slmreasoning/junh/causal-error/CausalError"

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V
"$PYTHON" -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_count:', torch.cuda.device_count())"

# Pre-flight package check for run_uq_eval.py dependencies.
"$PYTHON" -c "import importlib.util,sys;mods=['torch','vllm','transformers','tqdm','tiktoken','dotenv','litellm','numpy','sklearn','scipy'];missing=[m for m in mods if importlib.util.find_spec(m) is None];print('missing:',missing if missing else 'none');sys.exit(1 if missing else 0)"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# UQ ablation: causal+corr0.2 graph for propagation + span_index, NO graph injection in Pass 2 prompt.
# Ablates graph_inject: isolates whether the gain in graph_inject came from the richer graph
# or from the explicit graph text in the Pass 2 prompt.
# Output: UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_corr0.2_span_index/
"$PYTHON" -u UQ/run_uq_eval.py \
  --split GAIA \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.75 \
  --enforce_eager \
  --corr_threshold 0.2 \
  --span_index \
  --propagation_threshold 0.1 \
  "$@"

EC=$?
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: $EC"
exit $EC
