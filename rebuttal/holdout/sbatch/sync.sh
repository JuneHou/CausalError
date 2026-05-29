#!/usr/bin/env bash
# Sync held-out artifacts from wangserv -> VT ARC cluster.
# Edit ARC_USER / ARC_HOST / ARC_BASE to match your login.
#
# Pushes only the minimal artifacts needed for the gpt-oss-120b sbatch jobs:
#   data/assignments/      fold metadata (lightweight)
#   data/graphs/           held-out Suppes + empty effect_edges
#   data/test_traces/      symlinked TRAIL test traces + MAST test.jsonl
# Plus the sbatch files themselves.
#
# After the slurm job finishes on ARC, run sync_back.sh to pull predictions.

set -euo pipefail

ARC_USER="junh"
ARC_HOST="tinkercliffs2.arc.vt.edu"
ARC_BASE="/projects/slmreasoning/junh/causal-error/trail-benchmark/rebuttal/holdout"

LOCAL_BASE="/data/wang/junh/githubs/trail-benchmark/rebuttal/holdout"

# Resolve symlinks under test_traces so rsync copies actual files (not links).
rsync -avh --copy-links \
  --include="data/" \
  --include="data/assignments/***" \
  --include="data/graphs/***" \
  --include="data/test_traces/***" \
  --include="sbatch/***" \
  --exclude="*" \
  "$LOCAL_BASE/" \
  "${ARC_USER}@${ARC_HOST}:${ARC_BASE}/"

echo
echo "Pushed to ${ARC_USER}@${ARC_HOST}:${ARC_BASE}"
echo "Next: ssh in and submit"
echo "  ssh ${ARC_USER}@${ARC_HOST}"
echo "  cd ${ARC_BASE}/sbatch"
echo "  sbatch run_holdout_gpt_oss_120b_trail.sbatch"
echo "  sbatch run_holdout_gpt_oss_120b_mast.sbatch"
