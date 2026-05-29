#!/usr/bin/env bash
# Pull predictions from VT ARC cluster -> wangserv after sbatch jobs finish.

set -euo pipefail

ARC_USER="junh"
ARC_HOST="tinkercliffs2.arc.vt.edu"
ARC_BASE="/projects/slmreasoning/junh/causal-error/trail-benchmark/rebuttal/holdout"

LOCAL_BASE="/data/wang/junh/githubs/trail-benchmark/rebuttal/holdout"

rsync -avh \
  "${ARC_USER}@${ARC_HOST}:${ARC_BASE}/data/predictions/trail/gpt-oss-120b/edge/" \
  "$LOCAL_BASE/data/predictions/trail/gpt-oss-120b/edge/"

rsync -avh \
  "${ARC_USER}@${ARC_HOST}:${ARC_BASE}/data/predictions/mast/gpt-oss-120b/edge/" \
  "$LOCAL_BASE/data/predictions/mast/gpt-oss-120b/edge/"

echo
echo "Pulled predictions back to $LOCAL_BASE/data/predictions/{trail,mast}/gpt-oss-120b/edge/"
echo "Now run: python aggregate.py"
