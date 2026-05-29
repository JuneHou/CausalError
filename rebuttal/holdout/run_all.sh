#!/usr/bin/env bash
# Orchestrator for the EDGE held-out 80/20 experiment.
# Run after activating the project env:
#   conda activate "/data/wang/junh/envs/causal"
#   cd /data/wang/junh/githubs/trail-benchmark/rebuttal/holdout
#   bash run_all.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[1/5] build_folds.py ............... task #59  (80/20 split assignment)"
python build_folds.py

echo
echo "[2/5] build_fold_graphs.py ......... task #60  (graph from train side only)"
python build_fold_graphs.py

echo
echo "[3/5] run_fold_detection.py ........ task #61  (GPU heavy, ~4-6 hours, +EDGE on held-out)"
python run_fold_detection.py

echo
echo "[4/5] subset_baseline.py ........... task #62  (baseline predictions on held-out)"
python subset_baseline.py

echo
echo "[5/5] aggregate.py ................. task #63  (rebuttal table)"
python aggregate.py

echo
echo "Done. Results in: results/rebuttal_holdout_table.tex"
