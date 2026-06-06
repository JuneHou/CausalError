#!/bin/bash
# Queue all in-process-vLLM R3 jobs (mistral-24b ran locally without sbatch).
# baseline and edge are separate jobs; submit only what you need to (re)run.
set -euo pipefail
cd "$(dirname "$0")"

sbatch r3_gpt-oss-120b_trail_baseline.sbatch
sbatch r3_gpt-oss-120b_trail_edge.sbatch
sbatch r3_gpt-oss-120b_mast_baseline.sbatch
sbatch r3_gpt-oss-120b_mast_edge.sbatch
sbatch r3_gpt-oss-20b_trail_baseline.sbatch
sbatch r3_gpt-oss-20b_trail_edge.sbatch
sbatch r3_gpt-oss-20b_mast_baseline.sbatch
sbatch r3_gpt-oss-20b_mast_edge.sbatch
sbatch r3_gemma-3-27b_trail_baseline.sbatch
sbatch r3_gemma-3-27b_trail_edge.sbatch
sbatch r3_gemma-3-27b_mast_baseline.sbatch
sbatch r3_gemma-3-27b_mast_edge.sbatch
sbatch r3_qwenlong-32b_trail_baseline.sbatch
sbatch r3_qwenlong-32b_trail_edge.sbatch
sbatch r3_qwenlong-32b_mast_baseline.sbatch
sbatch r3_qwenlong-32b_mast_edge.sbatch
