#!/usr/bin/env bash
# Sweep corr_threshold across {causal-only, 0.30, 0.20, 0.15, 0.10, 0.05}
# for the +GI+SI graph-injection method, then score everything.
#
# Usage:
#   eval/run_threshold_sweep.sh <model> <split> [gpus] [output_dir] [backend]
#
#   <model>      e.g. openai/gpt-oss-20b, Tongyi-Zhiwen/QwenLong-L1-32B,
#                     gemini/gemini-2.5-flash
#   <split>      GAIA_dedup | SWE_Bench_dedup
#   [gpus]       CUDA_VISIBLE_DEVICES value (default: 0,1). Ignored for litellm.
#   [output_dir] default: outputs/zero_shot2
#   [backend]    vllm | litellm. If omitted, inferred from model:
#                  models starting with "gemini/" or "openai/gpt-4" -> litellm
#                  everything else -> vllm
#
# Examples:
#   eval/run_threshold_sweep.sh openai/gpt-oss-20b GAIA_dedup 0,1
#   eval/run_threshold_sweep.sh Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3
#   eval/run_threshold_sweep.sh gemini/gemini-2.5-flash GAIA_dedup "" outputs/zero_shot2 litellm

set -euo pipefail

MODEL="${1:?model required}"
SPLIT="${2:?split required}"
GPUS="${3:-0,1}"
OUTDIR="${4:-outputs/zero_shot2}"
BACKEND="${5:-}"

# Infer backend from model name if not specified
if [[ -z "$BACKEND" ]]; then
  case "$MODEL" in
    gemini/*|openai/gpt-4*|openai/o*) BACKEND="litellm" ;;
    *)                                BACKEND="vllm"   ;;
  esac
fi

case "$BACKEND" in
  vllm)    SCRIPT="eval/run_eval_graph_inject_vllm.py" ;;
  litellm) SCRIPT="eval/run_eval_graph_inject.py"      ;;
  *) echo "Unknown backend: $BACKEND" >&2; exit 1 ;;
esac

# Threshold list. "causal_only" is a sentinel handled below.
THRESHOLDS=(causal_only 0.30 0.20 0.15 0.10 0.05)

LOGDIR="${OUTDIR}/_sweep_logs"
mkdir -p "$LOGDIR"

run_cmd() {
  if [[ "$BACKEND" == "vllm" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT" "$@"
  else
    python "$SCRIPT" "$@"
  fi
}

for t in "${THRESHOLDS[@]}"; do
  tag="${MODEL//\//-}-${SPLIT}-t${t}"
  log="${LOGDIR}/${tag}.log"
  echo
  echo "============================================================"
  echo "[$(date +%T)] threshold=$t  model=$MODEL  split=$SPLIT  -> $log"
  echo "============================================================"

  if [[ "$t" == "causal_only" ]]; then
    run_cmd \
        --model "$MODEL" \
        --split "$SPLIT" \
        --causal_only --span_index \
        --output_dir "$OUTDIR" \
        2>&1 | tee "$log"
  else
    run_cmd \
        --model "$MODEL" \
        --split "$SPLIT" \
        --corr_threshold "$t" --span_index \
        --output_dir "$OUTDIR" \
        2>&1 | tee "$log"
  fi
done

echo
echo "============================================================"
echo "[$(date +%T)] All thresholds complete. Scoring $OUTDIR ..."
echo "============================================================"
python eval/calculate_scores.py --results_dir "$OUTDIR"

echo
echo "Sweep done. Per-threshold metrics files:"
ls -1 "${OUTDIR}"/*"${SPLIT}"*corr*-metrics.txt "${OUTDIR}"/*"${SPLIT}"*causal_only*-metrics.txt 2>/dev/null | sort
