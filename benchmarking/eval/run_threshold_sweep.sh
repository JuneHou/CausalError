#!/usr/bin/env bash
# Sweep corr_threshold across {0.35, 0.25, 0.20} for the +GI+SI
# graph-injection method, then score everything.  Also runs the
# random-12-edges baseline at the same +GI+SI setting.
#
# Threshold choice rationale (geomean = sqrt(precedence * PR_delta) on suppes_graph.json,
# union with the 12 intervention-validated causal edges):
#   0.35 -> 19 edges (knee: largest +5 jump from 14)
#   0.25 -> 21 edges (mid-regime)
#   0.20 -> 25 edges (saturating end of the sweep)
#
# Plus baselines (run separately; not part of the sweep loop):
#   causal_only (12 edges, intervention-validated)
#   random12_seed42 (12 random non-Suppes edges; null-graph control)
#
# NOTE: switching the threshold score from sqrt(P(B|A)*PR_delta) to
# sqrt(precedence*PR_delta) invalidates all previous corr-threshold
# outputs.  Existing outputs_thres/t0.20/ and outputs_corr/ results
# under the old score must be re-run.
#
# Usage:
#   eval/run_threshold_sweep.sh <model> <split> [gpus] [output_dir] [backend]
#
#   <model>      e.g. openai/gpt-oss-20b, Tongyi-Zhiwen/QwenLong-L1-32B,
#                     gemini/gemini-2.5-flash
#   <split>      GAIA_dedup | SWE_Bench_dedup
#   [gpus]       CUDA_VISIBLE_DEVICES value (default: 0,1). Ignored for litellm.
#   [output_dir] default: outputs_thres (per-threshold subdir t<t>/ is appended automatically)
#   [backend]    vllm | litellm. If omitted, inferred from model:
#                  models starting with "gemini/" or "openai/gpt-4" -> litellm
#                  everything else -> vllm
#
# Examples:
#   eval/run_threshold_sweep.sh openai/gpt-oss-20b GAIA_dedup 0,1
#   eval/run_threshold_sweep.sh Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3
#   eval/run_threshold_sweep.sh gemini/gemini-2.5-flash GAIA_dedup "" outputs_thres litellm

set -euo pipefail

MODEL="${1:?model required}"
SPLIT="${2:?split required}"
GPUS="${3:-0,1}"
OUTDIR="${4:-outputs_thres}"
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

# Derive tensor_parallel_size from the GPU list so 4 GPUs actually get used.
# (Inner vLLM script defaults to TP=2; without this, extra GPUs are ignored.)
if [[ "$BACKEND" == "vllm" ]]; then
  IFS=',' read -ra _gpu_arr <<< "$GPUS"
  TP="${#_gpu_arr[@]}"
else
  TP=""
fi

# Per-model max_model_len. QwenLong's YaRN config officially caps at 128K
# (not 131072); Mistral-Small-3.1 OOM'd at 131072 with TP=2 (vLLM estimated
# 109,888), so 108000 leaves headroom. Gemma-3-27B and the GPT-oss family
# are capped at 108000 by the same empirical KV-cache budget across
# thresholds (published context is 128K for all three, but 108000 has been
# the practical cap in prior sweep runs). Others fall back to the inner
# script's default (131072).
case "$MODEL" in
  Tongyi-Zhiwen/QwenLong-L1-32B*)                 MAX_MODEL_LEN=128000 ;;
  mistralai/Mistral-Small-3.1-24B-Instruct-2503*) MAX_MODEL_LEN=108000 ;;
  openai/gemma-3-27b-it*|google/gemma-3-27b-it*)  MAX_MODEL_LEN=108000 ;;
  openai/gpt-oss-20b*|openai/gpt-oss-120b*)       MAX_MODEL_LEN=108000 ;;
  *)                                              MAX_MODEL_LEN=""     ;;
esac

# Threshold list. "causal_only" and "random" are sentinels handled below.
# Sweep points under the new score sqrt(precedence * PR_delta):
#   0.35 -> 19 edges (knee), 0.25 -> 21, 0.20 -> 25 (saturating)
# Baselines (also run by default so each model gets the full ablation):
#   random      -> 12 random non-Suppes edges, seed=42 (null-graph control)
#   causal_only -> 12 intervention-validated edges (already on disk in
#                  outputs/zero_shot2/; included here only if not skipped)
# Set THRESHOLDS via env to override, e.g.:
#   THRESHOLDS="0.25 0.20" eval/run_threshold_sweep.sh ...
THRESHOLDS=(${THRESHOLDS:-random 0.35 0.25 0.20})

LOGDIR="${OUTDIR}/_sweep_logs"
mkdir -p "$LOGDIR"

run_cmd() {
  local extra=()
  if [[ -n "$MAX_MODEL_LEN" && "$BACKEND" == "vllm" ]]; then
    extra+=(--max_model_len "$MAX_MODEL_LEN")
  fi
  if [[ "$BACKEND" == "vllm" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT" --tensor_parallel_size "$TP" "${extra[@]}" "$@"
  else
    python "$SCRIPT" "$@"
  fi
}

for t in "${THRESHOLDS[@]}"; do
  tag="${MODEL//\//-}-${SPLIT}-t${t}"
  log="${LOGDIR}/${tag}.log"
  case "$t" in
    causal_only) sub="t_causal_only" ;;
    random)      sub="t_random12_seed42" ;;
    *)           sub="t${t}" ;;
  esac
  thres_outdir="${OUTDIR}/${sub}"
  mkdir -p "$thres_outdir"
  echo
  echo "============================================================"
  echo "[$(date +%T)] threshold=$t  model=$MODEL  split=$SPLIT  -> $log"
  echo "             output_dir=$thres_outdir"
  echo "============================================================"

  case "$t" in
    causal_only)
      run_cmd \
          --model "$MODEL" \
          --split "$SPLIT" \
          --causal_only --span_index \
          --output_dir "$thres_outdir" \
          2>&1 | tee "$log"
      ;;
    random)
      run_cmd \
          --model "$MODEL" \
          --split "$SPLIT" \
          --random_edges --span_index \
          --output_dir "$thres_outdir" \
          2>&1 | tee "$log"
      ;;
    *)
      run_cmd \
          --model "$MODEL" \
          --split "$SPLIT" \
          --corr_threshold "$t" --span_index \
          --output_dir "$thres_outdir" \
          2>&1 | tee "$log"
      ;;
  esac
done

echo
echo "============================================================"
echo "[$(date +%T)] All thresholds complete. Scoring per-threshold dirs under $OUTDIR ..."
echo "============================================================"
for t in "${THRESHOLDS[@]}"; do
  case "$t" in
    causal_only) sub="t_causal_only" ;;
    random)      sub="t_random12_seed42" ;;
    *)           sub="t${t}" ;;
  esac
  python eval/calculate_scores.py --results_dir "${OUTDIR}/${sub}"
done

echo
echo "Sweep done. Per-threshold metrics files:"
ls -1 "${OUTDIR}"/t*/*"${SPLIT}"*-metrics.txt 2>/dev/null | sort
