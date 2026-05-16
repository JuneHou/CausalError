#!/usr/bin/env bash
# Sister of run_threshold_sweep.sh that runs the +CG (one-pass,
# in-prompt graph guidance) method across the same threshold list
# instead of the +GI (two-pass, dynamic injection) method.
#
# Pair this with the +GI sweep to compare how each injection
# architecture scales with edge count under the same edge sets
# (§4.5 claim).
#
# Threshold list (identical to the +GI sweep so cells line up):
#   0.35 -> 19 edges (knee)
#   0.25 -> 21 edges
#   0.20 -> 25 edges (saturating)
#   random12_seed42 -> 12 random non-Suppes edges (null-graph control)
#
# IMPORTANT DIFFERENCES FROM run_threshold_sweep.sh:
#   1. Inner runner is one of:
#        vllm      -> eval/run_eval_with_graph_vllm.py          (--edge_threshold)
#        litellm   -> eval/run_eval_with_graph.py               (--edge_threshold)
#        deepinfra -> eval/run_eval_with_graph_api_deepinfra.py (--corr_threshold)
#        arc       -> eval/run_eval_with_graph_api_arc.py       (--corr_threshold)
#   2. The vllm/litellm runners do NOT support --corr_threshold (union of
#      causal + Suppes>=τ) or --random_edges; they only support plain
#      --edge_threshold (Suppes>=τ). The API runners support both.
#      Effect on cross-method comparison: API-backend cells use the same
#      edge sets as the +GI sweep (good); vllm/litellm cells use a slightly
#      different edge set at the same τ (close in practice — see Task C
#      notes in TODO.md). The 'random' threshold is SKIPPED for vllm/litellm
#      with a warning until those runners get --random_edges ported in.
#   3. --span_index is OFF by default to match the +CG row in
#      paper/tables/threshold_sweep_ablation.tex. Set SPAN_INDEX=1 to
#      flip it on.
#
# Output root: outputs_thres_cg/ (parallel to outputs_thres/ for +GI).
# Naming inside each t<τ>/ subdir:
#   API runners (corr_threshold):  outputs_<model>-<split>-graph_causal_corr<τ>/
#   vllm/litellm (edge_threshold): outputs_<model>-<split>-graph_t<τ>/
# Existing τ=0.20 +CG runs already live under outputs_thres_cg/t0.20/
# with the graph_t0.2 suffix (moved there from outputs/zero_shot2/).
#
# Usage (identical positional args to the +GI sweep):
#   eval/run_threshold_sweep_cg.sh <model> <split> [gpus] [output_dir] [backend]
#
# Examples:
#   eval/run_threshold_sweep_cg.sh openai/gemma-3-27b-it GAIA_dedup 0,1
#   eval/run_threshold_sweep_cg.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503 SWE_Bench_dedup 0,1
#   THRESHOLDS="0.25 0.20" SPAN_INDEX=1 eval/run_threshold_sweep_cg.sh \
#       Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3

set -euo pipefail

MODEL="${1:?model required}"
SPLIT="${2:?split required}"
GPUS="${3:-0,1}"
OUTDIR="${4:-outputs_thres_cg}"
BACKEND="${5:-}"

# Backend inference matches run_threshold_sweep.sh exactly.
if [[ -z "$BACKEND" ]]; then
  case "$MODEL" in
    gemini/*|openai/gpt-4*|openai/o*) BACKEND="litellm"   ;;
    openai/gpt-oss-*|google/*|mistralai/*)        BACKEND="deepinfra" ;;
    */*)                              BACKEND="vllm"      ;;
    *)                                BACKEND="arc"       ;;
  esac
fi

case "$BACKEND" in
  vllm)      SCRIPT="eval/run_eval_with_graph_vllm.py"          ;;
  litellm)   SCRIPT="eval/run_eval_with_graph.py"               ;;
  deepinfra) SCRIPT="eval/run_eval_with_graph_api_deepinfra.py" ;;
  arc)       SCRIPT="eval/run_eval_with_graph_api_arc.py"       ;;
  *) echo "Unknown backend: $BACKEND" >&2; exit 1 ;;
esac

# vllm/litellm only support --edge_threshold; API runners support
# --corr_threshold (union with causal_only, matches the +GI sweep) AND
# --random_edges. This flag controls which form the inner script gets.
case "$BACKEND" in
  vllm|litellm) SUPPORTS_CORR_AND_RANDOM=0 ;;
  *)            SUPPORTS_CORR_AND_RANDOM=1 ;;
esac

# tensor_parallel_size mirrors the +GI sweep so 4 GPUs actually get used.
if [[ "$BACKEND" == "vllm" ]]; then
  IFS=',' read -ra _gpu_arr <<< "$GPUS"
  TP="${#_gpu_arr[@]}"
else
  TP=""
fi

# Per-model max_model_len — same table as the +GI sweep.
case "$MODEL" in
  Tongyi-Zhiwen/QwenLong-L1-32B*)                 MAX_MODEL_LEN=128000 ;;
  mistralai/Mistral-Small-3.1-24B-Instruct-2503*) MAX_MODEL_LEN=108000 ;;
  mistralai/Mistral-Small-3.1-24B-Instruct-2503*) MAX_MODEL_LEN=108000 ;;
  openai/gemma-3-27b-it*|google/gemma-3-27b-it*)  MAX_MODEL_LEN=108000 ;;
  openai/gpt-oss-20b*|openai/gpt-oss-120b*)       MAX_MODEL_LEN=108000 ;;
  *)                                              MAX_MODEL_LEN=""     ;;
esac

# SPAN_INDEX=1 to enable --span_index on every call. Off by default.
SPAN_INDEX="${SPAN_INDEX:-0}"

# Threshold list. Same defaults as the +GI sweep so cells align.
# 'random' is currently a no-op for +CG (see header) — kept in the
# default list so you get a clear skip warning rather than silent
# omission; remove via THRESHOLDS=... if it's noise.
THRESHOLDS=(${THRESHOLDS:-random 0.35 0.25 0.20})

LOGDIR="${OUTDIR}/_sweep_logs"
mkdir -p "$LOGDIR"

run_cmd() {
  local extra=()
  if [[ -n "$MAX_MODEL_LEN" && "$BACKEND" == "vllm" ]]; then
    extra+=(--max_model_len "$MAX_MODEL_LEN")
  fi
  if [[ "$SPAN_INDEX" == "1" ]]; then
    extra+=(--span_index)
  fi
  if [[ "$BACKEND" == "vllm" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT" --tensor_parallel_size "$TP" "${extra[@]}" "$@"
  else
    python "$SCRIPT" "${extra[@]}" "$@"
  fi
}

for t in "${THRESHOLDS[@]}"; do
  tag="${MODEL//\//-}-${SPLIT}-cg-t${t}"
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
  echo "[$(date +%T)] [+CG] threshold=$t  model=$MODEL  split=$SPLIT  -> $log"
  echo "             output_dir=$thres_outdir  span_index=$SPAN_INDEX"
  echo "============================================================"

  case "$t" in
    causal_only)
      run_cmd \
          --model "$MODEL" \
          --split "$SPLIT" \
          --causal_only \
          --output_dir "$thres_outdir" \
          2>&1 | tee "$log"
      ;;
    random)
      if [[ "$SUPPORTS_CORR_AND_RANDOM" == "1" ]]; then
        run_cmd \
            --model "$MODEL" \
            --split "$SPLIT" \
            --random_edges \
            --output_dir "$thres_outdir" \
            2>&1 | tee "$log"
      else
        echo "WARN: +CG ${BACKEND} runner ($SCRIPT) has no --random_edges flag — skipping 'random' threshold." | tee "$log"
        echo "      Port the random-edges plumbing from run_eval_graph_inject_vllm.py into"  | tee -a "$log"
        echo "      $SCRIPT (lines 231-248, 502-506, 533-538) to enable on this backend."     | tee -a "$log"
        continue
      fi
      ;;
    *)
      if [[ "$SUPPORTS_CORR_AND_RANDOM" == "1" ]]; then
        # API runners: --corr_threshold (union with causal_only, matches +GI sweep)
        run_cmd \
            --model "$MODEL" \
            --split "$SPLIT" \
            --corr_threshold "$t" \
            --output_dir "$thres_outdir" \
            2>&1 | tee "$log"
      else
        # vllm/litellm: --edge_threshold (plain Suppes geomean only)
        run_cmd \
            --model "$MODEL" \
            --split "$SPLIT" \
            --edge_threshold "$t" \
            --output_dir "$thres_outdir" \
            2>&1 | tee "$log"
      fi
      ;;
  esac
done

echo
echo "============================================================"
echo "[$(date +%T)] [+CG] All thresholds complete. Scoring per-threshold dirs under $OUTDIR ..."
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
echo "[+CG] Sweep done. Per-threshold metrics files (filtered to +CG outputs):"
ls -1 "${OUTDIR}"/t*/*"${SPLIT}"*-graph_t*-metrics.txt 2>/dev/null | sort
