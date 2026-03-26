#!/usr/bin/env bash
# Run causal graph construction pipeline for TRAIL GAIA split.
# Execute from trail-benchmark/:
#   cd trail-benchmark && bash causal/graph/run_causal_gaia.sh

set -e
DATA_DIR="${DATA_DIR:-benchmarking/data}"
OUT_FILTERED="benchmarking/data/trail_filtered/gaia.jsonl"
OUT_SPAN_ORDER="benchmarking/data/trail_span_order/gaia.jsonl"
OUT_ONSETS="benchmarking/data/trail_derived/onsets_gaia.jsonl"

echo "=== Step 1: Build trace + annotation pairs (GAIA) ==="
python causal/graph/preprocess/trail_1_filter_split.py \
  --data_dir "$DATA_DIR" \
  --annotation_dir benchmarking \
  --out_path "$OUT_FILTERED" \
  --split GAIA

echo ""
echo "=== Step 2: Build candidate-span timeline + coverage ==="
python causal/graph/preprocess/trail_2_build_span_order.py \
  --filtered_path "$OUT_FILTERED" \
  --out_path "$OUT_SPAN_ORDER"

echo ""
echo "=== Step 3: Build onsets (present, onset, count, ties) ==="
python causal/graph/preprocess/trail_3_build_onsets.py \
  --filtered_path "$OUT_FILTERED" \
  --span_order_path "$OUT_SPAN_ORDER" \
  --out_path "$OUT_ONSETS"

echo ""
echo "Done. Onsets for downstream Suppes/CAPRI: $OUT_ONSETS"
