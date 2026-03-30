#!/usr/bin/env bash
# =============================================================================
# run_causal_train.sh — Build Suppes causal graph from TRAINING DATA ONLY.
#
# Combines:
#   - GAIA train traces     (from graph/splits/train_trace_ids.json)
#   - SWE-bench train traces (from graph/splits_swe/train_trace_ids.json)
#
# This avoids data leakage: no test-split label statistics influence the graph.
# The shared error taxonomy makes both datasets valid for joint graph construction.
#
# Prerequisites:
#   1. GAIA split exists:     graph/splits/train_trace_ids.json
#      (run: python graph/01_make_splits.py)
#   2. SWE-bench split exists: graph/splits_swe/train_trace_ids.json
#      (run: python graph/01_make_splits.py --datasets swe_bench \
#                  --out_dir graph/splits_swe/ --train_ratio 0.70 --val_ratio 0.15)
#
# Execute from trail-benchmark/:
#   bash causal_train/graph/run_causal_train.sh
# =============================================================================

set -e

BENCH_DIR="${BENCH_DIR:-benchmarking}"
DATA_DIR="${BENCH_DIR}/data"

# Split ID files
GAIA_TRAIN_IDS="graph/splits/train_trace_ids.json"
SWE_TRAIN_IDS="graph/splits_swe/train_trace_ids.json"

# Intermediate outputs
OUT_FILTERED_GAIA="${DATA_DIR}/trail_filtered_train/gaia_train.jsonl"
OUT_FILTERED_SWE="${DATA_DIR}/trail_filtered_train/swe_train.jsonl"
OUT_SPAN_ORDER_GAIA="${DATA_DIR}/trail_span_order_train/gaia_train.jsonl"
OUT_SPAN_ORDER_SWE="${DATA_DIR}/trail_span_order_train/swe_train.jsonl"
OUT_ONSETS_GAIA="${DATA_DIR}/trail_derived/onsets_gaia_train.jsonl"
OUT_ONSETS_SWE="${DATA_DIR}/trail_derived/onsets_swe_train.jsonl"
OUT_ONSETS_COMBINED="${DATA_DIR}/trail_derived/onsets_combined_train.jsonl"

# Final graph output
OUT_DIR="${DATA_DIR}/trail_causal_outputs_train"

# ---- Create output directories -----------------------------------------------
mkdir -p "$(dirname "$OUT_FILTERED_GAIA")"
mkdir -p "$(dirname "$OUT_FILTERED_SWE")"
mkdir -p "$(dirname "$OUT_SPAN_ORDER_GAIA")"
mkdir -p "$(dirname "$OUT_SPAN_ORDER_SWE")"
mkdir -p "$(dirname "$OUT_ONSETS_GAIA")"
mkdir -p "$OUT_DIR"

# ---- Validate prerequisites --------------------------------------------------
for f in "$GAIA_TRAIN_IDS" "$SWE_TRAIN_IDS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required split file not found: $f"
        echo "  GAIA:      python graph/01_make_splits.py"
        echo "  SWE-bench: python graph/01_make_splits.py --datasets swe_bench \\"
        echo "                 --out_dir graph/splits_swe/ --train_ratio 0.70 --val_ratio 0.15"
        exit 1
    fi
done

# ---- GAIA: preprocess train traces -------------------------------------------

echo "=== [GAIA] Step 1: Filter to train trace IDs ==="
python causal_train/graph/preprocess/trail_1_filter_split.py \
    --data_dir         "$DATA_DIR" \
    --annotation_dir   "$BENCH_DIR" \
    --out_path         "$OUT_FILTERED_GAIA" \
    --split            GAIA \
    --include_ids      "$GAIA_TRAIN_IDS"

echo ""
echo "=== [GAIA] Step 2: Build span order ==="
python causal_train/graph/preprocess/trail_2_build_span_order.py \
    --filtered_path    "$OUT_FILTERED_GAIA" \
    --out_path         "$OUT_SPAN_ORDER_GAIA"

echo ""
echo "=== [GAIA] Step 3: Build onsets ==="
python causal_train/graph/preprocess/trail_3_build_onsets.py \
    --filtered_path    "$OUT_FILTERED_GAIA" \
    --span_order_path  "$OUT_SPAN_ORDER_GAIA" \
    --out_path         "$OUT_ONSETS_GAIA"

# ---- SWE-bench: preprocess train traces --------------------------------------

echo ""
echo "=== [SWE-bench] Step 1: Filter to train trace IDs ==="
python causal_train/graph/preprocess/trail_1_filter_split.py \
    --data_dir         "$DATA_DIR" \
    --annotation_dir   "$BENCH_DIR" \
    --out_path         "$OUT_FILTERED_SWE" \
    --split            "SWE Bench" \
    --include_ids      "$SWE_TRAIN_IDS"

echo ""
echo "=== [SWE-bench] Step 2: Build span order ==="
python causal_train/graph/preprocess/trail_2_build_span_order.py \
    --filtered_path    "$OUT_FILTERED_SWE" \
    --out_path         "$OUT_SPAN_ORDER_SWE"

echo ""
echo "=== [SWE-bench] Step 3: Build onsets ==="
python causal_train/graph/preprocess/trail_3_build_onsets.py \
    --filtered_path    "$OUT_FILTERED_SWE" \
    --span_order_path  "$OUT_SPAN_ORDER_SWE" \
    --out_path         "$OUT_ONSETS_SWE"

# ---- Merge -------------------------------------------------------------------

echo ""
echo "=== Step 4: Merge GAIA-train + SWE-train onsets ==="
python causal_train/graph/preprocess/merge_onsets.py \
    --inputs   "$OUT_ONSETS_GAIA" "$OUT_ONSETS_SWE" \
    --out_path "$OUT_ONSETS_COMBINED"

# ---- CAPRI pipeline ----------------------------------------------------------

echo ""
echo "=== Step 5: Run Suppes + CAPRI on combined train onsets ==="
python causal_train/graph/run_causal_from_trail_onsets.py \
    --onsets_path      "$OUT_ONSETS_COMBINED" \
    --out_dir          "$OUT_DIR" \
    --skip_bootstrap \
    --skip_shuffle

echo ""
echo "======================================================"
echo "Done. New causal graph (train-only, GAIA+SWE) at:"
echo "  ${OUT_DIR}/suppes_graph.json"
echo ""
echo "Next: rebuild GNN graph data with the new Suppes graph:"
echo "  python graph/build_graph_data.py \\"
echo "      --suppes_path ${OUT_DIR}/suppes_graph.json \\"
echo "      --out_dir graph/outputs_train/"
echo ""
echo "Then rebuild graph_input.pt:"
echo "  python graph/04_build_graph_input.py --data_dir graph/data/"
echo "  (after copying graph/outputs_train/ -> graph/outputs/ or updating build_graph_data.py paths)"
echo "======================================================"
