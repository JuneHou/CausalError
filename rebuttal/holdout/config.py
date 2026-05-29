"""
Shared paths and constants for the EDGE held-out (80/20) experiment.

Protocol locked 2026-05-27 (revised):
  - One stratified 80/20 split per benchmark, not k-fold.
  - TRAIL = combined GAIA + SWE (148 traces) -> one held-out cell per backbone.
  - MAST split by unique trace_id (142 unique IDs, ~393 records) so all
    instances of the same task are on the same side of the split.
  - Two backbones: Mistral-Small-3.1-24B and GPT-oss-120B.
  - Comparison axis = full-corpus F1 from main paper's existing runs,
    rescored over the same combined trace set.
"""

from pathlib import Path

# ---------- repo roots ----------
TRAIL_ROOT = Path("/data/wang/junh/githubs/trail-benchmark")
MAST_ROOT = Path("/data/wang/junh/githubs/MAST")
HOLDOUT_ROOT = TRAIL_ROOT / "rebuttal" / "holdout"

# ---------- source onset files ----------
TRAIL_ONSETS = TRAIL_ROOT / "benchmarking" / "data" / "trail_derived" / "onsets_gaia_swe_full.jsonl"
MAST_ONSETS = MAST_ROOT / "causal_graph" / "data" / "onsets.jsonl"

# ---------- source trace data ----------
TRAIL_GAIA_TRACE_DIR = TRAIL_ROOT / "benchmarking" / "data" / "GAIA_dedup"
TRAIL_SWE_TRACE_DIR = TRAIL_ROOT / "benchmarking" / "data" / "SWE_Bench_dedup"
MAST_ANNOTATION_FILE = MAST_ROOT / "data" / "annotation" / "annotation_ag2_filtered.jsonl"

# ---------- pipeline entry points ----------
TRAIL_SUPPES_SCRIPT = TRAIL_ROOT / "causal" / "graph" / "CAPRI" / "2_suppes_screen.py"
TRAIL_EVAL_SCRIPT_API = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval_graph_inject_api_arc.py"   # Virginia Tech ARC LLM API
TRAIL_EVAL_SCRIPT_VLLM = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval_graph_inject_vllm.py"     # local vLLM
TRAIL_SCORER = TRAIL_ROOT / "benchmarking" / "eval" / "calculate_scores.py"

MAST_SUPPES_SCRIPT = MAST_ROOT / "causal_graph" / "CAPRI" / "2_suppes_screen.py"
MAST_EVAL_SCRIPT_VLLM = MAST_ROOT / "eval" / "run_eval_graph_inject.py"                             # in-process vLLM
MAST_EVAL_SCRIPT_API = MAST_ROOT / "eval" / "full_run_eval_graph_inject_api_arc.py"                 # Virginia Tech ARC LLM API
MAST_SCORER = MAST_ROOT / "eval" / "calculate_scores_yesno.py"

# ---------- experiment outputs ----------
ASSIGN_DIR = HOLDOUT_ROOT / "data" / "assignments"
ONSETS_DIR = HOLDOUT_ROOT / "data" / "onsets_train"
TEST_TRACE_DIR = HOLDOUT_ROOT / "data" / "test_traces"
GRAPHS_DIR = HOLDOUT_ROOT / "data" / "graphs"
PREDICTIONS_DIR = HOLDOUT_ROOT / "data" / "predictions"
RESULTS_DIR = HOLDOUT_ROOT / "results"

# ---------- benchmark configs ----------
# TRAIL is combined GAIA + SWE (split_filter=None loads all rows).
# split_id_field tells build_folds which field defines the held-out unit:
#   - "trace_id" for TRAIL (each row is a unique trace)
#   - "trace_id" for MAST too, but MAST rows share trace_id across model/prompt
#     instances of the same task; we dedup so all instances land together.
BENCHMARKS = {
    "trail": {"source": "trail", "split_filter": None, "test_frac": 0.20},
    "mast":  {"source": "mast",  "split_filter": None, "test_frac": 0.20},
}

# ---------- backbones (HF model names) ----------
# `backend`: "vllm" = local in-process vLLM (Mistral); "api" = litellm routed to a hosted
# endpoint (GPT-oss-120B via ARC). Determines which TRAIL eval script to invoke and
# whether vLLM tensor-parallel + GPU-mem flags apply.
BACKBONES = {
    "mistral-24b": {
        "hf_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "backend": "vllm",
        "tensor_parallel_size": 4,        # 4 x A100 per the paper's compute table
        "gpu_memory_utilization": 0.75,
        "trail_baseline_dirs": [
            TRAIL_ROOT / "benchmarking" / "outputs" / "zero_shot2" /
                "outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-GAIA_dedup",
            TRAIL_ROOT / "benchmarking" / "outputs" / "zero_shot2" /
                "outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup",
        ],
        "mast_baseline_dir": MAST_ROOT / "causal_graph" / "outputs" /
            "mistralai-Mistral-Small-3.1-24B-Instruct-2503-baseline",
    },
    "gpt-oss-120b": {
        "hf_name": "gpt-oss-120b",        # NO openai/ prefix on ARC
        "backend": "api",                 # Virginia Tech ARC LLM API (https://llm-api.arc.vt.edu/api/v1/)
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "trail_baseline_dirs": [
            TRAIL_ROOT / "benchmarking" / "outputs" / "zero_shot2" /
                "outputs_openai-gpt-oss-120b-GAIA_dedup",
            TRAIL_ROOT / "benchmarking" / "outputs" / "zero_shot2" /
                "outputs_openai-gpt-oss-120b-SWE_Bench_dedup",
        ],
        "mast_baseline_dir": MAST_ROOT / "outputs" / "gpt-oss-120b-yesno-baseline",
    },
}

# ---------- threshold (matches paper headline) ----------
CORR_THRESHOLD = 0.35

# ---------- suppes screen params (match paper) ----------
SUPPES_PARAMS = {
    "min_precedence": 0.55,
    "min_pr_delta": 0.05,
    "min_joint": 3,
}

# ---------- min traces per category to enter rotation (pinning threshold) ----------
MIN_PER_CAT = 5

# ---------- random seed for split assignment ----------
SEED = 42
