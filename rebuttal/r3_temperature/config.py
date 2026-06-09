"""
Shared paths and constants for the R3 temperature-robustness experiment.

Scope (locked 2026-05-31):
  - 5 open-weight backbones from Table 1: Mistral-Small-3.1-24B (sbatch),
    GPT-oss-120B (ARC API), GPT-oss-20B (sbatch), Gemma-3-27B-IT (sbatch),
    QwenLong-L1-32B (sbatch). Closed-weight backbones deferred.
  - Both benchmarks at full corpus (not held-out): TRAIL combined + MAST 393.
  - Main-paper graph at tau=0.35 (no rebuild).
  - Temperature 0.7, 3 i.i.d. samples per cell via independent invocations.
  - Compare against existing temp=0 Table 1 anchor.
"""

from pathlib import Path

# ---------- repo roots ----------
# Prefer the ARC cluster layout; fall back to the local checkout (this file lives
# at <repo>/rebuttal/r3_temperature/config.py) so aggregation can run anywhere.
_ARC_TRAIL_ROOT = Path("/projects/slmreasoning/junh/causal-error/CausalError")
if _ARC_TRAIL_ROOT.exists():
    TRAIL_ROOT = _ARC_TRAIL_ROOT
    MAST_ROOT  = Path("/projects/slmreasoning/junh/causal-error/CausalMAST")
else:
    TRAIL_ROOT = Path(__file__).resolve().parents[2]
    # The MAST repo may be checked out under a few different names locally.
    MAST_ROOT  = next(
        (c for c in (TRAIL_ROOT.parent / "CausalMAST", TRAIL_ROOT.parent / "MAST") if c.exists()),
        TRAIL_ROOT.parent / "CausalMAST",   # absent => MAST aggregation reuses precomputed -metrics.json
    )
R3_ROOT    = TRAIL_ROOT / "rebuttal" / "r3_temperature"

# ---------- experiment dials ----------
TEMPERATURE = 0.7
N_SAMPLES   = 3
CORR_THRESHOLD = 0.35

# ---------- eval entry points ----------
# TRAIL
TRAIL_BASELINE_SCRIPT_VLLM = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval_vllm.py"
TRAIL_BASELINE_SCRIPT_API  = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval.py"          # litellm; configure ARC endpoint via env vars
TRAIL_EDGE_SCRIPT_VLLM     = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval_graph_inject_vllm.py"
TRAIL_EDGE_SCRIPT_ARC      = TRAIL_ROOT / "benchmarking" / "eval" / "run_eval_graph_inject_api_arc.py"
TRAIL_SCORER               = TRAIL_ROOT / "benchmarking" / "eval" / "calculate_scores.py"

# MAST
MAST_BASELINE_SCRIPT_VLLM  = MAST_ROOT / "eval" / "run_eval_yesno_vllm.py"
MAST_BASELINE_SCRIPT_API   = MAST_ROOT / "eval" / "run_eval_yesno_api.py"
MAST_EDGE_SCRIPT_VLLM      = MAST_ROOT / "eval" / "run_eval_graph_inject.py"
MAST_EDGE_SCRIPT_ARC       = MAST_ROOT / "eval" / "full_run_eval_graph_inject_api_arc.py"
MAST_SCORER                = MAST_ROOT / "eval" / "calculate_scores_yesno.py"

# ---------- main-paper graph artifacts (no rebuild for R3) ----------
# TRAIL: tau=0.35 corr-union ∪ intervention-validated edges.
TRAIL_SUPPES_GRAPH = TRAIL_ROOT / "benchmarking" / "data" / "trail_causal_outputs_full_gaia_swe_AIC" / "suppes_graph.json"
TRAIL_EFFECT_EDGES = TRAIL_ROOT / "benchmarking" / "outputs" / "interventions_full_gaia_swe_merged" / "effect_edges.json"

# MAST: tau=0.35 corr-union (the "+EDGE" headline in MAST sbatch uses --edge_threshold 0.35
# with these two graph files; see rebuttal/holdout/sbatch/run_holdout_gpt_oss_120b_mast.sbatch for layout).
MAST_SUPPES_GRAPH  = MAST_ROOT / "causal_graph" / "outputs" / "suppes_graph.json"
MAST_EFFECT_EDGES  = MAST_ROOT / "causal_graph" / "outputs" / "interventions" / "effect_edges.json"

# ---------- TRAIL data dirs (full corpus) ----------
TRAIL_DATA_DIR = TRAIL_ROOT / "benchmarking" / "data"
TRAIL_GAIA_SPLIT = "GAIA_dedup"   # 117 traces
TRAIL_SWE_SPLIT  = "SWE_Bench_dedup"  # 31 traces

# ---------- MAST input ----------
MAST_ANNOTATION = MAST_ROOT / "data" / "annotation" / "annotation_ag2_filtered.jsonl"

# ---------- output predictions root ----------
PREDICTIONS_DIR = R3_ROOT / "data" / "predictions"
RESULTS_DIR     = R3_ROOT / "results"

# ---------- backbones ----------
# `model_tag` MUST match the existing Table-1 dir naming convention so the
# temp=0 anchor and temp=0.7 reruns line up under the same identifier.
#
# `backend`:
#   - "vllm"   = local sbatch + in-process vLLM (open-weight)
#   - "api"    = ARC API via OpenAI-style client
#
# `vllm_arc_path` is the path on ARC's shared models filesystem (used in sbatch
# scripts). `hf_name` is the HuggingFace identifier (used as --model in vLLM).
BACKBONES = {
    "mistral-24b": {
        "model_tag": "mistralai-Mistral-Small-3.1-24B-Instruct-2503",
        "hf_name":   "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "vllm_arc_path": "/common/data/models/mistralai--Mistral-Small-3.1-24B-Instruct-2503",
        "backend": "vllm",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.75,
        "max_model_len_trail": 131072,
        "max_model_len_mast":  32768,
    },
    "gpt-oss-120b": {
        "model_tag": "openai-gpt-oss-120b",
        "hf_name":   "gpt-oss-120b",  # no openai/ prefix on ARC endpoint
        "vllm_arc_path": None,
        "backend": "api",
    },
    "gpt-oss-20b": {
        "model_tag": "openai-gpt-oss-20b",
        "hf_name":   "openai/gpt-oss-20b",
        "vllm_arc_path": "/common/data/models/openai--gpt-oss-20b",
        "backend": "vllm",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.75,
        "max_model_len_trail": 131072,
        "max_model_len_mast":  32768,
    },
    "gemma-3-27b": {
        "model_tag": "openai-gemma-3-27b-it",  # matches main-paper dir naming
        "hf_name":   "google/gemma-3-27b-it",
        "vllm_arc_path": "/common/data/models/google--gemma-3-27b-it",
        "backend": "vllm",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.75,
        "max_model_len_trail": 131072,   # gemma-3-27b supports 128K; matches the other backbones
        "max_model_len_mast":  32768,
    },
    "qwenlong-32b": {
        "model_tag": "Tongyi-Zhiwen-QwenLong-L1-32B",
        "hf_name":   "Tongyi-Zhiwen/QwenLong-L1-32B",
        "vllm_arc_path": "/common/data/models/Tongyi-Zhiwen--QwenLong-L1-32B",
        "backend": "vllm",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.75,
        "max_model_len_trail": 131072,
        "max_model_len_mast":  32768,
    },
}

# ---------- benchmark configs ----------
BENCHMARKS = {
    "trail": {
        "splits": [TRAIL_GAIA_SPLIT, TRAIL_SWE_SPLIT],   # run both, pool by N
        "suppes_graph": str(TRAIL_SUPPES_GRAPH),
        "effect_edges": str(TRAIL_EFFECT_EDGES),
    },
    "mast": {
        "input": str(MAST_ANNOTATION),
        "suppes_graph": str(MAST_SUPPES_GRAPH),
        "effect_edges": str(MAST_EFFECT_EDGES),
    },
}


def output_dir(benchmark: str, backbone: str, variant: str, sample_idx: int, split: str = None) -> Path:
    """Canonical layout for a single (cell, sample) prediction folder.

      data/predictions/<bench>/<backbone>/<variant>/temp{T}_sample{N}/
          for MAST and for TRAIL (we keep both splits in subfolders below).
      data/predictions/trail/<backbone>/<variant>/temp{T}_sample{N}/<split>/
    """
    base = PREDICTIONS_DIR / benchmark / backbone / variant / f"temp{TEMPERATURE}_sample{sample_idx}"
    if split is not None:
        base = base / split
    return base
