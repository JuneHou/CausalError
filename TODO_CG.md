# TODO — +CG main-results gap fill (TRAIL)

**Goal**: add a +CG row at τ=0.35 (corr-union 19 edges, `graph_causal_corr0.35`)
to `main_results_trail.tex` for all 5 open-source models, so each model block has
3 rows: Baseline / +CG(τ=0.35) / +GI(τ=0.35) (= \our).

**Current state**: `benchmarking/outputs_thres_cg/t0.35/` covers Gemma, GPT-oss-120B,
and GPT-oss-20B with the correct corr-union graph. Two models are missing or wrong.

Closed-source (Gemini-Flash, Gemini-Pro) is out of scope for this file.

## Missing cells — 4 runs

| # | Model | Split | Backend | Note |
|---|---|---|---|---|
| 1 | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | GAIA_dedup | DeepInfra | the existing τ=0.35 cells use Mistral-**3.2**-24B-2506; main table needs 3.1 |
| 2 | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | SWE_Bench_dedup | DeepInfra | same |
| 3 | Tongyi-Zhiwen/QwenLong-L1-32B | GAIA_dedup | vLLM | existing dir is `graph_t0.35` (pure Suppes 15 edges via `--edge_threshold`), not the 19-edge corr-union; needs re-run with `--corr_threshold 0.35` |
| 4 | Tongyi-Zhiwen/QwenLong-L1-32B | SWE_Bench_dedup | vLLM | same |

## Runner gap — QwenLong (vLLM)

`benchmarking/eval/run_eval_with_graph_vllm.py` does **not** currently support
`--corr_threshold` (only `--edge_threshold` and `--causal_only`). Three options:

- **(a) Extend the runner** (recommended): mirror the `--corr_threshold` block
  from `run_eval_with_graph_api_deepinfra.py` (lines 178+) into
  `run_eval_with_graph_vllm.py`. ~10 lines of arg + graph-loading code.
- (b) Accept pure-Suppes graph: use existing `graph_t0.35` runs (15 edges, no
  causal union) — drops the apples-to-apples comparison with the +GI row.
- (c) Run QwenLong via DeepInfra instead of vLLM (cheaper one-off, no code change).

## Commands

```bash
# ============================================================
# (1, 2) Mistral-Small-3.1-24B via DeepInfra — 2 cells
# Auth: export DEEPINFRA_API_KEY=<key>
# Output lands in: outputs_thres_cg/t0.35/outputs_<model_tag>-<split>-graph_causal_corr0.35/
# ============================================================
cd benchmarking
for split in GAIA_dedup SWE_Bench_dedup; do
  python eval/run_eval_with_graph_api_deepinfra.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split "$split" \
    --corr_threshold 0.35 \
    --output_dir outputs_thres_cg/t0.35
done

# ============================================================
# (3, 4) QwenLong-L1-32B via vLLM — 2 cells (option a: after extending runner)
# IMPORTANT: --tensor_parallel_size 4 on A40s (KV cache otherwise OOMs at tp=2)
# ============================================================
for split in GAIA_dedup SWE_Bench_dedup; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/run_eval_with_graph_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split "$split" \
    --tensor_parallel_size 4 \
    --corr_threshold 0.35 \
    --output_dir outputs_thres_cg/t0.35
done
```

## Expected output paths

```
benchmarking/outputs_thres_cg/t0.35/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-{GAIA_dedup,SWE_Bench_dedup}-graph_causal_corr0.35/
benchmarking/outputs_thres_cg/t0.35/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-{GAIA_dedup,SWE_Bench_dedup}-graph_causal_corr0.35/
```

## Scoring + table integration

```bash
# Per run:
python eval/calculate_scores.py --pred_dir <output_path>
```

After all 4 cells finish:
- Pull metrics into a new `+CG (τ=0.35)` row per model block in
  `/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/tables/main_results_trail.tex`
- Re-rank bold/underline per column

## Cross-reference

MAST has 2 parallel missing cells (Mistral-3.1 + QwQ-32B). See
`/data/wang/junh/githubs/MAST/TODO_CG.md`. Combined total: **6 runs across both
benchmarks**.
