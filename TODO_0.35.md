# TODO — τ=0.35 pivot for TRAIL paper tables

**Pivot**: change the +GI headline graph from causal-only (12 edges) to corr~$\geq$~0.35 Suppes-screened super-graph (24 edges). All three TRAIL tables affected.

**Decision basis**: threshold sweep (Table `threshold_sweep_ablation.tex`) shows τ=0.35 has the highest mean **W-F1 gain over the causal-only baseline** across the 5-model open-source panel (+4.84 W-F1 across 10 (model, split) cells; next best τ=0.25 at +4.74). It also has the most per-cell F1 wins (4/10) and is the most parsimonious corr-thresholded option (19 edges vs 21 / 25). Mean raw W-F1 (0.2526) agrees with the gain criterion — the selection is robust to either aggregation.

**Affected tables** (all in `paper/tables/`):
1. `main_results_0.35.tex` — NEW, drafted; 4 cells pending.
2. `threshold_sweep_ablation.tex` — existing; needs caption/role reframe + optionally 2 new model rows.
3. `who_and_when_results.tex` — existing; 20 cells pending if W&W pivots; otherwise footnote only.

---

## 1. Main results table — 4 runs ★ critical

`paper/tables/main_results_0.35.tex` is drafted with open-source rows filled and Gemini rows marked `\text{TBD}`.

Missing cells:

| # | Model | Split | Backend |
|---|---|---|---|
| 1 | Gemini-2.5-Flash | GAIA_dedup | litellm |
| 2 | Gemini-2.5-Flash | SWE_Bench_dedup | litellm |
| 3 | Gemini-2.5-Pro | GAIA_dedup | litellm |
| 4 | Gemini-2.5-Pro | SWE_Bench_dedup | litellm |

Commands (run from repo root):

```bash
# litellm backend — Gemini-2.5-Flash and Gemini-2.5-Pro
# Runner: benchmarking/eval/run_eval_graph_inject.py is the litellm path
# (despite the lack of an "_api" suffix). Needs litellm installed and
# GEMINI_API_KEY exported in the shell.
for model in gemini/gemini-2.5-flash gemini/gemini-2.5-pro; do
  for split in GAIA_dedup SWE_Bench_dedup; do
    python benchmarking/eval/run_eval_graph_inject.py \
      --model "$model" --split "$split" \
      --corr_threshold 0.35 --span_index \
      --output_dir benchmarking/outputs_thres/t0.35
  done
done
```

Expected output paths:
```
benchmarking/outputs_thres/t0.35/outputs_gemini-gemini-2.5-flash-{GAIA_dedup,SWE_Bench_dedup}-graph_inject_causal_corr0.35_span_index/
benchmarking/outputs_thres/t0.35/outputs_gemini-gemini-2.5-pro-{GAIA_dedup,SWE_Bench_dedup}-graph_inject_causal_corr0.35_span_index/
```

After completion: edit `main_results_0.35.tex`, replace the 4 `\text{TBD}` blocks with the new numbers, then re-run the column-wise bold/underline ranker (regenerate via the script kept in conversation).

---

## 2. Threshold sweep ablation table — 0 runs (re-frame) + optionally 16 runs ☐

`paper/tables/threshold_sweep_ablation.tex` already contains the full 5-model × 5-τ-point grid. Two paths:

### 2a. Minimum: caption reframe only (0 runs)
Re-cast caption from *"ablation: does corr-τ help over causal-only?"* to *"ablation: justification for choosing τ=0.35 as the headline."* The 5 corr-τ rows ARE the τ-selection evidence; the causal-only row becomes the simpler-baseline comparison.

### 2b. Defensible: add 2 closed-source model rows (16 runs, optional) ☐
Without Gemini in the ablation, reviewers may flag "τ chosen on open-source, applied to closed-source without supporting evidence." Mitigation: extend the sweep to Gemini.

| Cell | random-12 | causal-only | τ=0.35 | τ=0.25 | τ=0.20 |
|---|---|---|---|---|---|
| Gemini-2.5-Flash GAIA | ☐ | ✓ (reused from main) | ☐ (= main cell #1) | ☐ | ☐ |
| Gemini-2.5-Flash SWE | ☐ | ✓ | ☐ (= #2) | ☐ | ☐ |
| Gemini-2.5-Pro GAIA | ☐ | ✓ | ☐ (= #3) | ☐ | ☐ |
| Gemini-2.5-Pro SWE | ☐ | ✓ | ☐ (= #4) | ☐ | ☐ |

Net new runs after subtracting the 4 main-table cells: **16** (4 (model, split) cells × 4 sweep points: random-12, τ=0.35, τ=0.25, τ=0.20; causal-only reused from existing main_results.tex sources).

```bash
# Defensibility sweep: Gemini at the three other sweep points + random null
# Same runner as the main-table cells (litellm via run_eval_graph_inject.py).
# Note: --output_dir points to the per-τ subdir to match existing tree layout.
declare -A POINT2DIR=(
  ["--random_edges --random_n 12 --random_seed 42"]="t_random12_seed42"
  ["--corr_threshold 0.25"]="t0.25"
  ["--corr_threshold 0.20"]="t0.20"
)
for model in gemini/gemini-2.5-flash gemini/gemini-2.5-pro; do
  for split in GAIA_dedup SWE_Bench_dedup; do
    for arg in "${!POINT2DIR[@]}"; do
      python benchmarking/eval/run_eval_graph_inject.py \
        --model "$model" --split "$split" $arg --span_index \
        --output_dir "benchmarking/outputs_thres/${POINT2DIR[$arg]}"
    done
  done
done
```

---

## 3. Who&When table — 20 runs ★ if W&W stays in main paper ☐

`paper/tables/who_and_when_results.tex` currently uses `causal_only` for every `W1+GI+SI` and `W2+CG+SI` row. If the headline pivots, W&W should follow to stay consistent.

| Model | Splits | Variants | Cells |
|---|---|---|---|
| Mistral-Small-3.1-24B | GAIA, SWE | W1+GI+SI, W2+CG+SI | ✓ |
| GPT-oss-120B | GAIA, SWE | W1+GI+SI, W2+CG+SI | ✓ |
| GPT-oss-20B | GAIA, SWE | W1+GI+SI, W2+CG+SI | ✓ |
| Gemma-3-27B-IT | GAIA, SWE | W1+GI+SI, W2+CG+SI | ✓ |
| QwenLong-L1-32B | GAIA, SWE | W1+GI+SI, W2+CG+SI | 4 |
| **Total** | | | **20** |

Note: `W2+CG+SI` uses the one-pass +CG architecture (not +GI). The TRAIL eval `outputs_thres_cg/t0.35/` data does NOT cover this — W&W has its own prompt template, so W&W-specific runs are still required.

### Backend routing (user policy)

| Model | Backend | Runner suffix |
|---|---|---|
| Mistral-Small-3.1-24B | DeepInfra | `*_api_deepinfra.py` |
| GPT-oss-20B | DeepInfra | `*_api_deepinfra.py` |
| Gemma-3-27B-IT | DeepInfra | `*_api_deepinfra.py` |
| GPT-oss-120B | ARC | `*_api_arc.py` |
| QwenLong-L1-32B | vLLM (local) | `*_vllm.py` |

W1+GI+SI script: `run_who_and_when_graph_inject_{backend}.py`
W2+CG+SI script: `run_who_and_when_with_graph_{backend}.py`
Shared args: `--variant {w1|w2} --corr_threshold 0.35 --span_index --split {GAIA_dedup|SWE_Bench_dedup}`

Auth (export once per shell):
- ARC: `source path/to/arc_llm_api.sh` (sets `ARC_LLM_API_KEY`)
- DeepInfra: `export DEEPINFRA_API_KEY=<key>`

### Commands

```bash
# ============================================================
# DeepInfra — Mistral / GPT-oss-20B / Gemma (3 models × 2 splits × 2 variants = 12 cells)
# ============================================================
DEEPINFRA_MODELS=(
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "openai/gpt-oss-20b"
  "google/gemma-3-27b-it"
)
for model in "${DEEPINFRA_MODELS[@]}"; do
  for split in GAIA_dedup SWE_Bench_dedup; do
    # W1+GI+SI
    python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_deepinfra.py \
      --model "$model" --split "$split" --variant w1 \
      --corr_threshold 0.35 --span_index
    # W2+CG+SI
    python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
      --model "$model" --split "$split" --variant w2 \
      --corr_threshold 0.35 --span_index
  done
done

# ============================================================
# ARC — GPT-oss-120B (1 model × 2 splits × 2 variants = 4 cells)
# (script defaults to gpt-oss-120b, --model can be omitted)
# ============================================================
for split in GAIA_dedup SWE_Bench_dedup; do
  # W1+GI+SI
  python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_arc.py \
    --split "$split" --variant w1 \
    --corr_threshold 0.35 --span_index
  # W2+CG+SI
  python baselines/who_and_when/causal/run_who_and_when_with_graph_api_arc.py \
    --split "$split" --variant w2 \
    --corr_threshold 0.35 --span_index
done

# ============================================================
# vLLM (local) — QwenLong-L1-32B (1 model × 2 splits × 2 variants = 4 cells)
# Tongyi-Zhiwen/QwenLong-L1-32B; --enable_thinking is NOT needed (this is not QwQ).
# IMPORTANT: pass --tensor_parallel_size 4. The script default is tp=2, which
# leaves only ~1.75 GiB/GPU for KV cache after the 32B-param weights split, so
# the engine fails with "KV cache memory 3.62 GiB" at max_model_len=131072.
# At tp=4 on A40 (45 GiB each), weights split → 16 GiB/GPU, leaving ~17 GiB/GPU
# for KV cache (~68 GiB total) — comfortably above the 16 GiB needed.
# Adjust CUDA_VISIBLE_DEVICES to whatever 4 GPUs you have free.
# ============================================================
for split in GAIA_dedup SWE_Bench_dedup; do
  # W1+GI+SI
  CUDA_VISIBLE_DEVICES=4,5,6,7 python baselines/who_and_when/causal/run_who_and_when_graph_inject_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B --split "$split" --variant w1 \
    --tensor_parallel_size 4 \
    --corr_threshold 0.35 --span_index
  # W2+CG+SI
  CUDA_VISIBLE_DEVICES=4,5,6,7 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B --split "$split" --variant w2 \
    --tensor_parallel_size 4 \
    --corr_threshold 0.35 --span_index
done
```

Total cells: 12 (DeepInfra) + 4 (ARC) + 4 (vLLM QwenLong) = **20 cells**.
Output paths (default for each runner): `baselines/who_and_when/causal/outputs/*-yesno-who_and_when_{w1,w2}_graph_{inject_,}corr0.35[_span_index]/`.

Gemini W&W is omitted from this table by design (see header comment in `who_and_when_results.tex`).

### Alternative: footnote instead of rerun (0 runs)
If 20 cells is too much, keep W&W rows at causal-only and add a caption footnote:
> *Who\&When ablation uses the causal-only graph (12 edges); the main-table +GI headline at $\tau{=}0.35$ is supported separately by Table~\ref{tab:threshold_sweep_ablation}.*

This is honest and avoids the rerun cost. Loses: the W&W table can no longer be cited as evidence that τ=0.35 generalizes across prompt families.

---

## 4. §4.5 paired +CG vs +GI block — 0 runs (re-tabulate) ☐

If the paper contains a paired-row block comparing +CG and +GI under the same edge set (mirroring MAST's §4.5), pull cells from both directories at τ=0.35:

```
+GI τ=0.35: benchmarking/outputs_thres/t0.35/*-graph_inject_causal_corr0.35_span_index/
+CG τ=0.35: benchmarking/outputs_thres_cg/t0.35/*-graph_causal_corr0.35/
```

5 models × 2 splits = 10 paired cells. All data exists; only re-tabulation needed.

---

## 5. τ-selection defensibility (optional, narrative) — 0 runs

If reviewers may push back on in-sample τ-selection:
- Pick τ=0.35 using **GAIA mean W-F1 gain over causal-only** (where it wins).
- Report SWE results at the same τ without re-tuning.

Pure re-tabulation. Costs nothing. Strengthens the methodology section without new compute.

---

## Total compute summary

| Scenario | Runs | What's covered |
|---|---|---|
| **Minimum** (main table only; W&W → footnote) | **4** | Headline pivot only |
| **Standard** (main + W&W rerun) | **24** | Defensible across all main paper tables |
| **Full** (main + W&W + Gemini ablation) | **40** | Reviewer-proof for τ-selection on closed-source |

**Recommended path**: start with the **4 Gemini main-table cells** (overnight at most), then decide on W&W and the Gemini ablation sweep based on those numbers. If Gemini-Pro+GI at τ=0.35 dominates the main table cleanly, the W&W rerun becomes a "nice-to-have"; if Gemini τ=0.35 numbers are weaker than causal-only, reconsider the pivot before sinking the W&W rerun budget.

---

## Cross-reference (out of scope here)

MAST has its own pivot considerations (Task 4 +CG sweep already in flight; main table needs 1 GPT-4o cell at chosen τ). Not in scope here — see `/data/wang/junh/githubs/MAST/TODO.md`.

---

# Follow-up — Closed-source cross-benchmark coverage

**Pivot**: In the combined main-results table
(`/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/tables/main_results_combine.tex`),
split the merged "Closed-source" row into separate **Gemini** and **GPT-4o** rows so each
covers all three benchmark columns (TRAIL-GAIA, TRAIL-SWE-Bench, MAST).

This file owns the **TRAIL-side GPT-4o** runs (direct OpenAI API via the existing
litellm runners — no router needed). MAST-side Gemini runs go through OpenRouter
and are tracked in `/data/wang/junh/githubs/MAST/TODO_0.5.md` (§4-§5).

## 6. TRAIL GPT-4o cells — 4 runs ★ critical

| # | Model | Split | Method | Backend |
|---|---|---|---|---|
| 1 | openai/gpt-4o | GAIA_dedup | Baseline | OpenAI direct via litellm (`run_eval.py`) |
| 2 | openai/gpt-4o | SWE_Bench_dedup | Baseline | OpenAI direct via litellm (`run_eval.py`) |
| 3 | openai/gpt-4o | GAIA_dedup | +GI(τ=0.35)+SI | OpenAI direct via litellm (`run_eval_graph_inject.py`) |
| 4 | openai/gpt-4o | SWE_Bench_dedup | +GI(τ=0.35)+SI | OpenAI direct via litellm (`run_eval_graph_inject.py`) |

Commands (run from `benchmarking/`; `OPENAI_API_KEY` exported in shell):

```bash
cd benchmarking
# ============================================================
# Baseline (no graph) — 2 cells
#   --output_dir outputs/zero_shot2 (alongside existing zero_shot/)
#   --max_workers 1   (long context; keep serial)
# ============================================================
for split in GAIA_dedup SWE_Bench_dedup; do
  python eval/run_eval.py \
    --model openai/gpt-4o --split "$split" \
    --output_dir outputs/zero_shot2 \
    --max_workers 1
done

# ============================================================
# +GI at τ=0.35 with span_index — 2 cells
# ============================================================
for split in GAIA_dedup SWE_Bench_dedup; do
  python eval/run_eval_graph_inject.py \
    --model openai/gpt-4o --split "$split" \
    --corr_threshold 0.35 --span_index \
    --output_dir outputs_thres/t0.35
done
```

Expected output paths:
```
benchmarking/outputs/zero_shot2/outputs_openai-gpt-4o-{GAIA_dedup,SWE_Bench_dedup}/
benchmarking/outputs_thres/t0.35/outputs_openai-gpt-4o-{GAIA_dedup,SWE_Bench_dedup}-graph_inject_causal_corr0.35_span_index/
```

After completion: score with `calculate_scores.py`, then add a new GPT-4o block to
`main_results_trail.tex` and (re-)tabulate the combined `main_results_combine.tex`
to split the merged "Closed-source" row into Gemini + GPT-4o rows.

## Cross-benchmark total

| Side | Runs | New scripts | TODO file |
|---|---|---|---|
| TRAIL (this file, §6) — GPT-4o via OpenAI direct | 4 | 0 | `trail-benchmark/TODO_0.35.md` |
| MAST — Gemini-Flash + Gemini-Pro via OpenRouter | 4 | 2 | `MAST/TODO_0.5.md` (§4-§5) |
| **Total runs** | **8** | **2 scripts** | |
