# EMNLP Main Table — Experiment TODO

All commands run from `benchmarking/` unless noted.
Target directory for all new runs: `outputs/zero_shot2/`

---

## Observational Graph Threshold Experiments (geomean scoring)

The non-causal graph path uses `score = sqrt(P(B|A) × PR_delta)` (geometric mean)
as both the injected edge weight and the filter criterion.
Both MAST and TRAIL use **geomean >= threshold** consistently for observational edges.

### Edge landscape (from suppes_graph.json)

| geomean threshold | N edges | Categories covered |
|---|---|---|
| causal_only (12 edges) | 12 | 11/20 |
| >= 0.30 | ~15 | 12/20 |
| >= 0.20 | 18 | 12/20 (+Environment Setup Errors) |
| >= 0.15 | ~20 | 13/20 |
| >= 0.10 (current default) | 21 | 13/20 (+Environment Setup Errors, +Task Orchestration) |
| >= 0.05 | ~25 | 13/20 |

7 categories are structurally uncoverable at any threshold (no co-occurrence data):
Instruction Non-compliance, Tool Definition Issues, Rate Limiting,
Service Errors, Resource Not Found, Resource Exhaustion, Timeout Issues.

### T-Obs-3 — Mistral GAIA_dedup, observational edges  ✓ done

Both [O3b] (`+GI causal + corr ≥ 0.20 + span index`,
W-F1 = 36.93) and [O3a] (`+CG observational, geomean ≥ 0.20`,
W-F1 = 29.33 GAIA / 9.31 SWE) are scored. Numbers folded into
`paper/ablation_graph_richness.tex` (Mistral row).

---

## NEW: Threshold Sweep — graph-richness × context-budget plot

Goal: characterise how W-F1 scales with the size of the injected graph,
and identify the threshold at which context-overflow failures (drops in
trace coverage `N`) start to dominate. Output is a 3-panel plot:
W-F1 vs τ, `N` vs τ, and Pass-2 prompt tokens vs τ.

### Sweep spec

- **Method**: `+GI+SI` only (two-pass; one-pass `+CG` is a separate ablation).
- **Thresholds**: `{causal-only, 0.30, 0.20, 0.15, 0.10, 0.05}` — 6 points.
- **Models** (priority order):
  1. **GPT-oss-20B** — largest corr0.2 gain on GAIA (+9.75 W-F1) → tests upside.
  2. **QwenLong-L1-32B** — only model with confirmed context fragility
     (corr0.2 dir came back empty) → tests where context budget breaks.
  3. *Optional:* **Gemma-3-27B-IT** — only model where corr0.2 *regresses*
     (-4.31 GAIA, -2.41 SWE) → tests whether the regression is monotonic
     in τ (graph too big) or specific to a single edge added at 0.20.
- **Splits**: GAIA_dedup only. SWE-Bench `N≤22` is too noisy for curve shape.
- **Total runs (priority 1+2)**: 2 models × 6 thresholds = 12 runs.

### Driver script

`eval/run_threshold_sweep.sh` runs all 6 thresholds for one
(model, split) pair and triggers scoring at the end:

```bash
# Usage: eval/run_threshold_sweep.sh <model> <split> [gpus] [output_dir] [backend]

# [TS-1] GPT-oss-20B GAIA_dedup
eval/run_threshold_sweep.sh openai/gpt-oss-20b GAIA_dedup 0,1

# [TS-2] QwenLong-L1-32B GAIA_dedup
#   NOTE: debug the t=0.20 failure first (empty output dir in outputs_corr/);
#   if context overflow is the cause, the sweep itself will quantify it.
eval/run_threshold_sweep.sh Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3

# [TS-3] (optional) Gemma-3-27B-IT GAIA_dedup
eval/run_threshold_sweep.sh openai/gemma-3-27b-it GAIA_dedup 0,1
```

The script writes per-threshold logs to
`outputs/zero_shot2/_sweep_logs/<model>-<split>-t<τ>.log`,
emits one `*-metrics.txt` per threshold, and runs
`calculate_scores.py` at the end.

### Plot

After the sweep completes, parse W-F1 / Loc / Joint / `N` from the
metrics files and plot τ on the x-axis (descending: causal-only on the
left, 0.05 on the right). Three panels: W-F1, `N`, and median Pass-2
prompt tokens (the third requires either logging in the inject script
or post-hoc tokenisation of the saved prompts).

---

---

## Outstanding gaps in `paper/ablation_graph_richness.tex`

Two cells in the TRAIL ablation table are not currently sourced from
`benchmarking/outputs_corr/`. Both need verification / re-runs before
the table can be locked.

### [G1] Locate or re-run Mistral +CG (suppes $t{=}0.2$) for both splits

The Mistral `+CG (suppes, t≥0.2)` row in
`paper/ablation_graph_richness.tex` cites W-F1 = 29.33 (GAIA) and
9.31 (SWE), but no metrics file under
`benchmarking/outputs_corr/` matches the Mistral one-pass suppes
condition. The TODO entry T-Obs-3 records this as done — likely
written to a different output directory. Action:

```bash
# 1. Search for any Mistral suppes-t0.2 metrics file in the repo
find . -type f -name "*Mistral*graph_suppes_t0.2*-metrics.txt" \
       -o -name "*Mistral*-graph-codename-t0.2*-metrics.txt" 2>/dev/null

# 2. If none found, re-run from benchmarking/ (one-pass +CG, suppes t=0.2)
python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup \
    --edge_threshold 0.2 \
    --output_dir outputs_corr

python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup \
    --edge_threshold 0.2 \
    --output_dir outputs_corr

# 3. Score
python eval/calculate_scores.py --results_dir outputs_corr
```

### [G2] Re-run QwenLong-L1-32B corr0.2 (+GI+SI) on both splits

The QwenLong corr0.2 dir
(`outputs_corr/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_dedup-graph_t0.2_span_index/`)
exists but is empty (0 outputs). Confirmed root cause: the corr-ablation
script `eval/run_eval_graph_inject_vllm.py` reuses the same buggy
JSON parser as the Who&When baseline runner — no Harmony stripping, no
balanced-brace JSON extractor, no open-`<think>` recovery, and no
reasoning-model `max_tokens` bump. QwenLong's reasoning chain blows past
`max_new_tokens`, the JSON is truncated mid-string, and `parse_json_output`
returns `None` for every trace. Action (in order):

```bash
# 1. Port the same 3 fixes from baselines/who_and_when/run_who_and_when_vllm.py
#    into benchmarking/eval/run_eval_graph_inject_vllm.py:
#      (a) _strip_harmony helper
#      (b) _balanced_json_object helper used by parse_json_output
#      (c) auto-bump max_new_tokens 8000 -> 24000 for reasoning models
#          matching r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)"

# 2. Re-run from benchmarking/ once parser is fixed
CUDA_VISIBLE_DEVICES=2,3 python eval/run_eval_graph_inject_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split GAIA_dedup \
    --causal_only --corr_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --max_model_len 32768 \
    --output_dir outputs_corr

CUDA_VISIBLE_DEVICES=2,3 python eval/run_eval_graph_inject_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split SWE_Bench_dedup \
    --causal_only --corr_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --max_model_len 32768 \
    --output_dir outputs_corr

# 3. Score
python eval/calculate_scores.py --results_dir outputs_corr
```

Note: gpt-oss runs in `outputs_corr/` are non-empty and produced
non-trivial W-F1, suggesting the parser bug only catastrophically breaks
reasoning models (long `<think>` chains that exceed the token budget).
For gpt-oss the Harmony channel leakage may still be silently degrading
quality — worth re-running them after the parser fix to quantify.

---

## Scoring

```bash
# From benchmarking/
python eval/calculate_scores.py --results_dir outputs/zero_shot2
```

## Completed (for reference)

| Run | Status |
|-----|--------|
| Flash GAIA all 4 conditions (zero_shot/) | ✓ |
| Flash GAIA_dedup no-graph (zero_shot2/) | ✓ |
| Flash SWE no-graph, causal_only (zero_shot/) | ✓ |
| Flash SWE causal_only+SI, graph_inject+SI (zero_shot2/, 31 traces) | ✓ |
| Mistral GAIA_dedup all 4 conditions (zero_shot2/) | ✓ |
| Mistral SWE_Bench_dedup no-graph + graph_inject+SI (zero_shot2/) | ✓ |
| Pro P1–P4 (GAIA_dedup + SWE_Bench_dedup, baseline + +GI+SI causal-only) | ✓ |
| T-Obs-3 Mistral GAIA_dedup +GI corr≥0.20+SI and +CG suppes t=0.20 (both splits) | ✓ |
| corr0.2 +GI+SI for {Mistral, Gemma, GPT-oss-120B, GPT-oss-20B} on GAIA+SWE (outputs_corr/) | ✓ |
| suppes t=0.20 +CG for {Gemma, GPT-oss-120B, GPT-oss-20B} on GAIA (outputs_corr/) | ✓ |

---

## Gemini-2.5-Pro — optional one-pass conditions (NOT for ablation study)

The ablation study (`paper/ablation_graph_richness.tex`) is scoped to
open-source models only — Gemini Flash / Pro are not part of it.
The conditions below remain listed only for the main results table,
in case the paper later needs a full one-pass Pro row alongside the
already-done P1–P4 (baseline + `+GI+SI` causal-only).

Model ID (litellm): `gemini/gemini-2.5-pro`

```bash
# [P5] GAIA_dedup causal-only (one-pass +CG)
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only \
    --output_dir outputs/zero_shot2

# [P6] GAIA_dedup causal-only + span index (+CG+SI)
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [P7] SWE_Bench_dedup causal-only (one-pass +CG)
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only \
    --output_dir outputs/zero_shot2

# [P8] SWE_Bench_dedup causal-only + span index (+CG+SI)
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2
```