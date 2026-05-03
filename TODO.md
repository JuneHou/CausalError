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