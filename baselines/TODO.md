# Baselines — Who&When EMNLP Experiment Plan

This file plans the runs that complete the Who&When ablation in the paper.
The plan is keyed to the cells in `paper/main_results_table.tex` (Tables 1–4)
so the resulting numbers slot directly into the existing main table layout.

All commands run from `baselines/who_and_when/` unless noted.
Output roots:
- Graph-free W1/W2/W3:           `baselines/outputs/`
- Graph-injected W2/W3 (ours):   `baselines/who_and_when/causal/outputs/`
- TRAIL-prompt (already in repo): `benchmarking/outputs/zero_shot2/`
  (Gemini-Flash uses `benchmarking/outputs/zero_shot/` — original split.)

Score with:
```bash
# from benchmarking/
python eval/calculate_scores.py --results_dir <output_dir>
```

---

## 1. What's already done (reference, do not re-run)

These come from `paper/main_results_table.tex` Tables 1 and 4 and define the
**TRAIL-prompt** baseline against which Who&When is the localization-strategy
ablation. Same six models, same two splits.

| Model                    | Split (TRAIL prompt)        | Conditions covered                |
|--------------------------|-----------------------------|-----------------------------------|
| Gemini-2.5-Flash         | GAIA (orig), SWE (orig)     | Baseline, +CG, +CG+SI, +GI+SI     |
| Mistral-Small-3.1-24B    | GAIA_dedup, SWE_Bench_dedup | Baseline, +CG, +CG+SI, +GI+SI     |
| GPT-oss-120B             | GAIA_dedup, SWE_Bench_dedup | Baseline, +CG, +CG+SI, +GI+SI     |
| GPT-oss-20B              | GAIA_dedup, SWE_Bench_dedup | Baseline, +CG, +CG+SI, +GI+SI     |
| Gemma-3-27B-IT           | GAIA_dedup, SWE_Bench_dedup | Baseline, +CG, +CG+SI, +GI+SI     |
| QwenLong-L1-32B          | GAIA_dedup, SWE_Bench_dedup | Baseline, +CG, +CG+SI, +GI+SI     |

Mistral-Small-3.1-24B already has W1 and W2 graph-free runs on `GAIA`,
`GAIA_dedup`, and `SWE Bench` (see `baselines/outputs/`). All other Who&When
cells below are new.

---

## 2. Coverage matrix — Who&When prompts × graph injection

**Reframing (2026-05-11).** W1/W2 are not competing methods on TRAIL — they
are alternative *prompt formats* over the same dataset; the TRAIL Baseline
row is itself an adapted W&W prompt (single→multi-error). The matrix below
therefore tests our method (+CG, +GI+SI) **on top of** each prompt format,
demonstrating that the gain is from causal-graph injection and not from a
lucky interaction with one specific prompt skeleton.

Same five open-source models × same two splits as Table 1.

| Model                    | W1 | W2 | W1+CG | W2+CG | W1+GI+SI | W2+GI+SI |
|--------------------------|----|----|-------|-------|----------|----------|
| Mistral-Small-3.1-24B    | ✅ | ✅ | T     | T     | T        | T        |
| GPT-oss-120B             | ✅ | ✅ | T     | T     | T        | T        |
| GPT-oss-20B              | ✅ | ✅ | T     | T     | T        | T        |
| Gemma-3-27B-IT           | ✅ | ✅ | T     | T     | T        | T        |
| QwenLong-L1-32B          | ✅ | ✅ | T     | T     | T        | T        |

`✅` already in `baselines/outputs/`. `T` = to-do.

**W3 (binary search) intentionally excluded** from the main matrix
(see §7). The runner still supports `--variant w3` for the appendix
sanity probe.

**Cost-staged priority ordering.** W1+graph is ~2× single-pass cost;
W2+graph is ~(N+1)× ≈ 9–10×. So we stage:

1. **W1+CG** then **W1+GI+SI** on the full panel (cheap — ~2× per run).
   This alone answers "does graph help on the cheap prompt format?" If
   yes for ≥3/5 models on both splits, the headline robustness claim
   is in.
2. **W2+CG** then **W2+GI+SI** on the headline cell first
   (Mistral GAIA_dedup), then on whichever 1–2 cells showed the largest
   W1 gain. Skip the rest unless reviewer pressure demands them.

Within each model, run in order:
`W1+CG → W1+GI+SI → W2+CG → W2+GI+SI`.
Graph-free W1/W2 numbers are already on disk for all five models —
no re-run needed.

---

## 3. Graph configuration

Use the **same graph artifacts** as TRAIL +GI+SI in Table 1, so the
prompt-robustness ablation isolates only the prompt-format axis:
- `--causal_only` (13 CAPRI-AIC validated edges) is the default.
- Graph paths default to
  `benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json`
  and `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json`.

Optionally also run the broader `--corr_threshold 0.20` graph for the
two best-performing model+split cells (decided post-hoc).

---

## 4. Run commands

Replace `${MODEL}` and `${SPLIT}` per cell. Two runners live in
`baselines/who_and_when/causal/`:

- `run_who_and_when_with_graph_vllm.py` — **+CG (one-pass graph in prompt).**
  Inserts the graph guidance block into the W1 single-call prompt, or into
  every per-step W2 prompt.
- `run_who_and_when_graph_inject_vllm.py` — **+GI+SI (two-pass dynamic
  injection).** Pass 1 = vanilla W1/W2 (no graph); propagate detected
  categories through the graph; Pass 2 = single trace-level targeted call
  with filtered subgraph. Cost profile: W1 → 2 calls/trace; W2 → N+1
  calls/trace (one Pass-2 call per trace, not per span).

### Graph-free W1 / W2 (already done; reference only)
```bash
# from baselines/who_and_when/
python run_who_and_when_vllm.py --model ${MODEL} --split ${SPLIT} --variant w1 --max_model_len 131072
python run_who_and_when_vllm.py --model ${MODEL} --split ${SPLIT} --variant w2 --max_model_len 32768
```

### Stage 1 — W1+CG and W1+GI+SI (cheap; full panel)
```bash
# from baselines/who_and_when/causal/

# W1 + CG (one call per trace)
python run_who_and_when_with_graph_vllm.py \
    --model ${MODEL} --split ${SPLIT} --variant w1 --causal_only \
    --max_model_len 131072

# W1 + GI+SI (two calls per trace)
python run_who_and_when_graph_inject_vllm.py \
    --model ${MODEL} --split ${SPLIT} --variant w1 --causal_only --span_index \
    --max_model_len 131072
```

### Stage 2 — W2+CG and W2+GI+SI (expensive; selected cells)
```bash
# from baselines/who_and_when/causal/

# W2 + CG (graph appears in every per-step call)
python run_who_and_when_with_graph_vllm.py \
    --model ${MODEL} --split ${SPLIT} --variant w2 --causal_only \
    --max_model_len 32768

# W2 + GI+SI (W2 sweep then 1 targeted trace-level pass)
python run_who_and_when_graph_inject_vllm.py \
    --model ${MODEL} --split ${SPLIT} --variant w2 --causal_only --span_index \
    --max_model_len 32768
```

### Concrete first-batch commands (Mistral GAIA_dedup, GPUs 1,2,6,7)
```bash
cd baselines/who_and_when/causal

# 1. W1 + CG — cheapest new cell; canary for whether the graph helps on W1 at all.
CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --variant w1 --causal_only \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.8 \
    --max_model_len 131072

# 2. W1 + GI+SI — confirms two-pass dynamic injection works on W1 prompt skeleton.
CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_graph_inject_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --variant w1 --causal_only --span_index \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.8 \
    --max_model_len 131072

# 3. W2 + GI+SI — headline cell from the previous plan; gates Stage 2.
CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_graph_inject_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --variant w2 --causal_only --span_index \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.8 \
    --max_model_len 32768
```

After GAIA_dedup is complete for each, repeat with `--split SWE_Bench_dedup`.

---

## 5. What the resulting table answers

The ablation table will have one row per (model, split) with six method
columns: W1 / W1+CG / W1+GI+SI / W2 / W2+CG / W2+GI+SI, reporting W-F1, Loc,
Joint — same metric block as Table 1.

Three questions it must answer:

1. **Does causal-graph injection help on prompt formats other than the
   TRAIL Baseline?** Compare the W1 → W1+CG → W1+GI+SI deltas
   against the Baseline → +CG → +GI+SI deltas in Table 1. If the +GI+SI
   delta is in the same direction (and ideally similar magnitude) for W1 as
   for the TRAIL prompt, the headline gain is a *method* contribution, not
   a configuration artifact.
2. **Is the gain prompt-format-sensitive?** Compare W1+GI+SI − W1 against
   W2+GI+SI − W2. If they differ substantially in either direction, we
   need a sentence about which prompt skeleton the method exploits best.
3. **(Original Q from prior plan)** Does W2's step-by-step localization
   consume the graph differently than W1's holistic format? Compare
   W2+CG − W2 (graph reasoning forced per span) vs W1+CG − W1 (graph
   reasoning in one shot). Theoretically W2 should benefit more because it
   can act on "given A, look for B downstream" at the span level.

Stage 1 (W1 cells) answers (1). Stage 2 (W2 cells, selectively) answers
(2) and (3).

---

## 6. Reporting

Pull each run's metrics with:
```bash
python benchmarking/eval/calculate_scores.py --results_dir baselines/outputs
python benchmarking/eval/calculate_scores.py --results_dir baselines/who_and_when/causal/outputs
```

Final table format mirrors `paper/main_results_table.tex` Table 1; numbers
go into a new sub-table immediately after Table 1 with the same column
layout (Model | Method | W-F1 | Loc | Joint × 2 splits).

Trace-count column (analogous to Table 4) should also be generated so any
incomplete runs are visible — Mistral +GI+SI on SWE has $N=9$ in Table 4,
so Who&When SWE runs will have similarly small $N$ and that needs to be
disclosed in the caption.

---

## 7. Why W3 (binary search) is excluded

The original Who&When W3 was designed under the paper's single-root-cause
assumption: one responsible agent per trace, located in O(log N) bisection
calls. Adapting W3 to TRAIL's multi-label setting forces two changes that
nullify this efficiency advantage:

1. Bisection must run **independently per error label** (~19× call
   multiplier on a 19-class taxonomy), since a single bisection cannot
   answer "where is each of the 19 labels."
2. Both halves of each interval must be allowed to test positive, since
   multiple errors of the same label may co-occur. Once both halves can be
   positive, the worst-case call count degrades from O(log N) to O(N) per
   label, and the expected count for any label that appears more than once
   collapses to roughly linear in N.

Net effect: W3 systematically costs more than the linear-scan W2 — which
already produces a per-span multi-label decision in N calls — while
offering no asymptotic localization benefit in the multi-error regime that
defines TRAIL.

**Paper framing (drop-in for the methodology section):**

> Who&When's W3 binary search was designed under the original paper's
> single-root-cause assumption: one responsible agent per trace, located
> in O(log N) bisection calls. Adapting W3 to TRAIL's multi-label setting
> requires (i) running an independent bisection per error label (~19×
> call multiplier on a 19-class taxonomy), and (ii) allowing both halves
> of each interval to test positive, since multiple errors of the same
> label may co-occur. Once both halves can be positive, the worst-case
> call count degrades from O(log N) to O(N) per label, and the expected
> count for any label that appears more than once collapses to roughly
> linear in N. The combined effect is that W3 systematically costs more
> than the linear-scan W2 — which already produces a per-span multi-label
> decision in N calls — while offering no asymptotic localization
> benefit in the multi-error regime that defines TRAIL. We therefore
> evaluate only W1 (all-at-once) and W2 (step-by-step) as Who&When
> localization controls, with W2+graph as the corresponding causal-
> injection variant.

**Empirical sanity probe.** Run W3 graph-free on Mistral GAIA_dedup once
(see §4) so the paper can cite a measured call count (e.g., "57 calls /
trace observed on Mistral GAIA_dedup, vs. 9 for W2") rather than relying
on the asymptotic argument alone. The runner still supports
`--variant w3` and `--variant w3_graph` for any appendix follow-up.

---

## 8. Cost analysis — why Gemini is excluded

Per-trace LLM-call multiplier vs. one TRAIL-prompt baseline run (= 1 call /
trace). GAIA traces have ~8 step spans (`N=8`), so ⌈log₂N⌉ = 3.

| Method                              | Calls / trace             | Multiplier vs. TRAIL Baseline |
|-------------------------------------|---------------------------|-------------------------------|
| TRAIL Baseline / +CG / +CG+SI       | 1                         | 1×                            |
| TRAIL +GI+SI (two-pass)             | 1–2                       | 1–2×                          |
| **W1**                              | 1 + 1 scores = 2          | ~2×                           |
| **W1 + CG**                         | 1 + 1 scores = 2          | ~2× (longer prompt)           |
| **W1 + GI+SI**                      | 2 + 1 scores = 3          | ~3×                           |
| **W2 (graph-free)**                 | N + 1 scores ≈ 9          | ~9×                           |
| **W2 + CG**                         | N + 1 ≈ 9 (longer prompt) | ~9× calls, ~1.5–2× tokens     |
| **W2 + GI+SI**                      | N + 1 sweep + 1 P2 ≈ 10   | ~10× calls                    |
| W3 (excluded — see §7)              | ~57 typical, ~200 worst   | ~30–200×                      |
| W3 + graph (excluded — see §7)      | ~17–26                    | ~17–26×                       |

W2's cumulative-prefix prompt grows ~O(N²) tokens across the N step calls,
so total *input tokens* per trace is ~4–5× the single-pass token count,
making W2's true cost ~30–40× a TRAIL Baseline run on input tokens
(though only ~9× in raw call count).

**Concrete projection — Gemini-2.5-Flash on GAIA orig (N=109 traces):**
- TRAIL Baseline:    ~109 calls
- W1:                ~218 calls (~2×)
- W2 (graph-free):   ~1,000 calls + ~5× tokens (~10–20×)
- W2 + graph:        ~1,000 calls (~10–20×)

Full Gemini Who&When row (W1, W2, W2+graph after the W3 drop) on GAIA + SWE
is still roughly **30–40× the cost of one Gemini TRAIL Baseline run**, with
an expected within-noise W1 ≈ W2 result and a graph-injection delta the
open-source rows already establish.

**Decision: Gemini is excluded from the Who&When matrix.** The same ablation
runs for free on the five open-source models via local vLLM. Gemini remains
in Table 1 as the TRAIL-prompt frontier reference, and the Who&When table
will be open-source only.

**Cost-saving choices kept for the open-source rows:**
- W3 / W3+graph dropped from the matrix (see §7); only the one-cell sanity
  probe on Mistral GAIA_dedup runs.
- `--w2_max_spans` lowered to 10 for `SWE_Bench_dedup` (longer traces) to
  cap W2 / W2+graph runtime.
