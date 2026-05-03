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

## 2. Coverage matrix — Who&When localization strategies

Same six models × same two splits as Table 1. Each row in this matrix
becomes a row in the Who&When ablation table in the paper.

Splits used **must match** the TRAIL-prompt rows so numbers are comparable:
all open-source models use `GAIA_dedup` and `SWE_Bench_dedup`.

| Model                    | W1 | W2 | W2+graph |
|--------------------------|----|----|----------|
| Mistral-Small-3.1-24B    | ✅ | ✅ | T        |
| GPT-oss-120B             | T  | T  | T        |
| GPT-oss-20B              | T  | T  | T        |
| Gemma-3-27B-IT           | T  | T  | T        |
| QwenLong-L1-32B          | T  | T  | T        |

`✅` already in `baselines/outputs/`. `T` = to-do.

**Gemini-2.5-Flash is intentionally excluded** from Who&When (see §8
on cost). Open-source-only matrix; Gemini stays in Table 1 as the
TRAIL-prompt frontier reference.

**W3 (binary search) is intentionally excluded** from the main matrix (see
§7 on the methodology argument). Recommend running W3 graph-free on
**Mistral GAIA_dedup only** as a one-cell sanity probe to cite empirical
call counts in the paper footnote. The runner still supports
`--variant w3` and `--variant w3_graph` for that sanity cell and any
appendix follow-up.

**Priority ordering.** Run in this order so the headline numbers land first:

1. **Mistral-Small-3.1-24B** (already has W1/W2): W2+graph on `GAIA_dedup`,
   then `SWE_Bench_dedup`. Mistral is the established open-source headline
   model in Table 1. Optionally also run a one-shot W3 graph-free on
   `GAIA_dedup` to record the empirical call-count blowup for the
   methodology footnote.
2. **GPT-oss-120B** and **Gemma-3-27B-IT** (full rows, dedup splits). Both
   show large +GI+SI gains in Table 1, so the W2+graph ablation is most
   informative here.
3. **GPT-oss-20B** and **QwenLong-L1-32B** (full rows, dedup splits). Smaller
   open-source models — fill in for completeness.

Within each model, run in order:
W1 → W2 → W3 → W2+graph → W3+graph
(graph-free first; graph variants depend on no-graph numbers existing in
the same table).

---

## 3. Graph configuration

Use the **same graph artifacts** as TRAIL +GI+SI in Table 1, so the
localization-strategy ablation isolates exactly the bisection vs. holistic
question with the same causal evidence:
- `--causal_only` (13 CAPRI-AIC validated edges)
- Graph paths default to
  `benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json`
  and `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json`.

Optionally also run the broader `--corr_threshold 0.20` graph for the
two best-performing model+split cells (decided post-hoc, after the
`--causal_only` numbers are in).

---

## 4. Run commands

Replace `${MODEL}` and `${SPLIT}` per cell. GPU count and
`tensor_parallel_size` follow the same conventions as the TRAIL +GI+SI runs
for that model.

### Graph-free W1 / W2
```bash
# from baselines/who_and_when/
python run_who_and_when_vllm.py \
    --model ${MODEL} \
    --split ${SPLIT} \
    --variant w1 \
    --max_model_len 131072

python run_who_and_when_vllm.py \
    --model ${MODEL} \
    --split ${SPLIT} \
    --variant w2 \
    --max_model_len 32768
```

### Graph-injected W2 (this PR)
```bash
# from baselines/who_and_when/causal/
python run_who_and_when_causal_vllm.py \
    --model ${MODEL} \
    --split ${SPLIT} \
    --variant w2_graph \
    --causal_only \
    --max_model_len 32768
```

### Concrete first-batch commands (Mistral GAIA_dedup, GPUs 1,2,6,7)
```bash
cd baselines/who_and_when/causal

# 1. W2 + graph — the headline new cell for Mistral
CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_causal_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --variant w2_graph \
    --causal_only \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \
    --max_model_len 32768
```

After GAIA_dedup is complete, repeat with `--split SWE_Bench_dedup`.

Optional methodology-footnote sanity probe (Mistral GAIA_dedup only):
```bash
cd baselines/who_and_when

# W3 graph-free — record the empirical call-count blowup for §7
CUDA_VISIBLE_DEVICES=1,2,6,7 python run_who_and_when_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --variant w3 \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.34 \
    --max_model_len 32768
```

---

## 5. What the resulting table answers

The Who&When ablation table in the paper will have one row per model and
three method columns (W1, W2, W2+graph), reporting W-F1, Loc, Joint per
split — same metric block as Table 1.

Two questions it must answer:

1. **Does localization strategy matter without the graph?** Compare
   W1 vs. W2 columns. If they are within noise (which `plan.md` predicts),
   it confirms the TRAIL-prompt baseline already captures whatever
   localization strategy contributes on its own.
2. **Does the graph help more in localization-aware variants than in W1?**
   Compare the W2 → W2+graph delta against the Baseline → +GI+SI delta
   from Table 1. The Suppes graph is precedence-filtered, so theoretically
   W2+graph should benefit more, since W2 advances span-by-span and can
   actually consume the "given A, look for B downstream" signal.

Both questions are downstream of getting the runs done — start with #1
(Mistral GAIA_dedup W2+graph) before launching the rest.

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
| **W2 (graph-free)**                 | N + 1 scores ≈ 9          | ~9×                           |
| **W2 + graph**                      | N + 1 ≈ 9 (longer prompt) | ~9× calls, ~1.5–2× tokens     |
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
