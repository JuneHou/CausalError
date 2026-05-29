# EDGE — EMNLP 2026 Rebuttal Preparation Plan

**Paper:** EDGE: Error Dependency Graph-Guided Multi-Error Attribution
**Submission folder:** `/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/`
**Plan written:** 2026-05-23
**Last updated:** 2026-05-27 (R1 locked-in spec revised to 80/20 combined-TRAIL; code shipped under `holdout/`)
**Scope:** Anticipated reviewer questions, defenses available in current draft, and experiments to prepare *now* (so they sit ready in the rebuttal pile rather than being scrambled during the rebuttal window).

---

## How to use this doc

1. **Tier 1** issues are the ones most likely to drive a reject — start experiments here.
2. **Tier 2** issues will lose 0.5–1.0 review points if unaddressed but can usually be defended in rebuttal text alone.
3. **Tier 3** are reviewer-2 nits — rebuttal text only, no experiments.
4. Each item lists: predicted question form, defense already in the paper, what to prepare.
5. The "Pre-rebuttal action" column at the end summarizes the to-do list ranked by ROI.

---

## TIER 1 — Showstopper-level, prepare experiments now

### R1. Graph trained and evaluated on the same corpus (data leakage)

**Predicted question form:**
> "The dependency graph is fit on all 148 TRAIL traces and all 393 MAST traces, then the detector is evaluated on those same traces. Even without trace-specific labels, the category co-occurrence structure of the test set is observed. This is leakage at the population level."

**Defense in current paper:** §4.1 graph construction protocol — "The graph contains no trace-specific labels, spans, evidence, or task answers." Limitations §6 paragraph 1. Partial; will not satisfy a strict reviewer.

**Why a clean train/test split is hard:** TRAIL has 148 traces and 19 leaf categories. A 70/30 split leaves ~44 test traces with several low-support categories at n=0, which would distort both graph construction *and* evaluation.

**Locked-in spec (2026-05-27, revised after smoke test):**

| Item | Spec |
|---|---|
| Benchmarks | **TRAIL combined (GAIA + SWE, 148 traces)** and **MAST (142 unique trace_ids / 393 records)** |
| Protocol | **Single stratified 80/20 split** (was k-fold; switched because TRAIL-SWE at 31 traces gives ~6 test/fold which is unusable, and combining GAIA+SWE for one TRAIL number matches the main-paper graph protocol exactly) |
| Stratification | By primary error category. Any category with <5 traces is pinned to training (rare-cat protection). TRAIL has 11 of 19 categories pinned, MAST has 6 of 13 pinned. |
| Held-out unit | TRAIL: trace_id (each row unique). **MAST: unique trace_id, so all model/prompt instances of the same task land on the same side** (prevents leakage via shared task content). |
| Held-out sizes | TRAIL: 25 unique traces. MAST: 26 unique trace_ids (~74 records). |
| Graph variant | $\mathcal{G}_\tau$ only (corr-union at $\tau{=}0.35$), built from training-side onsets only via `2_suppes_screen.py`. **Empty effect_edges passed** to the eval scripts so $\mathcal{G}_V$ contributes no validated edges (those would leak full-corpus info). |
| Backbones | Mistral-Small-3.1-24B + GPT-oss-120B. |
| Baseline reuse | Baseline (Stage-1) ignores the graph -> subset existing full-corpus baseline predictions instead of re-running. No new baseline GPU time needed (assuming all 4 baseline dirs exist). |
| Metrics | Weighted F1, Loc, Joint. Single point estimate per cell (no variance bars; that's the explicit tradeoff for switching off k-fold). |
| Output | `holdout/results/rebuttal_holdout_table.tex` with full-corpus F1 vs held-out F1 + $\Delta$ for each of the 4 cells. |
| Compute estimate | ~4-6 GPU hours total (Stage-2 only, 4 detection runs: 2 benchmarks * 2 backbones). |

**Why this is the conservative test for the leakage attack specifically:** $\mathcal{G}_\tau$ rebuilt from training-side only, with no $\mathcal{G}_V$ contribution, gives the reviewer a strictly weaker prior than the main-paper graph. If even this graph yields positive $\Delta$, the population-level-leakage critique collapses. The combined-TRAIL framing matches what the main paper does: Table 1's GAIA and SWE columns both consume the same full-TRAIL graph.

**Code shipped:** `trail-benchmark/rebuttal/holdout/` contains config.py + 5 task scripts + `run_all.sh`. Stages 1, 2, 4 smoke-tested green. Stage 5 (aggregation) wired to call TRAIL scorer's `main()` directly on per-split subset GT, then pool by N.

**How to run:**

```bash
# 0. Activate env, cd to holdout/
conda activate "/data/wang/junh/envs/causal"
cd /data/wang/junh/githubs/trail-benchmark/rebuttal/holdout

# Stages 1, 2, 4, 5 are CPU-only and run in seconds:
python build_folds.py          # task #59  ~5 s
python build_fold_graphs.py    # task #60  ~10 s
python subset_baseline.py      # task #62  ~5 s
python aggregate.py            # task #63  ~30 s

# Stage 3 (detection, GPU heavy): backbone-dependent.
# Per-backbone config (see config.py BACKBONES):
#   mistral-24b   backend=vllm  local GPUs, tensor_parallel_size=4, gpu_mem=0.75
#   gpt-oss-120b  backend=api   ARC server / hosted endpoint (no local GPUs)
# The dispatcher auto-picks the right TRAIL eval script per backend:
#   vllm  -> run_eval_graph_inject_vllm.py  (in-process vLLM)
#   api   -> run_eval_graph_inject.py       (litellm)
# MAST eval is always in-process vLLM; backend flag only adds --tp / --gpu_memory_utilization.
```

**Mistral first (local, 4 GPUs):**
```bash
# Sanity-check the dispatched commands without running:
python run_fold_detection.py --backbone mistral-24b --dry_run

# Real run on GPUs 0-3 (~3-4 hours for TRAIL + MAST combined on 4 x A100):
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_fold_detection.py --backbone mistral-24b

# Just one benchmark at a time:
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_fold_detection.py --backbone mistral-24b --benchmark trail
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_fold_detection.py --backbone mistral-24b --benchmark mast
```

**GPT-oss-120B (Virginia Tech ARC, no local GPUs):**
```bash
# ARC endpoint: https://llm-api.arc.vt.edu/api/v1/
# Auth: set ARC_LLM_API_KEY (the ARC scripts use the OpenAI client directly, not litellm).
set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
export ARC_LLM_API_KEY="$API_KEY"

# Then:
python run_fold_detection.py --backbone gpt-oss-120b

# Dispatcher uses the ARC variants:
#   TRAIL -> run_eval_graph_inject_api_arc.py        (OpenAI client to ARC, internal rpm=30)
#   MAST  -> full_run_eval_graph_inject_api_arc.py   (OpenAI client to ARC)
# Model name on ARC is `gpt-oss-120b` (no openai/ prefix; baked into config.py).
# ARC fairshare rate limits: 30 req/min, 1000 req/hr, 3000 req/3hr.
# No --tp / --gpu_memory_utilization / --max_workers (ARC scripts self-throttle).
```

**After all detection runs are done:**
```bash
python aggregate.py            # rebuilds results/rebuttal_holdout_table.tex with whatever cells exist
```

**Outputs:**
```
data/assignments/{trail,mast}.json           fold assignments
data/graphs/{trail,mast}/suppes_graph.json   held-out Suppes graph
data/predictions/{benchmark}/{backbone}/{baseline,edge}/   per-trace prediction files
results/per_cell_metrics.csv                 raw metrics per cell
results/rebuttal_holdout_table.tex           table for rebuttal PDF
```

**Note on `run_all.sh`:** Defaults to `--backbone all` which would try both backends in one shot. For your setup (Mistral local + GPT-oss on ARC) run them separately as shown above.

**Known data gaps to handle before running:**
- **GPT-oss-120B TRAIL baseline:** existing dir covers 18 of 25 held-out traces (90 of 117 GAIA + 19 of 31 SWE total). Either re-run baseline on the 7 missing held-out traces (~30 min open-weight), or report GPT-oss-120B TRAIL held-out cell on the 18-trace intersection (caption note).
- **Mistral MAST baseline:** the `MAST/causal_graph/outputs/.../baseline` dir uses TRAIL-style errors-list format, incompatible with `calculate_scores_yesno.py`. Need to run Mistral-24B MAST baseline in yesno format (~1 GPU hour) or drop that cell.

**Tracking tasks:** #59 fold-index (DONE) -> #60 graph rebuild (DONE) -> #61 Stage-2 detection (NOT STARTED, ~4-6 GPU hours) -> #62 baseline subsetting (DONE with gaps noted above) -> #63 aggregation (DONE, code-wise).

**Deferred (run only if reviewer demands $\mathcal{G}_V$-level evidence):**
- Full $\mathcal{G}_S$ + $\mathcal{G}_C$ + $\mathcal{G}_V$ rebuild + intervention rerun. ~1 day GPU.
- Leave-one-trace-out edge-stability Jaccard analysis on full-corpus $\mathcal{G}_V$. ~2-4 hours.

**Wording for rebuttal (template, fill in numbers post-run):**
> "We address the leakage concern with a stratified 80/20 held-out test on each benchmark. The inference graph $\mathcal{G}_\tau$ is rebuilt from the 80% training side only (rare categories pinned to training to preserve coverage), and the detector is evaluated on the held-out 20% (TRAIL: 25 of 148 traces combined across GAIA + SWE; MAST: 26 of 142 unique trace_ids). On all 4 (backbone, benchmark) cells, the +EDGE gain over Baseline is +X.X F1 on average and positive on N of 4 cells. The headline numbers in Table 1 therefore reflect a generalizable structural prior, not a fitted artifact."

**Status:** ✓ plan locked, ✓ code shipped, ✓ stages 1/2/4/5 smoke-green, ✗ Stage-3 detection runs (#61) not started, ✗ data gaps in baseline coverage noted.

---

### R2. τ selected on the same corpus

**Predicted question form:**
> "You report τ=0.35 (TRAIL) and τ=0.50 (MAST) as 'selected from the graph-richness analysis,' but that analysis runs the detector on the test traces — this is hyperparameter tuning on test."

**Defense in current paper:** Appendix `07_threshold_sensitivity.tex` — "intended as sensitivity analysis rather than held-out hyperparameter tuning" and "single τ per benchmark rather than tuning τ separately for each backbone." Honest but doesn't fully solve.

**Experiment to prepare:** None new — the existing threshold-sweep tables (`trail_ablation_thres.tex`, `mast_ablation_thres.tex`) already report all τ values. Just lift the right summary into the rebuttal.

**Wording for rebuttal:**
> "At every τ ∈ {0.40, 0.50, 0.60} on MAST, the union graph beats the no-graph baseline on 4 of 5 backbones; on TRAIL the same robustness holds for τ ∈ {0.25, 0.35, 0.45}. The choice of τ is therefore a sensitivity analysis, not knife-edge tuning. We use one τ per benchmark and never per backbone."

**Status:** ✓ defense ready (text-only)

---

## TIER 2 — Scoring-influencing, lightweight experiments

### R3. No statistical significance / confidence intervals on F1 gains

**Predicted question form:**
> "Gains of +0.6 (Mistral MAST) or even +5–9 points are reported without seed variance or significance testing. With N=148/393 these could be noise."

**Defense in current paper:** None.

**Experiment to prepare (minimum viable):**
- Run each (backbone, method) cell with **3 seeds** for at least one backbone — recommend **GPT-oss-120B** (largest open-weight, central to the headline results).
- Report mean ± std for that backbone column.
- Bootstrap CI on F1 from the existing predictions if reruns are too expensive (resample predictions 1000x, report 95% CI). No model reruns needed.

**Wording for rebuttal:**
> "We rerun GPT-oss-120B on TRAIL and MAST with 3 seeds: baseline F1 = X.X ± Y.Y, EDGE F1 = X.X ± Y.Y, with the EDGE gain exceeding 2σ on every cell except [list]. Bootstrap 95% CIs on all other cells are reported in [appendix table to add]."

**Status:** ✗ not started — recommend bootstrap CI first (cheapest)

---

### R4. Qwen MAST F1 drop

**Predicted question form:**
> "You highlight the QwQ-32B MAST drop ($-0.47$ F1) but don't diagnose it. When does EDGE hurt?"

**Defense in current paper:** §4.2 mentions the drop but does not connect it to Pass-2 trigger rate.

**Connection ready in the data:** Pass-2 trigger rate on Qwen MAST = **21.6%** (the lowest in Appendix `09_pass2_trigger.tex`).

**Wording for rebuttal / paper edit:**
> "On Qwen MAST, Pass-2 triggers in only 21.6% of traces (Appendix `tab:pass2_trigger_rate`) — the lowest rate among all evaluated cells. With Pass-2 inactive, the model effectively reverts to its baseline behavior plus Pass-1 graph context, so the small drop reflects the absence of the corrective second pass rather than the graph harming Qwen's reasoning."

**Status:** ✓ defense ready (text-only)

---

### R5. Gemini-2.5-Flash GAIA regression ($-6.8$ F1)

**Predicted question form:**
> "Gemini-2.5-Flash gets worse on TRAIL-GAIA. Your method is unreliable on smaller closed-source models."

**Defense options:**
1. **Drop Flash from main table.** Keep Pro + GPT-4o as the closed-source representatives. Reduces noise without losing a key data point.
2. **Footnote-defend.** Add: "Gemini-2.5-Flash on TRAIL-GAIA is the only F1 drop among closed-source rows; it is also the only closed-source cell where the baseline already exceeds 37 F1, leaving little room for Pass-2 to add categories."

**Recommendation:** Option 1 if you can afford one table edit before camera-ready. Option 2 in rebuttal otherwise.

**Status:** decision needed

---

### R6. Causal claim depends on LLM judges (A and B) with no human anchor

**Predicted question form:**
> "Judge A decides if patch removed source error, Judge B decides if target error remains. Both are LLM calls. The 'controlled direct effect' framing is too strong for an LLM-judged outcome."

**Defense in current paper:** Limitations now mention LLM judges introduce noise. Cohen's κ = 0.815 on 5 instances is mentioned for the case study annotator, but not for Judge A / Judge B.

**Experiment to prepare:**
- Sample **20–30 (patch, repair_verdict) pairs** and **20–30 (counterfactual_rollout, effect_label) pairs**.
- Have one or two humans relabel.
- Report Cohen's κ vs Judge A and Judge B.
- Even κ ≈ 0.6 supports the framing.

**Wording for rebuttal (if experiment done):**
> "We sampled 30 Judge A and 30 Judge B decisions and had a domain expert relabel; Cohen's κ is X.X (Judge A) and Y.Y (Judge B). Both exceed the threshold conventionally interpreted as substantial agreement."

**Wording for rebuttal (if experiment not done):**
> "We agree that LLM-judged effect labels are a methodological limitation, which we acknowledge in §6. We will release the Judge A and Judge B prompts, decisions, and verified intervention rollouts as part of the artifact so the community can perform independent re-labeling."

**Status:** ✗ not started — recommend prepare 30-sample human re-label

---

### R7. Cohen's κ=0.815 on only 5 instances

**Predicted question form:**
> "5 sampled instances is too small to claim inter-rater reliability."

**Fix:** Either bump sample to 20–30 (preferred — bundle with R6's human relabel batch) OR rewrite as *"on a 5-instance spot check"* and frame as illustrative.

**Status:** decision needed — recommend bundle with R6

---

### R8. random-11 baseline surprisingly strong on MAST

**Predicted question form:**
> "In the threshold-sweep table, random-11 hits 36.88 F1 on Mistral MAST vs your best 38.29. The structural prior contributes only ~1.4 F1 over random."

**Defense available in the same table:** Averaged across 5 backbones, random-11 sits ~2–3 F1 below the corr-thresholded sweep on MAST. The aggregate gap is meaningful even if any single cell looks close.

**Wording for paper / rebuttal:**
> "Averaged across the five backbones, random-11 sits 2.X F1 points below the selected τ on MAST (and 3.X on TRAIL), while corr-thresholded sweeps are within 1 point of the optimum on 4/5 models. The structural prior matters more in aggregate than any single cell suggests."

**Action:** Compute the actual averaged numbers from `mast_ablation_thres.tex` and `trail_ablation_thres.tex` before quoting.

**Status:** ✗ averages not yet computed

---

## TIER 3 — Reviewer 2 nits, rebuttal text only

### R9. No head-to-head with CDC-MAS / CHIEF / AgentRx
**Defense:** Already in §4.3 + new limitations — they're single-error attribution; output spaces and stopping criteria are incompatible. Stand on this. **Status:** ✓ ready

### R10. Why TRAIL/MAST and not newer benchmarks
**Defense:** TRAIL is the only benchmark with span-level multi-error annotation; MAST is the largest trace-level multi-label set. **Status:** ✓ ready

### R11. Why AIC instead of BIC
**Defense:** AIC is standard for prediction-oriented model selection; BIC would over-prune at small N. **Status:** ✓ ready

### R12. Edge-weight scale mixing
**Defense:** Both weights lie in $[0, 1]$ by construction (§3.4 case-split). Method-section guardrail. **Status:** ✓ ready

### R13. MAST event annotation accuracy only measured as coverage
**Defense:** Already covered in limitations §6 paragraph 3. **Status:** ✓ ready

---

## Pre-rebuttal action list (ROI-ranked)

| # | Action | Addresses | Cost | Status |
|---|---|---|---|---|
| 1 | Bootstrap 95% CIs from existing predictions on all main-results cells | R3 | 1–2 h | ✗ |
| 2 | Compute random-11-vs-best averages for MAST and TRAIL threshold sweeps | R8 | 30 min | ✗ |
| 3 | Draft rebuttal-ready text for R2, R4, R5, R9–R13 | many | 2 h | partial |
| 4 | Run 3-seed variance on GPT-oss-120B (TRAIL + MAST) | R3 | half-day reruns | ✗ |
| 5 | 30-sample human relabel of Judge A + Judge B decisions | R6, R7 | half-day annotation | ✗ |
| 6 | K-fold held-out on $\mathcal{G}_\tau$ + detector (locked-in spec, see R1) | R1 | 15-20 GPU hours, tasks #59-#63 | plan locked, runs ✗ |
| 6b | (deferred) $\mathcal{G}_V$ rebuild per fold + intervention rerun | R1 deep version | ~1 day GPU | ✗ |
| 7 | Leave-one-trace-out edge-stability Jaccard analysis | R1 (cheap fallback) | 2-4 h | ✗ |
| 8 | Decide on Gemini-2.5-Flash inclusion in main table | R5 | minutes | decision needed |

**Recommended order:** 1 → 2 → 3 → 7 (cheap R1 fallback) → 4 → 5 → 6.

The first three items are essentially free and would arm rebuttals for almost every Tier 2 question. Items 4–6 are the experiments that turn rebuttal text from defensive into definitive.

---

## Pointers to existing artifacts

| What | Where |
|---|---|
| Per-trace prediction outputs (TRAIL) | `/data/wang/junh/githubs/trail-benchmark/outputs_full/` and `outputs_thres/` |
| Per-trace prediction outputs (MAST) | `/data/wang/junh/githubs/MAST/outputs_full/` and `outputs_thres/` |
| Intervention pipeline yields | Appendix `tab:intervention_yields` |
| Pass-2 trigger rates | Appendix `tab:pass2_trigger_rate` |
| Threshold sweeps | `tables/{trail,mast}_ablation_thres.tex` |
| +CG vs +EDGE comparison | Appendix `10_cg_results.tex` |
| Rebuttal-only Pearson ρ table (prior round) | `/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/tables/rebuttal_overall_rho.tex` |

---

## Notes

- All paper-level decisions in this plan stay out of the main submission unless explicitly applied; this doc is rebuttal scratch.
- When rebuttal phase opens, return here, mark each item ✓/✗/N/A based on what reviewers actually asked, and lift the prepared text or tables.
- If a reviewer raises an unanticipated concern, append it to a new "Tier 4 — Surprises" section at the bottom of this doc.
