# EMNLP Main Table — Experiment TODO

All commands run from `benchmarking/` unless noted.
Target directory for all new runs: `outputs/zero_shot2/` (or `outputs_thres/`
for the threshold sweep).

---

## Currently In Progress

Open-source panel (no Gemini Flash/Pro): **gpt-oss-20b, gpt-oss-120b, Mistral-24B, Qwen-32B, Gemma-3-27B**.
Both splits: GAIA_dedup, SWE_Bench_dedup.

### Task A — Who&When W1+graph\_inject (+GI+SI) / W2+with\_graph (+CG+SI)

- **W1** uses two-pass graph injection (`graph_inject`): `run_who_and_when_graph_inject_{vllm|api_deepinfra}.py`
- **W2** uses one-pass in-prompt graph guidance (`with_graph`): `run_who_and_when_with_graph_{vllm|api_deepinfra}.py`
- Both variants add `--span_index` (`_span_index` suffix in output dir name)

Output dir: `baselines/who_and_when/causal/outputs/`
Naming: W1 → `outputs_{model}-{split}-who_and_when_w1_graph_inject_causal_only_span_index/`
        W2 → `outputs_{model}-{split}-who_and_when_w2_graph_causal_only_span_index/`

| Model | Split | W1 (+GI+SI) | W2 (+CG+SI) |
|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | ✓ | ✓ |
| gpt-oss-120b | SWE_Bench_dedup | ✓ | ✓ |
| gpt-oss-20b | GAIA_dedup | ✓ | ✓ |
| gpt-oss-20b | SWE_Bench_dedup | ✓ | ✓ |
| Mistral-Small-24B | GAIA_dedup | ✓ | ✓ |
| Mistral-Small-24B | SWE_Bench_dedup | ✓ | ✓ |
| Qwen-32B | GAIA_dedup | ✓ | ✓ |
| Qwen-32B | SWE_Bench_dedup | ✓ | ✓ |
| Gemma-3-27B | GAIA_dedup | ✓ arc | ✓ arc |
| Gemma-3-27B | SWE_Bench_dedup | ✓ arc | ✓ arc |

gpt-oss models use DeepInfra API (Qwen not supported there; use vLLM for Qwen).
Score: `python benchmarking/eval/calculate_scores.py` pointed at the output dir.

#### Commands (run from repo root; source API keys first)

```bash
# ARC:       source path/to/arc_llm_api.sh   → sets ARC_LLM_API_KEY
# DeepInfra: export DEEPINFRA_API_KEY=<key>

# === ARC API — gpt-oss-120b (default model) ===

# [TA-ARC-W1] W1+GI+SI
python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_arc.py \
    --variant w1 --split GAIA_dedup --causal_only --span_index
python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_arc.py \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index

# [TA-ARC-W2] W2+CG+SI
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_arc.py \
    --variant w2 --split GAIA_dedup --causal_only --span_index
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_arc.py \
    --variant w2 --split SWE_Bench_dedup --causal_only --span_index

# === DeepInfra API — Mistral-Small-24B ===

# [TA-DI-W1] W1+GI+SI (GAIA already done via vLLM; only SWE missing)
python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_deepinfra.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index

# [TA-DI-W2] W2+CG+SI (both splits missing)
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --variant w2 --split GAIA_dedup --causal_only --span_index
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --variant w2 --split SWE_Bench_dedup --causal_only --span_index

# === DeepInfra API — Gemma-3-27B-IT (W1 and W2, both splits missing) ===

# [TA-DI-Gemma-W1] W1+GI+SI
python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_deepinfra.py \
    --model google/gemma-3-27b-it \
    --variant w1 --split GAIA_dedup --causal_only --span_index
python baselines/who_and_when/causal/run_who_and_when_graph_inject_api_deepinfra.py \
    --model google/gemma-3-27b-it \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index

# [TA-DI-Gemma-W2] W2+CG+SI
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model google/gemma-3-27b-it \
    --variant w2 --split GAIA_dedup --causal_only --span_index
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model google/gemma-3-27b-it \
    --variant w2 --split SWE_Bench_dedup --causal_only --span_index
```

### Task B — Threshold Sweep (`+GI+SI`, τ ∈ {0.35, 0.25, 0.20, random-12})

Output dir: `benchmarking/outputs_thres/t<τ>/`
Script: `benchmarking/eval/run_threshold_sweep.sh`

| Model | Split | τ=0.35 | τ=0.25 | τ=0.20 | random-12 |
|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-120b | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-20b | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-20b | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| Mistral-Small-24B | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| Mistral-Small-24B | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| Qwen-32B | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| Qwen-32B | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| Gemma-3-27B | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| Gemma-3-27B | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |

All sweep cells complete; see `paper/ablation_graph_richness.tex`
for the assembled table.

#### [TB-Rerun-gpt20b-SWE] RESOLVED 2026-05-14 — reasoning_effort='low' fix

**Issue (now fixed).** All four `+GI+SI` sweep variants for gpt-oss-20b
on SWE_Bench_dedup originally landed with $N\in\{9,10,13,12\}$ vs.\
causal-only's $N{=}19$. Two failure modes were identified from
`outputs_thres/_sweep_logs/openai-gpt-oss-20b-SWE_Bench_dedup-*.log`:

1. **Pass-1 JSON parse failures (~5/run, the fixable mode).**
   DeepInfra defaults gpt-oss to `reasoning_effort='high'`, which
   burns most of the 24k `max_tokens` budget on thinking and
   truncates the JSON tail. Fixed in
   `benchmarking/eval/run_eval_graph_inject_api_deepinfra.py` —
   `--reasoning_effort auto` (default) now sets `'low'` automatically
   when the model id matches `gpt-oss-20b`. After fix: 0–2 Pass-1
   parse failures per variant.
2. **Context-length errors (~12/run, unfixable client-side).**
   Some SWE_Bench traces produce prompts of 130k–570k tokens,
   exceeding DeepInfra's 131,071-token cap. These 12 traces still
   400 after the rerun; they are excluded from N regardless of
   reasoning_effort.

**Post-rerun result (2026-05-14)** — coverage and W-F1, before → after:

| Variant   | N (before → after) | W-F1 (before → after) |
|---|---|---|
| random-12 | 9  → **18** | 12.53 → **23.50** |
| τ=0.35    | 10 → **19** | 18.23 → **29.89** |
| τ=0.25    | 13 → **17** | 19.56 → **27.71** |
| τ=0.20    | 12 → **18** | 19.12 → **21.13** |

`paper/ablation_graph_richness.tex` has been updated to drop the four
$\dagger$ flags on the GPT-oss-20B SWE block, replace the W-F1/Loc/Joint
values, and re-bold per (model, split, metric).

**Procedure that was used** (kept for future reference / similar
reruns; the original Step 1 used a literal-glob form that silently
no-op'd, so it's been replaced with a proper `for d in glob` pattern
that survives the no-match case):

```bash
# Run from benchmarking/; needs DEEPINFRA_API_KEY.

# 1. Park the broken sweep dirs aside. The nested-for-loop pattern
#    expands the glob safely (a literal "*" in `mv $src ...` silently
#    fails when the glob doesn't match — the old form did that).
for sub in t_random12_seed42 t0.35 t0.25 t0.20 ; do
  for d in outputs_thres/${sub}/outputs_openai-gpt-oss-20b-SWE_Bench_dedup-graph_inject_*span_index ; do
    [ -d "$d" ] || continue
    mv "$d" "${d}_BROKEN_pre_reasoning_low_$(date +%Y%m%d_%H%M)"
    echo "moved: $d"
  done
done

# 2. Verify Step 1 worked — should list 4 _BROKEN_ dirs and no live
#    sweep dirs. If you still see live dirs, Step 1 didn't actually
#    move them and Step 3 will skip everything ("Pending: 0").
ls -d outputs_thres/*/outputs_openai-gpt-oss-20b-SWE_Bench_dedup-graph_inject_* 2>/dev/null

# 3. Re-run the 4 flagged thresholds (causal_only NOT in the default
#    THRESHOLDS, so it won't be touched).
bash eval/run_threshold_sweep.sh openai/gpt-oss-20b SWE_Bench_dedup
# First lines of the log should show:
#   [INFO] gpt-oss-20b detected; setting reasoning_effort='low' ...
#   Found 31 traces. Pending: 31 (skipping 0 already done)
# If "Pending: 0" appears, Step 1 was missed — stop and redo it.

# 4. Confirm coverage recovered.
for t in t_random12_seed42 t0.35 t0.25 t0.20 ; do
  f=outputs_thres/${t}/outputs_openai-gpt-oss-20b-SWE_Bench_dedup-graph_inject_*span_index-metrics.txt
  echo "=== $t ==="; grep "overall" $f | awk '{print "  N="$4}'
done
```

#### [TB-Rerun-mistral-SWE] Open — clean rerun via DeepInfra

**Issue.** Mistral-Small-3.1-24B on SWE_Bench_dedup is flagged at
causal-only ($N{=}9$) and $\tau{=}0.25$ ($N{=}10$) in
`paper/ablation_graph_richness.tex`. Both are <75% of the same model's
achievable max ($\tau{=}0.35$ has $N{=}14$). Coverage spread across
the five variants is `random=12 / causal=9 / t0.35=14 / t0.25=10 /
t0.20=12`. Because the variation is *per variant within the same
model+split*, the missing traces are **Pass-1 JSON parse failures**,
not the 131k context cap (cap-bound traces would fail uniformly
across all five variants). The structural ceiling is $N{=}14$.

**Why no code fix.** Mistral-Small-3.1-24B is not a reasoning model,
so there is no `reasoning_effort` knob to lower (the gpt-oss-20b fix
does not apply). A clean rerun with the existing DeepInfra runner
should lift parse-failure traces back into N. Target post-rerun:
$N\to 14$ on both variants.

**Backend.** `run_threshold_sweep.sh` defaults `mistralai/*` to vLLM
— for this rerun we override to DeepInfra (5th positional arg) to
match the gpt-oss-20b workflow. Greedy decoding (`temperature=0`)
keeps backend-cross-checks tight in expectation.

**Procedure** (run from `benchmarking/`; needs `DEEPINFRA_API_KEY`):

```bash
# 1. Park the two broken output dirs aside. Same nested-for-loop
#    pattern as the gpt-oss-20b rerun: a literal "*" in `mv $src ...`
#    silently fails when the glob doesn't expand — the for-loop
#    form survives the no-match case.
for d in outputs/zero_shot2/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_only_span_index ; do
  [ -d "$d" ] || continue
  mv "$d" "${d}_BROKEN_pre_rerun_$(date +%Y%m%d_%H%M)"
  echo "moved: $d"
done
for d in outputs_thres/t0.25/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_corr0.25_span_index ; do
  [ -d "$d" ] || continue
  mv "$d" "${d}_BROKEN_pre_rerun_$(date +%Y%m%d_%H%M)"
  echo "moved: $d"
done

# 2. Verify Step 1 actually moved both dirs — should print 2 *_BROKEN_*
#    paths and zero live dirs. If you still see live dirs, Step 1 didn't
#    fire and Step 3 will skip everything ("Pending: 0").
ls -d outputs/zero_shot2/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_only_span_index* 2>/dev/null
ls -d outputs_thres/t0.25/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_corr0.25_span_index* 2>/dev/null

# 3a. Re-run causal-only directly so the new output lands back at
#     outputs/zero_shot2/ (the path the table is already sourcing from).
python eval/run_eval_graph_inject_api_deepinfra.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# 3b. Re-run t=0.25 via the sweep with the DeepInfra backend override.
#     THRESHOLDS env restricts the loop to a single threshold; 5th
#     positional arg "deepinfra" forces DeepInfra (default for
#     mistralai/* would be vLLM).
THRESHOLDS="0.25" bash eval/run_threshold_sweep.sh \
    mistralai/Mistral-Small-3.1-24B-Instruct-2503 SWE_Bench_dedup 0,1 outputs_thres deepinfra

# 4. Score the rerun outputs.
python eval/calculate_scores.py --results_dir outputs/zero_shot2
# (sweep already scores t0.25 at the end of Step 3b)

# 5. Confirm coverage recovered.
echo "=== causal-only ==="
grep "overall" outputs/zero_shot2/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_only_span_index-metrics.txt | awk '{print "  N="$4}'
echo "=== t=0.25 ==="
grep "overall" outputs_thres/t0.25/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-SWE_Bench_dedup-graph_inject_causal_corr0.25_span_index-metrics.txt | awk '{print "  N="$4}'
```

**Post-rerun follow-up** (after Step 5 confirms $N$ recovered):
update `paper/ablation_graph_richness.tex` — replace the six
$\dagger$ cells on the Mistral SWE causal-only and $\tau{=}0.25$
rows with the new W-F1 / Loc / Joint values, re-bold per (model,
split, metric), and trim the abnormal-coverage paragraph in the
caption + the take-away mentioning "Mistral SWE causal-only ($N{=}9$)
/ $\tau{=}0.25$ ($N{=}10$)".

#### Commands (run from benchmarking/; source API keys same as Task A)

All cells go through `eval/run_threshold_sweep.sh`, which writes per-threshold
subdirs to `outputs_thres/t<τ>/` and runs scoring at the end. Backend is
inferred from the model ID (see driver header); override with the 5th
positional arg if needed.

```bash
# === ARC API — gpt-oss-120b ===     (backend inferred: bare name → arc)
# Auth: source path/to/arc_llm_api.sh   → sets ARC_LLM_API_KEY

# [TB-ARC-G] GAIA_dedup — all 4 thresholds in one call
bash eval/run_threshold_sweep.sh gpt-oss-120b GAIA_dedup
# [TB-ARC-S] SWE_Bench_dedup
bash eval/run_threshold_sweep.sh gpt-oss-120b SWE_Bench_dedup

# === DeepInfra API — gpt-oss-20b ===  (inferred: openai/gpt-oss-* → deepinfra)
# Auth: export DEEPINFRA_API_KEY=<key>

# [TB-DI-G] GAIA_dedup
bash eval/run_threshold_sweep.sh openai/gpt-oss-20b GAIA_dedup
# [TB-DI-S] SWE_Bench_dedup
bash eval/run_threshold_sweep.sh openai/gpt-oss-20b SWE_Bench_dedup

# === Gemma-3-27B-IT === (pick ONE backend; use ID matching the backend)

# Option A: vLLM  (inferred: openai/gemma-* → vllm)
# [TB-vLLM-Gemma-G] GAIA_dedup
bash eval/run_threshold_sweep.sh openai/gemma-3-27b-it GAIA_dedup 0,1
# [TB-vLLM-Gemma-S] SWE_Bench_dedup
bash eval/run_threshold_sweep.sh openai/gemma-3-27b-it SWE_Bench_dedup 0,1

# Option B: DeepInfra  (inferred: google/* → deepinfra)
# [TB-DI-Gemma-G] GAIA_dedup
# bash eval/run_threshold_sweep.sh google/gemma-3-27b-it GAIA_dedup
# [TB-DI-Gemma-S] SWE_Bench_dedup
# bash eval/run_threshold_sweep.sh google/gemma-3-27b-it SWE_Bench_dedup
```

Note: ARC uses bare model IDs (`gpt-oss-120b`, no prefix); DeepInfra and vLLM
use the HF-style ID (`openai/`, `google/`, `mistralai/` …). The driver passes
`--model` through verbatim, so the ID and the inferred backend stay in sync as
long as you use the canonical form for each provider.

### Task C — Threshold Sweep (`+CG`, τ ∈ {0.35, 0.25, 0.20, random-12})

Same sweep as Task B but for the **one-pass** in-prompt graph guidance
(`+CG`, no two-pass dynamic injection, no span-index by default).
Lets the threshold-sweep ablation table contrast +CG vs +GI under the
same edge sets and show whether the two-pass injector is what scales
with edge count (the §4.5 claim) — currently only the +GI side has
sweep data.

Inner runners (per backend):
- vllm    → `eval/run_eval_with_graph_vllm.py` (`--edge_threshold τ`)
- litellm → `eval/run_eval_with_graph.py`      (`--edge_threshold τ`)
- arc     → `eval/run_eval_with_graph_api_arc.py`       (`--corr_threshold τ`)
- deepinfra → `eval/run_eval_with_graph_api_deepinfra.py` (`--corr_threshold τ`)

`--corr_threshold τ` (API runners) gives the **union of intervention-
validated causal edges and Suppes geomean ≥ τ** — matches the +GI sweep
edge sets exactly. `--edge_threshold τ` (vllm/litellm) gives plain
Suppes geomean ≥ τ with no causal union. Close in practice, slightly
different edge counts; flag in the §4.5 caption if both backends end
up in the final table.

Output root: `benchmarking/outputs_thres_cg/t<τ>/` (parallel to
`outputs_thres/t<τ>/` for +GI).
Naming inside each `t<τ>/` subdir (per the inner script):
  API runners:    `outputs_<model>-<split>-graph_causal_corr<τ>/`
  vllm/litellm:   `outputs_<model>-<split>-graph_t<τ>/`
  → with `--span_index`: `outputs_..._span_index/` (optional; default
  is +CG only, no SI, to match the +CG row in the main ablation table).

| Model | Split | τ=0.35 | τ=0.25 | τ=0.20 | random-12 |
|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-120b | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-20b | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| gpt-oss-20b | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| Mistral-Small-24B | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| Mistral-Small-24B | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |
| Qwen-32B | GAIA_dedup | ⬜ | ⬜ | ✓ (`outputs_thres_cg/t0.20/-graph_t0.2/`) | ⬜ |
| Qwen-32B | SWE_Bench_dedup | ⬜ | ⬜ | ⬜ | ⬜ |
| Gemma-3-27B | GAIA_dedup | ✓ | ✓ | ✓ | ✓ |
| Gemma-3-27B | SWE_Bench_dedup | ✓ | ✓ | ✓ | ✓ |

The three ✓ cells were generated through `run_eval_with_graph_vllm.py`
with `--edge_threshold 0.2` (plain Suppes, no causal union); they were
moved from `outputs/zero_shot2/` into `outputs_thres_cg/t0.20/` on
2026-05-15. Edge-set caveat: these τ=0.20 cells differ slightly from
what the API runners produce at the same τ (which use `--corr_threshold`,
i.e. causal ∪ Suppes ≥ τ). For the API rows we can either:
(i) accept the small edge-set mismatch and reuse these cells, or
(ii) re-run the three cells through `--corr_threshold 0.20` for a
clean union-everywhere story. Note that `--causal_only` (12 edges) is
already a subset of the union at any τ < 1.0, so the union variant is
the cleaner of the two.

Cells expected after the sweep: 4 thresholds × 5 models × 2 splits = 40,
minus the 3 vllm cells already on disk = **37 new runs** (or 40 if you
re-run the three to switch to `--corr_threshold` semantics).

#### Driver

Shipped as a **sister script** `eval/run_threshold_sweep_cg.sh`
(mirrors `run_threshold_sweep.sh` but calls the one-pass +CG runner
on every backend). Same positional args, same threshold list, same
per-model `max_model_len`, same logging + scoring at the end. The
default OUTDIR is `outputs_thres_cg` so cells land alongside the
existing τ=0.20 runs without colliding with the +GI `outputs_thres/`.

Caveats vs the +GI sweep:
1. **vllm/litellm runners use `--edge_threshold`** (plain Suppes,
   no causal union). The sweep script auto-dispatches `--corr_threshold`
   for API runners and `--edge_threshold` for vllm/litellm. Edge sets
   at the same τ differ slightly; see the caveat above.
2. **`--random_edges` only works on API backends.** vllm/litellm
   skip the `random` threshold with a warning until the random-edges
   plumbing is ported into `run_eval_with_graph_vllm.py` (mirror lines
   231–248, 502–506, 533–538 of `run_eval_graph_inject_vllm.py`).
3. **`--span_index` is OFF by default** to match the existing +CG row
   in `paper/tables/threshold_sweep_ablation.tex`. Pass `SPAN_INDEX=1`
   as an env var to flip it on.

#### Commands (run from `benchmarking/`; same auth as Task B)

```bash
# === ARC API — gpt-oss-120b ===   (auth: source path/to/arc_llm_api.sh)
# [TC-ARC-G] GAIA_dedup
bash eval/run_threshold_sweep_cg.sh gpt-oss-120b GAIA_dedup
# [TC-ARC-S] SWE_Bench_dedup
bash eval/run_threshold_sweep_cg.sh gpt-oss-120b SWE_Bench_dedup

# === DeepInfra API — gpt-oss-20b ===  (export DEEPINFRA_API_KEY=<key>)
# [TC-DI-G] GAIA_dedup
bash eval/run_threshold_sweep_cg.sh openai/gpt-oss-20b GAIA_dedup
# [TC-DI-S] SWE_Bench_dedup
bash eval/run_threshold_sweep_cg.sh openai/gpt-oss-20b SWE_Bench_dedup

# === vLLM — Mistral / Qwen / Gemma === (local GPUs)
# [TC-vLLM-Mistral-G] GAIA_dedup
bash eval/run_threshold_sweep_cg.sh \
    mistralai/Mistral-Small-3.1-24B-Instruct-2503 GAIA_dedup 0,1
# [TC-vLLM-Mistral-S] SWE_Bench_dedup
bash eval/run_threshold_sweep_cg.sh \
    mistralai/Mistral-Small-3.1-24B-Instruct-2503 SWE_Bench_dedup 0,1

# [TC-vLLM-Qwen-G] GAIA_dedup
bash eval/run_threshold_sweep_cg.sh \
    Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3
# [TC-vLLM-Qwen-S] SWE_Bench_dedup
bash eval/run_threshold_sweep_cg.sh \
    Tongyi-Zhiwen/QwenLong-L1-32B SWE_Bench_dedup 2,3

# [TC-vLLM-Gemma-G] GAIA_dedup
bash eval/run_threshold_sweep_cg.sh openai/gemma-3-27b-it GAIA_dedup 0,1
# [TC-vLLM-Gemma-S] SWE_Bench_dedup
bash eval/run_threshold_sweep_cg.sh openai/gemma-3-27b-it SWE_Bench_dedup 0,1
```

#### Follow-up to the threshold-sweep table

Once Task C is complete, regenerate
`paper/tables/threshold_sweep_ablation.tex` with paired +CG / +GI rows
per variant — most natural layout: same 5 graph variants on the row
axis but a `Method` sub-column splitting +CG vs +GI under each
variant, OR two side-by-side tables (one per method) with identical
row structure. Choose at table-build time based on whether the §4.5
contrast reads better as paired-rows or paired-tables.

### Task D — Main Results Table +CG+SI anchor gaps (Table 1)

Fill the 3 missing `+CG+SI` (causal-only) cells in
`paper/main_results_table.tex`. These are anchor runs at the 12
intervention-validated edges — no threshold sweep. Paired with the
existing `+GI+SI` column to show "+CG buys most of the gain; +GI
closes the rest" directly in the headline table.

Output dir: `outputs/zero_shot2/`
Naming: `outputs_{model}-{split}-graph_causal_only_span_index/`

| Model | Split | +CG+SI |
|---|---|---|
| Gemini-2.5-Pro | GAIA_dedup | ⬜ |
| Gemini-2.5-Pro | SWE_Bench_dedup | ⬜ |
| Mistral-Small-3.1-24B | SWE_Bench_dedup | ✓ |

#### Commands (run from `benchmarking/`)

```bash
# [TD-Pro-G] Gemini-2.5-Pro × GAIA_dedup
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [TD-Pro-S] Gemini-2.5-Pro × SWE_Bench_dedup
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [TD-Mistral-S] Mistral-Small-3.1-24B × SWE_Bench_dedup
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 108000 \
    --output_dir outputs/zero_shot2
```

Score with `python eval/calculate_scores.py --results_dir outputs/zero_shot2`.

Task D subsumes the optional P5/P7 commands at the bottom of this file
(those used the same model + split + `--causal_only` plus `--span_index`
on Pro). Drop the bottom-of-file optional block after Task D lands.

After all three cells are on disk, update `paper/main_results_table.tex`
to add a `+CG+SI` row under Pro and Mistral, and drop the `\text{---}`
markers in `Table 4` (trace coverage) for Pro and Mistral SWE.

### Task E — Who&When W1+CG+SI panel (Table 3)

Fill the 9 missing `W1+CG+SI` cells in
`paper/tables/who_and_when_results.tex`. This is the +CG analogue of
the existing `W1+GI+SI` column — one-pass static causal graph injected
into the W1 (all-at-once) prompt, span-indexed.

The runner `run_who_and_when_with_graph_*.py` already supports
`--variant w1 --causal_only --span_index`; no code change needed.

Output dir: `baselines/who_and_when/causal/outputs/`
Naming: `outputs_{model}-{split}-who_and_when_w1_graph_causal_only_span_index/`

| Model | Split | W1+CG+SI |
|---|---|---|
| Mistral-Small-3.1-24B | GAIA_dedup | ⬜ |
| Mistral-Small-3.1-24B | SWE_Bench_dedup | ✓ on disk |
| GPT-oss-120B | GAIA_dedup | ✓ |
| GPT-oss-120B | SWE_Bench_dedup | ✓ |
| GPT-oss-20B | GAIA_dedup | ✓ |
| GPT-oss-20B | SWE_Bench_dedup | ✓ |
| Gemma-3-27B-IT | GAIA_dedup | ✓ |
| Gemma-3-27B-IT | SWE_Bench_dedup | ⬜ |
| QwenLong-L1-32B | GAIA_dedup | ⬜ |
| QwenLong-L1-32B | SWE_Bench_dedup | ⬜ |

#### Commands (run from repo root; source the same API keys as Task A)

```bash
# === ARC API — gpt-oss-120b ===  (source path/to/arc_llm_api.sh)
# [TE-ARC-G] GPT-oss-120B × GAIA_dedup
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_arc.py \
    --variant w1 --split GAIA_dedup --causal_only --span_index
# [TE-ARC-S] GPT-oss-120B × SWE_Bench_dedup
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_arc.py \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index

# === DeepInfra API — gpt-oss-20b ===  (export DEEPINFRA_API_KEY=<key>)
# [TE-DI-G] GPT-oss-20B × GAIA_dedup
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model openai/gpt-oss-20b \
    --variant w1 --split GAIA_dedup --causal_only --span_index
# [TE-DI-S] GPT-oss-20B × SWE_Bench_dedup
python baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py \
    --model openai/gpt-oss-20b \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index

# === vLLM — Mistral / Gemma / QwenLong ===

# [TE-vLLM-Mistral-G] Mistral × GAIA_dedup  (SWE already on disk)
CUDA_VISIBLE_DEVICES=0,1 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --variant w1 --split GAIA_dedup --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 108000

# [TE-vLLM-Gemma-G] Gemma × GAIA_dedup
CUDA_VISIBLE_DEVICES=0,1 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model openai/gemma-3-27b-it \
    --variant w1 --split GAIA_dedup --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 108000
# [TE-vLLM-Gemma-S] Gemma × SWE_Bench_dedup
CUDA_VISIBLE_DEVICES=0,1 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model openai/gemma-3-27b-it \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 108000

# [TE-vLLM-Qwen-G] QwenLong × GAIA_dedup
CUDA_VISIBLE_DEVICES=2,3 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --variant w1 --split GAIA_dedup --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 128000
# [TE-vLLM-Qwen-S] QwenLong × SWE_Bench_dedup
CUDA_VISIBLE_DEVICES=2,3 python baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --variant w1 --split SWE_Bench_dedup --causal_only --span_index \
    --tensor_parallel_size 2 --max_model_len 128000
```

Score with the standard W&W scoring path
(`python benchmarking/eval/calculate_scores.py` pointed at each
output dir).

After the 9 cells land, add the `W1+CG+SI` row to the W1 block of
`paper/tables/who_and_when_results.tex` (parallel to the existing
`W1+GI+SI` row), and re-bold per-column maxima across all five
graph-variant rows per model.

---

## Ablation plan (post-stage-pipeline drop)

We are NOT running the stage-by-stage construction ablation
(Suppes-only / CAPRI-pre-validation as separate graph variants).
The remaining ablation set is three studies, each defending one paper claim:

| # | Name | Defends | Status |
|---|---|---|---|
| **1** | **Span-Index Orthogonality** (`+SI` alone) | "+SI is orthogonal to the graph" (§4.3) — separates SI gain from graph gain | only QwenLong/GAIA-orig on disk; need open-source panel |
| **2** | **Edge-Richness × Injection Architecture** (`paper/ablation_graph_richness.tex`) | "static `+CG` does not scale with edge count, but two-pass `+GI+SI` does" (§4.5) | mostly done at τ=0.20; outstanding cells listed under [A2-G1] / [A2-G2] / [A2-CGSI] below |
| **3** | **Edge-Richness Threshold Sweep** (extension of Ablation 2) | "the graph-richness curve has a knee; chosen τ is on the right side of it" | in progress via `run_threshold_sweep.sh` over {0.18, 0.15, 0.05} on full open-source panel × both splits |

Existing on-disk artifacts to reuse:
- causal-only +GI+SI (every model × both splits) — `outputs/zero_shot2/*-graph_inject_causal_only_span_index/`
- corr 0.20 +GI+SI and +CG — moved into `outputs_thres/t0.20/` (was `outputs_corr/`)
- Threshold sweep dirs land in `outputs_thres/t<τ>/` per the sweep script.

---

## Ablation 1 — Span-Index Orthogonality (`+SI` alone)

### Goal

Add a single `+SI` row per (model, split) to decompose:
- SI-only gain  = `+SI` − Baseline
- Graph-only gain = `+CG` − Baseline
- Graph × SI interaction = `+GI+SI` − (`+SI` + `+CG` − Baseline)

### Existing runs

Only one `+SI` run is on disk:

| Model | Split | Path | N | W-F1 / Loc / Joint |
|---|---|---|---|---|
| QwenLong-L1-32B | GAIA (orig) | `outputs/zero_shot/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_span_index/` | 117 | 16.77 / 11.80 / 3.24 |

This row is on the **non-dedup** GAIA split, so it isn't directly comparable
to the dedup main-results table. Treat as a sanity-check anchor, not a table
cell.

### Missing runs (open-source panel, dedup splits)

Method: zero-shot baseline + `--span_index` (no graph). Use:
- `eval/run_eval_vllm.py --span_index` for vLLM models.
- `eval/run_eval.py --span_index` for litellm/Gemini.

Output naming: `outputs_{model_tag}-{split}_span_index/`.

```bash
# === GAIA_dedup ===
# [A1-G1] Mistral-Small-24B
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --span_index \
    --tensor_parallel_size 2 --max_model_len 108000 \
    --output_dir outputs/zero_shot2

# [A1-G2] GPT-oss-120B
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/run_eval_vllm.py \
    --model openai/gpt-oss-120b --split GAIA_dedup --span_index \
    --tensor_parallel_size 4 --output_dir outputs/zero_shot2

# [A1-G3] GPT-oss-20B
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_vllm.py \
    --model openai/gpt-oss-20b --split GAIA_dedup --span_index \
    --tensor_parallel_size 2 --output_dir outputs/zero_shot2

# [A1-G4] Gemma-3-27B-IT
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_vllm.py \
    --model openai/gemma-3-27b-it --split GAIA_dedup --span_index \
    --tensor_parallel_size 2 --output_dir outputs/zero_shot2

# [A1-G5] QwenLong-L1-32B  (re-run on dedup, not orig)
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split GAIA_dedup --span_index \
    --tensor_parallel_size 2 --max_model_len 128000 \
    --output_dir outputs/zero_shot2

# === SWE_Bench_dedup ===  (same 5 models, --split SWE_Bench_dedup)
# [A1-S1..S5] mirror the GAIA commands above with --split SWE_Bench_dedup
```

Score with `python eval/calculate_scores.py --results_dir outputs/zero_shot2`.

Closed-source Gemini Flash / Pro: optional — only Flash already has all four
graph variants in Table 2; adding +SI to that single row is the cheapest
demonstration of orthogonality.

---

## Ablation 2 — Edge-Richness × Injection Architecture

Source table: `paper/ablation_graph_richness.tex` (currently three rows per
model: `+CG (corr, t≥0.2)`, `+GI+SI (causal-only)`, `+GI+SI (corr ≥0.2)`).

### Outstanding cells

Three known gaps, all noted previously in this TODO and now grouped under
Ablation 2:

#### [A2-G1] Mistral `+CG (corr t=0.2)` — locate or re-run on both splits

The Mistral `+CG (corr, t≥0.2)` row in `paper/ablation_graph_richness.tex`
cites W-F1 = 29.33 (GAIA) and 9.31 (SWE), but the source metrics file
location was unclear after the move from `outputs_corr/` to
`outputs_thres/t0.20/`. Confirm presence; re-run if missing.

```bash
# 1. Search for any Mistral suppes/corr-t0.2 metrics file in the repo
find . -type f \( -name "*Mistral*graph_suppes_t0.2*-metrics.txt" \
                  -o -name "*Mistral*-graph_t0.2-metrics.txt" \) 2>/dev/null

# 2. If none found, re-run from benchmarking/ (one-pass +CG, edge_threshold=0.2)
python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup \
    --edge_threshold 0.2 \
    --output_dir outputs_thres/t0.20

python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup \
    --edge_threshold 0.2 \
    --output_dir outputs_thres/t0.20

# 3. Score
python eval/calculate_scores.py --results_dir outputs_thres/t0.20
```

#### [A2-G2] QwenLong `+GI+SI (corr ≥ 0.2)` — finish GAIA partial coverage

Existing `outputs_thres/t0.20/outputs_openai-Tongyi-Zhiwen-...-graph_inject_suppes_t0.2_span_index/`
covers only 84/117 GAIA traces (33 missing); the run was launched against
a non-dedup file set. SWE-dedup is fully covered (31/31). The current paper
table flags GAIA with $\dagger$ for partial coverage.

Root cause to fix before re-running: the corr-ablation script
`eval/run_eval_graph_inject_vllm.py` reuses the same JSON parser as the
Who&When baseline runner — no Harmony stripping, no balanced-brace JSON
extractor, no open-`<think>` recovery, and no reasoning-model `max_tokens`
bump. QwenLong's reasoning chain blows past `max_new_tokens`, the JSON is
truncated mid-string, and `parse_json_output` returns `None` for affected
traces.

```bash
# 1. Port the 3 fixes from baselines/who_and_when/run_who_and_when_vllm.py
#    into benchmarking/eval/run_eval_graph_inject_vllm.py:
#      (a) _strip_harmony helper
#      (b) _balanced_json_object helper used by parse_json_output
#      (c) auto-bump max_new_tokens 8000 -> 24000 for reasoning models
#          matching r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)"

# 2. Re-run from benchmarking/ once parser is fixed
CUDA_VISIBLE_DEVICES=2,3 python eval/run_eval_graph_inject_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split GAIA_dedup \
    --corr_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --max_model_len 128000 \
    --output_dir outputs_thres/t0.20

CUDA_VISIBLE_DEVICES=2,3 python eval/run_eval_graph_inject_vllm.py \
    --model Tongyi-Zhiwen/QwenLong-L1-32B \
    --split SWE_Bench_dedup \
    --corr_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --max_model_len 128000 \
    --output_dir outputs_thres/t0.20

# 3. Score
python eval/calculate_scores.py --results_dir outputs_thres/t0.20
```

After the re-run produces a cleanly-named
`outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_dedup-graph_inject_causal_corr0.2_span_index/`
folder, drop the $\dagger$ from the QwenLong corr0.2 row and remove the
partial-coverage caption note.

Note: gpt-oss runs in `outputs_thres/t0.20/` are non-empty and produced
non-trivial W-F1, suggesting the parser bug only catastrophically breaks
reasoning models (long `<think>` chains that exceed the token budget). For
gpt-oss the Harmony channel leakage may still be silently degrading quality
— worth re-running them after the parser fix to quantify.

#### [A2-CGSI] Add `+CG+SI (corr t=0.2)` row across the open-source panel

```bash
# Open-source panel × both splits, +CG+SI corr0.2.
# Output dir naming: outputs_{model}-{split}-graph_t0.2_span_index/

# Mistral
python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup --edge_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --output_dir outputs_thres/t0.20
python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup --edge_threshold 0.2 --span_index \
    --tensor_parallel_size 2 --output_dir outputs_thres/t0.20

# Repeat for: openai/gemma-3-27b-it, openai/gpt-oss-120b (TP=4),
# openai/gpt-oss-20b, Tongyi-Zhiwen/QwenLong-L1-32B (max_model_len=128000)
```

Score with `python eval/calculate_scores.py --results_dir outputs_thres/t0.20`.

---

## Ablation 3 — Edge-Richness Threshold Sweep

Extends Ablation 2 by sweeping the corr / geomean threshold τ across more
points so the graph-richness curve has a knee that justifies the chosen τ
in `paper/ablation_graph_richness.tex`. Output is a 3-panel plot per
(model, split): W-F1 vs τ, `N` vs τ, and Pass-2 prompt tokens vs τ.

### Edge landscape (reference, from `suppes_graph.json`)

Threshold score is `sqrt(precedence * pr_delta)` (the two graded Suppes
statistics; geomean is monotone in each and independent across them).
Edge counts are the **union with the 12 intervention-validated causal
edges** (the inject code admits each Suppes edge that either passes the
threshold or is in the causal set).

| geomean τ | ∪ causal edges | Notes |
|---|---|---|
| causal_only (intervention-validated) | 12 | anchor; 11/20 categories |
| ≥ 0.50 | 12 | only causal — redundant |
| ≥ 0.45 | 13 | first observational edge enters |
| ≥ 0.40 | 14 | |
| ≥ 0.35 | 19 | knee: largest +5 jump |
| ≥ 0.30 | 20 | |
| ≥ 0.25 | 21 | mid-regime |
| ≥ 0.20 | 25 | saturating end of sweep |
| random-12 (seed 42) | 12 | null-graph control (non-Suppes pairs over full taxonomy) |

The action lives in τ ∈ [0.35, 0.20]. Above 0.45 collapses to causal-only;
below 0.20 adds the 2 remaining low-score Suppes edges.

7 categories are structurally uncoverable in the Suppes graph (no
co-occurrence data): Instruction Non-compliance, Tool Definition Issues,
Rate Limiting, Service Errors, Resource Not Found, Resource Exhaustion,
Timeout Issues. The random-12 baseline samples over the full 20-category
taxonomy (so it can land on these uncoverable categories — by design,
to avoid handing the random baseline domain knowledge).

### Sweep spec

- **Method**: `+GI+SI` only (two-pass; one-pass `+CG` is a separate ablation).
- **Thresholds (3 sweep points + 1 baseline)**: `{0.35, 0.25, 0.20}` plus
  the **random-12** null-graph control.
  → corr edge counts **19 → 21 → 25** (monotone), plus **random-12** at
  12 edges (count-matched to causal-only).
  Causal-only stays as the main-table anchor row but is not re-run by
  the sweep script (it's already on disk for every model × split).
- **Score change vs prior sweep**: the threshold score moved from
  `sqrt(P(B|A)*pr_delta)` to `sqrt(precedence*pr_delta)`. The previous
  factors were not independent (P(B|A) appeared in both), so the geomean
  did not have its intended "both signals must be substantial"
  interpretation. The new score uses the two independent Suppes
  statistics (precedence and probability-raising). **All prior corr-threshold
  outputs under `outputs_thres/t0.20/` and `outputs_corr/` are
  invalidated** and need re-running.
- **Models — full open-source panel**:
  1. **GPT-oss-20B** — largest corr0.2 gain on GAIA (+9.75 W-F1).
  2. **GPT-oss-120B** — largest combined GAIA+SWE lift (+6.30 / +8.19).
  3. **QwenLong-L1-32B** — context-fragility candidate; score the new
     `outputs_corr/outputs_openai-Tongyi-Zhiwen-...-graph_inject_suppes_t0.2_span_index/`
     dirs first to recover the existing τ=0.20 point.
  4. **Mistral-Small-3.1-24B** — already has clean +6.17 GAIA at corr0.2.
  5. **Gemma-3-27B-IT** — only regression model (-4.31 GAIA, -2.41 SWE);
     tests whether the regression is monotonic in τ or driven by a
     single bad edge.
- **Splits**: **both** `GAIA_dedup` and `SWE_Bench_dedup`. SWE `N` is
  noisy but corr0.2 lift is large for GPT-120B/20B — directionally informative.
- **Reused runs**: `causal_only` is on disk for every (model, split);
  `corr0.2` is on disk for every (model, split) except QwenLong (score
  pending). The sweep script runs only the 3 new thresholds:
  **3 × 5 models × 2 splits = 30 new runs**.

### Driver script

`eval/run_threshold_sweep.sh` (already updated to the 5-point list)
runs every threshold for one (model, split) pair and triggers scoring
at the end:

```bash
# Usage: eval/run_threshold_sweep.sh <model> <split> [gpus] [output_dir] [backend]

# === GAIA_dedup ===
# [TS-G1] GPT-oss-20B
eval/run_threshold_sweep.sh openai/gpt-oss-20b           GAIA_dedup 0,1
# [TS-G2] GPT-oss-120B
eval/run_threshold_sweep.sh openai/gpt-oss-120b          GAIA_dedup 0,1,2,3
# [TS-G3] QwenLong-L1-32B   (requires parser fix from [A2-G2])
eval/run_threshold_sweep.sh Tongyi-Zhiwen/QwenLong-L1-32B GAIA_dedup 2,3
# [TS-G4] Mistral-Small-24B
eval/run_threshold_sweep.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503 GAIA_dedup 0,1
# [TS-G5] Gemma-3-27B-IT
eval/run_threshold_sweep.sh openai/gemma-3-27b-it        GAIA_dedup 0,1

# === SWE_Bench_dedup ===
# [TS-S1] GPT-oss-20B
eval/run_threshold_sweep.sh openai/gpt-oss-20b           SWE_Bench_dedup 0,1
# [TS-S2] GPT-oss-120B
eval/run_threshold_sweep.sh openai/gpt-oss-120b          SWE_Bench_dedup 0,1,2,3
# [TS-S3] QwenLong-L1-32B
eval/run_threshold_sweep.sh Tongyi-Zhiwen/QwenLong-L1-32B SWE_Bench_dedup 2,3
# [TS-S4] Mistral-Small-24B
eval/run_threshold_sweep.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503 SWE_Bench_dedup 0,1
# [TS-S5] Gemma-3-27B-IT
eval/run_threshold_sweep.sh openai/gemma-3-27b-it        SWE_Bench_dedup 0,1
```

The script writes per-threshold logs to
`outputs/zero_shot2/_sweep_logs/<model>-<split>-t<τ>.log`,
emits one `*-metrics.txt` per threshold, and runs
`calculate_scores.py` at the end. Re-running existing
`causal_only` / `corr_threshold 0.20` thresholds is idempotent — the
inner script overwrites prior outputs in the same dir.

### Pre-flight

1. **Score existing QwenLong corr0.2 dirs** so the τ=0.20 point is
   available without re-running:
   ```bash
   python eval/calculate_scores.py --results_dir outputs_corr
   ```
   Targets: `outputs_openai-Tongyi-Zhiwen-...graph_inject_suppes_t0.2_span_index/`
   GAIA (177 trace files) and SWE (63 trace files).
2. **Reconcile edge-count discrepancy** (15 vs. 17/18 at τ=0.20). Dump
   the edge set once with `--corr_threshold 0.20` to confirm the script
   is unioning causal-validated edges with geomean-filtered ones, then
   correct Table 6 caption in `paper/ablation_graph_richness.tex`.

### Plot

After the sweep completes, parse W-F1 / Loc / Joint / `N` from the
metrics files and plot τ on the x-axis (descending: causal-only on the
left, 0.05 on the right). Three panels per (model, split): W-F1, `N`,
and median Pass-2 prompt tokens (the third requires either logging in
the inject script or post-hoc tokenisation of the saved prompts).
Optional 4th panel: false-positive rate (edges suggested but
category-wrong) as a hallucination proxy at high-edge thresholds.

---

## Optional Ablation — BIC vs AIC capri criterion (skip if time-boxed)

**Status: optional, likely deferred — only ~2 weeks until deadline.**

The methodology paper says we use AIC for CAPRI structure learning (favors
sensitivity over sparsity given the moderate trace count). Both repos are
now coded to default AIC:

- TRAIL: eval scripts already point at `data/trail_causal_outputs_full_gaia_swe_AIC/`
  (criterion=AIC, 13 edges → 12 intervention-validated). Active artifact.
- MAST: defaults flipped from BIC → AIC in `run_causal_pipeline.py`,
  `CAPRI/3_capri_prune.py`, `CAPRI/4_bootstrap_stability.py`,
  `visualize_graphs.py`, `workflow.md` (commit pending). On-disk
  `outputs/capri_graph.json` was produced under BIC and is still the
  artifact consumed by intervention validation; AIC artifacts will only
  take effect after re-running the pipeline + intervention stage.

If a reviewer asks "why AIC?", the defensible answer is the BIC artifact set:

| Repo | BIC capri | AIC capri |
|---|---|---|
| TRAIL | 7 edges (in `trail_causal_outputs_BIC_old/`) | **13 edges** (active) |
| MAST  | 14 edges (current `outputs/capri_graph.json`) | 23 edges (in `outputs/capri_graph_aic.json`) |

So the *graph counts already exist* on both sides — the ablation only requires:

### What would need to run

1. **TRAIL — already partially defensible from on-disk BIC artifact**
   - `data/trail_causal_outputs_BIC_old/capri_graph.json` (7 edges) is on disk.
   - Would need: intervention validation re-run on the 7 BIC edges, then a
     small panel run (1 model × 1 split, +GI+SI) under both BIC-validated
     and AIC-validated edges to show the W-F1 delta.
   - Compute: ~7 intervention runs (vs. 13 already done for AIC) +
     1 inference run per (model, split) we want in the table.

2. **MAST — full pipeline + intervention re-run**
   - Re-run pipeline with `--criterion BIC` (already what the existing
     `outputs/capri_graph.json` reflects) AND `--criterion AIC` to
     produce both intervention edge sets cleanly.
   - Run inference under each on the AG2 panel.

### Recommendation given the deadline

Ship as a 1-paragraph note in the implementation appendix citing the
existing on-disk edge counts (TRAIL 7 vs 13; MAST 14 vs 23) and the
methodology argument that AIC's lower complexity penalty is appropriate
for moderate-N corpora. Only run the full ablation if a reviewer
specifically demands it post-submission.

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


Two modes now available for TRAIL:

  # Default: only validated edges' nodes (11 nodes, 12 edges) — clean
  python causal/graph/visualize_graphs.py \
      --effect_edges  benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json \
      --hierarchy     benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/hierarchy_levels.json \
      --out_dir       figures/graph

  # With --show_isolated --suppes: all 13 Suppes-universe categories (full taxonomy under test)
  # Renders Environment Setup Errors + Task Orchestration as isolated nodes
  python causal/graph/visualize_graphs.py \
      --effect_edges  benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json \
      --suppes        benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json \
      --hierarchy     benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/hierarchy_levels.json \
      --show_isolated \
      --out_dir       figures/graph

  The current figures/graph/graph_causal.png is regenerated with --show_isolated, so it shows all 13
  Suppes-universe categories. Two are isolated:
  - Environment Setup Errors — appeared in the Suppes graph (had observational co-occurrence) but no
  edge survived intervention validation 
  - Task Orchestration — same situation

  This is informative: the figure now communicates "we tested 13 categories; intervention validation
  kept causal edges between 11 of them." If you want the cleaner version without isolated nodes, drop
  --show_isolated --suppes.