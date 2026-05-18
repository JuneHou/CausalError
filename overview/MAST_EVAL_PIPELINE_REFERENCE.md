# MAST Evaluation Pipeline — Reference

> One-stop reference for the MAST LLM-as-judge eval pipeline (`MAST/eval/`). Takes the Stage 1-2 artifacts from `MAST_CAUSAL_PIPELINE_REFERENCE.md` and produces the metrics that drive paper tables.

**Module path**: `/data/wang/junh/githubs/MAST/eval/` (lives in the sibling MAST repo; doc kept here because MAST blocked writes during generation)

**Compiled**: 2026-05-16 by 5 parallel sonnet agents (A baseline, B +CG, C +GI, E sweep, F scoring).

**MAST-specific design facts**:
- One dataset: 393 AG2 traces (`data/annotation/annotation_ag2_filtered.jsonl`); no train/test split
- 13-leaf taxonomy (`1.1–1.5`, `2.1–2.4`, `2.6`, `3.1–3.3`); `2.5` is always 0 in AG2 and silently dropped
- **No location/span prediction** — `--span_index` is a no-op accepted only for sweep-CLI parity
- `full_` prefix denotes the **production** runners (the non-`full_` ones are legacy/superseded)
- Single scorer for everything: `eval/calculate_scores_yesno.py`

---

## Architecture decision matrix

| | Baseline (A) | +CG (B) | +GI (C) |
|---|---|---|---|
| LLM calls per trace | 1 | 1 | 2 (Pass-1 + targeted Pass-2 when triggered) |
| Graph in prompt? | no | yes (full block) | yes (Pass-1 full; Pass-2 filtered subgraph) |
| Pass-2 merge rule | n/a | n/a | **logical OR** (Pass-2 can only flip no→yes, never reverts a Pass-1 detection) |
| `--span_index`? | no-op | no-op | no-op |
| Scorer output | `*-metrics.json` | `*-metrics.json` | `*-metrics.json` |

---

## Pipeline overview

```
                ┌──────────────── Stage 1-2 artifacts ────────────────┐
                │  • suppes_graph.json   (M2 in causal pipeline)      │
                │  • effect_edges.json   (M7 in causal pipeline)      │
                │  see MAST_CAUSAL_PIPELINE_REFERENCE.md              │
                └────────────────────┬────────────────────────────────┘
                                     │  loaded once via load_graph_edges()
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                          Stage 3: Eval                                │
   │                                                                       │
   │   Module A (baseline)     Module B (+CG)         Module C (+GI)      │
   │   run_eval_yesno*.py     full_run_eval_with_     full_run_eval_      │
   │                            graph*.py              graph_inject*.py    │
   │                                                                       │
   │   Module E sweep drivers (run_threshold_sweep[_cg].sh)               │
   │   dispatch B or C across {random-11, τ=0.60, τ=0.50, τ=0.40}         │
   │                                  │                                    │
   │                                  ▼                                    │
   │                          Module F (scoring)                           │
   │                          calculate_scores_yesno.py                    │
   │                                  │                                    │
   └──────────────────────────────────┼────────────────────────────────────┘
                                      ▼
                           <pred_dir>-metrics.json
                           (paper-table values)
```

---

# Module A: Baseline Eval (no graph)

**Purpose**: Predict yes/no for each of MAST's 13 leaf categories on every AG2 trace using a single LLM call per trace, with no graph guidance.

**Code**:
| Script | Backend |
|---|---|
| `eval/run_eval_yesno.py` | litellm (original o1 paper baseline) |
| `eval/run_eval_yesno_api.py` | litellm (adds `--sample_indices`; **liberal annotation prompt**) |
| `eval/run_eval_yesno_vllm.py` | in-process vLLM (`LLM.chat`) |

No DeepInfra/ARC baseline runners. The full panel uses one of the three above.

### Inputs
- `data/annotation/annotation_ag2_filtered.jsonl` (393 lines)
- `taxonomy_definitions_examples/definitions.txt` + `examples.txt` (loaded at module init)
- No train/test split — all 393 traces are the eval set

### Output
- Dir: `<output_dir>/<model_tag>-yesno-baseline[/-thinking]/` (vLLM adds `-thinking` when `--enable_thinking` set)
- Per-trace JSON: `{rec_id, trace_id, predictions: {13 codes: 0/1}, raw_response, thinking?}`
- Metrics: `<dir>-metrics.json` written by Module F (not by the runner)

### Key CLI args / defaults
| Arg | `run_eval_yesno.py` | `_api.py` | `_vllm.py` |
|---|---|---|---|
| `--model` | `openai/o1` | `openai/o1` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `--input` | `data/annotation/annotation_ag2_filtered.jsonl` | same | same |
| `--output_dir` | `outputs` | `outputs_o1` | `outputs` |
| `--max_workers` | `1` | `1` | — |
| `--sample_indices` | — | `None` | — |
| `--tp` | — | — | auto from `CUDA_VISIBLE_DEVICES` |
| `--batch_size` | — | — | `8` (docstring claims 32 — stale) |
| `--max_tokens` | — | — | `8000` |
| `--max_model_len` | — | — | `108000` |
| `--gpu_memory_utilization` | — | — | `0.9` |
| `--enable_thinking` | — | — | `False` (adds `-thinking` suffix; sets `chat_template_kwargs={"enable_thinking": True}`) |

### Prompt structure
One LLM call per trace; **all 13 categories asked in a single prompt** (not one per category).
- `run_eval_yesno.py`: replicates the original `llm_judge_pipeline.ipynb` format; definitions + examples appended after trace.
- `run_eval_yesno_vllm.py` and `_api.py`: cleaner reformat — definitions + examples **prepended** before trace; `@@`-delimited answer block.
- `_api.py` additionally uses a **liberal annotation standard** ("Every trace contains AT LEAST 2 failure modes; lean toward yes"). Conservative variants in the other two scripts.

### Output parsing
`parse_response()` strips `@@` markers, then for each of the 13 codes tries 3 regex patterns: `code…: yes/no`, `code yes/no`, `code\n yes/no`. First match wins; default `0`. `ContextWindowExceededError` → all-zero. vLLM additionally calls `strip_thinking` first.

### Concurrency
- `run_eval_yesno.py` / `_api.py`: `ThreadPoolExecutor(max_workers=N)`; idempotent (skip existing files)
- `_vllm.py`: single-threaded driver over `--batch_size`-sized batches; pending-record filter runs before model load (cheap restarts)

### Production-run intermediate statistics

| Model | Script | n | W-F1 |
|---|---|---|---|
| `openai/o1` (100-trace subset) | `_api.py` | 100 | 0.0908 |
| `openai/gpt-4o` (conservative) | `run_eval_yesno.py` | 393 | 0.0946 |
| `openai/gpt-4o` (**liberal**) | `_api.py` | 393 | **0.2287** |
| `google/gemma-3-27b-it` | vLLM | 393 | 0.1418 |
| `gpt-oss-120b` | vLLM | 393 | 0.1784 |

**The GPT-4o liberal-vs-conservative prompt swap (W-F1 0.0946 → 0.2287, +0.134) is a much bigger effect than any graph augmentation**. The published "GPT-4o baseline = 0.2287" implicitly uses the liberal prompt.

### Gotchas
- `.env` API keys loaded via `python-dotenv`; `litellm.drop_params = True` set globally
- Reasoning model detection (`o1|o3|o4|anthropic|gemini-2.5`): `reasoning_effort="high"`, no temp/top_p
- vLLM `strip_thinking` handles both `<think>…</think>` pairs AND orphan `</think>` (QwQ pattern where chat template injects the opening tag, so generated tokens begin mid-thought)
- **Annotation-standard divergence** between the three scripts has more impact than graph methods on GPT-4o; document the variant used when comparing

---

# Module B: +CG — One-Pass Causal Graph Guidance

**Purpose**: Prepend a static causal-graph block to the single MAST yes/no prompt.

**Code** (4 production runners, all with `full_` prefix):
| File | Backend |
|---|---|
| `eval/full_run_eval_with_graph.py` | vLLM (canonical) |
| `eval/full_run_eval_with_graph_api.py` | litellm (GPT-4o, o1) |
| `eval/full_run_eval_with_graph_api_deepinfra.py` | DeepInfra |
| `eval/full_run_eval_with_graph_api_arc.py` | ARC (gpt-oss-120b) |

**Legacy/superseded**: `eval/run_eval_with_graph.py`, `eval/run_eval_with_graph_api.py` (older, no `full_` prefix, no `code(name)` edge format).

API runners import `load_graph_edges`, `format_graph_guidance`, `format_trace`, `get_prompt`, `strip_thinking`, `parse_response` from `full_run_eval_with_graph` via `sys.path.insert`. Single source of truth.

### Backend routing
| Model | Backend |
|---|---|
| `gpt-oss-120b` | ARC |
| `openai/gpt-oss-20b`, `google/gemma-3-27b-it`, `mistralai/Mistral-Small-3.1-*` | DeepInfra |
| `Qwen/QwQ-32B` (with `--enable_thinking`) | vLLM |
| `openai/gpt-4o`, `openai/o1` | litellm |

### Output dir naming
```
<output_dir>/<model_tag>-yesno-with-graph-codename-<graph_tag>[-thinking]/
```
Defaults: `outputs_full` (vLLM/DI/ARC) or `outputs_full_api` (litellm).

`<graph_tag>`: `causal_only`, `corr<τ>` (e.g. `corr0.5`), `t<thr>`, `random<N>_seed<S>`.

**Note**: "codename" is hardcoded in the naming pattern (line 402 of vLLM file), not a CLI flag. It indicates the `code(name)` edge rendering format `1.1(Disobey Task Specification) -> 3.3(...)  (causal effect: X.XX)`.

### Key CLI args / defaults
Graph mode (mutually exclusive; runtime-enforced):
- `--causal_only` (False) — load 11 validated edges, weight = `abs(delta)`
- `--corr_threshold τ` (1.0) — UNION (Suppes geomean ≥ τ) ∪ (validated causal), weight = geomean
- `--edge_threshold t` (0.2) — pure Suppes (no causal union)
- `--random_edges` (False) — `--random_n 11`, `--random_seed 42`

Graph paths: `--effect_edges` (default `causal_graph/outputs/interventions/effect_edges.json`), `--suppes_graph` (default `causal_graph/outputs/suppes_graph.json`), `--stability_graph` (accepted for parity, unused).

Common: `--model`, `--input`, `--output_dir`, `--max_tokens` (vLLM 8000; DI/ARC 2000 → auto-bump 16000 for reasoning models; litellm 4000/8000), `--model_tag`.

vLLM: `--tp` (auto), `--batch_size 8`, `--max_model_len 128000`, `--gpu_memory_utilization 0.8`, `--enable_thinking` (False).

DeepInfra: `--rpm 600`, `--max_retries 5`.
ARC: `--rpm 30`, `--rph 1000`, `--rp3h 3000`, `--max_retries 5`.
litellm: `--max_workers 1`, `--reasoning_effort "high"`.

**`--span_index`** confirmed no-op (line 341 vLLM; line 245 DeepInfra; line 232 ARC).

### Prompt structure
`get_prompt(trace_text, graph_guidance)`: role preamble → DEFINITIONS → EXAMPLES → **graph guidance block** → annotation instructions (liberal in `_api.py`) → `@@`-delimited answer template + worked example → trace.

`format_graph_guidance`:
- causal-only: `CAUSAL ERROR PATTERNS (intervention-validated): ... 1.3(Step Repetition) -> 3.3(No or Incorrect Verification)  (causal effect: 0.14)`
- observational: `CORRELATED ERROR PATTERNS (observational, precedence-filtered): ... (observational score: X.XX)`
- random: `RANDOM ERROR PATTERN BASELINE (uncalibrated): ...` (no weights)

### Concurrency / backend gotchas
- **DeepInfra context-overflow short-circuit** (line 129): wraps 400s in 500s; retry loop checks `str(e)` for `"maximum context length"` / `"BadRequestError"` and re-raises immediately
- **ARC 3-rule fairshare**: `[(30, 60), (1000, 3600), (3000, 10800)]`; longest required sleep applied
- **DeepInfra reasoning auto-bump**: `max_tokens 2000 → 16000` for models matching `gpt-oss|qwenlong|-l1-|deepseek-r1|qwq|thinking` (line 269); `-thinking` dir suffix is separately controlled by `"thinking" in args.model.lower()`
- **vLLM `--enable_thinking`**: passes `chat_template_kwargs={"enable_thinking": True}`; injects `<think>` so output begins mid-thought (orphan `</think>` handled by `strip_thinking`)

### Production-run intermediate statistics (causal_only)

| Model | Output | W-F1 |
|---|---|---|
| Mistral-Small-24B | `outputs_full/...mistralai-...-codename-causal_only/` | **0.4731** (winner) |
| gpt-oss-120b | `outputs_full/gpt-oss-120b-...-causal_only/` | 0.2162 |
| gpt-oss-20b | `outputs_full/openai-gpt-oss-20b-...-causal_only/` | 0.1872 |
| Gemma-3-27B | `outputs_full/google-gemma-3-27b-it-...-causal_only/` | 0.1323 |

Mistral's 0.4731 is driven by very high recall on `2.6` (F1=0.80, recall=0.93), `1.3` (F1=0.55, recall=0.53), `3.3` (F1=0.42, recall=0.80).

### Threshold-sweep coverage (`outputs_thres_cg/`)
| τ tag | gpt-oss-120b W-F1 |
|---|---|
| `corr0.4` | 0.2249 |
| `corr0.5` | 0.2163 |
| `corr0.6` | 0.2169 |
| `random11_seed42` (null) | 0.2309 |

No corr-sweep cells yet for Mistral or QwQ under `outputs_thres_cg/` (pending — see Task 4 in MAST/TODO.md).

---

# Module C: +GI — Two-Pass Dynamic Graph Injection

**Purpose**: Pass-1 with full graph block → propagate detected categories → Pass-2 with filtered subgraph → merge.

**Code** (4 production runners with `full_` prefix):
| File | Backend |
|---|---|
| `eval/full_run_eval_graph_inject.py` | vLLM (**canonical**) |
| `eval/full_run_eval_graph_inject_api.py` | litellm |
| `eval/full_run_eval_graph_inject_api_deepinfra.py` | DeepInfra |
| `eval/full_run_eval_graph_inject_api_arc.py` | ARC |

API variants import all prompts/parsing/graph logic from `full_run_eval_graph_inject` via `sys.path.insert`.

### Per-trace flow
1. **Pass-1**: full trace + full graph block → yes/no for all 13 categories
2. **Propagate**: `propagate_confidence(detected_cats, edges, threshold=0.10)` — accumulate incoming edge weight per undetected target; survivors → Pass-2 targets
3. **Pass-2**: targeted re-call with **filtered subgraph + Pass-1 results**; LLM revisits only the propagated targets; "err on the side of YES when there is any plausible indication — even indirect"
4. **Merge**: **logical OR** — `if val == 1 and merged.get(cat, 0) == 0: merged[cat] = 1`. Pass-2 cannot revert Pass-1 detections. `pass2_upgrades` records net additions.

### Edge cases
- Pass-1 detects zero categories → `detected = []`; propagation not called; Pass-2 skipped
- Propagation yields zero edges → Pass-2 skipped; `pass2_triggered: false`
- Pass-1 context overflow → all-zero `p1_pred`; Pass-2 inevitably skipped (zero detections)
- Pass-2 failure → kept Pass-1 results; `pass2_error` field set

### Output dir naming
```
<output_dir>/<model_tag>-yesno-graph-inject-codename-<graph_tag>[-thinking]/
```
Defaults: `outputs_full` (vLLM/DI/ARC), `outputs_full_api` (litellm).

Per-trace JSON: same fields as +CG plus `pass2_triggered`, `pass2_upgrades`, `pass2_raw` (API only), `pass2_error?`.

### Key CLI args (additional to +CG)
- `--propagation_threshold 0.10`: min summed incoming weight for a target to enter Pass-2

### Production-run intermediate statistics (causal_only)

| Model | +GI W-F1 | +CG W-F1 | Winner |
|---|---|---|---|
| **GPT-4o** | **0.2570** | 0.1857 | +GI (+7.1pp) |
| Mistral-Small-24B | 0.3701 | **0.4731** | +CG (+10.3pp) |
| gpt-oss-120b | **0.2526** | 0.2162 | +GI (+3.6pp) |
| gpt-oss-20b | **0.2168** | 0.1872 | +GI (+3.0pp) |
| Gemma-3-27B | **0.2025** | 0.1323 | +GI (+7.0pp) |
| QwQ-32B codename | 0.1369 | **0.1513** | +CG |
| QwQ-32B non-codename | **0.1717** | — | (reported in TODO) |

**Pattern**: closed-source/API models prefer +GI; instruction-tuned Mistral and thinking QwQ prefer +CG. Different lesson than TRAIL where +GI wins more uniformly.

### Threshold-sweep coverage
+GI sweep (`outputs_thres/`): **20/20 complete** across 5 models × 4 sweep points + 5 causal-only anchors in `t_causal_only/`. The duplicate metrics file `mistralai-...v2-...causal_only-metrics.json` in `outputs_full/` has identical scores to the canonical Mistral run.

---

# Module E: Threshold Sweep Drivers

**Code**:
- `eval/run_threshold_sweep.sh` (+GI)
- `eval/run_threshold_sweep_cg.sh` (+CG)

### Usage
```
eval/run_threshold_sweep[_cg].sh <model> [gpus] [output_dir] [backend]
```
**No `<split>` positional** (TRAIL has GAIA/SWE; MAST is one dataset).

### Backend inference (identical in both scripts)
| Model pattern | Backend |
|---|---|
| `gemini/*`, `openai/gpt-4*`, `openai/o*`, `anthropic/*` | litellm |
| `openai/gpt-oss-*`, `google/*` | deepinfra |
| any `<ns>/<name>` with `/` | vllm |
| bare name | arc |

(TRAIL's +CG script additionally routes `mistralai/*` to DeepInfra; MAST does not.)

### Inner-script dispatch
| Backend | +GI | +CG |
|---|---|---|
| vllm | `full_run_eval_graph_inject.py` | `full_run_eval_with_graph.py` |
| litellm | `full_run_eval_graph_inject_api.py` | `full_run_eval_with_graph_api.py` |
| deepinfra | `full_run_eval_graph_inject_api_deepinfra.py` | `full_run_eval_with_graph_api_deepinfra.py` |
| arc | `full_run_eval_graph_inject_api_arc.py` | `full_run_eval_with_graph_api_arc.py` |

### Sweep points
Default `THRESHOLDS=(random 0.6 0.5 0.4)`. Subdir names:
- `random` → `t_random11_seed42/` (`--random_edges --random_n 11`)
- `0.6/0.5/0.4` → `t0.6/`, `t0.5/`, `t0.4/` (`--corr_threshold τ`)

### Per-model `max_model_len` (vLLM only, both scripts identical)
| Model glob | `max_model_len` |
|---|---|
| `Tongyi-Zhiwen/QwenLong-L1-32B*` | 128000 |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503*` | 108000 |
| `{openai,google}/gemma-3-27b-it*` | 108000 |
| `openai/gpt-oss-20b*`, `openai/gpt-oss-120b*` | 108000 |
| `Qwen/QwQ-32B*` | 40960 |

### Thinking flag
Only `Qwen/QwQ-32B*` gets `--enable_thinking` (line 122/111). `THINKING_SUFFIX="-thinking"` then appended to expected output dir in scoring loop.

### Output dir layout
```
outputs_thres[/_cg]/
  t_random11_seed42/
    <model_tag>-yesno-{graph-inject|with-graph}-codename-random11_seed42[-thinking]/
  t0.6/ t0.5/ t0.4/    (analogous)
  t_causal_only/       (only +GI has this; +CG anchor is in outputs_full/)
  _sweep_logs/
    <model_tag>-{gi|cg}-<graph_tag>.log
```

### Scoring loop
Iterates same `THRESHOLDS`, resolves each to subdir, calls `python eval/calculate_scores_yesno.py --pred_dir <dir>`. Missing dirs emit `WARNING:` and skip. +CG also skips `random` for litellm (`SUPPORTS_RANDOM != 1`).

### Differences (+GI vs +CG)
1. Inner-script family
2. Output dir root: `outputs_thres/` vs `outputs_thres_cg/`
3. Per-cell subdir prefix: `-yesno-graph-inject-codename-` vs `-yesno-with-graph-codename-`
4. `--span_index` flag passthrough: +GI hardcodes on every vLLM call; +CG omits entirely (no-op anyway)
5. **+GI `random` seed mismatch**: passes `--random_seed 111` (line 157) but names subdir `t_random11_seed42` (line 189). **All existing +GI random data was actually generated with seed 111**, not 42. +CG correctly uses seed 42.
6. +CG detects `SUPPORTS_RANDOM=0` for litellm and skips the random point; +GI has no such guard

### Production-run sweep coverage

**Task 3 — +GI sweep** (`outputs_thres/`): **20/20 complete** (5 models × 4 sweep points), plus 5 `t_causal_only/` anchors.

**Task 4 — +CG sweep** (`outputs_thres_cg/`): **12/20 complete** — Mistral-Small-24B and QwQ-32B are entirely missing (3 models × 4 = 12 done; 2 models × 4 = 8 pending).

### Gotchas
- **Legacy `<method>` positional rejected** with explicit error (was previously `cg|gi`)
- **+GI random seed/subdir mismatch** (see above)
- **`--random_n`/`--random_seed` not passed explicitly** — uses inner-script defaults

---

# Module F: Scoring & Metrics

**Purpose**: Score per-trace yes/no predictions against ground truth; produce per-category P/R/F1 + Cohen's kappa variants + aggregate W-F1/macro-F1.

**Code**: `eval/calculate_scores_yesno.py`

### Usage
```bash
python eval/calculate_scores_yesno.py --pred_dir <path> [--annotation <path>]
```
| Arg | Default |
|---|---|
| `--pred_dir` | (required) — single dir OR parent of multiple dirs (each scored independently) |
| `--annotation` | `data/annotation/annotation_ag2_filtered.jsonl` |

### Inputs
- Annotations: JSONL keyed by line position (`rec_id = f"{idx:04d}"`); reads `mast_annotation` dict (13 codes, 0/1)
- Predictions: `<rec_id>.json` per trace; reads `predictions` dict (13 codes, 0/1); other fields ignored
- **`2.5` silently dropped** at both ends; only the 13 codes in `MAST_MODES` are scored

### Output: `<pred_dir>-metrics.json` (sibling, not inside)

```json
{
  "n_traces": 393,
  "weighted_f1": 0.0946, "macro_f1": 0.0656,
  "macro_precision": 0.2615, "macro_recall": 0.0519, "macro_accuracy": 0.6784,
  "kappa_per_label": -0.0054, "kappa_pooled": 0.0167, "kappa_per_trace": 0.0183,
  "category_metrics": {
    "1.1": {"name": "Disobey Task Specification",
            "precision": 0.4286, "recall": 0.0174, "f1": 0.0335,
            "accuracy": 0.5598, "kappa": -0.0007,
            "support": 172, "pred_positives": 7, "detection_rate": 0.0178},
    ...
  }
}
```

JSON only — no txt mirror (TRAIL's scorer emits both).

### Metric definitions
- **Weighted F1**: sklearn `f1_score(..., average="weighted", zero_division=0)` on full `(393, 13)` multilabel matrix. Headline paper metric.
- **Macro F1 / P / R**: unweighted across 13 labels. More sensitive to rare categories (e.g. `1.2` support=7).
- **Per-category P/R/F1**: raw TP/FP/FN counts pooled across traces.
- **Detection rate**: `pred_positives / n_traces` — under/over-detection diagnostic. TRAIL has nothing equivalent.
- **Three kappa variants**: per-label (macro-averaged column-wise), pooled (flattened), per-trace (macro-averaged row-wise). `kappa_per_trace` considered most comparable to inter-annotator agreement.

### Edge cases
- Missing prediction file → silently skipped (not counted in `n_traces`)
- `rec_id` not in GT → warning printed, file omitted
- Malformed JSON → **uncaught; aborts entire directory's scoring**
- Empty `pred_dir` → `None` returned; no metrics file written
- All-zero predictions → scores normally (every category gets P=R=F1=0); this is GPT-4o's failure mode

### "Codename" — three prompt generations
| Gen | Edge format | Example |
|---|---|---|
| v1 | name-only | `Disobey Task Specification -> No or Incorrect Verification` |
| v2 | code-only | `1.1 -> 3.3` |
| **current** (`outputs_full/`) | **codename** | `1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)` |

The scorer is format-agnostic — reads bare `predictions` dict keyed by codes. Edge format only changes what the LLM sees in the prompt.

### Production-run results (verified)

| Model | Condition | W-F1 | Source |
|---|---|---|---|
| Mistral-Small-24B | baseline (v2 prompt) | 0.3773 | `outputs_v2/` — **no longer on disk**; recorded in `experiments.md` row #4 |
| Mistral-Small-24B | +CG codename causal_only | **0.4731** | `outputs_full/` |
| Mistral-Small-24B | +GI codename causal_only | 0.3701 | `outputs_full/` |
| GPT-4o | baseline (liberal) | 0.2287 | `outputs_full_api/` |
| GPT-4o | +CG codename causal_only | 0.1857 | `outputs_full_api/` |
| GPT-4o | +GI codename causal_only | **0.2570** | `outputs_full_api/` |
| QwQ-32B | baseline thinking | 0.1608 | `outputs_think/` |
| QwQ-32B | +GI non-codename causal_only | **0.1717** | `outputs_think/` |
| QwQ-32B | +GI codename causal_only | 0.1369 | `outputs_think/` |
| QwQ-32B | +CG codename causal_only | 0.1513 | `outputs_think/` |

**Important**: The paper Mistral baseline (0.3773) lives in `experiments.md` only — the `outputs_v2/` directory is no longer on disk. The `causal_graph/outputs/mistralai-...-baseline-metrics.json` file is a DIFFERENT older experiment scoring 0.2103 — **not** the paper number.

### Gotchas
- **No try/except on `json.load`** — one malformed prediction file aborts the entire directory
- **Three "kappa" variants** are easy to confuse; paper uses `kappa_per_trace`
- **Three prompt generations** ("codename" etc.) affect LLM input, not scorer behavior; the scorer is format-agnostic

---

# Cross-cutting notes

## File → module map

| File | Module |
|---|---|
| `run_eval_yesno*.py` (3 files) | A |
| `full_run_eval_with_graph*.py` (4 files) | B |
| `full_run_eval_graph_inject*.py` (4 files) | C |
| `run_threshold_sweep.sh`, `run_threshold_sweep_cg.sh` | E |
| `calculate_scores_yesno.py` | F |

**Legacy/superseded** (mentioned, not documented): `run_eval_with_graph.py`, `run_eval_with_graph_api.py`, `run_eval_graph_inject.py` (non-`full_` versions). Use the `full_` prefix for production.

## Architecture stacking summary

| Run | Module | Output dir pattern |
|---|---|---|
| Baseline | A | `outputs[_o1]/<model>-yesno-baseline[/-thinking]/` |
| +CG (causal_only) | B | `outputs_full[_api]/<model>-yesno-with-graph-codename-causal_only/` |
| +CG (corr-τ) | B+E | `outputs_thres_cg/t<τ>/<model>-yesno-with-graph-codename-corr<τ>/` |
| +GI (causal_only) | C | `outputs_full[_api]/<model>-yesno-graph-inject-codename-causal_only/` |
| +GI (corr-τ) | C+E | `outputs_thres/t<τ>/<model>-yesno-graph-inject-codename-corr<τ>/` |

## Per-model backend pairing (production)
| Model | Backend | Notes |
|---|---|---|
| GPT-4o, o1 | litellm | reasoning_effort="high"; liberal prompt for o1 / `_api` variants |
| Mistral-Small-3.1-24B | DeepInfra or vLLM | tokenizer works on both |
| Mistral-Small-3.2-24B | DeepInfra only | new tokenizer rejects vLLM regex fix |
| gpt-oss-120b | ARC | only provisioned location |
| gpt-oss-20b | DeepInfra | `reasoning_effort='low'` auto-set |
| Gemma-3-27B | vLLM or DeepInfra | |
| QwQ-32B | vLLM | `--enable_thinking`; max_model_len=40960 |

## Two known structural quirks

1. **+GI random seed bug**: `run_threshold_sweep.sh` passes `--random_seed 111` but names subdir `t_random11_seed42`. All `outputs_thres/t_random11_seed42/` data was actually generated with seed 111. +CG sister script is correct.

2. **Liberal vs conservative prompt** in baseline: the GPT-4o "0.2287" paper number uses the liberal annotation prompt in `_api.py`. Changing to conservative (`run_eval_yesno.py`) drops it to 0.0946 — a bigger swing than any graph augmentation. Be explicit when comparing conditions.
