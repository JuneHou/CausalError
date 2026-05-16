# TRAIL Evaluation Pipeline — Reference

> One-stop reference for the LLM-as-judge eval pipeline (`benchmarking/eval/`). Stage-3 of the project: takes the artifacts from `CAUSAL_PIPELINE_REFERENCE.md` (Stages 1-2) and produces the metrics in `paper/tables/`.

**Compiled**: 2026-05-16 by 6 parallel sonnet agents (A=baseline, B=+CG, C=+GI, D=+SI, E=sweep, F=scoring).

---

## Pipeline overview

```
                      ┌─────────────────────────────────────────┐
                      │   Stage 1-2 artifacts                   │
                      │  • suppes_graph.json (M2)               │
                      │  • effect_edges.json (M8d)              │
                      │  see CAUSAL_PIPELINE_REFERENCE.md       │
                      └────────────────────┬────────────────────┘
                                           │ loaded once via
                                           │ load_graph_edges()
                                           ▼
       ┌───────────────────────────────────────────────────────────────────┐
       │                       Stage 3: Eval                                │
       │                                                                    │
       │   Module A (baseline)     Module B (+CG)        Module C (+GI)    │
       │   one prompt, no graph    one-pass + graph      two-pass dynamic  │
       │                                  │                     │           │
       │                                  └──── Module D (+SI) ─┤           │
       │                                       span-index flag             │
       │                                       (composable with B and C)   │
       │                                                                    │
       │   Module E driver runs Modules B/C across τ ∈ {random,0.35,0.25,0.20}
       │           outputs land in outputs_thres[/_cg]/t<τ>/                 │
       │                                  │                                  │
       │                                  ▼                                  │
       │                          Module F (scoring)                         │
       │                          calculate_scores.py                        │
       │                                  │                                  │
       └──────────────────────────────────┼──────────────────────────────────┘
                                          ▼
                              <pred_dir>-metrics.txt
                              (paper-table values)
```

## Architecture decision matrix

| | Baseline (A) | +CG (B) | +GI (C) | +SI (D) |
|---|---|---|---|---|
| LLM calls per trace | 1 (+1 rubric) | 1 | 2 | (flag) |
| Graph in prompt? | no | yes (full block) | yes (Pass-1 full; Pass-2 filtered subgraph) | no impact on graph |
| Span-id prediction? | no | optional via +SI | optional via +SI | yes |
| Loc/Joint metrics computable? | no | with +SI | with +SI | required |
| Combinable | — | +SI | +SI | adds to B or C |

The paper's headline method is **+GI+SI** (Module C with `--span_index`).

---

# Module A: Baseline Eval (no graph)

**Purpose**: LLM-as-judge over the full TRAIL error taxonomy with no causal guidance — produces per-trace error predictions + scalar quality scores.

**Code**:
- `benchmarking/eval/run_eval.py` — litellm (OpenAI/Gemini/Anthropic)
- `benchmarking/eval/run_eval_vllm.py` — vLLM (local HF models)
- (no DeepInfra or ARC baseline runners — those exist only for graph variants)

### Inputs
- Per-trace JSON files at `data/<split>/*.json`; trace passed as-is in the prompt
- Splits: `GAIA` (117), `SWE Bench` (31), `GAIA_dedup`, `SWE_Bench_dedup`
- Ground truth is used only by Module F at scoring time, not at inference time

### Output
- Dir name: `outputs_<model_tag>-<split>[_span_index]/` (model_tag = model with `/` → `-`)
- Per-trace file: `<trace_id>.json` containing the **raw LLM response string** (no parsing done at inference)
- Expected response schema (LLM is asked to produce, no enforcement):
  ```json
  {
    "errors": [{"category", "location": "<span_id hex>", "evidence", "description", "impact": "HIGH|MEDIUM|LOW"}],
    "scores": [{"reliability_score": 0-5, "security_score": 0-5,
                "instruction_adherence_score": 0-5, "plan_opt_score": 0-5,
                "overall": 0-5, plus *_reasoning strings}]
  }
  ```

### Key CLI args
| Arg | `run_eval.py` default | `run_eval_vllm.py` default |
|---|---|---|
| `--model` | `openai/gpt-4o` | `Tongyi-Zhiwen/QwenLong-L1-32B` |
| `--data_dir` | `data` | `data` |
| `--output_dir` | `outputs/zero_shot` | `outputs/zero_shot` |
| `--split` | `GAIA` | `GAIA` |
| `--max_workers` | `1` | — |
| `--tensor_parallel_size` | — | `4` |
| `--max_model_len` | — | `131072` |
| `--gpu_memory_utilization` | — | `0.75` |
| `--max_new_tokens` | — | `8000` |
| `--enforce_eager` | — | `True` (use `--no_enforce_eager` to disable) |
| `--span_index` | NOT supported | `False` (Module D) |

### Prompt structure
Single user message (no system message). Contents in order:
1. Header instructing model to follow taxonomy
2. Full 19-leaf taxonomy as ASCII box-drawing tree
3. Bullet instructions: be exhaustive, leaf categories only, output bare JSON, no markdown
4. JSON output template with inline comments
5. Worked example (one-error trace) + no-errors example
6. Optional span-index block (vLLM only, Module D)
7. `"The data to analyze is as follows:"` + raw trace JSON
8. Closing instructions; **Resource Abuse uses LAST instance, all other errors use FIRST instance**

For reasoning models (`o1`, `o3`, `o4`, `anthropic/*`, `gemini-2.5`), `run_eval.py` passes `reasoning_effort="high"` and omits temperature. Otherwise `temperature=0.0, top_p=1`.

### Output parsing
Done by **Module F**, not by the eval. Eval writes raw text; scorer applies `re.search(r"\{.*\}", ..., re.DOTALL)` to extract the outermost JSON, with iterative-tail-trimming fallback.

### Concurrency
- `run_eval.py`: `ThreadPoolExecutor` with `--max_workers`; 5-retry exponential backoff (60, 120, 240, 480, 960s) on `RateLimitError`. Context-overflow writes sentinel `"Context window exceeded. No output generated."` and continues.
- `run_eval_vllm.py`: single-process sequential; vLLM internal batching. Qwen3-30B-A3B gets special handling (DUAL_CHUNK_FLASH_ATTN backend, `VLLM_USE_V1=0`, chunked prefill, `max_num_seqs=1`). Mistral models without chat_template fall back to `<s>[INST]...[/INST]`.

### Gotchas
- **Idempotent**: both scripts skip files where `os.path.exists(output_file)`. Resumable.
- `run_eval.py` always appends `/{split}` to `data_dir`; cannot accept flat dir
- `litellm.drop_params=True` set globally (line 271)
- `security_score` Pearson correlation is consistently `nan` (zero-variance: models output 5/5 deterministically)

---

# Module B: +CG — One-Pass Causal Graph Guidance

**Purpose**: Single LLM call per trace; the prompt embeds a pre-loaded text block listing causal or correlated edges from the TRAIL graph. No span-index head, no two-pass dynamics.

**Code** (4 backends):
| File | Backend |
|---|---|
| `benchmarking/eval/run_eval_with_graph.py` | litellm (Gemini, GPT-4o, Anthropic) |
| `benchmarking/eval/run_eval_with_graph_vllm.py` | vLLM (open-source models) |
| `benchmarking/eval/run_eval_with_graph_api_deepinfra.py` | DeepInfra REST |
| `benchmarking/eval/run_eval_with_graph_api_arc.py` | ARC LLM API (gpt-oss-120b) |

**Note**: the API runners (`*_deepinfra`, `*_arc`) import `load_graph_edges`, `format_graph_guidance`, and `build_span_index` from `run_eval_graph_inject_vllm.py` (Module C's file) — that's where the union/random surface lives. They import `get_prompt` from `run_eval_with_graph_vllm.py`.

### Backend routing
| Model | Backend |
|---|---|
| `gemini/*`, `openai/gpt-4o`, `anthropic/*` | litellm |
| `Tongyi-Zhiwen/QwenLong-L1-32B`, `mistralai/Mistral-Small-*`, `Qwen3-30B-A3B` | vLLM |
| `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `google/gemma-3-27b-it`, `mistralai/*` | DeepInfra |
| `gpt-oss-120b` (no prefix) | ARC |

### Output
Dir: `outputs_<model_tag>-<split>-graph_<tag>[_span_index]/`. `<tag>` resolution order:
| Mode | Tag |
|---|---|
| `--random_edges` | `random<N>_seed<S>` |
| `--causal_only` | `causal_only` |
| `--corr_threshold <τ<1>` (API only) | `causal_corr<τ>` |
| `--edge_threshold <τ>` (litellm/vLLM) | `suppes_t<τ>` (litellm) or `graph_t<τ>` (vLLM) |

Per-trace files contain the raw LLM response (parsed at scoring time).

### Key CLI args
Graph mode: `--causal_only`, `--corr_threshold` (default 1.0; API only), `--edge_threshold` (default 0.20), `--random_edges`, `--random_seed` (42), `--random_n` (12).
Graph paths: `--causal_graph` (default `effect_edges.json` from `interventions_full_gaia_swe_merged/`), `--suppes_graph` (default `suppes_graph.json` from `trail_causal_outputs_full_gaia_swe_AIC/`).
Common: `--model`, `--data_dir` (`data`), `--output_dir` (`outputs/zero_shot` or `zero_shot2`), `--split`.
API: `--max_tokens` (8000, auto-bumped to 24000 for reasoning models), `--max_workers` (litellm 5), `--rpm` (DeepInfra 600 / ARC 30), `--rph`/`--rp3h` (ARC 1000/3000), `--max_retries` (5), `--limit_traces`, `--model_tag`, `--reasoning_effort` (DeepInfra; `"auto"` → `"low"` for gpt-oss-20b).
vLLM: `--tensor_parallel_size` (4), `--max_model_len` (131072), `--gpu_memory_utilization` (0.8), `--max_new_tokens` (8000), `--enforce_eager` (True).

### Prompt structure
Single user message:
```
[taxonomy tree]
[graph guidance block]   ← inserted after taxonomy, before instructions
- Based on the taxonomy above, analyze the LLM agent trace below...
[instructions + JSON template + 2 examples]
The data to analyze is as follows:
[optional span_index block (Module D)]
[trace JSON]
[closing format reminders]
```

`format_graph_guidance(edges, causal_only=...)` produces three variants:
- `--causal_only`: header `# Causal Error Patterns (intervention-validated)`, edges formatted as `[Source] → [Consequent]  (causal effect: X.XX)`
- corr/Suppes: header `# Correlated Error Patterns (observational, precedence-filtered)`, `(observational score: X.XX)`
- `--random_edges`: header `# Random Error Pattern Baseline (uncalibrated)`, no scores

### Output parsing
Deferred to Module F. The eval writes raw LLM text.

### Concurrency
- vLLM: sequential per-trace; pre-tokenizes prompt and skips if `tok_len ≥ max_model_len`
- litellm: `ThreadPoolExecutor`; 3 retries on `RateLimitError` (60s sleep); `ContextWindowExceededError` → skip trace
- DeepInfra/ARC: sequential with `RateLimiter` (sliding-window)

### Backend gotchas
- **DeepInfra**: 5-retry exponential backoff (1+2+4+8+16s, capped 60s). **Recent patch**: short-circuit when error string contains `"maximum context length"` or `"BadRequestError"` (DeepInfra wraps upstream 400s in 500). `--reasoning_effort auto` → `"low"` for `gpt-oss-20b` (default `"high"` truncates JSON tail). Auth: `DEEPINFRA_API_KEY` or `API_KEY`.
- **ARC**: 3-rule fairshare limiter (30/min, 1000/hr, 3000/3hr). Bare model name (`gpt-oss-120b`, no `openai/` prefix). No `reasoning_effort`. Auth: `ARC_LLM_API_KEY`. Base URL `https://llm-api.arc.vt.edu/api/v1/`.
- **vLLM**: For `Qwen3-30B-A3B`, set `VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN`, `VLLM_USE_V1=0`, chunked_prefill. Mistral models require `fix_mistral_regex=True` on tokenizer load (works around regex bug in some Mistral-3.1/3.2 releases). Trace skipped with sentinel if `tok_len ≥ max_model_len`.
- **litellm**: reasoning models (`o1`/`o3`/`o4`/`anthropic/*`/`gemini-2.5`/`gpt-oss`) → `max_completion_tokens=8000, reasoning_effort="high"`.

---

# Module C: +GI — Two-Pass Dynamic Graph Injection

**Purpose**: Paper's headline architecture. Pass-1 detects categories with the full graph block visible; propagation filters edges to those reachable from detected categories; Pass-2 is a targeted re-call constrained to that subgraph; results are merged.

**Code** (4 backends, canonical implementation is the vLLM file):
| File | Backend |
|---|---|
| `benchmarking/eval/run_eval_graph_inject.py` | litellm (Gemini canonical) |
| `benchmarking/eval/run_eval_graph_inject_vllm.py` | **canonical** — defines `load_graph_edges`, `build_span_index`, `propagate_confidence`, `validate_locations`, `parse_json_output` |
| `benchmarking/eval/run_eval_graph_inject_api_deepinfra.py` | DeepInfra |
| `benchmarking/eval/run_eval_graph_inject_api_arc.py` | ARC (gpt-oss-120b) |

The DeepInfra and ARC variants import all graph logic + prompt builders + parsers from the vLLM file; they add only their own `call_chat` wrapper and `RateLimiter`.

### Per-trace flow
1. **Pass-1**: full trace + global graph block → `{"errors": [...], "scores": [...]}`
2. **Propagate**: extract detected categories from Pass-1 errors; call `propagate_confidence(detected_cats, edges, propagation_threshold)`
3. **Pass-2**: if `filtered_edges` non-empty, re-call with Pass-1 summary + filtered subgraph; returns only new errors (no scores)
4. **Merge**: drop any Pass-2 error whose category is already in Pass-1 (category-level dedup, NOT instance-level); append survivors; scores come from Pass-1 only

### The propagation rule
`propagate_confidence` (line 324 of `run_eval_graph_inject_vllm.py`):
```python
boosted = {}
for src, dst, w in edges:
    if src in detected_set:
        boosted[dst] = boosted.get(dst, 0.0) + w
return [
    (src, dst, w) for src, dst, w in edges
    if src in detected_set and dst not in detected_set
    and boosted.get(dst, 0.0) > threshold
]
```
Edge survives Pass-2 iff: source detected, target NOT detected (no point re-finding what Pass-1 found), accumulated incoming weight from detected sources exceeds `propagation_threshold` (default `0.10`).

### Output
Dir: `outputs_<model_tag>-<split>-graph_inject_<graph_tag>[_span_index]/`. `<graph_tag>` patterns: `causal_only`, `causal_corr<τ>`, `random<N>_seed<S>`, `suppes_t<thr>`.

Two files per trace:
- **`<trace_id>.json`**: merged Pass-1 + Pass-2 errors + Pass-1 scores
- **`_meta_<trace_id>.json`**: `{trace_id, pass1_detected, pass2_triggered, pass2_filtered_edges, pass2_new_errors, pass2_parse_failed, p1_dropped, p2_dropped}` (plus `graph`, `api`, `model` in deepinfra/arc variants)

On Pass-2 parse failure, raw response goes to `_debug_p2_<trace_id>.txt`. On context overflow, output contains `{"errors": [], "scores": [], "_error": "context_overflow"}`.

### Key CLI args (additional to +CG)
- `--propagation_threshold` (`0.10`) — minimum accumulated weight for target to trigger Pass-2
- `--validate_span_id` (`True`) — drop errors whose location isn't a known span hex (`--no_validate_span_id` to disable)

### Pass-2 prompt structure
Pass-2 template (`GRAPH_INJECT_TEMPLATE`, lines 176-204 of vllm file) differs from Pass-1:
- `TAXONOMY_BLOCK` (same)
- `"TARGETED SECOND-PASS analysis"` declaration
- **PASS 1 RESULTS** block: bulleted list of categories + locations already found
- **CAUSAL GRAPH CONTEXT** block: only `filtered_edges` (`"src" → "dst" [weight: X.XX]`)
- Output only NEW errors, no scores; return `{"errors": [...]}` only
- litellm: Pass-1 uses `reasoning_effort="high"`; Pass-2 uses `use_reasoning=False` (line 676)

### Output parsing
`parse_json_output` (lines 346-362 of vllm file): strips `<thinking>...</thinking>` and `<think>...</think>` (for Gemini 2.5 / Claude extended thinking), strips markdown fences, attempts `json.loads`, falls back to regex `{...}` scan.

Pass-2 dedup is **category-level**: if Pass-2 finds the same category at a different span as Pass-1, it's dropped. Span-id validation is independent per-pass and runs before dedup.

### Edge cases
- Pass-1 detects zero categories → Pass-2 skipped (`if detected_cats and edges:` guard fails); meta records `pass2_triggered: false`
- Propagation yields zero edges → log `no relevant edges for {trace_id} — skipping Pass-2`; only Pass-1 errors kept
- vLLM pre-flight: `tok_len + 2048 > max_model_len` → context_overflow sentinel
- Pass-1 JSON parse failure → raw text written to output file, `p1_parse_failed: True` in meta
- Pass-2 JSON parse failure → `_debug_p2_<trace_id>.txt` saved, P1 results still written

### Concurrency
- litellm: `ThreadPoolExecutor` with `--max_workers` (default 1); 6-second `time.sleep` after each call (enforces free-tier 10 RPM)
- vLLM: sequential per-trace; vLLM internal batching
- DeepInfra/ARC: sequential with `RateLimiter`; both passes count toward limit

---

# Module D: +SI — Span Index Prefix

**Purpose**: Prepend a compact enumeration of agent-step span IDs (and immediate children) to the prompt, so the LLM populates `location` with verifiable hex span IDs. Required to make Loc and Joint metrics meaningful.

### What it does
`build_span_index` (defined in `run_eval_graph_inject_vllm.py:399`; duplicated verbatim in `run_eval_with_graph_vllm.py:42`, `run_eval_with_graph.py:54`, `run_eval_graph_inject.py:493`) — calls `parse_trace_to_step_level` (from `benchmarking/span_level_parser.py`) to extract agent-step spans (`CodeAgent.run`, `ToolCallingAgent.run`, `Step N` patterns at any depth). Records each step span's `span_id` + name, then walks one level deeper for immediate children (two-space indent). Deduplicated. Returns multi-line string, or `""` on parse failure.

### Resulting prompt block
```
Span index for this trace (use these exact span_id hex values for the location field):
  span_id "a1b2c3d4e5f60000"  (CodeAgent.run)
    span_id "7f3e9a1200000000"  (Step 1)
  span_id "b9c1234500000000"  (ToolCallingAgent.run)
    ...
```

### Where it lands in the prompt
- **+CG (Module B)** (`run_eval_with_graph_vllm.py:279`): after graph guidance block, before `{trace}`
- **+GI Pass-1** (`PASS1_PROMPT_TEMPLATE:171`): before `The data to analyze:`
- **+GI Pass-2** (`GRAPH_INJECT_TEMPLATE:191`): before `INSTRUCTIONS:`

Span index is computed once per trace and passed to both passes.

### Output naming
| Mode | Dir suffix |
|---|---|
| without `--span_index` | (none) |
| with `--span_index` | `_span_index` appended after the graph tag |

Examples: `outputs_<m>-<s>-graph_causal_only_span_index/`, `outputs_<m>-<s>-graph_inject_causal_corr0.35_span_index/`.

### Output schema impact
Same `{"errors": [...], "scores": [...]}` shape. The `location` field now holds verified hex span IDs. With `--validate_span_id True` (default for +GI), the runner calls `extract_span_ids` to build a full-tree `{span_id: span_name}` dict and `validate_locations` drops any predicted error whose location isn't a key (independently per pass; recorded as `p1_dropped` / `p2_dropped` in meta). Use `--no_validate_span_id` to disable.

### Gotchas
- `--validate_span_id` is a +GI flag; +CG runners have no equivalent (but the validation feature is independent of `--span_index` — it runs regardless when present)
- `build_span_index` shows only step spans + their **direct children**; deeper spans aren't enumerated. But `extract_span_ids` for validation walks the full tree, so a deep-nested valid span_id won't be dropped even though it wasn't in the index.
- **MAST runners** don't expose `--span_index` (MAST has no location prediction)

---

# Module E: Threshold Sweep Drivers

**Purpose**: One driver, one model, runs every sweep point sequentially. Dispatches to the right inner script per backend; scores everything at the end.

**Code**:
- `benchmarking/eval/run_threshold_sweep.sh` — +GI sweep
- `benchmarking/eval/run_threshold_sweep_cg.sh` — +CG sweep (sister)

### Usage
```
bash run_threshold_sweep[_cg].sh <model> <split> [gpus] [output_dir] [backend]
```

`[gpus]` default `0,1`; comma-count drives `--tensor_parallel_size` auto-detection. `[output_dir]` defaults to `outputs_thres` (+GI) or `outputs_thres_cg` (+CG). Backend explicit override: `vllm | litellm | deepinfra | arc`; otherwise inferred from model name.

### Backend inference (case statement)
| Model pattern | Backend |
|---|---|
| `gemini/*`, `openai/gpt-4*`, `openai/o*` | `litellm` |
| `openai/gpt-oss-*`, `google/*` | `deepinfra` (+GI); +CG also adds `mistralai/*` |
| any other `<ns>/<name>` (contains `/`) | `vllm` |
| bare name (no `/`) | `arc` |

### Inner-script dispatch
| Backend | +GI sweep | +CG sweep |
|---|---|---|
| vllm | `run_eval_graph_inject_vllm.py` | `run_eval_with_graph_vllm.py` |
| litellm | `run_eval_graph_inject.py` | `run_eval_with_graph.py` |
| deepinfra | `run_eval_graph_inject_api_deepinfra.py` | `run_eval_with_graph_api_deepinfra.py` |
| arc | `run_eval_graph_inject_api_arc.py` | `run_eval_with_graph_api_arc.py` |

### Sweep points
Default `THRESHOLDS="random 0.35 0.25 0.20"`; override via env var. Recognized values:
- `random` → `--random_edges` (inner-script defaults: `random_n=12 random_seed=42`); subdir `t_random12_seed42/`
- `0.35`, `0.25`, `0.20` → `--corr_threshold τ` (API) or `--edge_threshold τ` (+CG vllm/litellm); subdir `t<τ>/`
- `causal_only` (recognized but not in default list); subdir `t_causal_only/`

### Per-model `max_model_len` table (vLLM only)
| Model | `max_model_len` |
|---|---|
| `Tongyi-Zhiwen/QwenLong-L1-32B*` | 128000 |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503*` | 108000 |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503*` | 108000 (**+CG only**) |
| `openai/gemma-3-27b-it*`, `google/gemma-3-27b-it*` | 108000 |
| `openai/gpt-oss-20b*`, `openai/gpt-oss-120b*` | 108000 |
| else | inner-script default (131072) |

### Output dir layout
```
outputs_thres[/_cg]/
  t_random12_seed42/
    outputs_<model_tag>-<split>-graph_inject_random12_seed42_span_index/  # +GI
    outputs_<model_tag>-<split>-graph_random12_seed42/                    # +CG API
  t0.35/
    outputs_<model_tag>-<split>-graph_inject_causal_corr0.35_span_index/  # +GI API
    outputs_<model_tag>-<split>-graph_causal_corr0.35/                    # +CG API
    outputs_<model_tag>-<split>-graph_t0.35/                              # +CG vllm/litellm (edge_threshold path)
  t0.25/ ...
  t0.20/ ...
  _sweep_logs/
    <model_tag>-<split>-t<τ>.log        # +GI
    <model_tag>-<split>-cg-t<τ>.log     # +CG
```

### Differences (+GI vs +CG drivers)
1. Inner-script family
2. Default output dir root
3. Per-cell subdir naming
4. **Edge selection flag**: +GI always passes `--corr_threshold`; +CG passes `--corr_threshold` for API backends but `--edge_threshold` for vllm/litellm (different edge set!)
5. `--span_index` passthrough: +GI hardcodes it on every call; +CG gates it on `SPAN_INDEX=1` env var (default off)
6. +CG routes `mistralai/*` to DeepInfra (not present in +GI)
7. +CG adds the Mistral 3.2 max_model_len row

### Scoring loop
After all sweep points, iterates `THRESHOLDS` again, resolves each to its subdir, calls `python eval/calculate_scores.py --results_dir <outdir>/<subdir>` (Module F).

### Gotchas
- **`random` skipped for vllm/litellm in +CG**: `SUPPORTS_CORR_AND_RANDOM=0` for those backends. Driver emits `WARN:` and `continue`s. Log file created but contains only the warning.
- `--random_n` / `--random_seed` not passed explicitly — relies on inner-script defaults (12, 42)
- `causal_only` not in default sweep list; add manually if needed
- No `--enable_thinking` flag in the bash drivers themselves — reasoning models handle thinking blocks inside their inner scripts
- Legacy `<method>` positional (former `cg|gi`) rejected with explicit error in `_cg.sh`

---

# Module F: Scoring & Metrics

**Purpose**: Turn per-trace prediction JSONs into the three headline metrics (W-F1, Loc, Joint), per-category Precision/Recall/F1, and Pearson score correlations.

**Code**: `benchmarking/eval/calculate_scores.py` — single scorer. There is **no** `calculate_scores_yesno.py` in TRAIL (that's the MAST scorer).

### Usage
```bash
# MUST run from benchmarking/ — ground-truth path is relative
python eval/calculate_scores.py --results_dir <path>
```
The script globs `<results_dir>/*` and scores each subdir in turn. Only one CLI arg (`--results_dir`, default `outputs/zero_shot`). Ground-truth path is auto-derived (line 389): basename inspection for `swe_bench`/`gaia` tokens → `processed_annotations_{swe_bench|gaia}/`.

### Inputs
- Ground truth: `benchmarking/processed_annotations_{gaia,swe_bench}/` (filename = `<trace_id>.json`; same schema as predictions)
- Predictions: `<results_dir>/<subdir>/<trace_id>.json` — read as raw text; JSON extracted via `extract_json_from_text` (regex `{.*}` with `re.DOTALL`, plus iterative tail-trim if `json.loads` fails)
- Expected schema: `{"errors": [{"category", "location": "<hex>", ...}], "scores": [...]}`

### Output
`<results_dir>/<subdir>-metrics.txt` (sibling of subdir, not inside):
```
Weighted F1: 0.3415
Average Location Accuracy: 0.2571
Average Location-Category Joint Accuracy: 0.1227

Score Correlations (Pearson r):
--------------------------------------------------------------------------------
Score Type                Correlation     p-value         N
--------------------------------------------------------------------------------
reliability                  ...
security                     nan
instruction_adherence        ...
...

Per-Category Statistics:
--------------------------------------------------------------------------------
Category                  Precision   Recall   F1       Support
...
```

### Metric definitions
All three are **per-trace, then macro-averaged across files_processed** (matched + substituted-empty all count).

- **Weighted F1**: per-trace, build a 21-dim binary vector (set membership over 21 hardcoded categories at lines 115-122). Stack into 2D array. Call `sklearn.metrics.f1_score(..., average='weighted', zero_division=0)`. `'weighted'` = each label's F1 weighted by support across all traces.
- **Location Accuracy**: per-trace, `|GT_locations ∩ pred_locations| / |GT_locations|`. Set intersection. Returns 0 if `GT_locations` empty. Averaged.
- **Joint Accuracy**: per-trace, zip GT errors into `(location, category)` tuples; same for predictions; `|intersect| / |GT|`. Set semantics. Averaged.

### Edge cases
- **Missing prediction file** → substituted as `{"errors": [], "scores": []}`; counted toward denominator; warning printed
- **Parse failure** → same empty-prediction substitution; `files_processed` incremented
- **Working-directory sensitivity**: relative path `processed_annotations_{split}` means scoring from anywhere other than `benchmarking/` yields silent zero metrics. The `glob.glob` returns `[]`; no error raised.
- `security` Pearson correlation is `nan` because predictions are constant (zero variance); `pearsonr` returns nan, propagates through
- Categories with zero support: P/R/F1 zero-divided to 0; only rows with `support > 0` are written (zero-support categories computed but suppressed)

### Cross-module
Called by Module E's scoring loop at end of each sweep iteration. Direct source of paper-table values.

---

# Cross-cutting notes

## Quick file → module map

| File | Module |
|---|---|
| `run_eval.py`, `run_eval_vllm.py` | A |
| `run_eval_with_graph*.py` (4 files) | B |
| `run_eval_graph_inject*.py` (4 files) | C |
| `build_span_index` (lives in `run_eval_graph_inject_vllm.py`), `--span_index` flag | D |
| `run_threshold_sweep.sh`, `run_threshold_sweep_cg.sh` | E |
| `calculate_scores.py` | F |

## Where does each artifact end up?

```
benchmarking/outputs/zero_shot/         A (closed-source baseline + +CG litellm + +GI litellm)
benchmarking/outputs/zero_shot2/        A (open-source baseline + +CG API)
benchmarking/outputs_thres/             E (+GI sweep)
benchmarking/outputs_thres_cg/          E (+CG sweep)
```

## Architecture stacking summary

| Run | Module(s) | Output dir pattern |
|---|---|---|
| Baseline | A | `outputs_<m>-<s>/` |
| Baseline + SI | A + D | `outputs_<m>-<s>_span_index/` |
| +CG (causal_only) | B | `outputs_<m>-<s>-graph_causal_only/` |
| +CG + SI (τ=0.35) | B + D | `outputs_<m>-<s>-graph_causal_corr0.35_span_index/` |
| +GI (causal_only) | C | `outputs_<m>-<s>-graph_inject_causal_only/` |
| **+GI + SI (τ=0.35)** ← paper headline | C + D | `outputs_<m>-<s>-graph_inject_causal_corr0.35_span_index/` |

## Recently-applied patches worth remembering

1. **Context-overflow short-circuit** in retry loops of 4 API runners (DeepInfra + ARC × +CG + +GI): on `"maximum context length"` or `"BadRequestError"` in error string, raise immediately instead of burning ~31s of exponential backoff
2. **Mistral 3.2 entry** added to `_cg.sh` per-model `max_model_len` table
3. **Path bug**: the litellm +GI runner is `run_eval_graph_inject.py` (no `_api` suffix), often misremembered as `run_eval_with_graph_inject_api.py` (does not exist)

## Per-model backend pairing in published runs

| Model | Backend used | Notes |
|---|---|---|
| Gemini-2.5-Flash | litellm | |
| Gemini-2.5-Pro | litellm | |
| Mistral-Small-3.1-24B | vLLM | tp=4 for QwenLong; Mistral fine at default tp |
| Mistral-Small-3.2-24B | DeepInfra (+CG only) | newer tokenizer breaks vLLM |
| GPT-oss-120B | ARC | only place this model is provisioned |
| GPT-oss-20B | DeepInfra | `reasoning_effort='low'` auto-set |
| Gemma-3-27B-IT | vLLM or DeepInfra | |
| QwenLong-L1-32B | vLLM | needs `--tensor_parallel_size 4` for W&W variants |
