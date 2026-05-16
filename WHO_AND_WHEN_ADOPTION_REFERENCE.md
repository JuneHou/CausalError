# Who&When Adoption — Reference

> How the Who&When (Yin et al., arXiv:2505.00212, ICML 2025 Spotlight) prompts were ported to TRAIL's multi-label span-level error annotation task, and how causal graph guidance was layered on top.

**Compiled**: 2026-05-16 by 2 parallel sonnet agents (Part 1 = vanilla adoption, Part 2 = graph-guided).

**Companion docs**:
- `CAUSAL_PIPELINE_REFERENCE.md` — how `effect_edges.json` and `suppes_graph.json` are produced
- `EVAL_PIPELINE_REFERENCE.md` — the main TRAIL eval pipeline; W&W reuses many of its functions

---

# Part 1: Vanilla W&W Adoption (no graph)

**Purpose**: Port the W&W prompting strategies to TRAIL's multi-label setting.

## Original W&W vs TRAIL adaptation

The original W&W identifies *which agent* (Who) caused failure and *at which step* (When), in a multi-agent setting. It always outputs exactly **one** prediction (one agent + one step) and is conditioned on ground-truth (used as hint). Three localization strategies: W1 all-at-once, W2 step-by-step, W3 binary search.

TRAIL is **single-agent, multi-label** over a 19-leaf taxonomy (`TAXONOMY_LEAF_CATEGORIES` actually lists 20; the module docstring says 19). The mapping:
- "Who" → **What** (TRAIL error category)
- "When" → **Where** (`span_id` hex in the trace)

Per-variant adaptation (per `run_who_and_when_vllm.py` lines 23-34 and `plan.md`):
- **W1**: single-error free-text answer → multi-label JSON; consider all categories independently; ground-truth dropped (TRAIL is reference-free)
- **W2**: per-step Yes/No → per-step multi-label JSON; the "early-exit on first Yes" rule is removed (all spans are scanned, results aggregated); "avoid being overly critical" calibration sentence retained verbatim
- **W3**: bisection runs separately per label (19 independent runs); both halves can be positive at each call (`lower_half_present` AND `upper_half_present`); recurse into all positive halves. **Implemented but excluded from headline experiments** — the O(log N) efficiency argument doesn't survive multi-label adaptation.

## File layout

| File | Purpose |
|---|---|
| `baselines/who_and_when/run_who_and_when_vllm.py` | **The sole vanilla runner**. Covers W1, W2, W3 via `--variant`. vLLM with `bfloat16`. No DeepInfra or ARC variants at this level. |
| `baselines/who_and_when/old_run_who_and_when_vllm.py` | Archived prior version; not used |
| `baselines/who_and_when/plan.md` | Design doc: adaptation philosophy, per-variant change list, cost per trace, output schema, evaluation protocol |
| `baselines/who_and_when/_normalize_qwenlong_w2_categories.py` | Post-hoc normalizer for QwenLong W2 outputs that emit hierarchical paths (`"Information Processing -> Tool Output Misinterpretation"`); splits on `-> / > / / / :: / .` separators, takes last token, rewrites to sibling `_leafnorm/` dir. Used only for QwenLong-L1-32B |

## W1 prompt structure (holistic)

`W1_PROMPT_TEMPLATE` at lines 143-177. Single user message with four blocks:
1. **Role + task framing**: `"You are an AI assistant tasked with analyzing... The problem is: {task_description}"`
2. **Taxonomy block** (`TAXONOMY_BLOCK`, lines 93-127)
3. **Serialized trace**: all step-level spans as `--- Step N: {name} (span_id: "{hex}") ---` + up to 1200 chars of content (`format_trace_for_prompt`, line 488)
4. **Multi-label instruction + JSON schema**:
   ```json
   {"errors": [{"category": "...", "location": "<span_id>", "evidence": "...",
                "description": "...", "impact": "HIGH/MEDIUM/LOW"}],
    "scores": []}
   ```

After the error call, a separate `SCORES_PROMPT_TEMPLATE` call generates rubric scores (5 fields). Final output merges both.

**Span-id validation** at line 624: predicted errors whose `location` isn't in the trace's known span_id set are silently dropped.

## W2 prompt structure (step-by-step)

`W2_STEP_PROMPT_TEMPLATE` at lines 179-205. **One LLM call per step span.** For span at position `i`:
1. Role + task framing
2. Taxonomy block (same as W1)
3. **Cumulative prefix**: serialized text of all spans from step 1 through step `i`, grown incrementally in the loop (lines 679-741)
4. **Current step identification**: `"The most recent step (Step {step_num}, span_id "{span_id}") was: {step_name}."`
5. **Per-step instruction**: errors in current step only; calibration sentence retained

Per-step output:
```json
{"step_id": N, "span_id": "...", "has_error": true/false,
 "errors": [{"category": "...", "evidence": "...", "description": "...", "impact": "..."}]}
```

**Aggregation**: skip `has_error=false` and JSON-parse failures. Dedup `(category, span_id)` pairs via `seen_pairs` (line 719). After all spans, one additional rubric scores call.

**Cost**: N+1 calls per trace. A typical GAIA trace (~8 step spans) makes ~9 calls. A 50-step trace would be ~51 calls. Context grows with each call; **context-overflow guard at line 698** breaks the loop early if `tok_len + per_step_budget > max_model_len`; meta records `"error": "context_overflow_at_step_N"`.

## Output

**Directory naming** (line 938):
```
outputs_{model_tag}-{split}-who_and_when_{variant}/
```
`model_tag = args.model.replace("/", "-")`.

**Per-trace files** (two each):
- `{trace_id}.json` — standard TRAIL prediction schema (errors + scores) — readable by `calculate_scores.py`
- `_meta_{trace_id}.json` — diagnostics (W1: `{tok_len, n_raw_errors, scores_tok_len, ...}`; W2: `{calls, error, scores_tok_len, ...}`)

## Key CLI args

| Arg | Default |
|---|---|
| `--model` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `--split` | `GAIA_dedup` |
| `--variant` | `w1` (or `w2` / `w3`) |
| `--data_dir` | `{BENCH_DIR}/data` |
| `--output_dir` | `/data/wang/junh/githubs/trail-benchmark/baselines/outputs` |
| `--tensor_parallel_size` | `2` |
| `--max_model_len` | `131072` |
| `--gpu_memory_utilization` | `0.8` |
| `--max_new_tokens` | `8000` (auto-bumped to 24000 for reasoning models matching `(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)`) |
| `--enforce_eager` | `True` (`--no_enforce_eager` to disable) |

**Not present**: `--rpm`, `--causal_only`, `--corr_threshold`, `--span_index`, `--random_edges`. Those exist only in the `causal/` subdir (Part 2).

## Concurrency

vLLM batches generation for W1 (one call per trace) and W3 (one call per bisect interval). W2 calls `llm.generate([prompt], sp)` one span at a time — **strictly sequential within each trace**, no cross-span batching.

## Gotchas

- **Mistral tokenizer regex bug** (line 955-959): `AutoTokenizer.from_pretrained` passes `fix_mistral_regex=True` when model name contains `"Mistral"`/`"mistral"` — works around a tokenizer bug in some Mistral-3.1/3.2 releases (note: 3.2's new MistralCommonTokenizer rejects this kwarg, so 3.2 must be routed via DeepInfra).
- **No `--output_dir` `zero_shot2` sub-path**: the module docstring (line 68) mentions it, but the actual `out_dir` is written directly under `args.output_dir`.

## Scoring

Both W1 and W2 write standard TRAIL JSON and are scored with `benchmarking/eval/calculate_scores.py` (Module F in `EVAL_PIPELINE_REFERENCE.md`) **unchanged**. There is no separate `calculate_scores_yesno.py` in TRAIL — the "yesno" framing in the docstring refers to W2's per-step binary judgment in the prompt, not to a separate scorer.

## Compared to TRAIL main eval

| Dimension | W1 | W2 | TRAIL main eval (baseline) |
|---|---|---|---|
| Calls per trace | 2 (error + rubric) | N+1 | 1 |
| Localization strategy | Holistic | Sequential per-span | Holistic |
| Context shape | Full trace | Growing prefix | Full trace |
| Multi-label | ✓ | ✓ | ✓ |
| Graph guidance | no | no | no (baseline); yes (+GI/+CG) |
| Calibration sentence | — | retained | — |
| Has W3 analog | — | — | no |

W2's sequential decoding with cumulative prefix has **no analog in the main eval pipeline**.

---

# Part 2: W&W + Causal Graph Guidance

**Purpose**: Add the TRAIL causal graph as prompt context to W&W. Test whether prior knowledge about which error categories cause which others improves the W&W-style annotators' precision.

## Pairing decision: W1+GI vs W2+CG

The W&W table reports two graph-guided variants, each a deliberate pairing of W&W prompt style × graph-injection strategy:

- **W1 (holistic) + GI (two-pass) + SI**: Pass-1 runs W1 or W2 with the whole graph block visible; detected categories drive `propagate_confidence`; Pass-2 is a **single trace-level** targeted call (W1-style regardless of whether W1 or W2 was used in Pass-1), returning only new errors. `+SI` = `--span_index` (W2 ignores it by design — see below).
- **W2 (step-by-step) + CG (one-pass) + SI**: The graph guidance block is embedded directly into **every per-step W2 prompt**, no second pass.

**Why this pairing?** Adding a second pass to W2 would make cost O(2N) — a typical 8-step trace would jump from ~9 calls to ~17. A trace-level Pass-2 preserves the N+1 cost profile. From `run_who_and_when_graph_inject_vllm.py` lines 14-18:

> "Two-pass × per-span on W2 doubles the W2 call cost (2N calls/trace), which is already 9× a single-pass baseline. A trace-level Pass 2 preserves the N+1 cost profile and matches the structural choice documented in `paper/baseline_who_and_when.tex`."

## Code location

All under `baselines/who_and_when/causal/`. Six active scripts + five `old_*` backups (pre-patch state, do not use directly).

**Backend routing matrix**:

| Variant | vLLM | DeepInfra API | ARC API |
|---|---|---|---|
| W&W + GI (two-pass) | `run_who_and_when_graph_inject_vllm.py` | `run_who_and_when_graph_inject_api_deepinfra.py` | `run_who_and_when_graph_inject_api_arc.py` |
| W&W + CG (one-pass) | `run_who_and_when_with_graph_vllm.py` | `run_who_and_when_with_graph_api_deepinfra.py` | `run_who_and_when_with_graph_api_arc.py` |

No litellm variant for either (Gemini W&W is omitted from the paper table; the prohibitive O(N) cost per trace makes Gemini W&W experiments too expensive).

## Imports from the main eval

The import split is a deliberate post-patch design. Each group of three files manages it consistently:

**+GI runners** (e.g., `run_who_and_when_graph_inject_vllm.py:74-84`):
- `load_graph_edges`, `build_span_index`, `format_graph_guidance`, `propagate_confidence`, `validate_locations`, `DEFAULT_CAUSAL_GRAPH`, `DEFAULT_SUPPES_GRAPH` — all from `run_eval_graph_inject_vllm` (Module C)
- W1/W2 templates + `_build_pass2_prompt` — from `run_who_and_when_graph_inject_vllm` (API runners re-import the vLLM sibling's templates)
- Base W&W utilities (`TAXONOMY_BLOCK`, `apply_chat_template`, `extract_span_ids`, etc.) — from `run_who_and_when_vllm`

**+CG runners** (e.g., `run_who_and_when_with_graph_vllm.py:71-81`):
- `format_graph_guidance`, `DEFAULT_CAUSAL_GRAPH`, `DEFAULT_SUPPES_GRAPH` — from `run_eval_with_graph_vllm` (Module B)
- **`load_graph_edges`, `build_span_index` — from `run_eval_graph_inject_vllm`** (Module C). This is the **post-patch invariant**: all six W&W graph scripts load edges from the +GI eval's `load_graph_edges`, which has the full `--corr_threshold` / `--random_edges` signature.
- W1/W2 templates — from `run_who_and_when_with_graph_vllm` (API runners re-import the vLLM sibling's templates)

**Pre-patch state** (preserved in `old_*` files): all three +CG runners imported `load_graph_edges` from `run_eval_with_graph_vllm`, which used the legacy two-argument `(causal_only, threshold)` signature lacking `--corr_threshold` and `--random_edges` support. Calling those flags raised `TypeError: load_graph_edges() got an unexpected keyword argument 'corr_threshold'`.

`format_graph_guidance` remains in `run_eval_with_graph_vllm` for +CG scripts (it formats the static one-pass block), while +GI scripts use the version from `run_eval_graph_inject_vllm` which also handles `random_edges` formatting. Same function name, two slightly different implementations.

## Prompt templates

**W1+GI**: Pass-1 template = `W1_PASS1_TEMPLATE` in `run_who_and_when_graph_inject_vllm.py:119-158`. Slots: `{taxonomy_block}`, `{graph_guidance_block}`, `{span_index_block}`, `{task_description}`, `{trace}`. Graph block placed between taxonomy and span-index/instructions. Adds **leaf-node enforcement** ("The category field must be a FINAL LEAF subcategory") and the **Resource-Abuse last-instance rule** (absent from plain W1).

Pass-2 reuses `GRAPH_INJECT_TEMPLATE` imported from `run_eval_graph_inject_vllm` via `_build_pass2_prompt` (lines 87-109), which formats the Pass-1 summary and filtered subgraph.

**W2+GI**: Pass-1 step template = `W2_PASS1_STEP_TEMPLATE` in `run_who_and_when_graph_inject_vllm.py:160-189`. Same `{graph_guidance_block}` slot between taxonomy and cumulative step history. Also adds the leaf-node constraint. Pass-2 is still a **single trace-level** `_build_pass2_prompt` call — no per-step Pass-2.

**W1+CG** and **W2+CG**: Templates in `run_who_and_when_with_graph_vllm.py:93-158`. Same `{graph_guidance_block}` slot position. Omit the Resource-Abuse last-instance rule (these templates predate that addition in the +GI side). W2 receives an empty `{span_index_block}` by design (line 421):
```python
span_index_text = build_span_index(trace_str) if (args.span_index and args.variant == "w1") else ""
```

## Key CLI args (all 6 runners, post-patch)

| Arg | Default | Notes |
|---|---|---|
| `--variant {w1,w2}` | `w1` (vLLM); required (API) | |
| `--split` | `GAIA_dedup` | |
| `--causal_only` | `False` | Use only intervention-validated edges |
| `--corr_threshold` | `1.0` | Causal-union threshold (geomean `sqrt(precedence × pr_delta) ≥ τ`); ignored if `--causal_only` |
| `--edge_threshold` | `0.20` | Pure-Suppes threshold (no causal union); used only when neither `--causal_only` nor `--corr_threshold<1` |
| `--random_edges` | `False` | Null-graph control |
| `--random_seed` | `42` | |
| `--random_n` | `12` | |
| `--span_index` | `False` | W1 only; W2 ignores by design |
| `--causal_graph` | `None` (uses `DEFAULT_CAUSAL_GRAPH`) | Override `effect_edges.json` path |
| `--suppes_graph` | `None` (uses `DEFAULT_SUPPES_GRAPH`) | Override `suppes_graph.json` path |
| `--propagation_threshold` | `0.10` | **+GI only**: edge weight cutoff for Pass-2 subgraph |
| `--validate_span_id` | `True` | **+GI only**: drop Pass-2 errors with invalid location |
| `--max_model_len` | `131072` (vLLM) | |
| `--tensor_parallel_size` | `2` (vLLM) | **Override to 4 for QwenLong-L1-32B** |
| `--max_tokens` / `--max_new_tokens` | `8000` (auto-bumped to 24000 for reasoning models) | |
| `--rpm` | `600` (DeepInfra), `30` (ARC) | |
| `--rph` / `--rp3h` | `1000` / `3000` (ARC only) | 3-rule fairshare |

**Post-patch invariant**: `--corr_threshold` and `--random_edges` (+ `--random_seed`, `--random_n`) are present in all six scripts.

## Output directory naming

**+GI runners** (`run_who_and_when_graph_inject_*.py`):
```
outputs_{model_tag}-{split}-who_and_when_{w1|w2}_graph_inject_{graph_tag}[_span_index]/
```

**+CG runners** (`run_who_and_when_with_graph_*.py`):
```
outputs_{model_tag}-{split}-who_and_when_{w1|w2}_graph_{graph_tag}[_span_index]/
```

`graph_tag` values:
| Mode | +GI tag | +CG tag |
|---|---|---|
| `--causal_only` | `causal_only` | `causal_only` |
| `--corr_threshold τ` | `causal_corr<τ>` | `causal_corr<τ>` |
| pure-Suppes | `suppes_t<thr>` | `t<thr>` (no `suppes_` prefix) |
| `--random_edges` | `random<N>_seed<S>` | `random<N>_seed<S>` |

Each trace writes two files: `{trace_id}.json` (errors + scores) and `_meta_{trace_id}.json` (provenance). Resumable: existing output files are skipped.

## Recent patches applied (three changes)

1. **`load_graph_edges` import switched** in all three +CG runners (vLLM, DeepInfra, ARC): from `run_eval_with_graph_vllm` (legacy 2-arg) → `run_eval_graph_inject_vllm` (full signature). Old behavior preserved in `old_run_who_and_when_with_graph_{vllm,api_deepinfra,api_arc}.py`.

2. **`--corr_threshold`, `--random_edges`, `--random_seed`, `--random_n`** added to all three +CG CLI parsers to match the +GI runners.

3. **Context-overflow short-circuit** added to all four API retry loops (DeepInfra + ARC, both +GI and +CG): in `call_chat`, if exception string contains `"maximum context length"` or `"BadRequestError"`, raise immediately instead of sleeping through ~31s of exponential backoff. Pre-patch state in `old_run_who_and_when_graph_inject_api_{deepinfra,arc}.py`.

## Backend-specific gotchas

**vLLM (+GI or +CG)**:
- Large traces can overflow `--max_model_len 131072` even at `--tensor_parallel_size 2`. **For QwenLong-L1-32B, pass `--tensor_parallel_size 4`** (math: 32B params × 2 bytes = 64GB → 16GB/GPU at tp=4 → ~17GB free per GPU for KV cache vs ~1.75GB at tp=2).
- W1 uses 8192-token headroom guard. W2 uses `per_step_budget = min(4096, max(512, max_new_tokens // 8))`.
- Reasoning models (regex `qwenlong|-l1-|gpt-oss|deepseek-r1|qwq`): `max_new_tokens` auto-bumped 8000 → 24000.

**DeepInfra API**:
- `gpt-oss-20b` with default `reasoning_effort='high'` burns full 24k budget on chain-of-thought, truncating JSON tail. `--reasoning_effort auto` (default) → `"low"` for `gpt-oss-20b`, `None` otherwise.
- 5-retry exponential backoff (1, 2, 4, 8, 16s, capped at 60s); context-length errors short-circuit immediately.
- Auth: `DEEPINFRA_API_KEY` or `API_KEY` env var.

**ARC API**:
- 3-rule sliding-window fairshare: `--rpm 30`, `--rph 1000`, `--rp3h 3000`.
- No `reasoning_effort` parameter.
- Same 5-retry backoff with context-overflow short-circuit.
- Default model: `gpt-oss-120b` (bare name, no `openai/` prefix).
- Auth: `ARC_LLM_API_KEY` or `API_KEY`. Key file: `/data/wang/junh/.cache/keys/arc_llm_api.sh`.

## Scoring

Outputs are scored with `benchmarking/eval/calculate_scores.py` (Module F in `EVAL_PIPELINE_REFERENCE.md`) — same scorer as the main eval. Paper table `paper/tables/who_and_when_results.tex` draws on these outputs for 5 open-source models × 2 splits × 2 graph-guided variants (W1+GI+SI, W2+CG+SI).

## Quick-reference: what's W1+GI+SI vs W2+CG+SI?

| Property | W1+GI+SI | W2+CG+SI |
|---|---|---|
| W&W prompt | W1 (holistic) | W2 (per-step) |
| Graph injection | two-pass dynamic (Pass-1 full, Pass-2 filtered) | one-pass static (graph block in every per-step prompt) |
| Span index | yes (W1 prompt) | yes (Pass-1 step prompts) — but actually no, W2 ignores `--span_index` by design |
| Calls per trace | 2 (Pass-1 + Pass-2 single trace-level) | N+1 (N step calls + 1 rubric scores) |
| File family | `run_who_and_when_graph_inject_*` | `run_who_and_when_with_graph_*` |

**Note** on W2+CG+SI naming: although the paper table column is labeled `W2+CG+SI`, the actual W2 runners pass `span_index_text = ""` when `variant=="w2"`. The "+SI" in the column name refers to the **span-id validation** that runs regardless of `--span_index` (errors with invalid `location` hex are dropped), not to the span-index prompt block being inserted. This is consistent with W2's per-step prompts already enumerating span IDs in the step headers, making the block redundant.

---

# Cross-references

- `CAUSAL_PIPELINE_REFERENCE.md` — provenance of `effect_edges.json` and `suppes_graph.json`; the `load_graph_edges` function signature
- `EVAL_PIPELINE_REFERENCE.md` — Module B (+CG) and Module C (+GI) define `format_graph_guidance` and `load_graph_edges` respectively; W&W reuses both
- `baselines/who_and_when/plan.md` — design rationale for the W1/W2/W3 adaptation choices
- `baselines/who_and_when/_normalize_qwenlong_w2_categories.py` — QwenLong-specific post-processor for hierarchical category outputs
- `paper/tables/who_and_when_results.tex` — the table that consumes these outputs (currently at causal_only; would need 20 reruns to pivot to τ=0.35; see `TODO_0.35.md` section 3)
