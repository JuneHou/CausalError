# MAST Who&When Adoption — Reference

> How Who&When (Yin et al., arXiv:2505.00212, ICML 2025) was ported to MAST's multi-label-per-trace yes/no setting (13 leaves, no location prediction), and how causal graph guidance was layered on top.

**Module path**: `/data/wang/junh/githubs/MAST/baselines/who&when/` (lives in the sibling MAST repo; doc kept here because MAST blocked writes during generation)

**Compiled**: 2026-05-16 by 2 parallel sonnet agents (Part 1 vanilla, Part 2 graph-guided).

**Companion docs**:
- `MAST_CAUSAL_PIPELINE_REFERENCE.md` — provenance of `effect_edges.json` + `suppes_graph.json`
- `MAST_EVAL_PIPELINE_REFERENCE.md` — main MAST eval; W&W reuses some functions and the same scorer
- `WHO_AND_WHEN_ADOPTION_REFERENCE.md` — TRAIL's analog

---

# Part 1: Vanilla W&W Adoption (no graph)

## Original W&W → MAST adaptation

| Dimension | Original W&W | MAST adaptation |
|---|---|---|
| Setting | multi-agent (who? when?) | single-trace, multi-label yes/no per 13 leaf categories |
| Mapping | "Who" → which agent; "When" → which step | "What" → which error category; "Where" → **dropped (no location)** |
| Output | single (agent, step) pair | binary vector of 13 yes/no answers |
| Ground-truth as hint | used | **dropped** (TRAIL/MAST are reference-free) |
| W3 (binary search) | implemented | **NOT implemented**; docstring line 33: *"binary search efficiency claim breaks under multi-label"* |

## File layout

3 active runners + plan doc + one normalizer:
| File | Backend |
|---|---|
| `baselines/who&when/run_who_and_when_vllm.py` | vLLM (local GPU) |
| `baselines/who&when/run_who_and_when_api_deepinfra.py` | DeepInfra |
| `baselines/who&when/run_who_and_when_api_arc.py` | ARC |
| `baselines/who&when/plan.md` | design doc |
| `baselines/who&when/old_run_who_and_when_vllm.py` | pre-patch backup |

The two API runners import all prompt/parsing logic from `run_who_and_when_vllm.py` via `sys.path.insert` — single source of truth.

## W1 prompt structure (holistic)
One call per trace. Structure:
1. Role framing with `{task_description}` (extracted via `extract_task_description()`: strips `[agent_role]` tags, truncates to 600 chars)
2. Full taxonomy + definitions + examples
3. Serialized trace (all steps as `[id]\n<content>`, max ~1200 chars/step)
4. `@@`-delimited answer template asking yes/no for all 13 categories
5. Optional separate rubric-scores call after the error call

Pre-patch backup (`old_*`) did NOT have task-description extraction.

## W2 prompt structure (step-by-step)
**One LLM call per step** (N+1 total per trace, including final rubric-scores call). Per-step prompt has:
1. Role framing
2. Taxonomy + definitions + examples
3. Cumulative prefix (steps 1..i, grown incrementally)
4. Current-step identification: `"Step i, span_id ..."`
5. Per-step instruction (errors in current step only); "avoid being overly critical" calibration sentence retained

Aggregation: skip `has_error=false` and parse failures. Dedup `(category, span_id)` pairs.

Cost: a typical 8-step trace = ~9 calls. Context-overflow guard breaks loop early if `tok_len + per_step_budget > max_model_len`.

## Output naming
```
outputs_<model_tag>-yesno-who_and_when_{w1|w2}[/-thinking]/
```
Two files per trace: `{rec_id}.json` (errors + scores) + `_meta_{rec_id}.json` (diagnostics).

## Key CLI args / defaults

| Arg | Default |
|---|---|
| `--variant` | `w1` (vLLM); required (API) — choices `w1`, `w2` (NOT `w3`) |
| `--input` | `data/annotation/annotation_ag2_filtered.jsonl` |
| `--output_dir` | `baselines/who&when/outputs` |
| `--model` | vLLM: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `--tensor_parallel_size` | vLLM: `2` |
| `--max_model_len` | vLLM: `131072` |
| `--max_new_tokens` | `8000` (auto-bumped to 24000 for reasoning models matching `qwenlong|-l1-|gpt-oss|deepseek-r1|qwq`) |
| `--rpm` | DI: 600, ARC: 30 |
| `--rph`/`--rp3h` | ARC: 1000 / 3000 |

**NOT present**: `--causal_only`, `--corr_threshold`, `--span_index`, `--random_edges`. Those exist only in `causal/` subdir runners (Part 2).

## Production-run intermediate statistics

All 10 vanilla W&W cells (5 models × 2 variants) complete; W-F1:

| Model | W1 | W2 |
|---|---|---|
| Mistral-Small-24B | 0.2308 | **0.0589** ← W2 severely under-detects |
| gpt-oss-120b | 0.2285 | 0.2944 |
| gpt-oss-20b | 0.1636 | 0.2655 |
| Gemma-3-27B | 0.1813 | 0.1184 |
| QwQ-32B | 0.1696 | 0.1828 |

**Critical clarification**: The "Mistral=0.3773 baseline" in MAST/TODO.md is the **main MAST baseline** (`run_eval_yesno_vllm.py` in Module A of `MAST_EVAL_PIPELINE_REFERENCE.md`), **not** W&W W1. Mistral W&W W1 is 0.2308, substantially lower.

## Backend gotchas
- **DeepInfra**: 5-retry exponential backoff (1+2+4+8+16s); recently patched to short-circuit on `"maximum context length"` / `"BadRequestError"` in error string
- **ARC**: 3-rule fairshare limiter (30/min, 1000/hr, 3000/3hr)
- **vLLM Mistral**: `fix_mistral_regex=True` on tokenizer load (works around regex bug in Mistral-3.1 tokenizers; rejected by Mistral-3.2's new tokenizer)
- **QwQ-32B post-processing**: `_normalize_qwenlong_w2_categories.py` is a sibling normalizer for QwenLong outputs that emit hierarchical category paths (`"Information Processing -> Tool Output Misinterpretation"`) — splits on common separators, takes last token, writes to `_leafnorm/`

## Scoring
Same as Module F: `eval/calculate_scores_yesno.py --pred_dir <out_dir>`. No separate W&W scorer.

## Compared to TRAIL W&W

| | TRAIL W&W | MAST W&W |
|---|---|---|
| Taxonomy | 19 leaves (hierarchical) | 13 codes (flat) |
| Location prediction | yes (span_id hex) | **none** |
| W3 | implemented but excluded from headline | **not implemented** |
| Splits | GAIA_dedup / SWE_Bench_dedup | none |
| Scorer | `calculate_scores.py` | `calculate_scores_yesno.py` |

---

# Part 2: W&W + Causal Graph Guidance

## Pairing decision (W1+GI vs W2+CG)

Same rationale as TRAIL. W1's single call per trace can absorb a Pass-2 (cost: 2 calls). W2 already costs N+1 calls per trace — adding a per-step Pass-2 would make it 2N+1, prohibitive. So:
- **W1+GI+SI** = W1 (holistic) ↔ +GI (two-pass dynamic)
- **W2+CG+SI** = W2 (step-by-step) ↔ +CG (one-pass static in every per-step prompt)

The +SI suffix in the paper-column name refers to **span-id validation**, NOT to a span-index prompt block (MAST has no location prediction). It's vestigial naming inherited from TRAIL.

Code comment in `run_who_and_when_graph_inject_vllm.py:14-18`:
> "Two-pass × per-span on W2 doubles the W2 call cost (2N calls/trace), which is already 9× a single-pass baseline. A trace-level Pass 2 preserves the N+1 cost profile and matches the structural choice in paper/baseline_who_and_when.tex."

## Code location

6 scripts in `baselines/who&when/causal/`:

| Variant | vLLM | DeepInfra | ARC |
|---|---|---|---|
| W&W + GI (two-pass) | `run_who_and_when_graph_inject_vllm.py` | `*_api_deepinfra.py` | `*_api_arc.py` |
| W&W + CG (one-pass) | `run_who_and_when_with_graph_vllm.py` | `*_api_deepinfra.py` | `*_api_arc.py` |

No litellm variants (Gemini/GPT-4o W&W omitted by design — too expensive at N+1 calls/trace).

## Imports from MAST eval

**MAST W&W causal/ scripts re-implement `load_graph_edges` and `format_graph_guidance` INLINE** (lines 99-180 and 88-171 of the respective vLLM scripts). They do NOT import from `MAST/eval/full_run_eval_*.py`. This differs from TRAIL, where these functions are imported.

The API variants (`api_deepinfra`, `api_arc`) import these from their sibling vLLM script via `sys.path.insert`. Self-contained.

Base W&W utilities (`format_trace`, `extract_task_description`, `build_cumulative_text`, `strip_thinking`, `parse_response`) are ALSO re-implemented inline (documented as "verbatim from MAST W&W baseline"), not imported from `run_who_and_when_vllm.py`.

## Prompt templates

**+CG (`run_who_and_when_with_graph_vllm.py`)**: Two functions, `get_w1_prompt` and `get_w2_step_prompt`. Each inserts a `{graph_block}` slot after EXAMPLES and before the trace/cumulative-history block. For W1 the block appears once; for W2 it's injected per step.

**+GI (`run_who_and_when_graph_inject_vllm.py`)**: Three functions:
- `get_w1_pass1_prompt` — **ignores** the `graph_guidance` argument (`del graph_guidance`); Pass-1 for W1+GI is graph-free, graph context reserved for Pass-2
- `get_w2_step_pass1_prompt` — DOES inject graph block per step (Pass-1 W2)
- `get_pass2_prompt` — single trace-level targeted call; shows Pass-1 detections + filtered subgraph; asks yes/no on target categories with "err on the side of YES when there is any plausible indication"

## Key CLI args / defaults

| Arg | Default | Notes |
|---|---|---|
| `--causal_only` | False | Use 11 validated edges from `effect_edges.json` |
| `--corr_threshold` | 1.0 | Union (geomean ≥ τ) ∪ (validated causal); ignored if `--causal_only` |
| `--effect_edges` | `causal_graph/outputs/interventions/effect_edges.json` | |
| `--suppes_graph` | `causal_graph/outputs/suppes_graph.json` | |
| `--variant` | (required) | `w1` or `w2` |
| `--w2_max_history_chars` | `80000` | Per-step cumulative-history char budget |
| `--propagation_threshold` | `0.1` | **+GI only** |
| `--span_index` | False | **No-op** for +CG (accepted for CLI parity); **NOT present** in +GI runners |
| `--model` | vLLM: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | |
| `--tp` | auto from `CUDA_VISIBLE_DEVICES` | |
| `--batch_size` | +GI: 4; +CG: 8 | |
| `--max_tokens` | 2000 (auto-bump 16000 for reasoning models) | |
| `--max_model_len` | 108000 (W&W vLLM; differs from main eval's 131072) | |
| `--gpu_memory_utilization` | 0.9 | |
| `--enable_thinking` | False | QwQ-32B and other thinking models |
| `--rpm`, `--rph`, `--rp3h` | DI: 600 / —; ARC: 30 / 1000 / 3000 | |

**NOT ported from TRAIL**: `--random_edges`, `--random_seed`, `--random_n`, `--edge_threshold`. Only `--causal_only` and `--corr_threshold` are available. No null-graph control for MAST W&W.

## Output directory naming

**+GI**: `<output_dir>/<model_tag>-yesno-who_and_when_{w1|w2}_graph_inject_<graph_tag>[-thinking]/`

**+CG**: `<output_dir>/<model_tag>-yesno-who_and_when_{w1|w2}_graph_<graph_tag>[-thinking][_span_index]/`

`<graph_tag>`: `causal_only` or `corr<value>` (e.g. `corr0.5`).

## Production-run intermediate statistics

Per MAST/TODO.md Task 2: the design pairs **W1+GI and W2+CG** (10 cells = 5 models × 2 variants). Plus a parallel **W1+CG** evaluation also exists on disk. Current scored cells (4 with metrics files):

| Cell | Dir name | W-F1 |
|---|---|---|
| Mistral W1+GI causal_only | `mistralai-...-w1_graph_inject_causal_only` | **0.2812** |
| Mistral W1+CG causal_only | `mistralai-...-w1_graph_causal_only` | 0.2435 |
| Gemma W1+CG causal_only | `google-gemma-3-27b-it-...-w1_graph_causal_only` | 0.1860 |
| QwQ-32B W1+CG causal_only thinking | `qwq-32b-...-w1_graph_causal_only-thinking` | 0.1641 |

**Pending per TODO.md Task 2**:
- Gemma W2+CG (infra issue; needs DeepInfra rerun)
- gpt-oss-20b W1+GI (393 files on disk, no metrics yet — needs scoring); W2+CG not started
- gpt-oss-120b: TODO says "done" but no metrics in `causal/outputs/` (likely scored under different output_dir via ARC)
- Mistral W2+CG and W2+GI: missing entirely
- QwQ-32B W2+CG and W1+GI rerun: not done (TODO notes W1+GI broken, needs rerun)

## Backend gotchas

- **vLLM W2 context-overflow**: `build_cumulative_text` applies `--w2_max_history_chars` (default 80,000) and silently trims earlier steps when exceeded (no hard break like TRAIL's `context_overflow_at_step_N`)
- **ARC fairshare** is the bottleneck for W2+GI runs: at 30 RPM × ~3500 calls per full run = ~117 minutes minimum before hourly caps bite
- **Reasoning auto-bump**: matches `gpt-oss|thinking|qwq|deepseek-r1|qwenlong`; bumps `max_tokens` to 16000. ARC variant checks only `"thinking"` in model name (no `gpt-oss` branch) — pass `--max_tokens 16000` explicitly for non-thinking gpt-oss on ARC if needed.
- **Harmony token artifacts**: `strip_harmony` removes `assistantfinal`, `<|channel|>…<|message|>`, `<|start|>/<|end|>/<|return|>` tokens that appear in raw vLLM output for some Mistral checkpoints; applied before parsing in all six scripts.

## Scoring
Same as Part 1: `python eval/calculate_scores_yesno.py --pred_dir <out_dir>`.

---

# Cross-references

- `MAST_CAUSAL_PIPELINE_REFERENCE.md` — `effect_edges.json` and `suppes_graph.json` provenance + edge-count breakdowns
- `MAST_EVAL_PIPELINE_REFERENCE.md` — Modules B (+CG) and C (+GI) in the main eval pipeline; W&W causal/ scripts mirror but re-implement inline
- `baselines/who&when/plan.md` — original design rationale + side-by-side adaptation table per variant
- `baselines/who&when/_normalize_qwenlong_w2_categories.py` — QwenLong-specific post-processor for hierarchical category outputs
- `paper/tables/who_and_when_results.tex` (presumably) — consumes these outputs; would need 10 reruns to pivot to τ=0.50 per `TODO_0.5.md` section 3
