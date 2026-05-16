# TRAIL Causal Pipeline — Reference

> One-stop reference for the causal graph construction + intervention validation pipeline. Use this instead of re-reading the code every time you need to remember what produces what.

**Compiled**: 2026-05-16 by 7 parallel sonnet agents (one per module). Module 4 (bootstrap stability + shuffle control) is omitted — those scripts exist in `causal/graph/CAPRI/` but are not invoked in the current production config (`_AIC`), and their outputs are not consumed by any downstream experiment.

---

## TL;DR — what eval actually reads

After all modules complete, **only two files** are touched at evaluation time:

| File | Source module | Eval-time role |
|---|---|---|
| `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json` | M2 (Suppes screen) | candidate edges + per-edge `precedence` / `pr_delta` scores; consumed by `load_graph_edges` for `--corr_threshold` and pure `--edge_threshold` modes |
| `benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json` | M8d (effect eval) | 12 intervention-validated edges (`validated=True`); the causal anchor in `--causal_only` mode and the union set in `--corr_threshold` mode |

Everything else (`capri_graph.json`, `hierarchy_levels.json`, `eligible_traces.json`, `edge_stability.*`, `controls_shuffle.json`) is intermediate or analytical — **none of it is read by `benchmarking/eval/`**.

## Edge counts at runtime (production `_AIC` Suppes graph)

| Variant | Edges | How produced |
|---|---|---|
| `--causal_only` | **12** | `effect_edges.json`, filter `validated=True` |
| `--corr_threshold 0.35` | **19** | Suppes geomean `sqrt(precedence × pr_delta) ≥ 0.35` (15) **∪** causal-validated edges below τ (4) |
| `--corr_threshold 0.25` | **21** | 19 above τ + 2 causal-anchor below τ |
| `--corr_threshold 0.20` | **25** | 24 above τ + 1 causal-anchor below τ |
| `--edge_threshold τ` (pure Suppes) | varies | Suppes geomean ≥ τ, no causal union |
| `--random_edges --random_n 12` | 12 | random pairs from taxonomy, excluding all Suppes pairs |

All edges (regardless of source) carry the same **Suppes geomean weight** `sqrt(precedence × pr_delta)` at eval time. The `abs(delta)` score from interventions is read only in `--causal_only` mode (per `_parse_causal_graph`).

## `_AIC` vs `_fc` Suppes graph

| Dir | `min_precedence` | `min_pr_delta` | `min_joint` | Total edges | Used at eval? |
|---|---|---|---|---|---|
| `trail_causal_outputs_full_gaia_swe_AIC/` | 0.55 | 0.05 | 3 | **27** | ✓ runtime default |
| `trail_causal_outputs_full_gaia_swe_fc/` | 0.0 | 0.0 | 1 | 146 | ✗ legacy snapshot |

Despite the `_AIC` name, **this has nothing to do with the Akaike Information Criterion**. The three threshold values are hand-picked. `_fc` ("full candidates") is the unfiltered baseline — every (a,b) pair with at least one joint co-occurrence; it exists only as a diagnostic reference and is NOT what the runs in `outputs_thres/` consumed.

---

## Pipeline overview

```
┌─────────────────────── Stage 1: Graph construction (causal/graph/) ───────────────────────┐
│                                                                                            │
│  M1 preprocess         M1 build_order_pairs       M2 suppes_screen      M3 capri_prune    │
│  trail_{1,2,3}_*.py    1_build_order_pairs.py     2_suppes_screen.py    3_capri_prune.py  │
│        │                       │                         │                      │          │
│  raw TRAIL                 onsets_*.jsonl ──► order_pairs.jsonl ──► suppes_graph.json ──► capri_graph.json
│  traces +                                                  │                      │          │
│  annotations                                               │                      │          │
│                                                            ▼                      ▼          │
│                                              [eval consumes directly]   [unused at eval]    │
│                                                                                              │
│                                            M5 export_hierarchy                              │
│                                            6_export_hierarchy.py                            │
│                                                    │                                        │
│                                              hierarchy_levels.json                          │
│                                              [unused at eval; viz only]                     │
│                                                                                              │
│                       Orchestrated by M6 run_causal_from_trail_onsets.py                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────── Stage 2: Intervention validation (causal/patch/ + causal/intervention/) ─────────┐
│                                                                                                │
│  capri_graph.json ──┐                                                                          │
│                     ▼                                                                          │
│   M7 run_pipeline.py ──► 8-step intervention pipeline:                                        │
│        Step 0 filter_traces      ──► eligible_traces.json                                     │
│        Step 1 case_builder       ──► a_instances.jsonl, edge_pairs.jsonl                      │
│        Step 2 patch_generator    ──► patch_results.jsonl                                      │
│        Step 3 rerun_harness      ──► rerun_results.jsonl                                      │
│        Step 4 judge_a_resolved   ──► a_resolved.jsonl                                         │
│        Step 5 judge_b_effect     ──► b_effect.jsonl                                           │
│        Step 6 effect_aggregator  ──► effect_edges.json  ◄── consumed by eval                 │
│                                                                                                │
│   M8 effect_eval.py — alternative aggregator path (annotation proxy, no rerun)                │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Stage 1 — Graph Construction

## Module 1: Preprocess + Order Pairs

**Purpose**: Convert raw TRAIL annotation JSONs and execution-trace JSONs into per-trace onset tables (the first occurrence rank of each error category within a trace), then enumerate all pairwise temporal-precedence relations between error categories.

**Code**: `causal/graph/preprocess/trail_{1,2,3}_*.py`, `causal/graph/CAPRI/1_build_order_pairs.py`

### Inputs
- `benchmarking/data/{GAIA,SWE Bench}/<trace_id>.json` — raw TRAIL execution traces (nested span trees)
- `benchmarking/processed_annotations_{gaia,swe_bench}/<trace_id>.json` — human annotations with `category` + `location` (span_id)

### Outputs

| File | Producer | Schema |
|---|---|---|
| `benchmarking/data/trail_filtered/{gaia,swe}.jsonl` | `trail_1_filter_split.py` | `{trace_id, split, trace_path, annotation_path, n_errors, error_locations}` |
| `benchmarking/data/trail_span_order/{gaia,swe}.jsonl` | `trail_2_build_span_order.py` | `{trace_id, split, span_rank, missing_annotated_span_ids}` — span_rank is `{span_id → 0-based chronological position}` |
| `benchmarking/data/trail_derived/onsets_gaia_swe_full.jsonl` | `trail_3_build_onsets.py` | `{trace_id, split, present: {cat: 1/0}, onset: {cat: step_idx}, count: {cat: n}, ties: [[catA, catB], ...]}` |
| `benchmarking/data/trail_causal_outputs_*/order_pairs.jsonl` | `1_build_order_pairs.py` | `{trace_id, pairs: [[A, B], ...]}` — `[A, B]` means A's onset step `<` B's onset step in this trace |

### Key CLI args
- `trail_1`: `--data_dir` (`"data"`), `--annotation_dir` (cwd), `--split` (`"both"`)
- `trail_2`: `--fallback_all_spans` (`True`) — if annotated span IDs aren't in the LLM/TOOL/CHAIN candidate set, rank all spans
- `trail_3`: `--include_ties` (`True`)
- `1_build_order_pairs`: `--in_path`, `--out_path`

### Algorithm
`trail_1` pairs traces with annotations by filename. `trail_2` walks the span tree, ranks spans chronologically by ISO timestamp (tie-break on `span_id`), **filters to spans with `openinference.span.kind ∈ {LLM, TOOL, CHAIN}`** (falls back to all spans if a required annotated span is missing). `trail_3` computes each category's onset as the **minimum rank** across spans where it was annotated. `1_build_order_pairs.py` enumerates all category pairs in a trace, emits `[A, B]` iff `onset[A] < onset[B]`. **Exact ties are silently dropped** in order-pair output (but listed in the `ties` field of the onsets file).

### Gotchas
- The `present`/`count` fields have a **fixed-width vocabulary** of all categories seen globally. Adding a new split later invalidates old onsets files.
- Intermediates `trail_filtered/` and `trail_span_order/` are persistent — re-running `trail_3` without re-running 1+2 silently reuses stale span ranks.

---

## Module 2: Suppes Screen

**Purpose**: Apply Suppes' probabilistic theory of causation — temporal precedence + probability raising — to every ordered pair of failure modes, discarding pairs that fail minimum-support thresholds. Produces the candidate edge set used by all downstream graph variants.

**Code**: `causal/graph/CAPRI/2_suppes_screen.py`

### Inputs
- `order_pairs.jsonl` — actually the script reads the **onsets** JSONL directly (not `order_pairs.jsonl`); recomputes ordering from `onset` dict per trace

### Output: `suppes_graph.json`

```json
{
  "params": {"min_precedence": 0.55, "min_pr_delta": 0.05, "min_joint": 3, "in_path": "..."},
  "n_traces": 148, "n_modes": 19, "n_edges": 27,
  "edges": [
    {"a": "Tool Selection Errors", "b": "Goal Deviation",
     "precedence": 1.0, "precedence_n": 9,
     "p_b_given_a": 0.7838, "p_b_given_not_a": 0.3153, "pr_delta": 0.4685},
    ...
  ]
}
```

### Per-edge fields
| Field | Meaning |
|---|---|
| `precedence` | Among traces where both A and B have onset entries, fraction where `onset[A] < onset[B]`. Ties excluded from numerator AND denominator. |
| `precedence_n` | Denominator of `precedence` — count of non-tied joint-onset traces |
| `p_b_given_a` | P(B has onset \| A has onset), estimated over ALL traces |
| `p_b_given_not_a` | P(B has onset \| A does NOT have onset), over ALL traces |
| `pr_delta` | `p_b_given_a − p_b_given_not_a` |

### Key CLI args (defaults at lines 33-41)
- `--in_path` (`"data/derived/onsets.jsonl"`)
- `--out_path` (`"outputs/suppes_graph.json"`)
- `--min_precedence` (**0.55**) — help string says 0.6 but actual default is 0.55 (stale help)
- `--min_pr_delta` (**0.05**) — help says 0.02; actual is 0.05
- `--min_joint` (**3**) — help says 30; actual is 3

### Survival rule
An edge A→B is retained iff **all three** conditions hold:
`precedence_n ≥ min_joint` AND `precedence ≥ min_precedence` AND `pr_delta ≥ min_pr_delta`.

### Important: the geomean `sqrt(precedence × pr_delta)` is NOT computed here
It is computed downstream at eval time by `benchmarking/eval/run_eval_graph_inject_vllm.py:load_graph_edges` (lines 262, 269) when applying `--corr_threshold`. This module only writes raw `precedence` and `pr_delta`.

---

## Module 3: CAPRI Prune

**Purpose**: Among the Suppes candidate edges, select the highest-scoring DAG subset using greedy hill-climbing over a BIC/AIC-penalized Bayesian network score. Produces `capri_graph.json` — a sparser, acyclic graph.

**Code**: `causal/graph/CAPRI/3_capri_prune.py`

### Inputs
- `suppes_graph.json` (M2) — used only as the **candidate set**; Suppes scores are NOT carried forward
- `onsets.jsonl` (M1) — re-read here to build the binary design matrix `X` (traces × modes) for likelihood scoring

### Output: `capri_graph.json`

Top-level keys: `params`, `n_traces`, `n_modes`, `suppes_n_edges`, `pruned_n_edges`, `score_history`, `edges`.
Per-edge: `{a, b}` only — **no scores carried over, no new scores assigned**.
Production AIC run: 27 Suppes candidates → **13 CAPRI edges** after 13 accepted hill-climbing moves.

### Key CLI args (lines 209-215)
- `--criterion` (`"BIC"`) — choices: `BIC` or `AIC`. **Production uses `AIC`.**
- `--max_parents` (`None`) — optional cap on in-degree per node
- `--max_iters` (`500`)

### Algorithm
Greedy hill-climbing starting from the empty graph. At each iteration, enumerate single-edge moves (add / remove / reverse), accepting whichever gives the largest score decrease. Local score per node `b` with parent set `pa`: penalized log-likelihood of a binary CPT with Laplace smoothing; penalty is `k_params × log(n_traces)` (BIC) or `2 × k_params` (AIC) where `k_params = 2^|pa|`. **Acyclicity is enforced** via DFS before admitting `add`/`reverse` moves.

### Status at eval time
- **NOT the default** — `DEFAULT_CAUSAL_GRAPH` in eval scripts points to `effect_edges.json`, NOT `capri_graph.json`.
- **Can be used as an override** by passing `--causal_graph .../capri_graph.json`. The eval's `_parse_causal_graph` handles both formats: dict-of-edges (effect_edges.json) → filters `validated=True`; list-of-edges (capri_graph.json) → no filter, returns all edges with Suppes geomean as weight.
- In practice the runs in `outputs_thres/` and `outputs/zero_shot{,2}/` use the DEFAULT (effect_edges.json), so `capri_graph.json` does NOT influence published numbers directly.

---

## Module 5: Hierarchy Export

**Purpose**: Topological-depth stratification of nodes in the (pruned) DAG — for visualization and human-readable structure only.

**Code**: `causal/graph/CAPRI/6_export_hierarchy.py`

### Inputs
- `capri_graph.json` (M3)
- `edge_stability.json` (M4, OPTIONAL — falls back to all edges if missing)

### Output: `hierarchy_levels.json`

```json
{
  "params": {"capri_path": "...", "stability_path": "...", "stability_threshold": 0.3},
  "n_levels": 5, "n_nodes": 12,
  "levels": {
    "level_0": ["Formatting Errors", "Tool Selection Errors"],
    ...
    "level_4": ["Goal Deviation"]
  }
}
```

### CLI args
`--capri_path`, `--stability_path`, `--out_path`, `--stability_threshold` (**0.3**)

### Algorithm
1. Filter edges below `--stability_threshold` (if stability file present)
2. Detect cycles via DFS; break each by removing its lowest-stability edge
3. BFS-style level propagation: parents-free nodes → level 0; each other node → `max(parent levels) + 1`

### Status at eval time
**NOT consumed by any eval script.** Only two downstream uses:
1. `run_causal_from_trail_onsets.py` orchestrator (it just writes the file)
2. `visualize_graphs.py` — optional `--hierarchy` flag for column layout in figures

---

## Module 6: Graph Construction Orchestrator

**Purpose**: End-to-end driver that invokes M1→M5 in sequence and writes all intermediates to one per-config output directory.

**Code**:
- `causal/graph/run_causal_from_trail_onsets.py` — Python driver
- `causal/graph/run_causal_gaia.sh` — Bash wrapper that runs the preprocess (M1) scripts; **does NOT call the Python driver itself**

### Key CLI args of `run_causal_from_trail_onsets.py`
| Flag | Default | Notes |
|---|---|---|
| `--onsets_path` | `benchmarking/data/trail_derived/onsets_gaia.jsonl` | M1 output |
| `--out_dir` | `benchmarking/data/trail_causal_outputs` | output root |
| `--min_precedence` | `0.55` | → M2 |
| `--min_pr_delta` | `0.05` | → M2 |
| `--min_joint` | `3` | → M2 |
| `--criterion` | `"BIC"` | → M3 (production uses `AIC`) |
| `--max_parents` | `None` | → M3 |
| `--n_bootstrap` | `100` | M4a (skippable) |
| `--n_shuffles` | `50` | M4b (skippable) |
| `--skip_bootstrap` | `False` | omit M4a |
| `--skip_shuffle` | `False` | omit M4b |

### Execution sequence
M1 → M2 → M3 → M4a (opt) → M4b (opt) → M5. Each step is invoked via `subprocess.run` with cwd set to `benchmarking/`. M1 failure is non-fatal; M2/M3 failures are fatal.

### Output directory layout (full run)
```
out_dir/
  order_pairs.jsonl          (M1)
  suppes_graph.json          (M2)
  capri_graph.json           (M3)
  edge_stability.csv         (M4a, if not skipped)
  edge_stability.json        (M4a)
  controls_shuffle.json      (M4b, if not skipped)
  hierarchy_levels.json      (M5)
  eligible_traces.json       (added later by Stage-2 filter)
  eligible_gaia/, eligible_swe/  (added later by Stage-2 filter)
```

### Production configs on disk
- **`trail_causal_outputs_full_gaia_swe_AIC/`** — runtime default; min_precedence=0.55, min_pr_delta=0.05, min_joint=3, criterion=AIC; 27 Suppes / 13 CAPRI edges; **bootstrap & shuffle skipped** (`edge_stability.*` and `controls_shuffle.json` absent)
- **`trail_causal_outputs_full_gaia_swe_fc/`** — legacy unfiltered; min_*=0/0/1; 146 Suppes edges; **not used at eval**

### Visualization sidecar: `visualize_graphs.py`
Reads `effect_edges.json` (and optionally `suppes_graph.json` + `hierarchy_levels.json`); renders `graph_causal.png` (intervention-validated DAG, weighted by `|Δ(A→B)|`) and optionally `graph_corr_t<τ>.png` (Suppes ∪ validated at threshold τ).

---

# Stage 2 — Intervention Validation

## Module 7: Patch Generation Infrastructure

**Purpose**: For every (A, B) candidate edge in the CAPRI graph, surgically remove error A from real trace spans, re-run the agent counterfactually, and judge whether B disappears. Produces `effect_edges.json` — the intervention-validated graph that becomes the causal anchor at eval time.

**Code**: `causal/patch/` (everything)
**Top-level driver**: `causal/patch/run_pipeline.py`

### 8-step pipeline (steps numbered as in the driver, not as separate modules)

| Step | Script | Output | Purpose |
|---|---|---|---|
| 0 | `filter_traces.py` | `eligible_traces.json` | keep traces with ≥ `min_errors` annotations and at least one A-typed error from the graph |
| 0b (opt) | `sample_coverage.py` | `eligible_traces_test.json` | greedy set-cover sub-sample for cheap test runs |
| 1 | `case_builder.py` | `a_instances.jsonl`, `edge_pairs.jsonl`, `intervention_location_conflicts.jsonl` | turn each (trace, error) into an `AInstanceRecord` + (A-instance × B-type) `EdgePair`s; dedup by intervention location |
| 2 | `patch_generator.py` | `patch_results.jsonl`, `postcheck_failures.jsonl` | LLM-generates patches with rule-based postcheck + retries |
| 3 | `rerun_harness.py` | `rerun_results.jsonl` | apply patch at t_A, replay tool responses, re-call LLM for ≤ `max_steps_after` (default 12) post-intervention steps |
| 4 | `judge_a_resolved.py` | `a_resolved.jsonl` | LLM check: did the patch actually remove A? (treatment validity) |
| 5 | `judge_b_effect.py` | `b_effect.jsonl` | LLM check per (A-instance, B-type): one of 8 labels (`disappeared`, `delayed`, `unchanged`, `earlier`, `weakened`, `strengthened`, `emerged`, `not_observable`) + `target_present_after` (bool) |
| 6 | `effect_aggregator.py` | `effect_edges.json` | per-edge Δ(A→B), validation flag, placebo null |

### Key concept: AInstanceRecord vs EdgePair
- **AInstanceRecord** = one (trace, error_id) instance of error A. Carries `local_snippet`, `replace_span_output` vs `replace_span_input`, the effective intervention span ID (which may differ from annotated when a TOOL error must be patched at its parent LLM span), system prompt, available tools.
- **EdgePair** = one (A-instance × B-type) graph edge to evaluate.
- **Many EdgePairs share the same A-instance** → one rerun is shared across all that A-instance's downstream B-types.

### `patch_generator.py` (the active one) vs `patch_generator_llm.py` (older, unused)
The active generator (`patch_generator.py`) makes a single combined LLM call per A-instance using `PATCH_SYSTEM` + `PATCH_USER_TEMPLATE`, returning `{slot_values, patch_payload, postcheck}`. Then runs an independent rule-based postcheck (required markers, no novel `<...>` tokens, patch differs from snippet) with `--max_retries` (default 3).

The older `patch_generator_llm.py` is a two-call diagnosis-first approach. It is NOT invoked by `run_pipeline.py` but its `_call_llm` helper is imported by both judges and the main generator.

### `patch_library.json`
Per-category patch templates. Top-level keys are TRAIL A-category names. Each entry has `category`, `trail_definition`, `patch_side_default`, `slot_schema`, `error_type_spec_text` (full text block injected into prompt), plus `repair_instruction` / `forbidden_actions` / `postcheck`.

### `effect_aggregator.py`
Δ(A→B) = `mean(target_present_after | resolved) − mean(b_present_baseline | resolved)`. Negative Δ ⇒ fixing A reduced B.
Edge is `validated=True` when `Δ < −threshold` (default `0.15`) AND `n ≥ min_n` (default `3`).
Also computes a **cross-edge placebo** (each edge's rerun vector kept, baseline labels drawn from OTHER edges' baselines; `placebo_seeds=100` bootstrap iterations).

### `recompute_placebo.py`
One-shot utility — recomputes only placebo from existing `a_resolved.jsonl` + `b_effect.jsonl`, no LLM re-calls.

### `rerun_harness.py` (Step 3)
Converts smolagents `tool-call` / `tool-response` roles to OpenAI `assistant+tool_calls` / `tool+tool_call_id`. For `replace_span_output`: injects patch as the assistant message at t_A. For `replace_span_input`: replaces last user message and re-calls the LLM to regenerate t_A's output. Continuation loop runs one LLM call per subsequent step (including pure planning steps) up to `max_steps_after`; terminates early on `final_answer` tool call.
Default 12 steps covers ~94% of A→B distances on 102 eligible GAIA pairs.

### `run_pipeline.py` CLI surface (selected)

| Arg | Default | Notes |
|---|---|---|
| `--trace_dir` | `data/GAIA` | |
| `--annotations_dir` | `processed_annotations_gaia` | |
| `--causal_graph` | `data/trail_causal_outputs_full_gaia_swe_AIC/capri_graph.json` | **NOTE: Stage-2 takes the CAPRI-pruned graph as its candidate edge set, NOT the Suppes graph** |
| `--patch_library` | `causal/patch/patch_library.json` | |
| `--out_dir` | `outputs/interventions` | |
| `--model` | `openai/gpt-4o` | for patch gen + Judge A + Judge B |
| `--rerun_model` | `openai/o3-mini` | for counterfactual rerun; should match trace model |
| `--threshold` | `0.15` | Δ-validation cutoff |
| `--min_n` | `1` | min interventions |
| `--max_steps_after` | `12` | rerun budget |
| `--skip_filter` / `--skip_cases` / `--skip_patches` / `--skip_rerun` / `--skip_judge_a` / `--skip_judge_b` | all `False` | resume / skip steps |
| `--merge_from` | `None` | merge prior run's results before aggregation |

---

## Module 8: Intervention Application + Effect Evaluation (lightweight path)

**Purpose**: An alternative, simpler pipeline alongside M7. Where M7 actually re-runs the agent with the patched trace, M8 uses an **annotation-based proxy** — for each trace where A was patched, count how often B appeared downstream of A in the **baseline annotations** (no agent rerun). Produces an upper-bound prevention effect.

**Code**:
- `causal/intervention/intervene.py` — main intervention loop (rule-based patches, no LLM)
- `causal/intervention/patch_apply.py` — patch loader + applicator
- `causal/intervention/trace_replay.py` — extracts ordered LLM/TOOL steps; supports `rerun` mode counterfactuals
- `causal/intervention/rerun_intervention.py` — standalone single-intervention re-run pipeline
- `causal/intervention/effect_eval.py` — Δ aggregator → `effect_edges.json`
- `causal/summarize_effects.py` — standalone aggregation utility for M7's `b_effect.jsonl` (NOT a shim)
- Top-level shims (12 lines each, just path setup + import): `causal/intervene.py`, `causal/effect_eval.py`, `causal/rerun_intervention.py`

### Step 8a: `intervene.py`
Routes each annotated error to an operator family via `ERROR_TYPE_TO_FAMILY`, instantiates a rule-based patch (no LLM), validates, writes the patch log.
**CLI**: `--trace_dir`, `--annotations_dir`, `--patch_specs_dir` (`data/patches`), `--out_dir`, `--window` (`0` = only annotated span; `1` = ±1 sibling).
**Outputs**: `patch_log.jsonl`, `patched_traces.jsonl`, `intervention_stats.json`

### Step 8b: `patch_apply.py` + `trace_replay.py`
Validation rules: patch must (1) change something, (2) change `<60` lines, (3) not fabricate `Observation:` content, (4) for BUDGET_GUARD patches contain guard keywords. **Patch never mutates the trace object** — only string copies.
Trace is intentionally NOT truncated after the patched span (later steps are needed for causal evaluation). Truncation is reserved for patch-validity checks only.
**`patch_only` mode** (used for production `effect_edges.json`): LLM NOT re-called; patched text injected directly. **`rerun` mode**: extracts exact prefix messages, builds observation tape from original post-intervention tool outputs, re-calls LLM for each subsequent LLM step.

### Step 8c: `rerun_intervention.py`
For each error A_i in each trace, creates one counterfactual `do(A_i ≈ 0)` trace and saves in GAIA format. In `rerun` mode also saves the full rerun transcript.
**Outputs**:
- `out_dir/rerun_log.jsonl` — one line per intervention
- `gaia_output_dir/<trace_id>_do_<i>_<safe_id>.json` (mode `patch_only`)
- `gaia_rerun_dir/<trace_id>_do_<i>_<safe_id>_rerun.json` (mode `rerun`)

### Step 8d: `effect_eval.py` — produces `effect_edges.json`

**Annotation-proxy Δ formula**:
`Δ(A→B) = n_traces_where_B_follows_A / n_patches_targeting_A`
Edge `validated=True` when `Δ ≥ validation_threshold` (CLI default `0.1`; the merged production file uses `0.15`).

### `effect_edges.json` schema (per the live merged file)

Top-level:
```json
{
  "edges": { "Formatting Errors -> Context Handling Failures": { ... }, ... },
  "placebo": {"null_delta_mean": ..., "null_delta_std": ..., "n_placebo_samples": ...},
  "patch_failure_by_category": { "Tool Selection Errors": 0.12, ... },
  "validation_threshold": 0.15,
  "min_n": 1
}
```

Per-edge:
| Field | Meaning |
|---|---|
| `a`, `b` | cause / effect error type strings |
| `n_valid_interventions` | successful patch interventions on A |
| `patch_failure_rate` | fraction of patch attempts on this category that failed validation |
| `b_present_baseline_rate` | fraction where B was present downstream of A in original trace |
| `b_present_rerun_rate` | fraction where B was present after `do(A=0)` |
| `delta` | `b_present_rerun_rate − b_present_baseline_rate`; **negative = causal prevention** |
| `placebo_mean`, `placebo_std` | null distribution stats |
| `effect_label_distribution` | counts of per-trace effect labels (`disappeared`, `weakened`, `unchanged`, etc.) |
| `validated` | `True` iff `|delta| ≥ validation_threshold` AND delta is negative |
| `in_capri_graph` | whether this edge is also in `capri_graph.json` |

### `effect_eval.py` CLI args (lines 401-421)
| Arg | Default |
|---|---|
| `--patch_log` | `outputs/interventions/patch_log.jsonl` |
| `--annotations_dir` | `processed_annotations_gaia` |
| `--stage1_edges` | `None` |
| `--out` | `outputs/interventions/effect_edges.json` |
| `--threshold` | `0.1` |

### `summarize_effects.py`
Standalone 294-line aggregator (NOT a shim). Merges `b_effect.jsonl` files from one or more M7 run directories. Validation threshold `delta ≤ −0.3` (line 54). Prints formatted table; optional `--out <path>.csv`.

### Connection to eval
`benchmarking/eval/run_eval_graph_inject_vllm.py` (line 42):
`DEFAULT_CAUSAL_GRAPH = BENCH_DIR / "outputs" / "interventions_full_gaia_swe_merged" / "effect_edges.json"`

`_parse_causal_graph()` (lines 211-222) opens this file, reads `data["edges"]`, filters to entries with `v.get("validated", False) == True`, returns `(a, b, abs(delta))` triples. **These 12 validated edges form the causal anchor** used in every `--causal_only` and `--corr_threshold` eval variant.

---

# Cross-cutting notes

## "M7 vs M8" — which one produces the production `effect_edges.json`?

Two different aggregators exist:
- **M7's `effect_aggregator.py`**: full pipeline with actual agent reruns and LLM judges; writes `effect_edges.json` to its own `--out_dir`.
- **M8's `effect_eval.py`**: annotation-proxy aggregator; faster but coarser.

The production file at `benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json` carries both `placebo` stats and `effect_label_distribution` — both signatures point at **M7's aggregator** (M8's effect_eval doesn't compute placebo from M7's b_effect.jsonl format). Treat M7 as the canonical producer; M8 is the lightweight alternative path.

## Quick-reference grep cheat sheet

| If you're asking… | grep this |
|---|---|
| What produces `suppes_graph.json`? | `causal/graph/CAPRI/2_suppes_screen.py` |
| What produces `effect_edges.json`? | `causal/patch/effect_aggregator.py` (or `causal/intervention/effect_eval.py` for the proxy path) |
| What does the eval read? | `DEFAULT_(CAUSAL\|SUPPES)_GRAPH` in `benchmarking/eval/run_eval_graph_inject*.py` |
| How is the corr-union computed at eval time? | `load_graph_edges` in `benchmarking/eval/run_eval_graph_inject_vllm.py` (around line 230-275) |
| Where are the rule-based patch specs? | `benchmarking/data/patches/` (M8 path) and `causal/patch/patch_library.json` (M7 path) |
| Where is the production `effect_edges.json`? | `benchmarking/outputs/interventions_full_gaia_swe_merged/effect_edges.json` |
| Where is the production `suppes_graph.json`? | `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json` |

## What "validated" means precisely

In `effect_edges.json`, `validated: True` requires:
- `|delta| ≥ validation_threshold` (production file: `0.15`)
- `delta < 0` (intervention suppressed B, not strengthened it)
- `n_valid_interventions ≥ min_n` (production: `1`)

So a `validated=True` edge has **measurable causal prevention** of B by `do(A=0)`. The 12 edges with this flag at production-merged config are the anchor used by `--causal_only` and the union-pin set used by `--corr_threshold`.

## Production-run intermediate statistics

Numbers below come from the actual on-disk artifacts (`trail_causal_outputs_full_gaia_swe_AIC/` + `interventions_full_gaia_swe_merged/effect_edges.json`). Use these to understand both data scale and quality at each stage — not just schemas.

### M1 → M5 (graph construction)

| Stage | Output | N |
|---|---|---|
| M1 preprocess | `onsets_gaia_swe_full.jsonl` traces | **148** |
| M1 preprocess | distinct error-mode categories observed | **19** |
| M2 Suppes (production `_AIC`: min_precedence=0.55, min_pr_delta=0.05, min_joint=3) | edges in `suppes_graph.json` | **27** |
| M2 Suppes (`_fc`: no filter) | edges in `suppes_graph.json` | 146 (not used at eval) |
| M3 CAPRI (AIC, no max_parents cap) | edges in `capri_graph.json` after 13 accepted hill-climbing moves | **13** |
| M5 hierarchy | levels × nodes in `hierarchy_levels.json` | **5 levels / 12 nodes** |

### M7 + M8 (intervention validation — `effect_edges.json`)

**Top-level run config**:
- `validation_threshold` = **0.15** (the `|Δ| ≥ τ AND Δ < 0` cutoff for `validated=True`)
- `min_n` = 1 (minimum interventions per edge)

**Edge yield** (per `effect_edges.json`):

| | Count |
|---|---|
| Candidate edges tested (= CAPRI graph) | 13 |
| Validated edges (`validated=True`) | **12** |
| Validation rate | 92% (12/13) |

The 1 unvalidated edge is `Tool Selection Errors → Task Orchestration` (Δ = -0.143, just below the 0.15 threshold).

**Sample size per edge** (`n_valid_interventions`): min 1, median **16**, max 73, **total = 308 successful intervention trials** across the 13 candidate edges.

**Per-edge effect strengths** (sorted by Δ, strongest causal effect first):

| A → B | n | Δ | b base→rerun | validated |
|---|---|---|---|---|
| Tool Selection → Goal Deviation | 29 | -0.724 | 0.93 → 0.21 | ✓ |
| Formatting → Incorrect Problem Identification | 19 | -0.579 | 0.84 → 0.26 | ✓ |
| Formatting → Resource Abuse | 73 | -0.548 | 0.92 → 0.37 | ✓ |
| Formatting → Context Handling Failures | 63 | -0.540 | 0.83 → 0.29 | ✓ |
| Tool-related → Goal Deviation | 16 | -0.500 | 0.94 → 0.44 | ✓ |
| Incorrect Problem Identification → Language-only | 7 | -0.429 | 0.57 → 0.14 | ✓ |
| Tool Selection → Language-only | 16 | -0.375 | 0.62 → 0.25 | ✓ |
| Formatting → Poor Information Retrieval | 42 | -0.310 | 0.79 → 0.48 | ✓ |
| Poor Information Retrieval → Resource Abuse | 15 | -0.267 | 0.53 → 0.27 | ✓ |
| Incorrect Problem Identification → Tool Output Misinterpretation | 8 | -0.250 | 0.62 → 0.38 | ✓ |
| Resource Abuse → Tool-related | 5 | -0.200 | 0.40 → 0.20 | ✓ |
| Resource Abuse → Authentication Errors | 1 | -1.000 | 1.00 → 0.00 | ✓ |
| Tool Selection → Task Orchestration | 14 | -0.143 | 0.50 → 0.36 | **✗** |

**Patch validation success rate** (`patch_failure_by_category`, per source category — only A-categories that had patches generated):

| Category | Patch-failure rate |
|---|---|
| Resource Abuse | 11.1% (worst) |
| Formatting Errors | 10.6% |
| Tool Selection Errors | 2.9% |
| Tool-related | 0.0% |
| Poor Information Retrieval | 0.0% |
| Incorrect Problem Identification | 0.0% |
| **Mean across 6 categories** | **4.1%** |
| **Categories with zero failures** | **3 of 6** |

In short: the rule-based patch generators succeed ~96% of the time on average; only `Resource Abuse` and `Formatting Errors` patches occasionally fail their internal post-checks.

**Aggregated effect-label distribution** (Judge-B counts across all validated edges, n = 294 trials):

| Label | Count | % |
|---|---|---|
| `disappeared` | 169 | **57.5%** |
| `unchanged` | 42 | 14.3% |
| `emerged` | 31 | 10.5% |
| `not_observable` | 29 | 9.9% |
| `weakened` | 21 | 7.1% |
| `strengthened` | 2 | 0.7% |

So 64.6% of intervention trials produced `disappeared` or `weakened` (the expected causal-prevention outcomes); 0.7% produced the counter-causal `strengthened` outcome.

**Placebo null distribution** (cross-edge bootstrap, 1300 samples):
- mean Δ = **-0.5188**
- std Δ = 0.1964

The null mean is large and negative because all candidate edges have high baseline `b_present` rates by construction (CAPRI screening). What matters is each real edge's Δ relative to its **edge-specific** placebo (`placebo_mean` / `placebo_std` per edge), not the cross-edge aggregate. For example, Formatting → Context Handling Failures has Δ = -0.540 vs `placebo_mean` = -0.509 (essentially at null), while Tool Selection → Goal Deviation Δ = -0.724 sits ~2× std above its placebo.

### Eval-stage trace counts (M7's M7 Step 0 + main eval data)

| Split | n_traces |
|---|---|
| `GAIA_dedup` | 117 (some metrics files show 125 — verify; the production run files show 117) |
| `SWE_Bench_dedup` | 31 |

---

## What is NOT used at eval time (current production config)

- `capri_graph.json` — only used by Stage-2 patch pipeline (M7's `--causal_graph` arg) as the candidate edge set for which (A,B) to test. NOT injected into LLM prompts.
- `hierarchy_levels.json` — visualization only.
- `edge_stability.{csv,json}`, `controls_shuffle.json` — bootstrap+shuffle skipped in current `_AIC` config; files absent on disk.
- `order_pairs.jsonl` — re-read by M2 from onsets directly; the file itself isn't needed downstream.
- `eligible_traces.json` — only consumed by Stage-2 (M7 Step 0); not eval-relevant.
