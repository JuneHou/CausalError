# MAST Causal Pipeline — Reference

> One-stop reference for the MAST-AG2 causal graph construction + intervention validation pipeline. Use instead of re-reading code every time.

**Module path**: `/data/wang/junh/githubs/MAST/causal_graph/` (lives in the sibling MAST repo; doc kept here because MAST blocked writes during generation)

**Compiled**: 2026-05-16 by 6 parallel sonnet agents (M1 preprocess, M2 Suppes, M3 CAPRI, M5 hierarchy, M6 orchestrator, M7 intervention validation). M4 (bootstrap + shuffle) is documented inline rather than a separate module — MAST DOES run bootstrap (unlike TRAIL's `_AIC` config which skips it).

**Companion docs**:
- `CAUSAL_PIPELINE_REFERENCE.md` — the TRAIL analog
- `MAST_EVAL_PIPELINE_REFERENCE.md` — Stage-3 (LLM eval) for MAST
- `MAST_WHO_AND_WHEN_ADOPTION_REFERENCE.md` — W&W adoption for MAST

---

## TL;DR — what eval actually reads

| File | Source module | Eval-time role |
|---|---|---|
| `causal_graph/outputs/suppes_graph.json` | M2 | candidate edges + per-edge `precedence` / `pr_delta`; consumed by `load_graph_edges` for `--corr_threshold` and `--edge_threshold` |
| `causal_graph/outputs/interventions/effect_edges.json` | M7 | 11 intervention-validated edges (`validated=True`) at τ=0.15; the causal anchor in `--causal_only` mode |

Everything else (`capri_graph.json`, `hierarchy_levels.json`, `edge_stability.json`, `controls_shuffle.json`, `eligible_traces.json`) is intermediate or analytical — NOT read by MAST eval scripts.

## Headline edge counts

| Variant | Edges | Source |
|---|---|---|
| `--causal_only` | **11** | `effect_edges.json`, filter `validated=True` (NB: TRAIL has 12) |
| `--corr_threshold 0.60` (union) | ~18 | Suppes geomean ≥ 0.60 ∪ validated causal |
| `--corr_threshold 0.50` (union) | ~25 | Suppes geomean ≥ 0.50 ∪ validated causal |
| `--corr_threshold 0.40` (union) | ~29 | Suppes geomean ≥ 0.40 ∪ validated causal |
| `--random_edges --random_n 11` | 11 | random non-Suppes pairs (count-matched to causal-only) |

## MAST vs TRAIL — key differences

| | TRAIL | MAST |
|---|---|---|
| Traces | 148 (GAIA+SWE) | **393** (AG2) |
| Taxonomy modes | 19 leaves (hierarchical) | **13 codes** (flat, e.g. `1.1`, `2.6`) |
| Suppes graph edges | 27 (`_AIC` config) | **43** |
| CAPRI graph edges | 13 (AIC) | **23 AIC** (or 14 BIC) |
| Validated causal edges | 12 (92.3% of candidates) | **11 (47.8% of 23 candidates)** |
| Bootstrap+shuffle | skipped | **run** (`n_bootstrap=100`, `n_shuffles=50`) |
| Hierarchy | 5 levels / 12 nodes | **3 levels / 11 nodes** |
| Dominant Judge-B label | `disappeared` (57.5%) | **`not_observable` (~70%)** |
| Span/location prediction | yes (`+SI`) | **none** |
| Splits | GAIA_dedup / SWE_Bench_dedup | none (one dataset) |

---

## Pipeline overview

```
┌─────────────────── Stage 1: Graph construction (causal_graph/) ───────────────────┐
│                                                                                    │
│  M1 preprocess          M1 build_order_pairs    M2 suppes_screen   M3 capri_prune │
│  ag2_build_gt.py        CAPRI/1_build_order_    CAPRI/2_suppes_    CAPRI/3_capri_ │
│  ag2_to_onsets.py        pairs.py                screen.py          prune.py       │
│        │                       │                       │                 │         │
│  AG2 raw                 onsets.jsonl ──► order_pairs.jsonl ──► suppes_graph.json ──► capri_graph.json
│  annotation +                                                          │                 │
│  data/gt/ (393                                                  [eval consumes]  [unused at eval]
│  per-trace files)                                                                        │
│                                                                                          │
│            M4a 4_bootstrap_stability ──► edge_stability.{csv,json}  [used by M5 only]   │
│            M4b 5_shuffle_control     ──► controls_shuffle.json      [diagnostic only]   │
│                                                                                          │
│                    M5 6_export_hierarchy ──► hierarchy_levels.json  [viz only]          │
│                                                                                          │
│             Orchestrated by M6 run_causal_pipeline.py (steps 0-6)                       │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────── Stage 2: Intervention validation (causal_valid/) ─────────────────┐
│                                                                                    │
│  capri_graph.json ──┐                                                              │
│                     ▼                                                              │
│   M7 run_pipeline.py ──► 7-step do(A=0) intervention pipeline:                    │
│        Step 0  filter_traces      ──► eligible_traces.json                        │
│        Step 1  case_builder       ──► a_instances.jsonl, edge_pairs.jsonl         │
│        Step 2  patch_generator    ──► patch_results.jsonl                         │
│        Step 3  rerun_harness      ──► rerun_results.jsonl                         │
│        Step 4  judge_a_resolved   ──► a_resolved.jsonl                            │
│        Step 5  judge_b_effect     ──► b_effect.jsonl                              │
│        Step 6  effect_aggregator  ──► effect_edges.json  ◄── consumed by eval    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Module 1: Preprocess + Order Pairs

**Purpose**: Convert AG2 annotated traces into per-trace ground-truth JSON, fixed-vocabulary onsets, and pairwise temporal-precedence pairs.

**Code**:
- `causal_graph/ag2_build_gt.py`
- `causal_graph/ag2_to_onsets.py`
- `causal_graph/CAPRI/1_build_order_pairs.py`

### Outputs

| File | Records | Schema |
|---|---|---|
| `data/gt/{idx:04d}.json` (393 files) | one per trace, keyed by row index | `{errors: [{category, location}]}` — only first occurrence per category |
| `data/onsets.jsonl` | 393 | `{trace_id, present: {cat: 0/1}, onset: {cat: step_idx}, count: {cat: n}}` — fixed-width 13-category vectors |
| `data/order_pairs.jsonl` | 393 | `{trace_id, pairs: [[A, B], ...]}` — A strictly precedes B; ties dropped |

### Key CLI args / defaults
- `ag2_build_gt.py`: `--input` (`../data/annotation/annotation_ag2_filtered.jsonl`), `--out_dir` (`data/gt`)
- `ag2_to_onsets.py`: `--input` (same), `--out_path` (`data/onsets.jsonl`)
- `CAPRI/1_build_order_pairs.py`: `--in_path` (default `data/derived/onsets.jsonl` — **stale, points to a dir that doesn't exist**; orchestrator overrides to `data/onsets.jsonl`)

### Production intermediate statistics

| Stat | Value |
|---|---|
| AG2 traces | **393** |
| GT files with ≥1 annotation | 375 |
| GT files with no annotations | 18 |
| Distinct leaf categories | **13** (all of `1.1–1.5`, `2.1–2.4`, `2.6`, `3.1–3.3`) |
| Records in `onsets.jsonl` / `order_pairs.jsonl` | 393 |
| Total order pairs | **1,347** |
| Avg pairs/trace | **3.4** (vs TRAIL's higher count from 8.4-step traces) |

**Per-category presence rate** (top 5):
| Cat | Rate |
|---|---|
| 1.3 Step Repetition | 52.7% |
| 2.6 Action-Reasoning Mismatch | 49.6% |
| 1.5 Unaware of Termination Conditions | 37.9% |
| 2.2 Fail to Ask for Clarification | 34.1% |
| 1.1 Disobey Task Specification | 33.6% |

Rarest: `1.2` (1.5%), `2.4` (3.1%).

### Gotchas
- **Category `2.5` divergence**: `ag2_to_onsets.py` hardcodes only 13 categories in `MAST_CATEGORIES` (omits `2.5`), but `ag2_build_gt.py` doesn't filter — so any `2.5=1` annotation appears in `data/gt/` but is silently excluded from `onsets.jsonl`.
- **Ties dropped silently** in `order_pairs.jsonl` (no warning, no counter)
- **`gt/` indexed by row position**, not `trace_id` (`trace_id` is non-contiguous: 3, 5, 7, ...)
- **Stale default path**: `1_build_order_pairs.py` defaults reference `data/derived/` which doesn't exist; only safe to invoke via orchestrator

---

# Module 2: Suppes Screen

**Purpose**: Apply Suppes probabilistic causation (temporal precedence + probability raising + min support) to candidate (A,B) pairs.

**Code**: `causal_graph/CAPRI/2_suppes_screen.py`

### Output: `suppes_graph.json`

```json
{
  "params": {"min_precedence": 0.55, "min_pr_delta": 0.05, "min_joint": 3, "in_path": "data/onsets.jsonl"},
  "n_traces": 393, "n_modes": 13, "n_edges": 43,
  "edges": [...]
}
```

Note: MAST uses **short AG2 codes** (`"1.4"`, `"2.1"`) as node labels, not human-readable strings.

### Key CLI args / defaults
Three flags with **stale help strings** — actual defaults at lines 37-43:
- `--min_precedence` (**0.55**; help says 0.6)
- `--min_pr_delta` (**0.05**; help says 0.02)
- `--min_joint` (**3**; help says 30)

Survival rule: `precedence_n ≥ min_joint` AND `precedence ≥ min_precedence` AND `pr_delta ≥ min_pr_delta`.

### Production intermediate statistics

| Stat | Value |
|---|---|
| Total Suppes edges | **43** |
| Edges with geomean ≥ 0.60 | 8 |
| Edges with geomean ≥ 0.50 | 15 |
| Edges with geomean ≥ 0.40 | 23 |
| Edges with geomean ≥ 0.30 | 32 |
| Edges with geomean ≥ 0.20 | 43 (all) |

Geomean = `sqrt(precedence × pr_delta)` is **not computed here** — done at eval time by `load_graph_edges`.

---

# Module 3: CAPRI Prune

**Purpose**: Greedy hill-climbing over BIC/AIC-penalized DAG score; one move per iteration; acyclicity enforced via DFS.

**Code**: `causal_graph/CAPRI/3_capri_prune.py`

### Output: `capri_graph.json`

Top-level: `params, n_traces, n_modes, suppes_n_edges, pruned_n_edges, score_history, edges`. Per-edge: `{a, b}` only — Suppes scores NOT carried forward.

### Key CLI args / defaults
- `--criterion` (`AIC` — production default; `BIC` also available)
- `--max_parents` (`None` — no cap)
- `--max_iters` (`500`)

### Algorithm
For each move (add / remove / reverse): compute local-score change at affected nodes only (incremental cache); accept largest improvement. Local score `b` with parents `pa`: penalized log-likelihood of binary CPT with Laplace smoothing (pseudo-count 0.5). `k_params = 2^|pa|`. Penalty: `k·log(n)` (BIC) or `2k` (AIC).

### Production intermediate statistics

| File | Criterion | Suppes → CAPRI | Accepted moves | Initial → Final score |
|---|---|---|---|---|
| `capri_graph.json` (= `capri_graph_aic.json`, byte-identical) | **AIC** | 43 → **23** | 25 | 4967.03 → 4480.25 |
| `capri_graph_BIC.json` | BIC | 43 → **14** | 14 | 5018.69 → 4680.61 |

AIC's weaker penalty (2k) retains substantially more edges than BIC (k·log(393)≈5.97k). Production uses AIC.

### Status at eval time

**NOT consumed by eval.** Both `eval/full_run_eval_with_graph.py` and `eval/full_run_eval_graph_inject.py` define:
```python
DEFAULT_EFFECT_EDGES = _GRAPH_DIR / "interventions" / "effect_edges.json"
DEFAULT_SUPPES_GRAPH  = _GRAPH_DIR / "suppes_graph.json"
```
Neither references `capri_graph.json`. The CAPRI graph is only Stage-2 (M7) input and M5 input.

---

# Module 4 (inline): Bootstrap + Shuffle (MAST runs both)

Unlike TRAIL (skipped), MAST **runs both** Module 4a (`4_bootstrap_stability.py`, `n_bootstrap=100`) and Module 4b (`5_shuffle_control.py`, `n_shuffles=50`). Outputs:
- `edge_stability.csv` + `edge_stability.json` — per-edge bootstrap-survival fraction; **consumed by M5** stability filter
- `controls_shuffle.json` — shuffle-null edge counts; diagnostic only

Neither is read by eval scripts.

---

# Module 5: Hierarchy Export

**Purpose**: Topological-depth stratification of the CAPRI DAG — visualization only.

**Code**: `causal_graph/CAPRI/6_export_hierarchy.py`

### Output: `hierarchy_levels.json`

```json
{
  "params": {..., "stability_threshold": 0.3},
  "n_levels": 3, "n_nodes": 11,
  "levels": {
    "level_0": ["1.1", "2.1", "2.2"],
    "level_1": ["1.3", "1.4", "2.3", "2.6", "3.3"],
    "level_2": ["1.5", "2.4", "3.1"]
  }
}
```

**3 levels / 11 nodes**. One of 13 MAST categories has no surviving edges and is invisible to the level loop.

### Algorithm
Filter edges below `--stability_threshold 0.3`; cycle-break by lowest-stability edge; BFS level propagation; orphan bulk-assign fallback.

### Status at eval time
NOT consumed by any eval/baselines script. Only `run_causal_pipeline.py` (writes it) and `visualize_graph.py --hierarchy` (column layout).

---

# Module 6: Pipeline Orchestrator

**Code**: `causal_graph/run_causal_pipeline.py` + `causal_graph/workflow.md` (README)

### Key CLI args / defaults
| Flag | Default |
|---|---|
| `--input` | `../data/annotation/annotation_ag2_filtered.jsonl` |
| `--data_dir` | `data` |
| `--out_dir` | `outputs` |
| `--min_precedence` / `--min_pr_delta` / `--min_joint` | 0.55 / 0.05 / 3 |
| `--criterion` | **`AIC`** |
| `--max_parents` | `None` |
| `--n_bootstrap` | 100 |
| `--seed` | 42 |
| `--n_shuffles` | 50 |
| `--skip_shuffle` | False |
| `--stability_threshold` | 0.3 |
| `--start_step` | 0 (resume from step N: 0=onset, 1=order_pairs, ..., 6=hierarchy) |

**No `--skip_bootstrap` flag** (TRAIL has one) — use `--start_step` to bypass.

### Execution sequence
Steps 0→6 launched as subprocesses with cwd = `causal_graph/`. M1 failure is non-fatal; M2 onward is fatal. The driver does **NOT** invoke Module 7 — that's run separately via `causal_valid/run_pipeline.py`.

### Output directory layout (post full run)
```
outputs/
  suppes_graph.json              (M2)
  capri_graph.json (= _aic)      (M3)
  capri_graph_BIC.json           (M3 alt run)
  capri_graph.jso                ← stale partial-write artifact (harmless)
  edge_stability.{csv,json}      (M4a)
  controls_shuffle.json          (M4b)
  hierarchy_levels.json          (M5)
  graph_causal.png               (visualize_graphs.py)
  aic/                           ← earlier viz archive (not pipeline output)
  interventions/                 (M7 AIC run; default eval target)
  interventions_BIC/             (M7 BIC parallel experiment)
  openai-gpt-4o-baseline/        ← Stage-3 eval lands here when --output_dir=outputs
  mistralai-...-baseline/        ← same
```

`outputs/` is **shared** between Stage-1 pipeline outputs and Stage-3 eval outputs.

### Visualization sidecars
- `visualize_graph.py` (singular): CAPRI bootstrap graph; stability-weighted; default `causal_graph.png`
- `visualize_graphs.py` (plural): intervention-validated graph; `|Δ(A→B)|`-weighted; renders `graph_causal.png` (production figure); optional `graph_corr_t<τ>.png` for corr-union

---

# Module 7: Intervention Validation (`causal_valid/`)

**Purpose**: Do(A=0) intervention experiments. Produces `effect_edges.json` — the validated-causal anchor consumed by eval.

**Code**: `causal_graph/causal_valid/`
**Top-level driver**: `run_pipeline.py`

### 7-step pipeline

| Step | Script | Output | Key behavior |
|---|---|---|---|
| 0 | `filter_traces.py` | `eligible_traces.json` | filter to traces with annotations on candidate A-categories |
| 1 | `case_builder.py` | `a_instances.jsonl`, `edge_pairs.jsonl`, `intervention_location_conflicts.jsonl` | turn each (trace, error) into `AInstanceRecord` + `EdgePair`s |
| 2 | `patch_generator.py` (active) — `_llm.py` legacy | `patch_results.jsonl`, `postcheck_failures.jsonl` | LLM-generates patches with rule-based postcheck + retries |
| 3 | `rerun_harness.py` | `rerun_results.jsonl` | apply patch via `replace_step_content`; **LLM-simulate** downstream steps (no live agent re-execution) |
| 4 | `judge_a_resolved.py` | `a_resolved.jsonl` | Judge-A: was A actually removed? |
| 5 | `judge_b_effect.py` | `b_effect.jsonl` | Judge-B: 8 labels (`disappeared`, `delayed`, `unchanged`, `earlier`, `weakened`, `strengthened`, `emerged`, `not_observable`) |
| 6 | `effect_aggregator.py` | **`effect_edges.json`** | Δ(A→B) per edge; validation; placebo |

### `effect_edges.json` schema

Top-level: `edges` (dict keyed by `"A -> B"`), `placebo` (`null_delta_mean`, `null_delta_std`, `n_placebo_samples`), `patch_failure_by_category`, `validation_threshold`, `min_n`.

Per-edge: `a`, `b`, `n_valid_interventions`, `patch_failure_rate`, `b_present_baseline_rate`, `b_present_rerun_rate`, `delta`, `placebo_mean`, `placebo_std`, `effect_label_distribution`, `validated`, `in_capri_graph`.

### Production intermediate statistics (AIC variant)

| | Value |
|---|---|
| Candidate edges | **23** |
| Validated edges | **11 (47.8%)** at threshold=0.15, min_n=1 |
| `n_valid_interventions`: min/median/max/total | 0 / 8 / 81 / 396 |
| 6 edges have n=0 | categories `2.1`, `2.3→1.4`, `2.3→2.4` had no eligible interventions |
| Mean patch failure rate | **~4.6%** (range 0.0%–11.1%, 8 active A-categories) |
| Placebo null | mean=−0.124, std=0.252, n=1700 |

**Strongest validated effect**: `2.6 → 3.1` (Δ = -0.833). **Weakest validated**: `1.1 → 2.6` (Δ = -0.161).

**Aggregated Judge-B labels across validated edges**:
| Label | Count | (% of ~340) |
|---|---|---|
| `not_observable` | ~231 | **~70%** |
| `disappeared` | ~79 | ~23% |

Strikingly different from TRAIL (`disappeared` 57.5%). Interpretation: MAST uses LLM-simulated reruns (no live agent re-execution), so Judge-B often cannot determine whether B happened.

### BIC variant (alternative)
`interventions_BIC/effect_edges.json` has 13 candidate edges, 7 validated — but uses `validation_threshold=0.0` (any negative Δ), so edge-count comparison vs AIC is not apples-to-apples.

### Eval connection
All four production eval scripts hardcode:
```python
DEFAULT_EFFECT_EDGES = _GRAPH_DIR / "interventions" / "effect_edges.json"
```
`_parse_causal_graph` filters `validated=True` → 11 edges used as the causal anchor.

---

# Quick-reference

## File → module map

| File | Module |
|---|---|
| `ag2_build_gt.py`, `ag2_to_onsets.py`, `CAPRI/1_build_order_pairs.py` | M1 |
| `CAPRI/2_suppes_screen.py` | M2 |
| `CAPRI/3_capri_prune.py` | M3 |
| `CAPRI/4_bootstrap_stability.py`, `CAPRI/5_shuffle_control.py` | M4 (inline) |
| `CAPRI/6_export_hierarchy.py` | M5 |
| `run_causal_pipeline.py`, `workflow.md`, `visualize_graph*.py` | M6 |
| `causal_valid/` (everything) | M7 |

## What's not used at eval

- `capri_graph.json`, `capri_graph_aic.json`, `capri_graph_BIC.json` — Stage-2 input only
- `hierarchy_levels.json` — visualization only
- `edge_stability.{csv,json}`, `controls_shuffle.json` — M5 input + diagnostic
- `eligible_traces.json` — Stage-2 internal
- `data/gt/`, `data/onsets.jsonl`, `data/order_pairs.jsonl` — Stage-1 internal

## Latent bugs / inconsistencies

1. **Category `2.5` divergence** between `ag2_build_gt.py` and `ag2_to_onsets.py`
2. **Stale defaults in `1_build_order_pairs.py`** point to non-existent `data/derived/`
3. **Stale help strings** in `2_suppes_screen.py` (0.6/0.02/30 vs actual 0.55/0.05/3)
4. **Stale partial-write artifact**: `outputs/capri_graph.jso` (filename truncated, harmless)
5. **Eval output dirs mixed** with Stage-1 outputs under `outputs/`
