# Causal Dependency Exploration on TRAIL: Plan

This document analyzes whether your REM_MAST causal discovery pipeline can be applied to TRAIL, what adaptations are needed, and what is missing.

---

## 1. Can the Pipeline Be Applied? Yes, with Adaptations

Your pipeline (Suppes + CAPRI) is **well-suited** to TRAIL because:

- **Same causal question**: Do certain failure types tend to precede others (temporal priority) and raise the probability of later failures (probability raising)?
- **Same output structure**: Onset table `{trace_id, present, onset}` → Suppes graph → CAPRI pruning → hierarchy.
- **TRAIL advantage**: Human-annotated onset locations (span_id) already exist. You **skip the segment-labeling step (vLLM)**—no LLM needed to localize failures.

The main work is mapping TRAIL’s format into the onset format your downstream scripts expect.

---

## 2. REM_MAST vs TRAIL: Data Comparison

| Pipeline Stage      | REM_MAST (MAD)                          | TRAIL                                       |
|---------------------|------------------------------------------|---------------------------------------------|
| **Input**           | MAD traces: `mas_name`, `trajectory`, `mast_annotation` | Trace JSON (spans) + annotation JSON (errors with location=span_id) |
| **Filter**          | By `mas_name` (AG2, AppWorld, …)         | By split: GAIA vs SWE Bench                  |
| **Segments**        | Text chunks from trajectory (JSON/role/char) | Candidate spans (e.g. OpenInference LLM/TOOL/CHAIN) with span_id, timestamp |
| **Segment index t** | 0, 1, 2, … from segmentation             | Rank from candidate-span ordering (timestamp, tie-break span_id); coverage report for missing annotated span_ids |
| **Labels**          | vLLM labels each segment for failure IDs | **Not needed**: annotations give (category, location=span_id) |
| **Onset**           | Earliest segment index t per failure     | Rank of span_id for each error category     |
| **Present**         | Trace-level `mast_annotation`            | Derived from annotations (which categories occur) |
| **Taxonomy**        | MAST IDs (1.1, 1.2, 2.3, …)             | TRAIL categories (Instruction Non-compliance, Tool-related, Goal Deviation, …) |

---

## 3. What Is Missing or Different in TRAIL

### 3.1 Candidate-span timeline (not “all spans”)

TRAIL gives `span_id` as location, not a segment index. You need a **deterministic ordering** of spans to define onset rank. You have two legitimate choices; **pick one and state it in the paper**.

- **Option 2A (recommended): Timeline = OpenInference action spans**  
  Define **candidate spans** as those where  
  `span_attributes["openinference.span.kind"] ∈ {"LLM", "TOOL", "CHAIN"}`.  
  Then order by timestamp (tie-break by `span_id`) and assign ranks.  
  **Why this is good:** These correspond to agent actions (LLM call, tool call, step wrapper) → good intervention hooks. Rich timeline without low-level wrappers. If other agent actions exist, add them as a new key.

- **Option 2B: Timeline = all spans**  
  Use every span in the trace (e.g. recursive collect + timestamp order). Simpler but noisier.

**Recommendation:** Option 2A. In Step 2 you produce `span_rank` for the **candidate set** and a **coverage report**: which annotated `span_id`s are missing from that set (see Step 2 output below).

**When annotated span_ids are missing from the candidate set**, choose one policy (state it in the paper):

1. **Fallback-to-all-spans ranking for that trace only** (robust; recommended for causality work).
2. **Fallback-to-parent span**: if the annotation points to a child span not in the candidate set, use the parent’s rank if the parent is a candidate.
3. **Drop that annotation**: only if extremely rare and reported.

### 3.2 Multiple errors at same location

TRAIL can have multiple error categories at the same `span_id` (e.g. Tool-related and Goal Deviation both at `bc20feefb97e11e5`). They share the same onset rank—that is acceptable for Suppes/CAPRI.

### 3.3 Taxonomy

TRAIL uses its own taxonomy (21 categories). Use these category names as failure IDs; no mapping to MAST IDs needed.

---

## 4. Adaptation Plan: TRAIL-Specific Pipeline

### 4.1 New TRAIL adapter scripts

| Script                     | Role                                                         |
|----------------------------|--------------------------------------------------------------|
| `causal_explore/preprocess/trail_1_filter_split.py`  | Build trace + annotation pairs from HF split; output `trail_filtered/{split}.jsonl` with `trace_id`, `split`, `trace_path`, `annotation_path`, plus sanity fields: `n_errors`, `error_locations` (set of span_ids in annotations). Enables immediate detection of annotations referring to span_ids not in the trace. |
| `causal_explore/preprocess/trail_2_build_span_order.py` | Build **candidate-span** timeline (Option 2A: OpenInference action spans LLM/TOOL/CHAIN); timestamp order, tie-break by span_id. Output `span_rank` and **coverage**: `missing_annotated_span_ids` per trace. Apply chosen fallback when missing non-empty (e.g. fallback-to-all-spans for that trace). |
| `causal_explore/preprocess/trail_3_build_onsets.py`  | Merge annotations + span_rank → onset table. `present[A]`, `onset[A]`, plus `count[A]` (# occurrences) and optional `ties[A,B]` (categories sharing same span). REM_MAST-compatible. |

### 4.2 Reuse existing scripts (unchanged)

| Script                     | Input                          |
|----------------------------|--------------------------------|
| `causal_explore/CAPRI/1_build_order_pairs.py`   | Onsets from `trail_3_build_onsets.py` |
| `causal_explore/CAPRI/2_suppes_screen.py`       | Same onsets                    |
| `causal_explore/CAPRI/3_capri_prune.py`         | Same                           |
| `causal_explore/CAPRI/4_bootstrap_stability.py` | Same                           |
| `causal_explore/CAPRI/5_shuffle_control.py`     | Same                           |
| `causal_explore/CAPRI/6_export_hierarchy.py`   | Same                           |

### 4.3 Optional TRAIL adaptations

- **Stratified runs**: Run pipeline separately for GAIA vs SWE Bench to compare causal structure.
- **Split by agent type**: GAIA = multi-agent; SWE Bench = single-agent. Stratify if you want to study differences.
- **Category pruning**: Exclude very rare TRAIL categories if they cause noise in Suppes (e.g. `min_joint`).

---

## 5. Data Flow

```
TRAIL data/
├── benchmarking/data/GAIA/*.json           (traces)
├── benchmarking/data/SWE Bench/*.json      (traces)
├── benchmarking/processed_annotations_gaia/*.json
└── benchmarking/processed_annotations_swe_bench/*.json

Step 1: causal_explore/preprocess/trail_1_filter_split.py
  Input: HF split rows (or equivalent: trace dir + annotation dir per split).
  → data/trail_filtered/{split}.jsonl  (e.g. gaia.jsonl, swe_bench.jsonl)
  Each line:
  {
    "trace_id": "...",
    "split": "GAIA",
    "trace_path": "...",
    "annotation_path": "...",
    "n_errors": 3,
    "error_locations": ["sid1", "sid2", "sid3"]
  }
  Why: n_errors and error_locations let you detect if a trace’s annotations refer to span_ids not present in the trace JSON.

Step 2: causal_explore/preprocess/trail_2_build_span_order.py
  Build candidate-span timeline (Option 2A: openinference.span.kind ∈ {LLM, TOOL, CHAIN}); order by timestamp, tie-break by span_id.
  → data/trail_span_order/gaia.jsonl (or swe_bench.jsonl)
  Each line:
  {
    "trace_id": "...",
    "split": "GAIA",
    "span_rank": {"sid1": 0, "sid2": 1, ...},
    "missing_annotated_span_ids": ["sidX", ...]
  }
  If missing_annotated_span_ids is non-empty, apply chosen fallback (e.g. fallback-to-all-spans for that trace).

Step 3: causal_explore/preprocess/trail_3_build_onsets.py
  Merge annotations + span_rank. Onset = min(rank(location_span_id)) per category.
  Output (REM_MAST-compatible, with extras):
  {
    "trace_id": "...",
    "split": "GAIA",
    "present": {"Instruction Non-compliance": 1, "Tool-related": 1, "Goal Deviation": 0, ...},
    "onset": {"Instruction Non-compliance": 5, "Tool-related": 12, ...},
    "count": {"Instruction Non-compliance": 1, "Tool-related": 2, ...},
    "ties": [["A", "B"], ...]   // optional: categories sharing same span
  }
  → data/trail_derived/onsets_gaia.jsonl (or onsets_swe_bench.jsonl)

Step 4–6: causal_explore/CAPRI/ (order pairs, Suppes, CAPRI prune, bootstrap, shuffle, hierarchy)
  Use onsets_gaia.jsonl or onsets_swe_bench.jsonl as input.
  Outputs: suppes_graph, capri_graph, edge_stability, shuffle_control, hierarchy_levels.
```

### 5.1 Running causal graph construction (steps 1–6 in CAPRI)

After you have `data/trail_derived/onsets_gaia.jsonl` (from `run_causal_gaia.sh`), run the causal steps from **benchmarking/** using the helper script. Scripts live in `causal_explore/CAPRI/` (no REM_MAST dependency).

**From `benchmarking/` directory:**

```bash
python run_causal_from_trail_onsets.py --onsets_path data/trail_derived/onsets_gaia.jsonl
```

This runs, in order:

- **Step 1**: `causal_explore/CAPRI/1_build_order_pairs.py` → `data/trail_causal_outputs/order_pairs.jsonl`
- **Step 2**: `2_suppes_screen.py` → `suppes_graph.json`
- **Step 3**: `3_capri_prune.py` → `capri_graph.json`
- **Step 4**: `4_bootstrap_stability.py` (optional) → `edge_stability.csv` / `edge_stability.json`
- **Step 5**: `5_shuffle_control.py` (optional) → `controls_shuffle.json`
- **Step 6**: `6_export_hierarchy.py` → `hierarchy_levels.json`

**Options:**

- `--out_dir data/trail_causal_outputs` (default) to write all outputs there.
- `--min_joint 5` (or 10) if many TRAIL categories are rare (Suppes requires at least this many traces with both A and B).
- `--skip_bootstrap` / `--skip_shuffle` for a quick run without validation.

TRAIL onset rows have `present` and `onset` keys; CAPRI scripts ignore extra keys (`trace_id`, `split`, `count`, `ties`), so the format is compatible.

---

## 6. Implementation Checklist

### Phase 1: Adapters (TRAIL-specific)

- [ ] **causal_explore/preprocess/trail_1_filter_split.py**: Build trace + annotation pairs from HF split; write `trail_filtered/{split}.jsonl` with `trace_id`, `split`, `trace_path`, `annotation_path`, `n_errors`, `error_locations` (set of span_ids from annotation file).
- [ ] **causal_explore/preprocess/trail_2_build_span_order.py**: Build candidate-span timeline (Option 2A: OpenInference kinds LLM, TOOL, CHAIN); sort by timestamp, tie-break by span_id; output `span_rank` and `missing_annotated_span_ids` per trace; implement chosen fallback when missing non-empty (e.g. fallback-to-all-spans for that trace).
- [ ] **causal_explore/preprocess/trail_3_build_onsets.py**: Merge annotations + span_rank; output `present`, `onset`, `count`, optional `ties`; REM_MAST-compatible onset table.

### Phase 2: Integration

- [ ] Run `causal_explore/CAPRI/1_build_order_pairs.py` through `6_export_hierarchy.py` with TRAIL onset paths (via `run_causal_from_trail_onsets.py`).
- [ ] Confirm Suppes/CAPRI parameters (e.g. `min_precedence`, `min_pr_delta`, `min_joint`) are suitable for TRAIL’s size and sparsity.

### Phase 3: Analysis

- [ ] Run pipeline for GAIA and SWE Bench separately.
- [ ] Compare causal graphs and hierarchy levels across splits.
- [ ] Optionally: bootstrap and shuffle controls to validate stability and rule out artifacts.

---

## 7. Expected Outputs

- **Causal graph**: Which TRAIL error categories tend to precede others (e.g. Instruction Non-compliance → Tool-related).
- **Hierarchy levels**: Root failures (level 0) vs downstream (level 1, 2, …).
- **Split comparison**: Whether GAIA (multi-agent) vs SWE Bench (single-agent) differ in causal structure.
- **Stability / controls**: Bootstrap and shuffle results to assess reliability.

---

## 8. Stage II: Causal Intervention (Validation)

Stage I produces a **correlational** causal graph (Suppes + CAPRI). To validate edges with **interventions** (repair/inject/placebo) and produce a weighted causal graph and mitigation policy, see:

- **[CAUSAL_STAGE2_INTERVENTION_PLAN.md](CAUSAL_STAGE2_INTERVENTION_PLAN.md)** — Coding plan for Stage II: how to use failure logs and annotations to replicate scenarios, run interventions, and deliver validated edge set G_C, operator library, controllability scores, and graph-guided mitigation policy.

---

## 9. Summary

| Question                               | Answer                                                                 |
|----------------------------------------|------------------------------------------------------------------------|
| Can the pipeline be applied to TRAIL?  | Yes, with TRAIL-specific adapters for filter, candidate-span timeline, and onsets. |
| What is missing in TRAIL?              | Candidate-span ordering (span_id → rank) and coverage (missing_annotated_span_ids); annotations give locations. |
| Timeline definition?                   | Option 2A (recommended): OpenInference action spans (LLM, TOOL, CHAIN); state choice in the paper. |
| When annotations miss the candidate set? | One policy: fallback-to-all-spans per trace (recommended), fallback-to-parent, or drop (if rare and reported). |
| What must be new?                      | Step 1 with n_errors/error_locations; Step 2 with candidate timeline + coverage + fallback; Step 3 with present/onset/count/optional ties. |
