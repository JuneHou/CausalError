# causal_train: Train-Only Causal Graph Construction & Intervention

This document records the methodology and results for rebuilding the causal graph
using training data only (GAIA train + SWE-bench train), replacing the original
graph which used all GAIA traces including the test split.

---

## Motivation

The original causal graph (`trail_causal_outputs_fully_connected`, `trail_causal_outputs_AIC`)
was built from all 117 GAIA traces, including 13 test-split traces. This causes data leakage:
rare error-type edge weights are influenced by test annotations. Additionally, only GAIA
traces were used despite SWE-bench sharing the same error taxonomy.

---

## Changes vs. Original Pipeline

| Aspect | Original | Train-only (this folder) |
|--------|----------|--------------------------|
| GAIA traces | 117 (train + val + test) | 92 (train only) |
| SWE-bench traces | 0 | 24 (train only) |
| Total traces | 117 | 116 |
| Test split excluded | No | Yes |
| Datasets | GAIA only | GAIA + SWE-bench |

---

## Stage I: Graph Construction

### Step 1 — Onset extraction

```bash
bash causal_train/graph/run_causal_train.sh
```

Outputs:
- `benchmarking/data/trail_derived/onsets_gaia_train.jsonl` — 92 GAIA train traces
- `benchmarking/data/trail_derived/onsets_swe_train.jsonl` — 24 SWE-bench train traces
- `benchmarking/data/trail_derived/onsets_combined_train.jsonl` — 116 traces combined

### Step 2 — Correlation graph (for GNN adjacency)

```bash
python causal_train/graph/run_causal_from_trail_onsets.py \
    --onsets_path benchmarking/data/trail_derived/onsets_combined_train.jsonl \
    --out_dir benchmarking/data/trail_causal_outputs_train_fc \
    --min_precedence 0.0 --min_pr_delta 0.0 --min_joint 1 \
    --skip_bootstrap --skip_shuffle
```

Output: `benchmarking/data/trail_causal_outputs_train_fc/suppes_graph.json`
- 116 traces, **150 edges** (vs. 156 in original fully-connected graph)

### Step 3 — Causal graph (CAPRI-AIC, for intervention)

```bash
python causal_train/graph/run_causal_from_trail_onsets.py \
    --onsets_path benchmarking/data/trail_derived/onsets_combined_train.jsonl \
    --out_dir benchmarking/data/trail_causal_outputs_train_AIC \
    --min_precedence 0.55 --min_pr_delta 0.05 --min_joint 3 \
    --criterion AIC --skip_bootstrap --skip_shuffle
```

Output: `benchmarking/data/trail_causal_outputs_train_AIC/capri_graph.json`
- 116 traces, 22 Suppes edges → **10 CAPRI-AIC edges**

| Edge |
|------|
| Formatting Errors → Context Handling Failures |
| Formatting Errors → Incorrect Problem Identification |
| Formatting Errors → Poor Information Retrieval |
| Formatting Errors → Resource Abuse |
| Formatting Errors → Tool Output Misinterpretation |
| Incorrect Problem Identification → Tool Output Misinterpretation |
| Instruction Non-compliance → Context Handling Failures |
| Task Orchestration → Language-only |
| Tool Selection Errors → Goal Deviation |
| Tool-related → Goal Deviation |

---

## Stage II: Intervention (do(A=0) counterfactual)

### Eligible traces and A-instances

Filter to train+val GAIA traces only (test excluded):

```bash
cd benchmarking/
python ../causal_train/patch/filter_traces.py \
    --annotations_dir processed_annotations_gaia \
    --causal_graph data/trail_causal_outputs_train_AIC/capri_graph.json \
    --out_dir data/trail_causal_outputs_train_AIC \
    --min_errors 2 --strict
```

Results:
- Total GAIA annotated traces: 125
- Eligible (all splits): 77
- **Train+val eligible: 63** (9 test excluded, 5 `old_*` excluded)
- Saved: `data/trail_causal_outputs_train_AIC/eligible_traces_train.json`

Build A-instances (span-level, deduped by intervention location):

| A-type | A-instances |
|--------|-------------|
| Formatting Errors | 96 |
| Tool Selection Errors | 27 |
| Tool-related | 19 |
| Task Orchestration | 7 |
| Incorrect Problem Identification | 6 |
| Instruction Non-compliance | 4 |
| **Total** | **159** |

Saved:
- `data/trail_causal_outputs_train_AIC/a_instances.jsonl` — 159 A-instances
- `data/trail_causal_outputs_train_AIC/edge_pairs.jsonl` — 262 edge pairs

### Incremental run strategy

Reused results from prior runs (train+val traces only, test excluded):

| Source | Reusable patches |
|--------|-----------------|
| `outputs/full_run_gpt4o` | 125 |
| `outputs/full_run_incremental` | 3 (unique, not in gpt4o) |
| **Total reusable** | **128** |
| **Novel (new API calls)** | **31** |

Novel 31 A-instances saved to:
- `data/trail_causal_outputs_train_AIC/a_instances_novel.jsonl`
- `data/trail_causal_outputs_train_AIC/edge_pairs_novel.jsonl`

Novel 31 run:
```bash
python ../causal/patch/run_pipeline.py \
    --trace_dir        data/GAIA \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_train_AIC/capri_graph.json \
    --a_instances_file data/trail_causal_outputs_train_AIC/a_instances_novel.jsonl \
    --edge_pairs_file  data/trail_causal_outputs_train_AIC/edge_pairs_novel.jsonl \
    --patch_library    ../causal/patch/patch_library.json \
    --out_dir          outputs/interventions_train_novel \
    --model            openai/gpt-4o \
    --rerun_model      openai/o3-mini \
    --max_steps_after  12 --max_retries 3 --threshold 0.15 --min_n 1
```

Results: 31 patches, 30 reruns, 27 resolved, 26 b_effect records.

### Final aggregation

Pre-merged all train+val sources (test traces excluded) into:
`outputs/interventions_all_trainval/` — **205 patches, 184 a_resolved, 165 b_effect**

Sources merged:
1. `outputs/full_run_gpt4o` (filtered to train+val)
2. `outputs/full_run_incremental` (filtered to train+val)
3. `outputs/interventions_train_novel` (31 novel, all train+val)

Final aggregation:
```bash
python ../causal_train/patch/run_pipeline.py \
    --trace_dir        data/GAIA \
    --annotations_dir  processed_annotations_gaia \
    --causal_graph     data/trail_causal_outputs_train_AIC/capri_graph.json \
    --a_instances_file data/trail_causal_outputs_train_AIC/a_instances_novel.jsonl \
    --edge_pairs_file  data/trail_causal_outputs_train_AIC/edge_pairs_novel.jsonl \
    --patch_library    ../causal_train/patch/patch_library.json \
    --out_dir          outputs/interventions_train_final3 \
    --merge_from       outputs/interventions_all_trainval \
    --skip_patches --skip_rerun --skip_judge_a --skip_judge_b \
    --threshold 0.15 --min_n 1
```

---

## Stage II Results: Validated Causal Edges

All 10 CAPRI-AIC edges validated (Δ < -0.15, n ≥ 1):

| Edge | n | Δ(A→B) | Validated |
|------|---|--------|-----------|
| Formatting Errors → Context Handling Failures | 14 | -0.214 | YES |
| Formatting Errors → Incorrect Problem Identification | 1 | -1.000 | YES |
| Formatting Errors → Poor Information Retrieval | 1 | -1.000 | YES |
| Formatting Errors → Resource Abuse | 54 | -0.315 | YES |
| Formatting Errors → Tool Output Misinterpretation | 5 | -1.000 | YES |
| Incorrect Problem Identification → Tool Output Misinterpretation | 7 | -0.429 | YES |
| Instruction Non-compliance → Context Handling Failures | 3 | -1.000 | YES |
| Task Orchestration → Language-only | 3 | -0.667 | YES |
| Tool Selection Errors → Goal Deviation | 12 | -0.583 | YES |
| Tool-related → Goal Deviation | 16 | -0.688 | YES |

Placebo null: mean=-0.6684, std=0.2751

Result file: `outputs/interventions_train_final3/effect_edges.json`

---

## Changes vs. Original CAUSAL_EDGES

| Removed (old, 5 edges) | Added (new, 4 edges) |
|------------------------|----------------------|
| Poor Information Retrieval → Resource Abuse | Formatting Errors → Poor Information Retrieval |
| Resource Abuse → Authentication Errors | Instruction Non-compliance → Context Handling Failures |
| Resource Abuse → Tool-related | Task Orchestration → Language-only |
| Task Orchestration → Context Handling Failures | Tool-related → Goal Deviation |
| Tool Selection Errors → Language-only | |

6 edges carried over unchanged from original CAUSAL_EDGES.

Updated in: `graph/build_graph_data.py` → `CAUSAL_EDGES`
