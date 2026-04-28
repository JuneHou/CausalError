# EMNLP Main Table — Experiment TODO

All commands run from `benchmarking/` unless noted.
Target directory for all new runs: `outputs/zero_shot2/`

---

## Remaining: Gemini-2.5-Flash — GAIA_dedup (zero_shot2 run)

zero_shot/ has all 4 GAIA conditions. zero_shot2/ only has the no-graph baseline.
Need the 3 graph conditions in zero_shot2/ to complete the averaging pair.

```bash
# [A] Causal-only graph
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --causal_only \
    --output_dir outputs/zero_shot2

# [B] Causal-only + span index
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [C] Graph inject (causal + corr>=0.2) + span index
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2
```

---

## New: Gemini-2.5-Pro — GAIA_dedup + SWE_Bench_dedup

Run the two bookend conditions (no-graph and best graph) on both splits.
This is the minimal set to establish whether Pro benefits from graph injection
and how it compares to Flash. Add intermediate conditions only if results are
surprising or if a full Pro row is needed in the main table.

Model ID (litellm): `gemini/gemini-2.5-pro`

### GAIA_dedup

```bash
# [P1] No-graph baseline
python eval/run_eval.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --output_dir outputs/zero_shot2

# [P2] Graph inject (causal-only) + span index  ← best condition
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2
```

### SWE_Bench_dedup

```bash
# [P3] No-graph baseline
python eval/run_eval.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --output_dir outputs/zero_shot2

# [P4] Graph inject (causal-only) + span index  ← best condition
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2
```

### Optional (if full Pro row needed in main table)

```bash
# [P5] GAIA_dedup causal-only
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only \
    --output_dir outputs/zero_shot2

# [P6] GAIA_dedup causal-only + span index
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [P7] SWE_Bench_dedup causal-only
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only \
    --output_dir outputs/zero_shot2

# [P8] SWE_Bench_dedup causal-only + span index
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-pro \
    --split SWE_Bench_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2
```

---

## Observational Graph Threshold Experiments (new geomean scoring)

The non-causal graph path now uses `score = sqrt(P(B|A) × PR_delta)` (geometric mean)
as both the injected edge weight and the filter criterion.
Both MAST and TRAIL now use **geomean >= threshold** consistently for observational edges.

### Edge landscape (from suppes_graph.json)

| geomean threshold | N edges | Categories covered |
|---|---|---|
| causal_only (12 edges) | 12 | 11/20 |
| >= 0.20 | 18 | 12/20 (+Environment Setup Errors) |
| >= 0.10 (current default) | 21 | 13/20 (+Environment Setup Errors, +Task Orchestration) |

7 categories are structurally uncoverable at any threshold (no co-occurrence data):
Instruction Non-compliance, Tool Definition Issues, Rate Limiting,
Service Errors, Resource Not Found, Resource Exhaustion, Timeout Issues.

Recommended threshold: **geomean >= 0.20** (18 edges, 12/20 cats) as the primary
experiment — strong associations only, analogous to MAST geomean >= 0.20.
Also run geomean >= 0.10 as the full-coverage comparison (adds Task Orchestration
at the cost of extra weak edges).

### T-Obs-1 — Gemini-Flash GAIA_dedup: +CG observational, two thresholds

```bash
# [O1a] +CG observational, geomean >= 0.20  ← primary
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --edge_threshold 0.20 \
    --output_dir outputs/zero_shot2

# [O1b] +CG observational + span index, geomean >= 0.20
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --edge_threshold 0.20 --span_index \
    --output_dir outputs/zero_shot2

# [O1c] +CG observational, geomean >= 0.10  ← full-coverage comparison
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --edge_threshold 0.10 \
    --output_dir outputs/zero_shot2
```

Expected output dirs (in `outputs/zero_shot2/`):
- `outputs_gemini-gemini-2.5-flash-GAIA_dedup-graph_suppes_t0.2/`
- `outputs_gemini-gemini-2.5-flash-GAIA_dedup-graph_suppes_t0.2_span_index/`
- `outputs_gemini-gemini-2.5-flash-GAIA_dedup-graph_suppes_t0.1/`

---

### T-Obs-2 — Gemini-Flash GAIA_dedup: +GI mixed (causal + observational)

TRAIL's graph inject scripts support `--corr_threshold` to combine validated
causal edges with suppes edges above a secondary threshold.  This is the most
powerful mode and has no MAST analog.

```bash
# [O2a] +GI causal + corr >= 0.20 + span index  ← recommended
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --corr_threshold 0.20 --span_index \
    --output_dir outputs/zero_shot2

# [O2b] +GI observational-only, geomean >= 0.20
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA_dedup \
    --edge_threshold 0.20 \
    --output_dir outputs/zero_shot2
```

Expected output dirs:
- `outputs_gemini-gemini-2.5-flash-GAIA_dedup-graph_inject_causal_corr0.2_span_index/`
- `outputs_gemini-gemini-2.5-flash-GAIA_dedup-graph_inject_suppes_t0.2/`

---

### T-Obs-3 — Mistral GAIA_dedup: +CG observational (geomean >= 0.20)

Run only after T-Obs-1 confirms that t=0.20 is the right threshold for Flash.
Mirrors the MAST T10 design (Mistral geomean >= 0.20).

```bash
# [O3a] +CG observational, geomean >= 0.20
CUDA_VISIBLE_DEVICES=<gpus> python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup \
    --edge_threshold 0.20 \
    --output_dir outputs/zero_shot2

# [O3b] +GI causal + corr >= 0.20 + span index
CUDA_VISIBLE_DEVICES=<gpus> python eval/run_eval_graph_inject_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup \
    --corr_threshold 0.20 --span_index \
    --output_dir outputs/zero_shot2
```

---

## Scoring

```bash
# From benchmarking/
python eval/calculate_scores.py --results_dir outputs/zero_shot2
```

---

## Completed (for reference)

| Run | Status |
|-----|--------|
| Flash GAIA all 4 conditions (zero_shot/) | ✓ |
| Flash GAIA_dedup no-graph (zero_shot2/) | ✓ |
| Flash SWE no-graph, causal_only (zero_shot/) | ✓ |
| Flash SWE causal_only+SI, graph_inject+SI (zero_shot2/, 31 traces) | ✓ |
| Mistral GAIA_dedup all 4 conditions (zero_shot2/) | ✓ |
| Mistral SWE_Bench_dedup no-graph + graph_inject+SI (zero_shot2/) | ✓ |
