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
