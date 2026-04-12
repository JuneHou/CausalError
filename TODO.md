# EMNLP Main Table — Experiment TODO

All commands run from `benchmarking/` unless noted. The goal is to populate
`outputs/zero_shot2/` using the new 13-edge full GAIA+SWE graph, then average
`zero_shot/` and `zero_shot2/` scores for the final reported numbers.

---

## Step 0 — No setup needed

The eval scripts now load edges directly from the causal JSON files.
No `graph_input.pt`, no torch, no GNN pipeline.

Default paths (already exist):
- Causal edges (13): `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/capri_graph.json`
- Suppes edges (27): `benchmarking/data/trail_causal_outputs_full_gaia_swe_AIC/suppes_graph.json`

Remove `--graph_input` from all commands below — it is no longer a valid flag.
Use `--causal_graph` / `--suppes_graph` only if you want to point at different files.

---

## Step 1 — Run experiments (all from benchmarking/)

All experiments write to `outputs/zero_shot2/`. Each script skips already-
completed files, so partial runs are safe to resume.

### 1a. Gemini-2.5-Flash — GAIA (4 conditions)

```bash
# [1] No-graph baseline
python eval/run_eval.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --output_dir outputs/zero_shot2

# [2] Causal-only graph
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --causal_only \
    --output_dir outputs/zero_shot2

# [3] Causal-only + span index
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [4] Extended graph (causal + correlation edges w>=0.2) + span index
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --causal_only --corr_threshold 0.2 --span_index \
    --output_dir outputs/zero_shot2
```

### 1b. Gemini-2.5-Flash — SWE Bench (4 conditions)

Note: the zero_shot/ no-graph baseline only ran 14/31 traces (incomplete run).
Re-run in zero_shot2/ on all 31 traces to get a clean comparable pair.
Note: `data/SWE Bench_dedup` is available (−80.9% size) if context window errors persist.

```bash
#  DONE [5] No-graph baseline (all 31 SWE Bench traces)
python eval/run_eval.py \
    --model gemini/gemini-2.5-flash \
    --split "SWE Bench" \
    --output_dir outputs/zero_shot2

#  DONE [6] Causal-only graph
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split "SWE Bench" \
    --causal_only \
    --output_dir outputs/zero_shot2

# [7] Causal-only + span index
python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split "SWE Bench" \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [8] Extended graph (causal + correlation edges w>=0.2) + span index
python eval/run_eval_graph_inject.py \
    --model gemini/gemini-2.5-flash \
    --split "SWE Bench" \
    --causal_only --corr_threshold 0.2 --span_index \
    --output_dir outputs/zero_shot2
```

### 1c. Mistral-Small-3.1-24B — GAIA_dedup (3 conditions)

All three conditions use the same GAIA_dedup split so the comparison is fair.
Run [9] is the only missing run needed to complete the Mistral rows in the table.

```bash
# [9] No-graph baseline on GAIA_dedup  ← CRITICAL MISSING RUN
CUDA_VISIBLE_DEVICES=3,4,5,6 python eval/run_eval_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --data_dir data/GAIA_dedup \
    --split GAIA_dedup \
    --output_dir outputs/zero_shot2

# DONE [10] Causal-only graph
CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --data_dir data --split GAIA_dedup \
    --causal_only \
    --tensor_parallel_size 2 \
    --output_dir outputs/zero_shot2

# DONE [11] Causal-only + span index
CUDA_VISIBLE_DEVICES=3,4,5,6 python eval/run_eval_with_graph_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --data_dir data --split GAIA_dedup \
    --causal_only --span_index \
    --output_dir outputs/zero_shot2

# [12] Extended graph (causal + correlation edges w>=0.2) + span index
CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_graph_inject_vllm.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --data_dir data --split GAIA_dedup \
    --causal_only --corr_threshold 0.2 --span_index \
    --tensor_parallel_size 2 \
    --output_dir outputs/zero_shot2
```

---

## Step 2 — Score zero_shot2

```bash
# From benchmarking/
python eval/calculate_scores.py --results_dir outputs/zero_shot2
```

This generates `*-metrics.txt` files alongside each output directory.

---

## Step 3 — Average zero_shot and zero_shot2

After both runs are scored, average wF1, Location Accuracy, and Joint Accuracy
across the two runs for the final reported number. Report mean ± std.

Matching pairs (zero_shot → zero_shot2):

| Condition | zero_shot | zero_shot2 |
|-----------|-----------|------------|
| Gemini GAIA no-graph | 0.3951 | TBD |
| Gemini GAIA causal_only | 0.4277 | TBD |
| Gemini GAIA causal_only + span_index | 0.4218 | TBD |
| Gemini GAIA graph_inject + span_index | 0.4326 | TBD |
| Gemini SWE no-graph | *(incomplete, skip)* | TBD |
| Gemini SWE causal_only | 0.3118 | TBD |
| Gemini SWE causal_only + span_index | TBD | TBD |
| Gemini SWE graph_inject + span_index | TBD | TBD |
| Mistral GAIA_dedup no-graph | *(missing, skip)* | TBD |
| Mistral GAIA_dedup causal_only | 0.2896 | TBD |
| Mistral GAIA_dedup causal_only + span_index | 0.3350 | TBD |
| Mistral GAIA_dedup graph_inject + span_index | TBD | TBD |

---

## Final Main Table Structure

```
GAIA (N=117):

Model                         | No Graph | +Causal | +Causal+SI | +GraphInject+SI
------------------------------|----------|---------|------------|----------------
Gemini-2.5-Flash              | mean±std | mean±std | mean±std  | mean±std
Mistral-Small-24B (GAIA_dedup)| mean±std | mean±std | mean±std  | mean±std

SWE Bench (N=31):

Model                         | No Graph | +Causal | +Causal+SI | +GraphInject+SI
------------------------------|----------|---------|------------|----------------
Gemini-2.5-Flash              | z2 only  | mean±std | mean±std  | mean±std
```

SI = span_index | GraphInject = two-pass graph inject (run_eval_graph_inject*.py, causal + corr≥0.2)
For SWE no-graph: use zero_shot2 only (zero_shot run was incomplete).
