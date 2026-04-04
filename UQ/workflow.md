# UQ Workflow: Error Detection Pipeline for LLM Agent Traces

## Overview

Four pipeline configurations are documented here, from simplest to most capable.
All share the same model (`Tongyi-Zhiwen/QwenLong-L1-32B` via vLLM) and the same
evaluation harness (`benchmarking/eval/calculate_scores.py`).

| Config | W-F1 | Loc Acc | Joint | Script |
|--------|------|---------|-------|--------|
| **1. Zero-shot baseline** | 0.154 | 0.071 | 0.020 | `run_eval_vllm.py` |
| **2. Zero-shot + span index** | 0.180 | 0.126 | 0.035 | `run_eval_vllm.py --span_index` |
| **3. UQ + causal graph only** | 0.137 | 0.044 | 0.013 | `run_uq_eval.py --causal_only` |
| **4. UQ + span index** | 0.165 | 0.135 | 0.027 | `run_uq_eval.py --causal_only --span_index` |
| **5. UQ + span index + graph probe** | **0.206** | **0.135** | 0.024 | `run_uq_eval.py --causal_only --span_index --graph_probe` |
| Gemini zero-shot (reference) | 0.395 | 0.366 | 0.136 | `run_eval.py` |

---

## Config 1 — Zero-Shot Baseline

Single-pass inference. Full taxonomy prompt → model outputs errors + scores JSON.
No graph, no confidence, no second pass.

```
Trace (JSON)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Pass 1 — QwenLong-L1-32B (vLLM)           │
│  Full taxonomy prompt → JSON output         │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Span-ID Gate        │  drop errors whose location is not
         │                     │  a valid 16-char hex span_id
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Output JSON         │
         │ errors + scores     │
         └────────────────────┘
```

### Command

```bash
# From trail-benchmark/
CUDA_VISIBLE_DEVICES=0,3,4,7 python benchmarking/eval/run_eval_vllm.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager
```

Output: `benchmarking/outputs/zero_shot/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA/`

---

## Config 2 — Zero-Shot + Span Index

Same as Config 1, but a compact `span_id → span_name` reference table is prepended
to the prompt. The model can look up the exact hex ID from the table instead of
scanning a 100K-char document.

```
Trace (JSON)
    │
    ├──► build_span_index()
    │    (agent-step spans + direct children, ~24 spans)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Span index block + taxonomy prompt         │
│  Pass 1 — QwenLong-L1-32B (vLLM)           │
│  → JSON output                              │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Span-ID Gate        │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Output JSON         │
         └────────────────────┘
```

**Why it helps**: location accuracy jumped from 0.071 → 0.126 (+77% relative).
The model already produced valid hex format 98% of the time; it was picking the wrong
span. The reference table gives it semantic → hex lookup.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python benchmarking/eval/run_eval_vllm.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --span_index
```

Output: `benchmarking/outputs/zero_shot/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_span_index/`

---

## Config 3 — UQ + Causal Graph Only

Two-pass pipeline with label-token logprob confidence and causal graph propagation.
**This config underperforms the zero-shot baseline** due to compressed confidence
(p50 = 0.984) making graph propagation indiscriminate.

```
Trace (JSON)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Pass 1 — QwenLong-L1-32B                  │
│  logprobs=1 per generated token             │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Span-ID Gate        │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Confidence Proxy    │  conf(c) = exp(mean logprob of
         │ (label-token        │  category-name tokens)
         │  logprobs)          │  ⚠ p50 = 0.984 — too compressed
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Causal Graph        │  boosted(B) = Σ conf(A) × w(A→B)
         │ Propagation         │  11 Suppes-causal edges
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Pass 2 — holistic   │  full trace re-submitted
         │ re-verification     │  "check these N categories"
         │ (flagged cats only) │  fires 15/117 traces, adds 1 error
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Merge + Output JSON │
         └────────────────────┘
```

**Known failure modes**:
- Confidence compressed → graph fires broadly → false-positive category detections
- Pass 2 prompt is holistic ("check all of these"), not contrastive → model agrees with itself
- Span-ID gate drops some predictions that zero-shot would keep → lower location accuracy

### Command

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only \
    --validate_span_id \
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only/`

---

## Config 4 — UQ + Span Index

Config 3 with span index added to both Pass 1 and Pass 2 prompts. Span index recovers
location accuracy (0.044 → 0.135) but the confidence problem persists — F1 still
trails Config 2.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only \
    --validate_span_id \
    --span_index \
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index/`

---

## Config 5 — UQ + Span Index + Graph Probe (Current Best)

Replaces compressed logprob confidence with **hard-binary confidence** (conf=1.0 if
detected in Pass 1, conf=0.0 otherwise) and replaces the holistic Pass 2 with
**targeted per-category probes** — one focused LLM call per graph-propagated category.

```
Trace (JSON)
    │
    ├──► build_span_index()
    │
    ▼
┌─────────────────────────────────────────────┐
│  Pass 1 — QwenLong-L1-32B                  │
│  Span index + taxonomy prompt               │
│  → JSON: errors + scores                    │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Span-ID Gate        │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Hard-Binary Conf    │  conf(A) = 1.0 if A ∈ detected
         │                     │  conf(A) = 0.0 otherwise
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Causal Graph        │  boosted(B) = Σ conf(A) × w(A→B)
         │ Propagation         │  11 Suppes-causal edges
         │                     │  to_probe = {B: boosted(B) > 0.10
         │                     │              AND B ∉ detected}
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Graph Probe         │  one LLM call per category in
         │ (per-category)      │  to_probe; span index included;
         │                     │  contrastive framing
         │                     │  → {"present": true/false,
         │                     │     "location": span_id}
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Span-ID Gate (p2)   │  drop probe results with invalid
         │                     │  span_ids before merge
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Merge + Output JSON │
         └────────────────────┘
```

**Why this works better**:
- Hard-binary conf eliminates false propagation from weak/uncertain Pass 1 detections
- Per-category contrastive probe ("a causally related error was found; does X also exist?")
  forces the model to take a stance rather than holistically re-verifying
- Span index in probe prompt → probe results also have accurate locations

### Command

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only \
    --validate_span_id \
    --span_index \
    --graph_probe \
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe/`

---

## Planned Configs (Round 3)

### 3A — Explicit Causal Encoding + Extended Graph

```bash
# 3A-enc: explicit encoding, causal-only graph
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 --enforce_eager \
    --causal_only --validate_span_id --span_index --graph_probe \
    --explicit_causal_encoding \
    --propagation_threshold 0.10

# 3A-graph: extended graph (causal + correlation w>=0.20), implicit encoding
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 --enforce_eager \
    --corr_threshold 0.20 --validate_span_id --span_index --graph_probe \
    --propagation_threshold 0.10

# 3A-both: explicit encoding + extended graph
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 --enforce_eager \
    --corr_threshold 0.20 --validate_span_id --span_index --graph_probe \
    --explicit_causal_encoding \
    --propagation_threshold 0.10
```

### 3B — 2-Sample Consistency Confidence

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 --enforce_eager \
    --causal_only --validate_span_id --span_index --graph_probe \
    --explicit_causal_encoding \
    --consistency_confidence \
    --propagation_threshold 0.10
```

### 3C — Offline Span Re-ranking (run on top of Config 5 outputs)

```bash
conda activate /data/wang/junh/envs/causal
python UQ/rerank_spans.py \
    --input_dir  UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe \
    --trace_dir  benchmarking/data/GAIA \
    --output_dir UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_2C_reranked \
    --top_k 5
```

---

## Input Format

**Traces**: `benchmarking/data/{split}/*.json`
Each file is one agent trace: `{"trace_id": "...", "spans": [...]}`.
Each span has `span_id` (16-char hex), `span_name`, `span_attributes`
(`input.value`, `output.value`), and `child_spans` (recursive).

**Causal graph**: `graph/data/graph_input.pt`

| Field | Description |
|-------|-------------|
| `node_names` | 20 strings: 19 error categories + "Correct" |
| `edge_index` | (2, E) directed edges |
| `edge_weight` | bootstrap stability ∈ [0,1] per edge |
| `edge_is_causal` | 1.0 for the ~11 fully-validated causal edges |

---

## Output Format

Each output file is a JSON object with the same schema as `run_eval.py`:

```json
{
    "errors": [
        {
            "category": "Tool Selection Errors",
            "location": "3ce413bb6e7e4dcd",
            "evidence": "...",
            "description": "...",
            "impact": "HIGH | MEDIUM | LOW"
        }
    ],
    "scores": [
        {
            "reliability_score": 3,
            "reliability_reasoning": "...",
            "security_score": 5,
            "instruction_adherence_score": 4,
            "plan_opt_score": 3,
            "overall": 3.75
        }
    ],
    "_uq_meta": { ... }
}
```

Directly compatible with `benchmarking/eval/calculate_scores.py`.

---

## Scoring

```bash
# From benchmarking/
python eval/calculate_scores.py --results_dir ../UQ/outputs/outputs_<tag>
```

Reports weighted F1, location accuracy, joint accuracy, per-category precision/recall/F1,
and Pearson correlations between model-generated scores and ground-truth scores.

---

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `GAIA` | Dataset split |
| `--model` | `Tongyi-Zhiwen/QwenLong-L1-32B` | HuggingFace model ID |
| `--tensor_parallel_size` | `4` | Number of GPUs |
| `--max_model_len` | `131072` | Context window cap |
| `--max_new_tokens` | `8000` | Max output tokens per Pass 1 call |
| `--gpu_memory_utilization` | `0.75` | vLLM GPU memory fraction |
| `--enforce_eager` | `True` | Disable CUDA graph capture (avoids OOM) |
| `--validate_span_id` | `True` | Drop errors with invalid span_id locations |
| `--repair_location` | `False` | Try fuzzy span-name → span_id repair before dropping |
| `--propagation_threshold` | `0.10` | Minimum boosted_score to trigger a probe |
| `--edge_threshold` | `0.10` | Minimum Suppes edge weight (when not using causal_only) |
| `--causal_only` | off | Use only the 11 bootstrap-validated causal edges |
| `--corr_threshold` | `1.0` | **(3A)** Include causal + correlation edges with w ≥ this value (e.g. `0.20`) |
| `--span_index` | off | Prepend span_id → name reference table to prompts |
| `--graph_probe` | off | Per-category targeted probe instead of holistic Pass 2 |
| `--explicit_causal_encoding` | off | **(3A)** Name source category, span, and edge weight in probe prompt |
| `--consistency_confidence` | off | **(3B)** Run Pass 1 twice (T=0 + T=0.7); use agreement as confidence |
| `--graph_input` | `graph/data/graph_input.pt` | Path to Suppes graph file |
