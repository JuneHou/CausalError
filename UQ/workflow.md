# UQ Workflow: White-Box Uncertainty + Causal Graph Propagation

## Overview

Two-pass error detection pipeline for LLM agent traces.
Pass 1 produces draft predictions with token-level confidence scores.
An **output validity gate** checks that predicted locations are exact `span_id` hex
values before any confidence or graph logic runs.
The causal graph propagates per-category confidence to flag likely-missed downstream
errors. Pass 2 re-verifies only the flagged categories.

```
Trace (JSON)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Pass 1 — QwenLong-L1-32B (vLLM)           │
│  Full taxonomy prompt → JSON output         │
│  + logprobs=1 per generated token           │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Output Validity     │  ← span_id gate:
         │ / Span-ID Gate      │    reject span names,
         │                     │    attempt repair or skip
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Local Confidence    │
         │ Proxy               │
         │ (token logprobs)    │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Causal Graph        │
         │ Propagation         │
         │ (Suppes edges)      │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Pass 2 — targeted   │
         │ re-verification     │
         │ (flagged cats only) │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Merge + Validate    │
         │ + Output JSON +     │
         │ diagnostics meta    │
         └────────────────────┘
```

---

## Input

**Source**: `benchmarking/data/{split}/*.json`
Each file is one agent trace with fields `trace_id` and `spans` (nested span tree).
Each span has a `span_id` (hex string, e.g. `"77fb7128d6f04862"`) and a `span_name`
(human-readable path). The full JSON is passed as a string directly into the prompt.

**Causal graph**: `graph/data/graph_input.pt`
Built by `graph/04_build_graph_input.py` from GAIA training traces.
Contains:
- `node_names`: 20 error-type names (19 error categories + "Correct")
- `edge_index`: (2, E) directed edges between error-type nodes
- `edge_weight`: bootstrap stability score in [0, 1] per edge (how often the edge
  survived Suppes significance tests across bootstrap resamples)
- `edge_is_causal`: binary flag marking the ~11 edges with stability = 1.0

---

## Step 1 — Pass 1: Draft Error Detection

**Model**: `Tongyi-Zhiwen/QwenLong-L1-32B`
**Infrastructure**: vLLM, `tensor_parallel_size=4` (4× A40)
**Context**: `max_model_len=131072`

The prompt contains the full 20-category taxonomy and instructs the model to output
a JSON object with an `errors` list and a `scores` list (same schema as `run_eval.py`).

### Schema constraints

Each error entry must satisfy:
- `category`: an exact leaf name from the taxonomy (e.g. `"Tool Selection Errors"`,
  not a full path like `"Reasoning Errors/Decision Making/Tool Selection Errors"`)
- `location`: the **exact `span_id` hex string** from the trace JSON
  (e.g. `"77fb7128d6f04862"`), **not** the `span_name` or a path like
  `"CodeAgent.run → Step 1 → LiteLLMModel.__call__"`

The prompt example and instructions explicitly state the `span_id` requirement.
A post-generation repair step (see Step 1b) handles residual non-compliance.

### Step 1b — Output Validity / Span-ID Gate

After raw JSON generation, before confidence extraction:

1. Parse the JSON and build the set of valid `span_id`s from the input trace.
2. For each predicted error, check whether `location` is in the valid span_id set.
3. If not:
   - **Repair**: try to fuzzy-match the span name against `span_name` fields and
     substitute the corresponding `span_id` (enabled by `--repair_location`).
   - **Drop**: if repair fails or is disabled, remove the error entry and log it.
4. Normalize `category` full-path strings to leaf names (e.g. strip prefix up to
   the last `/`).

This gate is essential: 100% of locations in the first run were span names, not hex
IDs, causing location accuracy = 0. The gate prevents downstream confidence and
graph logic from running on evaluation-incompatible outputs.

vLLM is called with `logprobs=1`, which attaches to every generated token:
- `decoded_token`: the text of the sampled token
- `logprob`: log-probability of that token under the model distribution

---

## Step 2 — Local Confidence Proxy (Version 1)

**Goal**: assign an initial confidence proxy `conf(c) ∈ (0, 1]` to every detected
error category `c`. This is a **coarse surface-level signal**, not a calibrated
uncertainty estimate.

**Current method** (v1): geometric-mean token probability of the category-name string.

1. Reconstruct the generated text from `decoded_token` strings and build a
   character-to-token index.
2. For each detected category `c` (e.g. `"Tool Selection Errors"`), find the
   substring in the reconstructed text via regex.
3. Map the character span to token indices using the char-to-token index.
4. Extract the logprob of each token in that span:
   `logprobs = [lp_t1, lp_t2, ..., lp_tk]`
5. Compute:

```
mean_logprob(c) = mean(logprobs)
conf(c)         = exp(mean_logprob(c))   ∈ (0, 1]
```

**Known limitation**: observed confidence values cluster in the 0.85–0.99 range,
indicating the signal is too compressed and too tied to surface token generation
of the category label string. It does not capture uncertainty about the *span location*
or the *presence* of the error in the trace.

**Future variants** (not yet implemented):
- `object_logprob`: mean logprob over the entire error entry (category + location + evidence)
- `judge_binary`: second model call asking "is this error present?" → yes/no probability
- `self_consistency`: K=3 samples, fraction agreeing on category detection

**Undetected categories** are assigned `conf = 0` and do not contribute to propagation.

---

## Step 3 — Causal Graph Propagation

### Step 3A — Category-level propagation (current implementation)

**Goal**: identify error categories that were not detected in Pass 1 but are likely
present given what was detected — using the causal graph as a prior.

**Suppes edges** encode statistically-derived causal relationships: edge A → B means
that in training traces where A was annotated, B tended to follow causally.
`edge_weight` is the bootstrap stability of that relationship (fraction of bootstrap
rounds in which the Suppes precedence + probability-raising tests both passed).

**Propagation rule** (additive, one-hop):

```
boosted_score(B) = Σ_{A → B ∈ edges} conf(A) × edge_weight(A → B)
```

Only edges with `edge_weight ≥ edge_threshold` (default 0.10) are included.
Alternatively, `--causal_only` restricts to the ~11 edges with `edge_is_causal = 1`
(bootstrap stability = 1.0), which was the best-performing setting in the Gemini
zero-shot experiment (+0.034 weighted F1 on GAIA).

Categories with `boosted_score > propagation_threshold` and not already detected
are queued for Pass 2 verification.

### Step 3B — Candidate-span graph reranking (planned)

The current category-level propagation only recovers missed error *types*. The final
evaluation metric requires exact `(span_id, category)` pairs. A planned revision will:

1. For each flagged category B, enumerate candidate spans where B might occur
   (e.g. spans sharing span type with the causal predecessor A's location).
2. Score `(span, B)` pairs using the propagated confidence and span features.
3. Pass the top-K candidate spans to Pass 2 for span-level adjudication.

This is not yet implemented. The current pipeline only propagates at the category
level and relies on Pass 2 to independently locate the error within the trace.

---

## Step 4 — Pass 2: Targeted Re-verification

If any categories were flagged by propagation, a second call is made to the same model.

### Current mode: category verification

The prompt includes:
1. The full trace (same as Pass 1)
2. A summary of errors already detected in Pass 1
3. An explicit list of categories to specifically verify

The model is instructed to output only **newly found** errors (not repeating Pass 1)
as a minimal JSON `{"errors": [...]}` without scores.

If no categories were flagged (`to_verify` is empty), Pass 2 is skipped entirely.

### Planned mode: span-level adjudication

For each flagged category B, provide:
1. The K candidate spans identified by Step 3B
2. Ask the model: "For category B, which of these spans (if any) is the correct location?"

This aligns Pass 2 more directly with the exact-match joint accuracy metric.
Not yet implemented — requires Step 3B span reranking to be built first.

---

## Step 5 — Merge, Validate, and Output

Pass 1 and Pass 2 errors are merged by deduplicating on `(category, location)` pairs.
The final JSON output:

```json
{
    "errors": [
        {
            "category": "...",
            "location": "span_id_hex",
            "evidence": "...",
            "description": "...",
            "impact": "HIGH | MEDIUM | LOW"
        }
    ],
    "scores": [ { "reliability_score": ..., ... } ]
}
```

This schema is identical to `run_eval.py` output and is directly compatible with
`benchmarking/eval/calculate_scores.py`.

A companion `_meta_{trace_id}.json` is saved alongside each output file with:
- `pass1_detected`: categories found in Pass 1
- `pass1_prompt_tokens` / `pass1_response_tokens` / `pass1_finish_reason`
- `pass1_raw_response`: full model output for inspection
- `confidence`: `conf(c)` for each detected category
- `boosted_scores`: `boosted_score(B)` for all B > 0
- `pass2_verify`: categories sent to Pass 2
- `pass2_raw_response`: Pass 2 output when triggered
- **Validation diagnostics**:
  - `valid_json`: whether Pass 1 output parsed successfully
  - `span_id_valid_count`: number of errors with valid hex span_id locations
  - `span_id_invalid_count`: number of errors with span-name locations
  - `repaired_count`: number of locations successfully repaired
  - `dropped_count`: number of errors dropped due to invalid location

---

## Output

```
UQ/outputs/
└── outputs_QwenLong-L1-32B-{split}-uq_{graph_tag}/
    ├── {trace_id}.json          ← scored output (errors + scores)
    └── _meta_{trace_id}.json    ← UQ metadata + diagnostics
```

---

## Commands

All commands run from the **repo root** (`trail-benchmark/`) unless noted.

### 1. QwenLong zero-shot baseline (no graph)

```bash
# Run inference
CUDA_VISIBLE_DEVICES=0,3,4,7 python benchmarking/eval/run_eval_vllm.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager

# Score (from benchmarking/)
cd benchmarking && python eval/calculate_scores.py --results_dir outputs/zero_shot
```

Output: `benchmarking/outputs/zero_shot/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA/`

---

### 2. QwenLong + UQ causal graph (two-pass, current)

```bash
# Default: all Suppes edges with weight >= 0.10, drop invalid span_ids
CUDA_VISIBLE_DEVICES=1,3,4,5 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager

# Restrict to 11 fully validated causal edges only (best Gemini setting)
CUDA_VISIBLE_DEVICES=1,3,4,5 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only

# Diagnostic only (NOT for evaluation): how many locations could be recovered via fuzzy repair
# --repair_location substitutes span names with span_ids post-hoc — inflates location accuracy
# and is NOT comparable to the Gemini baseline which outputs correct span_ids natively.
# Use only to understand model span-ID compliance rate, not as a scored experiment.
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --repair_location \
    --output_dir UQ/outputs/diagnostic_repair

# Score (from benchmarking/)
cd benchmarking && python eval/calculate_scores.py --results_dir ../UQ/outputs
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_suppes_t0.1/`
(or `uq_causal_only` with `--causal_only`)

---

### 3. Gemini baselines (reference, already run)

```bash
# Zero-shot baseline
cd benchmarking && python eval/run_eval.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --max_workers 5

# With 11 validated causal edges injected into prompt
cd benchmarking && python eval/run_eval_with_graph.py \
    --model gemini/gemini-2.5-flash \
    --split GAIA \
    --causal_only \
    --max_workers 5
```

---

### 4. Score all experiments at once

```bash
cd benchmarking

# All zero-shot and graph-guided Gemini/QwenLong results
python eval/calculate_scores.py --results_dir outputs/zero_shot

# UQ two-pass results
python eval/calculate_scores.py --results_dir ../UQ/outputs
```

---

### 5. Inspect UQ diagnostics (span-ID gate stats)

```bash
python3 -c "
import json, glob, collections

meta_dir = 'UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_suppes_t0.1'
total_valid, total_repaired, total_dropped = 0, 0, 0
pass2_triggered = 0

for f in glob.glob(f'{meta_dir}/_meta_*.json'):
    m = json.load(open(f))
    diag = m.get('location_diagnostics', {})
    total_valid   += diag.get('span_id_valid_count', 0)
    total_repaired += diag.get('repaired_count', 0)
    total_dropped  += diag.get('dropped_count', 0)
    if m.get('pass2_verify'):
        pass2_triggered += 1

print(f'Span-ID gate:  valid={total_valid}  repaired={total_repaired}  dropped={total_dropped}')
print(f'Pass 2 triggered: {pass2_triggered} traces')
"
```

---

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--split` | `GAIA` | Dataset split (`GAIA` or `SWE Bench`) |
| `--model` | `Tongyi-Zhiwen/QwenLong-L1-32B` | HuggingFace model ID |
| `--tensor_parallel_size` | `4` | Number of GPUs |
| `--max_model_len` | `131072` | Max sequence length (model cap) |
| `--max_new_tokens` | `8000` | Max output tokens (8000 is sufficient; observed max ~1855) |
| `--gpu_memory_utilization` | `0.75` | Fraction of GPU memory for vLLM |
| `--enforce_eager` | `True` | Disable CUDA graph capture (avoids OOM during warmup) |
| `--propagation_threshold` | `0.10` | Minimum `boosted_score` to trigger Pass 2 |
| `--edge_threshold` | `0.10` | Minimum Suppes edge weight to include |
| `--causal_only` | off | Restrict to ~11 fully validated causal edges only |
| `--validate_span_id` | `True` | Enable span-ID gate (drop/repair invalid locations) |
| `--repair_location` | `False` | Attempt fuzzy span-name → span-ID repair before dropping |
| `--graph_input` | `graph/data/graph_input.pt` | Path to Suppes graph |

Planned parameters (not yet implemented):
- `--pass2_mode {category_verify, span_rerank}` — switch Pass 2 between current category
  verification and future span-level adjudication
- `--top_k_candidate_spans` — number of candidate spans to pass to span_rerank mode
- `--confidence_mode {category_logprob, object_logprob, judge_binary, self_consistency}`
  — switch the local confidence proxy method

---

## Relation to Prior Experiments

| Approach | Graph use | Confidence | Pass 2 | Location |
|---|---|---|---|---|
| `run_eval.py` (Gemini baseline) | none | none | none | span_id |
| `run_eval_vllm.py` (QwenLong baseline) | none | none | none | span_id (prompt-enforced) |
| `run_eval_with_graph.py` | edges injected into prompt | implicit (uniform) | none | span_id |
| `run_uq_eval.py` v1 (current) | category-level score propagation | category token logprob proxy | category verification | span_id (gate + repair) |
| `run_uq_eval.py` v2 (planned) | span-level candidate reranking | object/judge logprob | span adjudication | span_id (exact match target) |

**Current position**: `run_uq_eval.py` is a **category-level graph-guided recovery
pipeline**. It uses the causal graph to recover missed error *types* and relies on
Pass 2 to independently locate them. It is not yet a span-level graph-aware reranker.

`run_eval_with_graph.py` is a degenerate case of this pipeline where `conf(A) = 1`
for all categories regardless of whether A was detected — it always propagates all
edges, adding noise from absent errors. `run_uq_eval.py` conditions propagation on
instance-specific confidence, suppressing edges where the upstream error was not
confidently detected.

**Key metrics (GAIA, all 117 traces)**:

| Model | Weighted F1 | Loc Acc | Joint Acc |
|---|---|---|---|
| Gemini baseline | 0.3951 | 0.3656 | 0.1358 |
| Gemini + causal_only graph | 0.4277 | 0.3754 | 0.1552 |
| QwenLong baseline | 0.1540 | 0.0712 | 0.0195 |
| QwenLong + UQ causal (v1) | TBD | TBD | TBD |
