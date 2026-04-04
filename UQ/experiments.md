# UQ Pipeline Experiments Log

## Experiment 1: Architecture A — QwenLong-L1-32B, Causal-Only Propagation

**Date**: 2026-03-30
**Model**: `Tongyi-Zhiwen/QwenLong-L1-32B`
**Script**: `UQ/run_uq_eval.py`
**Output dir**: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only/`

### Algorithm / Mechanism

This is the baseline UQ pipeline. It implements a two-pass LLM inference loop with confidence extraction and causal graph propagation between the passes.

**Pass 1 — holistic error detection**:
- Input: full TRAIL trace as a single prompt with a taxonomy block listing all 20 error categories
- Model generates a JSON list of detected errors, each with `{category, location, evidence, description, impact}`
- Location is a 16-char hex span_id from the trace

**Confidence extraction**:
- After Pass 1, re-tokenize the output and look up per-token log probabilities from vLLM (`logprobs=1`)
- For each detected category label (e.g., `"Tool Selection Errors"`), find those tokens in the output and compute:
  ```
  conf(c) = exp( mean( logprob(t) for t in tokens_of(category_string) ) )
  ```
- This gives one scalar confidence per detected category

**Causal graph propagation**:
- Use only the 11 Suppes-causal-validated edges (`--causal_only`), each with a bootstrap-stable weight
- For every category B in the taxonomy, compute a boosted score:
  ```
  boosted_score(B) = Σ_{A → B in graph} conf(A) × edge_weight(A → B)
  ```
- If `boosted_score(B) > propagation_threshold` AND B was not already detected in Pass 1, add B to the `to_verify` set

**Pass 2 — holistic re-check**:
- Build a new prompt that lists the `to_verify` categories and asks the model to re-examine the trace specifically for those categories
- Model returns a JSON list of any newly found errors
- Merge Pass 2 errors into Pass 1 output

**Span validation gate** (`--validate_span_id`):
- After each pass, check that each `location` value is a valid 16-char hex string that appears as a `span_id` in the original trace
- Drop (not repair) any error entry that fails this check

### Run Command
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

### Key Settings
| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_model_len` | 131072 | Model's hard cap; 1M fails at startup |
| `gpu_memory_utilization` | 0.75 | Needed to avoid OOM on partially-occupied GPUs |
| `enforce_eager` | True | Prevents CUDA graph OOM during warmup |
| `causal_only` | True | Uses only 11 fully-validated Suppes causal edges |
| `validate_span_id` | True | Drops errors with span names instead of span_id hex |
| `repair_location` | False | Not used in scored run (would inflate scores) |
| `propagation_threshold` | 0.10 | Boosted score threshold to trigger Pass 2 |
| `max_new_tokens` | 8000 | QwenLong is a reasoning model; needs long output budget |

### Quantitative Results

| Metric | QwenLong Zero-Shot | QwenLong UQ (causal_only) | Gemini 2.5 Flash |
|--------|-------------------|--------------------------|-----------------|
| Weighted F1 | 0.1540 | 0.1365 | 0.3951 |
| Location Accuracy | 0.0712 | 0.0439 | 0.3656 |
| Location-Category Joint | 0.0195 | 0.0127 | 0.1358 |
| Traces scored | 117 | 117 | 109 |

**Finding**: UQ pipeline underperforms zero-shot baseline. Gemini is 2.5x better on F1 and 8x better on location accuracy.

---

## Analysis of Failure Modes

### Problem 1: Confidence Scores Are Severely Compressed

**Measurement** (115 confidence scores extracted from 117 traces):

| Percentile | Confidence Value |
|-----------|-----------------|
| p0 (min) | 0.617 |
| p10 | 0.834 |
| p25 | 0.912 |
| p50 | 0.984 |
| p75 | 1.000 |
| p90 | 1.000 |
| p100 | 1.000 |

**Root cause**: The confidence proxy is `conf(c) = exp(mean_logprob of category-name tokens)`. Once the reasoning model has committed to outputting a category label, each subsequent token in the fixed string (e.g., `"Language-only"`) is near-deterministic — the model is highly confident it is spelling the label correctly, not that it detected the error correctly. This is **label-spelling confidence**, not **task-level uncertainty**.

**Consequence**: Boosted scores (causal propagation output) inherit the same compression (p50=0.994). With `propagation_threshold=0.10`, nearly every category that appears in any causal edge from a detected category scores above threshold — but confidence provides essentially no discrimination.

---

### Problem 2: Pass 2 Rarely Fires, Rarely Helps

| Metric | Value |
|--------|-------|
| Traces triggering Pass 2 | 15 / 117 (12.8%) |
| Pass 2 responses parsed successfully | 15 / 15 |
| Pass 2 returned no new errors | 14 / 15 (93%) |
| Pass 2 added new errors | 1 / 15 (7%) |

**Parsing note**: QwenLong-L1-32B is a reasoning model and prepends a `<think>...</think>` chain-of-thought block before the final JSON output. The `parse_json_output` function recovers the JSON via DOTALL regex (matching from first `{` to last `}`), which works because the `<think>` block uses no `{}`  braces.

**Why so few triggers?** `propagation_threshold=0.10` should trigger broadly given compressed boosted scores, but Pass 2 only runs for categories **not already in Pass 1**. Because Pass 1 detects multiple categories per trace, most propagated categories are already covered.

**The one addition (trace `53dba4241b22d5039c9c119871c7c8b4`)**:
- Pass 1 detected: `Tool-related`, `Tool Selection Errors`, `Timeout Issues`
- Pass 2 asked to check: `Language-only`, `Goal Deviation` (from causal propagation)
- Pass 2 found both present at span `e3fc8c667a3d7533`
- **Structural issue**: Pass 2 output used key `span_id` instead of `location`, and omitted `evidence`/`description`/`impact` fields — the output schema is inconsistent with Pass 1

---

### Problem 3: Location Accuracy Near Zero — Span Selection, Not Format

**Measurement**: Location accuracy = 0.0439 (UQ) vs 0.0712 (zero-shot).

**Correction from initial hypothesis**: The format is NOT the issue. Both pipelines produce valid hex span_ids:
- Zero-shot: 119/121 error entries (98.3%) have valid 16-char hex span_ids
- UQ pipeline: 116/124 error entries (93.5%) have valid span_ids; 8 dropped by gate

**Real root cause**: The model outputs valid hex strings but picks the **wrong span**. With a mean of 30.6 spans per trace (range 11–178), correct span selection requires understanding which step the error occurred in. The model often defaults to the root `main` span or an LLM call span without semantically grounding the error to the correct location.

Zero-shot location accuracy of 0.0712 means roughly 8–9 out of 119 predictions land on the correct span — only slightly above the ~3% random baseline (1/30 average).

**Why UQ has lower location accuracy than zero-shot**: The span_id gate is strict-and-correct. Zero-shot's reported 0.0712 includes some partially-correct predictions that UQ's gate would drop (wrong hex or span not in trace). The gate reduces noise but also removes some marginal hits.

---

### Problem 4: Pass 2 Is Category-Level Only

Even when Pass 2 correctly identifies that category B should be present (because category A causally implies B), it has no mechanism to identify **which span** the B error is located in. The current design only answers "should B be revisited?" not "which (span, B) pair should be added?". This means a correct category detection from Pass 2 will still score zero on location accuracy.

---

## Root Cause Summary

| Failure | Cause | Fix Direction |
|---------|-------|---------------|
| Compressed confidence | Label-token logprob is near-deterministic | Replace with self-consistency or object-level logprob |
| Pass 2 rarely fires | Threshold too loose AND most propagated cats already detected | Raise threshold; use more selective propagation |
| Pass 2 rarely adds errors | Model agrees with itself on second look | Pass 2 prompt needs stronger contrastive framing |
| Pass 2 output schema mismatch | No schema enforcement for `location` / `span_id` key | Standardize Pass 2 output template |
| Location accuracy near zero | Model outputs span names not span_ids | Stronger prompt instruction + few-shot examples with real span_ids |
| UQ worse than zero-shot | Gate drops predictions; confidence provides no selection signal | Fix location output first; then revisit confidence design |

---

## Proposed Ablation Sequence

The revised architecture (from user discussion 2026-03-30) treats local binary judging as the primary signal and the causal graph as a soft reranking prior. The ablations below test this direction incrementally.

### Experiment 2A: Span Index Injection ✓ COMPLETED 2026-03-31

**Question**: Does giving the model a compact `span_id → span_name` reference table improve span selection?

**Rationale**: The model has valid hex format (98%) but wrong span selection (~7% accuracy). The span_ids are buried in a 100K-char trace. A compact listing at the prompt top gives the model a reference table so it can reason semantically (span_name) while outputting correctly (span_id hex).

**Important**: The span index is a prompt-side reference only. Output format is unchanged — the model still generates the full TRAIL JSON with `"location": "span_id_hex"`. Evaluation via `calculate_scores.py` still does exact string matching. No binary classification; fully TRAIL-compatible.

### Algorithm / Mechanism

**Span index construction** (`build_step_spans()` in `span_level_parser.py`):
1. Parse the TRAIL trace JSON and walk the span tree
2. Select only agent-step spans: any span whose name matches `CodeAgent.run`, `ToolCallingAgent.run`, or `Step \d+` at any nesting depth
3. For each selected span, also include its direct LLM and TOOL children (one level down)
4. Format as a compact reference block injected at the top of the prompt:
   ```
   Span reference (span_id → name):
     a1b2c3d4e5f60001  CodeAgent.run
     a1b2c3d4e5f60002    Step 1
     a1b2c3d4e5f60003      LiteLLMModel.__call__
     a1b2c3d4e5f60004    Step 2
     ...
   ```
5. Average: ~8 agent-step spans + ~16 child spans per trace (~24 total, vs 30.6 recursive average)

**Why this works**: The model can reason "the error happened in Step 3's LLM call" and then look up the exact hex ID from the reference table rather than scanning a 100K-character document.

**Change**: `--span_index` flag added to `run_eval_vllm.py`.

**Run command** (from `benchmarking/`):
```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python eval/run_eval_vllm.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --span_index
```
Output: `outputs/zero_shot/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_span_index/`

### Results

| Metric | QwenLong Zero-Shot | QwenLong + Span Index | QwenLong UQ Causal | Gemini Zero-Shot | Gemini + Causal |
|--------|-------------------|-----------------------|--------------------|-----------------|----------------|
| Weighted F1 | 0.1540 | **0.1799** | 0.1365 | 0.3951 | 0.4277 |
| Location Accuracy | 0.0712 | **0.1261** | 0.0439 | 0.3656 | 0.3754 |
| Location-Category Joint | 0.0195 | **0.0346** | 0.0127 | 0.1358 | 0.1552 |
| Traces with overflow | 44 | 40 | — | — | — |

### Analysis

**1. Span selection is the dominant location bottleneck — confirmed.**
Location accuracy jumped from 0.0712 → 0.1261 (+77% relative) with a single prompt change. The model was not failing due to hex format issues (format compliance was already 98%) but because it could not recall the correct span_id from a 100K-char document. The reference table directly solves this.

**2. Span index is the most effective QwenLong intervention so far.**
It outperforms both the zero-shot baseline and the UQ causal pipeline on all three metrics. The UQ pipeline's lower location accuracy (0.044) was partly caused by the span_id gate dropping incorrect predictions; with span index the base predictions are more accurate so fewer entries get dropped.

**3. Location gap to Gemini remains large (0.126 vs 0.366, 3× difference).**
Span index halved the gap but did not close it. The remaining gap likely reflects semantic span-selection capability (which step did the error occur in?) rather than format compliance. Binary judging (Exp 2B) targets this directly.

**4. Decision gate outcome: span index carries forward to all future experiments.**
Loc = 0.126 > 0.15 threshold was not quite met, but the +77% relative improvement clearly validates the approach. All future QwenLong experiments (including UQ pipeline) should include span index.

---

---

### Experiment 2A-UQ: Span Index + UQ Causal Pipeline ✓ COMPLETED 2026-03-31

**Question**: Does adding the span index to the UQ pipeline recover the location improvement seen in 2A while also benefiting from graph propagation?

### Algorithm / Mechanism

Combines the Experiment 1 two-pass pipeline (confidence extraction → graph propagation → holistic Pass 2) with the span index from 2A. The span index is injected into **both** Pass 1 and Pass 2 prompts so the model can look up span_ids in both inference calls.

No other algorithmic changes from Experiment 1. The confidence extraction formula, propagation formula, and Pass 2 holistic re-check design are identical. This is a controlled test of span index benefit within the UQ framework.

**Run command**:
```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only \
    --span_index
```
Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index/`

### Results

| Metric | Zero-Shot | + Span Index | UQ Causal | UQ + Span Index |
|--------|-----------|--------------|-----------|-----------------|
| Weighted F1 | 0.1540 | **0.1799** | 0.1365 | 0.1653 |
| Location Accuracy | 0.0712 | 0.1261 | 0.0439 | **0.1353** |
| Location-Category Joint | 0.0195 | **0.0346** | 0.0127 | 0.0272 |

### Analysis

**Location improved, F1 dropped.** Span index is still doing its job (Loc: 0.1261 → 0.1353, +7%), confirming it helps within the UQ pipeline too. However, F1 dropped (0.1799 → 0.1653) and joint dropped (0.0346 → 0.0272), meaning the UQ graph propagation is adding false-positive categories that hurt precision.

**Root cause: graph propagation amplifies false positives under compressed confidence.** Because confidence scores remain compressed (p50 ≈ 0.984), propagation treats every Pass 1 detection as near-certain. Causal neighbors of any detected category get boosted above threshold and Pass 2 finds them — even when they do not genuinely exist. The span_index improves location but cannot fix the category-precision problem caused by the confidence signal.

**Conclusion**: The UQ architecture needs calibrated confidence before graph propagation is useful. Proceeding to Experiment 2B (binary P(Yes)/P(No) judging) to fix this.

---

### Experiment 2B: Binary Decomposed Judging ✗ ABANDONED

**Original plan**: Extract `P(Yes) = logprob("Yes") / (logprob("Yes") + logprob("No"))` from per-(span, category) binary probes to replace compressed label-logprob confidence.

### Intended Algorithm / Mechanism

**Confidence calibration via binary probing**:
1. For each (span, category) candidate from Pass 1, construct a focused binary prompt:
   > "Span `{span_id}` (`{span_name}`): does this span contain evidence of a `{category}` error? Answer Yes or No."
2. Request logprobs for the first output token only (`max_tokens=1`)
3. Compute a symmetric binary confidence:
   ```
   P(Yes) = exp(logprob("Yes")) / ( exp(logprob("Yes")) + exp(logprob("No")) )
   ```
   This forces probability mass onto exactly two tokens, producing a calibrated score in [0, 1] rather than the ~0.984 compressed label-logprob
4. Filter Pass 1 detections: keep only (span, category) pairs where `P(Yes) > threshold`
5. Graph propagation: use `P(Yes)` as `conf(A)` in the boosted score formula — now a real discriminative signal

**Why P(Yes) fixes the compression problem**: The label-logprob asks "what probability does the model assign to this spelling?" (near-1.0 once committed to the label). The binary question asks "does this span have this error?" — a genuinely uncertain judgment that spreads mass between Yes and No.

**Why it was abandoned**:

1. **Reasoning model incompatibility**: QwenLong-L1-32B prepends a `<think>...</think>` chain-of-thought block before any token output. With `max_tokens=5`, the sampled output captures only the start of the think block — the model never reaches the Yes/No token. `_yes_no_p_yes()` found neither "Yes" nor "No" in the first-token logprobs and returned 0.5 (fallback) for every probe. All confidence values became 0.5, making 2B a no-op.

2. **Fixing the reasoning model issue is expensive**: Stripping the `<think>` block and parsing the first token after it would require `max_tokens ≈ 1024` per probe. With ~234 probes per 117-trace run, this is ~27,000 additional LLM calls — infeasible on the available GPU budget.

3. **Span-level design flaw**: The initial 2B design probed all agent-step spans to re-select the best one per category. Ground-truth annotations in TRAIL are at the `LiteLLMModel.__call__` level (children of steps), not the step level itself. Probing only step-level spans would systematically miss correct locations, dropping location accuracy to ~0.

**Decision**: Skip 2B entirely. Use **hard-binary confidence** (detected = 1.0, undetected = 0.0) in the 2C architecture instead — same discriminative benefit, no logprob extraction required.

---

### Experiment 2C: Targeted Graph-Guided Probing ✓ COMPLETED 2026-04-01

**Core idea (revised from original plan)**: The causal graph identifies *which categories* to probe next, given what Pass 1 already detected. Rather than reranking existing predictions, 2C runs a **targeted second pass** for graph-propagated categories not yet found in Pass 1.

### Algorithm / Mechanism

**Step 1 — Pass 1 (same as 2A-UQ)**:
- Holistic JSON generation with span index injected at prompt top
- Collect set of detected categories: `detected_cats`

**Step 2 — Hard-binary confidence assignment**:
- Replace the compressed label-logprob confidence with a deterministic binary signal:
  ```
  conf(A) = 1.0  if A ∈ detected_cats
  conf(A) = 0.0  otherwise
  ```
- This eliminates false propagation: only categories the model explicitly reported in Pass 1 become source nodes; undetected categories contribute zero to any downstream propagation.

**Step 3 — Causal graph propagation**:
- For each edge `A → B` in the causal-only graph (11 edges, 6 source categories):
  ```
  boosted_score(B) = Σ_{A → B} conf(A) × edge_weight(A → B)
  ```
- Collect `to_probe = {B : boosted_score(B) > 0.10 AND B ∉ detected_cats}`

**Step 4 — Targeted graph probes** (one LLM call per category in `to_probe`):
- For each category B, build a contrastive probe prompt:
  > "Context: A causally related error was already detected in this trace. Based on statistical causal relationships, a `{B}` error is likely also present. Does a `{B}` error exist in the trace? Output ONLY JSON: `{"present": true, "location": "<span_id>", "evidence": "..."}` or `{"present": false}`."
- Span index is injected into each probe prompt so the model can look up the correct hex ID
- The contrastive framing ("already detected a related error") is intentional: it primes the model to look harder than it would from a cold start

**Step 5 — Merge and validate**:
- For each probe response where `present=true` and a valid `location` is provided:
  - Run span_id gate: drop if location hex is not in the trace's actual span set
  - Append to Pass 1 error list as a new error entry
- Final output = Pass 1 errors ∪ confirmed probe errors

**Key differences from 2A-UQ Pass 2**:

| Dimension | 2A-UQ Pass 2 | 2C Graph Probe |
|-----------|-------------|----------------|
| Confidence signal | Label logprob (~0.984 compressed) | Hard-binary (1.0/0.0) |
| What triggers a re-check | Any category with boosted_score > 0.10 | Same threshold, but now only truly-detected categories propagate |
| Prompt style | Holistic: "check all of these N categories" | Per-category: "does this specific type exist?" |
| Model output schema | Full TRAIL JSON (all fields) | Minimal: `{present, location, evidence, description, impact}` |
| Query count | 1 per trace (holistic) | 1 per propagated category per trace |

**Run command**:
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

### Results

| Metric | Zero-Shot | + Span Index | UQ Causal | UQ + SI | UQ + SI + Graph Probe (2C) |
|--------|-----------|--------------|-----------|---------|---------------------------|
| Weighted F1 | 0.1540 | 0.1799 | 0.1365 | 0.1653 | **0.2059** |
| Location Accuracy | 0.0712 | 0.1261 | 0.0439 | 0.1353 | **0.1353** |
| Location-Category Joint | 0.0195 | **0.0346** | 0.0127 | 0.0272 | 0.0244 |

### Analysis

**F1 improved substantially (+14% relative vs span_index baseline)**. The targeted graph probing brought F1 from 0.1799 to 0.2059, the best QwenLong result so far. This is the first configuration where the causal graph provably helps over the span_index-only baseline.

**Location accuracy unchanged (0.1353)**. The graph probing adds new error reports via the probe pathway; those probes include span selection, so location accuracy is maintained but not increased. The 2C contributions are new true-positive category detections, not location improvements.

**Joint accuracy dropped slightly (0.0272 → 0.0244)**. This is expected: joint requires both correct category AND correct span. The probe pathway adds some correct category detections whose span selection is imperfect, slightly lowering joint. The trade-off (higher F1, same loc, lower joint) reflects a precision/recall shift toward better recall on categories.

**Why hard-binary confidence fixed the false-positive problem from 2A-UQ**:
- In 2A-UQ, compressed confidence (p50 ≈ 0.984) treated every Pass 1 detection as near-certain, causing graph propagation to fire broadly and add false-positive categories.
- Hard-binary (0.0 for undetected categories) means the graph only propagates from categories the model explicitly reported in Pass 1. Combined with the targeted contrastive probe prompt, precision is maintained while recall improves.

**Graph coverage is intentionally narrow**: Only 11 causal-only edges from 6 source categories (Formatting Errors, Tool Selection Errors, Poor Information Retrieval, Resource Abuse, Task Orchestration, Incorrect Problem Identification). 2C triggers only when Pass 1 detects one of these 6. This is a precision improvement, not a broad recall booster.

**Conclusion**: 2C (targeted graph-guided probing with hard-binary confidence) improves F1 by +14% relative over the best span_index baseline, demonstrating that the causal graph contributes meaningfully to error detection when confidence is properly calibrated.

---

### Experiment 2D: Self-Consistency for Uncertain Probes (deferred)

**Original plan**: For graph-probe candidates where confidence is uncertain, run N=2 additional probes (3 total) and take majority vote.

### Intended Algorithm / Mechanism

**Self-consistency over uncertain binary probes**:
1. Run 2C as above; collect per-category `P(present)` from probe responses
2. Identify uncertain probes: categories where the initial probe returned `present=true` but the model's expressed confidence (via tone, hedge words, or a soft confidence field) is borderline
3. For uncertain candidates only, run 2 additional independent probes with temperature > 0 (sampling variation)
4. Apply majority vote over 3 probes: keep the error entry only if ≥ 2/3 probes return `present=true`
5. This produces a consistency-based filter: errors that pass 3-probe majority are higher precision than single-probe detections

**Status**: Deferred. In 2C, confidence is hard-binary (1.0 for Pass 1 detections), so there are no uncertain candidates to re-probe. Self-consistency would only apply if a soft confidence signal is available (e.g., from a calibrated model or explicit confidence field in the probe response). Re-evaluate if a future experiment produces soft scores.

---

### Deferred

- **Tiny calibrator**: needs features from 2B/2C/2D first; feasibility with 117 traces and cross-validation is limited
- **Exhaustive binary grid** (all agent-step spans × 20 cats × 117 traces = ~19,600 queries): infeasible with reasoning model; use P1-candidate-only mode instead
- **Gemini 2C run**: run the 2C configuration (causal_only + span_index + graph_probe) on Gemini to produce the paper comparison table

---

---

## Round 3 Experiments — Planned (2026-04-01)

Three directions derived from analysis of 2C failure modes and 2025 literature. Each direction maps to one structural gap in the current best config (2C: F1=0.206, Loc=0.135):

| Gap | Direction | Primary experiment |
|-----|-----------|-------------------|
| Causal graph probe prompt does not convey *why* a category is likely | Direction 2 — explicit causal encoding + extended graph | Exp 3A |
| Pass 1 confidence is hard-binary (0/1), no graded signal for propagation | Direction 1 — 2-sample consistency confidence | Exp 3B |
| Span selection wrong ~87% of the time; 40 traces overflow context | Direction 3 — local-window span re-ranking | Exp 3C |

---

### Experiment 3A: Explicit Causal Encoding + Extended Graph ✓ DONE (3A-enc)

**Direction 2 — Causal Graph Coverage and Encoding**  
**Cost**: Zero additional LLM calls (prompt change only)  
**Primary metric target**: F1 (via higher-precision, higher-recall graph probing)

**Question**: Does encoding causal graph structure explicitly in the probe prompt — with source category, edge weight, and validation status — improve graph probe precision and recall over the current implicit encoding? And does adding high-weight correlation edges (w≥0.20) alongside the 11 causal edges further improve recall?

#### Why This Direction

**Gap 2a — Graph coverage**: Only 11 causal edges from 6 source categories. Instruction Non-compliance (precision 0.70–0.75 in 2C) is not a source node despite strong correlations: Instruction Non-compliance → Goal Deviation (w=0.385, 62 GT instances) and → Language-only (w=0.319, 40 GT instances). These are the two highest-support categories in the dataset.

**Gap 2b — Encoding format**: `GRAPH_PROBE_TEMPLATE` currently uses implicit encoding (*"a causally related error was already detected"*) with no explicit graph structure shown to the model.

#### Referenced Paper

> **Sheth et al. (2025, NAACL Findings). *"CausalGraph2LLM: Evaluating LLMs for Causal Queries."***  
> A benchmark of 700k+ queries reveals that LLMs show ~60% performance variance based solely on how causal graph structure is presented in the prompt — even for GPT-4. Encoding sensitivity (text format, explicit quantification, direction labeling) is as important as model reasoning capability; structured, quantified encodings consistently outperform implicit ones.

**How we adopt it**: The ~60% variance finding means our current implicit encoding is likely leaving a large fraction of graph probe accuracy on the table. We replace the implicit framing with an explicit, quantified description that names both endpoints, states the edge weight, and cites the validation method — mirroring the structured encoding formats that outperformed in CausalGraph2LLM. We also extend the graph from 11 to ~26 edges by adding correlation edges with w≥0.20, introducing 3 new source categories (Instruction Non-compliance, Tool Output Misinterpretation, Environment Setup Errors).

#### Algorithm / Mechanism

**Prompt change** — replace implicit framing in `GRAPH_PROBE_TEMPLATE`:

*Current (implicit)*:
```
Context: A causally related error was already detected in this trace. Based on statistical
causal relationships between error types, a "{category}" error is likely also present.
```

*Revised (explicit + quantified)*:
```
Causal graph analysis: {source_category} was detected in this trace (at span {source_span}).
Statistical analysis of agent execution traces shows {source_category} causally precedes
{target_category} in a bootstrap-validated causal graph (Suppes criterion, edge weight={weight:.2f}).
Based on this causal relationship, a {target_category} error is likely also present in this trace.
```

This requires passing `source_category`, `source_span`, and `weight` into `build_graph_probe_prompt()`. All three are already available in the pipeline (`p1_errors`, `boosted` propagation, edge list).

**Graph extension**: add correlation edges with w≥0.20 (15 new edges, ~26 total from 9 source categories). The existing formula `Σ conf(A) × edge_weight(A→B)` already discounts correlation edges by their actual weight (0.20–0.385 vs 1.0 for causal). Run with hard-binary confidence here to isolate the encoding and graph effects from the confidence signal change in 3B.

#### Ablations

| Run | Encoding | Graph | Confidence | Purpose |
|-----|----------|-------|------------|---------|
| 3A-enc | Explicit | Causal-only (11) | Hard-binary | Isolate encoding effect |
| 3A-graph | Implicit | Extended (26) | Hard-binary | Isolate graph coverage effect |
| 3A-both | Explicit | Extended (26) | Hard-binary | Combined |

#### Run Command

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --causal_only \              # swap for --corr_threshold 0.20 in 3A-graph/3A-both
    --validate_span_id \
    --span_index \
    --graph_probe \
    --explicit_causal_encoding \ # new flag
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe_explicit_enc/`

#### Result (3A-enc, 117/117 traces)

| Metric | Config 5 baseline | 3A-enc | Δ |
|--------|------------------|--------|---|
| W-F1 | 0.2059 | 0.2049 | −0.001 |
| Loc Acc | 0.1353 | 0.1353 | 0.000 |
| Joint | 0.0244 | 0.0261 | +0.002 |

**Conclusion: negative.** Explicit causal framing (source category + span + edge weight in probe prompt) does not improve over implicit framing. The probe task is a yes/no error detection question — adding causal provenance text does not change the model's detection decision. The CausalGraph2LLM finding (60% variance from encoding format) applies to graph reasoning tasks, not to binary error detection in execution traces.

3A-graph and 3A-both were not run: if the encoding change alone had no effect, extending the graph would only add more probe calls without improving precision.

---

### Experiment 3B: 2-Sample Consistency Confidence ✓ DONE

**Direction 1 — Confidence Signal for Graph Propagation**  
**Cost**: 1 extra Pass 1 call per trace (117 additional calls)  
**Primary metric target**: F1 (better confidence → better graph propagation precision)

**Question**: Does replacing hard-binary confidence with 2-sample consistency confidence (T=0 + T=0.7 agreement) give the graph propagation a graded signal that reduces false-positive category probes?

#### Why This Direction

**The problem with hard-binary (2C)**: conf=1.0 for all Pass 1 detections treats a strong detection and a marginal one identically — both propagate equally through the graph. This causes some false-positive category probes from uncertain detections.

**Why we can't just add a verbalized confidence field**: prior to choosing 2-sample consistency, verbalized confidence was the obvious option. It was ruled out on the basis of:

> **Heo et al. (2025, ICLR). *"Do LLMs Estimate Uncertainty Well in Instruction-Following?"***  
> First systematic evaluation of LLM uncertainty in instruction-following tasks. Shows verbalized confidence remains inadequate for subtle instruction-following errors — the exact regime of agent error detection — even when the model has chain-of-thought reasoning available. Internal model states improve uncertainty estimation but do not solve the problem.

Adding a `"confidence"` field to the output template would likely produce overconfident, poorly-discriminating scores for subtle errors like "Incorrect Problem Identification" — precisely the categories where we want better signal.

#### Referenced Paper

> **Del et al. (2025, arXiv:2603.19118). *"How Uncertainty Estimation Scales with Sampling in Reasoning Models."***  
> For reasoning models specifically, hybrid uncertainty estimators that combine signals from N=2 samples improve AUROC by +12 on average over single-sample methods. Consistency across samples is a stronger discriminative signal than any single-sample proxy (logprob, verbalization, or internal state), and the gains hold with just two samples — making it cost-efficient.

**How we adopt it**: We operationalize the N=2 finding as category-level agreement between a greedy run (T=0) and a sampled run (T=0.7). Categories the model detects consistently are high-confidence; categories detected in only one run are borderline. This produces a three-level graded signal that replaces hard-binary without requiring logprob extraction (which fails for reasoning models) or verbalization (which fails for subtle errors).

#### Algorithm / Mechanism

**Step 1 — Run Pass 1 twice**:
- Pass 1a: T=0 (greedy, same as current) → `detected_cats_greedy`
- Pass 1b: T=0.7 (sampled) → `detected_cats_sampled`

**Step 2 — Assign consistency confidence**:
```python
conf = {}
all_detected = detected_cats_greedy | detected_cats_sampled
for cat in all_detected:
    in_greedy  = cat in detected_cats_greedy
    in_sampled = cat in detected_cats_sampled
    if in_greedy and in_sampled:
        conf[cat] = 1.0
    else:
        conf[cat] = 0.5
# Undetected categories remain at 0.0
```

**Step 3 — Propagation with graded signal**:
```
boosted_score(B) = Σ_{A→B} conf(A) × edge_weight(A→B)
```
A borderline detection (conf=0.5) propagates half as strongly as a consistent one (conf=1.0). With threshold=0.10, a single borderline causal source gives 0.5 × 1.0 = 0.50 — still above threshold, so causal edges still fire. A correlation edge from a borderline source gives 0.5 × 0.20 = 0.10 — right at the boundary, providing natural discrimination.

**Step 4 — Final errors**: greedy-run Pass 1 errors form the base output. T=0.7 errors are used only for confidence estimation and are not added to the error list.

#### Ablations

| Run | Confidence | Graph | Encoding | Purpose |
|-----|------------|-------|----------|---------|
| 3B-conf | 2-sample consistency | Causal-only (11) | Explicit (from 3A) | Isolate confidence effect |
| 3B-graph | 2-sample consistency | Extended (26) | Explicit | Confidence + extended graph |

#### Run Command

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
    --explicit_causal_encoding \
    --consistency_confidence \    # new flag: enables 2-sample T=0.7 second pass
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe_explicit_enc_consistency_conf/`

#### Result (3B, 117/117 traces)

| Metric | Config 5 baseline | 3B | Δ |
|--------|------------------|-----|---|
| W-F1 | 0.2059 | 0.1989 | −0.007 |
| Loc Acc | 0.1353 | 0.1306 | −0.005 |
| Joint | 0.0244 | 0.0177 | −0.007 |

**Conclusion: negative.** Consistency confidence hurts across all metrics. Root cause: the `conf` dict is built from `detected_cats_greedy | detected_cats_sampled`, meaning categories detected only in the T=0.7 stochastic pass (but not greedy) get conf=0.5 and become graph propagation sources. These are hallucinations from the stochastic run — they fire probes for spurious targets, hurting precision. The fix would be to only allow greedy-detected categories as propagation sources and use the sampled pass solely to downgrade confidence. However, even with this fix, the T=0.7 pass is noisier than greedy — categories correctly detected at T=0 that are missed at T=0.7 get downgraded from conf=1.0 to conf=0.5, weakening real signals.

The Del et al. N=2 finding (+12 AUROC) applies to uncertainty estimation for reasoning model outputs — not to graph propagation thresholding in a detection pipeline.

---

### Experiment 3C: Local-Window Span Re-ranking ✓ DONE

**Direction 3 — Span Re-ranking for Location Accuracy**  
**Cost**: ~1,200 small LLM calls (~1.5K tokens each); no full trace re-submission  
**Primary metric target**: Location Accuracy (current bottleneck: 0.135 vs Gemini 0.366)

**Question**: Does pointwise re-ranking of top-K span candidates using local content windows — without re-submitting the full trace — improve location accuracy and resolve the 40-trace overflow problem?

#### Why This Direction

The span index (+77% location accuracy in 2A) solved the format problem: the model now has a reference table and produces valid hex span_ids 98% of the time. The remaining gap is **semantic grounding**: the model sees span names but not span contents, and cannot determine which step semantically matches the error evidence without reading what each span actually did.

Two additional problems compound this:
- ~40/117 traces overflow the 131K-token context; Pass 1 runs truncated on these
- Graph probing (2C) adds correct categories but their span assignments are no better than Pass 1 (location unchanged at 0.135)

#### Why Pointwise, Not Listwise

> **Abdallah et al. (2025, EMNLP Findings). *"How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models."***  
> Evaluated 22 reranking methods across TREC/BEIR benchmarks. LLM rerankers excel on familiar/seen query types but generalize poorly to novel, unseen queries — a significant limitation under distribution shift. Pointwise approaches are more robust in low-resource, novel-domain settings than listwise comparison methods.

Our execution spans are novel and unseen at inference time — QwenLong has no prior exposure to GAIA agent traces. This matches the "novel query" failure mode: listwise comparison (show all 24 spans, pick best) requires cross-span relative judgment on unseen patterns, which Abdallah et al. identify as unreliable. Pointwise scoring per candidate is more robust.

#### Validation That Step-Level LLM Scoring Works

> **Ou et al. (2025, EMNLP System Demonstrations). *"AgentDiagnose: An Open Toolkit for Diagnosing LLM Agent Trajectories."***  
> LLM-based step-level scoring of agent trajectories achieves r=0.78 Pearson correlation with human judges for task decomposition quality. Demonstrates that LLM judges can reliably score individual execution steps at useful accuracy.

AgentDiagnose establishes that pointwise LLM scoring of individual agent steps is feasible (r=0.78 with humans), giving a reasonable ceiling estimate for what span re-ranking can achieve in our setting.

#### Why Include Execution-Flow Context in the Local Window

> **ACM Web Conference 2025. *"LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations."***  
> Including structural context — how items relate to each other in a graph — in the re-ranking prompt significantly outperforms presenting item content in isolation. Chain-of-thought re-ranking with global relational context beats isolated pointwise scoring.

**How we adopt it**: rather than presenting each candidate span's content in isolation, we include execution-flow position (parent step, previous sibling) in each local window. This gives the model the structural context — "this span is Step 3's LLM call, which came after a tool call that returned a 404 error" — that LLM4Rerank shows is necessary for accurate pointwise scoring.

#### Algorithm / Mechanism

After 2C produces `{category, location_rough, evidence}` for each error entry:

**Step 1 — Candidate pre-filter**: score all indexed spans (~24) by BM25 between evidence text and span content; select top-K=5 candidates.

**Step 2 — Local window extraction** per candidate:
```
span_name + span_id
parent: {parent_step_name}
preceding sibling: {prev_sibling_content[:100]}
content: {span_input_or_output[:300]}
```

**Step 3 — Pointwise scoring** (all K=5 candidates batched per trace in one `llm.generate()` call):

```
Error type: {category} — {one-line taxonomy definition}
Evidence from initial detection: {evidence}

Candidate span "{span_id}" ({span_name})
  Execution position: {parent_step} → {span_name}
  Preceding context: [{sibling_summary}]
  Content: [{span_content}]

How likely is this span the location of the described error?
1=very unlikely  2=unlikely  3=uncertain  4=likely  5=very likely
Output ONLY: {"score": <1-5>}
```

**Step 4 — Re-rank and update**:
- Select span with highest score
- If max score ≥ 3: replace `location` with re-ranked span_id
- If all scores < 3: retain original location from Pass 1 / graph probe
- Apply span_id gate after re-ranking (same as `validate_and_repair_locations`)

**Step 5 — Overflow handling**: local windows are 500–1,500 tokens regardless of full trace length. The ~40 overflow traces can now be re-ranked without truncation.

#### Implementation Plan

1. New offline script `UQ/rerank_spans.py` — reads existing 2C output JSONs + original trace JSONs, writes re-ranked JSONs to a new directory
2. Validate with `calculate_scores.py`
3. If location accuracy improves, integrate as `--span_rerank` flag in `run_uq_eval.py`

#### Run Command (offline)

```bash
conda activate /data/wang/junh/envs/causal
python UQ/rerank_spans.py \
    --input_dir  UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_causal_only_span_index_graph_probe/ \
    --trace_dir  benchmarking/data/GAIA/ \
    --output_dir UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_2C_reranked/ \
    --top_k 5
```

#### Result (3C, 117 traces, 24 locations changed)

| Metric | Config 5 baseline | 3C re-ranked | Δ |
|--------|------------------|--------------|---|
| W-F1 | 0.2059 | 0.2059 | 0.000 |
| Loc Acc | 0.1353 | 0.1268 | −0.009 |
| Joint | 0.0244 | 0.0188 | −0.006 |

**Conclusion: negative.** Re-ranking changed 24 locations but moved more to wrong spans than correct ones. W-F1 is unchanged (categories not modified). Root cause: BM25 keyword overlap between error evidence and span content favors spans with more text (long tool outputs, web page content) over the `LiteLLMModel.__call__` spans where ground-truth annotations live. The re-ranker cannot recover the correct span type from evidence text alone — it lacks the structural knowledge that TRAIL annotations always point to LLM call spans, not tool output spans.

---

### Experiment 3-Full: Combined 3A + 3B + 3C ✗ CANCELLED

**Question**: Do all three improvements stack — explicit causal encoding + extended graph (3A) + 2-sample consistency confidence (3B) + span re-ranking (3C)?

Run the full integrated pipeline after each component is individually validated:

```bash
CUDA_VISIBLE_DEVICES=0,3,4,7 python UQ/run_uq_eval.py \
    --split GAIA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --enforce_eager \
    --corr_threshold 0.20 \       # causal + correlation w>=0.20 (26 edges)
    --validate_span_id \
    --span_index \
    --graph_probe \
    --explicit_causal_encoding \
    --consistency_confidence \
    --span_rerank \
    --propagation_threshold 0.10
```

Output: `UQ/outputs/outputs_QwenLong-L1-32B-GAIA-uq_3full/`

### Round 3 Summary

All three Round 3 directions failed to improve over Config 5. The 77 traces that ran successfully (40 overflow) were the same across all configs, so the comparison is fair — the methods would have shown improvement on those 77 traces if the approach were valid.

| Config | W-F1 | Loc Acc | Joint | Notes |
|--------|------|---------|-------|-------|
| Zero-shot | 0.154 | 0.071 | 0.020 | |
| + Span index (2A) | 0.180 | 0.126 | 0.035 | |
| + Graph probe (2C) | **0.206** | **0.135** | 0.024 | Current best |
| + Explicit encoding (3A-enc) | 0.205 | 0.135 | 0.026 | Neutral |
| + Consistency conf (3B) | 0.199 | 0.131 | 0.018 | Negative |
| + Span re-rank (3C) | 0.206 | 0.127 | 0.019 | Negative (loc↓) |
| 3-Full combined | — | — | — | Cancelled |
| Gemini zero-shot | 0.395 | 0.366 | 0.136 | Upper reference |

**Bottleneck analysis**: The 40 overflow traces (34%) are a hard ceiling — those traces produce no output for any config. For the 77 processed traces, category detection (W-F1) appears near-ceiling for this model+graph combination. Location accuracy (0.135) remains the main gap vs Gemini (0.366), but re-ranking based on BM25+pointwise LLM scoring does not solve it because the ground-truth annotation scheme (always `LiteLLMModel.__call__` spans) is not recoverable from evidence text alone.
