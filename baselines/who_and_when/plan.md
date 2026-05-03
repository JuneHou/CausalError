# Who&When Adaptation Plan for TRAIL Benchmark

**Source paper:** "Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems"  
**arXiv:** 2505.00212 (ICML 2025 Spotlight)  
**GitHub:** https://github.com/mingyin1/Agents_Failure_Attribution  
**Last updated:** 2026-04-29

---

## 1. Core Adaptation Philosophy

Who&When defines three **localization strategies** — all-at-once (W1), step-by-step (W2), binary search (W3) — for answering: *which agent caused the failure, and at which step?* The strategies differ in how they search for the error location; they all produce a single prediction because the original task assumes one responsible agent.

TRAIL is a **multi-label detection** task: every trace may contain zero, one, or many `(error_category, span_id)` pairs. We adapt the three Who&When localization strategies to TRAIL while removing the single-error constraint from each. The three variants therefore differ only in **how they search for error locations**, not in how many errors they predict.

**Mapping Who&When → TRAIL:**
- "Who" (which agent) → **What** (which error category from TRAIL's 19-category taxonomy)
- "When" (which step) → **Where** (which span_id in the trace)

**The single-error assumption is broken in each variant as follows:**
- **W1**: output all categories present, not just the most critical one
- **W2**: do not stop at the first detected error; continue through all spans and aggregate
- **W3**: run bisection separately per label; allow both halves to contain the target label; recurse into all positive halves

---

## 1a. Changes from the Original Who&When Implementation

The original code lives in `Automated_FA/Lib/utils.py` and `local_model.py` at https://github.com/mingyin1/Agents_Failure_Attribution.

### What the original does (verbatim from the repo)

| Aspect | Original | Our adaptation |
|---|---|---|
| **Task** | Multi-agent: predict which *agent* failed and at which *step* | Single-agent: predict which *error category* and at which *span_id* |
| **Output count** | Always 1 (one responsible agent + one step) | 0 or more `(category, span_id)` pairs |
| **Input format** | Problem, ground-truth answer, chat history with agent names | TRAIL trace JSON serialized as text; span index prepended |
| **Model** | Azure OpenAI GPT-4 / local Llama / Qwen | vLLM with Mistral-Small-3.1-24B |
| **Scoring** | Agent accuracy + step accuracy | W-F1, Location Accuracy, Joint Accuracy |

### W1 changes
**Original prompt** asks: *"Who is responsible? Which step is the root cause?"* — one agent name + one step number.  
**Our change (lines W1_PROMPT_TEMPLATE in code):** Replace "identify the SINGLE most critical error" with per-category multi-label instruction ("decide independently for each of the 19 categories"). Remove `errors = [errors[0]]` truncation in `run_w1()`.

### W2 changes
**Original logic** (`step_by_step` function): For each step, ask "Does this step have an error — Yes/No?" Stop immediately when first "Yes" is detected.  
**Our change (run_w2() in code):** Remove early-exit `return` on first "Yes". Change per-step prompt to return a JSON list of zero or more error categories (multi-label per step). Continue through all spans. Aggregate all `(category, span_id)` pairs at the end; deduplicate.

### W3 changes
**Original logic** (`binary_search` function + `_find_error_in_segment_recursive`): Bisect to find the single step containing *the* most critical error. At each level asks "upper half or lower half?" (exactly one). Terminates with one span → one classification call → one prediction.  
**Our change (run_w3() in code):** Run bisection **separately for each of the 19 labels**. At each bisect call, ask two independent booleans: `lower_half_present` and `upper_half_present` (both can be true). Recurse into all positive halves. Final leaf call asks `present: true/false` for the specific label at the specific span. This is label-conditioned recursive bisection with multi-label aggregation.

### wcat (new variant, not in original)
Not in the original Who&When paper. Added as a single structured call that explicitly declares all 19 categories and asks for independent yes/no per category — removing any single-error anchoring bias from W1's open-ended framing.

---

## 2. Input/Output Format for TRAIL

### Input
Each TRAIL trace is a sequence of spans, each having:
- A hex `span_id` (e.g., `037ba72b...`)
- A human-readable name (e.g., `Step 3: page_down`)
- Span kind: LLM, TOOL, or CHAIN
- Content: LLM input/output messages, tool call results, etc.

Spans are serialized in execution order and numbered (Step 1, Step 2, …). The span index table (span_id → human-readable name) from our +SI variant is reused to prevent span_id hallucination.

### Output (all three variants)
Standard TRAIL JSON, same schema as all other conditions:
```json
{
    "errors": [
        {
            "category": "<leaf category>",
            "location": "<exact span_id hex string>",
            "evidence": "<brief quote>",
            "description": "<one-sentence description>",
            "impact": "HIGH"
        }
    ],
    "scores": []
}
```
Zero or more error entries per trace. Scored with `calculate_scores.py` unchanged.

---

## 3. Three Method Variants

### Variant W1: All-at-Once (single pass, multi-label)
**LLM calls per trace:** 1

**What changed from original:**
The original gives the full trace and asks for one responsible agent + one step. We keep the full-trace-in-one-prompt design but replace the single-agent/single-step instruction with a multi-label instruction: output all error categories present, each with its first-occurrence span.

**Prompt:**
```
You are an AI assistant analyzing an agent execution trace.

The agent was solving this task: {task_description}

{span_index}

Here is the full execution trace:
{trace}

This is a multi-label task. Zero, one, or multiple error types may be present.
Do NOT force a label if there is no clear evidence.

For each error category in the taxonomy below, decide independently whether it is
present in the trace. For every category that IS present, identify the span where
it FIRST occurs.

TRAIL Error Taxonomy:
{taxonomy}

Valid category names (use EXACTLY as written — do not paraphrase):
Language-only, Tool-related, Poor Information Retrieval, Tool Output Misinterpretation,
Incorrect Problem Identification, Tool Selection Errors, Formatting Errors,
Instruction Non-compliance, Tool Definition Issues, Environment Setup Errors,
Rate Limiting, Authentication Errors, Service Errors, Resource Not Found,
Resource Exhaustion, Timeout Issues, Context Handling Failures,
Resource Abuse, Goal Deviation, Task Orchestration

Output strictly valid JSON — only include categories where your answer is YES:
{"errors": [{"category": "...", "location": "<span_id>", "evidence": "...",
             "description": "...", "impact": "HIGH"}, ...], "scores": []}
If no errors are present: {"errors": [], "scores": []}
```

**Difference from TRAIL Baseline:** The TRAIL baseline uses an open-ended "find all errors" instruction. W1 frames it as an explicit per-category binary judgment ("decide independently for each category"), which reduces anchoring on error count.

---

### Variant W2: Step-by-Step (sequential scan, multi-label)
**LLM calls per trace:** N (number of step spans, no early exit)

**What changed from original:**
The original stops at the first "Yes" step. We remove the early-exit and continue through **all** spans. Each call judges the current span for all taxonomy categories. Results are aggregated across all spans at the end.

**Per-step prompt:**
```
You are an AI assistant evaluating one step of an agent execution trace.

The agent was solving: {task_description}

Conversation history up to the current step:
{cumulative_spans_up_to_step_i}

The CURRENT step is:
  Step {i}  (span_id: "{span_id}")  —  {step_name}
  Content: {span_content}

Your task: determine whether the CURRENT STEP contains any error from the taxonomy.
This is a multi-label task — a single step may have zero, one, or multiple error types.
Predict an error only if the current step itself provides evidence (not just a downstream
consequence of an earlier error).

Valid category names (use EXACTLY as written):
Language-only, Tool-related, ... [full list]

Return JSON:
{
  "step_id": "{i}",
  "span_id": "{span_id}",
  "has_error": true/false,
  "errors": [
    {"category": "...", "evidence": "...", "description": "..."}
  ]
}
```

**Aggregation after all steps:** Collect all per-step errors. For each unique `(category, span_id)` pair, keep it. Apply span_id validation. Final output is the standard TRAIL JSON.

**Cost note:** N calls per trace where N = number of step spans (~8 for GAIA). Cumulative context grows with each step. Cap context by truncating the cumulative prefix if it exceeds the token budget, keeping only the most recent K spans as context.

---

### Variant W3: Binary Search per Label (label-conditioned recursive bisection)
**LLM calls per trace:** ≈ 19 × ⌈log₂(N)⌉ in the typical case

**What changed from original:**
The original bisects to find the single most critical step and forces a choice between upper/lower half. We run bisection **separately for each of the 19 labels** and allow **both halves to be positive** at each bisection step, recursing into all positive halves.

**Bisect prompt (per label, per interval):**
```
You are an AI assistant localizing one specific error type in an agent trace.

The agent was solving: {task_description}

Target error type: {label}
Definition: {label_definition}

Here are the spans in the current interval (Steps {low}–{high}):
{span_window}

Does "{label}" occur in the LOWER HALF (Steps {low}–{mid})?
Does "{label}" occur in the UPPER HALF (Steps {mid+1}–{high})?

Answer yes for a half only if that half contains direct evidence of the target error.
If a half only shows downstream consequences of an error that occurred before this
interval, answer no unless the target error independently recurs here.

Return JSON:
{
  "label": "{label}",
  "lower_half_present": true/false,
  "upper_half_present": true/false,
  "lower_evidence": "...",
  "upper_evidence": "..."
}
```

**Final localization prompt (when interval = 1 span):**
```
Target error type: {label}
Definition: {label_definition}

Span: Step {i} (span_id: "{span_id}") — {step_name}
Content: {span_content}

Is "{label}" present at this exact span?
Return JSON:
{"label": "{label}", "present": true/false, "evidence": "...", "span_id": "{span_id}"}
```

**Algorithm (pseudocode):**
```python
def bisect_label(label, spans):
    if len(spans) == 0:
        return []
    if len(spans) == 1:
        result = ask_single_span(label, spans[0])
        return [spans[0]['span_id']] if result.present else []
    mid = len(spans) // 2
    lower, upper = spans[:mid], spans[mid:]
    result = ask_bisect(label, spans, lower, upper)
    locations = []
    if result.lower_half_present:
        locations += bisect_label(label, lower)
    if result.upper_half_present:
        locations += bisect_label(label, upper)
    return locations

all_errors = []
for label in TAXONOMY_LEAF_CATEGORIES:
    locations = bisect_label(label, ordered_step_spans)
    for span_id in locations:
        all_errors.append({"category": label, "location": span_id, ...})
```

**Cost note:** For GAIA with ~8 step spans, ⌈log₂(8)⌉ = 3 levels. If a label is absent (both halves negative), bisection terminates after 1 call. Expected total calls per trace ≈ 19 × 1–4 depending on how many labels are present. Worst case (all 19 labels present throughout): 19 × (3 intermediate + 8 leaf) = ~209 calls; typical (3–5 labels present): ~30–60 calls.

---

## 4. Evaluation Protocol

### Scoring
All three variants output standard TRAIL JSON. Scored identically with `calculate_scores.py`:
- **W-F1:** weighted category F1 over the 19-class taxonomy
- **Loc:** fraction of ground-truth span_ids recovered
- **Joint:** fraction of ground-truth (category, span_id) pairs exactly matched

### What the comparison shows

All numbers for Mistral-Small-3.1-24B on GAIA_dedup.

| Condition | Localization strategy | Multi-label? | Graph? | Expected W-F1 |
|---|---|---|---|---|
| Who&When W1 | Full trace holistic | Yes | No | Similar to Baseline |
| Who&When W2 | Sequential per-span | Yes | No | Similar to Baseline |
| Who&When W3 | Per-label bisection | Yes | No | Similar to Baseline |
| **TRAIL Baseline** | Full trace holistic | Yes | No | 24.06 |
| **+GI+SI (ours)** | Full trace + causal inject | Yes | Yes (dynamic) | 30.76 |

The comparison isolates the contribution of the **causal graph** specifically. W1/W2/W3 are all multi-label and graph-free; the gap between them and +GI+SI is due to the causal structure alone. The three variants additionally show that the localization strategy (holistic vs. sequential vs. bisection) makes little difference without causal guidance.

---

## 5. Implementation Plan

### Step 1: W1 — remove single-error truncation
In `run_w1()`: remove the `errors = [errors[0]]` line. Update the prompt to use multi-label framing ("decide independently for each category").

### Step 2: W2 — remove early exit, aggregate across all spans
In `run_w2()`: remove the `break` on first `Error Found: Yes`. Continue through all spans. Collect all `(category, span_id)` pairs. After the loop, deduplicate by `(category, span_id)` and return.

### Step 3: W3 — per-label recursive bisection
Replace the current single-label bisection with `bisect_label(label, spans)` running for each of the 19 labels. Each bisect call returns `lower_half_present` and `upper_half_present` (both can be true). Recurse into all positive halves. Final localization call at leaf (1-span interval).

### Step 4: Scoring
```
python eval/calculate_scores.py --results_dir outputs/zero_shot2
```
Output directories:
```
outputs_{model}-{split}-who_and_when_w1/
outputs_{model}-{split}-who_and_when_w2/
outputs_{model}-{split}-who_and_when_w3/
```

### Step 5: Model and split
Mistral-Small-3.1-24B-Instruct on `GAIA_dedup`. Open-source only (no Gemini/GPT).
GPUs: 1,2,6,7 with `--tensor_parallel_size 4 --gpu_memory_utilization 0.34`.
- W1: `--max_model_len 131072` (full trace in context)
- W2, W3: `--max_model_len 32768` (per-call prompts are much shorter)

---

## 6. Files

```
baselines/who_and_when/
├── plan.md                         ← this file
└── run_who_and_when_vllm.py        ← runner: --variant w1|w2|w3|wcat
```
