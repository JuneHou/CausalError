# ECHO Adaptation Plan for TRAIL Benchmark

**Source paper:** *Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution*
**arXiv:** 2510.04886
**GitHub:** none — paper-only reimplementation
**Last updated:** 2026-04-29

---

## 1. Why ECHO

ECHO solves the same general problem as Who\&When (single-answer attribution: pick one `(agent, step)` per failed trace) using a different mechanism: **hierarchical context windowing + multi-persona LLM ensemble + confidence-weighted vote**. Unlike CHIEF it has no oracle dependency, no per-trace causal-graph construction, and no multi-agent assumption.

For our paper, ECHO is the cleanest competing hypothesis: *do you need a learned causal graph at all, or does multi-perspective LLM ensembling already saturate what an LLM can extract from a single trace?* If our +GI+SI method beats ECHO on TRAIL W-F1/Loc/Joint, that supports the claim that **population-level causal structure** adds signal beyond **multi-perspective ensembling**.

ECHO complements the Who\&When W1/W2/W3 baselines we already have:
- W1/W2/W3 vary the **localization strategy** (holistic / sequential / bisection) — single-LLM-call paradigm
- ECHO varies the **ensembling + context-windowing strategy** — different axis

---

## 2. Adaptation Philosophy

ECHO outputs a single `(agent, step)`; TRAIL needs 0–N `(category, span_id)` pairs. We apply the same multi-label transform we used for Who\&When:

- **Output schema change:** each analyst emits per-step, per-category likelihoods instead of one prediction
- **Voting per (category, span_id) pair:** confidence-weighted aggregation, retained if weighted vote ≥ δ
- **Drop the agent-attribution branch:** TRAIL is single-agent; the `mistake_agent` voting collapses

Beyond that, the algorithm is preserved as published.

---

## 2a. Changes from the Original ECHO Implementation

The original ECHO algorithm is described in Sections 3–4 of the paper. Verbatim summary of what's there and what we change.

### What the original does (paper §3–4)

| Aspect | Original ECHO | Our adaptation |
|---|---|---|
| **Task** | Single-answer multi-agent attribution: one `(agent, step)` per trace | Multi-label classification: 0–N `(category, span_id)` per trace |
| **Output count** | Always 1 | 0 or more |
| **Hierarchy axis** | Position-relative compression (L1 = ±1 step from candidate, L2 = ±2–3, L3 = ±4–6, L4 = rest) | Same hierarchy, span-indexed |
| **Persona panel** | 6 fixed personas; k = 3 sampled per trace | Same 6 personas, same k = 3 sampling |
| **Vote target** | (attribution_type, agent, step) | (category, span_id) per pair |
| **Confidence threshold δ** | 0.3 | 0.3 (kept) |
| **Backbone LLM** | Claude 3.5 Sonnet v2, temperatures ∈ {0.3, 0.6, 0.9} | Same temperature schedule; LLM is configurable (Mistral via vLLM, Gemini via litellm) |
| **Per-trace cost** | ~54k tokens, ~\$0.15 | Similar; depends on trace length |
| **Scoring** | Top-1 agent acc + top-1 step acc | W-F1 / Location Accuracy / Joint Accuracy via `calculate_scores.py` |

### Phase 1 changes (Hierarchical Context)
**Original:** four compression layers around each candidate step τᵢ:
- L1 (τᵢ±1): full content
- L2 (τᵢ±2..3): key reasoning chains, routine ops filtered
- L3 (τᵢ±4..6): outcome summaries, state transitions, warnings
- L4 (rest): milestone-based

**Our change:** keep the four-layer scheme. Re-define "step" as **TRAIL step span** (output of `span_level_parser.parse_trace_to_step_level`). Compression details (L2/L3/L4) are not pinned down by the paper; we make them concrete:
- **L1 (±1 step):** include the full step span content (LLM input/output messages, tool call args/results)
- **L2 (±2..3 steps):** strip span_attributes that aren't behavioral — keep `llm.input_messages`, `llm.output_messages`, `tool.name`, `tool.parameters`, `tool.output`, exception events; drop token counts, mime types, model config (this is exactly what `compress_traces.py --dedup` already does)
- **L3 (±4..6 steps):** for each step, emit a 1–2 sentence summary line: span_kind, span_name, status_code, brief outcome description (truncate llm/tool content to first 200 chars)
- **L4 (rest):** emit only milestones — first/last span, any span with status_code = ERROR, any span with exception events. One line each.

This concretization is documented as a re-implementation choice; we cite ECHO's prose descriptions and note where they leave gaps.

### Phase 2 changes (Persona Panel)
**Original:** 6 fixed personas, k = 3 sampled per analysis, temperatures spread across 0.3–0.9. Each emits investigation summary, per-step error-likelihood scores ∈ [0,1], primary conclusion, alternative hypotheses, confidence σⱼ.

**Our change:** keep the 6 personas verbatim from the paper (Conservative / Liberal / Detail-Focused / Pattern-Focused / Skeptical / General). Re-write each persona's system prompt to ask for a **per-(category, span_id) likelihood matrix** instead of a single primary conclusion. Each persona outputs:
```json
{
  "investigation_summary": "...",
  "errors": [
    {"category": "<one of TRAIL's 19>", "span_id": "<hex>",
     "likelihood": 0.xx, "evidence": "...", "description": "..."}
  ],
  "self_confidence": 0.xx
}
```

### Phase 3 changes (Consensus Voting)
**Original:** confidence-weighted vote on `(attribution_type, agent, step)`; threshold δ = 0.3.

**Our change:** vote per `(category, span_id)` pair. For each unique pair appearing in any persona's output:
- Compute weighted vote: V(c, s) = Σⱼ σⱼ · likelihoodⱼ(c, s) where the sum is over personas that emitted that pair
- Apply threshold: keep pair iff V(c, s) ≥ δ × Σⱼ σⱼ (i.e., likelihood-weighted majority)
- For evidence/description in the final output, pick the entry from the highest-confidence persona

Drop the agent-attribution voting branch entirely (TRAIL is single-agent).

---

## 3. Input/Output Format for TRAIL

### Input
Standard TRAIL trace JSON: `{trace_id, spans: [...]}` plus the corresponding entry from `processed_annotations_gaia/<trace_id>.json` for evaluation only (not at inference). Adapter A (already mostly built — see `baselines/chief/plan.md` §1) flattens spans to ordered step list with `span_id ↔ step_index` table.

### Output
Standard TRAIL JSON, identical schema to all other baselines:
```json
{
  "errors": [
    {
      "category": "<TRAIL leaf category>",
      "location": "<exact span_id hex>",
      "evidence": "<persona-supplied quote>",
      "description": "<persona-supplied one-sentence>",
      "impact": "HIGH"
    }
  ],
  "scores": []
}
```
Zero or more entries. Scored with `eval/calculate_scores.py` unchanged.

---

## 4. Algorithm in Detail

### Phase 1 — Build Hierarchical Context Windows

For a trace with N step spans `[s_1, s_2, ..., s_N]`:

```python
def build_context(trace_steps, anchor_idx):
    L1 = [full_content(s) for s in trace_steps[anchor_idx-1 : anchor_idx+2]]
    L2 = [reasoning_filter(s) for s in trace_steps[anchor_idx-3 : anchor_idx+4] - L1]
    L3 = [outcome_summary(s) for s in trace_steps[anchor_idx-6 : anchor_idx+7] - L1 - L2]
    L4 = [milestone(s) for s in trace_steps - L1 - L2 - L3]
    return {"L1": L1, "L2": L2, "L3": L3, "L4": L4}
```

For TRAIL, since we're emitting per-(category, span_id) likelihoods rather than focusing on one anchor, we build **one global hierarchical view** per trace, then run all personas over it. The "anchor" idea is collapsed because every span is a candidate. (The paper's anchor was implicit because it was searching for the single failure step.) Concretely: L1 contains the full content of every step, L2 = compressed-attributes view, L3 = summary view, L4 = milestone view, and the prompt presents all four layers stacked.

**Trade-off:** the global view is larger than a per-anchor view but avoids running personas N times. If trace is too long (>token budget), fall back to anchored windows where each persona is asked about a sliding window of K consecutive spans.

### Phase 2 — Run Persona Panel

```python
PERSONAS = [
    "Conservative", "Liberal", "Detail-Focused",
    "Pattern-Focused", "Skeptical", "General",
]

def run_panel(trace_id, context, k=3, temps=(0.3, 0.6, 0.9), seed=None):
    sampled = random.sample(PERSONAS, k)
    outputs = []
    for persona, temp in zip(sampled, temps):
        prompt = build_persona_prompt(persona, context, taxonomy)
        response = call_llm(prompt, temperature=temp)
        parsed = parse_json(response)
        outputs.append({"persona": persona, "temp": temp, **parsed})
    return outputs
```

**Persona system prompts (verbatim wording adapted from paper §3.2):**
- *Conservative*: "Require strong, clear evidence before attributing an error. Prefer single-error attributions where the evidence is unambiguous. Maintain high confidence thresholds. Output high likelihood (>0.7) only when the trace contains direct, unmistakable evidence."
- *Liberal*: "Consider multi-error scenarios. Identify subtle error patterns. Accept moderate confidence thresholds. Output likelihoods broadly across categories that show even partial evidence."
- *Detail-Focused*: "Examine specific evidence and exact wording in tool calls, LLM outputs, and error messages. Identify subtle inconsistencies in agent reasoning."
- *Pattern-Focused*: "Recognize broader reasoning chains across the trace. Track how earlier reasoning steps propagate into later errors."
- *Skeptical*: "Question underlying assumptions. Explore alternative explanations for each apparent error. Flag categories that are plausible but uncertain with moderate likelihoods."
- *General*: "Maintain a balanced perspective. Consider all evidence types equally. Use no preferred error category."

**Per-persona prompt template (see §6 for full text):**
```
You are {persona_role}: {persona_description}

You will see an agent execution trace organized into four context layers
(L1 = full detail, L2 = filtered, L3 = summarized, L4 = milestones).

Identify all errors in the trace from the TRAIL taxonomy.
For each (error_category, span_id) you believe is present, output a
likelihood score in [0,1] reflecting your confidence.
Also report your overall analysis confidence (self_confidence ∈ [0,1]).

TRAIL Taxonomy:
{taxonomy_block}

Trace context:
L1 (full detail): {L1}
L2 (filtered): {L2}
L3 (summary): {L3}
L4 (milestones): {L4}

Output strictly valid JSON: {schema}
```

### Phase 3 — Confidence-Weighted Consensus Vote

```python
def consensus(outputs, delta=0.3):
    # Collect all (category, span_id) pairs across personas
    pair_votes = defaultdict(list)  # (cat, span) -> [(persona_conf, likelihood, entry)]
    for o in outputs:
        sigma = o["self_confidence"]
        for e in o["errors"]:
            key = (e["category"], e["span_id"])
            pair_votes[key].append((sigma, e["likelihood"], e))

    total_conf = sum(o["self_confidence"] for o in outputs)
    final_errors = []
    for (cat, span), votes in pair_votes.items():
        weighted = sum(sigma * lik for sigma, lik, _ in votes)
        if weighted >= delta * total_conf:
            best = max(votes, key=lambda v: v[0] * v[1])
            final_errors.append({
                "category": cat,
                "location": span,
                "evidence": best[2]["evidence"],
                "description": best[2]["description"],
                "impact": "HIGH",
            })
    return final_errors
```

The δ × Σσ threshold is the multi-label generalization of ECHO's single-prediction δ = 0.3 floor.

---

## 5. Cost Model

| Component | Cost |
|---|---|
| LLM calls per trace | k = 3 (one per sampled persona) |
| Per-call prompt size | ~30k tokens (full hierarchical view of trace) for GAIA, larger for SWE Bench |
| Per-trace token cost | ~90k tokens (3 × 30k) |
| Total cost vs ECHO paper | Roughly 2× the paper's ~54k figure, because we ask for full per-pair likelihoods rather than a single attribution |
| LLM | Mistral-Small-3.1-24B (vLLM) or Gemini-2.5-pro (litellm) |

If cost is a concern, two reductions:
- **k=3 → k=1** ablation (single-persona, no ensembling) — measures the contribution of the persona ensemble vs. just the hierarchical context
- **Anchored windowing** when trace exceeds budget: split trace into overlapping windows of M=10 spans, run each persona over each window, vote across all (window, persona) outputs

---

## 6. Implementation Plan

### Files
```
baselines/echo/
├── plan.md                ← this file
└── run_echo.py            ← runner: --backend {vllm, litellm} --model ... --split GAIA_dedup
```

(Single file; no separate vLLM/litellm scripts. Backend chosen by `--backend` flag.)

### Steps
1. **Adapter A reuse:** Import the existing trace loading + flattening from `benchmarking/span_level_parser.py`. No new code.
2. **Hierarchical view builder:** Implement `build_context(trace_steps)` producing the L1/L2/L3/L4 view. ~50 lines.
3. **Persona prompt builder:** Define 6 persona templates as a dict; assemble per-persona prompt with taxonomy + context. ~30 lines.
4. **Panel runner:** Sample k=3 personas, dispatch parallel LLM calls (litellm `ThreadPoolExecutor` or vLLM batched generate). ~40 lines.
5. **Voting:** Implement `consensus(outputs, delta=0.3)` per §4. ~30 lines.
6. **Output:** Write per-trace TRAIL JSON to `outputs/zero_shot2/outputs_{model}-{split}-echo/`.

Total estimated size: ~150–200 lines plus boilerplate.

### Hyperparameters (initial defaults from paper)
- `k = 3` personas sampled from 6
- `temperatures = [0.3, 0.6, 0.9]` mapped to the k sampled personas
- `delta = 0.3` confidence threshold
- Random seed: configurable via `--seed`; default 42 for reproducibility

### Run targets
Mirror the Who&When experiment matrix:
```bash
# Mistral on GAIA_dedup
CUDA_VISIBLE_DEVICES=1,2,6,7 python baselines/echo/run_echo.py \
    --backend vllm \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split GAIA_dedup \
    --output_dir outputs/zero_shot2

# Mistral on SWE_Bench_dedup
CUDA_VISIBLE_DEVICES=1,2,6,7 python baselines/echo/run_echo.py \
    --backend vllm \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --split SWE_Bench_dedup \
    --output_dir outputs/zero_shot2

# Gemini on GAIA_dedup (optional — for closed-source cross-check)
python baselines/echo/run_echo.py \
    --backend litellm \
    --model gemini/gemini-2.5-pro \
    --split GAIA_dedup \
    --output_dir outputs/zero_shot2
```

### Scoring
```bash
python eval/calculate_scores.py --results_dir outputs/zero_shot2
```
Output directories follow the same convention as Who&When and CHIEF baselines:
```
outputs_{model_tag}-{split}-echo/
outputs_{model_tag}-{split}-echo-metrics.txt
```

---

## 7. Expected Result Comparison

All numbers for Mistral-Small-3.1-24B on GAIA_dedup. ECHO numbers are predictions, not measured.

| Condition | Multi-persona? | Hierarchical context? | Causal graph? | Expected W-F1 |
|---|---|---|---|---|
| TRAIL Baseline | No | No | No | 24.06 (measured) |
| Who&When W1 | No | No | No | ≈ Baseline |
| **ECHO** | **Yes (k=3)** | **Yes (L1–L4)** | **No** | unknown — interesting |
| **+GI+SI (ours)** | No | No | **Yes (cross-trace, validated)** | 30.76 (measured) |

**Two informative outcomes:**
1. ECHO ≪ +GI+SI → causal graph adds signal beyond persona ensembling. Strong support for our contribution.
2. ECHO ≈ +GI+SI → the gain is in *how the LLM is queried*, not the graph itself. We'd need to discuss this honestly in the paper.

Both outcomes are publishable — they just lead to different framings.

---

## 8. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Paper's L2/L3/L4 compression is under-specified — our concretization may differ from authors' implementation | Document the choice. Cite paper prose and explicitly note re-implementation. |
| Per-pair likelihood output may be unstable across personas (LLMs disagree on `span_id`s) | Use the `span_index` block from our existing pipeline so all personas reference the same span_id table |
| 19 categories × N spans = large output matrix; LLM may truncate or hallucinate span_ids | Constrain output to JSON list (only emit non-zero pairs); validate span_ids against trace's span_id set; drop hallucinated ones |
| k = 3 randomly sampled may give variance across runs | Fix `--seed` for the main results; report variance via 3-seed average if reviewers ask |
| Cost: 3 calls × full-trace prompt may exceed budget for SWE_Bench_dedup | Implement anchored windowing as fallback (see §5) |

---

## 9. Reproducibility note for the paper

In §4 (Experiments) we cite ECHO and explicitly note: "We re-implement ECHO from the paper (arXiv 2510.04886); no public code is available. We preserve the 6-persona panel, k=3 random sampling, temperature schedule {0.3, 0.6, 0.9}, and confidence threshold δ=0.3 specified in §3 of the paper. The L2–L4 context-compression heuristics are not fully specified by the paper; we make concrete the following choices: (i) L2 = compressed span attributes (drop token counts, model config, mime types); (ii) L3 = 1–2 sentence summary per step; (iii) L4 = milestone-only listing. Single-answer ECHO is generalized to multi-label by per-(category, span_id) confidence-weighted voting under threshold δ × Σσ."

This makes our re-implementation auditable and signals to reviewers that the comparison is in good faith.
