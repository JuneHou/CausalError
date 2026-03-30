# Uncertainty Quantification for LLM-as-a-Judge: Brainstorm

---

## 1. Literature Review

### Paper 1: SAUP — Situation Awareness Uncertainty Propagation (ACL 2025)
**Link**: https://aclanthology.org/2025.acl-long.302

**Data**
- Evaluated on multi-step QA and agent benchmarks where the final answer is produced after a chain of reasoning steps.
- Each trace is a sequence of reasoning steps (tool calls, intermediate conclusions).

**Model**
- Any existing one-step uncertainty estimator (e.g., token-level entropy, semantic similarity across samples) plugged in at each step.
- A "situation awareness" scorer assigns importance weights to each step based on its estimated contribution to the final outcome.

**Workflow**
1. Run LLM agent on a task, producing a step-by-step trace.
2. At each step t, compute local uncertainty u(t) using a base estimator.
3. Compute situational importance s(t) — how consequential is this step to the final output?
4. Aggregate: `U_final = Σ_t s(t) × u(t)` — a weighted sum propagated forward along the chain.
5. Use U_final to flag unreliable traces or calibrate confidence in the final answer.

**Implication for our project**
- Our agent traces ARE multi-step reasoning chains. Each span (LLM call or tool call) is a reasoning step.
- SAUP's propagation structure maps directly: uncertainty at an early span (e.g., a misidentified problem) should increase uncertainty in later spans (downstream errors).
- The causal graph edges define exactly these propagation paths — they encode which earlier errors cause later errors, i.e., the "situation awareness" structure.
- SAUP treats time as the chain; we replace time with causal order.

---

### Paper 2: C3 — Consistency-based Confidence Calibration (ACL 2025)
**Link**: https://aclanthology.org/2025.acl-long.1184

**Data**
- Evaluated on NQ (open-domain QA), HotpotQA (multi-hop QA), MMLU (multiple-choice).
- Single-turn question answering; the model either knows the answer or it does not.

**Model**
- Any autoregressive LLM with access to pre-generation hidden layer representations.
- A lightweight probe trained on top of hidden states to predict "known / unknown" before any token is generated.
- Combined with question reformulation to measure output consistency across surface-level prompt variations.

**Workflow**
1. Reformulate the input question into K paraphrases.
2. For each paraphrase, extract the hidden representation before generation begins.
3. Measure consistency of the model's confidence across the K reformulations — inconsistency signals uncertain knowledge.
4. Optionally combine with post-generation token probability to produce a calibrated final confidence score.
5. Use the score for selective generation (abstain if confidence below threshold).

**Implication for our project**
- The reformulation consistency idea (C3) is model-agnostic at the output level: run the judge on the same trace with K prompt variants, measure consistency of which errors are detected.
- This gives an external, post-hoc confidence proxy that does not require access to internal states.
- The pre-generation hidden state probe is the internal mechanistic component — requires white-box access to the judge.
- Key insight: the "known/unknown" boundary maps to "error detected confidently / error detection uncertain" in our setting.

---

### Resource 3: ACL 2025 Tutorial — Uncertainty Quantification for LLMs
**Link**: https://sites.google.com/view/acl2025-uncertainty-for-llms/
**Toolkit**: LM-Polygraph (unifies 12+ UQ methods)

**Scope**
- Systematic survey of why classification-era UQ does not transfer to generation.
- Covers white-box (entropy, hidden states, attention), black-box (consistency, self-evaluation), and calibration methods.
- Selective generation: use uncertainty to decide when to abstain or defer to a stronger model.

**Workflow covered**
1. White-box: extract token entropy, attention entropy, or probed hidden states during generation.
2. Black-box: sample N outputs, measure semantic or exact-match consistency.
3. Calibration: map raw uncertainty scores to well-calibrated probabilities.
4. Selective generation: set a threshold; below it, abstain or escalate.

**Implication for our project**
- LM-Polygraph provides off-the-shelf implementations of consistency-based and entropy-based estimators.
- The selective generation idea is directly applicable: if the judge's confidence in detecting a specific error type is below threshold, escalate to a second LLM pass with targeted prompting.
- The calibration step is important: raw consistency scores need calibration before they can be used to weight causal graph propagation.

---

## 2. Proposed Method: Causal-Graph-Structured Uncertainty Propagation

### What Is Novel vs. Prior Work

| | SAUP | C3 | Our System |
|---|---|---|---|
| Graph structure | Linear chain (time steps) | None | Causal DAG (error types) |
| Uncertainty source | Per-step estimator | Internal hidden states | LLM output confidence / consistency |
| Propagation | Forward along time | None | Forward along causal edges |
| Goal | Final answer reliability | Knowledge boundary | Error detection completeness |

**Core novelty**: causal-graph-structured uncertainty propagation for LLM agent evaluation — uncertainty about one error type propagates to calibrate search intensity for causally downstream error types.

SAUP propagates uncertainty forward in time. We propagate forward along causal edges. The causal graph replaces the temporal chain as the propagation backbone, enabling non-linear, branching propagation paths that match how errors actually cause other errors in agent traces.

### Full Proposed Workflow

```
Input: agent trace T
       causal graph G = {A → B, ...} with bootstrap-validated edges

Step 1 — First-pass judge call
  Run judge LLM on T (zero-shot or with causal graph edges injected).
  Output: detected errors E1 = {(category_i, location_i, impact_i)}

Step 2 — Confidence estimation (external, post-hoc)
  For each error category c in the taxonomy:
    Option A (single pass): use impact level (HIGH=1.0, MEDIUM=0.67, LOW=0.33) as proxy.
    Option B (multi-sample): run judge K=3–5 times with prompt perturbations;
                             confidence(c) = fraction of runs that detect c.
  Result: confidence vector conf = {c: score} for all detected and undetected categories.

Step 3 — Causal graph propagation
  For each edge A → B in G:
    boosted_score(B) += conf(A) × edge_weight(A→B)
  edge_weight: bootstrap stability score (already computed from our Suppes graph).
  Result: boosted_score(B) reflects how much evidence for B is implied by upstream detections.

Step 4 — Threshold and select
  For each category B not detected in E1:
    if boosted_score(B) > propagation_threshold:
      flag B as "verify in pass 2"
  For each category B already detected in E1:
    if boosted_score(B) is high: reinforce (skip re-verification)
    if boosted_score(B) is low but conf(B) was low: include in "verify in pass 2"

Step 5 — Second-pass judge call (targeted)
  Construct a targeted prompt:
    "In your first analysis, the following errors were detected: [E1].
     Based on causal relationships, the following error types are likely
     present but may have been missed. Please specifically verify whether
     each is present, providing span_id location and evidence: [flagged categories]."
  Run judge on T again with this targeted prompt.
  Output: refined error set E2.

Step 6 — Merge and output
  Merge E1 and E2 (deduplicate by category + location).
  Output final error list with scores.
```

### Why Two Passes Are Needed

The first-pass LLM output provides the uncertainty signal (Step 2) that drives causal propagation (Step 3). The second pass acts on the propagated signal to close detection gaps. Without two passes, there is no signal to propagate — the propagation is contingent on first observing which errors were detected and with what confidence.

### Connection to Existing Results

Our current causal graph injection (run_eval_with_graph.py) is a degenerate version of this: it injects all edges with uniform weight, with no uncertainty estimation, effectively setting conf(A) = 1.0 for all A regardless of whether A was detected. The proposed method adds:
1. Instance-specific confidence (conf varies per trace)
2. Edge weighting by bootstrap stability
3. Selective second pass (only re-verify what propagation flags, not everything)

This explains the existing mixed results: categories where A was not present in the trace still had their downstream edges injected, adding noise. Propagation conditioned on per-trace confidence would suppress those edges.

---

## 3. The Internal vs. External Coherence Problem

### Why Internal Mechanistic Signals Are Attractive

- Attention weights when the judge generates a specific error category reveal which input spans it "read" to reach that decision — directly useful for location prediction.
- Hidden representations before generating each error entry encode how committed the model is to that prediction — a richer confidence signal than output-level proxies.
- Attention co-activation: if the judge's attention when predicting error A and error B consistently focuses on the same spans across traces, this is mechanistic evidence for the A→B causal edge — potentially allowing the causal graph itself to be learned or validated from judge internals.

### The Coherence Constraint

**Internal mechanistic signal and final prediction must come from the same model.**

If internal states are extracted from open-source model M1 (e.g., Llama-3.1-70B) but the final prediction is made by closed model M2 (e.g., Gemini 2.5 Flash):
- conf(A) reflects M1's uncertainty surface, not M2's
- M1 and M2 have different representations, different failure modes, different training distributions
- Propagating M1's confidence to guide M2's second pass assumes aligned uncertainty surfaces — this is not guaranteed and likely incorrect
- The internal and external signals are measuring different processes

**Conclusion**: the internal mechanistic angle is only coherent if the same open-source model handles both signal extraction and final prediction end-to-end. This is a quality trade-off: open-source models are currently weaker than Gemini 2.5 Flash on complex multi-step trace evaluation.

### Two Coherent Architecture Choices

**Architecture A: Open-source model, end-to-end**
```
Llama-3.1-70B (or Qwen-2.5-72B)
  ├── Pass 1: generate draft error predictions
  ├── Extract: attention weights → span-level location confidence
  │            hidden states → per-error-type confidence scores
  │            attention co-activation → edge weight calibration
  ├── Propagate: through causal graph using extracted confidence
  └── Pass 2: targeted re-verification prompt → final predictions
```
- Coherent: same model, signal and prediction aligned
- Feasible: open-source models can be run locally with attention/hidden state access
- Risk: lower prediction quality than closed models
- Research value: mechanistic interpretability of error detection; causal edge validation from attention

**Architecture B: Closed model, external uncertainty only (recommended for benchmark performance)**
```
Gemini 2.5 Flash
  ├── Pass 1: generate draft error predictions
  ├── Estimate confidence from own outputs:
  │     - multi-sample consistency (K=3 runs with prompt variants)
  │     - impact distribution (HIGH/MEDIUM/LOW as confidence proxy)
  │     - self-reported confidence (explicit confidence request in prompt)
  ├── Propagate: through causal graph using output-level confidence
  └── Pass 2: targeted re-verification → final predictions
```
- Coherent: same model, signal and prediction aligned
- Maintains closed-model quality
- No internal access needed — fully API-compatible
- External signal is weaker (noisier) than internal mechanistic signal
- K=3 multi-sample adds cost (~3× API calls per trace)

### Recommended Path

For **benchmark performance** (TRAIL metrics): Architecture B.
For **research insight** (understanding error detection mechanics, validating causal edges): Architecture A.

Both can be run and compared. Architecture A provides interpretability findings (which spans the judge attends to for each error type, whether attention co-activation mirrors causal edges) that can motivate and validate the causal structure used in Architecture B.

The combined contribution: Architecture A reveals the mechanism; Architecture B operationalizes it efficiently at scale.
