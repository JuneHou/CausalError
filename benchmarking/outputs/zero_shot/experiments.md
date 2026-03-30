# Experiment: Causal Graph Guidance for LLM-as-Judge Evaluation

**Model**: Gemini-2.5-Flash
**Dataset**: GAIA split (117 traces; 8 skipped due to context window overflow, 109 scored)
**Comparison**: `run_eval.py` (baseline) vs. `run_eval_with_graph.py --causal_only` (causal graph guided)
**Graph**: 11 fully validated causal edges (bootstrap stability = 1.0) from the Suppes causal graph built over TRAIL training traces

---

## Overall Metrics

| Metric | Baseline | + Causal Graph | Δ |
|--------|----------|---------------|---|
| Weighted F1 | 0.4163 | **0.4504** | +0.0341 |
| Location Accuracy | 0.3925 | **0.4030** | +0.0105 |
| Joint Accuracy | 0.1457 | **0.1666** | +0.0209 |

All three metrics improve with causal graph guidance.

---

## Score Correlations with Human Annotations (Pearson r)

| Score Type | Baseline | + Causal Graph | Δ |
|------------|----------|---------------|---|
| Reliability | 0.5554 | **0.6221** | +0.0667 |
| Instruction Adherence | **0.3622** | 0.2590 | −0.1032 |
| Plan Optimization | **0.2980** | 0.2952 | −0.0028 |
| Overall | **0.4935** | 0.4673 | −0.0262 |

Reliability correlation improves notably. The other scores slip slightly — the graph likely causes the judge to flag more errors, which adds noise to the subjective quality scores.

---

## Per-Category F1 (★ = type appears in a causal graph edge)

| Category | Baseline | + Causal Graph | Δ |
|----------|---------|---------------|---|
| ★ Context Handling Failures | 0.0952 | **0.3478** | **+0.2526** |
| ★ Incorrect Problem Identification | 0.0769 | **0.2667** | **+0.1898** |
| ★ Tool Selection Errors | 0.1905 | **0.3529** | **+0.1624** |
| ★ Tool-related | 0.2800 | **0.3871** | **+0.1071** |
| ★ Task Orchestration | 0.1667 | **0.2400** | **+0.0733** |
| Instruction Non-compliance | 0.5091 | **0.5714** | +0.0623 |
| Tool Definition Issues | 0.3333 | **0.4000** | +0.0667 |
| ★ Formatting Errors | 0.6349 | **0.6567** | +0.0218 |
| ★ Language-only | 0.5417 | 0.5376 | −0.0041 |
| ★ Resource Abuse | 0.6792 | 0.6667 | −0.0125 |
| ★ Goal Deviation | 0.4691 | 0.4524 | −0.0167 |
| ★ Poor Information Retrieval | 0.5789 | 0.5116 | −0.0673 |
| Environment Setup Errors | 0.2000 | 0.0000 | −0.2000 |
| ★ Tool Output Misinterpretation | 0.4000 | 0.1600 | −0.2400 |
| ★ Authentication Errors | 0.8889 | 0.3333 | −0.5556 |

---

## Causal Graph Edges (causal_only mode)

The 11 validated edges injected into the prompt as guidance:

| Source Error | → | Consequent Error |
|---|---|---|
| Formatting Errors | → | Context Handling Failures |
| Formatting Errors | → | Incorrect Problem Identification |
| Formatting Errors | → | Resource Abuse |
| Formatting Errors | → | Tool Output Misinterpretation |
| Incorrect Problem Identification | → | Tool Output Misinterpretation |
| Poor Information Retrieval | → | Resource Abuse |
| Resource Abuse | → | Authentication Errors |
| Resource Abuse | → | Tool-related |
| Task Orchestration | → | Context Handling Failures |
| Tool Selection Errors | → | Goal Deviation |
| Tool Selection Errors | → | Language-only |

---

## Summary

The 11 causal edges involve 12 error types. Among those, **5 show large gains** — all categories that were poorly detected at baseline (F1 < 0.20): `Context Handling Failures`, `Incorrect Problem Identification`, `Tool Selection Errors`, `Tool-related`, `Task Orchestration`. The graph explicitly chains these as downstream consequences of other errors, prompting the LLM to actively look for them.

The two notable regressions — `Authentication Errors` (−0.56) and `Tool Output Misinterpretation` (−0.24) — were already well-detected at baseline. The graph guidance appears to cause over-detection of their causal neighbors, reducing precision on these categories.

Overall, the causal graph is effective at surfacing **underdetected, cascading error types** with minimal overhead (11 edges, <100 tokens added to the prompt). The improvement in weighted F1 (+0.034) is driven by gains on previously low-performing categories rather than uniform uplift across all types.
