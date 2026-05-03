# Related Work Replication Assessment

**Date:** 2026-04-28  
**Purpose:** Assess which related methods to replicate on TRAIL for comparison with our causal graph-guided evaluation pipeline (+GI+SI).

---

## Methods Evaluated

### 1. AgentRx
**Paper:** "AgentRx: Diagnosing AI Agent Failures from Execution Trajectories"  
**Authors:** Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal (Microsoft Research)  
**arXiv:** 2602.02475  
**GitHub:** https://github.com/microsoft/AgentRx  
**Dataset:** https://huggingface.co/datasets/microsoft/AgentRx

**What it does:**  
6-stage pipeline: IR normalization → static constraint generation → dynamic constraint generation → guarded step-by-step evaluation → LLM judge → report. The LLM judge receives a "validation log" (evidence of constraint violations) plus a failure taxonomy and outputs the critical failure step and category.

**Failure taxonomy:** 9 categories (Plan Adherence Failure, Invention of New Information, Invalid Invocation, Misinterpretation of Tool Output, Intent–Plan Misalignment, Under-specified User Intent, Intent Not Supported, Guardrails Triggered, System Failure).

**Results on own benchmark:** +23.6% absolute improvement in failure localization vs. prompting baselines, across 115 annotated trajectories (τ-bench, Flash, Magentic-One).

**Critical finding:** NOT tested on TRAIL. Uses a different 9-category taxonomy and its own benchmark format.

**Replication strategy for TRAIL:**  
- Skip stages 1–4 (constraint generation and guarded evaluation — these are domain-specific)
- Extract the LLM judge prompt from `src/judge/` in the GitHub repo
- Substitute TRAIL's 19-category taxonomy into the judge prompt
- Adapt output format to TRAIL's `(category, span_id)` pair format
- Run as a single-pass LLM-as-judge call

**Effort:** Medium  
**Recommendation:** **Best main non-causal baseline.** Replicate the adapted judge variant.

---

### 2. Who&When Prompting
**Paper:** "Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems"  
**Authors:** Shaokun Zhang, Ming Yin, Jieyu Zhang, et al.  
**arXiv:** 2505.00212 (ICML 2025 Spotlight)  
**Dataset:** https://huggingface.co/datasets/Kevin355/Who_and_When  
**Code:** Available (open source)

**What it does:**  
Dataset paper + prompting evaluation for multi-agent failure attribution. Formalizes two questions: (1) *Who* — which agent caused the failure? (2) *When* — at which step did the failure occur? Evaluates structured prompting baselines; best results: 53.5% agent-level accuracy, 14.2% step-level accuracy. Even SOTA reasoning models (o1, DeepSeek R1) fail to achieve practical usability.

**Critical finding:** Designed for multi-agent attribution (blame assignment across multiple agents), not single-agent multi-label error classification. Not tested on TRAIL. Their prompting approach is structurally very close to TRAIL's own zero-shot baseline.

**Replication strategy for TRAIL:**  
- Implement as a minimal zero-shot prompt: "identify all error types present and at which span each first occurs" — no graph, no span index
- Serves as "naive zero-shot prompting" anchor, confirming that our baseline is already competitive with the SOTA in simple prompting

**Effort:** Low  
**Recommendation:** **Simple prompting baseline.** Confirms the floor; low differentiation from TRAIL's own baseline.

---

### 3. CDC-MAS
**Paper:** "Automatic Failure Attribution and Critical Step Prediction Method for Multi-Agent Systems Based on Causal Inference"  
**Authors:** Guoqing Ma, Jia Zhu, Hanghui Guo, et al.  
**arXiv:** 2509.08682  
**GitHub:** None found (not publicly available)

**What it does:**  
Causal discovery algorithm for failure attribution in multi-agent systems. Four phases: (1) context-aware feature preparation using a Transformer encoder; (2) temporal causal structure discovery via conditional independence tests; (3) confounding-aware edge orientation using context as a confounder proxy; (4) causal path analysis and ranking via Shapley values.

**Results on TRAIL-GAIA:** 44.6% step-level accuracy (vs. 24.8% baseline). On SWE-Bench: 14.3% (vs. 10.5%).

**Critical finding:** Their metric is **top-1 step exact match** (single critical failure step), NOT TRAIL's W-F1/Loc/Joint. As noted in our paper (§6): "a model that correctly identifies only the onset step would score 1.0 under CDC-MAS but only 0.10 location accuracy under TRAIL on a trace with 10 annotated error spans." The evaluations are fundamentally incomparable. No public code.

**Effort:** Very high (no code; requires full causal discovery infrastructure with Transformer encoder + Shapley values)  
**Recommendation:** **Related work discussion only.** Cite results in text with explicit metric incompatibility caveat; do not attempt replication.

---

### 4. CHIEF
**Paper:** "From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems"  
**Authors:** Yawen Wang, Wenjie Wu, Junjie Wang, Qing Wang  
**arXiv:** 2602.23701  
**GitHub:** Anonymous link referenced in paper; not publicly discoverable

**What it does:**  
3-module framework: (1) hierarchical causal graph construction via RAG-based task decomposition + OTAR (Observation-Thought-Action-Result) parsing; (2) oracle-guided backtracking that generates virtual oracles per subtask and narrows failure scope top-down; (3) counterfactual attribution via four progressive filters (local, planning-control, data-flow, deviation-aware). Evaluated on Who&When benchmark: 77.59%/29.31% agent/step accuracy.

**Critical finding:** NOT tested on TRAIL. Evaluated only on Who&When. No accessible code. The hierarchical causal graph framing is very close to our own method — including it as a baseline would blur the contribution story.

**Effort:** Very high (no code, not on TRAIL, 3-module system)  
**Recommendation:** **Avoid as baseline.** Mention in related work to distinguish our cross-trace error-type DAG from their per-trace step-level DAG.

---

## Summary Table

| Method | Replicate? | Effort | Tested on TRAIL | Code Available | Recommendation |
|--------|-----------|--------|-----------------|---------------|----------------|
| AgentRx | **Yes — main baseline** | Medium | No (adapt judge) | Yes (GitHub) | Best non-causal baseline |
| Who&When | **Yes — trivial baseline** | Low | No (adapt prompt) | Yes | Zero-shot prompting anchor |
| CDC-MAS | No | Very high | Yes (diff. metrics) | No | Related work, cite with caveat |
| CHIEF | No | Very high | No | No | Related work only |

---

## Concrete Next Steps

1. **AgentRx adapted judge:** Read `src/judge/` from https://github.com/microsoft/AgentRx, extract the judge prompt structure, substitute TRAIL's 19-category taxonomy, adapt output to `(category, span_id)` pairs, run as single-pass LLM-as-judge.

2. **Who&When zero-shot prompt:** Implement a minimal prompt asking the LLM to identify error types and their first-occurrence span — no graph, no span index, no chain-of-thought.

3. **CDC-MAS in paper:** Discuss in §6 (Related Work), noting the metric incompatibility. Their 44.6% step accuracy ≠ our location accuracy metric.

4. **CHIEF in paper:** Distinguish our method from CHIEF by emphasizing: (a) cross-trace DAG vs. per-trace DAG; (b) intervention-validated edges vs. observational graph; (c) LLM-as-judge augmentation vs. step localization.
