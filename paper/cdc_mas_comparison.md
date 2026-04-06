# Comparison with CDC-MAS (Ma et al., 2025)

**Paper**: "Automatic Failure Attribution and Critical Step Prediction Method for Multi-Agent Systems Based on Causal Inference"  
**arXiv**: 2509.08682

---

## One-Paragraph Summary (for Related Work)

Ma et al. (2025) propose CDC-MAS, a causal-inference-based framework for failure attribution in multi-agent systems. While CDC-MAS also uses the TRAIL-GAIA benchmark and reports a step-level accuracy of 44.6%, the two methods are **not directly comparable** due to task scope differences. CDC-MAS adopts the Who&When (Zhang et al., 2025) evaluation protocol, which measures top-1 exact match on a single "decisive error step" — a binary, single-answer prediction. Our method targets the full TRAIL task: identifying **all** error locations (multi-label recall over all annotated spans), **all** error types (Category F1 over 19 taxonomy classes), and their joint prediction (Joint Accuracy). Crucially, CDC-MAS does not report Category F1, as error type classification is outside their task definition. A concrete example illustrates the gap: in a trace with 13 annotated errors across 10 spans and 5 error types, CDC-MAS is evaluated on whether a model correctly identifies the single onset step (binary 0/1), while our Location Accuracy scores partial credit across all 10 spans and our Category F1 requires predicting all 5 error types. The two evaluations measure different aspects of diagnostic completeness and cannot be reduced to a common number.

---

## Key Differences Table

| Dimension | CDC-MAS (Ma et al., 2025) | Our Method |
|---|---|---|
| Metric source | Who&When (Zhang et al., 2025b) | TRAIL (Deshpande et al., 2025) |
| Location metric | Top-1 step exact match (binary) | Recall over all GT spans |
| Category metric | Not reported | Weighted F1 over 19 classes |
| Joint metric | Not reported | Joint (span + type) accuracy |
| # answers expected | 1 (decisive step) | All errors in the trace |
| Partial credit | No | Yes |
| TRAIL-GAIA number | 44.6% step accuracy | 0.450 W-F1 / 0.403 Loc / 0.167 Joint (Gemini+causal) |
| Causal graph type | Per-trace step-level DAG | Cross-trace error-type Suppes graph |
| LLM role | Baseline only; causal inference is primary | LLM is the classifier; graph shapes prompting |

---

## Metric Incomparability — Concrete Example

**Trace**: `59365b27641e501d105b0e8f5e7c5af7` (GAIA)  
**Task**: Count Mercedes Sosa studio albums. Agent repeatedly called `page_down` with wrong arguments across 10 steps.

**Ground truth**: 13 errors, 10 unique spans, 5 error types  
(Formatting Errors ×9, Tool-related, Context Handling Failures, Resource Abuse, Poor Info Retrieval)

**CDC-MAS evaluation**: predict the single decisive step (onset step 24, first Formatting Error).  
→ Score = 1.0 if correct, 0.0 if wrong. One prediction, binary.

**Our evaluation**:
- Location Accuracy = |predicted spans ∩ GT spans| / 10 → partial credit for finding each of the 10 spans
- Category F1 → must identify all 5 error types
- Joint Accuracy → must get both span and type correct per error

A model that correctly identifies only the onset step (step 24) scores 1.0 under CDC-MAS but only 0.10 location accuracy under TRAIL (1/10 spans found).

---

## Suggested Related Work Sentence

> Ma et al. (2025) apply causal inference to multi-agent failure attribution on the TRAIL benchmark, reporting 44.6% step-level accuracy on GAIA. However, their evaluation follows the Who&When protocol (Zhang et al., 2025) — top-1 binary prediction of a single decisive error step — and does not report error type classification metrics. Our work targets the full TRAIL evaluation: multi-label error localization, 19-class category F1, and joint accuracy, which constitute a strictly broader and harder task.
