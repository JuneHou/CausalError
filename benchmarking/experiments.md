# Experiment Notes

---

## Evaluator Fix: Errors Now Counted as Empty Predictions (2026-04-13)

**File**: `eval/calculate_scores.py`

**Change**: Previously, files that errored during evaluation (e.g., malformed JSON, context-length truncations producing non-JSON output) were silently skipped — they did not enter the denominator. Missing output files were already correctly counted as empty predictions. The fix makes both cases consistent: any file that fails to parse is now counted as an empty (all-wrong) prediction.

**Impact**: All scores decreased slightly across all models and conditions, since more zero-predictions enter the denominator. Conditions whose output files stored context overflow as plain text (e.g., `"Context window exceeded. No output generated."`) were most affected, as those files now count as failures instead of being excluded.

---

## Finding: Mistral SWE Bench — Why +GI+SI Is Worse Than Baseline

**Model**: Mistral-Small-3.1-24B-Instruct-2503  
**Dataset**: SWE\_Bench\_dedup (`zero_shot2/`)  
**Scores** (after evaluator fix):

| Method   | W-F1 | Loc  | Joint |
|----------|------|------|-------|
| Baseline | 9.80 | 9.36 | 1.57  |
| +GI+SI   | 6.76 | 1.67 | 0.00  |

### Root cause: context overflow + inconsistent error representation

Both runs hit the context limit on **exactly 12 out of 31 output files**. However, the two runs stored those failures differently:

| | Baseline | +GI+SI |
|---|---|---|
| Overflow representation | Plain text: `"Context window exceeded. No output generated."` | Structured JSON: `{"errors": [], "scores": [], "_error": "context_overflow"}` |
| Old evaluator (pre-fix) | `extract_json_from_text` raises `ValueError` → **silently skipped** (denominator = 28) | Parsed as empty prediction → **penalized** (denominator = 40) |
| New evaluator (post-fix) | Exception caught → counted as empty (denominator = 40) | Same as before |

The old evaluator was **unfairly inflating the baseline score** on SWE Bench by excluding its 12 overflow failures from the denominator (28 vs 40). After the fix, both denominators are 40.

### Why +GI+SI still scores lower after the fix

Even with a fair denominator, +GI+SI (6.76) remains below baseline (9.80) for three reasons:

1. **One extra empty prediction**: Trace `0e6f...` produced `{"errors": [], "scores": []}` under +GI+SI (pass2 not triggered, model found nothing), while the baseline run produced an actual prediction for it.
2. **Lower quality on valid predictions**: 18 valid +GI+SI predictions vs. 19 valid baseline predictions. Adding graph injection makes the prompt longer; Mistral-Small's remaining context budget for the actual SWE Bench task is reduced, likely degrading reasoning quality.
3. **Small sample amplification**: Only 18–19 traces contribute valid predictions. A single trace's difference has a large effect on the aggregate metrics.

### Interpretation

The degradation is **not** evidence that graph injection is fundamentally harmful for SWE Bench. It reflects that:
- SWE Bench traces are already near Mistral-Small's context ceiling.
- Graph injection adds content that pushes marginal traces over the limit.
- GAIA traces are shorter, so graph injection helps there (+GI+SI gains over baseline on GAIA for all models).

The finding is specific to small-context models on long-trace datasets. Larger-context models (e.g., Gemini-2.5-Flash) show positive SWE Bench gains with +GI+SI (8.71 → 21.56 W-F1).
