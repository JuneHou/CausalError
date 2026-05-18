# Workstream B: Case Study Candidate Selection

This folder contains code/results to shortlist TRAIL case-study examples using
existing outputs only (no new experiments).

## Structure

- `code/select_case_candidates.py`:
  compares baseline vs `+GI causal-only` vs `+GI corr0.35` per instance.
- `results/instance_comparison_summary.csv`:
  all comparable instances with per-instance score proxies and deltas.
- `results/model_priority_by_metric.csv`:
  per-model support summary (mean delta + win rate for W-F1, Loc, Joint).
- `results/top_working_candidates.csv`:
  top candidates where corr-union outperforms both baseline and causal-only.
- `results/top_not_working_candidates.csv`:
  top candidates where corr-union regresses relative to causal-only.
- Metric-specific candidate lists:
  - `results/top_working_candidates_wf1.csv`
  - `results/top_working_candidates_loc.csv`
  - `results/top_working_candidates_joint.csv`
  - `results/top_not_working_candidates_wf1.csv`
  - `results/top_not_working_candidates_loc.csv`
  - `results/top_not_working_candidates_joint.csv`
- `results/run_manifest.json`:
  run metadata and counts.

## Default assumptions used

- Model scope: open-source panel only (can override with `--models`).
- Splits: `GAIA_dedup` and `SWE_Bench_dedup`.
- Ranking objective:
  - balanced proxy (`wf1 + location + joint`) and
  - metric-specific ranking for each of `wf1`, `loc`, `joint`.

## Re-run

```bash
python3 analysis/workstream_b_case_study/code/select_case_candidates.py
```

Optional model override:

```bash
python3 analysis/workstream_b_case_study/code/select_case_candidates.py \
  --models openai-gpt-oss-120b openai-gpt-oss-20b google-gemma-3-27b-it
```

## Important note

Some output files are malformed/empty JSON in existing directories; the script
skips those and reports skip counts in `run_manifest.json`.
