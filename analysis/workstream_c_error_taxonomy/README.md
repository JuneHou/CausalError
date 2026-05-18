# Workstream C: Causal-vs-Corr Contrast Analysis (existing outputs only)

This folder implements the Workstream-C pipeline as a contrast analysis that is
explicitly different from Workstream B:

- Workstream B: baseline-centered case study (wins/fails vs baseline).
- Workstream C: causal-vs-corr mechanism contrast (when corr adds over causal,
  when causal is already sufficient, and when corr hurts).

This folder provides:

- freeze two-layer schema (benchmark taxonomy + mechanism/pattern labels),
- pilot GPT labeling (10 examples) and refine,
- run full GPT labeling on a compact contrast set (default: 30 examples),
- manual QC on 8-12 examples with agreement report,
- adjudicate and lock final labels,
- aggregate to paper-ready summaries and protocol sentences.

## Structure

- `code/build_trail_audit_sheet.py`
  - consumes Workstream-B `instance_comparison_summary.csv`,
  - stratifies into simplified causal-vs-corr contrast strata
    (`causal_strong`, `corr_strong`, `all_tie`),
  - writes audit sheet + taxonomy overlays + frozen schema files.
- `code/annotation_schema.py`
  - frozen label space for both layers.
- `code/gpt5_label_workstream_c.py`
  - GPT-5 labeling for pilot/main modes.
- `code/prepare_qc_subset.py`
  - creates QC subset and blank manual relabel file.
- `code/evaluate_qc_agreement.py`
  - computes field-wise agreement and kappa.
- `code/adjudicate_labels.py`
  - merges GPT + human override into final locked labels.
- `code/summarize_trail_audit.py`
  - summarizes final labels (mechanism + taxonomy distribution).
- `code/build_protocol_summary.py`
  - emits one-sentence protocol and one-sentence QC text for paper.
- `results/`
  - generated artifacts.

## Default workflow

1) Build TRAIL audit sheet:

```bash
python3 analysis/workstream_c_error_taxonomy/code/build_trail_audit_sheet.py
```

Default compact contrast size is 30 (10 per group). To change size:

```bash
python3 analysis/workstream_c_error_taxonomy/code/build_trail_audit_sheet.py \
  --sampling_mode contrast --contrast_n_per_group 10
```

2) Freeze schema + build audit sheet:

```bash
python3 analysis/workstream_c_error_taxonomy/code/build_trail_audit_sheet.py
```

3) Pilot (10 rows) with GPT-5.2:

```bash
export OPENAI_API_KEY=...
python3 analysis/workstream_c_error_taxonomy/code/gpt5_label_workstream_c.py --mode pilot --pilot_n 10
```

4) Review pilot CSV and refine schema/prompts if needed.

5) Main pass (all selected rows, outputs to `trail_audit_sheet_gpt52_labeled.csv`):

```bash
python3 analysis/workstream_c_error_taxonomy/code/gpt5_label_workstream_c.py --mode main
```

Important:
- `mechanism_bucket` is now deterministic and assigned by rule from performance deltas
  (`delta_corr_vs_causal`, and baseline tie-handling).
- GPT labels only the supportive fields (`pattern_tags`, characteristics,
  corr role, severity, confidence, evidence_note) unless
  `--label_mechanism_bucket` is explicitly enabled.
- Prompt is fixed to compact mode (no runtime switch).

6) QC subset for manual relabeling (8-12 rows):

```bash
python3 analysis/workstream_c_error_taxonomy/code/prepare_qc_subset.py --n_qc 10
```

Fill `results/trail_qc_subset_manual_blank.csv`, then save as
`results/trail_qc_subset_manual_filled.csv`.

7) Evaluate agreement:

```bash
python3 analysis/workstream_c_error_taxonomy/code/evaluate_qc_agreement.py
```

8) Adjudicate and lock:

```bash
python3 analysis/workstream_c_error_taxonomy/code/adjudicate_labels.py
```

9) Summarize final locked labels:

```bash
python3 analysis/workstream_c_error_taxonomy/code/summarize_trail_audit.py \
  --audit_csv analysis/workstream_c_error_taxonomy/results/trail_audit_sheet_final_locked.csv
python3 analysis/workstream_c_error_taxonomy/code/build_protocol_summary.py
```

## Two-layer labels to fill/review

Layer-1 benchmark taxonomy is auto-derived from leaf categories in each variant.

Layer-2 mechanism fields:

- `mechanism_bucket`
  - one of: `causal-backed-gain` / `corr-added-gain` / `causal-preserving-neutral` / `corr-induced-harm` / `shared-failure`
- `pattern_tags` (pipe-separated, e.g. `missing-context-recovery|dependency-chain`)
- `pattern_characteristics` (short phrase; what characterizes this pattern)
- `corr_edge_role` (one of `beneficial` / `neutral` / `harmful` / `unknown`)
- `impact_severity` (`low` / `medium` / `high`)
- `confidence` (`high` / `medium`)
- `evidence_note`
- `annotator`

## Output files

- `results/trail_audit_sheet.csv`: annotation sheet for TRAIL.
- `results/trail_audit_candidates.json`: same rows in JSON for easier inspection.
- `results/trail_sampling_manifest.json`: selection counts and diversity checks.
- `results/annotation_schema_freeze.{json,md}`: frozen label schema.
- `results/trail_audit_sheet_pilot_labeled.csv`: GPT pilot labels.
- `results/trail_audit_sheet_gpt52_labeled.csv`: GPT-5.2 main labels.
- `results/trail_qc_subset_reference_gpt5.csv`: GPT labels for QC subset.
- `results/trail_qc_subset_manual_blank.csv`: blank manual QC file.
- `results/trail_qc_agreement_summary.csv`: field-wise agreement/kappa.
- `results/trail_qc_disagreements.csv`: disagreements for adjudication.
- `results/trail_audit_sheet_final_locked.csv`: adjudicated final labels.
- `results/trail_mechanism_bucket_summary.csv`: mechanism-bucket counts and percentages.
- `results/trail_split_summary.csv`: split-level counts.
- `results/trail_split_mechanism_bucket_summary.csv`: within-split mechanism distribution.
- `results/trail_pattern_tag_summary.csv`: pattern/characteristic frequency summary.
- `results/trail_taxonomy_leaf_summary.csv`: layer-1 leaf category frequencies.
- `results/trail_taxonomy_l1_summary.csv`: top-level taxonomy distribution.
- `results/trail_taxonomy_l2_summary.csv`: mid-level taxonomy distribution.
- `results/trail_representative_cases.csv`: one representative row per bucket.
- `results/trail_audit_summary.json`: final compact summary + net-effect line.
- `results/trail_protocol_summary.{json,md}`: paper-ready protocol/QC sentences.

## Notes

- Uses existing completed outputs only; no new experiments.
- Selection defaults match the plan target (22 working, 10 not-working, 6 neutral).
- If `meets_min_models` is false in the sampling manifest, rerun with a different seed
  or adjust bucket counts.

## Intermediate run snapshot

Recorded from terminal output (`build_trail_audit_sheet.py`), for traceability of the
current mid-step state:

```json
{
  "inputs": {
    "input_summary_csv": "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_b_case_study/results/instance_comparison_summary.csv",
    "seed": 42
  },
  "targets": {
    "n_working": 22,
    "n_not_working": 10,
    "n_neutral": 6,
    "n_total": 38
  },
  "selected_counts": {
    "n_selected_raw": 38,
    "n_selected_enriched": 38,
    "by_bucket": {
      "working": 22,
      "not_working": 10,
      "neutral": 6
    },
    "by_split": {
      "GAIA_dedup": 26,
      "SWE_Bench_dedup": 12
    },
    "model_diversity": 1,
    "meets_min_models": false
  },
  "read_stats": {},
  "outputs": {
    "trail_audit_sheet_csv": "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_sheet.csv",
    "trail_audit_candidates_json": "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/trail_audit_candidates.json",
    "schema_json": "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/annotation_schema_freeze.json",
    "schema_md": "/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results/annotation_schema_freeze.md"
  },
  "notes": [
    "Uses existing outputs only; no new experiments.",
    "Audit labels are intentionally left blank for manual annotation.",
    "Use mechanism_bucket as the primary label: causal-backed-gain / corr-added-gain / causal-preserving-neutral / corr-induced-harm / shared-failure.",
    "If model diversity is below threshold, rerun with different seed or bucket sizes."
  ]
}
```
