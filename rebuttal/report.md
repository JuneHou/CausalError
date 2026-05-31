# EDGE — EMNLP 2026 Rebuttal Reports

Companion to `REBUTTAL_PLAN.md`. This file holds the **actual results and
rebuttal-ready text** for each reviewer concern as the experiments come in.
The plan says *what* to do; this file says *what we found and how to phrase it*.

**Status legend:** ✓ done · ◐ in progress · ✗ not started · N/A not applicable

---

## R1 — Graph trained and evaluated on the same corpus (leakage) ✓

**Status:** Done 2026-05-31. Headline = **4/4 cells positive, avg +6.3 W-F1**.

### Setup

- Single stratified 80/20 split per benchmark (not k-fold).
- TRAIL = combined GAIA + SWE (148 traces). 11 of 19 rare categories pinned to training.
- MAST = split by unique trace_id (142 unique IDs, 393 records). 6 of 13 rare categories pinned to training.
- $\mathcal{G}_\tau$ rebuilt from 80% training-side onsets only via `2_suppes_screen.py` at $\tau{=}0.35$; empty `effect_edges` so $\mathcal{G}_V$ contributes nothing (strictly weaker prior than the main paper's graph).
- Backbones: Mistral-Small-3.1-24B (local vLLM), GPT-oss-120B (Virginia Tech ARC API).
- Held-out sizes: TRAIL 25 traces; MAST 26 unique trace_ids / 74 records.
- Scorer paths: `dump_location_hits.py` (TRAIL, holdout `gt_trail/`), `dump_yesno_hits.py` (MAST, holdout `test.jsonl`).

### Results

| Benchmark | Backbone | n | Baseline | +EDGE | $\Delta$ |
|---|---|---:|---:|---:|---:|
| TRAIL | mistral-24B | 25 | 0.2474 | 0.3314 | **+0.0840** |
| TRAIL | gpt-oss-120B | 25 | 0.3108 | 0.3348 | **+0.0240** |
| MAST | mistral-24B | 74 | 0.3453 | 0.3644 | **+0.0191** |
| MAST | gpt-oss-120B | 74 | 0.2209 | 0.3450 | **+0.1241** |

**4/4 cells positive, mean $\Delta$ = +0.063 W-F1.**

### Where the numbers live on disk

```
rebuttal/holdout/data/predictions/
  trail/{mistral-24b,gpt-oss-120b}/{baseline,edge}-location_hits.json
  mast/mistral-24b/mistralai-...-yesno-baseline-yesno_hits.json
  mast/mistral-24b/edge-yesno_hits.json
  mast/gpt-oss-120b/{baseline,edge}-yesno_hits.json
```

### Pipeline issues encountered (for reproducibility)

1. `mast/gpt-oss-120b/baseline/` was originally named with **global** record indices (`0000, 0003, 0006, …, 0390`) carried over from the upstream `MAST/outputs/gpt-oss-120b-yesno-baseline/`. `dump_yesno_hits.py` keys the holdout `test.jsonl` by **local** line index `0000..0073`, so file `0003.json` was being scored against the wrong GT row. Fixed in-place by sorting baseline files and renaming to `0000..0073.json` (sort order matches the holdout `test.jsonl` line order by `trace_id`).
2. `dump_location_hits.py` does not auto-resolve nested subdirs (unlike `dump_yesno_hits.py`'s `resolve_pred_dir`). TRAIL edge predictions live in `edge/outputs_<model>-test-graph_inject_causal_corr0.35_span_index/*.json`; invoke with that nested path and redirect output via `--out` to `…/edge-location_hits.json` for the conventional layout.
3. Pre-fix, `aggregate.py` produced a stale `baseline-metrics.json` (n=18) and inconsistent CSV values for the same cell. The new dump-script flow supersedes `aggregate.py` for rebuttal numbers; `per_cell_metrics.csv` and `rebuttal_holdout_table.tex` are stale.

### Rebuttal-ready text (compact, fits any reply textbox)

> **R1 (leakage).** Stratified 80/20 held-out, one split per benchmark. $\mathcal{G}_\tau$ rebuilt from the 80% training side only (corr-union $\tau{=}0.35$, no $\mathcal{G}_V$ edges) — a strictly weaker prior than the main-paper graph. Held out: TRAIL 25/148 traces (GAIA+SWE combined); MAST 26/142 unique trace_ids (74 records). +EDGE W-F1 over baseline: TRAIL mistral +8.4, gpt-oss-120B +2.4; MAST mistral +1.9, gpt-oss-120B +12.4. **4/4 cells positive, avg +6.3 F1.** Population-level leakage cannot explain the gains.

### Optional table form (when space allows)

> | | n | Base | +EDGE | $\Delta$ |
> |---|---:|---:|---:|---:|
> | TRAIL mistral-24B | 25 | 24.7 | 33.1 | **+8.4** |
> | TRAIL gpt-oss-120B | 25 | 31.1 | 33.5 | **+2.4** |
> | MAST mistral-24B | 74 | 34.5 | 36.4 | **+1.9** |
> | MAST gpt-oss-120B | 74 | 22.1 | 34.5 | **+12.4** |

