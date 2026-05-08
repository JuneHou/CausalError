# Per-trace count-bias CDF

This document describes the figure `_plot_count_bias_cdf_v2.png` (and its
unstyled sibling `_plot_count_bias_cdf.png`), how it is generated, and how
it should be read. The Likert / diverging-stacked-bar variant is **not**
the figure of interest and is excluded from this document.

---

## What the figure shows

For every `(model, dataset split, method)` triple in TRAIL, we compute one
**count-bias score** per trace and plot the empirical CDF of those scores,
one curve per model. The figure is faceted as

```
                        Baseline       +CG       +GI+SI
        TRAIL-GAIA       (panel)      (panel)    (panel)
        TRAIL-SWE        (panel)      (panel)    (panel)
```

so that horizontal sweeps reveal the effect of adding the causal graph,
and vertical sweeps reveal the difference between the two TRAIL splits.

### The score

For each trace,

$$
s \;=\; \log\!\left(\frac{\mathrm{pred} + 1}{\mathrm{gold} + 1}\right),
$$

where `pred` is the number of distinct error categories the model flags
in the trace and `gold` is the number of distinct error categories in the
ground-truth annotation.

* `s < 0` → trace is **under-predicted** (model named fewer categories than gold has).
* `s > 0` → trace is **over-predicted** (model named more).
* `s = 0` → exact match in count.

The `+1` smoothing makes the score symmetric around zero and finite when
either side is zero (e.g. trace with `gold = 0`, `pred = 0`).

### The y-axis

For a model's curve at score `x`,

$$
\mathrm{CDF}(x) \;=\; \frac{\#\{\text{traces with } s \leq x\}}{\text{total traces}}.
$$

So the curve always starts at `(−∞, 0)` and ends at `(+∞, 1)`; each step
upward is one trace.

---

## How to read a single curve

Three landmarks tell you almost everything.

1. **Where the curve crosses the bold vertical at `x = 0`.**
   That y-value is the fraction of traces the model under-predicts on.
   Subtract from 1 to get the fraction it over-predicts (plus exact matches).

2. **Where the curve crosses the dotted horizontal at `y = 0.5`.**
   That x-value is the **median per-trace bias**. The v2 figure additionally
   marks this point with a filled circle and prints the numeric median in
   a small column at the right edge of each panel.
   * Median left of zero → typical trace under-predicts.
   * Median right of zero → typical trace over-predicts.
   * Median at zero → typical trace is well-calibrated in count.

3. **Slope around `y = 0.5`.**
   * Steep curve = bias is consistent across traces (model is reliably off
     by about the same amount).
   * Flat curve = bias varies trace-by-trace (some traces strongly over,
     others strongly under; average looks fine but individual traces don't).

A vertical jump at `x = 0` means many traces have `pred == gold` exactly.
The bigger the jump, the more "neutral" traces.

---

## How to compare curves

### Within the same panel (cross-model)

* **Curve shifted left** of another → that model is more conservative
  (under-predicts more often / more deeply).
* **Curve shifted right** → more exploratory.
* **Curves crossing each other** → no uniform ordering; one model is more
  conservative on easy traces but more exploratory on hard ones (or
  vice-versa). A crossing at `y ≈ 0.5` is particularly informative: it
  implies similar median bias but different tails.
* **Vertical gap at any fixed `x`** between two curves = the
  Kolmogorov–Smirnov-style difference in `Pr[s ≤ x]` between the two models.

### Across rows (GAIA vs SWE)

For the same model and method, comparing GAIA (top row) to SWE (bottom row)
reveals dataset effects more than model effects: SWE traces have fewer
gold errors, so a model that emits the same average number of categories
on both splits will look exploratory on SWE and conservative on GAIA.
Read row-to-row shifts as **dataset properties first, model behavior second**.

### Across columns (Baseline vs +CG vs +GI+SI)

This is the headline read for "does the causal graph change conservativeness?"
Track a single colored curve from left to right within the same row:

* Curve drifts **right** → the graph encourages more flagging (less conservative).
* Curve drifts **left** → the graph encourages restraint (more conservative).
* Curve does not move → graph guidance does not change *count* behavior,
  only *which categories* get flagged (this would still affect macro-P/R
  but not bias).

### What the data actually shows

Most curves cross `x = 0` near `y = 0.4`–`0.6`, meaning roughly half of
traces under-predict and half over-predict for any given model. This may
look surprising next to the macro-PR plot (where most points sit below
the `P = R` diagonal, i.e. systematically conservative). The two
observations are not contradictory — they describe different cuts:

* **Macro-PR** aggregates *category-level* binary decisions across all
  traces. Conservativeness there is driven by many false-negative
  categories (gold contains a category, model omitted it).
* **Count-bias CDF** aggregates *per-trace* count differences. Positive
  and negative trace-level errors can cancel out at the trace level even
  while category-level recall is uniformly low.

The tails on the CDF are also short: most curves do nearly all their
rising between `s = −1` and `s = +1`. So when a model misses, it typically
misses by a factor of ~½ to ~2 (e.g. gold = 4, pred = 2 or 8), not by a
factor of 10. There is no "predicted 12 when gold was 4" long tail in TRAIL.

---

## How the figure is generated

### Inputs

* **Predictions** — JSON files under
  `benchmarking/outputs/zero_shot2/`,
  `benchmarking/outputs/zero_shot/`, and
  `benchmarking/outputs/zero_shot/compressed/`. The collector follows the
  source-of-truth conventions documented at the top of
  `paper/main_results_table.tex`:
  * Gemini-2.5-Flash → `zero_shot/`, original split (full GAIA / SWE Bench).
  * Gemini-2.5-Pro, Mistral, GPT-oss-{20B,120B}, Gemma → `zero_shot2/`, dedup splits.
  * QwenLong-L1-32B GAIA → `zero_shot/compressed/GAIA_dedup-*` (full traces).
  * QwenLong-L1-32B SWE → `zero_shot2/`.
  * The Gemini-Flash GAIA `+GI+SI` cell falls back to the
    `graph_inject_causal_corr0.2_span_index` run (the `causal_only` variant
    was not produced for that cell). The fallback was verified by matching
    the resulting weighted-F1 (0.4075) against the value reported in
    `main_results_table.tex` (40.75).

* **Ground truth** — `benchmarking/processed_annotations_gaia/` and
  `benchmarking/processed_annotations_swe_bench/`. Predictions are matched
  to GT by file hash; traces with no matching GT are skipped. Many model
  outputs are wrapped in markdown code fences (```` ```json … ``` ````);
  the loader strips fences and falls back to balanced-brace extraction so
  these are not silently dropped.

### Method classification

Directory suffix → method, with priority `causal_only` > `corr0.2`:

| suffix on output dir name | method |
|---|---|
| *(no suffix)* | Baseline |
| `…-graph_causal_only` | +CG |
| `…-graph_inject_causal_only_span_index` | +GI+SI |
| `…-graph_inject_causal_corr0.2_span_index` | +GI+SI (fallback for Gemini-Flash GAIA only) |
| `…-graph_causal_only_span_index`, `…-graph_t0.2`, `…-graph_suppes`, `…-who_and_when_*`, `…-calibrated`, `…-graph_causal_only_train` | excluded |

Only the seven models that appear in `paper/main_results_table.tex` are
kept (Gemini-Flash, Gemini-Pro, Mistral-Small-24B, GPT-oss-120B,
GPT-oss-20B, Gemma-3-27B, QwenLong-L1-32B). Other outputs in the
directories (e.g. `o4-mini`) are filtered out.

### Pipeline

1. `_plot_count_bias_likert.py` (in `baselines/outputs/`) exposes
   `collect()` and `per_trace_scores()`. These walk the priority-ordered
   output dirs, classify each directory, deduplicate by
   `(model, split, method)` (preferring earlier search dirs and
   `pref = 0` methods), match predictions to GT by filename hash, parse
   both JSONs leniently, count distinct error categories on both sides,
   and emit one row per trace.
2. `_plot_count_bias_cdf_v2.py` (in `figures/`) imports `collect()`,
   sorts each model's per-trace scores, builds the empirical CDF, and
   renders the 2 × 3 grid with the Wong colorblind-safe palette and
   median markers.

### How to (re)generate

From the repo root, with the project conda env active:

```bash
conda activate "/data/wang/junh/envs/causal"
python figures/_plot_count_bias_cdf_v2.py
```

Outputs:

| file | description |
|---|---|
| `figures/_plot_count_bias_cdf_v2.png` | Wong palette, serif typography, median markers (current best version). |
| `figures/_plot_count_bias_cdf.png` | Original tab10 palette, no median markers (kept for reference). |

### Styling choices

The v2 figure follows Nature/Science conventions:

* **Wong colorblind-safe palette** (Bang Wong, *Nature Methods* 2011,
  doi:10.1038/nmeth.1618). Eight categorical colors with controlled
  luminance, distinguishable in deuteranopia and protanopia. Family-aware
  assignment: warm tones for Gemini, cool tones for GPT-oss, distinct
  hues for the three open singletons.
* **Serif body font** (DejaVu Serif fallback), thinner axis lines (0.6 pt),
  no top/right spines, light gridlines (alpha 0.25). Curves drawn at
  1.4 pt with rounded caps so individual trace steps remain visible
  without dominating.
* **Median markers** on each curve (filled circle at `(median, 0.5)`,
  white halo for separation), with the numeric medians printed in a
  monospace column at the right edge of each panel, sorted from most
  conservative to most exploratory. This makes the median read-off
  numerical rather than visual.

### Caveats

* The percentile-based bin thresholds used by the Likert variant
  (`0.29 / 0.69` for over, `0.41 / 0.69` for under) compress against the
  short tails in this dataset and are **not** used in this CDF figure.
  The CDF is threshold-free.
* Trace counts per cell follow the prediction availability documented in
  Table 4 of `paper/main_results_table.tex`. Some `+CG` cells are absent
  by design (Gemini-Pro and Mistral SWE) and appear as missing curves
  rather than as flat lines.
