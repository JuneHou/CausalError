# R3 — Temperature Robustness

Companion to `rebuttal/REBUTTAL_PLAN.md` §R3. Defends the headline +EDGE deltas
against the "no significance / could be noise" reviewer concern by rerunning the
**main-results setup** (full corpus, $\tau{=}0.35$ graph) at `temperature=0.7`
with **3 i.i.d. samples** per cell across all 5 open-weight backbones.

## Scope (60 inference runs)

| Backbones | Mistral-Small-3.1-24B, GPT-oss-120B, GPT-oss-20B, Gemma-3-27B-IT, QwenLong-L1-32B |
|---|---|
| Benchmarks | TRAIL combined (GAIA 117 + SWE 31) + MAST (393) — full corpus |
| Variants | baseline + +EDGE at $\tau{=}0.35$ |
| Decoding | `temperature=0.7`, 3 independent invocations per cell |
| Anchor | existing temp=0 Table-1 dirs (no rerun) |

## Layout

```
r3_temperature/
  config.py        scope, backbones, graph file pointers
  run_r3.py        command builder / dispatcher
  aggregate.py     scoring + LaTeX table
  sbatch/
    r3_template.sbatch  parametric sbatch (consumes JOB_NAME + CMDS env)
  data/predictions/<bench>/<backbone>/<variant>/temp0.7_sample{1,2,3}/...
  results/
    r3_per_cell_metrics.csv
    r3_temperature_table.tex
```

## Code patches (already applied)

Six eval scripts now accept `--temperature` (default `0.0`, preserves Table 1):

| Repo | Script |
|---|---|
| trail-benchmark | `benchmarking/eval/run_eval_vllm.py` |
| trail-benchmark | `benchmarking/eval/run_eval_graph_inject_vllm.py` |
| trail-benchmark | `benchmarking/eval/run_eval.py` (litellm baseline) |
| trail-benchmark | `benchmarking/eval/run_eval_graph_inject_api_arc.py` |
| MAST | `eval/run_eval_yesno_vllm.py` |
| MAST | `eval/run_eval_yesno_api.py` |
| MAST | `eval/run_eval_graph_inject.py` |
| MAST | `eval/full_run_eval_graph_inject_api_arc.py` |

Originals backed up with `old_` prefix in each `eval/` dir.

## Dispatch

### 1. Inspect the command list

```bash
cd rebuttal/r3_temperature
python run_r3.py                                  # print all 78 commands
python run_r3.py --backbone mistral-24b           # one backbone
python run_r3.py --backbone gpt-oss-120b --benchmark mast
```

### 2. Local sbatch (4 vllm backbones × 4 cells = 16 jobs)

The dispatcher prints sbatch invocations wrapping the cell's 3-sample loop.
You can edit `sbatch/r3_template.sbatch` for partition / time / GPU config
specific to your cluster; current defaults match `rebuttal/holdout/sbatch/`.

After syncing the patched repo over to the ARC paths (mirror
`rebuttal/holdout/sync.sh`), pipe the dispatcher into `sbatch`:

```bash
python run_r3.py --execute                        # submits all vllm cells + backgrounds ARC
# or copy the printed lines and sbatch them by hand
```

### 3. ARC API (gpt-oss-120B × 18 invocations)

```bash
set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
export ARC_LLM_API_KEY="$API_KEY"
# For TRAIL baseline via litellm, also set:
export OPENAI_API_BASE="https://llm-api.arc.vt.edu/api/v1/"
export OPENAI_API_KEY="$ARC_LLM_API_KEY"
python run_r3.py --backbone gpt-oss-120b --execute
```

### 4. After all runs complete

```bash
python aggregate.py
# writes results/{r3_per_cell_metrics.csv, r3_temperature_table.tex}
```

## Verification

1. **Stochasticity sanity** — diff `temp0.7_sample1/.../<id>.json` vs `temp0.7_sample2/.../<id>.json` for any one cell; outputs must differ.
2. **Deterministic-preservation sanity** — rerun one main-paper cell with `--temperature 0.0` via the patched script and compare to its Table-1 number; should match within rounding.
3. **Variance claim** — on each cell, check whether `|Δ_mean| > 2·max(σ_baseline, σ_edge)`. Cells that satisfy this carry the rebuttal directly; borderline cells should be reported honestly.

## Estimated wall-clock

- Local sbatch (4 backbones × 4 cells, each loops 3 samples internally): **6–24 h** depending on queue.
- ARC API (12 cells, ~200–400 calls each, ARC self-throttles at 30 rpm): **3–5 h**.
- Concurrent: **~half a day** if 4 nodes are free; **~1 day** if queue is busy.
