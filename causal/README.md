# Causal pipeline

- **`intervention/`** — Patch application, single-intervention re-runs (do(A_i≈0)), and effect evaluation (Δ(A→B)). Run from `benchmarking/`:
  - `python intervene.py` or `python causal/intervention/intervene.py`
  - `python effect_eval.py` or `python causal/intervention/effect_eval.py`
  - `python rerun_intervention.py` or `python causal/intervention/rerun_intervention.py`
  - See `intervention/INTERVENTION_PIPELINE.md` for full usage.

- **`graph/`** — Causal graph construction: preprocess (trail_1–3) and CAPRI pipeline (order pairs, Suppes, CAPRI prune, bootstrap, shuffle, hierarchy). Run from `benchmarking/`:
  - `bash run_causal_gaia.sh` or `bash causal/graph/run_causal_gaia.sh` — build onsets from GAIA traces.
  - `python run_causal_from_trail_onsets.py --onsets_path data/trail_derived/onsets_gaia.jsonl` or `python causal/graph/run_causal_from_trail_onsets.py ...` — run CAPRI on onsets.
