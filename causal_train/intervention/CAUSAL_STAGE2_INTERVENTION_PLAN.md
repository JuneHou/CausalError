# Stage II: Causal Intervention Plan (TRAIL)

This document designs the **coding plan** for Stage II: validating Stage I causal edges via **intervention** (not just correlation), using existing TRAIL data (failure logs, annotations, onsets, traces) to replicate scenarios and run repair/inject/placebo arms.

---

## 1. How Existing Data Supports Intervention

### 1.1 Data you already have

| Asset | Location / source | Use in Stage II |
|-------|-------------------|------------------|
| **Failure log (annotations)** | `processed_annotations_gaia/*.json`, `processed_annotations_swe_bench/*.json` | Per-trace list of `{ category, location (span_id), evidence, description, impact }`. Defines which failures occurred and **where**; evidence/description feed **repair(A)** and **inject(A)** prompt templates. |
| **Onsets** | `data/trail_derived/onsets_gaia.jsonl` | `trace_id`, `present[A]`, `onset[A]` (earliest rank per category). Used to **select traces** per edge (e.g. traces with both A and B, or A only), and to compute baseline P(A), P(B) and **ΔT_A** (onset shift) after intervention. |
| **Span order** | `data/trail_span_order/gaia.jsonl` | `trace_id` → `span_rank` (span_id → rank). Defines timeline for **ΔT_A** (did repair move the onset of A?). |
| **Stage I graph** | `data/trail_causal_outputs/capri_graph.json`, `hierarchy_levels.json` | **Edges A→B** to validate; **roots** (level_0) for intervention operators and controllability. |
| **Traces** | `data/GAIA/*.json`, `data/SWE Bench/*.json` | Full span tree. Used to **extract run config** (system prompt, input prompt, model) and to locate **intervention hooks** (LLM/TOOL/CHAIN span at onset of A). |
| **Filtered index** | `data/trail_filtered/gaia.jsonl` | `trace_id`, `trace_path`, `annotation_path`, `split`, `n_errors`, `error_locations`. Links traces ↔ annotations. |

### 1.2 Replicating the scenario to run intervention

Two viable strategies; implement **Path A** first for true causal validity, then **Path B** as a cheaper proxy.

- **Path A — Full re-run (recommended for deliverables)**  
  For each trace and each arm (control Z0, repair Z-, inject Z+):
  1. **Extract** from the trace: system prompt, input/task prompt, model (see `REPLICATE_ERRORS_README.md`).
  2. **Apply** the intervention: e.g. append or prepend to the system (or task) prompt a block from the **intervention operator library** (repair(A), inject(A), or placebo).
  3. **Re-run** the same agent on the same task (same task_id / question) with the modified prompt.
  4. **Label** the new trace: run the existing TRAIL error detector (`run_eval.py`) on the new trace → get `errors[]` → derive binary A, B (and optionally other categories) per trace.
  5. Aggregate over traces and seeds: P(A|Z-), P(A|Z+), P(B|Z-), P(B|Z+), placebo, then compute ΔP(A), ACE, spillover, CI.

  **What the trace already has:** For GAIA, the TRAIL trace JSON **already contains** the two prompts and the model (see [REPLICATE_ERRORS_README.md](REPLICATE_ERRORS_README.md)): (1) **System prompt** — in LLM spans as `span_attributes["llm.input_messages.0.message.content"]`; (2) **Input / task prompt** — in early logs as `"question"` and in LLM spans as `llm.input_messages.1.message.content` (e.g. “New task: … Here is the task: …”); (3) **Model** — `span_attributes["llm.model_name"]` (e.g. `"o3-mini"`). So you do **not** need the GAIA data source or GitHub repo to obtain prompts and model; they are in the trace.  
  **Gap:** The repo does not yet contain an **agent re-run harness** (OpenDeepResearch for GAIA, CodeAct for SWE Bench). You need the GAIA/SWE **execution** repo only to **run** the agent: take the extracted (system_prompt, input_prompt, model) and produce a new trace. The script `extract_run_config.py` (to be added) simply reads these from the trace and writes a run-config JSON (and optionally prompt .txt files) so any re-run harness can consume them.

- **Path B — Eval-only proxy (faster, weaker causality)**  
  Do **not** re-run the agent. For each trace where A occurs at span_id `s`:
  1. Run **error detection** (`run_eval.py`) twice on the **same** trace:
     - **Control:** default eval prompt.
     - **Intervention:** eval prompt augmented with: “Assume the agent was instructed at this step to avoid [failure type A] (e.g. [short repair(A) description]). Still report all other errors.”
  2. Compare reported errors: did “B” appear less often in the intervention condition? Use this as a **proxy** for ACE (clearly document as “eval-model proxy,” not full causal).

  Path B reuses `run_eval.py` and annotations only; no new agent runs. Use for quick iteration and as fallback if Path A is delayed.

---

## 2. Output 1 — Validated Causal Edge Set G_C with Weights

### 2.1 Per-edge experiment design

For each Stage I edge **A → B** (from `capri_graph.json`):

1. **Trace sampling**
   - **Cohort:** Traces where **A** appears (from onsets: `present[A]==1`). Optionally require also `present[B]==1` for power on B.
   - **Arms:** Z- (repair(A)), Z+ (inject(A)), Z0 (control or placebo). Each trace is run with one arm per “experiment” (or use between-trace randomization if re-running is expensive).
   - **Seeds:** Multiple seeds per (trace, arm) for variance and bootstrap CI (e.g. n_seeds=40, n_trials_per_arm=5 as in your spec).

2. **Metrics to compute**
   - **Manipulation check**
     - **ΔP(A)** = P(A=1|Z-) − P(A=1|Z0) (repair should decrease A) and/or P(A=1|Z+) − P(A=1|Z0) (inject should increase A).
     - **ΔT_A** (optional): If you track onset of A in reruns (via span_rank of first A in new trace), report mean onset shift (e.g. repair delays or removes A → positive ΔT_A or N/A).
   - **Interventional effect (ACE)**
     - **ACE_A→B** = P(B=1|Z+) − P(B=1|Z-) (or vs control: P(B=1|Z-) − P(B=1|Z0) for “does repairing A reduce B?”).
   - **Uncertainty:** Bootstrap over seeds (and optionally over traces) → **CI95** for ACE.
   - **Placebo:** **placebo_ACE** = effect when using placebo prompt (same length/style, no A-specific content) to check for non-specific effects.
   - **Spillover:** For other failure types C ∉ {A,B}, **ΔP(C)** per C (e.g. spillover_L1 = max |ΔP(C)| or a named spillover for a key category).

3. **Storage (CSV/JSON)**

   Per edge, store something like:

```json
{
  "A": "Tool Selection Errors",
  "B": "Goal Deviation",
  "n_seeds": 40,
  "n_trials_per_arm": 5,
  "deltaP_A_repair": -0.22,
  "deltaT_A_repair": 1.1,
  "ACE_A_to_B": -0.18,
  "CI95": [-0.27, -0.08],
  "placebo_ACE": -0.01,
  "spillover_L1": 0.12
}
```

   Keep a **validated edge list** G_C: retain edge A→B only if manipulation check is satisfied (e.g. |ΔP(A)| above threshold) and optionally |ACE| above a minimum.

### 2.2 Using existing data for aggregation

- **Baseline P(A), P(B):** From current annotations (or from current onsets): for the cohort of traces used in the experiment, P(A)=1 and P(B)=1 are just the fractions with `present[A]==1` and `present[B]==1`. After re-runs, P(A|Z) and P(B|Z) come from the new traces’ labels (from `run_eval.py` or from your re-run harness’s output).
- **Trace–edge index:** Build a small table: for each edge (A,B), list `trace_id`s where both A and B occur (or at least A). Use `data/trail_derived/onsets_gaia.jsonl` + `capri_graph.json` to build this once (see script in § 5).

---

## 3. Output 2 — Intervention Operator Library

For each root-candidate failure type **A** (and optionally for every category that appears as a cause in the graph), define:

| Operator | Goal | Use in Stage II |
|----------|------|------------------|
| **repair(A)** | Reduce P(A) | Z- arm: append/prepend to system (or task) prompt a short instruction that discourages failure type A. |
| **inject(A)** | Increase P(A) | Z+ arm: instruction that makes A more likely (e.g. “prefer quick answers without checking tools” for a tool-related A). |
| **placebo** | Same length/style, no A content | Control for non-specific prompt effects. |
| **wrong_target(A')** | Optional | Instruction targeting a different failure A' to check specificity. |

**Implementation:**

- **Storage:** YAML or JSON under e.g. `data/trail_causal_outputs/intervention_operators/`:
  - One file per category or one file with all operators (e.g. `operators.yaml`) with parameter slots: `{failure_type}`, optional `{evidence_snippet}`, optional `{span_id}`.
- **Filling slots:** Use annotation fields: for each (category, location), annotations provide `evidence` and `description`. A generic template for repair(A) could be: “Before the step at which the error occurred, ensure you do not [description]. In particular avoid [evidence].” Inject(A) can be the inverse (e.g. “It is acceptable to [behavior that leads to A].”).
- **Usage:** The re-run harness (Path A) or the eval prompt builder (Path B) loads the template, substitutes A (and optionally evidence/span_id), and injects the text into the system or task prompt.

Example structure:

```yaml
# operators.yaml (conceptually)
repair:
  Tool Selection Errors: |
    Additional instruction: When choosing a tool, explicitly check that the tool's capability matches the current subtask. Avoid using a tool whose description does not cover the operation you need.
inject:
  Tool Selection Errors: |
    Additional instruction: Prefer using any available tool quickly even if its description is only loosely related to the step.
placebo: |
  Additional instruction: Please keep your responses concise and well-structured.
```

---

## 4. Output 3 — Controllability Score per Failure Type A

Define:

- **Ctrl(A)** = |P(A=1|Z+) − P(A=1|Z-)| (how much you can move P(A) with your operators).
- **Cost:** repair_token_cost, inject_token_cost (and optionally extra tool calls, runtime).

**Data source:** From Stage II runs: for each A you already compute P(A|Z+) and P(A|Z-) (and token counts if you log them). Aggregate per failure type A (over all edges where A is the cause).

**Storage (e.g. JSON):**

```json
{
  "A": "Tool Selection Errors",
  "Ctrl": 0.35,
  "repair_token_cost": 180,
  "inject_token_cost": 50
}
```

One file per A or a single `controllability.json` with keys = failure types.

---

## 5. Output 4 — Graph-Guided Mitigation Policy

Derive from Stage II outputs:

- **Benefit(A)** = ∑_{B ∈ Ê \ {A}} max(0, −ACÊ_{A→B}).  
  (Only count B that appear in the detected set Ê; negative ACE means “repairing A reduces B.”)
- **Priority(A)** = Ctrl(A) · Benefit(A) / Cost(repair(A)).
- **Policy:** Given a detected set of failures Ê in a trace:
  - **Stage-1 prompt:** Include repair instructions for the top root(s) by Priority(A).
  - **Stage-2 prompt (optional):** For remaining failures in Ê not resolved, add repairs in order of Priority.

**Inputs:** Validated edge set G_C (with ACE and CI), controllability and cost per A, and the set Ê (from runtime detection or from annotations for evaluation). No new data beyond Stage II outputs.

---

## 6. Implementation Order (Coding Plan)

### Phase 1 — Data and indexing (no re-run yet)

1. **Trace–edge index**
   - **Script:** e.g. `benchmarking/causal_explore/stage2/1_build_trace_edge_index.py`
   - **Inputs:** `onsets_gaia.jsonl`, `capri_graph.json`, optional `trail_filtered/gaia.jsonl`
   - **Output:** e.g. `data/trail_causal_outputs/stage2/trace_edge_index.json`: for each edge (A,B), list of `trace_id` with present A (and optionally present B), plus baseline counts.

2. **Extract run config**
   - **Script:** `benchmarking/extract_run_config.py` (create; referenced in REPLICATE_ERRORS_README)
   - **Input:** path to one trace JSON (GAIA or SWE Bench).
   - **Output:** JSON with `system_prompt`, `input_prompt`, `model`, and optionally `*_system_prompt.txt`, `*_input_prompt.txt`. Implement by scanning spans for `openinference.span.kind` and `llm.input_messages` (or GAIA’s logs with `question` / model_id) as in the README.

3. **Intervention operator library**
   - **Script:** `benchmarking/causal_explore/stage2/2_build_operator_library.py`
   - **Input:** TRAIL taxonomy (or list of categories from onsets/capri), optional annotations for evidence/description samples.
   - **Output:** `data/trail_causal_outputs/intervention_operators/operators.yaml` (or JSON) with repair(A), inject(A), placebo, and parameter slots.

### Phase 2 — Intervention runs and labeling

4. **Apply operator to run config**
   - **Script:** `benchmarking/causal_explore/stage2/3_apply_operator.py`
   - **Input:** extracted run config JSON, operator name (repair_A, inject_A, placebo), category A.
   - **Output:** modified run config (new system or task prompt string) ready for the re-run harness.

5. **Re-run harness (Path A)**
   - **Script or integration:** Either call external agent runner (OpenDeepResearch / CodeAct) or add a minimal `run_agent_single.py` that takes (system_prompt, input_prompt, model, seed) and writes a trace JSON. This is the main “gap” and may depend on your infra.
   - **Orchestrator:** For each edge (A,B), for each trace in the edge’s cohort, for each arm (Z0, Z-, Z+), for each seed: extract config → apply operator → run agent → save trace. Optionally run `run_eval.py` on each new trace and store (trace_id, arm, seed, errors[]).

6. **Eval-only proxy (Path B)**
   - **Script:** `benchmarking/causal_explore/stage2/4_eval_proxy_intervention.py`
   - **Input:** trace path, operator text for A (repair or placebo), eval model, existing `run_eval.py` entrypoint.
   - **Output:** Two eval outputs (control vs intervention prompt); parse errors[] and compute proxy P(A), P(B) for that trace.

### Phase 3 — Aggregation and deliverables

7. **Aggregate per edge**
   - **Script:** `benchmarking/causal_explore/stage2/5_aggregate_edge_metrics.py`
   - **Input:** Run results (per trace, arm, seed: labels A, B, others) and optionally onset/span_rank for ΔT_A.
   - **Output:** Per-edge JSON/CSV: deltaP_A_repair, deltaT_A_repair, ACE_A_to_B, CI95, placebo_ACE, spillover vector. Write to `data/trail_causal_outputs/stage2/validated_edges.json` (or CSV).

8. **Controllability**
   - **Script:** `benchmarking/causal_explore/stage2/6_controllability_per_failure.py`
   - **Input:** Aggregated runs (P(A|Z+), P(A|Z-), token counts per A).
   - **Output:** `data/trail_causal_outputs/stage2/controllability.json`.

9. **Mitigation policy**
   - **Script:** `benchmarking/causal_explore/stage2/7_mitigation_policy.py`
   - **Input:** validated_edges.json, controllability.json, and (for evaluation) a list of detected failures Ê or full onset rows.
   - **Output:** Priority ordering, recommended stage-1 and optional stage-2 repair set; optionally a small JSON that Stage III can consume.

### Phase 4 — Orchestration and docs

10. **Runner script**
    - **Script:** `benchmarking/run_stage2_intervention.py` (or similar)
    - **Role:** CLI with flags for Path A vs B, edge subset, n_seeds, n_trials_per_arm, output dir. Calls the scripts above in order and writes deliverables under `data/trail_causal_outputs/stage2/`.

11. **Docs**
    - Extend `CAUSAL_EXPLORATION_PLAN.md` or this file with: final paths, how to run Stage II, interpretation of validated edges and Priority.

---

## 7. Summary: Using Failure Log and Annotation to Replicate and Intervene

- **Replicate scenario:** Use **trace_path** from `trail_filtered` + **extract_run_config** to get (system_prompt, input_prompt, model). That is the “scenario”; re-running with the same config reproduces the original setup.
- **Intervention:** Modify the prompt with **repair(A)** / **inject(A)** / **placebo** from the operator library (filled with category A and optionally annotation evidence/description). Re-run with the modified config → new trace → label with `run_eval.py` → P(A), P(B), etc.
- **Failure log and annotation:** (1) Define **which** traces have A and B (onsets + capri edges). (2) Define **where** A occurs (span_id) for ΔT_A and for evidence in operators. (3) Provide **evidence/description** to fill repair/inject templates. (4) Baseline P(A), P(B) from current annotations; post-intervention P(A), P(B) from new traces’ labels.

This plan yields the four deliverables (validated edge set with weights, operator library, controllability, and graph-guided mitigation policy) and reuses your existing data and eval pipeline as much as possible.
