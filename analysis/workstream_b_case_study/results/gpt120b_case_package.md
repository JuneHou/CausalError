# GPT-120B Workstream B Case Package

Selected cases (2 working + 2 not-working) with traceable judge logic and reusable writing snippets.

## Pipeline

- Source shortlist: `gpt120b_top4_shortlist.csv` (locked 4 IDs).
- For each case, gather artifacts from existing outputs only:
  - gold annotation, baseline prediction, causal-only prediction, corr-union prediction, corr metadata, and trace data.
- Normalize and summarize into one structured package for writing.
- Emit both machine-readable JSON and reviewer-facing Markdown.

## Filters and Selection Logic (already applied before this package)

- Model fixed to `GPT-oss-120B`.
- Case composition fixed to `2 working + 2 not_working`.
- Working eligibility:
  - `Δ(corr-causal) > 0`, `Δ(corr-baseline) > 0`, `ΔW-F1 > 0`, `ΔLoc > 0`.
- Joint handling:
  - prefer `ΔJoint > 0`; if insufficient candidates, relax joint only.
- User-approved hard include:
  - SWE working case `c104d0e28f4f8dddeea1dd90b4138e5a`.
- Not-working eligibility:
  - `Δ(corr-causal) < 0` with at least one negative metric delta.

## Output Schema (What each case includes)

- case type (`working` / `not_working`)
- split and trace ID
- delta metrics (`ΔW-F1`, `ΔLoc`, `ΔJoint`, plus corr-vs-causal/base)
- prompt/question snippet (when extractable)
- gold / baseline / causal-only / corr category heads
- score snapshot (reliability, instruction adherence, plan optimization, overall)
- corr metadata:
  - `pass1_detected`
  - `pass2_triggered`
  - `pass2_filtered_edges`
  - `pass2_new_errors`
- draft takeaway sentence
- writing slots for:
  - activated-edge evidence
  - reviewer-convincing rationale

## Case 1: `dbc070b918d4a052c0b686081408fb52` (working, GAIA_dedup)

- **Prompt snippet**: On July 15, 2008, Phys.org published an article about a catastrophe. Find the explosive force of this catastrophe according to Encyclopedia Britannica, then find the name of the US nuclear test that had the same yield. Your answer should only be the last word
- **Reference answer**: `Bravo`
- **Delta vs causal-only**: W-F1 `+0.100`, Loc `+1.000`, Joint `+0.667`
- **Pass-1 detected**: Instruction Non-compliance, Goal Deviation, Incorrect Problem Identification, Language-only, Tool Selection Errors
- **Pass-2 filtered edges**: `3`; **new errors**: `2`
- **Gold categories (head)**: Language-only, Goal Deviation, Instruction Non-compliance
- **Baseline categories (head)**: Language-only, Instruction Non-compliance, Tool Selection Errors, Goal Deviation
- **Causal-only categories (head)**: Instruction Non-compliance, Language-only, Formatting Errors, Incorrect Problem Identification, Poor Information Retrieval
- **Corr-union categories (head)**: Language-only, Tool Selection Errors, Instruction Non-compliance, Incorrect Problem Identification, Goal Deviation
- **Draft takeaway**: Corr-union improves over causal-only (ΔW-F1 +0.100, ΔLoc +1.000, ΔJoint +0.667) while preserving explicit graph-trigger traceability.
- **Writing slots**:
  - Activated-edge evidence: `<fill from pass2 / prompt trace>`
  - Why this is convincing for reviewers: `<1-2 lines>`

## Case 2: `c104d0e28f4f8dddeea1dd90b4138e5a` (working, SWE_Bench_dedup)

- **Prompt snippet**: You will be provided with a partial code base and an issue statement explaining a problem to resolve.  <issue> Rule L060 could give a specific error message At the moment rule L060 flags something like this:    ```  L:  21 | P:   9 | L060 | Use 'COALESCE' inst
- **Delta vs causal-only**: W-F1 `+0.171`, Loc `+0.333`, Joint `+0.000`
- **Pass-1 detected**: Instruction Non-compliance, Resource Abuse
- **Pass-2 filtered edges**: `3`; **new errors**: `2`
- **Gold categories (head)**: Instruction Non-compliance, Formatting Errors, Instruction Non-compliance, Context Handling Failures
- **Baseline categories (head)**: Instruction Non-compliance
- **Causal-only categories (head)**: Instruction Non-compliance, Resource Abuse
- **Corr-union categories (head)**: Instruction Non-compliance, Resource Abuse, Context Handling Failures, Tool Output Misinterpretation
- **Draft takeaway**: Corr-union improves over causal-only (ΔW-F1 +0.171, ΔLoc +0.333, ΔJoint +0.000) while preserving explicit graph-trigger traceability.
- **Writing slots**:
  - Activated-edge evidence: `<fill from pass2 / prompt trace>`
  - Why this is convincing for reviewers: `<1-2 lines>`

## Case 3: `ea313eef484bb042ddb079771359c8e6` (not_working, GAIA_dedup)

- **Prompt snippet**: In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this location called in the official script for the episode? Give the setting exactly as it appears in the first scene heading.
- **Reference answer**: `THE CASTLE`
- **Delta vs causal-only**: W-F1 `+0.067`, Loc `-1.000`, Joint `-1.000`
- **Pass-1 detected**: Instruction Non-compliance, Tool Selection Errors
- **Pass-2 filtered edges**: `3`; **new errors**: `2`
- **Gold categories (head)**: Instruction Non-compliance
- **Baseline categories (head)**: Tool Selection Errors, Tool Definition Issues
- **Causal-only categories (head)**: Tool Selection Errors, Instruction Non-compliance, Formatting Errors, Goal Deviation, Incorrect Problem Identification
- **Corr-union categories (head)**: Tool Selection Errors, Instruction Non-compliance, Goal Deviation, Formatting Errors
- **Draft takeaway**: Corr-union regresses vs causal-only (ΔW-F1 +0.067, ΔLoc -1.000, ΔJoint -1.000); use as bounded failure evidence.
- **Writing slots**:
  - Activated-edge evidence: `<fill from pass2 / prompt trace>`
  - Why this is convincing for reviewers: `<1-2 lines>`

## Case 4: `72822db6e120878d916b515c2501246b` (not_working, SWE_Bench_dedup)

- **Prompt snippet**: You will be provided with a partial code base and an issue statement explaining a problem to resolve.

<issue>
dbt postgres fix command errors with UnicodeEncodeError and also wipes the .sql file
_If this is a parsing or linting issue, please include a minimal
- **Delta vs causal-only**: W-F1 `-0.250`, Loc `-1.000`, Joint `+0.000`
- **Pass-1 detected**: Tool Output Misinterpretation, Instruction Non-compliance, Poor Information Retrieval, Incorrect Problem Identification
- **Pass-2 filtered edges**: `3`; **new errors**: `2`
- **Gold categories (head)**: Formatting Errors
- **Baseline categories (head)**: Instruction Non-compliance, Instruction Non-compliance, Tool Output Misinterpretation, Incorrect Problem Identification
- **Causal-only categories (head)**: Formatting Errors, Instruction Non-compliance, Tool Output Misinterpretation, Poor Information Retrieval, Context Handling Failures
- **Corr-union categories (head)**: Instruction Non-compliance, Incorrect Problem Identification, Tool Output Misinterpretation, Poor Information Retrieval, Resource Abuse
- **Draft takeaway**: Corr-union regresses vs causal-only (ΔW-F1 -0.250, ΔLoc -1.000, ΔJoint +0.000); use as bounded failure evidence.
- **Writing slots**:
  - Activated-edge evidence: `<fill from pass2 / prompt trace>`
  - Why this is convincing for reviewers: `<1-2 lines>`
