# Workstream C: Causal-vs-Corr Mechanism Ablation (Final)

This note documents the final Workstream C analysis in EMNLP ablation style, using completed outputs only.

## Setup

- Scope: 30 audited TRAIL traces.
- Design: contrast-stratified comparison of `corr-union +GI` vs `causal-only +GI` on matched traces.
- Annotation: benchmark taxonomy + mechanism labels (buckets/tags).
- QC: manual re-labeling subset with adjudication.

## Main mechanism outcomes

Mechanism bucket distribution (`n=30`):

- `corr-added-gain`: 10 (33.33%)
- `corr-induced-harm`: 10 (33.33%)
- `shared-failure`: 8 (26.67%)
- `causal-preserving-neutral`: 2 (6.67%)

Interpretation: correlation augmentation is not uniformly beneficial; gains and regressions occur at similar frequency under matched evidence.

## Benchmark taxonomy perspective

Gold taxonomy signal is concentrated in reasoning/planning behavior rather than system/API errors.

- L1 counts from gold leaves:
  - `Reasoning Errors`: 50
  - `Planning and Coordination Errors`: 27
- Most frequent gold leaves:
  - `Instruction Non-compliance`: 17
  - `Goal Deviation`: 15
  - `Tool-related`: 9
  - `Tool Selection Errors`: 7
  - `Formatting Errors`: 7
  - `Task Orchestration`: 6

This indicates the critical failure mass is in behavior-control categories where propagation design has first-order effect.

## Interaction between taxonomy and mechanism labels

At aggregate leaf level, corr-union shifts both error recovery and overprediction relative to causal-only:

- False negatives: 34 (corr) vs 48 (causal)  -> corr reduces misses by 14.
- False positives: 59 (corr) vs 45 (causal) -> corr adds 14 extra false activations.

This symmetric shift indicates a structured recall-precision trade-off, not random noise:
correlation edges recover missing categories while also propagating unsupported categories.

## Mechanism tags and corr-edge roles

Top pattern tags:

- `over-propagation-fp-chain`: 13 (43.33%)
- `already-solved-no-graph-needed`: 11 (36.67%)
- `spurious-correlation-trigger`: 8 (26.67%)
- `missing-context-recovery`: 5 (16.67%)
- `localization-drift`: 5 (16.67%)
- `precision-preserving-correction`: 5 (16.67%)

Corr-edge role distribution:

- `harmful`: 14
- `neutral`: 10
- `beneficial`: 5
- `unknown`: 1

Mechanism implication: recovery effects exist, but harmful propagation patterns are more prevalent overall.

## Reliability (QC)

QC agreement (manual vs GPT, `n=5` common traces):

- Mean agreement: 0.8667
- Mean Cohen's kappa: 0.8148
- Per field:
  - `mechanism_bucket`: agreement 1.0000, kappa 1.0000
  - `corr_edge_role`: agreement 0.6000, kappa 0.4444
  - `impact_severity`: agreement 1.0000, kappa 1.0000

Disagreements are concentrated in corr-edge polarity (`harmful` vs `neutral`), while structural labels remain stable.

## Threats to validity

- Small QC subset (`n=5`) means reliability evidence is strong but preliminary.
- Audited set is GAIA-heavy (27/30), so generalization to SWE-heavy settings should be stated conservatively.

## Final takeaway

Corr-union provides measurable category recovery on a subset of traces, but introduces comparable over-propagation risk; causal structure is therefore necessary to keep augmentation behavior reliable and controllable.

