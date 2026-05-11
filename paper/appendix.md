# Appendix Notes

Auxiliary clarifications and post-hoc analyses that don't belong in the main
methodology / implementation sections but are useful for reviewers and
reproducibility. Each subsection below is a self-contained note; new notes
should be appended as separate `## …` sections.

---

## A.1 Interpreting the intervention-validity threshold $\tau_\Delta$

The validation threshold $\tau_\Delta = 0.15$ is best read as a *minimum effect size of practical interest* rather than a statistical null-rejection criterion. The placebo null distributions (TRAIL: $\mu=-0.519$, $\sigma=0.196$; MAST: $\mu=-0.124$, $\sigma=0.252$) are shifted negative because the patch + rerun procedure systematically attenuates downstream errors irrespective of the patched cause — a confound that is not removed by within-trace onset permutation. Consequently, $\tau_\Delta$ does not bound a Type-I error rate in the classical sense; instead, an edge is reported as validated when the absolute reduction in $B$'s post-intervention occurrence exceeds 15 percentage points and the intervention sample is non-empty. We disclose the placebo distribution alongside each edge set as a sanity check on directional consistency rather than as a significance test. A stricter $\sigma$- or quantile-calibrated threshold is straightforward to apply post-hoc to the released `effect_edges.json` and does not require re-running the intervention pipeline.
