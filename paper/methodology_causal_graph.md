# Methodology: Causal Error Graph Construction and Validation

## Causal Graph Construction and Validation

We construct a **taxonomy-agnostic causal graph** over agent error categories via a three-stage pipeline: probabilistic screening (Suppes), structure learning (CAPRI), and counterfactual intervention validation. The pipeline operates on *error onset data* derived from annotated agent traces — sequences of (trace\_id, error\_category, onset\_rank) tuples — and makes no assumptions about the specific error taxonomy, making it reusable across agent benchmarks and annotation schemas.

### Stage I: Onset Extraction

For each annotated trace, we identify the *onset* of each error category as the rank (by timestamp) of the earliest span annotated with that category. This produces a per-trace onset table:

$$\text{onset}(A, t) = \min_{s \in \text{spans}(t)} \{ \text{rank}(s) : A \in \text{labels}(s) \}$$

A category $A$ is *present* in trace $t$ if at least one span is annotated with $A$ ($\text{present}(A, t) = 1$). Ties in onset are retained and handled conservatively in the precedence calculation.

### Stage II: Suppes Probabilistic Screening

We apply Suppes' probabilistic theory of causation~\cite{suppes1970probabilistic} to screen candidate causal edges. An edge $A \to B$ is retained as a *prima facie cause* if and only if two conditions hold simultaneously.

**Condition 1 — Temporal Precedence.** Among traces where both $A$ and $B$ co-occur with non-tied onsets, $A$ must precede $B$ in the majority of cases:

$$\text{prec}(A \to B) = \frac{|\{t : \text{onset}(A,t) < \text{onset}(B,t)\}|}{|\{t : \text{onset}(A,t) \neq \text{onset}(B,t), \text{present}(A,t)=\text{present}(B,t)=1\}|} \geq \tau_{\text{prec}}$$

**Condition 2 — Probability Raising.** The presence of $A$ must raise the probability of $B$ above its base rate:

$$\Delta\text{PR}(A \to B) = P(B{=}1 \mid A{=}1) - P(B{=}1 \mid A{=}0) \geq \tau_{\Delta\text{PR}}$$

where the probabilities are computed over all traces in the corpus, treating presence as a binary event. A minimum joint co-occurrence count $n_{AB} \geq \tau_{\text{joint}}$ is required to suppress spurious low-count edges.

The Suppes screen produces a *candidate graph* $\mathcal{G}_S$ with directed edges satisfying both conditions. This graph may contain spurious associations arising from common causes or transitive chains.

### Stage III: CAPRI Structure Learning

We apply CAPRI (Causal Analysis with Probabilistic Reasoning Infrastructure) to prune $\mathcal{G}_S$ to a minimal sufficient causal skeleton. CAPRI performs score-based DAG learning via hill-climbing, restricted to the Suppes candidate edge set.

**Scoring.** Each node $B$ is scored given its parent set $\text{pa}(B) \subseteq \mathcal{G}_S$ using a Bayesian information criterion. For each of the $2^{|\text{pa}(B)|}$ parent configurations $c$, we estimate the conditional probability of $B$ with Laplace smoothing:

$$\hat{p}_{B \mid c} = \frac{n_{B=1,c} + 0.5}{n_c + 1.0}$$

The per-node score is:

$$\text{score}(B, \text{pa}(B)) = \underbrace{-2\sum_c \left[n_{B=1,c} \log \hat{p} + n_{B=0,c} \log(1-\hat{p})\right]}_{\text{deviance}} + \lambda \cdot 2^{|\text{pa}(B)|} \cdot \log n$$

where $\lambda = 1$ for BIC and the penalty term $2^{|\text{pa}(B)|}$ counts free parameters. We use the AIC variant ($\lambda = 1$, replacing $\log n$ with 2) to favor sensitivity over sparsity, retaining edges that provide even modest explanatory gain.

**Hill-climbing.** Starting from the empty graph, the algorithm iteratively proposes three classes of moves — edge addition, removal, or reversal — subject to (i) the move must use an edge in $\mathcal{G}_S$, and (ii) the resulting graph must be acyclic. Each move is accepted if it strictly reduces the total score $\sum_B \text{score}(B, \text{pa}(B))$. The procedure terminates when no improving neighbor exists or after a maximum of 500 iterations.

The output is a directed acyclic graph $\mathcal{G}_C \subseteq \mathcal{G}_S$ — the CAPRI causal graph.

### Stage IV: Counterfactual Intervention Validation

To confirm that edges in $\mathcal{G}_C$ represent genuine causal relationships rather than residual associations, we validate each edge $A \to B$ via a $\text{do}(A{=}0)$ counterfactual intervention. This operationalizes Pearl's do-calculus~\cite{pearl2009causality} at the span level.

**Protocol.** For each eligible trace containing both error $A$ at span $t_A$ and error $B$ at a later span $t_B > t_A$:

1. **Patch generation.** An LLM rewrites the output of span $t_A$ to eliminate error $A$, using a category-specific repair template from a patch library. The patch is validated with rule-based postchecks (format compliance, semantic preservation, no ungrounded content).

2. **Counterfactual rerun.** The corrected span output is injected at position $t_A$, and the agent is re-continued for up to $K$ steps using an LLM, replaying the original tool execution results in order. This isolates the effect of the error correction from downstream stochasticity.

3. **Judge-A (treatment validity).** An LLM judge assesses whether error $A$ has been eliminated in the rerun. Only interventions judged as successfully resolving $A$ proceed to the outcome assessment.

4. **Judge-B (outcome label).** For each validated intervention, a separate LLM judge assesses the status of error $B$ in the counterfactual continuation, producing a presence label $\hat{y}_B^{\text{cf}} \in \{0, 1\}$.

**Effect aggregation.** For each edge $A \to B$, the causal effect is estimated as the change in $B$-presence under intervention:

$$\Delta(A \to B) = \mathbb{E}[\hat{y}_B^{\text{cf}} \mid \text{resolved}(A)] - \mathbb{E}[y_B^{\text{baseline}} \mid \text{resolved}(A)]$$

An edge is *validated* if $\Delta(A \to B) < -\tau_\Delta$ and the number of valid interventions $n \geq n_{\min}$, indicating that correcting $A$ reliably reduces the occurrence of $B$.

**Placebo control.** To calibrate the threshold $\tau_\Delta$, we estimate a null distribution by permuting error onset ranks within traces ($R=1000$ samples), yielding a null mean and standard deviation against which empirical effect sizes are compared.

### Taxonomy-Agnostic Design

The pipeline is fully agnostic to the specific error taxonomy. It requires only: (i) a set of error category labels, (ii) per-trace annotation files mapping spans to labels, and (iii) temporal ordering of spans. The Suppes and CAPRI algorithms operate over category names as abstract discrete events, with no category-specific logic embedded in the pipeline. This enables direct reuse across benchmarks with different annotation schemas, agent architectures, or task domains, provided the same annotation protocol is applied.
