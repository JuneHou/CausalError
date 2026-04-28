# Methodology: Causal Graph-Augmented LLM-as-Judge Evaluation

## 4. Methods

We cast error detection in agent traces as a structured prediction task and evaluate a family of LLM-as-judge methods that progressively incorporate structural and causal information. All methods share a common taxonomy $\mathcal{C}$ of 20 leaf-level error categories (e.g., *Hallucination: Language-only*, *Goal Deviation*, *Context Handling Failures*) organized into three top-level branches: Reasoning Errors, System Execution Errors, and Planning and Coordination Errors. The judge's task is to identify a set of error instances $\hat{E} = \{(c_i, \ell_i)\}$, where $c_i \in \mathcal{C}$ is an error category and $\ell_i$ is the `span_id` hex identifier of the span where the error first manifests.

### 4.1 Task Formulation

Let a trace $T = (s_1, s_2, \ldots, s_n)$ be a sequence of spans ordered by execution time, where each span $s_i$ carries a unique identifier $\text{id}(s_i)$, a human-readable name $\text{name}(s_i)$, and textual content recording the agent's inputs, outputs, and tool interactions. The ground-truth annotation for $T$ is a set $E^* = \{(c_j^*, \ell_j^*)\}$ of (category, span\_id) pairs. An automatic judge $\mathcal{J}$ receives $T$ (serialized as JSON) and a task prompt, and produces a prediction $\hat{E}(T)$.

We evaluate predictions against $E^*$ using three complementary metrics computed per trace and averaged macro across traces:

- **Weighted-F1 (W-F1):** Binary indicator vector over $\mathcal{C}$ (1 if any error of that category is predicted), compared to the ground-truth indicator vector using weighted F1-score.
- **Location Accuracy (Loc):** Fraction of ground-truth span locations $\ell^*$ that appear in the predicted set $\{\hat{\ell}_i\}$.
- **Joint Accuracy (Joint):** Fraction of ground-truth (category, span\_id) pairs $(c^*, \ell^*)$ that are exactly matched in $\hat{E}$.

### 4.2 Baseline: Zero-shot LLM-as-Judge

The baseline judge receives the full taxonomy definition $\mathcal{C}$ and the serialized trace $T$ in a single prompt, and is instructed to output a JSON object listing detected errors:

$$\hat{E}_{\text{base}} = \mathcal{J}(\,\text{prompt}(\mathcal{C},\, T)\,)$$

No external knowledge about error co-occurrence or causal structure is provided. The prompt instructs the model to identify the *first* span of each error (or the *last* span for Resource Abuse), and to output only leaf-level taxonomy categories.

### 4.3 Span Index Augmentation (+SI)

One failure mode of the baseline is span location hallucination: the judge predicts a span identifier that does not exist in $T$. To address this, the **Span Index** (+SI) variant prepends a compact navigation table $\mathcal{I}(T)$ to the prompt before the trace body:

$$\mathcal{I}(T) = \bigl\{\,(\text{id}(s_i),\; \text{name}(s_i))\,\bigr\}_{i=1}^{n}$$

This enumeration, ordered by execution step, lists every valid span identifier with its human-readable label, constraining the judge's output space for the `location` field to verified values. Invalid-location predictions are post-hoc filtered by checking each predicted $\hat{\ell}_i$ against the set $\{\text{id}(s_i)\}$. Span index augmentation is orthogonal to the graph methods below and can be combined with any of them.

### 4.4 Static Causal Graph Prompt Injection (+GI)

The causal graph $\mathcal{G}_C = (\mathcal{V}, \mathcal{E}_C)$ is derived from the CAPRI pipeline described in Section 3: each validated edge $(A \to B) \in \mathcal{E}_C$ carries a weight $w_{AB}$ corresponding to the counterfactually estimated causal effect of error $A$ on the subsequent occurrence of error $B$.

In the **static graph injection** (+GI) variant, the complete validated edge set $\mathcal{E}_C$ is formatted as a guidance block and prepended to the taxonomy in the prompt:

$$\text{guidance}(\mathcal{E}_C) = \bigl\{\,A \to B \;(\text{strength: } w_{AB})\,\bigr\}_{(A,B) \in \mathcal{E}_C}$$

The prompt instructs the judge: *"When you identify an error of type $A$, actively look for errors of type $B$ in subsequent spans, as $B$ has been found to causally follow $A$."* This single-pass approach augments the judge's holistic reading with data-driven attention cues, encouraging it to pursue causal consequence chains without altering the inference structure of the call:

$$\hat{E}_{+\text{GI}} = \mathcal{J}\!\left(\,\text{prompt}(\mathcal{C},\, \text{guidance}(\mathcal{E}_C),\, T)\,\right)$$

Because the graph is injected globally and identically for every trace, this method is **static**: the graph context does not condition on what errors have already been found in $T$.

### 4.5 Dynamic Causal Graph Injection (+GI+SI, Complete Model)

The limitation of static injection is that the full edge set introduces a fixed, trace-agnostic prior regardless of which error categories actually appear in the trace. We address this with a **dynamic two-pass** procedure that adapts the graph context to the errors detected in a first reading of the trace.

#### Pass 1 — Holistic Detection

The first pass uses the same prompt as +GI+SI (taxonomy + causal guidance block + span index), producing an initial error set $\hat{E}_1$ with detected categories $D = \{c : (c, \cdot) \in \hat{E}_1\}$.

#### Graph Propagation — Subgraph Filtering

Given $D$, we compute a boosted relevance score for each candidate target category $B \notin D$:

$$\text{boosted}(B) = \sum_{\substack{(A \to B) \in \mathcal{E}_C \\ A \in D}} w_{AB}$$

This aggregates the causal influence of all detected source errors on each undetected target. Pass 2 is triggered only when the filtered edge set

$$\mathcal{E}_2 = \bigl\{\,(A, B, w_{AB}) \in \mathcal{E}_C : A \in D,\; B \notin D,\; \text{boosted}(B) > \tau\,\bigr\}$$

is non-empty, where $\tau$ is a propagation threshold (default $\tau = 0.10$). Traces for which Pass 1 detected no graph source categories bypass Pass 2 entirely.

#### Pass 2 — Targeted Re-analysis

When $\mathcal{E}_2 \neq \emptyset$, a second LLM call is issued with a purpose-built prompt that provides: (i) the Pass 1 summary $\hat{E}_1$ as a list of already-detected errors, (ii) the trace-specific subgraph $\mathcal{E}_2$ formatted as $(A \to B \;[\text{weight: } w_{AB}])$ pairs, and (iii) the span index $\mathcal{I}(T)$. The judge is instructed to output **only** errors not already present in $\hat{E}_1$, targeting the specific category types $\{B : (A, B, \cdot) \in \mathcal{E}_2\}$ indicated as causally likely. To maximize output quality within the available token budget, Pass 2 disables extended reasoning (thinking) for reasoning-capable models, as the task is targeted and well-constrained.

#### Merge and Deduplication

The final prediction merges both passes, removing any Pass 2 prediction whose category was already identified in Pass 1:

$$\hat{E}_{+\text{GI}+\text{SI}} = \hat{E}_1 \;\cup\; \bigl\{\,(c, \ell) \in \hat{E}_2 : c \notin D\,\bigr\}$$

This two-pass dynamic design offers two advantages over static injection. First, by conditioning the subgraph on $D$, it presents only causally relevant edges to the second-pass judge, reducing noise and prompt length. Second, Pass 2's explicit listing of already-found errors prevents the judge from re-reporting the same categories, producing a cleaner union prediction. The overall pipeline per trace is summarized in Figure~\ref{fig:pipeline}.

### 4.6 Summary of Experimental Conditions

| Condition | Graph | Span Index | Passes |
|---|---|---|---|
| Baseline | — | — | 1 |
| +SI | — | ✓ | 1 |
| +GI | $\mathcal{E}_C$ (static) | — | 1 |
| +GI+SI | $\mathcal{E}_C$ (static) | ✓ | 1 |
| **+DGI+SI** (ours) | $\mathcal{E}_2$ (dynamic) | ✓ | **2** |

All conditions use the same underlying LLM judge and identical taxonomy. The causal graph $\mathcal{G}_C$ used in +GI, +GI+SI, and +DGI+SI is the set of intervention-validated edges from the CAPRI pipeline (Section 3), using only edges with counterfactually confirmed causal effects ($\delta < -\tau_\Delta$).
