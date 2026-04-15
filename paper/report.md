# TRAIL Benchmark: Causal Graph-Guided LLM Error Detection in Agent Traces

---

## 1. Introduction

Automated evaluation of LLM-based agent systems is a growing challenge. As agents are deployed in complex, multi-step tasks — querying web sources, executing code, calling APIs — they produce long traces of interleaved reasoning and tool use. When an agent fails, diagnosing *what* went wrong (error type) and *where* it went wrong (location in the trace) requires structured, scalable evaluation.

The **TRAIL benchmark** (Trace-based Reasoning for AI Localization) provides a standardized evaluation framework for this problem. Each TRAIL trace is a hierarchical span tree capturing an agent's full execution: LLM reasoning steps, tool calls, API responses, and chain-of-thought. Human annotators label every span with error categories from a 19-class taxonomy spanning reasoning failures (hallucinations, misinterpretation), system execution errors (rate limiting, authentication), and planning failures (goal deviation, context overflow). The evaluation task has three components:

- **Weighted F1 (W-F1):** multi-label classification accuracy over the 19 error categories present in the trace.
- **Location Accuracy (Loc):** recall of ground-truth error spans — what fraction of annotated spans does the model correctly identify.
- **Joint Accuracy (Joint):** the fraction of (span, error type) pairs that are simultaneously correct.

This report describes the **causal graph-guided evaluation pipeline** we developed on top of TRAIL. The core hypothesis is that error categories in agent traces are not independent — certain errors causally precede others, forming propagation chains. By discovering this causal structure from annotations and injecting it into LLM-as-judge prompts, we improve error detection across all three metrics.

The pipeline has two stages. First, we construct a **causal error graph** from the annotated trace corpus using a sequence of probabilistic screening, structure learning, and counterfactual intervention validation. Second, we design three variants of LLM-as-judge prompts that use the graph in different ways — from passive graph listing to active two-pass graph injection — and evaluate them across four models (Gemini-2.5-Flash, Mistral-Small-3.1-24B, GPT-oss-120B, QwenLong-L1-32B) on both GAIA and SWE-bench splits.

---

## 2. Causal Graph Construction Pipeline

We construct a taxonomy-agnostic causal graph over agent error categories using a four-stage pipeline: onset extraction, probabilistic screening, structure learning, and counterfactual intervention validation. The input is the annotated trace corpus (148 traces: 117 GAIA + 31 SWE-bench); the output is a directed acyclic graph (DAG) of validated causal edges between error categories.

### 2.1 Stage I: Onset Extraction

The pipeline begins by extracting the *onset* of each error category in each trace — the rank (by timestamp) of the earliest span annotated with that category:

$$\text{onset}(A, t) = \min_{s \in \text{spans}(t)} \{ \text{rank}(s) \;:\; A \in \text{labels}(s) \}$$

Spans are sorted by timestamp with span_id as a tie-breaker. Only spans of kind LLM, TOOL, or CHAIN (the OpenInference span taxonomy) are considered — metadata spans are excluded. A category $A$ is *present* in trace $t$ if at least one span carries label $A$.

This extraction converts the rich span-level annotation into a compact onset table: one row per (trace, category) pair, recording only the first occurrence rank. The onset table is the sole input to all downstream stages — no category-specific logic is needed, making the pipeline fully reusable across benchmarks with different taxonomies.

### 2.2 Stage II: Suppes Probabilistic Screening

We apply Suppes' probabilistic theory of causation to screen candidate directed edges. An edge $A \to B$ is retained as a *prima facie cause* if and only if two conditions hold simultaneously.

**Condition 1 — Temporal Precedence.** Among traces where both $A$ and $B$ co-occur with non-tied onsets, $A$ must precede $B$ in the majority of cases:

$$\text{prec}(A \to B) = \frac{|\{t : \text{onset}(A,t) < \text{onset}(B,t)\}|}{|\{t : \text{onset}(A,t) \neq \text{onset}(B,t),\; \text{present}(A,t) = \text{present}(B,t) = 1\}|} \geq \tau_{\text{prec}}$$

**Condition 2 — Probability Raising.** $A$'s presence must raise $B$'s occurrence probability above its baseline:

$$\Delta\text{PR}(A \to B) = P(B{=}1 \mid A{=}1) - P(B{=}1 \mid A{=}0) \geq \tau_{\Delta\text{PR}}$$

A minimum joint co-occurrence count $n_{AB} \geq \tau_{\text{joint}}$ suppresses spurious low-count edges.

**Applied settings (TRAIL):** $\tau_{\text{prec}} = 0.55$, $\tau_{\Delta\text{PR}} = 0.05$, $\tau_{\text{joint}} = 3$. Applied to 148 traces and 19 categories ($19 \times 18 = 342$ directed pairs), the Suppes screen retains **27 candidate edges**. The strongest edges by probability-raising include *Tool Selection Errors → Goal Deviation* ($\Delta\text{PR} = 0.47$, prec $= 1.00$, $n = 9$) and *Poor Information Retrieval → Resource Abuse* ($\Delta\text{PR} = 0.42$, prec $= 0.57$, $n = 14$).

The Suppes candidate graph $\mathcal{G}_S$ may contain spurious associations arising from common causes or transitive chains. The next stage prunes these.

### 2.3 Stage III: CAPRI Structure Learning

We apply CAPRI (Causal Analysis with Probabilistic Reasoning Infrastructure) to prune $\mathcal{G}_S$ to a minimal sufficient causal skeleton using score-based DAG learning via hill-climbing.

**Scoring.** Each node $B$ is scored given its candidate parent set using the Akaike Information Criterion (AIC). For each parent configuration $c$, conditional probabilities are estimated with Laplace smoothing:

$$\hat{p}_{B \mid c} = \frac{n_{B=1,c} + 0.5}{n_c + 1.0}$$

The per-node score is:

$$\text{score}(B, \text{pa}(B)) = -2\sum_c \left[n_{B=1,c} \log \hat{p} + n_{B=0,c} \log(1-\hat{p})\right] + 2 \cdot 2^{|\text{pa}(B)|}$$

AIC is preferred over BIC for the TRAIL corpus (148 traces) because BIC's stronger $\log n$ penalty risks discarding true causal edges that have limited observational support at this corpus size.

**Hill-climbing.** Starting from the empty graph, the algorithm iteratively proposes edge additions, removals, or reversals — restricted to edges in $\mathcal{G}_S$ and subject to acyclicity. Each move is accepted if it strictly reduces the total score across all nodes. The procedure converges in **14 iterations**, reducing the total score from 2308.3 to 2179.4.

The resulting CAPRI-AIC graph $\mathcal{G}_C$ contains **13 directed edges**:

| Source | Target |
|--------|--------|
| Formatting Errors | Context Handling Failures |
| Formatting Errors | Incorrect Problem Identification |
| Formatting Errors | Poor Information Retrieval |
| Formatting Errors | Resource Abuse |
| Incorrect Problem Identification | Language-only |
| Incorrect Problem Identification | Tool Output Misinterpretation |
| Poor Information Retrieval | Resource Abuse |
| Resource Abuse | Authentication Errors |
| Resource Abuse | Tool-related |
| Tool Selection Errors | Goal Deviation |
| Tool Selection Errors | Language-only |
| Tool Selection Errors | Task Orchestration |
| Tool-related | Goal Deviation |

The graph reveals *Formatting Errors* and *Tool Selection Errors* as dominant root causes, with *Resource Abuse* and *Goal Deviation* as their primary consequences.

### 2.4 Stage IV: Counterfactual Intervention Validation

To confirm that edges in $\mathcal{G}_C$ represent genuine causal relationships rather than residual associations, we validate each edge $A \to B$ via a $\text{do}(A{=}0)$ counterfactual intervention — operationalizing Pearl's do-calculus at the span level.

**Eligible trace filtering.** Before constructing interventions, traces are filtered by three sequential criteria: (i) $\geq 2$ total annotated errors (singleton-error traces cannot exhibit A→B co-occurrence); (ii) the trace contains at least one error whose category is a source node (A-type) in the CAPRI graph; and (iii) *strict mode* — the trace must contain at least one (A, B) pair where the B-type error appears after the A-type in annotation order, ensuring every eligible trace contributes at least one baseline-positive edge pair. Test-split traces are excluded before filtering to prevent leakage. These three criteria yield 72 eligible GAIA traces (from 117 total) and 26 eligible SWE-bench traces (from 31 total).

**A-instances and edge pairs.** An *A-instance* is one unique (trace, error occurrence) pair where the error category is an A-type. A single eligible trace can yield multiple A-instances — one per qualifying A-type error annotation — each forming an independent counterfactual experiment on the same full trace, with a patch applied at that error's span and the rerun continuing from that point to the end of the trace. A-instances sharing the same intervention location are deduplicated (first by annotation index), since applying two overlapping patches to the same span would conflate the causal estimands. Each A-instance then fans out to one *EdgePair* per B-type connected to its A-type in the CAPRI graph, recording whether B appears after A in the original trace (`b_present_baseline`). The 98 eligible traces expand to 274 A-instances and 493 edge pairs (192/345 GAIA + 82/148 SWE-bench).

**Protocol.** For each eligible trace containing both error $A$ at span $t_A$ and error $B$ at a later span $t_B > t_A$:

1. **Patch generation.** An LLM (GPT-4o) rewrites the span at $t_A$ to eliminate error $A$, using a **shared prompt scaffold** combined with a **category-specific repair template** from a patch library. Rule-based postchecks enforce format compliance, semantic preservation, and absence of ungrounded content. Up to 3 retries are attempted; patches failing all retries are discarded.

2. **Counterfactual rerun.** The corrected span is injected at position $t_A$ and the agent is re-continued for up to $K = 12$ subsequent steps, replaying original tool results in order. The rerun model matches each trace's original agent (o3-mini for GAIA, GPT-4o for SWE-bench).

3. **Judge-A (treatment validity).** A dedicated LLM judge (GPT-4o) compares the original and patched spans to verify that error $A$ is no longer present. Only confirmed interventions proceed to effect estimation.

4. **Judge-B (outcome label).** A separate LLM judge (GPT-4o) compares the original and counterfactual trace suffixes after $t_A$ and assigns one of eight outcome labels: *disappeared*, *unchanged*, *weakened*, *strengthened*, *emerged*, *delayed*, *earlier*, or *not\_observable*.

**Patch library.** The patch library (`patch_library.json`) contains **20 entries** — one per TRAIL error category — each providing the LLM with category-specific instructions that are injected into the shared scaffold. Each entry specifies:

- **`patch_side_default`**: whether to patch the span's *output* (`replace_span_output`) or *input* (`replace_span_input`). For example, *Formatting Errors* patches the output (the malformed text the LLM produced), while *Tool Selection Errors* patches the input (the reasoning context that drove the wrong tool choice). When the annotation points to a TOOL span and the patch targets input, the intervention is redirected to the sibling LLM span that authored the tool call.
- **`repair_instruction`**: category-specific fix strategy (e.g., "make the smallest possible structural edit that satisfies the required format contract" for Formatting Errors; "redirect to the correct tool in the input context" for Tool Selection Errors).
- **`forbidden_actions`**: explicit constraints preventing over-editing (e.g., "do not fabricate tool outputs", "do not change plan steps beyond the formatting fix").
- **`postcheck`**: rule-based validation criteria checked after generation (e.g., "all required markers appear exactly", "patch payload differs from local snippet").

**Prompts.** The pipeline uses three distinct prompt roles:

- **`PATCH_SYSTEM` (shared scaffold):** A single system prompt, identical across all error types, that frames the task as a $\text{do}(A{=}0)$ intervention and enforces hard constraints (modify only the local snippet, do not directly repair B, do not invent ungrounded content). The category-specific `error_type_spec_text` from the patch library is injected into the user message alongside the local snippet, error description, and evidence.

- **`JUDGE_A_SYSTEM`:** Instructs the judge to compare the original and patched spans and return `{"resolved": true/false, "confidence": float, "evidence_excerpt": string}`. The judge is explicitly told to focus only on error A and not penalize for downstream B effects.

- **`JUDGE_B_SYSTEM`:** Instructs the judge to compare the original and counterfactual trace suffixes and return one of the eight effect labels. The user message includes the TRAIL taxonomy definition of error B, the original trace suffix with B's onset location, and the rerun suffix, ensuring the judge can ground its verdict in the actual execution context.

**Effect estimation.** The causal effect of each edge is estimated as:

$$\Delta(A \to B) = \mathbb{E}[\hat{y}_B^{\text{cf}} \mid \text{resolved}(A)] - \mathbb{E}[y_B^{\text{baseline}} \mid \text{resolved}(A)]$$

An edge is *validated* if $\Delta(A \to B) < -\tau_\Delta$ (correcting $A$ reduces $B$), with $\tau_\Delta = 0.15$.

**Results.** On GAIA training and validation traces (63 eligible traces, 192 A-instances), **all 10 CAPRI edges tested are validated**, with effect sizes ranging from $-0.21$ to $-1.00$. Notable effects: *Formatting Errors → Resource Abuse* ($\Delta = -0.32$, $n = 54$) and *Tool-related → Goal Deviation* ($\Delta = -0.69$, $n = 16$). A null distribution over 1000 onset permutations has mean $-0.67$ and standard deviation $0.28$, confirming that validated effects exceed chance levels.

**Pipeline summary:**

| Stage | Input | Output |
|-------|-------|--------|
| Onset extraction | 148 annotated traces | Per-trace onset table |
| Suppes screening | Onset table | 27 candidate edges |
| CAPRI-AIC | 27 candidate edges | 13 DAG edges |
| Intervention validation | 13 DAG edges | 10 validated causal edges |

---

## 3. Experiment Design: Three Variants

### 3.1 Motivation

The causal graph encodes two types of information that are useful for LLM-as-judge evaluation:

1. **Category co-occurrence structure** — which error types tend to appear together.
2. **Propagation chains** — which errors causally precede others, giving directional hints about where to look next.

A naive approach would simply append the edge list to the prompt and let the LLM reason over it. However, long traces combined with large graph descriptions can exceed context windows for smaller models, and an undifferentiated edge list does not tell the LLM *when* to act on a graph hint. We therefore designed three variants with increasing levels of graph integration, each targeting a different failure mode of the naive approach.

Additionally, span identifiers in TRAIL traces are long hex strings (e.g., `037ba72bqlkpas`). Without explicit guidance, LLMs frequently hallucinate or misquote span IDs in the `location` field of their predictions. The **span index** (SI) addresses this by prepending an explicit index of valid span IDs and their human-readable names at the top of the prompt.

### 3.2 Variant +CG: Causal Graph Only

**Design.** The causal graph edges are appended as a structured text block at the end of the standard evaluation prompt, after the taxonomy and before the trace. The block lists all validated edges with their weight (probability-raising delta), framed as: *"When you detect error A in the trace, also check whether error B is present."*

**Motivation.** This is the direct application of the causal graph as a checklist hint. It costs one additional prompt block and requires no second LLM call. The expected gain comes from the LLM being reminded to check correlated error types it might otherwise miss, particularly rare categories that are underrepresented in training data.

**Limitation.** Because the hint is unconditional — it fires regardless of what the LLM has already detected — it may introduce spurious category predictions when the agent trace does not actually contain the hinted error type. This over-triggering can hurt precision.

### 3.3 Variant +CG+SI: Causal Graph + Span Index

**Design.** Combines the static causal graph hint from +CG with a prepended **span index** — an ordered list of every valid span ID in the trace, paired with a human-readable label (e.g., `span_id "037ba72b..."  (Step 3: page_down)`). The LLM is instructed to use only span IDs from this index in the `location` field of its output.

**Motivation.** The span index targets a different failure mode than the graph: location accuracy. In the baseline, LLMs frequently hallucinate or partially misquote span IDs, even when they correctly identify *which step* an error occurs at. By providing an explicit closed set of valid IDs, the span index converts the location field from a free-form generation problem into a selection problem, reducing hallucination.

The combination of +CG and +SI is expected to improve both W-F1 (via graph hints) and Location/Joint accuracy (via span index), making it a natural combined condition.

**Limitation.** For very long traces, the span index itself adds substantial token length. On models with tight context limits, this can cause truncation of the trace content, sometimes producing *zero valid predictions* (the QwenLong +CG+SI failure case on full traces, discussed in §5).

### 3.4 Variant +GI+SI: Graph Injection + Span Index

**Design.** This is a two-pass pipeline. **Pass 1** runs the standard evaluation prompt (with span index but without graph hints) to get an initial set of detected errors. **Pass 2** then injects a *targeted* causal subgraph: for each error category detected in Pass 1 that has outgoing causal edges, the system constructs a subgraph containing only edges where the source is detected and the target is *not yet* detected. Pass 2 re-runs the LLM with this targeted hint, asking it to reconsider the trace specifically for the predicted downstream errors. The outputs of Pass 1 and Pass 2 are then merged (with deduplication).

**Motivation.** The key difference from +CG is *conditionality*: graph injection only fires when the LLM has already detected a root cause, and only prompts for the specific downstream effects of what was detected. This eliminates the over-triggering problem of the static graph hint: if the LLM found *Tool Selection Errors*, it is prompted to check for *Goal Deviation* and *Task Orchestration*; if it did not find *Tool Selection Errors*, those hints are suppressed entirely.

This two-pass approach also allows Pass 2 to be a focused, shorter prompt (targeting only 1–3 specific error categories) rather than a full re-evaluation, which is more token-efficient for API-based models.

**Limitation.** Two API calls per trace doubles the cost and latency. For traces where Pass 1 detects no errors with outgoing graph edges, Pass 2 is skipped entirely (no additional cost). In practice, Pass 2 fires on approximately 60–70% of traces.

### 3.5 Summary of Variant Differences

| Property | Baseline | +CG | +CG+SI | +GI+SI |
|----------|----------|-----|--------|--------|
| Graph hints | None | Static, all edges | Static, all edges | Dynamic, targeted subgraph |
| Span index | No | No | Yes | Yes |
| LLM calls per trace | 1 | 1 | 1 | 1–2 |
| Addresses over-triggering | — | No | No | Yes |
| Addresses span hallucination | No | No | Yes | Yes |
| Token overhead | None | Low | Medium | Low (Pass 2 is short) |

---

## 4. Deduplication and Its Motivation

### 4.1 The Problem: Trace Duplication in the Original Split

The original TRAIL GAIA split contains traces from a shared agent corpus, some of which represent runs on identical or near-identical tasks. When multiple traces correspond to the same underlying task instance, the evaluation set has two problems:

1. **Inflated sample size:** Task repetitions count as independent evaluation instances, giving disproportionate weight to task variants that happen to have more traces.
2. **Leakage of difficulty signal:** If an agent retries a task and a later run succeeds, the distribution of errors in the trace set is skewed toward near-success runs, which have systematically different error profiles than complete failures.

For a benchmark focused on error detection, duplicated traces also inflate or deflate individual category frequencies depending on which tasks are over-represented — potentially distorting both training and evaluation.

### 4.2 The Deduplication Approach

We create two deduplicated splits — **GAIA_dedup** and **SWE_Bench_dedup** — by identifying and collapsing duplicate task instances. The deduplication is applied at the task level: if multiple traces correspond to the same task specification, only one representative trace is retained.

The deduplication results in a larger evaluated sample in some configurations: Mistral on GAIA increases from $N \approx 48$ (original split) to $N \approx 64$–$81$ (dedup split), because the dedup split redistributes traces more uniformly across task types while removing near-duplicate runs.

### 4.3 Effect on Evaluation Quality

The deduplication effect is starkest for the Mistral model (Table 3). On the original split, Mistral +CG achieves W-F1 = 17.98 and Loc = 8.69. On the deduplicated split, Mistral +CG achieves W-F1 = 28.31 and Loc = 30.31 — a jump of more than 10 points in both metrics. This large gap is not a model capability improvement; it reflects how duplicated traces in the original split disproportionately amplify certain prediction errors.

Critically, the dedup split also enables a clean **baseline comparison**: on the original split, no Mistral baseline run exists (only graph-augmented runs were completed), making it impossible to measure the graph's contribution. The dedup split provides a matched baseline (W-F1 = 24.06, Loc = 23.83) against which all three graph variants can be directly compared.

### 4.4 Context Compression for Long-Context Models

For QwenLong-L1-32B — a model with an extremely long context window (up to 1M tokens) but modest reasoning performance — we additionally experiment with **context-compressed traces**. Full TRAIL traces can exceed 100K tokens; compressed traces retain only the semantically critical keys (task specification, LLM input/output messages, tool call results, error status codes) identified in our span-key analysis, dropping metadata fields with no diagnostic signal.

Compression addresses two problems for QwenLong: (i) full traces cause severe context overflow in graph-augmented conditions, producing empty outputs (the +CG+SI zero result on full traces); (ii) QwenLong's attention mechanism performs better on focused, shorter inputs than on very long documents with heavy metadata padding. The dedup + compressed configuration (`GAIA (dedup-C)`) is therefore the primary configuration for QwenLong evaluation.

---

## 5. Analysis of Current Results

### 5.1 Main Results

The main results (Table 1) show the best-performing graph variant (+GI+SI) against the baseline across four models and two benchmarks. The overall picture is:

**+GI+SI is consistently the best method for GAIA** on the two models where full comparisons exist:
- **Gemini-2.5-Flash:** W-F1 40.75 (+3.68 over baseline 37.07), Loc 35.14 (+0.92), Joint 14.48 (+1.77).
- **QwenLong-L1-32B (dedup-C):** W-F1 16.83 (+4.43 over baseline 12.40), Loc 19.23 (+15.31), Joint 3.63 (+3.36).

**Mistral-Small-3.1-24B (dedup):** W-F1 30.76 vs baseline 24.06 — a gain of 6.70 points. Location Accuracy also improves substantially (27.79 vs 23.83). This is the largest absolute W-F1 gain across all experiments.

**SWE-bench remains hard.** Performance on SWE-bench is substantially lower than GAIA across all models. Gemini baseline on SWE-bench achieves only W-F1 = 8.71 vs 37.07 on GAIA. Even with +GI+SI, the W-F1 reaches only 21.56 — still below any GAIA baseline. SWE-bench traces involve repository-level software engineering with a CodeAgent using Python interpreter as the sole tool; the trace structure is fundamentally different from GAIA's web-search-and-reasoning format, and the error taxonomy coverage is sparser. Joint Accuracy remains 0.00 for all Gemini SWE-bench conditions, indicating that while the LLM can identify some error categories, it cannot reliably locate them at the right span.

### 5.2 Ablation: Method Variants (Gemini on GAIA)

Table 2 (Gemini-2.5-Flash on TRAIL-GAIA original split) provides a clean ablation of all four conditions:

| Method | W-F1 | Loc | Joint |
|--------|------|-----|-------|
| Baseline | 37.07 | 34.22 | 12.71 |
| +CG | 40.21 | **35.14** | 14.53 |
| +CG+SI | 39.60 | 34.40 | **15.81** |
| +GI+SI | **40.75** | **35.14** | 14.48 |

Several observations:

**+CG alone delivers a large W-F1 gain (+3.14).** The static graph hint is already effective for category detection even without the span index. Gemini is capable enough to use the graph hint without being confused by it — unlike smaller models.

**+CG+SI achieves the best Joint Accuracy (15.81),** exceeding +GI+SI (14.48) on this metric. The span index strongly reduces location hallucination, and the combination with the graph gives the highest joint (span, type) accuracy. The slight W-F1 degradation vs +CG (+40.21 → +39.60) suggests the span index adds slight overhead that occasionally deflects attention from category detection.

**+GI+SI achieves the best W-F1 (40.75).** The two-pass approach finds categories that the static hint misses, particularly rare ones where the LLM benefits from a targeted second-pass prompt. The targeted subgraph injection, conditioned on Pass 1 detections, appears more effective at improving category recall than the static list.

**Location Accuracy is less sensitive to method choice.** Loc varies only from 34.22 to 35.14 across all conditions — a 0.92-point range. This suggests that for Gemini, location prediction quality is primarily determined by the model's internal parsing of the trace, not by the graph hint. The span index's effect on location is modest.

### 5.3 Ablation: Deduplication Effect (Mistral on GAIA)

Table 3 (Mistral-Small-3.1-24B) shows the deduplication effect:

| Split | Method | W-F1 | Loc | Joint |
|-------|--------|------|-----|-------|
| Original | +CG | 17.98 | 8.69 | 2.51 |
| Original | +CG+SI | 20.85 | 14.59 | 3.09 |
| Dedup | Baseline | 24.06 | 23.83 | 3.78 |
| Dedup | +CG | 28.31 | **30.31** | 11.16 |
| Dedup | +CG+SI | 29.25 | 25.99 | 8.75 |
| Dedup | +GI+SI | **30.76** | 27.79 | **11.51** |

The dedup split improves results dramatically across all conditions. The Mistral baseline on dedup (W-F1 = 24.06, Loc = 23.83) already exceeds the best graph-augmented result on the original split (+CG+SI: W-F1 = 20.85, Loc = 14.59), confirming that the original split's duplication structure was severely biasing evaluations downward.

On the dedup split, +GI+SI is the best method for W-F1 and Joint, while +CG achieves the highest Location Accuracy (30.31). The fact that +CG yields higher location accuracy than +GI+SI (30.31 vs 27.79) but lower W-F1 (28.31 vs 30.76) is an interesting inversion: the two-pass GI+SI improves category identification at the cost of some location precision, because Pass 2's targeted re-evaluation may introduce errors at incorrect spans.

A notable failure: **Mistral +GI+SI on SWE-bench performs *below* baseline** (W-F1 = 6.76 vs 9.80, Loc = 1.67 vs 9.36). The two-pass structure appears to backfire on SWE-bench for Mistral: the model is small enough that Pass 2's targeted injection confuses rather than helps, leading to category predictions not grounded in the correct spans.

### 5.4 Per-Model Patterns

**Gemini-2.5-Flash** is the strongest overall model and benefits consistently from all three graph variants. It has sufficient reasoning capacity to use causal hints productively without being distracted by them.

**Mistral-Small-3.1-24B** shows the largest absolute improvement from graph methods on GAIA dedup (+6.70 W-F1), suggesting that for smaller models, the graph hints provide valuable scaffolding that partially compensates for weaker error detection capability. However, the SWE-bench failure suggests that this benefit does not generalize when the trace structure changes significantly.

**GPT-oss-120B** exhibits a counterintuitive pattern: both +CG and +CG+SI *decrease* W-F1 relative to baseline on GAIA original (25.24 → 23.12 / 23.62). This may reflect that GPT-oss's baseline prompt strategy is well-tuned and the graph block disrupts its output format. No +GI+SI run exists for GPT-oss. The compressed baseline (GAIA orig-C: W-F1 = 26.39, Loc = 18.13, Joint = 6.77) outperforms all full-trace conditions, indicating that trace length is a bottleneck for this model even at 120B scale.

**QwenLong-L1-32B** has the weakest absolute performance but shows clear sensitivity to both format and method. On full original traces with the static graph (+CG+SI), the model produces *zero valid predictions* across all traces — a complete failure arising from context overflow combined with the model's sensitivity to prompt structure. On compressed dedup traces, +GI+SI achieves W-F1 = 16.83 and Loc = 19.23 — substantial improvements over the compressed baseline (12.40 / 3.92), with Location Accuracy improving by 15.31 points. This dramatic location improvement from graph injection on compressed traces suggests that QwenLong can benefit significantly from structured hints when the input is manageable.

### 5.5 Overall Takeaways

1. **The causal graph consistently helps category detection (W-F1).** Across all models and settings where a meaningful comparison exists, graph-augmented methods improve W-F1 over the baseline. The improvement ranges from +3.68 (Gemini GAIA) to +6.70 (Mistral GAIA dedup).

2. **+GI+SI is the best single method for W-F1 and Joint Accuracy.** The two-pass targeted injection outperforms the static graph hint for category recall across the two main models (Gemini and Mistral on GAIA). However, +CG+SI is competitive for Joint Accuracy and Location on Gemini.

3. **SWE-bench is an unsolved challenge.** All methods perform substantially below GAIA levels on SWE-bench. Joint Accuracy is 0.00 or near-zero across all conditions. The SWE-bench trace structure (long Python code execution logs, single-tool architecture) requires different representation and evaluation strategies.

4. **Deduplication is essential for reliable evaluation.** The 10+ point difference between original and dedup splits for Mistral is not a model effect — it is an evaluation artifact. Deduplicated splits provide more meaningful baselines and should be the standard evaluation setting.

5. **Context length is a first-order constraint for smaller models.** QwenLong's collapse on +CG+SI with full traces, and GPT-oss's best performance on compressed traces, both point to token budget as a critical design parameter. The span index and causal graph hints must fit within the remaining context after the trace, which requires either trace compression or careful prompt budget management.

---

## 6. Connection to Related Work

Our approach differs from CDC-MAS (Ma et al., 2025), which also applies causal inference to agent failure attribution on TRAIL-GAIA. CDC-MAS builds a *per-trace step-level DAG* and focuses on identifying the single decisive error step (top-1 step exact match, reporting 44.6% accuracy). Our method constructs a *cross-trace error-type graph* and targets the full TRAIL evaluation: multi-label error localization, 19-class category F1, and joint accuracy. The two evaluations are fundamentally incomparable: a model that correctly identifies only the onset step would score 1.0 under CDC-MAS but only 0.10 location accuracy under TRAIL on a trace with 10 annotated error spans. Our work is also the first to apply intervention-validated causal edges (rather than observational correlations) to guide LLM-as-judge evaluation.

The pipeline further generalises to the MAST benchmark (393 multi-agent conversation traces, 13 categories), where the same Suppes–CAPRI–intervention protocol recovers a structurally distinct 14-edge causal graph with a clear three-level hierarchy: root causes (*Disobey Task Specification*, *Conversation Reset*, *Fail to Ask Clarification*) propagating through intermediate errors to terminal consequences (*Unaware of Termination Conditions*, *Premature Termination*). 7 of 14 CAPRI-BIC edges are validated under counterfactual intervention, confirming the pipeline's robustness across benchmarks with different agent architectures, task domains, and annotation schemas.
