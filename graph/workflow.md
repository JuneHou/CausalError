# Graph-based Error Classification — Design Workflow

**Model**: GAT (Graph Attention Network) + bilinear link prediction
**Task**: Given a new span embedding, predict which of the 19 error-type nodes it connects to (multi-label, span-level); aggregate span predictions to trace-level for evaluation.
**Evaluation metric**: Category F1 (weighted F1 over 19 error types, excluding the "Correct" node).

---

## Data Summary

Currently trained and evaluated on **GAIA only** (SWE-bench excluded; uncomment in `01_make_splits.py` to include).

| Source | Traces | Spans |
|---|---|---|
| GAIA | 117 | 1,062 (9.1 avg/trace) |

**19 error type nodes** (from `graph/outputs/graph_data.json`):
```
Authentication Errors, Context Handling Failures, Environment Setup Errors,
Formatting Errors, Goal Deviation, Incorrect Problem Identification,
Instruction Non-compliance, Language-only, Poor Information Retrieval,
Resource Abuse, Resource Exhaustion, Resource Not Found, Service Errors,
Task Orchestration, Timeout Issues, Tool Definition Issues,
Tool Output Misinterpretation, Tool Selection Errors, Tool-related
```
Plus 1 auxiliary **"Correct" node** (index 19) — spans with no error annotation.

**Rare types** (≤4 traces, `RARE_THRESHOLD=4`): 1 trace forced to test, rest to train:
- `Service Errors`: 2 traces
- `Timeout Issues`: 2 traces
- `Tool Definition Issues`: 3 traces
- `Resource Exhaustion`: 2 traces (shared test slot with Resource Not Found)

**Anchor types** (5–8 traces, `ANCHOR_THRESHOLD=8`): at least 1 forced to train before random split.

**Rescue-test**: after random split, if any error type has 0 test traces, a trace is moved from val (preferred) or train → test to ensure all 19 types have ≥1 test trace.

---

## Graph

**File**: `graph/outputs/graph_data.json`
**Nodes**: 19 (all error types, sorted alphabetically)
**Edges**: 155 weighted directed edges from Suppes causality test
**Causal edges**: 11 validated pairs (weight = 1.0)
**Edge weight formula**: `s_AB = (0.4 · precedence_AB + 0.6 · ΔPR̃_AB) · n_AB / (n_AB + 5)`

The **"Correct" node (index 19)** is added at runtime as an isolated node — no edges to/from error nodes. It is purely a classification target for unannotated spans and is never included in trace-level evaluation.

The graph **topology is frozen** during training — `edge_index` and `edge_weight` never change. The GAT layer parameters (attention weight matrices, linear transforms) are fully trainable.

---

## Step 1 — Train/Val/Test Split (`01_make_splits.py`)

**Input**: 117 GAIA trace IDs + `processed_annotations_gaia/`
**Ratio**: ~80 / 10 / 10 at trace level (actual: 92 / 12 / 13)
**Stratification rule**:
1. Rare types (≤4 traces): 1 trace → test, rest → train.
2. Anchor types (5–8 traces): at least 1 trace anchored to train before random split.
3. Random split on remaining traces.
4. Rescue-test: if any type has 0 test traces, move one trace from val/train → test.

**Output**:
```
graph/splits/train_trace_ids.json
graph/splits/val_trace_ids.json
graph/splits/test_trace_ids.json
```

---

## Step 2 — Span Extraction and Annotation Mapping (`02_build_span_dataset.py`)

### Span extraction

The encoding unit is the **LLM/TOOL action span** — the same level as annotation `location` span_ids. These are identified by `openinference.span.kind`:

- **`LLM`** spans (e.g. `LiteLLMModel.__call__`): the LLM call at each agent step. Always have `input.value` AND `output.value`.
- **`TOOL`** spans (e.g. `PageDownTool`, `FinalAnswerTool`, `SearchInformationTool`): tool invocations. Always have `input.value`; `output.value` may be absent for some tools.

Container spans (**`AGENT`**: CodeAgent.run, ToolCallingAgent.run; **`CHAIN`**: Step N) are not encoded — they have no LLM I/O of their own.

**Span collection** (two rules):

- **Rule 1 — Step-N level (primary)**: Every `Step N` span has exactly one direct LLM child. That LLM span is the encoding unit for that step, correct or annotated. Individual (non-annotated) TOOL children of Step N are not encoding units.
- **Rule 2 — Non-Step-N annotated spans**: Annotated spans whose parent is not a Step N (e.g. direct LLM child of `CodeAgent.run`, or an annotated TOOL span) are added explicitly, plus all direct LLM siblings within the same parent (correct spans at that level).

Sort all collected spans by start_time.

**Label assignment**: annotation `location` span_id → direct dictionary lookup. Spans not referenced by any annotation → "Correct".

### Span text content

**LLM spans** — either `input.value` or `output.value` is sufficient (span is not skipped if one is absent):
```
[SPAN] {span_name}

[INPUT]
{span_attributes["input.value"]}

[OUTPUT]
{span_attributes["output.value"]}
```

**TOOL spans** — `input.value` required; `output.value` optional:
```
[SPAN] {span_name}
[TOOL] {tool.name} — {tool.description}

[INPUT]
{span_attributes["input.value"]}

[OUTPUT]
{span_attributes["output.value"]}    ← omitted if absent
```

Logs (`function.name`, `function.output`, `severity_text`) are included when present on the span:
```
[LOGS]
severity: {severity_text}
function: {function.name}
output:   {function.output}
```

Truncate final text to 8K tokens before encoding.

### Label assignment

For each step span:
- If the annotation file references this span's ID (via `location` field walking up to step ancestor): label = the `category` of the error. Multi-label: one span can have multiple error categories if multiple errors reference the same step.
- If no annotation references this span: label = **"Correct"** (node index 19).

### Output schema

```json
{
  "trace_id": "...",
  "split": "train|val|test",
  "spans": [
    {
      "span_id": "...",
      "step_index": 0,
      "text": "... LLM input + output ...",
      "labels": ["Formatting Errors"],   // empty list = Correct
      "is_correct": false
    }
  ]
}
```

**Output**: `graph/data/span_dataset.jsonl` (one line per trace)

**Empirical results (02_build_span_dataset.py, GAIA only)**:
- 1,062 spans across 117 traces, avg 9.1/trace
- 99.3% annotation coverage — 581/585 annotation instances matched; 2 invalid "Span ID not found" annotations are the only unmatched
- 0 spans skipped (either input.value or output.value is sufficient)
- 387 annotated spans (36.4%), 675 correct spans (63.6%)

---

## Step 3 — Encoding and Prototype Construction (`03_encode_spans.py`)

**Encoder**: `Qwen/Qwen3-Embedding-8B`
**Representation**: last token hidden state (recommended for Qwen embedding family)
**Output dim**: 4096

### Encoding

Encode every span text in train, val, and test using the **same frozen Qwen model**. No fine-tuning.

```python
embedding = qwen_encode(span["text"])  # shape: (4096,)
```

### Prototype construction (train only)

For each of the 20 nodes (19 error types + Correct), compute a prototype by **mean-pooling** over all train spans assigned to that label:

```python
prototype[i] = mean( embed(s) for s in train_spans if label[i] in s.labels )
```

**Critical**: prototypes are computed from **train spans only**. Val and test spans are encoded with the same Qwen model but are never used to compute or update prototypes. This prevents information leakage.

**Output**:
```
graph/data/span_embeddings_train.pt     # {trace_id: {span_id: tensor(4096)}}
graph/data/span_embeddings_val.pt
graph/data/span_embeddings_test.pt
graph/data/prototypes.pt                # tensor(20, 4096) — train only
```

---

## Step 4 — Graph Input Construction (`04_build_graph_input.py`)

**Source**: `graph/outputs/graph_data.json`

Load the frozen graph and add the "Correct" node:

```python
# Original 19 nodes from graph_data.json
edge_index   = tensor(graph_data["edge_index"])    # (2, 155)
edge_weight  = tensor(graph_data["edge_attr"])     # (155,)
edge_is_causal = tensor(graph_data["is_causal"])   # (155,) bool

# Node 19 = "Correct" — isolated, no edges added
n_nodes = 20
```

**Initial node features** = `prototypes.pt` (shape: `(20, 4096)`)

The "Correct" node's prototype is the mean embedding of all non-annotated train spans. Its prototype is treated identically to error node prototypes — it is a learned representation updated through the GAT, but it has no edges and therefore only receives a self-transform (no message aggregation from neighbors).

**Output**: graph object ready for PyTorch Geometric
```
graph/data/graph_input.pt   # Data(x, edge_index, edge_weight, edge_is_causal)
```

---

## Step 5 — GAT Model and Training (`05_train_gat.py`)

### Architecture

```
Input node features:  (20, 4096)
                         |
            Linear projection layer
                         |
                      (20, D)        D = 256 (hidden dim)
                         |
              GAT Layer 1 (K=4 heads)
                         |
              GAT Layer 2 (K=4 heads)
                         |
            Refined node embeddings Z  (20, D)
```

Edge weights from `edge_weight` are used as attention bias in GAT (weighted adjacency).
The "Correct" node (index 19) has no neighbors, so both GAT layers apply only a self-transform (linear projection, no aggregation). Its refined embedding `z_19` is still a valid learnable vector.

### Per-span prediction (link prediction)

Given a span embedding `h_k` (4096-dim, projected to D-dim), score against each node:

```
h_k'   = ELU(proj(h_k))                            # project span emb to D-dim
bilinear = h_k'^T · M · z_i                        # (D,D) learnable matrix M
cos      = cosine_sim(h_k', z_i) / τ               # τ = exp(log_temp), init 0.07 (learned)
score(h_k, z_i) = bilinear + cos
p_hat[k, i]     = sigmoid(score)   for i in 0..19
```

### Loss functions

**1. Span-level link prediction loss** (primary)
```
L_span = BCE(p_hat[k, 0:20], y[k, 0:20])
```
Multi-label BCE, one row per span in the batch.

**2. Graph structure reconstruction loss** (auxiliary)
```
A_hat[i,j] = sigmoid(z_i^T · M_graph · z_j)
L_graph    = BCE(A_hat, A_gold)
```
where `A_gold` is the binarized adjacency from the golden graph (19×19, excluding Correct node).

This auxiliary loss ensures the GAT-refined node embeddings Z remain consistent with the known causal topology — error nodes that co-occur causally should be closer in embedding space. Without it, node embeddings are shaped only by span-to-node link prediction and may drift from the graph structure.

**No trace-level loss**: a trace-level BCE loss (pooling span predictions to trace level) is not included. Justification: we have ground-truth span-level labels from the annotation files (`location` → `category`). The span-level BCE already encodes the complete supervision signal. Adding a trace-level loss would re-weight the same labels redundantly and can introduce conflicting gradients. A trace-level auxiliary loss is only justified under Multi-Instance Learning (MIL) when span-level labels are unavailable (Maron & Lozano-Pérez 1998; Carbonneau et al. 2018). That condition does not hold here.

**Total loss**:
```
L = L_span + λ_graph · L_graph
```
Default: `λ_graph = 0.1` (tune on val Cat. F1).

### Regularization
- Dropout on GAT layers: 0.3
- Weight decay: 1e-4
- Qwen embeddings are frozen (no backprop through encoder)

### Training details
- Optimizer: AdamW
- Batch size: 32 spans
- Early stopping on val Cat. F1 (patience = 10 epochs)

---

## Step 6 — Evaluation (`06_evaluate.py`)

### Inference

For each span in a test trace:
1. Encode span text → `h_k` (frozen Qwen)
2. Project → `h_k'` (D-dim)
3. Score against refined `Z` (20 nodes)
4. `p_hat[k, i] = sigmoid(h_k'^T · M · z_i)`
5. Threshold at τ_val (val-tuned): `pred_span[k] = {i : p_hat[k,i] > τ_val, i < 19}`  (exclude Correct node i=19)

### Trace-level aggregation (max-pool)
```
trace_max[t, i] = max( p_hat[k, i] for k in spans of trace t )    i in 0..18
pred_trace[t]   = {i : trace_max[t, i] > τ_val}
```
Max-pool is used rather than mean-pool because 63.6% of spans are "Correct" — mean-pool would dilute error signals below threshold even for strongly predicted error spans.

### Metric
**Category F1** = `sklearn.metrics.f1_score(y_true, y_pred, average='weighted')`
Over 19 error type labels (indices 0–18).
"Correct" node (index 19) is excluded from all evaluation.
Zero-support classes in val/test (e.g. Service Errors, Timeout Issues) contribute F1=0.0 with support=0 — reported as-is in per-class breakdown.

### Threshold tuning
The class imbalance (36.4% annotated spans) causes sigmoid scores to sit well below 0.5. The threshold is tuned on val by sweeping [0.10, 0.15, ..., 0.50] and picking the value that maximises val weighted F1. The best threshold is saved in the model checkpoint (`val_threshold`) and used at test time. The sweep starts at 0.10 (not 0.05) to avoid the degenerate "predict everything" outcome when scores are uniformly low.

### Output
```
graph/outputs/eval_results_test.json  — per-class F1, support, overall Cat. F1
graph/outputs/confusion_matrix_test.png
```

### Empirical results (05_train_gat.py + 06_evaluate.py, GAIA only)
Val-tuned threshold: 0.15

| Metric      | Test  |
|-------------|-------|
| Weighted F1 | 0.338 |
| Macro F1    | 0.241 |
| Micro F1    | 0.369 |

Best per-class F1: Language-only (0.667), Tool Output Misinterpretation (0.667), Formatting Errors (0.500), Context Handling Failures (0.500), Incorrect Problem Identification (0.500).
Zero F1: 9 classes — mostly rare types (1–2 test traces) where model scores stay below threshold.

**Improvement directions**: Focal loss for rare classes; larger training set; contrastive span-level training to widen the score gap between true/false error types.

---

## How to Run

All commands are run from the **`benchmarking/`** directory (repo root for this pipeline).

### Prerequisites

```bash
pip install torch torch-geometric transformers scikit-learn matplotlib
```

The Qwen3-Embedding-8B encoder (Step 3) requires a GPU with ~20 GB VRAM. Steps 1, 2, 4 are CPU-only. Steps 5–6 need a GPU.

### Step-by-step

**Step 1 — Generate train/val/test splits** (CPU, ~2 s)
```bash
python3 graph/01_make_splits.py
# Output: graph/splits/{train,val,test}_trace_ids.json
```
Re-run whenever annotation files change. Splits are deterministic (seed=42).

**Step 2 — Extract and annotate spans** (CPU, ~30 s)
```bash
python3 graph/02_build_span_dataset.py
# Output: graph/data/span_dataset.jsonl
#         graph/data/label_to_node_idx.json
```
Requires: `processed_annotations_gaia/`, TRAIL trace data (`data/GAIA/`).

**Step 3 — Encode spans with Qwen3-Embedding-8B** (GPU, ~30–60 min)
```bash
python3 graph/03_encode_spans.py --gpu 0
# Output: graph/data/span_embeddings_{train,val,test}.pt
#         graph/data/prototypes.pt
```
Prototypes are computed from train spans only. This step is slow; skip if `.pt` files already exist.

**Step 4 — Build graph input** (CPU, ~2 s)
```bash
python3 graph/04_build_graph_input.py
# Output: graph/data/graph_input.pt
```
Requires: `graph/outputs/graph_data.json` and `graph/data/prototypes.pt`.

**Step 5 — Train GAT model** (GPU, ~10–20 min)
```bash
python3 graph/05_train_gat.py --gpu 0
# Output: graph/models/best_model.pt
```
Key options:
```
--hidden_dim 256    # GAT hidden dimension (default: 256)
--n_heads    4      # GAT attention heads (default: 4)
--epochs     100    # max epochs (default: 100)
--patience   10     # early stopping patience (default: 10)
--lambda_graph 0.1  # graph reconstruction loss weight (default: 0.1)
--gpu        0      # GPU index
```
Best threshold (tuned on val weighted F1) is saved inside `best_model.pt`.

**Step 6 — Evaluate on test set** (GPU, ~1 min)
```bash
python3 graph/06_evaluate.py --gpu 0
# Output: graph/outputs/eval_results_test.json
#         graph/outputs/confusion_matrix_test.png
```
To evaluate on val instead:
```bash
python3 graph/06_evaluate.py --split val --gpu 0
```

### Full pipeline (fresh run)

```bash
cd benchmarking/
python3 graph/01_make_splits.py
python3 graph/02_build_span_dataset.py
python3 graph/03_encode_spans.py --gpu 0
python3 graph/04_build_graph_input.py
python3 graph/05_train_gat.py --gpu 0
python3 graph/06_evaluate.py --gpu 0
```

### Re-running after annotation changes

If annotation files are updated:
```bash
python3 graph/01_make_splits.py          # regenerate splits
python3 graph/02_build_span_dataset.py   # remap labels
python3 graph/03_encode_spans.py --gpu 0 # recompute prototypes
python3 graph/04_build_graph_input.py    # rebuild graph input
python3 graph/05_train_gat.py --gpu 0   # retrain
python3 graph/06_evaluate.py --gpu 0    # re-evaluate
```

If only the graph topology changes (`graph_data.json` updated):
```bash
python3 graph/04_build_graph_input.py   # rebuild graph input (prototypes unchanged)
python3 graph/05_train_gat.py --gpu 0  # retrain
python3 graph/06_evaluate.py --gpu 0   # re-evaluate
```

---

## File Layout

```
graph/
├── workflow.md                  ← this file
├── build_graph_data.py          ← existing: builds graph_data.json
├── outputs/
│   ├── graph_data.json          ← 19 nodes, 155 edges (frozen)
│   ├── node_list.json
│   ├── edge_list.csv
│   └── adjacency_matrix.*
├── splits/                      ← Step 1 output
│   ├── train_trace_ids.json
│   ├── val_trace_ids.json
│   └── test_trace_ids.json
├── data/                        ← Steps 2–4 output
│   ├── span_dataset.jsonl
│   ├── span_embeddings_train.pt
│   ├── span_embeddings_val.pt
│   ├── span_embeddings_test.pt
│   ├── prototypes.pt
│   └── graph_input.pt
├── 01_make_splits.py
├── 02_build_span_dataset.py
├── 03_encode_spans.py
├── 04_build_graph_input.py
├── 05_train_gat.py
└── 06_evaluate.py
```

---

## Open Question: Correct Node Isolation

The "Correct" node (index 19) is isolated in the graph — it has no edges to or from error nodes. After GAT:
- Error nodes 0–18 receive graph messages from neighbors based on golden graph structure.
- Correct node 19 passes through a **self-transform only** (no aggregation).

This means the Correct node's refined embedding `z_19` is effectively `W · prototype_correct + b` — a linear transform of the mean embedding of non-annotated spans. It does not benefit from graph context. This is intentional: "Correct" is orthogonal to error topology.

**Consequence**: the model must rely entirely on the span embedding similarity to `z_19` to predict "this span is correct." This is actually desirable — it means the Correct prediction is driven by span content alone, not by error graph structure.
