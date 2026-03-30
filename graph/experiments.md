# Graph Model Experiments

## 1. Model Architecture

### Shared Input

| Component | Shape | Description |
|-----------|-------|-------------|
| Prototype features `x` | (20, 4096) | Mean-pooled Qwen3-Embedding-8B embeddings, one per error-type node (19 error + 1 Correct), built from training spans |
| Span embeddings `h_k` | (1, 4096) | Per-span Qwen3-Embedding-8B encoding, pre-computed and frozen |
| Adjacency `adj` | (20, 20) | Weighted directed graph over 19 error nodes; golden = Suppes correlation graph with validated causal overrides (varies by ablation condition) |

---

### Baseline: `NoGraphBaseline`

```
x  (20, 4096)
  └─ Linear(4096 → 256)  +  ELU  +  NodeDropout(0.3)
       └─ Z  (20, 256)          ← node embeddings, no graph mixing

h_k  (B, 4096)
  └─ Linear(4096 → 256)  +  ELU  +  SpanDropout(0.3)
       └─ h_k'  (B, 256)

score(h_k, z_i) = h_k' · M · z_i  +  cos(h_k', z_i) / τ
               = bilinear(B,256)×(256,256)×(256,20)  +  cosine/τ
               → (B, 20)

Loss (λ=0):  L = L_span = BCE_weighted(σ(scores), labels)
Loss (λ>0):  L = L_span + λ · L_graph
             L_graph = BCE(σ(Z[:19] · B · Z[:19]ᵀ), A_gold)
             B: (256, 256) separate graph_bilinear matrix
```

**Learnable parameters:**
- `proj`: Linear(4096, 256) — shared for nodes and spans — 1,048,832
- `bilinear` M: (256, 256) — span-to-node scorer — 65,536
- `graph_bilinear` B: (256, 256) — graph structure head (used only when λ>0) — 65,536
- `log_temp` τ: scalar — cosine temperature, init ln(0.07) — 1
- Total ≈ **1.18M parameters**

**Commands (from `trail-benchmark/`):**
```bash
# Train + evaluate (λ=0, best result)
python graph/baseline/run_baseline.py

# Train + evaluate with L_graph regulariser
python graph/baseline/run_baseline.py --lambda_graph 0.1

# Evaluate saved checkpoint only
python graph/baseline/run_baseline.py --eval_only
```

---

### Ablation Control: `GATModel` (self-loop, 1-layer)

Same architecture as GAT-1L but with `adj = I₂₀` (identity matrix). Each node attends
only to itself — the attention denominator has a single term, so `α_ii = 1.0` and all
off-diagonal weights are masked to −∞ before softmax. This isolates the effect of the
GATLayer *transformation* from the effect of cross-node *propagation*.

```
x  (20, 4096)
  └─ Linear(4096 → 256)  +  ELU  +  NodeDropout(0.3)
       └─ h  (20, 256)

  └─ DenseGATLayer(in=256, out=256, K=4 heads, concat=False)
       │  adj = I₂₀  (identity — only diagonal edges exist)
       │  W: Linear(256 → 4×256=1024)  →  Wh  (20, 4, 256)
       │  a: (4, 512) attention vector
       │  e_ij masked: only e_ii survives  →  α_ii = 1.0
       │  out = mean_heads(Wh_i)  (20, 256)   ← pure self-transform, no mixing
       └─ Z  (20, 256)  +  ELU

h_k scoring: same as Baseline
```

Activated via `--adj_type self_loop` in `05_train_gat.py`.
With `self_loop`, `A_gold = 0₁₉ₓ₁₉` so L_graph (when λ>0) pushes all pairwise
similarities to zero — a spurious orthogonality constraint with no real structure.

**Same parameter count as GAT-1L ≈ 1.45M**

**Commands (from `trail-benchmark/`):**
```bash
python graph/05_train_gat.py --n_layers 1 --adj_type self_loop --lambda_graph 0
python graph/05_train_gat.py --n_layers 1 --adj_type self_loop --lambda_graph 0.1
python graph/06_evaluate.py   # evaluate saved checkpoint
```

---

### Full Model: `GATModel` (1-layer variant)

```
x  (20, 4096)
  └─ Linear(4096 → 256)  +  ELU  +  NodeDropout(0.3)
       └─ h  (20, 256)

  └─ DenseGATLayer(in=256, out=256, K=4 heads, concat=False)
       │  W: Linear(256 → 4×256=1024)  →  Wh  (20, 4, 256)
       │  a: (4, 512) attention vector
       │  e_ij = LeakyReLU(a · [Wh_i ‖ Wh_j])  (20, 20, 4)
       │  + edge-weight bias; mask absent edges with -inf
       │  α = softmax(e, dim=1)  →  out = mean over heads  (20, 256)
       └─ Z  (20, 256)  +  ELU           ← refined node embeddings

h_k scoring: same as Baseline
```

**Commands (from `trail-benchmark/`):**
```bash
# Golden graph, λ=0.1 (best 1-layer GAT result)
python graph/05_train_gat.py --n_layers 1 --adj_type golden --lambda_graph 0.1

# Golden graph, λ=0
python graph/05_train_gat.py --n_layers 1 --adj_type golden --lambda_graph 0

# Random graph ablation
python graph/05_train_gat.py --n_layers 1 --adj_type random --lambda_graph 0
python graph/05_train_gat.py --n_layers 1 --adj_type random --lambda_graph 0.1

python graph/06_evaluate.py   # evaluate saved checkpoint
```

---

### Full Model: `GATModel` (2-layer variant, default)

```
x  (20, 4096)
  └─ Linear(4096 → 256)  +  ELU  +  NodeDropout(0.3)
       └─ h  (20, 256)

  └─ DenseGATLayer(in=256, out=256, K=4 heads, concat=True)
       └─ h  (20, 1024)    ← heads concatenated

  └─ DenseGATLayer(in=1024, out=256, K=4 heads, concat=False)
       └─ Z  (20, 256)  +  ELU

h_k scoring: same as Baseline
```

**Additional parameters vs. Baseline (per GAT layer):**
- `W`: Linear(in_dim, 4×256) — e.g. Linear(256, 1024) → 262,144
- `a`: (4, 512) attention vector → 2,048
- 1-layer GAT total ≈ **1.45M parameters**
- 2-layer GAT total ≈ **1.98M parameters**

**Commands (from `trail-benchmark/`):**
```bash
# Default: 2-layer, golden graph, λ=0.1
python graph/05_train_gat.py

# 2-layer, λ=0
python graph/05_train_gat.py --lambda_graph 0

python graph/06_evaluate.py   # evaluate saved checkpoint
```

---

### Training Hyperparameters (shared across all experiments)

| Hyperparameter | Value |
|----------------|-------|
| `hidden_dim` D | 256 |
| `n_heads` K | 4 |
| `dropout` | 0.3 |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Batch size | 32 spans |
| Max epochs | 100 |
| Early stopping patience | 10 (val Cat. F1) |
| Pos-weight | n_correct / n_error (~3.4×) to handle 77% correct-span imbalance |
| Threshold | Swept [0.05..0.50] on val, best used at test time |
| `lambda_graph` λ | 0.1 (original default) |
| Seed | 42 |

---

## 2. Ablation Results (Test Set)

All runs use the same seed, hyperparameters, and data splits.
`adj_type=golden` uses the Suppes correlation graph (directed edges screened by precedence fraction and probability raising, with 11 validated causal edges overridden to weight=1.0).
Precision and Recall are weighted averages. Hamming Acc = 1 − hamming_loss (per-label average). Subset Acc (exact match) = 0.000 for all configurations.

| # | Model | Layers | adj_type | λ | F1 (wtd) | F1 (mac) | F1 (mic) | Precision | Recall | Hamming Acc |
|---|-------|--------|----------|---|----------|----------|----------|-----------|--------|-------------|
| 1 | Baseline | — | none | 0 | **0.5621** | **0.4976** | 0.5109 | 0.4431 | **0.8545** | 0.6356 |
| 2 | GAT | 1 | self_loop | 0 | 0.5288 | 0.3852 | **0.5606** | 0.4733 | 0.6727 | **0.7652** |
| 3 | GAT | 1 | self_loop | 0.1 | 0.5112 | 0.3602 | 0.5390 | 0.4423 | 0.6909 | 0.7368 |
| 4 | GAT | 1 | golden | 0.1 | 0.4973 | 0.3839 | 0.4551 | 0.4380 | 0.6909 | 0.6316 |
| 5 | GAT | 1 | golden | 0 | 0.4558 | 0.3639 | 0.4462 | **0.4690** | 0.5273 | 0.7085 |
| 6 | GAT | 1 | random | 0.1 | 0.4179 | 0.3230 | 0.4138 | 0.3727 | 0.5455 | 0.6559 |
| 7 | GAT | 2 | golden | 0 | 0.4255 | 0.3085 | 0.4580 | 0.3767 | 0.5455 | 0.7126 |
| 8 | GAT | 2 | golden | 0.1 | 0.3383 | 0.2413 | 0.3689 | 0.4093 | 0.3455 | 0.7368 |
| 9 | GAT | 1 | random | 0 | 0.3468 | 0.2778 | 0.3421 | 0.3059 | 0.4727 | 0.5951 |
| 10 | Baseline | — | none | 0.1 | 0.5370 | 0.4678 | 0.4821 | 0.4152 | 0.8545 | 0.5911 |

Post-hoc Suppes graph inference experiments (additive propagation, causal predecessor
gating) are documented separately in `post_causal/experiments.md`.

All results above are **GAIA test set only** (13 traces). SWE-bench cross-dataset results are in Section 5.

---

## 3. What Each Ablation Revealed

### A. Baseline vs. GAT-1L self_loop (rows 1 → 2)
**Baseline 0.5621 → Self-loop 0.5288 (−0.033 weighted F1)**

The precision/recall breakdown reveals this is not simple degradation but a trade-off shift:

| | Precision | Recall |
|--|-----------|--------|
| Baseline | 0.4431 | 0.8545 |
| Self-loop | 0.4733 | 0.6727 |
| Δ | +0.030 | **−0.182** |

The GATLayer transformation makes the model more conservative: precision improves slightly
but recall collapses by 0.18. The large recall drop dominates, pulling weighted F1 down.
The baseline's high-recall strategy (predict broadly, accept lower precision) is better
suited to this class-imbalanced multi-label setting than the GATLayer's more selective
predictions. The simple `ELU(Proj(x))` preserves the recall advantage of the prototype
embeddings; the GATLayer suppresses it.

### B. Self-loop λ=0 vs. λ=0.1 (rows 2 → 3)
**Self-loop λ=0: 0.5288 → λ=0.1: 0.5112 (−0.018)**

With `self_loop`, `A_gold` is all-zeros. The L_graph loss becomes
`BCE(σ(Z·B·Zᵀ), 0)`, which pushes all pairwise cosine similarities
toward zero — forcing node embeddings to be approximately orthogonal.
This is a spurious structural constraint that hurts because there is
no meaningful graph to reconstruct.

### C. Self-loop vs. golden (rows 2 → 4, both λ=0.1)
**Self-loop 0.5112 → Golden 0.4973 (−0.014)**

Adding real Suppes edges (neighbor propagation) makes performance worse than
restricting each node to only its own embedding. The Suppes edges connect
error types that tend to co-occur and causally precede one another
(e.g. Tool Selection → Tool Output Misinterpretation), and GAT aggregation
mixes their prototype embeddings, reducing discriminability even though
the edges are semantically meaningful.

### D. Golden λ=0 vs. λ=0.1 (rows 5 → 4)
**Golden λ=0: 0.4558 → λ=0.1: 0.4973 (+0.042)**

L_graph with the real Suppes structure partially recovers from the propagation damage.
The `graph_bilinear` matrix B is separate from the scorer M, so the gradient from
L_graph shapes the embedding space without a direct blending operation. However, this
is *partial recovery* from GAT damage, not a genuine improvement: row 10 shows that
the same L_graph hurts the undamaged baseline (0.5621 → 0.5370). The apparent gain
here reflects how much GAT degraded the embeddings, not how beneficial L_graph is.

### E. 1-layer vs. 2-layer GAT (rows 4 → 8, both golden λ=0.1)
**1-layer 0.4973 → 2-layer 0.3383 (−0.159 weighted F1)**

The recall collapse is the primary symptom:

| | Precision | Recall |
|--|-----------|--------|
| GAT-1L golden λ=0.1 | 0.4380 | 0.6909 |
| GAT-2L golden λ=0.1 | 0.4093 | **0.3455** |
| Δ | −0.029 | **−0.345** |

Recall drops by 0.35 — the model becomes extremely conservative after two rounds of
neighborhood aggregation. The Suppes graph over-smoothing causes error-type embeddings
to converge, making the scorer uncertain about most error types and predicting very few
per trace. This is consistent with the over-smoothing interpretation: blended embeddings
lose the discriminative signal needed to confidently predict minority error types.

### F. Golden vs. random (rows 4 → 6, both 1-layer λ=0.1)
**Golden 0.4973 → Random 0.4179 (−0.079)**

Replacing the Suppes graph with a density-matched random directed graph
drops performance significantly, confirming that the golden graph's
structure carries real signal. The Suppes edges are not merely regularization
noise — they encode genuine precedence and probability-raising relationships
between error types. However, this signal is not enough to overcome the
over-smoothing damage from message passing.

### G. Random λ=0 vs. λ=0.1 (rows 9 → 6)
**Random λ=0: 0.3468 → λ=0.1: 0.4179 (+0.071)**

Even with random A_gold, L_graph improves over the random-λ=0 baseline — the largest
relative λ gain in the table. However, row 10 shows L_graph also hurts the undamaged
baseline (−0.025). The positive effect here is again *partial recovery*: random
propagation severely degrades the embeddings (0.3468), and L_graph — even with noisy
random A_gold — provides a consistent structural signal that stabilizes training and
partially counteracts the propagation noise. The takeaway is not that L_graph is
helpful in isolation, but that any consistent structural signal is better than
unguided random propagation.

### H. Baseline λ=0 vs. λ=0.1 (rows 1 → 10)
**Baseline λ=0: 0.5621 → λ=0.1: 0.5370 (−0.025 weighted F1)**

The new metrics reveal exactly where L_graph hurts:

| | Precision | Recall |
|--|-----------|--------|
| Baseline λ=0 | 0.4431 | **0.8545** |
| Baseline λ=0.1 | 0.4152 | **0.8545** |
| Δ | −0.028 | 0.000 |

Recall is identical (0.8545) — L_graph does not change which error types get detected.
It specifically reduces precision by −0.028: the structural constraint pulls connected
prototype embeddings closer together, making the scorer fire more broadly and producing
more false positives. The backward pass through `Z[:19]·B·Z[:19]ᵀ` modifies Z to make
connected pairs more similar, softening the inter-class margin and lowering the score
threshold at which spurious predictions appear.

**Conclusion on L_graph:** It never helps in an absolute sense. All positive λ effects
(rows D, G) were partial recovery from propagation damage. The mechanism of harm is
now precise: L_graph reduces precision without affecting recall.

---

## 4. Root Cause

Every mechanism that injects the Suppes graph into the *embedding space* — message
passing (GAT) or structural loss (L_graph) — degrades performance relative to the
plain baseline. The root cause has two parts:

**0. No configuration achieves exact label-set prediction.**
Subset accuracy (exact match across all 19 error types) = 0.000 for every configuration.
Multi-label prediction at the trace level is fundamentally hard with this dataset size —
the best any model does is partial overlap with the ground truth label set.

**1. Prototype embeddings are already near-optimal for recall.**
The prototype features `x` are mean-pooled Qwen3-8B embeddings over training spans.
The bilinear + cosine scorer achieves recall=0.8545 with no graph — the highest across
all configurations. Every graph mechanism reduces recall, sometimes drastically
(2-layer GAT: 0.3455). The Qwen3-8B embeddings are well-separated enough that the
scorer confidently identifies most error types present; graph refinement only disrupts
this by compressing the embedding space.

**2. The Suppes graph is an output-space structure, not an input-space structure.**
The Suppes graph encodes which error types tend to co-occur and causally precede one
another across traces. This is a property of the *label distribution* (output space),
not of the token-level text representations (input/embedding space). Injecting it into
the embedding space — whether via message passing or L_graph — creates a mismatch:
it pulls together prototype embeddings for semantically related error types, reducing
the very inter-class margin that makes the scorer work.

**Why GCN would not help:**
GCN applies the same message-passing principle as GAT but with degree-normalized
averaging instead of attention weighting. It would have the same fundamental problem
and likely perform worse, since degree normalization blends embeddings more uniformly
than GAT's selective attention.

**Overall conclusion (training-time graph mechanisms):**
Every training-time use of the Suppes graph — message passing (GAT), structural loss
(L_graph), or both — either ties or degrades the plain baseline. The baseline
(0.5621 weighted F1) using only frozen Qwen3-8B prototype embeddings and a bilinear
scorer is the best result within this training-time design space.

Post-hoc inference-time approaches (additive Suppes propagation, CMLL-inspired causal
predecessor gating) are documented in `post_causal/experiments.md`. Neither improves
weighted F1, though the causal gate shows a notable macro F1 gain (+0.090) by
redistributing predictions more evenly across rare error types.

---

## 5. SWE-bench Cross-Dataset Evaluation

### Setup

All models are trained on **GAIA only** (unchanged). SWE-bench evaluation is **zero-shot cross-dataset transfer** — no SWE-bench data is seen during training.

**SWE-bench data** (`graph/splits_swe/`):

| Split | Traces | LLM/TOOL Spans | Annotated | Annotation Coverage |
|-------|--------|----------------|-----------|---------------------|
| train | 24     | 374            | 146 (39%) | —                   |
| val   | 2      | 32             | 12 (38%)  | —                   |
| test  | 5      | 78             | 32 (41%)  | 98.8% (253/256)     |

Two malformed annotations (`location = "Span ID not foun"`) unmatched in 2 traces; 3 errors total go unmatched.

**Separate data directory**: `graph/data_swe/`
- `span_dataset.jsonl` — SWE-bench spans only (from `graph/splits_swe/`)
- `span_embeddings_{train,val,test}.pt` — Qwen3-8B encodings of SWE spans
- `graph_input.pt` — **copied from `graph/data/`** (GAIA-trained prototypes; same graph structure used during training)
- `label_to_node_idx.json` — generated by `03_encode_spans.py` (same taxonomy, same values)

### Build SWE data directory

```bash
cd /data/wang/junh/githubs/trail-benchmark

# Step 1: build span dataset
python graph/02_build_span_dataset.py \
    --splits_dir graph/splits_swe/ \
    --out_file graph/data_swe/span_dataset.jsonl

# Step 2: encode spans (GPU, ~same time as GAIA)
python graph/03_encode_spans.py \
    --data_file graph/data_swe/span_dataset.jsonl \
    --out_dir graph/data_swe/

# Step 3: copy GAIA graph_input.pt (model was trained with GAIA prototypes)
cp graph/data/graph_input.pt graph/data_swe/graph_input.pt
```

### Evaluate existing models on SWE test

```bash
# No-graph baseline (GAIA-trained, SWE zero-shot)
python graph/baseline/run_baseline.py --eval_only --split_tag swe
# Output: graph/baseline/outputs_swe/eval_results_{val,test}.json

# GAT golden (GAIA-trained, SWE zero-shot)
python graph/06_evaluate.py --split_tag swe --model_path graph/models/best_model.pt
# Output: graph/outputs_swe/eval_results_test.json

# Post-causal gate
python graph/post_causal/run_causal_inference.py --split_tag swe
# Output: graph/post_causal/outputs_swe/eval_results_causal_gate_{val,test}.json
```

### Train ablation models (GAIA data) + evaluate on both datasets

Ablation models are trained on GAIA data (default `graph/data/`). `--split_tag` for `05_train_gat.py` only changes the model and output save directories, not the data directory.

```bash
# Train ablations
python graph/05_train_gat.py --adj_type self_loop --split_tag self_loop
python graph/05_train_gat.py --adj_type random    --split_tag random
# Checkpoints: graph/models_self_loop/best_model.pt, graph/models_random/best_model.pt

# Evaluate ablations on GAIA test
python graph/06_evaluate.py \
    --model_path graph/models_self_loop/best_model.pt \
    --out_dir graph/outputs_self_loop/
python graph/06_evaluate.py \
    --model_path graph/models_random/best_model.pt \
    --out_dir graph/outputs_random/

# Evaluate ablations on SWE test (zero-shot)
python graph/06_evaluate.py --split_tag swe \
    --model_path graph/models_self_loop/best_model.pt \
    --out_dir graph/outputs_swe_self_loop/
python graph/06_evaluate.py --split_tag swe \
    --model_path graph/models_random/best_model.pt \
    --out_dir graph/outputs_swe_random/
```

### Note on evaluation strategy

SWE-only training (24 train traces, 2 val traces) is too small — val threshold tuning is unstable and the model converges to all-zero predictions. All SWE results below use **GAIA-trained models evaluated zero-shot on SWE test** (5 traces, 78 spans).

### Results

| Model | adj_type | λ | GAIA F1 (wtd) | SWE F1 (wtd) | SWE Recall (wtd) | SWE Loc. Acc |
|-------|----------|---|---------------|--------------|------------------|--------------|
| Baseline | — | 0 | 0.5203 | **0.4268** | 0.5652 | 1.0000 |
| GAT-2L | golden | 0.1 | — | — | — | — |
| GAT-1L | self_loop | 0 | 0.5288 | — | — | — |
| GAT-1L | random | 0.1 | 0.4179 | — | — | — |
| Post-causal gate | golden | — | 0.5171 | — | — | — |

GAIA F1 (wtd) for baseline = 0.5203 (test set, 13 traces). Full GAIA ablation table is in Section 2.
