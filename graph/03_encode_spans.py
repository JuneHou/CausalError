#!/usr/bin/env python3
"""
03_encode_spans.py — Encode step-span texts with Qwen3-Embedding-8B.

For each span in span_dataset.jsonl:
  - Encode its text with the Qwen3-Embedding-8B model using last-token pooling
    (the recommended pooling strategy for the Qwen3 embedding family).
  - L2-normalise the resulting vector.

Prototype construction (train only):
  - For each of the 20 nodes (19 error types + "Correct"), compute the mean of
    all TRAIN span embeddings labelled with that node.
  - A span with multiple labels contributes to each label's prototype mean.
  - The "Correct" node prototype is the mean of all train spans with is_correct=True.
  - Prototypes are never computed from val or test spans.

Outputs:
  graph/data/span_embeddings_train.pt   — {trace_id: {span_id: tensor(D,)}}
  graph/data/span_embeddings_val.pt
  graph/data/span_embeddings_test.pt
  graph/data/prototypes.pt              — tensor(20, D), order = node_list.json + Correct
  graph/data/label_to_node_idx.json     — {"Authentication Errors": 0, ..., "Correct": 19}
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BENCH_DIR / "graph" / "data" / "span_dataset.jsonl"
NODE_LIST = BENCH_DIR / "graph" / "outputs" / "node_list.json"
OUTPUT_DIR = BENCH_DIR / "graph" / "data"

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
MAX_LENGTH = 8192   # tokens; Qwen3-Embedding supports up to 131072
BATCH_SIZE = 8      # spans per forward pass (doubled for 2-GPU default)
DEFAULT_GPUS = "0,1"
CORRECT_NODE = "Correct"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Last-token pooling (Qwen3 recommended approach)
# ---------------------------------------------------------------------------

def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the last non-padding token's hidden state for each item in the batch.
    Works for both left-padded and right-padded inputs.
    """
    # Check if left-padded (all last positions are non-padding)
    left_padded = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    # Right-padded: find index of last real token per row
    seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
    batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_idx, seq_lens]            # (B, D)


# ---------------------------------------------------------------------------
# Encode a list of texts in batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    primary_device: torch.device,
    max_length: int,
    batch_size: int,
    desc: str = "Encoding",
) -> torch.Tensor:
    """
    Encode texts → L2-normalised embeddings, shape (N, D).
    Processes in batches. Shows tqdm progress with ETA.

    With DataParallel the model scatters each batch across GPUs and gathers
    last_hidden_state back on primary_device before pooling.
    """
    all_embeddings = []
    batches = range(0, len(texts), batch_size)

    with tqdm(batches, desc=desc, unit="batch",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for start in pbar:
            batch_texts = texts[start: start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Move to primary device; DataParallel distributes from there
            encoded = {k: v.to(primary_device) for k, v in encoded.items()}

            outputs = model(**encoded)

            # last_hidden_state is gathered on primary_device by DataParallel
            pooled = last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            normed = F.normalize(pooled.float(), p=2, dim=1)   # (B, D) fp32
            all_embeddings.append(normed.cpu())

            n_done = min(start + batch_size, len(texts))
            pbar.set_postfix(spans=f"{n_done}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0)               # (N, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Encode span texts with Qwen3-Embedding-8B")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL})")
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH,
                    help=f"Max token length per span (default: {MAX_LENGTH})")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                    help=f"Total spans per forward pass across all GPUs (default: {BATCH_SIZE})")
    ap.add_argument("--gpus", default=DEFAULT_GPUS,
                    help=f"Comma-separated GPU ids for DataParallel (default: {DEFAULT_GPUS}). "
                         "Use '' or 'cpu' to run on CPU.")
    ap.add_argument("--data_file", default=None,
                    help="Path to span_dataset.jsonl (default: graph/data/span_dataset.jsonl). "
                         "Use to encode a different dataset, e.g. graph/data_swe/span_dataset.jsonl.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for embeddings and prototypes "
                         "(default: graph/data/). Must contain node_list.json or "
                         "use --node_list to specify it separately.")
    args = ap.parse_args()

    # Override paths if specified
    global DATA_FILE, OUTPUT_DIR
    if args.data_file:
        DATA_FILE = Path(args.data_file)
    if args.out_dir:
        OUTPUT_DIR = Path(args.out_dir)

    # Parse GPU list
    if args.gpus.strip().lower() in ("", "cpu") or not torch.cuda.is_available():
        gpu_ids = []
        primary_device = torch.device("cpu")
    else:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build label → node index mapping
    # ------------------------------------------------------------------
    node_data = json.loads(NODE_LIST.read_text())          # {"0": "Auth...", ...}
    # node_data values are sorted error-type names, indices 0-18
    node_list = [node_data[str(i)] for i in range(len(node_data))]
    node_list.append(CORRECT_NODE)                         # index 19
    label_to_idx = {name: i for i, name in enumerate(node_list)}

    label_map_path = OUTPUT_DIR / "label_to_node_idx.json"
    label_map_path.write_text(json.dumps(label_to_idx, indent=2))
    log.info("Node list (%d nodes): %s ... %s",
             len(node_list), node_list[:3], node_list[-1])

    # ------------------------------------------------------------------
    # Load span dataset
    # ------------------------------------------------------------------
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found — run 02_build_span_dataset.py first")

    # Organise spans by split
    split_spans: dict[str, list[dict]] = defaultdict(list)   # split → flat list of span dicts
    # Each entry: {trace_id, span_id, text, labels, is_correct}

    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid   = rec["trace_id"]
            split = rec["split"]
            for sp in rec["spans"]:
                split_spans[split].append({
                    "trace_id":   tid,
                    "span_id":    sp["span_id"],
                    "text":       sp["text"],
                    "labels":     sp["labels"],
                    "is_correct": sp["is_correct"],
                })

    for sn in ("train", "val", "test"):
        log.info("  %s: %d spans", sn, len(split_spans[sn]))

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log.info("Loading tokenizer and model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", trust_remote_code=True,
    )
    base_model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(primary_device).eval()

    # Wrap with DataParallel if multiple GPUs requested
    if len(gpu_ids) > 1:
        model = nn.DataParallel(base_model, device_ids=gpu_ids)
        log.info("Using DataParallel on GPUs: %s (primary: cuda:%d)",
                 gpu_ids, gpu_ids[0])
    else:
        model = base_model
        log.info("Using single device: %s", primary_device)

    embed_dim = base_model.config.hidden_size
    log.info("Model loaded — embedding dim: %d", embed_dim)

    # ------------------------------------------------------------------
    # Encode all splits; save per-split embedding dicts
    # ------------------------------------------------------------------
    # train embeddings are kept in memory to compute prototypes
    train_embeddings: list[torch.Tensor] = []

    for split_name in ("train", "val", "test"):
        spans = split_spans[split_name]
        if not spans:
            log.warning("No spans for split '%s' — skipping", split_name)
            continue

        log.info("Encoding %s (%d spans)...", split_name, len(spans))
        texts = [s["text"] for s in spans]
        embs = encode_texts(
            texts, tokenizer, model, primary_device,
            args.max_length, args.batch_size,
            desc=f"{split_name:5s}",
        )                                                  # (N, D)

        # Package as {trace_id: {span_id: tensor(D,)}}
        emb_dict: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for i, sp in enumerate(spans):
            emb_dict[sp["trace_id"]][sp["span_id"]] = embs[i]

        out_path = OUTPUT_DIR / f"span_embeddings_{split_name}.pt"
        torch.save(dict(emb_dict), out_path)
        log.info("  Saved → %s  (shape per embedding: %s)", out_path, tuple(embs[0].shape))

        if split_name == "train":
            # Keep flat list paired with span metadata for prototype computation
            train_embeddings = [(embs[i], spans[i]) for i in range(len(spans))]

    # ------------------------------------------------------------------
    # Compute prototypes from train spans only
    # ------------------------------------------------------------------
    log.info("Computing prototypes from %d train spans...", len(train_embeddings))

    # Accumulate embeddings per node index
    proto_accum: dict[int, list[torch.Tensor]] = defaultdict(list)

    for emb, sp in train_embeddings:
        if sp["is_correct"]:
            idx = label_to_idx[CORRECT_NODE]              # index 19
            proto_accum[idx].append(emb)
        else:
            for label in sp["labels"]:
                if label in label_to_idx:
                    proto_accum[label_to_idx[label]].append(emb)
                else:
                    log.warning("Unknown label %r — skipping for prototype", label)

    # Build prototype tensor (20, D); mean-pool per node
    n_nodes = len(node_list)
    prototypes = torch.zeros(n_nodes, embed_dim, dtype=torch.float32)

    log.info("Prototype statistics:")
    log.info("  %-45s  %6s  %s", "Node", "Count", "Norm")
    for idx, name in enumerate(node_list):
        vecs = proto_accum.get(idx, [])
        if not vecs:
            log.warning("  Node %2d (%s): 0 train spans — prototype is zero vector", idx, name)
            continue
        stacked = torch.stack(vecs, dim=0)                 # (K, D)
        mean    = stacked.mean(dim=0)                      # (D,)
        mean    = F.normalize(mean, p=2, dim=0)            # unit-norm
        prototypes[idx] = mean
        log.info("  %-45s  %6d  %.4f", name, len(vecs), mean.norm().item())

    proto_path = OUTPUT_DIR / "prototypes.pt"
    torch.save(prototypes, proto_path)
    log.info("Saved prototypes → %s  shape=%s", proto_path, tuple(prototypes.shape))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Encoding complete")
    print(f"{'='*60}")
    for split_name in ("train", "val", "test"):
        n = len(split_spans[split_name])
        print(f"  {split_name:<8}: {n} spans → span_embeddings_{split_name}.pt")
    print(f"  prototypes : tensor({n_nodes}, {embed_dim})  → prototypes.pt")
    print(f"  label map  : {label_map_path}")


if __name__ == "__main__":
    main()
