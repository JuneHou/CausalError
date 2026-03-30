#!/usr/bin/env python3
"""
03b_resplit_embeddings.py — Re-partition existing span embeddings to a new train/val/test split.

Avoids re-running the expensive Qwen3-Embedding-8B encoding step.
All span embeddings from the original graph/data/ are merged, then
re-assigned to new splits according to the provided splits directory.
Prototypes are recomputed from the new train set.

Usage (7:1:2 split):
    python graph/03b_resplit_embeddings.py \\
        --src_data_dir  graph/data/ \\
        --splits_dir    graph/splits_712/ \\
        --dataset_file  graph/data_712/span_dataset.jsonl \\
        --out_dir       graph/data_712/

Outputs (in --out_dir):
    span_embeddings_train.pt
    span_embeddings_val.pt
    span_embeddings_test.pt
    prototypes.pt
    label_to_node_idx.json        (copied from src, not recomputed)
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = Path(__file__).resolve().parent

CORRECT_NODE = "Correct"


def load_all_embeddings(src_data_dir: Path) -> dict[str, dict]:
    """Merge train/val/test embedding dicts into one {trace_id: {span_id: tensor}}."""
    merged: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = src_data_dir / f"span_embeddings_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        d = torch.load(path, weights_only=True)
        overlap = set(merged) & set(d)
        if overlap:
            log.warning("Duplicate trace IDs across embedding files: %s", overlap)
        merged.update(d)
    log.info("Merged embeddings: %d traces total", len(merged))
    return merged


def load_splits(splits_dir: Path) -> dict[str, str]:
    """Return {trace_id: split_name} from *_trace_ids.json files."""
    mapping: dict[str, str] = {}
    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}_trace_ids.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run 01_make_splits.py first")
        ids = json.loads(p.read_text())
        for tid in ids:
            mapping[tid] = name
    return mapping


def load_span_labels(dataset_file: Path) -> dict[str, dict[str, list[str]]]:
    """Return {trace_id: {span_id: [label, ...]}} from span_dataset.jsonl."""
    result: dict[str, dict[str, list[str]]] = {}
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["trace_id"]
            result[tid] = {}
            for span in rec["spans"]:
                result[tid][span["span_id"]] = span["labels"]
    return result


def compute_prototypes(
    train_embs: dict[str, dict],
    span_labels: dict[str, dict[str, list[str]]],
    label_to_idx: dict[str, int],
    n_nodes: int,
    feat_dim: int,
) -> torch.Tensor:
    """Compute mean prototype embeddings from train spans (train only)."""
    sums   = torch.zeros(n_nodes, feat_dim)
    counts = torch.zeros(n_nodes)

    correct_idx = label_to_idx[CORRECT_NODE]

    for trace_id, span_dict in train_embs.items():
        labels_for_trace = span_labels.get(trace_id, {})
        for span_id, emb in span_dict.items():
            labels = labels_for_trace.get(span_id, [])
            if not labels:
                # Correct span
                sums[correct_idx]   += emb
                counts[correct_idx] += 1
            else:
                for label in labels:
                    if label in label_to_idx:
                        idx = label_to_idx[label]
                        sums[idx]   += emb
                        counts[idx] += 1

    # Normalize
    prototypes = torch.zeros(n_nodes, feat_dim)
    for i in range(n_nodes):
        if counts[i] > 0:
            proto = sums[i] / counts[i]
            prototypes[i] = F.normalize(proto, p=2, dim=0)
        else:
            log.warning("Node %d has no train examples — zero prototype", i)

    return prototypes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-partition existing span embeddings to a new train/val/test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--src_data_dir", default=str(GRAPH_DIR / "data"),
        help="Source data directory containing span_embeddings_*.pt and label_to_node_idx.json",
    )
    ap.add_argument(
        "--splits_dir", required=True,
        help="Directory with new *_trace_ids.json files (from 01_make_splits.py --out_dir)",
    )
    ap.add_argument(
        "--dataset_file", default=None,
        help="span_dataset.jsonl to read span labels from (default: --out_dir/span_dataset.jsonl)",
    )
    ap.add_argument(
        "--out_dir", required=True,
        help="Output directory for new embedding files and prototypes",
    )
    args = ap.parse_args()

    src_data_dir = Path(args.src_data_dir)
    splits_dir   = Path(args.splits_dir)
    out_dir      = Path(args.out_dir)
    dataset_file = Path(args.dataset_file) if args.dataset_file else out_dir / "span_dataset.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load label map (same across splits — defined by node_list.json)
    # ------------------------------------------------------------------
    label_map_src = src_data_dir / "label_to_node_idx.json"
    if not label_map_src.exists():
        raise FileNotFoundError(f"label_to_node_idx.json not found in {src_data_dir}")
    label_to_idx: dict[str, int] = json.loads(label_map_src.read_text())
    n_nodes  = max(label_to_idx.values()) + 1
    feat_dim = None  # determined from embeddings

    # Copy label map to out_dir
    (out_dir / "label_to_node_idx.json").write_text(
        json.dumps(label_to_idx, indent=2)
    )
    log.info("label_to_node_idx.json → %s", out_dir)

    # ------------------------------------------------------------------
    # Merge all existing embeddings
    # ------------------------------------------------------------------
    log.info("Loading embeddings from %s ...", src_data_dir)
    all_embs = load_all_embeddings(src_data_dir)

    # Infer feat_dim from first embedding
    for span_dict in all_embs.values():
        for emb in span_dict.values():
            feat_dim = emb.shape[0]
            break
        if feat_dim is not None:
            break
    assert feat_dim is not None, "No embeddings found"
    log.info("Embedding dim: %d", feat_dim)

    # ------------------------------------------------------------------
    # Load new split assignment
    # ------------------------------------------------------------------
    log.info("Loading new splits from %s ...", splits_dir)
    split_map = load_splits(splits_dir)

    # Verify all traces have embeddings
    missing = [tid for tid in split_map if tid not in all_embs]
    if missing:
        log.warning("%d traces in new splits have no embeddings: %s", len(missing), missing[:5])

    # ------------------------------------------------------------------
    # Partition embeddings by split
    # ------------------------------------------------------------------
    split_embs: dict[str, dict[str, dict]] = {"train": {}, "val": {}, "test": {}}
    for trace_id, split_name in split_map.items():
        if trace_id in all_embs:
            split_embs[split_name][trace_id] = all_embs[trace_id]

    for split_name, d in split_embs.items():
        log.info("  %s: %d traces", split_name, len(d))
        out_path = out_dir / f"span_embeddings_{split_name}.pt"
        torch.save(d, out_path)
        log.info("Saved → %s", out_path)

    # ------------------------------------------------------------------
    # Recompute prototypes from new train set
    # ------------------------------------------------------------------
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"{dataset_file} not found — run 02_build_span_dataset.py --out_file {dataset_file} first"
        )
    log.info("Loading span labels from %s ...", dataset_file)
    span_labels = load_span_labels(dataset_file)

    log.info("Computing prototypes from %d train traces ...", len(split_embs["train"]))
    prototypes = compute_prototypes(
        split_embs["train"], span_labels, label_to_idx, n_nodes, feat_dim
    )

    proto_path = out_dir / "prototypes.pt"
    torch.save(prototypes, proto_path)
    log.info("Saved prototypes %s → %s", tuple(prototypes.shape), proto_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Resplit summary")
    print(f"{'='*60}")
    for sn in ("train", "val", "test"):
        n_traces = len(split_embs[sn])
        n_spans  = sum(len(v) for v in split_embs[sn].values())
        print(f"  {sn:<8}  {n_traces:>4} traces  {n_spans:>6} spans")
    print(f"  Prototypes: {tuple(prototypes.shape)}")
    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
