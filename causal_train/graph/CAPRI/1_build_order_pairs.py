"""
Build per-trace order pairs A≺B from onsets.

For each trace, this script generates all pairs (A, B) where failure A's
onset precedes failure B's onset (temporal ordering).

Output format (one record per trace):
{
  "trace_id": "...",
  "pairs": [["1.5", "3.1"], ["1.4", "3.1"], ...]
}

Usage:
    python scripts/5_build_order_pairs.py
"""

import json
import os
import argparse
from tqdm import tqdm
from itertools import combinations


def main():
    ap = argparse.ArgumentParser(description="Build order pairs from onsets")
    ap.add_argument("--in_path", default="data/derived/onsets.jsonl",
                    help="Input onsets path")
    ap.add_argument("--out_path", default="data/derived/order_pairs.jsonl",
                    help="Output order pairs path")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print(f"Building order pairs from {args.in_path}...")
    
    total_traces = 0
    total_pairs = 0
    
    with open(args.in_path, "r") as fin, open(args.out_path, "w") as fout:
        for line in tqdm(fin, desc="Order pairs"):
            r = json.loads(line)
            onset = r.get("onset", {})
            modes = sorted(onset.keys())
            
            # Generate all pairs where A precedes B
            pairs = []
            for a, b in combinations(modes, 2):
                ta, tb = onset[a], onset[b]
                if ta < tb:
                    pairs.append([a, b])
                elif tb < ta:
                    pairs.append([b, a])
                # Ties (ta == tb) are ignored
            
            fout.write(json.dumps({
                "trace_id": r["trace_id"],
                "pairs": pairs
            }, ensure_ascii=False) + "\n")
            
            total_traces += 1
            total_pairs += len(pairs)
    
    print(f"\n✓ Processed {total_traces} traces")
    print(f"✓ Generated {total_pairs} order pairs (avg: {total_pairs/total_traces:.1f} per trace)")
    print(f"✓ Saved to {args.out_path}")


if __name__ == "__main__":
    main()
