"""
Shuffle control (negative control experiment).

This script performs a negative control by randomly permuting onset ranks
within each trace, then re-running the Suppes screen. If edges persist
after shuffling, it suggests artifacts rather than true causal structure.

Expected: edge count and strength should collapse after shuffling.

Usage:
    python scripts/9_shuffle_control.py
    python scripts/9_shuffle_control.py --n_shuffles 50
"""

import json
import os
import argparse
import random
from collections import Counter
from tqdm import tqdm
import subprocess
import sys


def shuffle_onsets(onset_dict, rng):
    """
    Randomly permute onset ranks within a trace using the provided RNG.
    (Do not reseed per trace — use one RNG per shuffle iteration so each
    trace gets a different permutation and the null is not weakened.)

    Args:
        onset_dict: Dictionary of {failure_id: onset_time}
        rng: random.Random instance (e.g. Random(seed + i) per iteration)

    Returns:
        Shuffled onset dictionary
    """
    failures = list(onset_dict.keys())
    times = list(onset_dict.values())
    rng.shuffle(times)
    return dict(zip(failures, times))


def main():
    ap = argparse.ArgumentParser(description="Shuffle control for negative control")
    ap.add_argument("--onsets_path", default="data/derived/onsets.jsonl",
                    help="Input onsets path")
    ap.add_argument("--suppes_path", default="outputs/suppes_graph.json",
                    help="Original Suppes graph (for parameters)")
    ap.add_argument("--out_path", default="outputs/controls_shuffle.json",
                    help="Output shuffle control results")
    ap.add_argument("--n_shuffles", type=int, default=50,
                    help="Number of shuffle iterations (default: 50)")
    ap.add_argument("--temp_dir", default="outputs/shuffle_temp",
                    help="Temporary directory")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    args = ap.parse_args()

    # Create directories
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Set random seed
    random.seed(args.seed)

    # Load original data
    print(f"Loading onsets from {args.onsets_path}...")
    onsets_data = []
    with open(args.onsets_path, "r") as f:
        for line in f:
            onsets_data.append(json.loads(line))
    print(f"Loaded {len(onsets_data)} traces")

    # Load Suppes parameters
    suppes_orig = json.load(open(args.suppes_path, "r"))
    suppes_params = suppes_orig["params"]
    
    print(f"Original Suppes graph has {suppes_orig['n_edges']} edges")

    # Shuffle and re-run
    print(f"\nRunning {args.n_shuffles} shuffle iterations...")
    shuffle_edge_counts = []
    shuffle_edges_all = Counter()
    
    for i in tqdm(range(args.n_shuffles), desc="Shuffling"):
        # One RNG per iteration so each trace gets a different permutation (no per-trace reseed)
        rng = random.Random(args.seed + i)
        shuffled_data = []
        for r in onsets_data:
            shuffled_onset = shuffle_onsets(r.get("onset", {}), rng=rng)
            shuffled_r = {**r, "onset": shuffled_onset}
            shuffled_data.append(shuffled_r)
        
        # Save shuffled data
        shuffled_path = os.path.join(args.temp_dir, f"shuffled_{i}.jsonl")
        with open(shuffled_path, "w") as f:
            for r in shuffled_data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Run Suppes screen (same-dir script in causal_explore/CAPRI/)
        suppes_out = os.path.join(args.temp_dir, f"suppes_shuffled_{i}.json")
        here = os.path.dirname(os.path.abspath(__file__))
        suppes_script = os.path.join(here, "2_suppes_screen.py")
        suppes_cmd = [
            sys.executable, suppes_script,
            "--in_path", shuffled_path,
            "--out_path", suppes_out,
            "--min_precedence", str(suppes_params["min_precedence"]),
            "--min_pr_delta", str(suppes_params["min_pr_delta"]),
            "--min_joint", str(suppes_params["min_joint"])
        ]
        subprocess.run(suppes_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load edges
        try:
            with open(suppes_out, "r") as f:
                result = json.load(f)
            n_edges = result["n_edges"]
            edges = [(e["a"], e["b"]) for e in result["edges"]]
        except Exception:
            n_edges = 0
            edges = []
        
        shuffle_edge_counts.append(n_edges)
        for edge in edges:
            shuffle_edges_all[edge] += 1
        
        # Cleanup temp files
        for path in [shuffled_path, suppes_out]:
            if os.path.exists(path):
                os.remove(path)

    # Calculate statistics
    mean_edges = sum(shuffle_edge_counts) / len(shuffle_edge_counts)
    max_edges = max(shuffle_edge_counts)
    min_edges = min(shuffle_edge_counts)

    # Most frequent edges in shuffled data
    most_common_shuffled = shuffle_edges_all.most_common(10)

    # Create output
    output = {
        "params": vars(args),
        "original_n_edges": suppes_orig["n_edges"],
        "shuffle_results": {
            "n_shuffles": args.n_shuffles,
            "mean_edges": round(mean_edges, 2),
            "max_edges": max_edges,
            "min_edges": min_edges,
            "edge_counts": shuffle_edge_counts,
        },
        "most_common_shuffled_edges": [
            {
                "a": a,
                "b": b,
                "count": count,
                "frequency": round(count / args.n_shuffles, 3)
            }
            for (a, b), count in most_common_shuffled
        ]
    }

    # Save output
    with open(args.out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Cleanup temp directory
    try:
        os.rmdir(args.temp_dir)
    except OSError:
        pass

    print(f"\n{'='*60}")
    print(f"SHUFFLE CONTROL RESULTS")
    print(f"{'='*60}")
    print(f"Original edges: {suppes_orig['n_edges']}")
    print(f"Shuffled (mean): {mean_edges:.1f}")
    print(f"Shuffled (range): [{min_edges}, {max_edges}]")
    print(f"Reduction: {100*(1 - mean_edges/max(1, suppes_orig['n_edges'])):.1f}%")
    
    if mean_edges < 0.2 * suppes_orig['n_edges']:
        print(f"\n✓ GOOD: Shuffling collapsed most edges → structure is not artifact")
    else:
        print(f"\n⚠ WARNING: Many edges persist after shuffling → check for artifacts")
    
    print(f"\n✓ Saved to {args.out_path}")


if __name__ == "__main__":
    main()
