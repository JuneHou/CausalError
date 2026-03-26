"""
Bootstrap stability analysis for causal edges.

This script performs bootstrap resampling to assess the stability of edges
in the causal graph. Edges that appear frequently across bootstrap samples
are more reliable.

Usage:
    python scripts/8_bootstrap_stability.py
    python scripts/8_bootstrap_stability.py --n_bootstrap 500
"""

import json
import os
import argparse
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import subprocess
import sys


def bootstrap_resample(data, seed=None):
    """Resample data with replacement."""
    if seed is not None:
        random.seed(seed)
    n = len(data)
    indices = [random.randint(0, n-1) for _ in range(n)]
    return [data[i] for i in indices]


def run_suppes_and_prune_on_sample(sample_onsets, temp_dir, sample_id, 
                                    suppes_params, capri_params):
    """
    Run Suppes + CAPRI on a bootstrap sample.
    Returns list of edges.
    """
    # Save sample to temp file
    sample_path = os.path.join(temp_dir, f"sample_{sample_id}.jsonl")
    with open(sample_path, "w") as f:
        for r in sample_onsets:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Get absolute paths to sibling scripts (causal_explore/CAPRI/)
    here = os.path.dirname(os.path.abspath(__file__))
    suppes_script = os.path.join(here, "2_suppes_screen.py")
    capri_script = os.path.join(here, "3_capri_prune.py")
    
    # Run Suppes screen
    suppes_out = os.path.join(temp_dir, f"suppes_{sample_id}.json")
    suppes_cmd = [
        sys.executable, suppes_script,
        "--in_path", sample_path,
        "--out_path", suppes_out,
        "--min_precedence", str(suppes_params["min_precedence"]),
        "--min_pr_delta", str(suppes_params["min_pr_delta"]),
        "--min_joint", str(suppes_params["min_joint"]),
    ]
    subprocess.run(suppes_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run CAPRI pruning (hill-climb + BIC/AIC; max_parents optional)
    capri_out = os.path.join(temp_dir, f"capri_{sample_id}.json")
    capri_cmd = [
        sys.executable, capri_script,
        "--onsets_path", sample_path,
        "--suppes_path", suppes_out,
        "--out_path", capri_out,
        "--criterion", str(capri_params.get("criterion", "BIC")),
    ]
    if capri_params.get("max_parents") is not None:
        capri_cmd.extend(["--max_parents", str(capri_params["max_parents"])])
    subprocess.run(capri_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Load edges
    try:
        with open(capri_out, "r") as f:
            result = json.load(f)
        edges = [(e["a"], e["b"]) for e in result["edges"]]
    except Exception:
        edges = []
    
    # Cleanup temp files
    for path in [sample_path, suppes_out, capri_out]:
        if os.path.exists(path):
            os.remove(path)
    
    return edges


def main():
    ap = argparse.ArgumentParser(description="Bootstrap stability analysis")
    ap.add_argument("--onsets_path", default="data/derived/onsets.jsonl",
                    help="Input onsets path")
    ap.add_argument("--suppes_path", default="outputs/suppes_graph.json",
                    help="Original Suppes graph (for parameters)")
    ap.add_argument("--capri_path", default="outputs/capri_graph.json",
                    help="Original CAPRI graph (for parameters)")
    ap.add_argument("--out_path", default="outputs/edge_stability.csv",
                    help="Output edge stability CSV")
    ap.add_argument("--n_bootstrap", type=int, default=100,
                    help="Number of bootstrap samples (default: 100)")
    ap.add_argument("--temp_dir", default="outputs/bootstrap_temp",
                    help="Temporary directory for bootstrap files")
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

    # Load parameters from original runs
    suppes_orig = json.load(open(args.suppes_path, "r"))
    capri_orig = json.load(open(args.capri_path, "r"))
    
    suppes_params = suppes_orig["params"]
    capri_params = capri_orig["params"]

    # Bootstrap resampling
    print(f"\nRunning {args.n_bootstrap} bootstrap samples...")
    edge_counter = Counter()
    
    for i in tqdm(range(args.n_bootstrap), desc="Bootstrap"):
        # Resample with replacement
        sample = bootstrap_resample(onsets_data, seed=args.seed + i)
        
        # Run pipeline on sample
        edges = run_suppes_and_prune_on_sample(
            sample, args.temp_dir, i, suppes_params, capri_params
        )
        
        # Count edges
        for edge in edges:
            edge_counter[edge] += 1

    # Calculate stability (frequency)
    print(f"\nCalculating edge stability...")
    results = []
    for (a, b), count in edge_counter.items():
        freq = count / args.n_bootstrap
        results.append({
            "a": a,
            "b": b,
            "frequency": round(freq, 4),
            "count": count
        })
    
    # Sort by frequency
    results_sorted = sorted(results, key=lambda x: x["frequency"], reverse=True)

    # Save CSV
    with open(args.out_path, "w") as f:
        f.write("source,target,frequency,count\n")
        for r in results_sorted:
            f.write(f"{r['a']},{r['b']},{r['frequency']},{r['count']}\n")

    # Save JSON version
    json_path = args.out_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "params": vars(args),
            "n_bootstrap": args.n_bootstrap,
            "n_edges": len(results_sorted),
            "edges": results_sorted
        }, f, indent=2)

    # Cleanup temp directory
    try:
        os.rmdir(args.temp_dir)
    except OSError:
        pass

    print(f"\n{'='*60}")
    print(f"BOOTSTRAP STABILITY RESULTS")
    print(f"{'='*60}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Unique edges found: {len(results_sorted)}")
    print(f"\nMost stable edges (frequency > 0.5):")
    stable = [r for r in results_sorted if r["frequency"] > 0.5]
    for i, r in enumerate(stable[:10], 1):
        print(f"  {i}. {r['a']} → {r['b']}: {r['frequency']:.2f} ({r['count']}/{args.n_bootstrap})")
    print(f"\n✓ Saved to {args.out_path}")
    print(f"✓ JSON version: {json_path}")


if __name__ == "__main__":
    main()
