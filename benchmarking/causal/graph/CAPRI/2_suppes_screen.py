"""
Suppes probabilistic causation screen.

Single event definition: event = has onset (so "present but onset missing" is
treated as non-occurrence). PR and precedence both use this definition.
PR is computed on all traces (A=1 iff A in onset, A=0 iff A not in onset);
precedence is computed on traces where both A and B have onset (non-tie).
This avoids inconsistent edges when missing onsets are non-random.

An edge A→B is kept if:
- Among traces where both A and B have onset (non-tie), A precedes B in >= min_precedence fraction
- P(B=1|A=1) - P(B=1|A=0) >= min_pr_delta (using onset-based definition on all traces)
- At least min_joint traces have both A and B with onsets (for precedence denominator)

Usage:
    python causal_explore/CAPRI/2_suppes_screen.py
    python causal_explore/CAPRI/2_suppes_screen.py --min_precedence 0.7 --min_pr_delta 0.05
"""

import json
import os
import argparse
from collections import defaultdict


def safe_div(a, b):
    """Safe division returning 0 if denominator is 0."""
    return a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser(description="Suppes probabilistic causation screen")
    ap.add_argument("--in_path", default="data/derived/onsets.jsonl",
                    help="Input onsets path")
    ap.add_argument("--out_path", default="outputs/suppes_graph.json",
                    help="Output Suppes graph path")
    ap.add_argument("--min_precedence", type=float, default=0.55,
                    help="Minimum fraction of traces where A precedes B (default: 0.6)")
    ap.add_argument("--min_pr_delta", type=float, default=0.05,
                    help="Minimum probability raising delta (default: 0.02)")
    ap.add_argument("--min_joint", type=int, default=3,
                    help="Minimum traces where both A and B occur with onsets (default: 30)")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Load all traces
    print(f"Loading onsets from {args.in_path}...")
    rows = []
    all_modes = set()

    with open(args.in_path, "r") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            # All modes that ever appear in onset (single event definition)
            all_modes.update((r.get("onset") or {}).keys())

    all_modes = sorted(all_modes)
    print(f"Loaded {len(rows)} traces with {len(all_modes)} unique failure modes")
    print("Event definition: has onset (PR and precedence both use this; PR on all traces, precedence on joint-onset subset)")

    # PR: use onset (not present) on all traces — single event definition
    n_A1 = defaultdict(int)
    n_A0 = defaultdict(int)
    n_B1_A1 = defaultdict(int)
    n_B1_A0 = defaultdict(int)

    for r in rows:
        onset = r.get("onset") or {}
        for a in all_modes:
            a1 = 1 if a in onset else 0
            if a1:
                n_A1[a] += 1
            else:
                n_A0[a] += 1
        for a in all_modes:
            a1 = 1 if a in onset else 0
            for b in all_modes:
                if a == b:
                    continue
                b1 = 1 if b in onset else 0
                if a1:
                    if b1:
                        n_B1_A1[(a, b)] += 1
                else:
                    if b1:
                        n_B1_A0[(a, b)] += 1

    # Precedence: among traces where both have onset (non-tie)
    prec_num = defaultdict(int)
    prec_den = defaultdict(int)

    for r in rows:
        onset = r.get("onset") or {}
        modes_with_onset = list(onset.keys())
        for i in range(len(modes_with_onset)):
            for j in range(i + 1, len(modes_with_onset)):
                a = modes_with_onset[i]
                b = modes_with_onset[j]
                ta, tb = onset[a], onset[b]
                if ta == tb:
                    continue
                if ta < tb:
                    prec_num[(a, b)] += 1
                    prec_den[(a, b)] += 1
                    prec_den[(b, a)] += 1
                else:
                    prec_num[(b, a)] += 1
                    prec_den[(a, b)] += 1
                    prec_den[(b, a)] += 1

    # Compute edges that pass Suppes criteria
    print("\nScreening for Suppes edges...")
    edges = []
    
    for a in all_modes:
        for b in all_modes:
            if a == b:
                continue
            
            # Compute probability raising
            p_b_a1 = safe_div(n_B1_A1[(a, b)], n_A1[a])
            p_b_a0 = safe_div(n_B1_A0[(a, b)], n_A0[a])
            pr_delta = p_b_a1 - p_b_a0
            
            # Compute precedence
            den = prec_den[(a, b)]
            if den < args.min_joint:
                continue
            
            p_prec = safe_div(prec_num[(a, b)], den)
            
            # Apply Suppes criteria
            if p_prec >= args.min_precedence and pr_delta >= args.min_pr_delta:
                edges.append({
                    "a": a,
                    "b": b,
                    "precedence": round(p_prec, 4),
                    "precedence_n": den,
                    "p_b_given_a": round(p_b_a1, 4),
                    "p_b_given_not_a": round(p_b_a0, 4),
                    "pr_delta": round(pr_delta, 4)
                })

    # Sort edges by strength (probability raising first, then precedence)
    edges_sorted = sorted(edges, key=lambda x: (x["pr_delta"], x["precedence"]), reverse=True)

    # Create output
    output = {
        "params": {
            "min_precedence": args.min_precedence,
            "min_pr_delta": args.min_pr_delta,
            "min_joint": args.min_joint,
            "in_path": args.in_path
        },
        "n_traces": len(rows),
        "n_modes": len(all_modes),
        "n_edges": len(edges_sorted),
        "edges": edges_sorted
    }

    # Save output
    with open(args.out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUPPES SCREEN RESULTS")
    print(f"{'='*60}")
    print(f"Traces analyzed: {len(rows)}")
    print(f"Failure modes: {len(all_modes)}")
    print(f"Prima facie edges found: {len(edges_sorted)}")
    print(f"\nTop 10 edges by probability raising:")
    for i, e in enumerate(edges_sorted[:10], 1):
        print(f"  {i}. {e['a']} → {e['b']}: PR={e['pr_delta']:.3f}, Prec={e['precedence']:.2f} (n={e['precedence_n']})")
    print(f"\n✓ Saved to {args.out_path}")


if __name__ == "__main__":
    main()
