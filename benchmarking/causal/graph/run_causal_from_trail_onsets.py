#!/usr/bin/env python3
"""
Run causal discovery steps (order pairs, Suppes, CAPRI, bootstrap, shuffle, hierarchy) on TRAIL onset data.

Uses TRAIL onset jsonl (present/onset per trace) and calls scripts in causal_explore/CAPRI/
(1_build_order_pairs → 2_suppes_screen → 3_capri_prune → 4_bootstrap_stability → 5_shuffle_control → 6_export_hierarchy).

Run from benchmarking/:
  python causal/graph/run_causal_from_trail_onsets.py --onsets_path data/trail_derived/onsets_gaia.jsonl
  python causal/graph/run_causal_from_trail_onsets.py --onsets_path data/trail_derived/onsets_gaia.jsonl --skip_bootstrap --skip_shuffle
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run causal steps (CAPRI pipeline) on TRAIL onsets")
    ap.add_argument("--onsets_path", default="data/trail_derived/onsets_gaia.jsonl",
                    help="TRAIL onset jsonl (present, onset per trace)")
    ap.add_argument("--out_dir", default="data/trail_causal_outputs",
                    help="Output directory for Suppes/CAPRI/bootstrap/hierarchy (default: data/trail_causal_outputs)")
    ap.add_argument("--min_precedence", type=float, default=0.55,
                    help="Suppes: min fraction A precedes B (default 0.55)")
    ap.add_argument("--min_pr_delta", type=float, default=0.05,
                    help="Suppes: min probability raising delta (default 0.05)")
    ap.add_argument("--min_joint", type=int, default=3,
                    help="Suppes: min traces with both A and B (default 3; use 5–10 for sparse categories)")
    ap.add_argument("--max_parents", type=int, default=None, help="CAPRI: optional cap on parents per node (default: none; BIC penalizes complexity)")
    ap.add_argument("--criterion", choices=["BIC", "AIC"], default="BIC", help="CAPRI: BIC or AIC score (default: BIC)")
    ap.add_argument("--n_bootstrap", type=int, default=100, help="Bootstrap samples (default 100)")
    ap.add_argument("--n_shuffles", type=int, default=50, help="Shuffle control permutations (default 50)")
    ap.add_argument("--skip_bootstrap", action="store_true", help="Skip bootstrap (faster)")
    ap.add_argument("--skip_shuffle", action="store_true", help="Skip shuffle control")
    args = ap.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.dirname(os.path.dirname(this_dir))  # benchmarking/
    capri_dir = os.path.join(this_dir, "CAPRI")
    if not os.path.isdir(capri_dir):
        print(f"CAPRI dir not found: {capri_dir}")
        sys.exit(1)

    onsets_abs = os.path.abspath(args.onsets_path)
    if not os.path.isfile(onsets_abs):
        print(f"Onsets file not found: {onsets_abs}")
        sys.exit(1)

    out_abs = os.path.abspath(args.out_dir)
    os.makedirs(out_abs, exist_ok=True)

    # Run scripts from causal/graph/CAPRI/; cwd=benchmarking so data/ paths resolve
    def run(script_name, cli_args):
        script = os.path.join(capri_dir, script_name)
        if not os.path.isfile(script):
            print(f"Script not found: {script}")
            return False
        cmd = [sys.executable, script] + cli_args
        print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
        r = subprocess.run(cmd, cwd=bench_dir)
        return r.returncode == 0

    # Step 1: Order pairs (optional)
    run("1_build_order_pairs.py", [
        "--in_path", onsets_abs,
        "--out_path", os.path.join(out_abs, "order_pairs.jsonl"),
    ])

    # Step 2: Suppes screen
    if not run("2_suppes_screen.py", [
        "--in_path", onsets_abs,
        "--out_path", os.path.join(out_abs, "suppes_graph.json"),
        "--min_precedence", str(args.min_precedence),
        "--min_pr_delta", str(args.min_pr_delta),
        "--min_joint", str(args.min_joint),
    ]):
        sys.exit(1)

    # Step 3: CAPRI prune
    capri_args = [
        "--onsets_path", onsets_abs,
        "--suppes_path", os.path.join(out_abs, "suppes_graph.json"),
        "--out_path", os.path.join(out_abs, "capri_graph.json"),
        "--criterion", args.criterion,
    ]
    if args.max_parents is not None:
        capri_args.extend(["--max_parents", str(args.max_parents)])
    if not run("3_capri_prune.py", capri_args):
        sys.exit(1)

    # Step 4: Bootstrap
    if not args.skip_bootstrap:
        run("4_bootstrap_stability.py", [
            "--onsets_path", onsets_abs,
            "--suppes_path", os.path.join(out_abs, "suppes_graph.json"),
            "--capri_path", os.path.join(out_abs, "capri_graph.json"),
            "--out_path", os.path.join(out_abs, "edge_stability.csv"),
            "--n_bootstrap", str(args.n_bootstrap),
        ])

    # Step 5: Shuffle control
    if not args.skip_shuffle:
        run("5_shuffle_control.py", [
            "--onsets_path", onsets_abs,
            "--suppes_path", os.path.join(out_abs, "suppes_graph.json"),
            "--out_path", os.path.join(out_abs, "controls_shuffle.json"),
            "--n_shuffles", str(args.n_shuffles),
        ])

    # Step 6: Export hierarchy (stability_path optional)
    stability_json = os.path.join(out_abs, "edge_stability.json")
    hierarchy_args = [
        "--capri_path", os.path.join(out_abs, "capri_graph.json"),
        "--out_path", os.path.join(out_abs, "hierarchy_levels.json"),
    ]
    if os.path.isfile(stability_json):
        hierarchy_args.insert(-2, "--stability_path")
        hierarchy_args.insert(-2, stability_json)
    run("6_export_hierarchy.py", hierarchy_args)

    print("\n" + "="*60)
    print("TRAIL causal discovery (CAPRI steps 1–6) complete")
    print("="*60)
    print(f"Outputs in: {out_abs}")
    print("  suppes_graph.json, capri_graph.json, hierarchy_levels.json")
    if not args.skip_bootstrap:
        print("  edge_stability.csv / edge_stability.json")
    if not args.skip_shuffle:
        print("  controls_shuffle.json")


if __name__ == "__main__":
    main()
