"""
Task #60: Rebuild Suppes graph per benchmark from training-side onsets only.

G_tau used in detection = corr-union derived inline by the eval script from
suppes_graph.json (edges with sqrt(precedence * pr_delta) >= tau). So per
benchmark we need a fresh suppes_graph.json built from the training-only
onsets, plus an empty effect_edges.json so the eval script's --causal_graph
contributes no validated G_V edges (those would leak full-corpus info).

Usage:
    python build_fold_graphs.py
"""

import json
import subprocess
import sys
from pathlib import Path

from config import (
    BENCHMARKS, GRAPHS_DIR, MAST_SUPPES_SCRIPT, ONSETS_DIR,
    SUPPES_PARAMS, TRAIL_SUPPES_SCRIPT,
)


def build_suppes(in_path: Path, out_path: Path, suppes_script: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(suppes_script),
        "--in_path", str(in_path),
        "--out_path", str(out_path),
        "--min_precedence", str(SUPPES_PARAMS["min_precedence"]),
        "--min_pr_delta", str(SUPPES_PARAMS["min_pr_delta"]),
        "--min_joint", str(SUPPES_PARAMS["min_joint"]),
    ]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def write_empty_effect_edges(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"edges": []}, f)


def main():
    summary = []
    for name, cfg in BENCHMARKS.items():
        in_path = ONSETS_DIR / f"{name}.jsonl"
        if not in_path.exists():
            print(f"  ! missing: {in_path}  (run build_folds.py first)")
            continue

        out_dir = GRAPHS_DIR / name
        suppes_out = out_dir / "suppes_graph.json"
        suppes_script = (
            TRAIL_SUPPES_SCRIPT if cfg["source"] == "trail" else MAST_SUPPES_SCRIPT
        )
        if not build_suppes(in_path, suppes_out, suppes_script):
            print(f"  ! suppes build failed for {name}")
            continue

        write_empty_effect_edges(out_dir / "effect_edges.json")

        with open(suppes_out) as f:
            g = json.load(f)
        n_edges = g.get("n_edges", len(g.get("edges", [])))
        summary.append((name, n_edges))
        print(f"  -> {name}: {n_edges} Suppes edges\n")

    print("\nSummary (Suppes edges per held-out training set):")
    for name, n in summary:
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
