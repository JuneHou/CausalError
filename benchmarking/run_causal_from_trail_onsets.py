#!/usr/bin/env python3
"""Entry point for causal discovery (CAPRI pipeline). Implementation in causal/graph/."""
import os
import sys
_bench = os.path.dirname(os.path.abspath(__file__))
_graph = os.path.join(_bench, "causal", "graph")
if _graph not in sys.path:
    sys.path.insert(0, _graph)
import run_causal_from_trail_onsets as runner
if __name__ == "__main__":
    runner.main()
