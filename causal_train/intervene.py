#!/usr/bin/env python3
"""Entry point for intervention pipeline. Implementation in intervention/."""
import os
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_bench = os.path.normpath(os.path.join(_here, "..", "benchmarking"))
_intervention = os.path.join(_here, "intervention")
for _p in (_bench, _intervention):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from intervene import main
if __name__ == "__main__":
    raise SystemExit(main())
