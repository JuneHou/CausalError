#!/usr/bin/env python3
"""Entry point for intervention pipeline. Implementation in causal/intervention/."""
import os
import sys
_bench = os.path.dirname(os.path.abspath(__file__))
_intervention = os.path.join(_bench, "causal", "intervention")
for _p in (_bench, _intervention):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from causal.intervention.intervene import main
if __name__ == "__main__":
    raise SystemExit(main())
