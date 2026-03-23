#!/usr/bin/env bash
# Run causal graph construction for TRAIL GAIA. Delegates to causal/graph/.
# Execute from benchmarking/:  cd benchmarking && bash run_causal_gaia.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/causal/graph/run_causal_gaia.sh"
