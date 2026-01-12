#!/bin/bash
# Evaluate Heuristics
# This script compares trained agents against heuristic baselines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "=========================================="
echo "Evaluating Heuristics"
echo "=========================================="

# Compare against heuristic baselines using Cython-compiled module
PY_CMD='from cogito.tools.eval_heuristics import main, Args; import tyro; main(tyro.cli(Args))'
python -c "$PY_CMD" --configs longcp wide --cases AL HS

echo ""
echo "Heuristic evaluation completed!"
