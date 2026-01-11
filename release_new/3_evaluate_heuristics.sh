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

# Compare against heuristic baselines
python eval_heuristics.py --configs longcp wide --cases AL HS

echo ""
echo "Heuristic evaluation completed!"
