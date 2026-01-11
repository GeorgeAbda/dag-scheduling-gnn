#!/usr/bin/env bash
# =============================================================================
# Evaluate Heuristic Baselines
# =============================================================================
# This script evaluates classical scheduling heuristics (Min-Min, Max-Min, HEFT, etc.)
# across all DAG topology cases for comparison with RL agents.
#
# Heuristics evaluated:
#   - Random
#   - Round Robin
#   - Min-Min (makespan-focused)
#   - Max-Min (makespan-focused)
#   - HEFT (Heterogeneous Earliest Finish Time)
#   - Energy-aware variants
#
# Host Specs Cases:
#   - AL: data/host_specs.json
#   - HP: data/host_specs_homoPower.json
#   - HS: data/host_specs_homospeed.json
#
# Usage:
#   ./bin/eval_heuristics.sh                    # Evaluate all cases
#   HOST_SPECS_PATH=data/host_specs_homoPower.json ./bin/eval_heuristics.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Output directory
OUT_DIR="${OUT_DIR:-logs/heuristic_eval}"

echo "=============================================="
echo "Evaluating Heuristic Baselines"
echo "=============================================="
echo "Output directory: $OUT_DIR"
echo "=============================================="

# Run multi-case heuristic evaluation
"$SCRIPT_DIR/../dist/eval_heuristics" \
    --out_dir "$OUT_DIR"

echo "=============================================="
echo "Heuristic evaluation complete!"
echo "Results saved to: $OUT_DIR"
echo "=============================================="
