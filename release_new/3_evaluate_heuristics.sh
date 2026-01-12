#!/bin/bash
# Evaluate Heuristics
# This script compares trained agents against heuristic baselines
#
# Usage:
#   bash 3_evaluate_heuristics.sh [--configs CONFIG1 CONFIG2 ...] [--cases CASE1 CASE2 ...]
#
# Options:
#   --configs    Configuration types to evaluate (default: longcp wide)
#   --cases      Test cases to run (default: AL HS)
#
# Examples:
#   bash 3_evaluate_heuristics.sh
#   bash 3_evaluate_heuristics.sh --configs longcp --cases AL
#   bash 3_evaluate_heuristics.sh --configs longcp wide --cases AL HS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

# Default values
CONFIGS="longcp wide"
CASES="AL HS"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --configs)
            shift
            CONFIGS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CONFIGS="$CONFIGS $1"
                shift
            done
            ;;
        --cases)
            shift
            CASES=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CASES="$CASES $1"
                shift
            done
            ;;
        --help)
            head -n 16 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Evaluating Heuristics"
echo "=========================================="
echo "Configs: $CONFIGS"
echo "Cases:   $CASES"
echo ""

# Compare against heuristic baselines using Cython-compiled module
PY_CMD='from cogito.tools.eval_heuristics import main; main()'
python -c "$PY_CMD" --configs $CONFIGS --cases $CASES

echo ""
echo "Heuristic evaluation completed!"
