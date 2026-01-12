#!/bin/bash
# Evaluate Trained Agents
# This script evaluates specialist agents across different host regimes
#
# Usage:
#   bash 4_evaluate_trained_agents.sh [--config CONFIG] [--regimes REGIME1 REGIME2 ...]
#
# Options:
#   --config PATH      Path to evaluation config YAML file (default: data/rl_configs/eval_agents_config.yaml)
#   --regimes          Regime labels to evaluate (default: all regimes from config)
#   --device           Device to use: cpu or cuda (default: cpu)
#   --repeats          Number of evaluation repeats per seed (default: from config)
#   --out_dir          Output directory (default: from config)
#
# Examples:
#   # Evaluate all regimes with default config
#   bash 4_evaluate_trained_agents.sh
#
#   # Evaluate specific regimes
#   bash 4_evaluate_trained_agents.sh --regimes AL HS
#
#   # Use custom config
#   bash 4_evaluate_trained_agents.sh --config my_eval_config.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

# Default values
CONFIG=""
REGIMES=""
DEVICE=""
REPEATS=""
OUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --regimes)
            shift
            REGIMES=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                REGIMES="$REGIMES $1"
                shift
            done
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --help)
            head -n 23 "$0" | grep "^#" | sed 's/^# \?//'
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
echo "Evaluating Trained Agents"
echo "=========================================="

# Build command
PY_CMD='from cogito.tools.eval_agents import main; main()'
CMD="python -c \"$PY_CMD\""

if [[ -n "$CONFIG" ]]; then
    CMD="$CMD --config $CONFIG"
fi

if [[ -n "$REGIMES" ]]; then
    CMD="$CMD --regimes $REGIMES"
fi

if [[ -n "$DEVICE" ]]; then
    CMD="$CMD --device $DEVICE"
fi

if [[ -n "$REPEATS" ]]; then
    CMD="$CMD --repeats $REPEATS"
fi

if [[ -n "$OUT_DIR" ]]; then
    CMD="$CMD --out_dir $OUT_DIR"
fi

echo "Running evaluation..."
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
