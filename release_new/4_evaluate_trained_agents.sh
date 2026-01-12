#!/bin/bash
# Evaluate Trained Agents
# This script evaluates specialist agents (LongCP and Wide) on specified host configurations
#
# Usage:
#   bash 4_evaluate_trained_agents.sh [OPTIONS]
#
# Options:
#   --host-specs PATH          Path to host specifications JSON file (required)
#   --longcp-checkpoint PATH   Path to LongCP specialist checkpoint (required)
#   --wide-checkpoint PATH     Path to Wide specialist checkpoint (optional)
#   --longcp-seeds PATH        Path to LongCP seed config (default: data/rl_configs/train_long_cp_p08_seeds.json)
#   --wide-seeds PATH          Path to Wide seed config (default: data/rl_configs/train_wide_p005_seeds.json)
#   --output-dir PATH          Output directory (default: evals_custom)
#   --device DEVICE            Device to use (default: cpu)
#   --num-repeats N            Number of repeats per seed (default: 1)
#   --max-seeds N              Maximum seeds to evaluate (default: 100)
#
# Examples:
#   # Evaluate LongCP specialist on AL host config
#   bash 4_evaluate_trained_agents.sh \
#       --longcp-checkpoint logs/longcp_AL/ablation/per_variant/hetero/hetero_best.pt \
#       --host-specs data/host_specs_AL.json \
#       --output-dir evals_custom/AL
#
#   # Evaluate both specialists
#   bash 4_evaluate_trained_agents.sh \
#       --longcp-checkpoint logs/longcp_AL/ablation/per_variant/hetero/hetero_best.pt \
#       --wide-checkpoint logs/wide_AL/ablation/per_variant/hetero/hetero_best.pt \
#       --host-specs data/host_specs_AL.json \
#       --output-dir evals_custom/AL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

# Default values
LONGCP_SEEDS="data/rl_configs/train_long_cp_p08_seeds.json"
WIDE_SEEDS="data/rl_configs/train_wide_p005_seeds.json"
OUTPUT_DIR="evals_custom"
DEVICE="cpu"
NUM_REPEATS="1"
MAX_SEEDS="100"
HOST_SPECS=""
LONGCP_CHECKPOINT=""
WIDE_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host-specs)
            HOST_SPECS="$2"
            shift 2
            ;;
        --longcp-checkpoint)
            LONGCP_CHECKPOINT="$2"
            shift 2
            ;;
        --wide-checkpoint)
            WIDE_CHECKPOINT="$2"
            shift 2
            ;;
        --longcp-seeds)
            LONGCP_SEEDS="$2"
            shift 2
            ;;
        --wide-seeds)
            WIDE_SEEDS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-repeats)
            NUM_REPEATS="$2"
            shift 2
            ;;
        --max-seeds)
            MAX_SEEDS="$2"
            shift 2
            ;;
        --help)
            head -n 30 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$HOST_SPECS" ]]; then
    echo "Error: --host-specs is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [[ -z "$LONGCP_CHECKPOINT" ]] && [[ -z "$WIDE_CHECKPOINT" ]]; then
    echo "Error: At least one of --longcp-checkpoint or --wide-checkpoint is required"
    echo "Run with --help for usage information"
    exit 1
fi

echo "=========================================="
echo "Evaluating Trained Agents"
echo "=========================================="
echo "Host specs:     $HOST_SPECS"
echo "Output dir:     $OUTPUT_DIR"
echo "Device:         $DEVICE"
echo "Num repeats:    $NUM_REPEATS"
echo "Max seeds:      $MAX_SEEDS"
echo "=========================================="
echo ""

# Build evaluation command based on which checkpoints are provided (delegate to private cogito)
PY_CMD='from cogito.tools.eval_hetero_agents_over_seed_configs import main, Args; import tyro; main(tyro.cli(Args))'
EVAL_CMD="python -c \"$PY_CMD\""
EVAL_CMD="$EVAL_CMD --host-specs-path $HOST_SPECS"
EVAL_CMD="$EVAL_CMD --device $DEVICE"
EVAL_CMD="$EVAL_CMD --eval-repeats-per-seed $NUM_REPEATS"

if [[ -n "$LONGCP_CHECKPOINT" ]]; then
    EVAL_CMD="$EVAL_CMD --longcp-ckpt $LONGCP_CHECKPOINT"
    EVAL_CMD="$EVAL_CMD --longcp-config $LONGCP_SEEDS"
    echo "LongCP checkpoint: $LONGCP_CHECKPOINT"
fi

if [[ -n "$WIDE_CHECKPOINT" ]]; then
    EVAL_CMD="$EVAL_CMD --wide-ckpt $WIDE_CHECKPOINT"
    EVAL_CMD="$EVAL_CMD --wide-config $WIDE_SEEDS"
    echo "Wide checkpoint:   $WIDE_CHECKPOINT"
fi

EVAL_CMD="$EVAL_CMD --out-csv ${OUTPUT_DIR}/hetero_eval.csv"

echo ""
echo "Running evaluation..."
# Suppress verbose output (req_div_mem messages)
eval $EVAL_CMD 2>&1 | grep -v "req_div_mem=" | grep -v "req_div_core="

echo ""
echo "✓ Evaluation complete"

echo "=========================================="
echo "Agent evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
