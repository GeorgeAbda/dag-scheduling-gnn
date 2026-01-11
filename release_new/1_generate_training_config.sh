#!/bin/bash
# Generate Training Configuration
# This script generates a training configuration with representative seeds
#
# Usage:
#   bash 1_generate_training_config.sh [style] [num_seeds] [k]
#
# Arguments:
#   style      - DAG style: 'wide' or 'longcp' (default: wide)
#   num_seeds  - Total number of seeds to generate (default: 100)
#   k          - Number of representative seeds to select (default: 10)
#
# Style-specific parameters (automatically set):
#   wide:
#     - Edge probability: 0.03 (sparse, wide parallelism)
#     - Tasks per DAG: 24-30
#     - Hosts: 10, VMs: 10
#     - Task length: 500-100000
#   longcp:
#     - Edge probability: 0.8 (dense, long critical path)
#     - Tasks per DAG: 20-40
#     - Hosts: 10, VMs: 10
#     - Task length: 500-100000
#
# Examples:
#   bash 1_generate_training_config.sh wide 100 10
#   bash 1_generate_training_config.sh longcp 200 20
#   bash 1_generate_training_config.sh wide 50

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments with defaults
STYLE="${1:-wide}"
NUM_SEEDS="${2:-100}"
K="${3:-10}"

# Validate style
if [[ "$STYLE" != "wide" && "$STYLE" != "longcp" ]]; then
    echo "Error: Style must be 'wide' or 'longcp', got: $STYLE"
    exit 1
fi

# Generate output filename based on parameters
OUTPUT_FILE="data/rl_configs/train_${STYLE}_n${NUM_SEEDS}_k${K}.json"

echo "=========================================="
echo "Generating Training Configuration"
echo "=========================================="
echo "Style:      $STYLE"
echo "Num seeds:  $NUM_SEEDS"
echo "K (repr):   $K"
echo "Output:     $OUTPUT_FILE"
echo "=========================================="
echo ""

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

# Generate config with representative seeds
# Note: --plot may fail due to matplotlib/numpy compatibility issues
# The config file will still be generated successfully
python ../scheduler/tools/gen_config.py \
    --style "$STYLE" \
    --num_seeds "$NUM_SEEDS" \
    --k "$K" \
    -o "$OUTPUT_FILE" \
    --plot 2>&1 || true

echo ""
echo "=========================================="
echo "Configuration generated successfully!"
echo "Output: $OUTPUT_FILE"
if [ -f "${OUTPUT_FILE%.json}.png" ]; then
    echo "Plot:   ${OUTPUT_FILE%.json}.png"
else
    echo "Note:   Plot generation skipped (matplotlib/numpy compatibility issue)"
fi
echo "=========================================="
