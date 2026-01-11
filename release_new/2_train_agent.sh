#!/bin/bash
# Train an Agent
# This script trains a specialist agent using the provided configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "=========================================="
echo "Training Agent"
echo "=========================================="

# Default: Train wide specialist
CONFIG="${1:-configs/train_wide_specialist.yaml}"

echo "Using configuration: $CONFIG"
echo ""

# Train wide specialist
python run_training.py --config "$CONFIG"

echo ""
echo "Training completed!"
echo ""
echo "To train with custom parameters, use:"
echo "  python run_training.py --config configs/train_wide_specialist.yaml \\"
echo "      --total_timesteps 500000 \\"
echo "      --learning_rate 0.0003"
