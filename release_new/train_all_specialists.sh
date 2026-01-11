#!/bin/bash
# Train all 8 specialist agents (LongCP and Wide across AL, NAL, HP, HS)
#
# Usage:
#   bash train_all_specialists.sh [parallel|sequential]
#
# Default: sequential (one after another)
# parallel: Launch all 8 in background (requires 8 GPUs or enough VRAM)

set -e

MODE="${1:-sequential}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Training All Specialist Agents"
echo "Mode: $MODE"
echo "=========================================="

# Training configurations
CONFIGS=(
    "configs/train_longcp_aligned.yaml"
    "configs/train_longcp_not_aligned.yaml"
    "configs/train_longcp_homopower.yaml"
    "configs/train_longcp_homospeed.yaml"
    "configs/train_wide_aligned.yaml"
    "configs/train_wide_not_aligned.yaml"
    "configs/train_wide_homopower.yaml"
    "configs/train_wide_homospeed.yaml"
)

# Function to train a single agent
train_agent() {
    local config=$1
    local name=$(basename "$config" .yaml)
    
    echo ""
    echo "=========================================="
    echo "Training: $name"
    echo "Config: $config"
    echo "Started: $(date)"
    echo "=========================================="
    
    python run_training.py --config "$config"
    
    echo "Completed: $name at $(date)"
}

if [ "$MODE" = "parallel" ]; then
    echo "Launching all 8 training jobs in parallel..."
    echo "WARNING: This requires significant GPU memory!"
    echo ""
    
    pids=()
    for config in "${CONFIGS[@]}"; do
        train_agent "$config" &
        pids+=($!)
        sleep 5  # Stagger launches slightly
    done
    
    echo "All jobs launched. Waiting for completion..."
    for pid in "${pids[@]}"; do
        wait $pid
        echo "Job $pid completed"
    done
    
else
    echo "Training sequentially (one after another)..."
    echo ""
    
    for config in "${CONFIGS[@]}"; do
        train_agent "$config"
    done
fi

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Checkpoints saved in: logs/"
echo ""
echo "Next steps:"
echo "  1. Evaluate agents: bash eval_new_checkpoints_all_cases.sh"
echo "  2. Analyze results: python ../scripts/analyze_objective_correlation_per_case.py"
