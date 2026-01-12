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

# Train specialist (use delegator if present, otherwise fallback to private module)
# Suppress verbose req_divisor messages but keep progress bars (use sed with unbuffered mode)
if [[ -f "run_training.py" ]]; then
  python run_training.py --config "$CONFIG" 2>&1 | sed -u '/req_divisor=/d; /max_req_memory_mb=/d; /max_req_cpu_cores=/d; /max_cores=/d; /max_memory=/d'
else
  PY_CMD='from cogito.gnn_deeprl_model.ablation_gnn_traj_main import main, Args; import tyro; main(tyro.cli(Args))'
  python -c "$PY_CMD" --config "$CONFIG" 2>&1 | sed -u '/req_divisor=/d; /max_req_memory_mb=/d; /max_req_cpu_cores=/d; /max_cores=/d; /max_memory=/d'
fi

echo ""
echo "Training completed!"
echo ""
echo "To train with custom parameters, use:"
echo "  python run_training.py --config configs/train_wide_specialist.yaml \\"
echo "      --total_timesteps 500000 \\"
echo "      --learning_rate 0.0003"
