#!/usr/bin/env bash
# =============================================================================
# Train Generalist Agent (Mixed DAG Topologies)
# =============================================================================
# This script trains a generalist agent on mixed DAG topologies.
# The agent learns to schedule both wide (sparse) and long-cp (dense) workflows.
#
# Host Specs Cases:
#   - AL (default): data/host_specs.json
#   - HP (homogeneous power): data/host_specs_homoPower.json
#   - HS (homogeneous speed): data/host_specs_homospeed.json
#
# Usage:
#   ./bin/train_generalist.sh                    # Uses default host_specs
#   HOST_SPECS_PATH=data/host_specs_homoPower.json ./bin/train_generalist.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
EXP_NAME="${EXP_NAME:-generalist_mixed}"
SEED="${SEED:-12345}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4000000}"  # Longer training for generalist
NUM_ENVS="${NUM_ENVS:-10}"
DEVICE="${DEVICE:-cpu}"

# Host specs (can be overridden via environment variable)
export HOST_SPECS_PATH="${HOST_SPECS_PATH:-$PROJECT_ROOT/data/host_specs.json}"

# Derive case name from host specs path for experiment naming
CASE_NAME="AL"
if [[ "$HOST_SPECS_PATH" == *"homoPower"* ]]; then
    CASE_NAME="HP"
elif [[ "$HOST_SPECS_PATH" == *"homospeed"* ]] || [[ "$HOST_SPECS_PATH" == *"homogenousSPEED"* ]]; then
    CASE_NAME="HS"
fi

FULL_EXP_NAME="${EXP_NAME}_${CASE_NAME}"

echo "=============================================="
echo "Training Generalist Agent (Mixed Topologies)"
echo "=============================================="
echo "Experiment: $FULL_EXP_NAME"
echo "Host Specs: $HOST_SPECS_PATH"
echo "Case: $CASE_NAME"
echo "Seed: $SEED"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "Num Envs: $NUM_ENVS"
echo "Device: $DEVICE"
echo "=============================================="

# Train with both wide and longcp configs for mixed training
"$SCRIPT_DIR/../dist/train_agent" \
    --exp_name "$FULL_EXP_NAME" \
    --seed "$SEED" \
    --train_only_variant hetero \
    --training_seed_mode controlled \
    --wide_config data/rl_configs/train_wide_p005_seeds.json \
    --longcp_config data/rl_configs/train_long_cp_p08_seeds.json \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --num_envs "$NUM_ENVS" \
    --device "$DEVICE" \
    --trajectory-enabled \
    --trajectory-collect-every 10 \
    --trajectory-method svd

echo "=============================================="
echo "Training complete!"
echo "Logs saved to: logs/$FULL_EXP_NAME"
echo "=============================================="
