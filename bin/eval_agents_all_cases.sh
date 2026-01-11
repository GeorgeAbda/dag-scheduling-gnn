#!/usr/bin/env bash
# =============================================================================
# Evaluate Trained Agents Across All Cases
# =============================================================================
# This script evaluates trained specialist and generalist agents across all
# host specification cases and DAG topology combinations.
#
# Evaluations performed:
#   - Long-CP agent on Long-CP seeds
#   - Long-CP agent on Wide seeds (cross-topology)
#   - Wide agent on Wide seeds
#   - Wide agent on Long-CP seeds (cross-topology)
#
# Host Specs Cases:
#   - AL: logs/AL_case/
#   - HP: logs/HP_controlled/
#   - HS: logs/HS_controlled/
#
# Usage:
#   ./bin/eval_agents_all_cases.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
DEVICE="${DEVICE:-cpu}"
EVAL_REPEATS="${EVAL_REPEATS:-5}"
OUT_DIR="${OUT_DIR:-logs/hetero_eval_all_cases}"

echo "=============================================="
echo "Evaluating Agents Across All Cases"
echo "=============================================="
echo "Device: $DEVICE"
echo "Eval repeats per seed: $EVAL_REPEATS"
echo "Output directory: $OUT_DIR"
echo "=============================================="

"$SCRIPT_DIR/../dist/eval_agents" \
    --device "$DEVICE" \
    --eval_repeats_per_seed "$EVAL_REPEATS" \
    --out_dir "$OUT_DIR"

echo "=============================================="
echo "Agent evaluation complete!"
echo "Results saved to: $OUT_DIR"
echo "=============================================="
