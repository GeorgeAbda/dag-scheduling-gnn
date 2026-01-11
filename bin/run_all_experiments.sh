#!/usr/bin/env bash
# =============================================================================
# Run All Paper Experiments
# =============================================================================
# This master script runs all experiments needed to reproduce the paper results.
# It trains specialists, evaluates heuristics, and generates comparison tables.
#
# WARNING: This will take a long time to complete (hours to days depending on hardware).
#
# Usage:
#   ./bin/run_all_experiments.sh
#   ./bin/run_all_experiments.sh --skip-training  # Only run evaluations
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SKIP_TRAINING=false
for arg in "$@"; do
    case $arg in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "Running All Paper Experiments"
echo "=============================================="
echo "Skip training: $SKIP_TRAINING"
echo "=============================================="

# Host specs cases to iterate over
HOST_SPECS_CASES=(
    "data/host_specs.json:AL"
    "data/host_specs_homoPower.json:HP"
    "data/host_specs_homospeed.json:HS"
)

if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo ">>> PHASE 1: Training Specialists <<<"
    echo ""
    
    for case_spec in "${HOST_SPECS_CASES[@]}"; do
        HOST_PATH="${case_spec%%:*}"
        CASE_NAME="${case_spec##*:}"
        
        echo "--- Training Long-CP Specialist ($CASE_NAME) ---"
        HOST_SPECS_PATH="$HOST_PATH" ./bin/train_longcp_specialist.sh
        
        echo "--- Training Wide Specialist ($CASE_NAME) ---"
        HOST_SPECS_PATH="$HOST_PATH" ./bin/train_wide_specialist.sh
    done
fi

echo ""
echo ">>> PHASE 2: Evaluating Heuristics <<<"
echo ""
./bin/eval_heuristics.sh

echo ""
echo ">>> PHASE 3: Evaluating Agents <<<"
echo ""
./bin/eval_agents_all_cases.sh

echo ""
echo ">>> PHASE 4: Generating Visualizations <<<"
echo ""
./bin/generate_dag_visualizations.sh

echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  - Training logs: logs/"
echo "  - Heuristic eval: logs/heuristic_eval/"
echo "  - Agent eval: logs/hetero_eval_all_cases/"
echo "  - Figures: figs/"
echo "=============================================="
