#!/usr/bin/env bash
# =============================================================================
# Run Pareto Front Analysis (NSGA-II Reference Search)
# =============================================================================
# This script runs multi-objective optimization to find Pareto-optimal schedules
# for comparison with RL agents and heuristics.
#
# Objectives:
#   - Minimize makespan
#   - Minimize energy consumption
#
# Usage:
#   ./bin/run_pareto_analysis.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
OUT_DIR="${OUT_DIR:-logs/pareto_analysis}"
POPULATION="${POPULATION:-100}"
GENERATIONS="${GENERATIONS:-200}"

echo "=============================================="
echo "Running Pareto Front Analysis"
echo "=============================================="
echo "Output directory: $OUT_DIR"
echo "Population: $POPULATION"
echo "Generations: $GENERATIONS"
echo "=============================================="

python scripts/run_nsga_and_indicators.py \
    --out_dir "$OUT_DIR" \
    --population "$POPULATION" \
    --generations "$GENERATIONS"

echo "=============================================="
echo "Pareto analysis complete!"
echo "Results saved to: $OUT_DIR"
echo "=============================================="
