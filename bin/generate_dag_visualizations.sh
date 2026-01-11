#!/usr/bin/env bash
# =============================================================================
# Generate DAG Topology Visualizations
# =============================================================================
# This script generates publication-quality DAG structure visualizations
# showing the difference between wide (sparse) and long-cp (dense) topologies.
#
# Usage:
#   ./bin/generate_dag_visualizations.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Output directory
OUT_DIR="${OUT_DIR:-figs/dag_illustrations}"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Generating DAG Visualizations"
echo "=============================================="
echo "Output directory: $OUT_DIR"
echo "=============================================="

python draw_dag_illustrations.py

echo "=============================================="
echo "DAG visualizations complete!"
echo "=============================================="
