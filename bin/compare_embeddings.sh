#!/usr/bin/env bash
# =============================================================================
# Compare Agent Embeddings (State Space Analysis)
# =============================================================================
# This script compares the learned state representations between specialist
# agents using optimal transport and embedding visualization.
#
# Analyses performed:
#   - PCA/t-SNE visualization of state embeddings
#   - Optimal transport distance between specialist embeddings
#   - State visitation distribution comparison
#
# Usage:
#   ./bin/compare_embeddings.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Output directory
OUT_DIR="${OUT_DIR:-logs/embedding_comparison}"

echo "=============================================="
echo "Comparing Agent Embeddings"
echo "=============================================="
echo "Output directory: $OUT_DIR"
echo "=============================================="

python scripts/compare_specialists_embeddings.py

echo "=============================================="
echo "Embedding comparison complete!"
echo "=============================================="
