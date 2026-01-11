#!/bin/bash

# Parallel evaluation script for all host cases using new checkpoints from release_new/logs
# Usage: bash eval_new_checkpoints_all_cases.sh

set -e

REPO_ROOT="/Users/anashattay/Documents/GitHub/DaDiL/to-github"
cd "$REPO_ROOT"

DEVICE="cpu"
EVAL_REPEATS=3
OUT_BASE="$REPO_ROOT/release_new/evals_new_ckpts"

# Create output directories
mkdir -p "$OUT_BASE/AL"
mkdir -p "$OUT_BASE/NAL"
mkdir -p "$OUT_BASE/HP"
mkdir -p "$OUT_BASE/HS"

echo "=========================================="
echo "Starting parallel evaluation of all cases"
echo "=========================================="
echo ""

# AL case
echo "Launching AL evaluation..."
(
  export HOST_SPECS_PATH="$REPO_ROOT/release_new/data/host_specs_AL.json"
  conda run -n drlgnn_test python scripts/eval_hetero_agents_over_seed_configs.py \
    --device "$DEVICE" \
    --eval-repeats-per-seed "$EVAL_REPEATS" \
    --out-csv "$OUT_BASE/AL/hetero_eval.csv" \
    --longcp-ckpt "$REPO_ROOT/release_new/logs/longcp_aligned/ablation/per_variant/hetero/hetero_best.pt" \
    --wide-ckpt "$REPO_ROOT/release_new/logs/wide_aligned/ablation/per_variant/hetero/hetero_best.pt" \
    --host-specs-path "$REPO_ROOT/release_new/data/host_specs_AL.json" \
    --longcp-config "$REPO_ROOT/release_new/data/rl_configs/train_long_cp_p08_seeds.json" \
    --wide-config "$REPO_ROOT/release_new/data/rl_configs/train_wide_p005_seeds.json"
  echo "✓ AL evaluation complete"
) &

# NAL case
echo "Launching NAL evaluation..."
(
  export HOST_SPECS_PATH="$REPO_ROOT/release_new/data/host_specs_NAL.json"
  conda run -n drlgnn_test python scripts/eval_hetero_agents_over_seed_configs.py \
    --device "$DEVICE" \
    --eval-repeats-per-seed "$EVAL_REPEATS" \
    --out-csv "$OUT_BASE/NAL/hetero_eval.csv" \
    --longcp-ckpt "$REPO_ROOT/release_new/logs/longcp_not_aligned/ablation/per_variant/hetero/hetero_best.pt" \
    --wide-ckpt "$REPO_ROOT/release_new/logs/wide_not_aligned/ablation/per_variant/hetero/hetero_best.pt" \
    --host-specs-path "$REPO_ROOT/release_new/data/host_specs_NAL.json" \
    --longcp-config "$REPO_ROOT/release_new/data/rl_configs/train_long_cp_p08_seeds.json" \
    --wide-config "$REPO_ROOT/release_new/data/rl_configs/train_wide_p005_seeds.json"
  echo "✓ NAL evaluation complete"
) &

# HP case
echo "Launching HP evaluation..."
(
  export HOST_SPECS_PATH="$REPO_ROOT/release_new/data/host_specs_homoPower.json"
  conda run -n drlgnn_test python scripts/eval_hetero_agents_over_seed_configs.py \
    --device "$DEVICE" \
    --eval-repeats-per-seed "$EVAL_REPEATS" \
    --out-csv "$OUT_BASE/HP/hetero_eval.csv" \
    --longcp-ckpt "$REPO_ROOT/release_new/logs/longcp_homopower/ablation/per_variant/hetero/hetero_best.pt" \
    --wide-ckpt "$REPO_ROOT/release_new/logs/wide_homopower/ablation/per_variant/hetero/hetero_best.pt" \
    --host-specs-path "$REPO_ROOT/release_new/data/host_specs_homoPower.json" \
    --longcp-config "$REPO_ROOT/release_new/data/rl_configs/train_long_cp_p08_seeds.json" \
    --wide-config "$REPO_ROOT/release_new/data/rl_configs/train_wide_p005_seeds.json"
  echo "✓ HP evaluation complete"
) &

# HS case
echo "Launching HS evaluation..."
(
  export HOST_SPECS_PATH="$REPO_ROOT/release_new/data/host_specs_homospeed.json"
  conda run -n drlgnn_test python scripts/eval_hetero_agents_over_seed_configs.py \
    --device "$DEVICE" \
    --eval-repeats-per-seed "$EVAL_REPEATS" \
    --out-csv "$OUT_BASE/HS/hetero_eval.csv" \
    --longcp-ckpt "$REPO_ROOT/release_new/logs/longcp_homospeed/ablation/per_variant/hetero/hetero_best.pt" \
    --wide-ckpt "$REPO_ROOT/release_new/logs/wide_homospeed/ablation/per_variant/hetero/hetero_best.pt" \
    --host-specs-path "$REPO_ROOT/release_new/data/host_specs_homospeed.json" \
    --longcp-config "$REPO_ROOT/release_new/data/rl_configs/train_long_cp_p08_seeds.json" \
    --wide-config "$REPO_ROOT/release_new/data/rl_configs/train_wide_p005_seeds.json"
  echo "✓ HS evaluation complete"
) &

echo ""
echo "All evaluations launched in parallel. Waiting for completion..."
echo ""

# Wait for all background jobs
wait

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - $OUT_BASE/AL/hetero_eval.summary.csv"
echo "  - $OUT_BASE/NAL/hetero_eval.summary.csv"
echo "  - $OUT_BASE/HP/hetero_eval.summary.csv"
echo "  - $OUT_BASE/HS/hetero_eval.summary.csv"
echo ""
