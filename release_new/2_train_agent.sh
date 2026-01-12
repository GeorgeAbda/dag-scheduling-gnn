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

# Train specialist using Cython-compiled module
# Parse YAML config and call the C extension directly
python -c "
import sys, os, yaml
sys.path.insert(0, os.path.abspath('..'))

# Load config
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

# Set host specs env var before importing
d = cfg.get('domain', {})
if d.get('host_specs_file'):
    os.environ['HOST_SPECS_PATH'] = os.path.abspath(d['host_specs_file'])

# Import and run
from cogito.gnn_deeprl_model.ablation_gnn_traj_main import main, Args

# Build args from config - only set attributes that exist
a = Args()
e = cfg.get('experiment', {})
if hasattr(a, 'exp_name'): a.exp_name = e.get('name', a.exp_name)
if hasattr(a, 'seed'): a.seed = e.get('seed', a.seed)
if hasattr(a, 'output_dir'): a.output_dir = e.get('output_dir', a.output_dir)
if hasattr(a, 'device'): a.device = e.get('device', a.device)

t = cfg.get('training', {})
if hasattr(a, 'total_timesteps'): a.total_timesteps = t.get('total_timesteps', a.total_timesteps)
if hasattr(a, 'learning_rate'): a.learning_rate = t.get('learning_rate', a.learning_rate)
if hasattr(a, 'num_envs'): a.num_envs = t.get('num_envs', a.num_envs)
if hasattr(a, 'num_steps'): a.num_steps = t.get('num_steps', a.num_steps)

# Seed control
sc = cfg.get('seed_control', {})
if sc.get('seeds_file') and hasattr(a, 'train_seeds_file'):
    a.train_seeds_file = sc['seeds_file']

# Run training
main(a)
"

echo ""
echo "Training completed!"
echo ""
echo "To train with custom parameters, use:"
echo "  python run_training.py --config configs/train_wide_specialist.yaml \\"
echo "      --total_timesteps 500000 \\"
echo "      --learning_rate 0.0003"
