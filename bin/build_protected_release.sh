#!/bin/bash
# Build protected release using bytecode compilation with additional obfuscation
# This provides reasonable protection while maintaining compatibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RELEASE_DIR="$PROJECT_ROOT/release_v2"

echo "=============================================="
echo "Building Release v2"
echo "=============================================="

# Backup generate_training_config.py before cleaning (if it exists with plotting)
GENCONFIG_BACKUP=""
if [ -f "$RELEASE_DIR/generate_training_config.py" ] && grep -q "plot_selected_dags" "$RELEASE_DIR/generate_training_config.py" 2>/dev/null; then
    GENCONFIG_BACKUP=$(mktemp)
    cp "$RELEASE_DIR/generate_training_config.py" "$GENCONFIG_BACKUP"
fi

# Clean previous builds
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo ""
echo "[1/6] Creating directory structure..."
echo "----------------------------------------------"

mkdir -p "$RELEASE_DIR/bin"
mkdir -p "$RELEASE_DIR/data"
mkdir -p "$RELEASE_DIR/runs/datasets"
mkdir -p "$RELEASE_DIR/scheduler"
mkdir -p "$RELEASE_DIR/scripts"

echo ""
echo "[2/6] Copying and compiling scheduler module..."
echo "----------------------------------------------"

# Copy scheduler directory
cp -r scheduler "$RELEASE_DIR/"

# Obfuscate: remove comments, docstrings
echo "  Obfuscating source code..."
python bin/obfuscate.py "$RELEASE_DIR/scheduler" 2>/dev/null || true

# Compile to optimized bytecode (level 2 = remove docstrings and asserts)
python -OO -m compileall -b -f "$RELEASE_DIR/scheduler" 2>/dev/null || true

# Remove all .py source files, keep only .pyc
find "$RELEASE_DIR/scheduler" -name "*.py" -type f -delete

echo "  Compiled scheduler/ to bytecode (obfuscated)"

echo ""
echo "[3/6] Copying and compiling scripts..."
echo "----------------------------------------------"

cp -r scripts "$RELEASE_DIR/"
echo "  Obfuscating scripts..."
python bin/obfuscate.py "$RELEASE_DIR/scripts" 2>/dev/null || true
python -OO -m compileall -b -f "$RELEASE_DIR/scripts" 2>/dev/null || true
find "$RELEASE_DIR/scripts" -name "*.py" -type f -delete
echo "  Compiled scripts/ to bytecode (obfuscated)"

echo ""
echo "[4/6] Copying data and configs..."
echo "----------------------------------------------"

cp -r data/* "$RELEASE_DIR/data/" 2>/dev/null || true
# Ensure dataset_defaults.yaml is copied
cp data/rl_configs/dataset_defaults.yaml "$RELEASE_DIR/data/rl_configs/" 2>/dev/null || true
echo "  Copied: data/"

cp -r runs/datasets/* "$RELEASE_DIR/runs/datasets/" 2>/dev/null || true
echo "  Copied: runs/datasets/"

# Copy the config generator script (keep as .py for user modification)
cp scheduler/rl_model/representative_eval.py "$RELEASE_DIR/" 2>/dev/null || true

# Create generate_training_config module (compiled to .pyc in __pycache__)
mkdir -p "$RELEASE_DIR/scheduler/tools"
echo "" > "$RELEASE_DIR/scheduler/tools/__init__.py"

# Copy tools from scheduler/tools
if [ -f "$PROJECT_ROOT/scheduler/tools/gen_config.py" ]; then
    cp "$PROJECT_ROOT/scheduler/tools/gen_config.py" "$RELEASE_DIR/scheduler/tools/gen_config.py"
elif [ -n "$GENCONFIG_BACKUP" ] && [ -f "$GENCONFIG_BACKUP" ]; then
    cp "$GENCONFIG_BACKUP" "$RELEASE_DIR/scheduler/tools/gen_config.py"
    rm -f "$GENCONFIG_BACKUP"
fi

if [ -f "$PROJECT_ROOT/scheduler/tools/eval_heuristics.py" ]; then
    cp "$PROJECT_ROOT/scheduler/tools/eval_heuristics.py" "$RELEASE_DIR/scheduler/tools/eval_heuristics.py"
fi

if [ -f "$PROJECT_ROOT/scheduler/tools/eval_agents.py" ]; then
    cp "$PROJECT_ROOT/scheduler/tools/eval_agents.py" "$RELEASE_DIR/scheduler/tools/eval_agents.py"
fi

# Obfuscate and compile tools
if [ -f "$RELEASE_DIR/scheduler/tools/gen_config.py" ]; then
    echo "  Obfuscating tools..."
    python bin/obfuscate.py "$RELEASE_DIR/scheduler/tools/" 2>/dev/null || true
    python3 -OO -m compileall -b "$RELEASE_DIR/scheduler/tools/" -q
    # Remove source files, keep .pyc
    rm -f "$RELEASE_DIR/scheduler/tools/gen_config.py"
    rm -f "$RELEASE_DIR/scheduler/tools/eval_heuristics.py"
    rm -f "$RELEASE_DIR/scheduler/tools/eval_agents.py"
    rm -f "$RELEASE_DIR/scheduler/tools/__init__.py"
    rm -rf "$RELEASE_DIR/scheduler/tools/__pycache__"
    echo "  Compiled: scheduler/tools/*.pyc (obfuscated)"
fi

# Create thin launcher for generate_training_config
cat > "$RELEASE_DIR/generate_training_config.py" << 'GENEOF'
#!/usr/bin/env python3
"""Generate Training Configuration - Launcher"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.tools.gen_config import main
if __name__ == "__main__": main()
GENEOF
chmod +x "$RELEASE_DIR/generate_training_config.py"
echo "  Created: generate_training_config.py (protected)"

# Create thin launcher for eval_heuristics
cat > "$RELEASE_DIR/eval_heuristics.py" << 'EVALEOF'
#!/usr/bin/env python3
"""Evaluate Heuristics - Launcher"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.tools.eval_heuristics import main
if __name__ == "__main__": main()
EVALEOF
chmod +x "$RELEASE_DIR/eval_heuristics.py"
echo "  Created: eval_heuristics.py (protected)"

# Create thin launcher for eval_agents
cat > "$RELEASE_DIR/eval_agents.py" << 'AGENTEOF'
#!/usr/bin/env python3
"""Evaluate Agents - Launcher"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.tools.eval_agents import main
if __name__ == "__main__": main()
AGENTEOF
chmod +x "$RELEASE_DIR/eval_agents.py"
echo "  Created: eval_agents.py (protected)"

echo ""
echo "[5/6] Creating launcher scripts..."
echo "----------------------------------------------"

# Create a Python wrapper that imports from .pyc
cat > "$RELEASE_DIR/run_training.py" << 'PYEOF'
#!/usr/bin/env python3
"""Training launcher - imports compiled modules."""
import sys
import os

# Add release directory to path
release_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, release_dir)

# Import and run
from scheduler.rl_model.ablation_gnn_traj_main import main, Args
import tyro

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
PYEOF

# Copy and modify shell scripts
for script in bin/train_*.sh bin/eval_*.sh; do
    if [ -f "$script" ]; then
        name=$(basename "$script")
        cp "$script" "$RELEASE_DIR/bin/$name"
        
        # Update the script to use the release directory
        sed -i '' 's|python -m scheduler|python run_training.py|g' "$RELEASE_DIR/bin/$name" 2>/dev/null || true
        chmod +x "$RELEASE_DIR/bin/$name"
        echo "  Created: bin/$name"
    fi
done

echo ""
echo "[6/6] Creating configs and documentation..."
echo "----------------------------------------------"

# Create configs directory
mkdir -p "$RELEASE_DIR/configs"

# Create config files
cat > "$RELEASE_DIR/configs/train_longcp_specialist.yaml" << 'YAMLEOF'
# Training Configuration: Long Critical Path Specialist
experiment:
  name: "longcp_specialist"
  seed: 12345
  output_dir: "logs"
  device: "cpu"

training:
  total_timesteps: 2000000
  learning_rate: 0.00025
  num_envs: 10
  num_steps: 256
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 4
  update_epochs: 4
  clip_coef: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  anneal_lr: true

evaluation:
  test_every_iters: 5
  robust_eval_alpha: 0.25

domain:
  longcp_config: "data/rl_configs/train_long_cp_p08_seeds.json"
  wide_config: null
  
  # Host hardware specifications - CHANGE THIS depending on experiment:
  # - "data/host_specs.json"           : Heterogeneous hosts (default)
  # - "data/host_specs_homospeed.json" : Homogeneous CPU speed
  # - "data/host_specs_homoPower.json" : Homogeneous power consumption
  host_specs_file: "data/host_specs.json"

# Training seeds are loaded from the domain config file (training_seeds field)
seed_control:
  mode: "controlled"

variant:
  name: "hetero"

trajectory:
  enabled: false
  collect_every: 10
  method: "svd"

logging:
  tensorboard: true
  log_every: 10
YAMLEOF

cat > "$RELEASE_DIR/configs/train_wide_specialist.yaml" << 'YAMLEOF'
# Training Configuration: Wide Specialist
experiment:
  name: "wide_specialist"
  seed: 12345
  output_dir: "logs"
  device: "cpu"

training:
  total_timesteps: 2000000
  learning_rate: 0.00025
  num_envs: 10
  num_steps: 256
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 4
  update_epochs: 4
  clip_coef: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  anneal_lr: true

evaluation:
  test_every_iters: 5
  robust_eval_alpha: 0.25

domain:
  longcp_config: null
  wide_config: "data/rl_configs/train_wide_p005_seeds.json"
  
  # Host hardware specifications - CHANGE THIS depending on experiment:
  # - "data/host_specs.json"           : Heterogeneous hosts (default)
  # - "data/host_specs_homospeed.json" : Homogeneous CPU speed
  # - "data/host_specs_homoPower.json" : Homogeneous power consumption
  host_specs_file: "data/host_specs.json"

# Training seeds are loaded from the domain config file (training_seeds field)
seed_control:
  mode: "controlled"

variant:
  name: "hetero"

trajectory:
  enabled: false
  collect_every: 10
  method: "svd"

logging:
  tensorboard: true
  log_every: 10
YAMLEOF

cat > "$RELEASE_DIR/configs/train_generalist.yaml" << 'YAMLEOF'
# Training Configuration: Generalist Agent (Mixed DAGs)
experiment:
  name: "generalist"
  seed: 12345
  output_dir: "logs"
  device: "cpu"

training:
  total_timesteps: 4000000
  learning_rate: 0.00025
  num_envs: 10
  num_steps: 256
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 4
  update_epochs: 4
  clip_coef: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  anneal_lr: true

evaluation:
  test_every_iters: 5
  robust_eval_alpha: 0.25

domain:
  longcp_config: "data/rl_configs/train_long_cp_p08_seeds.json"
  wide_config: "data/rl_configs/train_wide_p005_seeds.json"
  
  # Host hardware specifications - CHANGE THIS depending on experiment:
  # - "data/host_specs.json"           : Heterogeneous hosts (default)
  # - "data/host_specs_homospeed.json" : Homogeneous CPU speed
  # - "data/host_specs_homoPower.json" : Homogeneous power consumption
  host_specs_file: "data/host_specs.json"

# Training seeds are loaded from the domain config files (training_seeds field)
seed_control:
  mode: "controlled"

variant:
  name: "hetero"

trajectory:
  enabled: true
  collect_every: 20
  method: "svd"

logging:
  tensorboard: true
  log_every: 10
YAMLEOF

echo "  Created: configs/*.yaml"

# Create the launcher script
cat > "$RELEASE_DIR/run_training.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Training Launcher - Run with YAML config or CLI arguments.

Usage:
    python run_training.py --config configs/train_longcp_specialist.yaml
    python run_training.py --config configs/train_longcp_specialist.yaml --total_timesteps 500000
    python run_training.py --exp_name my_exp --total_timesteps 1000000
"""
import sys, os, argparse
release_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, release_dir)

def load_config(path):
    import yaml
    with open(path) as f: return yaml.safe_load(f)

def config_to_args(cfg):
    from scheduler.rl_model.ablation_gnn_traj_main import Args
    a = Args()
    e = cfg.get('experiment', {})
    a.exp_name = e.get('name', a.exp_name)
    a.seed = e.get('seed', a.seed)
    a.output_dir = e.get('output_dir', a.output_dir)
    a.device = e.get('device', a.device)
    t = cfg.get('training', {})
    a.total_timesteps = t.get('total_timesteps', a.total_timesteps)
    a.learning_rate = t.get('learning_rate', a.learning_rate)
    a.num_envs = t.get('num_envs', a.num_envs)
    a.num_steps = t.get('num_steps', a.num_steps)
    a.gamma = t.get('gamma', a.gamma)
    a.gae_lambda = t.get('gae_lambda', a.gae_lambda)
    a.num_minibatches = t.get('num_minibatches', a.num_minibatches)
    a.update_epochs = t.get('update_epochs', a.update_epochs)
    a.clip_coef = t.get('clip_coef', a.clip_coef)
    a.ent_coef = t.get('ent_coef', a.ent_coef)
    a.vf_coef = t.get('vf_coef', a.vf_coef)
    a.max_grad_norm = t.get('max_grad_norm', a.max_grad_norm)
    a.anneal_lr = t.get('anneal_lr', a.anneal_lr)
    a.norm_adv = t.get('norm_adv', a.norm_adv)
    a.clip_vloss = t.get('clip_vloss', a.clip_vloss)
    ev = cfg.get('evaluation', {})
    a.test_every_iters = ev.get('test_every_iters', a.test_every_iters)
    a.robust_eval_alpha = ev.get('robust_eval_alpha', a.robust_eval_alpha)
    d = cfg.get('domain', {})
    a.longcp_config = d.get('longcp_config')
    a.wide_config = d.get('wide_config')
    # Training seeds are now embedded in the domain config files (training_seeds field)
    # Set host specs path via environment variable
    host_specs = d.get('host_specs_file')
    if host_specs:
        import os as _os
        _os.environ['HOST_SPECS_PATH'] = _os.path.abspath(host_specs)
    s = cfg.get('seed_control', {})
    a.training_seed_mode = s.get('mode', a.training_seed_mode)
    v = cfg.get('variant', {})
    a.train_only_variant = v.get('name', a.train_only_variant)
    tr = cfg.get('trajectory', {})
    a.trajectory_enabled = tr.get('enabled', a.trajectory_enabled)
    a.trajectory_collect_every = tr.get('collect_every', a.trajectory_collect_every)
    a.trajectory_method = tr.get('method', a.trajectory_method)
    l = cfg.get('logging', {})
    a.no_tensorboard = not l.get('tensorboard', True)
    a.log_every = l.get('log_every', a.log_every)
    return a

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config', '-c', type=str)
    p.add_argument('--help', '-h', action='store_true')
    known, rest = p.parse_known_args()
    if known.config:
        print(f"Loading: {known.config}")
        args = config_to_args(load_config(known.config))
        if rest:
            op = argparse.ArgumentParser()
            op.add_argument('--exp_name', type=str)
            op.add_argument('--seed', type=int)
            op.add_argument('--device', type=str)
            op.add_argument('--total_timesteps', type=int)
            op.add_argument('--num_envs', type=int)
            op.add_argument('--trajectory_enabled', action='store_true')
            ov, _ = op.parse_known_args(rest)
            for k,v in vars(ov).items():
                if v is not None: setattr(args, k, v)
        from scheduler.rl_model.ablation_gnn_traj_main import main as train
        train(args)
    elif known.help:
        print(__doc__)
        print("\nConfigs:", *[f"  - {c}" for c in sorted(__import__('glob').glob("configs/*.yaml"))], sep='\n')
    else:
        import tyro
        from scheduler.rl_model.ablation_gnn_traj_main import main as train, Args
        train(tyro.cli(Args))

if __name__ == "__main__": main()
PYEOF

echo "  Created: run_training.py"

cat > "$RELEASE_DIR/README.md" << 'EOF'
# Protected Release

This release contains compiled Python bytecode that cannot be easily read.

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
cd release_protected
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Train long-CP specialist
./bin/train_longcp_specialist.sh

# Or run directly:
python run_training.py --exp_name my_experiment --total_timesteps 1000000
```

## Available Scripts

- `train_longcp_specialist.sh` - Train on long critical path DAGs
- `train_wide_specialist.sh` - Train on wide (sparse) DAGs
- `train_generalist.sh` - Train on mixed DAG types

## Output

Results are saved to `logs/<exp_name>/ablation/per_variant/<variant>/`
EOF

# Copy documentation
mkdir -p "$RELEASE_DIR/docs"
cp docs/API.md "$RELEASE_DIR/docs/" 2>/dev/null || true
cp docs/EXTENDING.md "$RELEASE_DIR/docs/" 2>/dev/null || true
cp docs/README_RELEASE.md "$RELEASE_DIR/README.md" 2>/dev/null || true
cp docs/demo_scheduling.ipynb "$RELEASE_DIR/docs/" 2>/dev/null || true
echo "  Copied: docs/"

# Copy requirements
cp requirements.txt "$RELEASE_DIR/" 2>/dev/null || echo "torch>=2.0.0
torch-geometric>=2.0.0
gymnasium>=0.29.0
numpy
pandas
matplotlib
tyro
tqdm" > "$RELEASE_DIR/requirements.txt"

# Create archive
echo ""
echo "Creating archive..."
cd "$PROJECT_ROOT"
rm -f release_v2.zip
zip -r release_v2.zip release_v2 -x "*.py" -x "*__pycache__*" -q

# Re-add the launcher .py file
cd release_v2
zip -u ../release_v2.zip run_training.py -q
cd "$PROJECT_ROOT"

echo ""
echo "=============================================="
echo "Release v2 Build Complete!"
echo "=============================================="
echo ""
echo "Output: $RELEASE_DIR"
echo "Archive: release_v2.zip ($(du -h release_v2.zip | cut -f1))"
echo ""
echo "Contents:"
ls -la "$RELEASE_DIR"
echo ""
echo "Bytecode files:"
find "$RELEASE_DIR" -name "*.pyc" | wc -l | xargs echo "  Total .pyc files:"
echo ""
echo "Source files removed:"
find "$RELEASE_DIR" -name "*.py" | wc -l | xargs echo "  Remaining .py files:"
echo "=============================================="
