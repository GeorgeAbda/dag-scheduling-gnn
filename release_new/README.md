# GNN-based DAG Scheduling Framework

A deep reinforcement learning framework for energy-aware workflow scheduling in cloud environments using Graph Neural Networks.

## Features

- **GNN-based Agent**: Heterogeneous graph neural network for task-VM assignment
- **Multi-objective Optimization**: Balance makespan and energy consumption
- **Specialist Training**: Domain-specific agents for different DAG topologies
- **Comprehensive Evaluation**: Heuristic baselines, EAF plots, statistical analysis

## Installation

```bash
# Create environment
conda create -n drlgnn python=3.10
conda activate drlgnn

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Configuration

```bash
# Generate config with representative seeds
python generate_training_config.py \
    --style wide \
    --num_seeds 100 \
    --k 10 \
    -o data/rl_configs/my_wide_config.json \
    --plot
```

### 2. Train an Agent

```bash
# Train wide specialist
python run_training.py --config configs/train_wide_specialist.yaml

# Or with custom parameters
python run_training.py --config configs/train_wide_specialist.yaml \
    --total_timesteps 500000 \
    --learning_rate 0.0003
```

### 3. Evaluate Heuristics

```bash
# Compare against heuristic baselines
python eval_heuristics.py --configs longcp wide --cases AL HS
```

### 4. Evaluate Trained Agents

```bash
# Evaluate specialists with EAF plots
python eval_agents.py --regimes AL HS HP
```

## Project Structure

```
├── run_training.py          # Training launcher
├── generate_training_config.py  # Config generator
├── eval_heuristics.py       # Heuristic evaluation
├── eval_agents.py           # Agent evaluation with EAF
├── configs/                  # YAML training configs
├── data/
│   ├── rl_configs/          # Seed configs and defaults
│   └── host_specs*.json     # Host specifications
└── scheduler/               # Core framework (compiled)
```

## Configuration Files

### Training Config (YAML)

```yaml
experiment:
  name: wide_specialist
  seed: 12345

training:
  total_timesteps: 2000000
  num_envs: 10
  learning_rate: 0.00025

variant:
  name: hetero
  gnn_layers: 2
  embedding_dim: 64
  hidden_dim: 128

domain:
  style: wide
  config_path: data/rl_configs/train_wide.json
```

### Dataset Defaults (YAML)

Edit `data/rl_configs/dataset_defaults.yaml` to change default parameters:

```yaml
wide:
  edge_probability: 0.03
  min_tasks: 24
  max_tasks: 30
  hosts: 10
  vms: 10
```

### Agent Evaluation Config (YAML)

Edit `data/rl_configs/eval_agents_config.yaml`:

```yaml
regimes:
  AL:
    host_specs: data/host_specs.json
    longcp_checkpoint: logs/.../hetero_best_return.pt
    wide_checkpoint: logs/.../hetero_best_return.pt
```

## Outputs

### Training
- `logs/{experiment}/ablation/per_variant/hetero/hetero_best_return.pt` - Best checkpoint
- `logs/{experiment}/metrics.csv` - Training metrics

### Evaluation
- `{regime}_eval.csv` - Per-seed results
- `{regime}_summary.csv` - Aggregated statistics
- `{regime}_eaf_*.png` - EAF comparison plots

## DAG Topologies

| Style | Description | Edge Probability | Tasks |
|-------|-------------|------------------|-------|
| `long_cp` | Long critical path | 0.75 | 12-24 |
| `wide` | Wide/parallel | 0.03 | 24-30 |

## Host Regimes

| Regime | Description |
|--------|-------------|
| AL | Heterogeneous (all different) |
| HS | Homogeneous speed |
| HP | Homogeneous power |

```

