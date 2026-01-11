# Executable Scripts for Paper Experiments

This directory contains executable scripts to reproduce the experiments from the paper:
**"On the Role of DAG Structure in Energy-Aware Scheduling with Deep Reinforcement Learning"**

## Quick Start

```bash
# Make all scripts executable
chmod +x bin/*.sh

# STEP 1: Build standalone executables (required once)
./bin/build_executables.sh

# STEP 2: Run experiments
./bin/train_longcp_specialist.sh
./bin/train_wide_specialist.sh
./bin/eval_heuristics.sh
```

## Building Executables

The training and evaluation code is compiled into standalone binary executables.
This allows distribution without exposing the source code.

```bash
# Build all executables (creates dist/ folder)
./bin/build_executables.sh
```

This creates:
- `dist/train_agent` - Training executable
- `dist/eval_heuristics` - Heuristic evaluation executable  
- `dist/eval_agents` - Agent evaluation executable

## Training Scripts

### `train_longcp_specialist.sh`
Trains a specialist agent on **long critical path (dense)** DAG topologies.
- Uses `gnp_p=0.75` (high edge density)
- Optimized for workflows with sequential dependencies

### `train_wide_specialist.sh`
Trains a specialist agent on **wide (sparse)** DAG topologies.
- Uses `gnp_p=0.03-0.05` (low edge density)
- Optimized for workflows with high parallelism

### `train_generalist.sh`
Trains a generalist agent on **mixed** DAG topologies.
- Alternates between wide and long-cp configurations
- Tests generalization across topology types

## Evaluation Scripts

### `eval_heuristics.sh`
Evaluates classical scheduling heuristics:
- Random, Round Robin
- Min-Min, Max-Min (makespan-focused)
- HEFT (Heterogeneous Earliest Finish Time)
- Energy-aware variants

### `eval_agents_all_cases.sh`
Evaluates trained RL agents across all host specification cases:
- AL (Aligned): Default heterogeneous hosts
- HP (Homogeneous Power): Same power consumption
- HS (Homogeneous Speed): Same CPU speed

### `run_pareto_analysis.sh`
Runs NSGA-II multi-objective optimization to find Pareto-optimal schedules.

### `compare_embeddings.sh`
Compares learned state representations between specialist agents.

## Visualization Scripts

### `generate_dag_visualizations.sh`
Generates publication-quality DAG topology illustrations.

## Configuration

All scripts support environment variable overrides:

```bash
# Change host specifications
HOST_SPECS_PATH=data/host_specs_homoPower.json ./bin/train_longcp_specialist.sh

# Change training parameters
SEED=42 TOTAL_TIMESTEPS=1000000 ./bin/train_longcp_specialist.sh

# Change device
DEVICE=cuda ./bin/train_wide_specialist.sh
```

### Host Specification Cases

| Case | File | Description |
|------|------|-------------|
| AL | `data/host_specs.json` | Default heterogeneous hosts |
| HP | `data/host_specs_homoPower.json` | Homogeneous power consumption |
| HS | `data/host_specs_homospeed.json` | Homogeneous CPU speed |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_SPECS_PATH` | `data/host_specs.json` | Path to host specifications |
| `SEED` | `12345` | Random seed for reproducibility |
| `TOTAL_TIMESTEPS` | `2000000` | Training timesteps |
| `NUM_ENVS` | `10` | Number of parallel environments |
| `DEVICE` | `cpu` | Training device (cpu/cuda) |

## Output Locations

- **Training logs**: `logs/<exp_name>/`
- **Heuristic evaluation**: `logs/heuristic_eval/`
- **Agent evaluation**: `logs/hetero_eval_all_cases/`
- **Pareto analysis**: `logs/pareto_analysis/`
- **Figures**: `figs/`

## Dependencies

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```
