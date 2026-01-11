# Structure-Aware Deep RL for Heterogeneous Cloud Scheduling

[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the code for reproducing the experiments in our paper:

> **Structure-Aware Deep Reinforcement Learning for Heterogeneous Cloud Scheduling**  
> [Authors]  
> Transactions on Machine Learning Research (TMLR), 2024

## Quick Start

### 1. Clone and Setup

```bash
git clone [repository-url]
cd to-github
conda create -n cloud_sched python=3.10
conda activate cloud_sched
pip install -r requirements.txt
```

### 2. Install Scheduler Library

The `scheduler` library is required but not included in this repository. See [INSTALL_SCHEDULER.md](INSTALL_SCHEDULER.md) for installation instructions.

**Quick setup** (if you have the scheduler locally):
```bash
export PYTHONPATH="/path/to/parent/of/scheduler:$PYTHONPATH"
```

### 3. Train a Single Agent

```bash
python release_new/run_training.py --config release_new/configs/train_longcp_aligned.yaml
```

### 4. Evaluate Trained Agent

```bash
python scripts/eval_hetero_agents_over_seed_configs.py \
    --checkpoint_path release_new/logs/longcp_aligned/ablation/per_variant/hetero/hetero_best.pt \
    --host_specs_path release_new/data/host_specs_AL.json \
    --seed_config_path release_new/data/rl_configs/train_long_cp_p08_seeds.json \
    --output_dir release_new/evals_new_ckpts/AL \
    --device cuda
```

## Repository Overview

**⚠️ Important Note**: The core `scheduler/` library is proprietary and not included in this repository due to copyright restrictions. For access during review or collaboration, please contact the authors.

### Key Scripts

| Script | Purpose |
|--------|---------|
| `release_new/run_training.py` | Main training launcher |
| `scripts/eval_hetero_agents_over_seed_configs.py` | Cross-domain evaluation |
| `scripts/analyze_objective_correlation_per_case.py` | Correlation analysis |
| `scripts/plot_state_space_pca_tsne.py` | State space visualization |

### Data Files

| Path | Description |
|------|-------------|
| `release_new/data/host_specs_AL.json` | Aligned host configuration |
| `release_new/data/host_specs_NAL.json` | Non-aligned host configuration |
| `release_new/data/host_specs_homoPower.json` | Homogeneous power configuration |
| `release_new/data/host_specs_homoSpeed.json` | Homogeneous speed configuration |
| `release_new/data/rl_configs/train_long_cp_p08_seeds.json` | LongCP training seeds |
| `release_new/data/rl_configs/train_wide_p005_seeds.json` | Wide training seeds |

### Configuration Files

Training configs for all 8 specialists are in `release_new/configs/`:
- `train_longcp_{aligned,not_aligned,homopower,homospeed}.yaml`
- `train_wide_{aligned,not_aligned,homopower,homospeed}.yaml`

## Experimental Setup

### Host Configurations

We evaluate on 4 host regimes:

1. **AL (Aligned)**: Fast VMs are energy-efficient (positive speed-power correlation)
2. **NAL (Non-Aligned)**: Fast VMs are power-hungry (negative speed-power correlation)
3. **HP (Homogeneous Power)**: All VMs have same power, different speeds
4. **HS (Homogeneous Speed)**: All VMs have same speed, different power

### DAG Topologies

We train specialists on 2 contrasting topologies:

1. **LongCP**: Long critical path (p=0.8 edge probability)
2. **Wide**: Wide parallel structure (p=0.05 edge probability)

### Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Architecture**: GIN (Graph Isomorphism Network)
- **Total timesteps**: 5,000,000 per specialist
- **Parallel environments**: 8
- **Learning rate**: 2.5e-4
- **Entropy coefficient**: 0.01
- **Training time**: ~6-12 hours per specialist on RTX 3090

### Evaluation Protocol

- **Seeds**: 100 evaluation seeds per domain
- **Repeats**: 1 episode per seed
- **Metrics**: Makespan, Active Energy, Policy Entropy
- **Cross-evaluation**: Each specialist tested on both LongCP and Wide domains

## Reproducing Paper Results

### Full Experimental Pipeline

```bash
# 1. Train all 8 specialists (can be parallelized)
cd release_new
bash train_all_specialists.sh

# 2. Evaluate all specialists on all host configs
bash eval_new_checkpoints_all_cases.sh

# 3. Generate correlation plots
cd ..
python scripts/analyze_objective_correlation_per_case.py

# 4. Generate state space visualizations
python scripts/plot_state_space_pca_tsne.py \
    --input_dir runs/state_space_random/AL_case \
    --output_dir runs/state_space_random/AL_case/plots \
    --kde_contours --save_pdf
```

### Expected Results

#### Table: Cross-Domain Performance

| Host | Agent  | Eval Domain | Makespan | Energy (10^7 J) |
|------|--------|-------------|----------|-----------------|
| HS   | LongCP | LongCP      | 2.08     | 39.09           |
| HS   | Wide   | LongCP      | 2.08     | 51.23           |
| HS   | LongCP | Wide        | 0.70     | 45.03           |
| HS   | Wide   | Wide        | 0.70     | 56.87           |
| HP   | LongCP | LongCP      | 3.71     | 21.76           |
| HP   | Wide   | LongCP      | 3.61     | 21.76           |
| HP   | LongCP | Wide        | 1.32     | 24.34           |
| HP   | Wide   | Wide        | 1.29     | 24.35           |
| AL   | LongCP | LongCP      | 3.23     | 39.37           |
| AL   | Wide   | LongCP      | 4.49     | 57.05           |
| AL   | LongCP | Wide        | 1.17     | 45.25           |
| AL   | Wide   | Wide        | 1.54     | 64.25           |
| NA   | LongCP | LongCP      | 4.05     | 35.82           |
| NA   | Wide   | LongCP      | 3.08     | 51.32           |
| NA   | LongCP | Wide        | 1.42     | 41.40           |
| NA   | Wide   | Wide        | 1.09     | 56.87           |

**Key Findings:**
- **HS**: LongCP wins on energy (same makespan)
- **HP**: Wide wins on makespan (same energy)
- **AL**: LongCP dominates on both objectives
- **NA**: True tradeoff (Wide faster, LongCP more efficient)

## Project Structure

```
.
├── README_REPRODUCIBILITY.md        # This file
├── REPRODUCIBILITY.md               # Detailed reproducibility guide
├── requirements.txt                 # Python dependencies
├── release_new/
│   ├── run_training.py             # Training launcher
│   ├── train_all_specialists.sh    # Train all 8 agents
│   ├── eval_new_checkpoints_all_cases.sh  # Evaluate all
│   ├── configs/                    # Training configurations
│   ├── data/
│   │   ├── host_specs_*.json       # Host configurations
│   │   └── rl_configs/             # Seed configurations
│   └── logs/                       # Training outputs (created)
├── scripts/
│   ├── eval_hetero_agents_over_seed_configs.py
│   ├── analyze_objective_correlation_per_case.py
│   └── plot_state_space_pca_tsne.py
└── scheduler/                      # Core library (private)
    ├── rl_model/                   # RL agent implementations
    ├── dataset_generator/          # Dataset generation
    └── ...
```

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB+ VRAM (for training)
- Storage: 50GB

### Recommended
- CPU: 8+ cores (AMD Ryzen 9 or Intel i9)
- RAM: 32GB+
- GPU: NVIDIA RTX 3090 (24GB VRAM) or better
- Storage: 100GB SSD

## Software Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU training)
- **Conda**: Recommended for environment management

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  num_envs: 4  # Reduce from 8
  num_steps: 64  # Reduce from 128
```

### Import Errors

Add project root to PYTHONPATH:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Slow Training

- Verify GPU is being used: `torch.cuda.is_available()`
- Reduce evaluation frequency: `test_every_iters: 100` → `200`
- Use multiple GPUs: `CUDA_VISIBLE_DEVICES=0,1`

## Advanced Usage

### Custom Host Configuration

Create a new host specs JSON:
```json
{
  "vm_types": [
    {
      "name": "custom_vm_1",
      "cpu_cores": 4,
      "cpu_speed_mips": 2500,
      "memory_mb": 8192,
      "power_idle_watt": 100,
      "power_peak_watt": 250
    }
  ]
}
```

### Custom Training Configuration

Modify YAML config:
```yaml
experiment:
  name: "my_custom_experiment"
  seed: 123

training:
  total_timesteps: 10000000  # Longer training
  learning_rate: 1e-4        # Different LR

domain:
  host_specs_file: "data/my_custom_hosts.json"
  edge_probability: 0.5      # Different topology
```

## Citation

```bibtex
@article{yourpaper2024,
  title={Structure-Aware Deep Reinforcement Learning for Heterogeneous Cloud Scheduling},
  author={Your Name and Collaborators},
  journal={Transactions on Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=xxxxx}
}
```

## License

This code is provided for research purposes only. Commercial use requires permission.

See [LICENSE](LICENSE) for details.

## Contact

- **Issues**: Open a GitHub issue
- **Email**: [your-email@domain.com]
- **Paper**: [OpenReview link]

## Acknowledgments

This work was supported by [funding sources].

---

**Note**: This is a private repository. Do not share or redistribute without permission.
