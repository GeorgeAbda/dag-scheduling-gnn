# Reproducibility Guide

This document provides instructions for reproducing the training and evaluation results presented in our paper.

## Repository Structure

```
.
├── release_new/
│   ├── run_training.py              # Training launcher script
│   ├── configs/                     # Training configurations
│   │   ├── train_longcp_aligned.yaml
│   │   ├── train_longcp_not_aligned.yaml
│   │   ├── train_longcp_homopower.yaml
│   │   ├── train_longcp_homospeed.yaml
│   │   ├── train_wide_aligned.yaml
│   │   ├── train_wide_not_aligned.yaml
│   │   ├── train_wide_homopower.yaml
│   │   └── train_wide_homospeed.yaml
│   └── data/
│       ├── host_specs_AL.json       # Aligned host configuration
│       ├── host_specs_NAL.json      # Non-aligned host configuration
│       ├── host_specs_homoPower.json
│       ├── host_specs_homoSpeed.json
│       └── rl_configs/
│           ├── train_long_cp_p08_seeds.json
│           └── train_wide_p005_seeds.json
├── scripts/
│   └── eval_hetero_agents_over_seed_configs.py  # Evaluation script
└── requirements.txt                 # Python dependencies
```

## Prerequisites

### System Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Software Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- PyTorch Geometric
- NumPy, Pandas
- Gymnasium
- tyro, pyyaml
- tqdm

## Training

### Overview

We train 8 specialist agents across 4 host configurations (AL, NAL, HP, HS) and 2 DAG topologies (LongCP, Wide).

### Single Agent Training

Train a single specialist using the configuration files:

```bash
python release_new/run_training.py --config release_new/configs/train_longcp_aligned.yaml
```

### Training All Specialists

Use the provided shell script to train all 8 specialists:

```bash
cd release_new
bash train_all_specialists.sh
```

Or train them individually:

```bash
# LongCP specialists
python run_training.py --config configs/train_longcp_aligned.yaml
python run_training.py --config configs/train_longcp_not_aligned.yaml
python run_training.py --config configs/train_longcp_homopower.yaml
python run_training.py --config configs/train_longcp_homospeed.yaml

# Wide specialists
python run_training.py --config configs/train_wide_aligned.yaml
python run_training.py --config configs/train_wide_not_aligned.yaml
python run_training.py --config configs/train_wide_homopower.yaml
python run_training.py --config configs/train_wide_homospeed.yaml
```

### Training Configuration

Each YAML config specifies:
- **Experiment name**: Identifies the run
- **Seed**: For reproducibility (default: 42)
- **Output directory**: Where checkpoints are saved
- **Device**: `cuda` or `cpu`
- **Training hyperparameters**:
  - Total timesteps: 5,000,000
  - Learning rate: 2.5e-4
  - Number of parallel environments: 8
  - GAE lambda: 0.95
  - Entropy coefficient: 0.01
- **Domain configuration**:
  - Host specs file path
  - DAG topology (edge probability)
  - Task count range
  - VM and host counts

### Expected Training Time

- **Per specialist**: ~6-12 hours on a single GPU (NVIDIA RTX 3090 or equivalent)
- **All 8 specialists**: ~48-96 hours (can be parallelized across GPUs)

### Checkpoints

Training produces checkpoints in the output directory:
- `hetero_best.pt`: Best checkpoint by validation performance
- `hetero_best_return.pt`: Best checkpoint by episodic return
- Pareto front checkpoints: `hetero_mean_pf_*.pt`, `hetero_cvar_pf_*.pt`, `hetero_worst_pf_*.pt`

## Evaluation

### Overview

Evaluation tests each specialist on both LongCP and Wide domains across all host configurations.

### Single Evaluation Run

Evaluate a trained checkpoint:

```bash
python scripts/eval_hetero_agents_over_seed_configs.py \
    --checkpoint_path release_new/logs/longcp_aligned/ablation/per_variant/hetero/hetero_best.pt \
    --host_specs_path release_new/data/host_specs_AL.json \
    --seed_config_path release_new/data/rl_configs/train_long_cp_p08_seeds.json \
    --output_dir release_new/evals_new_ckpts/AL \
    --device cuda \
    --num_repeats 1
```

### Evaluate All Cases

Use the provided evaluation script:

```bash
cd release_new
bash eval_new_checkpoints_all_cases.sh
```

This evaluates all 8 specialists on both evaluation domains (LongCP and Wide) across all 4 host configurations.

### Evaluation Parameters

- `--checkpoint_path`: Path to trained agent checkpoint
- `--host_specs_path`: Host configuration JSON file
- `--seed_config_path`: Seed configuration with dataset parameters
- `--output_dir`: Directory for evaluation results
- `--device`: `cuda` or `cpu`
- `--num_repeats`: Number of evaluation runs per seed (default: 1)

### Expected Evaluation Time

- **Per agent-domain-host combination**: ~10-30 minutes
- **Full cross-evaluation (8 agents × 2 domains × 4 hosts)**: ~5-10 hours

### Evaluation Outputs

For each evaluation run, the script generates:

1. **Summary CSV** (`hetero_eval.summary.csv`):
   ```csv
   agent_train_domain,eval_domain,seeds,mean_makespan,mean_energy_active,mean_entropy
   long_cp,long_cp,100.0,3.23,39.37,2.26
   long_cp,wide,100.0,1.17,45.25,2.50
   ```

2. **Per-seed CSV** (`hetero_eval.per_seed.csv`):
   - Detailed results for each evaluation seed
   - Makespan, energy, entropy per episode

3. **Metadata JSON** (`hetero_eval.metadata.json`):
   - Checkpoint path
   - Host specs
   - Evaluation timestamp
   - System information

## Reproducing Paper Results

### Table: Cross-Domain Evaluation

To reproduce Table X in the paper:

1. Train all 8 specialists (or use provided checkpoints)
2. Run full cross-evaluation:
   ```bash
   cd release_new
   bash eval_new_checkpoints_all_cases.sh
   ```
3. Results will be in `release_new/evals_new_ckpts/{AL,NAL,HP,HS}/hetero_eval.summary.csv`

Expected results (mean ± std):

| Host | Agent | Eval Domain | Makespan | Energy (10^7 J) |
|------|-------|-------------|----------|-----------------|
| AL   | LongCP| LongCP      | 3.23     | 39.37           |
| AL   | LongCP| Wide        | 1.17     | 45.25           |
| AL   | Wide  | LongCP      | 4.49     | 57.05           |
| AL   | Wide  | Wide        | 1.54     | 64.25           |
| ...  | ...   | ...         | ...      | ...             |

### Figure: Objective Correlation

To reproduce the correlation analysis:

```bash
python scripts/analyze_objective_correlation_per_case.py
```

Outputs:
- `logs/objective_correlation_analysis/objective_correlation_AL.png`
- `logs/objective_correlation_analysis/objective_correlation_NAL.png`

### Figure: State Space Visualization

To reproduce t-SNE state space plots:

```bash
# First, collect random agent states (if not already done)
python scripts/random_agents_state_distribution.py \
    --output_dir runs/state_space_random/AL_case \
    --host_specs release_new/data/host_specs_AL.json

# Then generate t-SNE plots
python scripts/plot_state_space_pca_tsne.py \
    --input_dir runs/state_space_random/AL_case \
    --output_dir runs/state_space_random/AL_case/plots \
    --kde_contours --save_pdf
```

## Seed Configuration

All experiments use fixed seeds for reproducibility:

- **Training seed**: 42 (specified in config files)
- **Evaluation seeds**: 100 seeds per domain (specified in `rl_configs/*.json`)
- **Random state**: Set via `torch.manual_seed()`, `np.random.seed()`, `random.seed()`

## Hardware and Environment

Our experiments were conducted on:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: AMD Ryzen 9 5950X
- **RAM**: 64GB
- **OS**: Ubuntu 20.04 / macOS
- **Python**: 3.10.12
- **PyTorch**: 2.0.1
- **CUDA**: 11.8

Results may vary slightly on different hardware due to floating-point precision differences.

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors during training:
- Reduce `num_envs` in config (e.g., from 8 to 4)
- Reduce `num_steps` (e.g., from 128 to 64)
- Use smaller batch sizes via `num_minibatches`

### Slow Training

- Ensure CUDA is available: `torch.cuda.is_available()`
- Use multiple GPUs: Set `CUDA_VISIBLE_DEVICES`
- Reduce `test_every_iters` to skip frequent validation

### Import Errors

Ensure the project root is in `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/DaDiL/to-github:$PYTHONPATH
```

## Citation

If you use this code or reproduce our results, please cite:

```bibtex
@article{yourpaper2024,
  title={Structure-Aware Deep Reinforcement Learning for Heterogeneous Cloud Scheduling},
  author={Your Name et al.},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## License

This code is provided for research purposes only. See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].
