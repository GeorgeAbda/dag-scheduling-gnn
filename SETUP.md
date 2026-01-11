# Setup Guide

This guide walks through setting up the environment and running your first experiment.

## Installation

### Step 1: Clone Repository

```bash
git clone [repository-url]
cd to-github
```

### Step 2: Create Environment

Using Conda (recommended):

```bash
conda create -n cloud_sched python=3.10
conda activate cloud_sched
```

Or using venv:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

Expected output:
```
PyTorch: 2.4.1
CUDA available: True
PyG: 2.6.1
```

## Quick Test

### Test Training (Short Run)

Create a test config `release_new/configs/test_quick.yaml`:

```yaml
experiment:
  name: "test_quick"
  seed: 42
  output_dir: "logs/test_quick"
  device: "cuda"

training:
  total_timesteps: 10000  # Very short for testing
  learning_rate: 0.00025
  num_envs: 2
  num_steps: 128
  gamma: 0.99
  gae_lambda: 0.95
  num_minibatches: 4
  update_epochs: 4
  clip_coef: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  anneal_lr: true
  norm_adv: true
  clip_vloss: true

evaluation:
  test_every_iters: 50
  test_num_episodes: 2

domain:
  host_specs_file: "data/host_specs_AL.json"
  edge_probability: 0.8
  min_tasks: 20
  max_tasks: 40
  num_hosts: 10
  num_vms: 10
```

Run test:

```bash
cd release_new
python run_training.py --config configs/test_quick.yaml
```

This should complete in ~2-5 minutes and verify your setup works.

### Test Evaluation

After the quick training completes:

```bash
cd ..
python scripts/eval_hetero_agents_over_seed_configs.py \
    --checkpoint_path release_new/logs/test_quick/ablation/per_variant/hetero/hetero_best.pt \
    --host_specs_path release_new/data/host_specs_AL.json \
    --seed_config_path release_new/data/rl_configs/train_long_cp_p08_seeds.json \
    --output_dir release_new/test_eval \
    --device cuda \
    --num_repeats 1 \
    --max_seeds 5
```

This evaluates on just 5 seeds for a quick test (~1-2 minutes).

## Directory Structure After Setup

```
to-github/
├── release_new/
│   ├── logs/                    # Created after training
│   │   └── test_quick/          # Test run output
│   ├── evals_new_ckpts/         # Created after evaluation
│   │   └── test_eval/           # Test evaluation output
│   ├── configs/
│   ├── data/
│   └── run_training.py
├── scripts/
├── scheduler/                   # Core library
├── requirements.txt
├── README_REPRODUCIBILITY.md
└── REPRODUCIBILITY.md
```

## Common Issues

### Issue: CUDA not available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Use CPU (slower): Set `device: "cpu"` in config

### Issue: Import errors for `scheduler`

**Symptom**: `ModuleNotFoundError: No module named 'scheduler'`

**Solution**: Add project root to PYTHONPATH:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Or in Python:
```python
import sys
sys.path.insert(0, '/path/to/to-github')
```

### Issue: Out of memory during training

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `num_envs` in config (e.g., 8 → 4)
2. Reduce `num_steps` (e.g., 128 → 64)
3. Increase `num_minibatches` (e.g., 4 → 8)
4. Use smaller batch size

### Issue: Slow training on CPU

**Symptom**: Training takes hours for small number of steps

**Solution**: Use GPU if available, or reduce problem size:
- Reduce `num_envs`
- Reduce `max_tasks` in domain config
- Reduce `num_vms` and `num_hosts`

## Next Steps

After successful setup:

1. **Read the full guide**: See `REPRODUCIBILITY.md` for detailed instructions
2. **Train specialists**: Run `bash release_new/train_all_specialists.sh`
3. **Evaluate**: Run `bash release_new/eval_new_checkpoints_all_cases.sh`
4. **Analyze**: Generate plots and tables

## Environment Variables

Optional environment variables:

```bash
# Force CPU (even if CUDA available)
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Set host specs path (overrides config)
export HOST_SPECS_PATH=/path/to/host_specs.json

# Disable wandb logging
export WANDB_MODE=disabled
```

## Verification Checklist

Before running full experiments, verify:

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] CUDA available (if using GPU)
- [ ] Quick test training completes successfully
- [ ] Quick test evaluation produces output CSV
- [ ] No import errors
- [ ] Sufficient disk space (50GB+)
- [ ] Sufficient RAM (16GB+)

## Getting Help

If you encounter issues:

1. Check this guide and `REPRODUCIBILITY.md`
2. Search existing GitHub issues
3. Open a new issue with:
   - Error message
   - System info (`python --version`, `nvidia-smi`)
   - Steps to reproduce
   - Config file used

## Performance Benchmarks

Expected performance on reference hardware (RTX 3090):

| Task | Time |
|------|------|
| Quick test (10k steps) | ~2-5 min |
| Full training (5M steps) | ~6-12 hours |
| Single evaluation (100 seeds) | ~10-30 min |
| Full cross-evaluation (64 runs) | ~5-10 hours |

Your results may vary based on hardware.
