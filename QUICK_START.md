# Quick Start Guide

This guide will help you get started with the cloud scheduling reproducibility package.

## Prerequisites

- Python 3.10+
- Git
- Access to the private `scheduler` library

## Step 1: Clone the Repository

```bash
git clone https://github.com/GeorgeAbda/dag-scheduling-gnn.git
cd dag-scheduling-gnn
```

## Step 2: Install Dependencies

```bash
pip install -r release_new/requirements.txt
```

## Step 3: Install Scheduler Library

The `scheduler` library is required but not included in this repository. You have three options:

### Option A: Set PYTHONPATH (Recommended for Testing)

If you have the scheduler library locally:

```bash
export PYTHONPATH="/path/to/parent/of/scheduler:$PYTHONPATH"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Option B: Install from Private Repository

```bash
pip install git+https://github.com/YOUR_USERNAME/scheduler-private.git
```

### Option C: Install Locally

```bash
cd /path/to/scheduler
pip install -e .
```

## Step 4: Verify Installation

```bash
python -c "import scheduler; print('Scheduler installed successfully')"
```

## Step 5: Run Your First Experiment

### Generate Training Configuration

```bash
cd release_new
bash 1_generate_training_config.sh longcp 10
```

### Train an Agent

```bash
bash 2_train_agent.sh configs/train_longcp_aligned.yaml
```

### Evaluate Heuristics

```bash
bash 3_evaluate_heuristics.sh
```

### Evaluate Trained Agents

```bash
bash 4_evaluate_trained_agents.sh \
    --longcp-checkpoint logs/longcp_aligned/ablation/per_variant/hetero/hetero_best.pt \
    --wide-checkpoint logs/wide_aligned/ablation/per_variant/hetero/hetero_best.pt \
    --host-specs data/host_specs_AL.json \
    --output-dir evals_test
```

## Common Issues

### Import Error: No module named 'scheduler'

**Solution**: Make sure you've set PYTHONPATH or installed the scheduler library.

```bash
export PYTHONPATH="/path/to/parent/of/scheduler:$PYTHONPATH"
```

### FileNotFoundError: host_specs.json

**Solution**: Use the correct host specs file path in your config:

```yaml
domain:
  host_specs_file: "data/host_specs_AL.json"
```

### Permission Denied

**Solution**: Make scripts executable:

```bash
chmod +x release_new/*.sh
```

## Next Steps

- See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed reproduction instructions
- See [SETUP.md](SETUP.md) for advanced configuration
- See [INSTALL_SCHEDULER.md](INSTALL_SCHEDULER.md) for scheduler installation details

## Getting Help

For issues or questions:
1. Check the documentation in this repository
2. Contact the authors via the submission system
3. Review the paper for methodology details
