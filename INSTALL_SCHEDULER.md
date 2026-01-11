# Installing the Scheduler Library

The `scheduler` library is required to run the training and evaluation scripts but is not included in this repository due to copyright restrictions.

## Option 1: Install from Private Repository (Recommended)

If you have access to the private scheduler repository:

```bash
# Via pip from private repo
pip install git+https://github.com/YOUR_USERNAME/scheduler-private.git

# Or via SSH
pip install git+ssh://git@github.com/YOUR_USERNAME/scheduler-private.git
```

## Option 2: Install from Local Copy

If you have a local copy of the scheduler:

```bash
# Install in development mode
cd /path/to/scheduler
pip install -e .

# Or install directly
pip install /path/to/scheduler
```

## Option 3: Add to PYTHONPATH

If you don't want to install:

```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc)
export PYTHONPATH="/path/to/scheduler:$PYTHONPATH"

# Or set before running scripts
export PYTHONPATH="/path/to/parent/of/scheduler:$PYTHONPATH"
python release_new/run_training.py --config ...
```

## Verify Installation

```bash
python -c "import scheduler; print('Scheduler installed successfully')"
python -c "from scheduler.rl_model.ablation_gnn_traj_main import Args; print('RL model available')"
```

## For Reviewers

If you are a reviewer and need access to the scheduler library, please contact the authors via the submission system.

## Structure

The scheduler library should provide:
- `scheduler.rl_model.ablation_gnn_traj_main` - Main training entry point
- `scheduler.tools.eval_agents` - Agent evaluation utilities
- `scheduler.tools.eval_heuristics` - Heuristic baseline evaluation
- `scheduler.dataset_generator` - DAG dataset generation

Once installed, all scripts in this repository will work normally.
