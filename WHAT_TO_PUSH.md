# What to Push to GitHub

This document lists exactly what should be included in the public repository.

## ✅ Files to Include

### Documentation (Root)
```
✓ README_REPRODUCIBILITY.md
✓ REPRODUCIBILITY.md
✓ SETUP.md
✓ INSTALL_SCHEDULER.md
✓ .gitignore
✓ requirements.txt
✓ LICENSE (create if needed)
```

### Scripts in `release_new/`
```
✓ 1_generate_training_config.sh
✓ 2_train_agent.sh
✓ 3_evaluate_heuristics.sh
✓ 4_evaluate_trained_agents.sh
✓ run_training.py
✓ eval_agents.py
✓ eval_heuristics.py
✓ generate_training_config.py
✓ train_all_specialists.sh
✓ eval_new_checkpoints_all_cases.sh (if exists)
```

### Configuration Files in `release_new/configs/`
```
✓ train_longcp_aligned.yaml
✓ train_longcp_not_aligned.yaml
✓ train_longcp_homopower.yaml
✓ train_longcp_homospeed.yaml
✓ train_wide_aligned.yaml
✓ train_wide_not_aligned.yaml
✓ train_wide_homopower.yaml
✓ train_wide_homospeed.yaml
```

### Data Files in `release_new/data/`
```
✓ host_specs_AL.json
✓ host_specs_NAL.json
✓ host_specs_homoPower.json
✓ host_specs_homoSpeed.json
✓ rl_configs/train_long_cp_p08_seeds.json
✓ rl_configs/train_wide_p005_seeds.json
```

### Analysis Scripts in `scripts/`
```
✓ eval_hetero_agents_over_seed_configs.py
✓ analyze_objective_correlation_per_case.py
✓ plot_state_space_pca_tsne.py
✓ random_agents_state_distribution.py (if needed)
```

### Scheduler Library (Minimal)
```
✓ scheduler/README.md (explains it's private)
✓ scheduler/__init__.py (empty or minimal)
```

## ❌ Files to Exclude (Already in .gitignore)

### Private Code
```
✗ scheduler/**/*.py (all Python files in scheduler/)
✗ scheduler/**/*.pyc
```

### Generated/Output Files
```
✗ logs/
✗ wandb/
✗ evals_*/
✗ runs/
✗ *.pt, *.pth (checkpoints)
✗ *.npz, *.npy
```

### Temporary Files
```
✗ __pycache__/
✗ *.pyc
✗ .DS_Store
✗ *.log
```

## 📋 Pre-Push Checklist

Before pushing to GitHub:

- [ ] Verify `.gitignore` excludes `scheduler/**/*.py`
- [ ] Ensure `scheduler/README.md` exists
- [ ] Update `README_REPRODUCIBILITY.md` with private library note
- [ ] All shell scripts are executable (`chmod +x *.sh`)
- [ ] All paths in scripts are relative (not absolute)
- [ ] `requirements.txt` is up to date
- [ ] No hardcoded API keys or secrets
- [ ] No author-identifying information (if anonymous review)

## 🔍 Verify What Will Be Pushed

```bash
cd /Users/anashattay/Documents/GitHub/DaDiL/to-github

# Check git status
git status

# Dry run to see what would be added
git add -n .

# Check if scheduler code is excluded
git ls-files scheduler/

# Should only show:
# scheduler/README.md
# scheduler/__init__.py (if exists)
```

## 📦 Push Commands

```bash
# Initialize git (if not already)
git init

# Add all files (respecting .gitignore)
git add .

# Check what's staged
git status

# Commit
git commit -m "Initial commit: Reproducibility package for TMLR submission"

# Add remote
git remote add origin git@github.com:username/repo-name.git

# Push
git push -u origin main
```

## 🔐 For Reviewers/Collaborators

Add this to your README:

```markdown
## Access to Private Code

The `scheduler/` library is proprietary. For access:

1. **Reviewers**: Contact authors via submission system
2. **Collaborators**: Request access at [email]
3. **After acceptance**: Code will be made available upon request

The library provides:
- RL model implementations
- Dataset generation utilities
- Evaluation tools
```

## Summary

**Total files to push**: ~30-40 files
- Documentation: 5 files
- Scripts: 15+ files
- Configs: 8 files
- Data: 6 files
- Scheduler stub: 2 files

**Repository size**: ~5-10 MB (without scheduler code, checkpoints, or outputs)

This creates a functional repository that:
✅ Shows your methodology
✅ Provides all configurations
✅ Includes data files
✅ Protects proprietary code
✅ Enables reproducibility (with access to scheduler library)
