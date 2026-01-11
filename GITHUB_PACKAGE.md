# GitHub Repository Package Summary

This document outlines what to include in your private GitHub repository for reproducibility.

## Repository Configuration

### Repository Settings

1. **Visibility**: Private
2. **Name**: `cloud-scheduling-structure-aware-rl`
3. **Description**: "Code for reproducing experiments in 'Structure-Aware Deep RL for Heterogeneous Cloud Scheduling' (TMLR 2024)"
4. **Topics**: `reinforcement-learning`, `cloud-computing`, `scheduling`, `graph-neural-networks`, `reproducibility`

### Branch Protection

- Main branch: Protected
- Require pull request reviews
- Require status checks

## Files to Include

### Essential Documentation

```
✓ README_REPRODUCIBILITY.md    # Main README
✓ REPRODUCIBILITY.md            # Detailed guide
✓ SETUP.md                      # Setup instructions
✓ LICENSE                       # Research license
✓ .gitignore                    # Ignore patterns
✓ requirements.txt              # Dependencies
```

### Core Scripts

```
✓ release_new/
  ✓ run_training.py             # Training launcher
  ✓ train_all_specialists.sh    # Batch training
  ✓ eval_new_checkpoints_all_cases.sh  # Batch evaluation
  
✓ scripts/
  ✓ eval_hetero_agents_over_seed_configs.py
  ✓ analyze_objective_correlation_per_case.py
  ✓ plot_state_space_pca_tsne.py
  ✓ random_agents_state_distribution.py
```

### Configuration Files

```
✓ release_new/configs/
  ✓ train_longcp_aligned.yaml
  ✓ train_longcp_not_aligned.yaml
  ✓ train_longcp_homopower.yaml
  ✓ train_longcp_homospeed.yaml
  ✓ train_wide_aligned.yaml
  ✓ train_wide_not_aligned.yaml
  ✓ train_wide_homopower.yaml
  ✓ train_wide_homospeed.yaml
```

### Data Files

```
✓ release_new/data/
  ✓ host_specs_AL.json
  ✓ host_specs_NAL.json
  ✓ host_specs_homoPower.json
  ✓ host_specs_homoSpeed.json
  ✓ rl_configs/
    ✓ train_long_cp_p08_seeds.json
    ✓ train_wide_p005_seeds.json
```

### Library Code

```
✓ scheduler/                    # Your core library
  ✓ rl_model/
  ✓ dataset_generator/
  ✓ (other modules as needed)
```

## Files to Exclude

### Large Files (use .gitignore)

```
✗ logs/                         # Training outputs (too large)
✗ wandb/                        # Weights & Biases logs
✗ *.pt, *.pth                   # Model checkpoints (too large)
✗ *.npz, *.npy                  # Numpy arrays
✗ evals_*/                      # Evaluation outputs
✗ runs/                         # Experiment runs
```

### Temporary Files

```
✗ __pycache__/
✗ *.pyc
✗ .ipynb_checkpoints/
✗ .DS_Store
✗ *.log
```

## Optional: Provide Pre-trained Checkpoints

If you want to provide pre-trained checkpoints without including them in the repo:

### Option 1: External Hosting

Host checkpoints on:
- Google Drive
- Zenodo (recommended for research)
- Institutional storage
- AWS S3 (with access control)

Create `CHECKPOINTS.md`:

```markdown
# Pre-trained Checkpoints

Download pre-trained checkpoints from: [link]

## Available Checkpoints

| Specialist | Host Config | Download Link | Size | MD5 |
|------------|-------------|---------------|------|-----|
| LongCP     | AL          | [link]        | 45MB | xxx |
| LongCP     | NAL         | [link]        | 45MB | xxx |
| ...        | ...         | ...           | ...  | ... |

## Usage

1. Download checkpoint
2. Place in `release_new/logs/[agent_name]/ablation/per_variant/hetero/`
3. Run evaluation
```

### Option 2: Git LFS (Large File Storage)

For GitHub:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
```

**Note**: GitHub LFS has storage limits (1GB free, then paid).

## Repository Structure

```
cloud-scheduling-structure-aware-rl/
├── README_REPRODUCIBILITY.md
├── REPRODUCIBILITY.md
├── SETUP.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── release_new/
│   ├── run_training.py
│   ├── train_all_specialists.sh
│   ├── eval_new_checkpoints_all_cases.sh
│   ├── configs/
│   │   ├── train_longcp_aligned.yaml
│   │   ├── train_longcp_not_aligned.yaml
│   │   ├── train_longcp_homopower.yaml
│   │   ├── train_longcp_homospeed.yaml
│   │   ├── train_wide_aligned.yaml
│   │   ├── train_wide_not_aligned.yaml
│   │   ├── train_wide_homopower.yaml
│   │   └── train_wide_homospeed.yaml
│   ├── data/
│   │   ├── host_specs_AL.json
│   │   ├── host_specs_NAL.json
│   │   ├── host_specs_homoPower.json
│   │   ├── host_specs_homoSpeed.json
│   │   └── rl_configs/
│   │       ├── train_long_cp_p08_seeds.json
│   │       └── train_wide_p005_seeds.json
│   ├── logs/.gitkeep
│   └── evals_new_ckpts/.gitkeep
│
├── scripts/
│   ├── eval_hetero_agents_over_seed_configs.py
│   ├── analyze_objective_correlation_per_case.py
│   ├── plot_state_space_pca_tsne.py
│   └── random_agents_state_distribution.py
│
└── scheduler/
    ├── __init__.py
    ├── rl_model/
    ├── dataset_generator/
    └── (other modules)
```

## GitHub Actions (Optional)

Create `.github/workflows/test.yml` for CI:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run quick test
      run: |
        cd release_new
        python run_training.py --config configs/test_quick.yaml
```

## README Badges

Add to top of `README_REPRODUCIBILITY.md`:

```markdown
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## Access Control

### Collaborators

Add reviewers/collaborators:
1. Go to Settings → Collaborators
2. Add by GitHub username or email
3. Set permissions (Read, Write, Admin)

### Access Levels

- **Read**: Can view and clone
- **Write**: Can push changes
- **Admin**: Full control

## Release Strategy

### Option 1: Private During Review

Keep private until paper acceptance, then:
1. Make public after acceptance
2. Create release with DOI (Zenodo)
3. Link from paper

### Option 2: Private Permanently

Keep private, provide access:
1. Add reviewers as collaborators
2. Provide access token for anonymous review
3. Archive on Zenodo with restricted access

### Option 3: Anonymous Review

For double-blind review:
1. Create anonymous repository
2. Remove author information
3. Use anonymous GitHub account
4. Provide link in paper submission

## Zenodo Integration

Archive your code with DOI:

1. Link GitHub to Zenodo: https://zenodo.org/account/settings/github/
2. Create release on GitHub
3. Zenodo automatically archives
4. Get DOI for citation

## License

Create `LICENSE` file:

```
Research License

Copyright (c) 2024 [Your Institution]

This code is provided for research and educational purposes only.

Permission is granted to use, copy, and modify this software for 
non-commercial research purposes, provided that:

1. The above copyright notice and this permission notice appear in 
   all copies.
2. Any publications resulting from use of this software cite the 
   original paper.
3. Commercial use requires explicit written permission from the 
   copyright holder.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

## Checklist Before Publishing

- [ ] All sensitive information removed (API keys, passwords, etc.)
- [ ] Author information anonymized (if needed for review)
- [ ] Documentation complete and clear
- [ ] All scripts tested and working
- [ ] Requirements.txt up to date
- [ ] .gitignore properly configured
- [ ] Large files excluded or in LFS
- [ ] License file included
- [ ] README has clear instructions
- [ ] Contact information provided
- [ ] Citation information included
- [ ] Links to paper/preprint added

## Post-Publication

After paper acceptance:

1. **Update README**: Add paper link, citation
2. **Create release**: Tag version (e.g., v1.0.0)
3. **Archive on Zenodo**: Get DOI
4. **Announce**: Twitter, blog post, etc.
5. **Monitor issues**: Respond to questions
6. **Maintain**: Fix bugs, update dependencies

## Example Repository Description

```
Structure-Aware Deep RL for Heterogeneous Cloud Scheduling

This repository contains the code for reproducing experiments in our 
TMLR 2024 paper. We provide:

✓ Training scripts for 8 specialist agents
✓ Evaluation across 4 host configurations
✓ Complete configuration files
✓ Detailed reproducibility guide

Key Results:
- LongCP specialist dominates in Aligned regime
- Wide specialist achieves lower makespan in Non-Aligned regime
- DAG structure fundamentally shapes learned policies

Paper: [link]
Preprint: [arXiv link]
```

## Contact Information

Include in README:

```markdown
## Contact

- **Issues**: Open a GitHub issue
- **Email**: [your-email@institution.edu]
- **Paper**: [OpenReview/arXiv link]
- **Institution**: [Your Lab/Department]
```

## Summary

**Minimum Required Files:**
1. README_REPRODUCIBILITY.md
2. REPRODUCIBILITY.md
3. SETUP.md
4. requirements.txt
5. .gitignore
6. LICENSE
7. Training script (run_training.py)
8. Evaluation script (eval_hetero_agents_over_seed_configs.py)
9. All config files
10. All data files (host specs, seed configs)
11. Core library code (scheduler/)

**Total Repository Size**: ~50-100MB (without checkpoints)

**With Checkpoints**: ~500MB-1GB (use LFS or external hosting)
