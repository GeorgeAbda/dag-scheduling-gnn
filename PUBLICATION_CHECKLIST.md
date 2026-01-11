# Publication Checklist for GitHub Repository

Use this checklist to prepare your private GitHub repository for paper submission and reproducibility.

## Pre-Publication Checklist

### Documentation

- [x] `README_REPRODUCIBILITY.md` - Main README with quick start
- [x] `REPRODUCIBILITY.md` - Detailed reproducibility guide
- [x] `SETUP.md` - Setup and installation instructions
- [x] `GITHUB_PACKAGE.md` - Repository packaging guide
- [ ] `LICENSE` - Research license file
- [ ] `CITATION.bib` - BibTeX citation
- [ ] `CHANGELOG.md` - Version history (optional)

### Core Scripts

- [x] `release_new/run_training.py` - Training launcher
- [x] `scripts/eval_hetero_agents_over_seed_configs.py` - Evaluation script
- [x] `release_new/train_all_specialists.sh` - Batch training script
- [x] `release_new/eval_new_checkpoints_all_cases.sh` - Batch evaluation script

### Configuration Files

- [ ] All 8 training configs in `release_new/configs/`
  - [ ] `train_longcp_aligned.yaml`
  - [ ] `train_longcp_not_aligned.yaml`
  - [ ] `train_longcp_homopower.yaml`
  - [ ] `train_longcp_homospeed.yaml`
  - [ ] `train_wide_aligned.yaml`
  - [ ] `train_wide_not_aligned.yaml`
  - [ ] `train_wide_homopower.yaml`
  - [ ] `train_wide_homospeed.yaml`

### Data Files

- [ ] Host specifications
  - [ ] `release_new/data/host_specs_AL.json`
  - [ ] `release_new/data/host_specs_NAL.json`
  - [ ] `release_new/data/host_specs_homoPower.json`
  - [ ] `release_new/data/host_specs_homoSpeed.json`
- [ ] Seed configurations
  - [ ] `release_new/data/rl_configs/train_long_cp_p08_seeds.json`
  - [ ] `release_new/data/rl_configs/train_wide_p005_seeds.json`

### Analysis Scripts

- [x] `scripts/analyze_objective_correlation_per_case.py` - Correlation plots
- [x] `scripts/plot_state_space_pca_tsne.py` - State space visualization
- [ ] `scripts/random_agents_state_distribution.py` - State collection (if needed)

### Repository Configuration

- [x] `.gitignore` - Ignore patterns for large files
- [ ] `.gitattributes` - Git LFS configuration (if using)
- [ ] `requirements.txt` - Python dependencies (already exists)
- [ ] `.github/workflows/` - CI/CD (optional)

### Code Quality

- [ ] Remove all hardcoded paths
- [ ] Remove all API keys/secrets
- [ ] Remove author-identifying information (if anonymous review)
- [ ] Add docstrings to main functions
- [ ] Test all scripts on clean environment
- [ ] Verify all imports work

### Testing

- [ ] Quick training test completes successfully
- [ ] Quick evaluation test produces output
- [ ] All config files are valid YAML
- [ ] All JSON data files are valid
- [ ] Scripts run without errors on fresh install

## Verification Steps

### 1. Clean Environment Test

```bash
# Create fresh environment
conda create -n test_env python=3.10 -y
conda activate test_env

# Clone repo (or use local copy)
cd /path/to/to-github

# Install dependencies
pip install -r requirements.txt

# Test training
cd release_new
python run_training.py --config configs/test_quick.yaml

# Test evaluation
cd ..
python scripts/eval_hetero_agents_over_seed_configs.py \
    --checkpoint_path release_new/logs/test_quick/ablation/per_variant/hetero/hetero_best.pt \
    --host_specs_path release_new/data/host_specs_AL.json \
    --seed_config_path release_new/data/rl_configs/train_long_cp_p08_seeds.json \
    --output_dir release_new/test_eval \
    --device cpu \
    --max_seeds 5
```

### 2. Documentation Review

- [ ] README is clear and complete
- [ ] All commands are copy-pasteable
- [ ] All file paths are correct
- [ ] Expected outputs are documented
- [ ] Hardware requirements are specified
- [ ] Estimated run times are provided

### 3. Reproducibility Check

- [ ] All random seeds are documented
- [ ] All hyperparameters are in configs
- [ ] Dataset generation is deterministic
- [ ] Evaluation protocol is clear
- [ ] Results match paper (within tolerance)

## Repository Size Check

```bash
# Check total size
du -sh .

# Check largest files
find . -type f -size +10M -exec ls -lh {} \;

# Check what's tracked
git ls-files | xargs du -h | sort -h | tail -20
```

**Target size**: <100MB (without checkpoints)

## Pre-trained Checkpoints (Optional)

If providing checkpoints:

### Option A: External Hosting

- [ ] Upload checkpoints to Zenodo/Google Drive
- [ ] Create `CHECKPOINTS.md` with download links
- [ ] Include MD5 checksums
- [ ] Test download and usage

### Option B: Git LFS

- [ ] Install Git LFS
- [ ] Configure `.gitattributes`
- [ ] Track checkpoint files
- [ ] Verify LFS is working

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
git lfs ls-files  # Verify
```

## GitHub Repository Setup

### 1. Create Repository

```bash
# On GitHub: Create new private repository
# Name: cloud-scheduling-structure-aware-rl
# Description: Code for "Structure-Aware Deep RL for Heterogeneous Cloud Scheduling"
# Private: Yes
```

### 2. Push Code

```bash
cd /path/to/to-github
git init
git add .
git commit -m "Initial commit: Reproducibility package for TMLR submission"
git branch -M main
git remote add origin git@github.com:username/cloud-scheduling-structure-aware-rl.git
git push -u origin main
```

### 3. Configure Repository

- [ ] Add description and topics
- [ ] Enable Issues
- [ ] Disable Wiki (or populate it)
- [ ] Set up branch protection
- [ ] Add collaborators (reviewers)

### 4. Create Release (After Acceptance)

```bash
git tag -a v1.0.0 -m "Release for TMLR 2024 paper"
git push origin v1.0.0
```

## Paper Submission

### In Paper

Include these statements:

**Code Availability:**
```
Code for reproducing all experiments is available at:
https://github.com/[username]/cloud-scheduling-structure-aware-rl

The repository includes:
- Training scripts for all 8 specialist agents
- Evaluation scripts for cross-domain testing
- Complete configuration files
- Detailed reproducibility guide

[Optional: Pre-trained checkpoints available at: [link]]
```

**Reproducibility Statement:**
```
All experiments use fixed random seeds (seed=42 for training,
100 fixed seeds for evaluation). Complete hyperparameters are
provided in YAML configuration files. Expected training time
is 6-12 hours per specialist on an NVIDIA RTX 3090 GPU.
```

### In Supplementary Material

- [ ] Link to GitHub repository
- [ ] Brief setup instructions
- [ ] Expected results table
- [ ] Hardware requirements

## Post-Acceptance Tasks

- [ ] Make repository public (if planned)
- [ ] Create Zenodo archive with DOI
- [ ] Update paper with DOI
- [ ] Add paper link to README
- [ ] Create release tag
- [ ] Announce on social media
- [ ] Monitor issues and questions

## Maintenance Plan

### Short-term (0-6 months)

- [ ] Respond to issues within 1 week
- [ ] Fix critical bugs immediately
- [ ] Update documentation as needed
- [ ] Add FAQ section based on questions

### Long-term (6+ months)

- [ ] Update dependencies annually
- [ ] Test with new PyTorch versions
- [ ] Archive if no longer maintained
- [ ] Consider transferring to organization

## Access During Review

### For Anonymous Review

Option 1: Anonymous GitHub account
- Create new GitHub account (no identifying info)
- Upload code
- Share link in paper

Option 2: Access token
- Keep repository private
- Generate read-only access token
- Share token with reviewers

Option 3: Supplementary material
- Include code as ZIP in submission
- Upload to review system

### For Non-Anonymous Review

- Add reviewers as collaborators
- Send invitation via email
- Provide clear instructions

## Final Checks

Before making repository public:

- [ ] All tests pass
- [ ] Documentation is complete
- [ ] No sensitive information
- [ ] License is appropriate
- [ ] Contact information is current
- [ ] Links to paper are correct
- [ ] Citation is formatted correctly
- [ ] Acknowledgments are included

## Estimated Timeline

| Task | Time |
|------|------|
| Documentation | 2-4 hours |
| Code cleanup | 2-3 hours |
| Testing | 1-2 hours |
| Repository setup | 1 hour |
| Review and polish | 1-2 hours |
| **Total** | **7-12 hours** |

## Support Resources

- GitHub Guides: https://guides.github.com/
- Git LFS: https://git-lfs.github.com/
- Zenodo: https://zenodo.org/
- Papers with Code: https://paperswithcode.com/

## Notes

- Keep a backup of everything
- Test on multiple machines if possible
- Get feedback from colleague before publishing
- Consider creating a demo notebook
- Plan for long-term maintenance

---

**Status**: ☐ Not Started | ◐ In Progress | ☑ Complete

Last updated: [Date]
