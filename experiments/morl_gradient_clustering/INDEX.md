# File Index: Gradient-Based Domain Clustering for MORL

This document provides an overview of all files in this implementation.

## 📚 Documentation Files

### **QUICK_START.md** (Start Here!)
Quick start guide to get running in under 5 minutes.
- Installation instructions
- First experiment walkthrough
- Output interpretation
- Troubleshooting tips

### **README.md**
Comprehensive documentation covering:
- Method overview and mathematical formulation
- Supported datasets and environments
- Installation and usage instructions
- Evaluation metrics
- Customization options
- References and citations

### **IMPLEMENTATION_SUMMARY.md**
Detailed technical documentation:
- Algorithm description
- Dataset details and characteristics
- Implementation architecture
- Example results and performance
- Computational considerations
- Limitations and future work

### **DATASET_DETAILS.md**
In-depth information about MORL datasets:
- MO-Gymnasium environments (all variants)
- D4MORL offline datasets
- MORL-Baselines benchmark
- Synthetic domains
- Comparison tables
- Download instructions
- Citation information

### **INDEX.md** (This File)
Overview of all files in the implementation.

## 💻 Source Code Files

### **gradient_domain_clustering.py** (Core Implementation)
Main implementation of the clustering method:
- `ClusteringConfig`: Configuration dataclass
- `SimplePolicy`: Neural network policy
- `GradientDomainClustering`: Main clustering class
  - Gradient computation
  - Similarity matrix construction
  - Spectral clustering
  - Evaluation metrics
  - Visualization

**Key Functions**:
- `compute_policy_gradient()`: Compute gradient for single domain
- `compute_domain_gradients()`: Batch gradient computation
- `build_similarity_matrix()`: Cosine similarity matrix
- `apply_spectral_clustering()`: Spectral clustering
- `evaluate_clustering()`: Compute metrics (ARI, NMI, Silhouette)
- `visualize_results()`: Generate heatmaps

### **mo_gymnasium_loader.py** (Environment Utilities)
Environment loading and management:
- `MOEnvironmentWrapper`: Wrapper for MO-Gymnasium
- `SyntheticMOEnvironment`: Built-in synthetic environment
- `load_mo_gymnasium_environments()`: Batch loading
- `get_default_mo_environments()`: Default configs
- `get_mo_mujoco_environments()`: MuJoCo configs
- `create_synthetic_mo_domains()`: Create synthetic domains
- `get_domain_info()`: Extract domain information

### **run_clustering_experiment.py** (Experiment Scripts)
Complete experiment pipeline:
- `run_synthetic_experiment()`: Synthetic domain clustering
- `run_mo_gymnasium_experiment()`: MO-Gymnasium clustering
- `main()`: Command-line interface with argparse

**Command-line Arguments**:
- `--experiment`: Type (synthetic, mo_gym_default, mo_gym_mujoco)
- `--n_domains`: Number of synthetic domains
- `--n_clusters`: Number of clusters
- `--n_gradient_samples`: Gradient estimation samples
- `--obs_dim`: Observation dimension
- `--action_dim`: Action dimension
- `--output_dir`: Results directory

### **test_implementation.py** (Testing)
Comprehensive test suite:
- Policy network tests
- Domain creation tests
- Gradient computation tests
- Similarity matrix tests
- Clustering tests
- Evaluation tests
- Summary generation tests

Run with: `python test_implementation.py`

### **__init__.py** (Package Interface)
Package initialization and exports:
- Imports all main classes and functions
- Defines `__all__` for clean imports
- Version information

## 📦 Configuration Files

### **requirements.txt**
Python package dependencies:
- Core: numpy, torch, scikit-learn, matplotlib, seaborn
- Optional: mo-gymnasium, gymnasium, gdown

Install with: `pip install -r requirements.txt`

## 📁 Directory Structure

```
morl_gradient_clustering/
├── Documentation/
│   ├── QUICK_START.md          # Start here
│   ├── README.md               # Main documentation
│   ├── IMPLEMENTATION_SUMMARY.md # Technical details
│   ├── DATASET_DETAILS.md      # Dataset information
│   └── INDEX.md                # This file
│
├── Source Code/
│   ├── gradient_domain_clustering.py  # Core implementation
│   ├── mo_gymnasium_loader.py         # Environment utilities
│   ├── run_clustering_experiment.py   # Experiment scripts
│   ├── test_implementation.py         # Test suite
│   └── __init__.py                    # Package interface
│
├── Configuration/
│   └── requirements.txt        # Dependencies
│
└── Results/ (created when running)
    ├── clustering_results_*.png    # Visualizations
    └── clustering_results_*.txt    # Text summaries
```

## 🚀 Quick Reference

### Installation
```bash
pip install numpy torch scikit-learn matplotlib seaborn
pip install mo-gymnasium  # Optional
```

### Run Test
```bash
python test_implementation.py
```

### Run Experiment
```bash
# Synthetic (fastest)
python run_clustering_experiment.py --experiment synthetic

# MO-Gymnasium
python run_clustering_experiment.py --experiment mo_gym_default

# MO-MuJoCo
python run_clustering_experiment.py --experiment mo_gym_mujoco
```

### Programmatic Usage
```python
from gradient_domain_clustering import GradientDomainClustering, ClusteringConfig
from mo_gymnasium_loader import create_synthetic_mo_domains

domains = create_synthetic_mo_domains(n_domains=9)
config = ClusteringConfig(n_clusters=3)
clustering = GradientDomainClustering(config)
clustering.compute_domain_gradients(domains, 4, 2)
clustering.build_similarity_matrix()
labels = clustering.apply_spectral_clustering()
```

## 📊 Output Files

When you run experiments, the following files are generated in `results/`:

### **clustering_results_*.png**
Visualization with two subplots:
1. **Similarity Matrix**: Heatmap of cosine similarities
2. **Cluster Assignments**: Binary matrix showing cluster membership

### **clustering_results_*.txt**
Text summary containing:
- Configuration parameters
- Clustering metrics (ARI, NMI, Silhouette)
- Cluster summary (which domains in which clusters)
- True vs. predicted labels

## 🔍 File Sizes

| File | Size | Purpose |
|------|------|---------|
| gradient_domain_clustering.py | ~14 KB | Core implementation |
| mo_gymnasium_loader.py | ~10 KB | Environment utilities |
| run_clustering_experiment.py | ~12 KB | Experiment scripts |
| test_implementation.py | ~6 KB | Testing |
| DATASET_DETAILS.md | ~13 KB | Dataset documentation |
| IMPLEMENTATION_SUMMARY.md | ~12 KB | Technical details |
| QUICK_START.md | ~9 KB | Quick start guide |
| README.md | ~8 KB | Main documentation |
| __init__.py | ~1 KB | Package interface |
| requirements.txt | <1 KB | Dependencies |

**Total**: ~85 KB of code and documentation

## 📖 Reading Order

For different purposes, read files in this order:

### **Getting Started** (5 minutes)
1. QUICK_START.md
2. Run test_implementation.py
3. Run first experiment

### **Understanding the Method** (15 minutes)
1. README.md (Overview and formulation)
2. IMPLEMENTATION_SUMMARY.md (Technical details)
3. gradient_domain_clustering.py (Code)

### **Working with Datasets** (10 minutes)
1. DATASET_DETAILS.md (All datasets)
2. mo_gymnasium_loader.py (Loading code)
3. run_clustering_experiment.py (Usage examples)

### **Development** (30 minutes)
1. All documentation files
2. All source code files
3. test_implementation.py (Testing patterns)

## 🎯 Key Features by File

### gradient_domain_clustering.py
✓ Policy gradient computation  
✓ Cosine similarity matrix  
✓ Spectral clustering  
✓ Evaluation metrics (ARI, NMI, Silhouette)  
✓ Visualization generation  

### mo_gymnasium_loader.py
✓ MO-Gymnasium integration  
✓ Synthetic environment creation  
✓ Domain variant generation  
✓ Environment information extraction  

### run_clustering_experiment.py
✓ Complete experiment pipeline  
✓ Command-line interface  
✓ Multiple experiment types  
✓ Automatic result saving  

### test_implementation.py
✓ Unit tests for all components  
✓ Integration tests  
✓ Verification of correctness  

## 🔗 Dependencies Between Files

```
gradient_domain_clustering.py (standalone)
    ↓
mo_gymnasium_loader.py (uses gradient_domain_clustering)
    ↓
run_clustering_experiment.py (uses both above)
    ↓
test_implementation.py (tests all above)
```

## 📝 Notes

- All Python files include comprehensive docstrings
- All functions have type hints
- Code follows PEP 8 style guidelines
- Documentation uses Markdown formatting
- Examples are provided for all major functions

## 🆘 Getting Help

1. **Quick questions**: Check QUICK_START.md
2. **Method details**: Read README.md
3. **Dataset info**: See DATASET_DETAILS.md
4. **Technical details**: Review IMPLEMENTATION_SUMMARY.md
5. **Code questions**: Read docstrings in source files
6. **Errors**: Run test_implementation.py

## ✅ Checklist for New Users

- [ ] Read QUICK_START.md
- [ ] Install dependencies from requirements.txt
- [ ] Run test_implementation.py
- [ ] Run first synthetic experiment
- [ ] Review generated visualizations
- [ ] Read README.md for full details
- [ ] Try MO-Gymnasium environments
- [ ] Customize for your use case

## 📄 License and Citation

This implementation is provided for research purposes. When using this code, please cite:

- The relevant dataset papers (see DATASET_DETAILS.md)
- MO-Gymnasium (Alegre et al., 2022)
- MORL-Baselines (Felten et al., 2023)

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Total Files**: 11 (10 implementation + 1 index)
