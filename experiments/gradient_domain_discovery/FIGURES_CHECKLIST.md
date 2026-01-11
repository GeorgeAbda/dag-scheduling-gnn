# WGDD Paper Figures Checklist

## All Figures Ready ✅

All figures needed for the TMLR paper are now available in `experiments/gradient_domain_discovery/figures/`

### Main Paper Figures

| Figure | Filename | Status | Description |
|--------|----------|--------|-------------|
| **Figure 1** | `wgdd_pipeline.png` | ✅ | 5-stage WGDD pipeline diagram |
| **Figure 2** | `k_selection.png` | ✅ | Automatic k-selection via silhouette scores |
| **Figure 3** | `wgdd_main_results.png` | ✅ | 4-panel: distance matrix, clusters, k-scores, Pareto alignment |
| **Figure 4** | `wgdd_embedding.png` | ✅ | t-SNE embeddings (predicted vs true labels) |
| **Figure 5** | `wgdd_conflict_analysis.png` | ✅ | Conflict tensor analysis |
| **Figure 6** | `gradient_subspace_pca_HETERO.png` | ✅ | PCA projection of gradients (Appendix) |

### Figure Details

#### Figure 1: WGDD Pipeline (`wgdd_pipeline.png`)
- **Created by**: `create_pipeline_figure.py`
- **Size**: 16×4 inches (300 DPI)
- **Shows**: Complete 5-stage pipeline from generalist training to domain discovery
- **Key elements**: 
  - Stage 1: Generalist pre-training (PPO on mixed data)
  - Stage 2: Multi-scale objective sampling (K=20)
  - Stage 3: Gradient distribution collection (R=5 replicates)
  - Stage 4: Wasserstein distance matrix (PCA + W₁)
  - Stage 5: Spectral clustering (auto k-selection)

#### Figure 2: k-Selection (`k_selection.png`)
- **Created by**: `extract_k_selection.py`
- **Size**: 6×4 inches (300 DPI)
- **Shows**: Silhouette scores for k=2 to k=6
- **Key result**: k*=2 clearly optimal (score=0.400)
- **Highlights**: Red marker and annotation at optimal k

#### Figure 3: Main Results (`wgdd_main_results.png`)
- **Created by**: `wasserstein_domain_discovery.py`
- **Size**: 4-panel figure
- **Panels**:
  - (a) Wasserstein distance matrix with block structure
  - (b) Discovered clusters vs edge probability (perfect separation)
  - (c) Silhouette scores for different k values
  - (d) Pareto alignment scores by MDP structure

#### Figure 4: Embedding (`wgdd_embedding.png`)
- **Created by**: `wasserstein_domain_discovery.py`
- **Size**: 2-panel figure
- **Shows**: t-SNE embedding of Wasserstein distances
- **Left panel**: Colored by predicted clusters
- **Right panel**: Colored by true labels
- **Key result**: Perfect alignment demonstrates successful recovery

#### Figure 5: Conflict Analysis (`wgdd_conflict_analysis.png`)
- **Created by**: `wasserstein_domain_discovery.py`
- **Size**: 3-panel figure
- **Panels**:
  - (a) Mean gradient conflict by edge probability
  - (b) Makespan-energy conflict by discovered cluster
  - (c) Bootstrap co-occurrence matrix (stability analysis)

#### Figure 6: Gradient Subspace PCA (`gradient_subspace_pca_HETERO.png`)
- **Created by**: `generate_figures_from_subspace.py`
- **Size**: 2-panel figure (Appendix)
- **Shows**: PCA projection of all gradient distributions
- **Left panel**: Gradients colored by objective weighting α
- **Right panel**: Per-MDP means colored by domain

### Compilation Instructions

```bash
cd experiments/gradient_domain_discovery

# Compile TMLR version
pdflatex paper_wgdd_tmlr.tex
bibtex paper_wgdd_tmlr
pdflatex paper_wgdd_tmlr.tex
pdflatex paper_wgdd_tmlr.tex

# Output: paper_wgdd_tmlr.pdf
```

### Figure Quality

- **Resolution**: All figures at 300 DPI (publication quality)
- **Format**: PNG with white background
- **Colors**: Consistent color scheme across all figures
  - Wide domain: Green tones
  - LongCP domain: Red tones
  - Optimal selections: Red highlights
  - Pipeline: Blue gradient

### Results Summary

From the figures, we can see:

1. **Perfect Domain Recovery**: NMI=1.0, ARI=1.0
2. **Automatic k-Selection**: Correctly identifies k*=2
3. **Clear Separation**: t-SNE shows distinct clusters
4. **Block Structure**: Distance matrix reveals domain structure
5. **Stability**: Bootstrap analysis confirms robustness

### File Locations

```
experiments/gradient_domain_discovery/
├── figures/
│   ├── wgdd_pipeline.png          ✅
│   ├── k_selection.png             ✅
│   ├── wgdd_main_results.png       ✅
│   ├── wgdd_embedding.png          ✅
│   ├── wgdd_conflict_analysis.png  ✅
│   └── gradient_subspace_pca_HETERO.png ✅
├── paper_wgdd_tmlr.tex            ✅
├── wgdd_references.bib            ✅
└── tmlr.sty                       ✅
```

## Ready for Submission! 🎉

All figures are publication-ready and properly referenced in the LaTeX source.
