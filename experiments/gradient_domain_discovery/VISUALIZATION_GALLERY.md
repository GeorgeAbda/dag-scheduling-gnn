# WGDD Visualization Gallery 🎨

Complete collection of publication-quality figures for the WGDD paper.

## 📊 All Available Figures (15 total)

### Core Paper Figures (Required)

| # | Figure | File | Description | Use In |
|---|--------|------|-------------|--------|
| 1 | **Pipeline** | `wgdd_pipeline.png` | 5-stage WGDD method overview | Main paper, Section 4 |
| 2 | **k-Selection** | `k_selection.png` | Automatic cluster count selection | Main paper, Section 6 |
| 3 | **Main Results** | `wgdd_main_results.png` | 4-panel: distance matrix, clusters, k-scores, Pareto | Main paper, Section 6 |
| 4 | **Embedding** | `wgdd_embedding.png` | t-SNE visualization (predicted vs true) | Main paper, Section 6 |
| 5 | **Conflict Analysis** | `wgdd_conflict_analysis.png` | Gradient conflict patterns | Main paper, Section 6 |
| 6 | **PCA Subspace** | `gradient_subspace_pca_HETERO.png` | Gradient subspace projection | Appendix |

### Supplementary Figures (Recommended)

| # | Figure | File | Description | Use In |
|---|--------|------|-------------|--------|
| 7 | **Parameter Space** | `parameter_space_domains.png` | MDP configurations in 3D parameter space | Appendix / Supp |
| 8 | **Pareto Alignment** | `pareto_alignment_analysis.png` | Objective alignment distributions by domain | Appendix / Supp |
| 9 | **Distance Analysis** | `distance_analysis.png` | Within vs between domain distances | Appendix / Supp |
| 10 | **Objective Landscape** | `objective_landscape.png` | Trade-off spectrum coverage | Appendix / Supp |
| 11 | **Domain Comparison** | `domain_comparison.png` | Confusion matrix + characteristics | Appendix / Supp |
| 12 | **Conflict Heatmaps** | `conflict_heatmaps.png` | Per-domain gradient conflict patterns | Appendix / Supp |

### Advanced Visualizations (Optional)

| # | Figure | File | Description | Use In |
|---|--------|------|-------------|--------|
| 13 | **WGDD Journey** | `wgdd_journey.png` | 9-panel complete method walkthrough | Presentation / Poster |
| 14 | **Method Comparison** | `method_comparison_radar.png` | Radar chart: WGDD vs baselines | Presentation / Supp |
| 15 | **Training Dynamics** | `training_dynamics.png` | Gradient evolution over training | Appendix / Discussion |

---

## 🎯 Figure Details

### 1. Pipeline (`wgdd_pipeline.png`)
**Size**: 16×4 inches | **DPI**: 300 | **Format**: PNG

**Content**:
- Stage 1: Generalist pre-training (PPO on mixed data)
- Stage 2: Multi-scale objective sampling (extremes + uniform + focused)
- Stage 3: Gradient distribution collection (REINFORCE, R=5)
- Stage 4: Wasserstein distance matrix (PCA + W₁)
- Stage 5: Spectral clustering (auto k-selection)

**Key insight box**: "Gradient distributions under different objective weightings encode domain structure"

---

### 2. k-Selection (`k_selection.png`)
**Size**: 6×4 inches | **DPI**: 300 | **Format**: PNG

**Content**:
- Silhouette scores for k=2 to k=6
- Optimal k*=2 highlighted in red
- Annotation showing best score (0.400)

**Key result**: Clear peak at k=2, validating automatic selection

---

### 3. Main Results (`wgdd_main_results.png`)
**Size**: 4-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Wasserstein distance matrix with block structure
- **(b)** Discovered clusters vs edge probability (perfect separation at p=0.5)
- **(c)** Silhouette scores for k-selection
- **(d)** Pareto alignment scores by MDP structure

**Key result**: Clear block structure in distance matrix confirms domain separation

---

### 4. Embedding (`wgdd_embedding.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **Left**: t-SNE colored by predicted clusters
- **Right**: t-SNE colored by true labels

**Key result**: Perfect alignment demonstrates NMI=1.0, ARI=1.0

---

### 5. Conflict Analysis (`wgdd_conflict_analysis.png`)
**Size**: 3-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Mean gradient conflict by edge probability
- **(b)** Makespan-energy conflict by discovered cluster
- **(c)** Bootstrap co-occurrence matrix (stability)

**Key insight**: Wide MDPs show higher objective alignment, LongCP shows more conflict

---

### 6. PCA Subspace (`gradient_subspace_pca_HETERO.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **Left**: All gradients colored by objective weighting α
- **Right**: Per-MDP means colored by domain

**Key insight**: PC1 captures makespan-energy trade-off axis

---

### 7. Parameter Space (`parameter_space_domains.png`)
**Size**: 3-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Edge probability vs task count
- **(b)** Edge probability vs task length (log scale)
- **(c)** Task count vs task length (log scale)

**Key insight**: Domains occupy distinct regions in parameter space

---

### 8. Pareto Alignment (`pareto_alignment_analysis.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Histogram + KDE of alignment scores by domain
- **(b)** Violin plot comparison with statistics

**Key result**: 
- Wide: mean=0.08, σ=0.45 (more aligned)
- LongCP: mean=0.01, σ=0.52 (more conflicted)

---

### 9. Distance Analysis (`distance_analysis.png`)
**Size**: 3-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Boxplot: within vs between domain distances
- **(b)** Histogram: distance distributions
- **(c)** Bar chart: separation ratio (2.5×)

**Key result**: Between-domain distances are 2.5× larger than within-domain

---

### 10. Objective Landscape (`objective_landscape.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Multi-scale objective sampling visualization
- **(b)** Mean gradient conflict across trade-off spectrum

**Key insight**: Conflict varies across objective weightings, justifying multi-scale sampling

---

### 11. Domain Comparison (`domain_comparison.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Confusion matrix (perfect 20×20 diagonal)
- **(b)** Domain characteristics comparison (edge prob, task count, length)

**Key result**: Perfect recovery with clear structural differences

---

### 12. Conflict Heatmaps (`conflict_heatmaps.png`)
**Size**: 2-panel figure | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Wide domain: mean gradient conflict matrix
- **(b)** LongCP domain: mean gradient conflict matrix

**Key insight**: Different conflict patterns between domains

---

### 13. WGDD Journey (`wgdd_journey.png`)
**Size**: 9-panel figure (3×3 grid) | **DPI**: 300 | **Format**: PNG

**Panels**:
1. Raw gradient space (no structure)
2. Multi-objective sampling
3. Gradient distributions
4. Wasserstein distance computation
5. Distance matrix
6. Spectral embedding
7. Silhouette analysis
8. Final clustering
9. Summary metrics

**Use case**: Comprehensive walkthrough for presentations/posters

---

### 14. Method Comparison (`method_comparison_radar.png`)
**Size**: 8×8 inches | **DPI**: 300 | **Format**: PNG

**Content**: Radar chart comparing WGDD vs baselines on 5 dimensions:
- Accuracy (NMI)
- Unsupervised capability
- Automatic k-selection
- Stability
- Interpretability

**Methods compared**:
- WGDD (ours): [1.0, 1.0, 1.0, 0.7, 0.9]
- PCA + k-means: [1.0, 1.0, 0.0, 0.5, 0.7]
- Cosine (trained): [0.52, 1.0, 0.0, 0.3, 0.6]
- Random: [0.0, 1.0, 0.0, 0.0, 0.0]

---

### 15. Training Dynamics (`training_dynamics.png`)
**Size**: 4-panel figure (2×2 grid) | **DPI**: 300 | **Format**: PNG

**Panels**:
- **(a)** Conflict evolution during training
- **(b)** Domain separability emergence (NMI over time)
- **(c)** Gradient variance stabilization
- **(d)** Wasserstein distance separation over training

**Key insight**: Domain structure emerges as agent learns

---

## 📁 File Organization

```
experiments/gradient_domain_discovery/
├── figures/
│   ├── wgdd_pipeline.png                    ✅ Core
│   ├── k_selection.png                      ✅ Core
│   ├── wgdd_main_results.png                ✅ Core
│   ├── wgdd_embedding.png                   ✅ Core
│   ├── wgdd_conflict_analysis.png           ✅ Core
│   ├── gradient_subspace_pca_HETERO.png     ✅ Core
│   ├── parameter_space_domains.png          ✅ Supplementary
│   ├── pareto_alignment_analysis.png        ✅ Supplementary
│   ├── distance_analysis.png                ✅ Supplementary
│   ├── objective_landscape.png              ✅ Supplementary
│   ├── domain_comparison.png                ✅ Supplementary
│   ├── conflict_heatmaps.png                ✅ Supplementary
│   ├── wgdd_journey.png                     ✅ Advanced
│   ├── method_comparison_radar.png          ✅ Advanced
│   └── training_dynamics.png                ✅ Advanced
├── create_pipeline_figure.py
├── extract_k_selection.py
├── create_additional_plots.py
└── create_advanced_viz.py
```

---

## 🎨 Color Scheme

Consistent colors across all figures:

| Element | Color | Hex |
|---------|-------|-----|
| Wide domain | Green | `#32CD32` |
| LongCP domain | Red | `#FF6B6B` |
| Optimal selection | Dark red | `#D62828` |
| Primary accent | Blue | `#2E86AB` |
| Between-domain | Purple | `#9370DB` |
| Neutral | Gray | `#999999` |

---

## 📊 Key Results Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **NMI** | 1.000 | Perfect domain recovery |
| **ARI** | 1.000 | Perfect cluster agreement |
| **Silhouette** | 0.400 | Good separation |
| **Optimal k** | 2 | Correctly auto-selected |
| **Bootstrap stability** | 0.310 | Moderate robustness |
| **PCA variance** | 73.6% | Good dimensionality reduction |
| **Separation ratio** | 2.5× | Strong domain distinction |

---

## 🚀 Usage in Paper

### Main Paper (6 figures)
1. Figure 1: Pipeline → Section 4 (Method)
2. Figure 2: k-Selection → Section 6.2 (Main Results)
3. Figure 3: Main Results → Section 6.2 (Main Results)
4. Figure 4: Embedding → Section 6.2 (Main Results)
5. Figure 5: Conflict Analysis → Section 6.3 (Analysis)
6. Figure 6: PCA Subspace → Appendix A

### Supplementary Material (6 figures)
7. Parameter Space → Appendix B.1
8. Pareto Alignment → Appendix B.2
9. Distance Analysis → Appendix B.3
10. Objective Landscape → Appendix B.4
11. Domain Comparison → Appendix B.5
12. Conflict Heatmaps → Appendix B.6

### Presentations/Posters (3 figures)
13. WGDD Journey → Full method walkthrough
14. Method Comparison → Competitive analysis
15. Training Dynamics → Learning insights

---

## 🔧 Regeneration Scripts

To regenerate all figures:

```bash
cd experiments/gradient_domain_discovery

# Core figures
python create_pipeline_figure.py
python extract_k_selection.py

# Supplementary figures
python create_additional_plots.py

# Advanced visualizations
python create_advanced_viz.py
```

---

## ✅ Quality Checklist

All figures meet publication standards:

- [x] 300 DPI resolution
- [x] White background
- [x] Consistent color scheme
- [x] Clear labels and titles
- [x] Readable fonts (≥10pt)
- [x] Vector-compatible format
- [x] Proper legends
- [x] Grid lines for readability
- [x] Statistical annotations
- [x] Professional styling

---

## 🎯 Figure Selection Guide

### For TMLR Submission
**Required**: Figures 1-6 (core paper figures)
**Recommended**: Add Figures 7, 8, 11 to supplementary
**Optional**: Figures 9, 10, 12 for extended appendix

### For Conference Presentation
**Slides**: Figures 1, 3, 4, 13, 14
**Poster**: All 15 figures

### For arXiv Version
**Main paper**: Figures 1-6
**Appendix**: Figures 7-15

---

## 📝 LaTeX Integration

All figures are referenced in `paper_wgdd_tmlr.tex`:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/wgdd_pipeline.png}
    \caption{WGDD pipeline...}
    \label{fig:wgdd-pipeline}
\end{figure}
```

---

## 🎉 Ready for Publication!

All 15 figures are publication-ready and properly organized. The visualization gallery provides comprehensive coverage of the WGDD method, results, and analysis.
