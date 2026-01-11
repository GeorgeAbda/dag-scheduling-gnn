"""
Generate additional WGDD plots and save to publication_figures/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# IO
root = Path('experiments/gradient_domain_discovery')
outdir = root / 'publication_figures'
outdir.mkdir(parents=True, exist_ok=True)

with open('logs/wgdd/wgdd_results.json', 'r') as f:
    results = json.load(f)

D = np.load('logs/wgdd/distance_matrix.npy')
clusters = np.array(results['clusters'])
edge_probs = np.array([c['edge_prob'] for c in results['mdp_configs']])
pareto = np.array(results['pareto_scores'])

# Affinity and Laplacian
sigma = np.median(D)
A = np.exp(-D**2 / (2 * sigma**2))
np.fill_diagonal(A, 0.0)
Deg = np.diag(A.sum(axis=1))
L = Deg - A

# ---------------------------------------------------------------------------
# 1) Isomap embedding (no extra deps)
# ---------------------------------------------------------------------------
try:
    iso = Isomap(n_neighbors=min(8, len(D)-2), n_components=2, metric='precomputed')
    coords = iso.fit_transform(D)

    plt.figure(figsize=(6,5))
    for label, color, name in [(0, '#32CD32', 'Wide'), (1, '#FF6B6B', 'LongCP')]:
        m = clusters == label
        plt.scatter(coords[m,0], coords[m,1], s=120, c=color, edgecolors='black', linewidths=1.5, alpha=0.8, label=name)
    plt.title('Isomap Embedding (precomputed distances)', fontsize=12, weight='bold')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.grid(True, alpha=0.3); plt.legend(framealpha=0.95)
    plt.tight_layout(); plt.savefig(outdir / '08_isomap_embedding.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
except Exception as e:
    print('Isomap failed:', e)

# ---------------------------------------------------------------------------
# 2) Laplacian eigenvalue spectrum (scree)
# ---------------------------------------------------------------------------
w, _ = np.linalg.eigh(L)
w = np.sort(w)
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(w)+1), w, 'o-', color='#2E86AB', linewidth=2)
plt.title('Graph Laplacian Eigenvalue Spectrum', fontsize=12, weight='bold')
plt.xlabel('Eigenvalue Index'); plt.ylabel('Eigenvalue')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(outdir / '09_laplacian_scree.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

# ---------------------------------------------------------------------------
# 3) Clustered distance heatmap with dendrogram ordering
# ---------------------------------------------------------------------------
Z = linkage(D, method='average')
order = dendrogram(Z, no_plot=True)['leaves']
D_ord = D[np.ix_(order, order)]
plt.figure(figsize=(7,6))
sns.heatmap(D_ord, cmap='viridis', cbar_kws={'label': 'Wasserstein distance'})
plt.title('Clustered Distance Heatmap (ordered by hierarchy)', fontsize=12, weight='bold')
plt.xlabel('MDP (ordered)'); plt.ylabel('MDP (ordered)')
plt.tight_layout(); plt.savefig(outdir / '10_clustered_distance_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

# ---------------------------------------------------------------------------
# 4) t-SNE perplexity grid (robustness)
# ---------------------------------------------------------------------------
perps = [5, 10, 20, 30]
fig, axes = plt.subplots(1, len(perps), figsize=(4*len(perps), 4))
for ax, perp in zip(axes, perps):
    tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42, perplexity=min(perp, len(D)-1))
    emb = tsne.fit_transform(D)
    for label, color, name in [(0, '#32CD32', 'Wide'), (1, '#FF6B6B', 'LongCP')]:
        m = clusters == label
        ax.scatter(emb[m,0], emb[m,1], s=80, c=color, edgecolors='black', linewidths=1, alpha=0.8, label=name)
    ax.set_title(f'perplexity={perp}')
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc='best', framealpha=0.95)
fig.suptitle('t-SNE Robustness Across Perplexities', fontsize=13, weight='bold')
plt.tight_layout(); plt.savefig(outdir / '11_tsne_perplexity_grid.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

# ---------------------------------------------------------------------------
# 5) 3D PCA embedding of (affinity) rows
# ---------------------------------------------------------------------------
X = A  # N x N affinity rows as features
pca = PCA(n_components=3, random_state=42)
X3 = pca.fit_transform(X)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
for label, color, name in [(0, '#32CD32', 'Wide'), (1, '#FF6B6B', 'LongCP')]:
    m = clusters == label
    ax.scatter(X3[m,0], X3[m,1], X3[m,2], s=60, c=color, edgecolors='black', linewidths=0.8, alpha=0.9, label=name, depthshade=True)
ax.set_title('3D PCA of Affinity Rows', fontsize=12, weight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.legend(framealpha=0.95)
plt.tight_layout(); plt.savefig(outdir / '12_pca3d_affinity.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

# ---------------------------------------------------------------------------
# 6) Edge probability histogram by domain
# ---------------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(edge_probs[clusters==0], bins=10, alpha=0.6, color='#32CD32', edgecolor='black', linewidth=1.5, label='Wide', density=True)
plt.hist(edge_probs[clusters==1], bins=10, alpha=0.6, color='#FF6B6B', edgecolor='black', linewidth=1.5, label='LongCP', density=True)
plt.title('Edge Probability Distribution by Domain', fontsize=12, weight='bold')
plt.xlabel('Edge Probability'); plt.ylabel('Density')
plt.grid(True, alpha=0.3); plt.legend(framealpha=0.95)
plt.tight_layout(); plt.savefig(outdir / '13_edgeprob_hist.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

print('✓ Created additional plots in', outdir)
