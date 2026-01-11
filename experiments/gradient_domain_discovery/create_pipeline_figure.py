"""
Create WGDD pipeline figure for paper
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(1, 1, figsize=(16, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')

# Colors
stage_colors = ['#E8F4F8', '#D4E9F7', '#B8DCF5', '#9ACEF3', '#7BBFF0']
arrow_color = '#2E86AB'
text_color = '#1A1A1A'

# Stage 1: Generalist Pre-training
x1, y1, w1, h1 = 0.3, 1.5, 2.5, 2
box1 = FancyBboxPatch((x1, y1), w1, h1, boxstyle="round,pad=0.1", 
                       edgecolor='#2E86AB', facecolor=stage_colors[0], linewidth=2)
ax.add_patch(box1)
ax.text(x1 + w1/2, y1 + h1 - 0.3, 'Stage 1', ha='center', fontsize=11, weight='bold', color=text_color)
ax.text(x1 + w1/2, y1 + h1/2 + 0.2, 'Generalist\nPre-training', ha='center', fontsize=10, color=text_color)
ax.text(x1 + w1/2, y1 + 0.3, 'PPO on mixed\nWide + LongCP', ha='center', fontsize=8, style='italic', color='#555')

# Arrow 1->2
arrow1 = FancyArrowPatch((x1 + w1, y1 + h1/2), (x1 + w1 + 0.4, y1 + h1/2),
                         arrowstyle='->', mutation_scale=20, linewidth=2, color=arrow_color)
ax.add_patch(arrow1)

# Stage 2: Multi-scale Objective Sampling
x2, y2, w2, h2 = 3.2, 1.5, 2.5, 2
box2 = FancyBboxPatch((x2, y2), w2, h2, boxstyle="round,pad=0.1",
                       edgecolor='#2E86AB', facecolor=stage_colors[1], linewidth=2)
ax.add_patch(box2)
ax.text(x2 + w2/2, y2 + h2 - 0.3, 'Stage 2', ha='center', fontsize=11, weight='bold', color=text_color)
ax.text(x2 + w2/2, y2 + h2/2 + 0.2, 'Multi-scale\nObjective Sampling', ha='center', fontsize=10, color=text_color)
ax.text(x2 + w2/2, y2 + 0.3, 'Extremes + Uniform\n+ Focused (K=20)', ha='center', fontsize=8, style='italic', color='#555')

# Arrow 2->3
arrow2 = FancyArrowPatch((x2 + w2, y2 + h2/2), (x2 + w2 + 0.4, y2 + h2/2),
                         arrowstyle='->', mutation_scale=20, linewidth=2, color=arrow_color)
ax.add_patch(arrow2)

# Stage 3: Gradient Distribution Collection
x3, y3, w3, h3 = 6.1, 1.5, 2.5, 2
box3 = FancyBboxPatch((x3, y3), w3, h3, boxstyle="round,pad=0.1",
                       edgecolor='#2E86AB', facecolor=stage_colors[2], linewidth=2)
ax.add_patch(box3)
ax.text(x3 + w3/2, y3 + h3 - 0.3, 'Stage 3', ha='center', fontsize=11, weight='bold', color=text_color)
ax.text(x3 + w3/2, y3 + h3/2 + 0.2, 'Gradient\nDistributions', ha='center', fontsize=10, color=text_color)
ax.text(x3 + w3/2, y3 + 0.3, 'REINFORCE\nR=5 replicates', ha='center', fontsize=8, style='italic', color='#555')

# Arrow 3->4
arrow3 = FancyArrowPatch((x3 + w3, y3 + h3/2), (x3 + w3 + 0.4, y3 + h3/2),
                         arrowstyle='->', mutation_scale=20, linewidth=2, color=arrow_color)
ax.add_patch(arrow3)

# Stage 4: Wasserstein Distance
x4, y4, w4, h4 = 9.0, 1.5, 2.5, 2
box4 = FancyBboxPatch((x4, y4), w4, h4, boxstyle="round,pad=0.1",
                       edgecolor='#2E86AB', facecolor=stage_colors[3], linewidth=2)
ax.add_patch(box4)
ax.text(x4 + w4/2, y4 + h4 - 0.3, 'Stage 4', ha='center', fontsize=11, weight='bold', color=text_color)
ax.text(x4 + w4/2, y4 + h4/2 + 0.2, 'Wasserstein\nDistance Matrix', ha='center', fontsize=10, color=text_color)
ax.text(x4 + w4/2, y4 + 0.3, 'PCA + W₁\nN×N distances', ha='center', fontsize=8, style='italic', color='#555')

# Arrow 4->5
arrow4 = FancyArrowPatch((x4 + w4, y4 + h4/2), (x4 + w4 + 0.4, y4 + h4/2),
                         arrowstyle='->', mutation_scale=20, linewidth=2, color=arrow_color)
ax.add_patch(arrow4)

# Stage 5: Spectral Clustering
x5, y5, w5, h5 = 11.9, 1.5, 2.5, 2
box5 = FancyBboxPatch((x5, y5), w5, h5, boxstyle="round,pad=0.1",
                       edgecolor='#2E86AB', facecolor=stage_colors[4], linewidth=2)
ax.add_patch(box5)
ax.text(x5 + w5/2, y5 + h5 - 0.3, 'Stage 5', ha='center', fontsize=11, weight='bold', color=text_color)
ax.text(x5 + w5/2, y5 + h5/2 + 0.2, 'Spectral\nClustering', ha='center', fontsize=10, color=text_color)
ax.text(x5 + w5/2, y5 + 0.3, 'Auto k-selection\nBootstrap stability', ha='center', fontsize=8, style='italic', color='#555')

# Output box
x6, y6, w6, h6 = 14.8, 1.8, 1.0, 1.4
box6 = FancyBboxPatch((x6, y6), w6, h6, boxstyle="round,pad=0.05",
                       edgecolor='#D62828', facecolor='#FFE5E5', linewidth=2.5)
ax.add_patch(box6)
ax.text(x6 + w6/2, y6 + h6/2, 'Domains\nk*=2', ha='center', fontsize=10, weight='bold', color='#D62828')

# Arrow to output
arrow5 = FancyArrowPatch((x5 + w5, y5 + h5/2), (x6, y6 + h6/2),
                         arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D62828')
ax.add_patch(arrow5)

# Title
ax.text(8, 3.7, 'Wasserstein Gradient Domain Discovery (WGDD) Pipeline', 
        ha='center', fontsize=14, weight='bold', color=text_color)

# Input annotation
ax.annotate('Input: N MDPs\n+ trained π_θ', xy=(0.5, 0.8), fontsize=9, 
            ha='left', color='#555', style='italic')

# Key insight box
insight_box = FancyBboxPatch((0.3, 0.1), 15.1, 0.5, boxstyle="round,pad=0.05",
                             edgecolor='#F77F00', facecolor='#FFF3E0', linewidth=1.5, linestyle='--')
ax.add_patch(insight_box)
ax.text(7.85, 0.35, '💡 Key Insight: Gradient distributions under different objective weightings encode domain structure', 
        ha='center', fontsize=9, style='italic', color='#D62828')

plt.tight_layout()
plt.savefig('experiments/gradient_domain_discovery/figures/wgdd_pipeline.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: experiments/gradient_domain_discovery/figures/wgdd_pipeline.png")
plt.close()
