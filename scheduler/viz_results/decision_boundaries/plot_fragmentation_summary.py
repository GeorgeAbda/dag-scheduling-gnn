import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Data from our analysis
p_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Fragmentation ratios (from the table)
frag_ratios = {
    'Homogeneous': [0.0628, 0.0700, 0.0807, 0.0876, 0.0885, 0.0903],
    'Heterogeneous': [0.0356, 0.0338, 0.0361, 0.0368, 0.0378, 0.0421],
    'MLP': [0.0416, 0.0415, 0.0416, 0.0424, 0.0438, 0.0442]
}

# In-hull regions (from the table)
regions = {
    'Homogeneous': [254, 302, 419, 473, 444, 465],
    'Heterogeneous': [58, 58, 59, 56, 66, 74],
    'MLP': [60, 57, 61, 64, 67, 69]
}

# Set up the figure with two subplots
plt.figure(figsize=(14, 6))

# Plot 1: Fragmentation Ratio
plt.subplot(1, 2, 1)
for model in ['Homogeneous', 'Heterogeneous', 'MLP']:
    plt.plot(p_values, frag_ratios[model], 'o-', label=model, markersize=8, linewidth=2)
plt.xlabel('GNP p value', fontsize=12)
plt.ylabel('Fragmentation Ratio', fontsize=12)
plt.title('Fragmentation Ratio vs. GNP p Value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Plot 2: Number of In-Hull Regions
plt.subplot(1, 2, 2)
for model in ['Homogeneous', 'Heterogeneous', 'MLP']:
    plt.plot(p_values, regions[model], 'o-', label=model, markersize=8, linewidth=2)
plt.xlabel('GNP p value', fontsize=12)
plt.ylabel('Number of In-Hull Regions', fontsize=12)
plt.title('Region Count vs. GNP p Value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'fragmentation_summary.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved summary plot to: {output_path}")

# Also save as PDF
output_pdf = output_path.replace('.png', '.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved summary plot to: {output_pdf}")

plt.show()
