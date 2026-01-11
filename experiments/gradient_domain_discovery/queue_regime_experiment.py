"""
Queue Regime Gradient Conflict Experiment

Tests the theoretical prediction that gradients from queue-free and bottleneck
regimes conflict when training a scheduler agent.

Queue regimes are controlled by:
- workflow_count: more workflows = higher load
- task_length: longer tasks = more VM occupation
- arrival_rate: faster arrivals = more queueing
"""

import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import from the scheduler codebase
try:
    from scheduler.dataset_generator.core.gen_dataset import (
        generate_dataset_long_cp_queue_free,
        generate_dataset_wide_queue_free,
        classify_queue_regime,
    )
    from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
    from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
    from scheduler.dataset_generator.gen_dataset import DatasetArgs
    HAS_SCHEDULER = True
except ImportError as e:
    print(f"Warning: Could not import scheduler modules: {e}")
    print("Running in synthetic mode...")
    HAS_SCHEDULER = False


@dataclass
class QueueRegimeConfig:
    """Configuration for a queue regime."""
    name: str
    workflow_count: int
    min_task_length: int
    max_task_length: int
    arrival_rate: float
    expected_load: str  # "queue_free", "light", "medium", "heavy"


# Define queue regime configurations
QUEUE_REGIMES = {
    "queue_free": QueueRegimeConfig(
        name="queue_free",
        workflow_count=1,
        min_task_length=500,
        max_task_length=5000,
        arrival_rate=0.001,
        expected_load="queue_free",
    ),
    "light_bottleneck": QueueRegimeConfig(
        name="light_bottleneck",
        workflow_count=3,
        min_task_length=5000,
        max_task_length=20000,
        arrival_rate=0.01,
        expected_load="light",
    ),
    "medium_bottleneck": QueueRegimeConfig(
        name="medium_bottleneck",
        workflow_count=5,
        min_task_length=10000,
        max_task_length=50000,
        arrival_rate=0.05,
        expected_load="medium",
    ),
    "heavy_bottleneck": QueueRegimeConfig(
        name="heavy_bottleneck",
        workflow_count=8,
        min_task_length=20000,
        max_task_length=100000,
        arrival_rate=0.1,
        expected_load="heavy",
    ),
}


class SimpleValueNetwork(nn.Module):
    """Simple value network for gradient collection."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.network(x)


class SimplePolicyNetwork(nn.Module):
    """Simple policy network for gradient collection."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, action_dim: int = 100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return torch.softmax(self.network(x), dim=-1)


def create_synthetic_regime_data(
    regime: str,
    num_samples: int = 100,
    obs_dim: int = 128,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic data that mimics different queue regimes.
    
    Key insight: The SAME state features lead to OPPOSITE reward signals
    in different regimes due to the energy decomposition shift.
    
    In queue-free: E_active dominates -> reward correlates POSITIVELY with
                   energy-efficient VM selection (slow but efficient)
    
    In bottleneck: E_idle dominates -> reward correlates POSITIVELY with
                   fast VM selection (to minimize makespan and idle energy)
                   
    This creates DIRECT CONFLICT because the same feature (VM speed/efficiency)
    has opposite effects on reward in different regimes.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate observations (shared state representation)
    obs = torch.randn(num_samples, obs_dim)
    
    # Key features:
    # dims 0-31: VM speed features (higher = faster VM)
    # dims 32-63: VM efficiency features (higher = more energy efficient per work)
    # dims 64-95: Queue length features (higher = longer queue)
    # dims 96-127: Task features
    
    vm_speed = obs[:, :32].mean(dim=1, keepdim=True)
    vm_efficiency = obs[:, 32:64].mean(dim=1, keepdim=True)
    queue_length = obs[:, 64:96].mean(dim=1, keepdim=True)
    
    # Simulate regime-specific reward patterns
    if regime == "queue_free":
        # All VMs idle -> E_active dominates
        # Reward: minimize active energy = maximize efficiency
        # Higher efficiency -> higher reward
        # Speed doesn't matter much (no queueing)
        rewards = vm_efficiency - 0.1 * vm_speed + torch.randn(num_samples, 1) * 0.05
    
    elif regime == "light_bottleneck":
        # Some queueing -> mix of E_active and E_idle
        # Both efficiency and speed matter
        rewards = 0.6 * vm_efficiency + 0.4 * vm_speed - 0.2 * queue_length + torch.randn(num_samples, 1) * 0.05
    
    elif regime == "medium_bottleneck":
        # More queueing -> E_idle starts to dominate
        # Speed matters more than efficiency (reduce makespan)
        rewards = 0.3 * vm_efficiency + 0.7 * vm_speed - 0.4 * queue_length + torch.randn(num_samples, 1) * 0.05
    
    else:  # heavy_bottleneck
        # Heavy queueing -> E_idle dominates completely
        # Reward: minimize makespan = maximize speed, avoid queues
        # Efficiency doesn't matter (idle energy dwarfs active)
        # THE CONFLICT: In queue-free we wanted efficiency, now we want speed
        rewards = -0.1 * vm_efficiency + vm_speed - 0.5 * queue_length + torch.randn(num_samples, 1) * 0.05
    
    return obs, rewards


def collect_gradients(
    model: nn.Module,
    obs: torch.Tensor,
    targets: torch.Tensor,
    num_batches: int = 10,
) -> List[torch.Tensor]:
    """Collect gradients from training on given data."""
    gradients = []
    batch_size = len(obs) // num_batches
    
    loss_fn = nn.MSELoss()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_obs = obs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        
        model.zero_grad()
        predictions = model(batch_obs)
        loss = loss_fn(predictions, batch_targets)
        loss.backward()
        
        # Flatten gradients
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        gradients.append(grad.detach().clone())
    
    return gradients


def compute_regime_similarity_matrix(
    regime_gradients: Dict[str, List[torch.Tensor]],
) -> Tuple[np.ndarray, List[str]]:
    """Compute cosine similarity matrix between regime gradients."""
    regimes = list(regime_gradients.keys())
    n_regimes = len(regimes)
    
    # Compute mean gradient per regime
    mean_gradients = {}
    for regime, grads in regime_gradients.items():
        stacked = torch.stack(grads)
        mean_gradients[regime] = stacked.mean(dim=0)
    
    # Compute similarity matrix
    similarity = np.zeros((n_regimes, n_regimes))
    for i, r1 in enumerate(regimes):
        for j, r2 in enumerate(regimes):
            g1 = mean_gradients[r1]
            g2 = mean_gradients[r2]
            cos_sim = torch.nn.functional.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0))
            similarity[i, j] = cos_sim.item()
    
    return similarity, regimes


def compute_conflict_rate(
    regime_gradients: Dict[str, List[torch.Tensor]],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise conflict rate (% of gradient pairs with negative inner product)."""
    regimes = list(regime_gradients.keys())
    conflict_rates = {}
    
    for i, r1 in enumerate(regimes):
        for j, r2 in enumerate(regimes):
            if i >= j:
                continue
            
            conflicts = 0
            total = 0
            
            for g1 in regime_gradients[r1]:
                for g2 in regime_gradients[r2]:
                    inner_product = torch.dot(g1, g2)
                    if inner_product < 0:
                        conflicts += 1
                    total += 1
            
            conflict_rates[(r1, r2)] = conflicts / total if total > 0 else 0.0
    
    return conflict_rates


def plot_results(
    similarity_matrix: np.ndarray,
    regimes: List[str],
    conflict_rates: Dict[Tuple[str, str], float],
    output_dir: str = "./figures",
    same_model_data: Optional[Dict] = None,
):
    """Generate publication-quality plots."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.facecolor': 'white',
    })
    
    # =========================================================================
    # FIGURE 1: Same-Model Gradient Conflict Analysis (NEW!)
    # =========================================================================
    if same_model_data is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        similarities = same_model_data['similarities']
        gradient_pairs_qf = same_model_data['gradient_pairs_qf']
        gradient_pairs_bn = same_model_data['gradient_pairs_bn']
        
        # (a) Cosine similarity per batch
        ax = axes[0, 0]
        colors = ['#CC3311' if s < 0 else '#009988' for s in similarities]
        ax.bar(range(len(similarities)), similarities, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=same_model_data['mean_similarity'], color='#0077BB', linewidth=2, 
                  linestyle='--', label=f"Mean: {same_model_data['mean_similarity']:.3f}")
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('(a) Per-Batch Gradient Similarity\n(Same Model, QF vs BN labels)', fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper right')
        
        # Add conflict zone annotation
        ax.fill_between(range(len(similarities)), -1.1, 0, alpha=0.1, color='red', label='Conflict Zone')
        ax.text(len(similarities)*0.7, -0.8, 'CONFLICT\nZONE', fontsize=12, color='#CC3311', 
               fontweight='bold', ha='center')
        
        # (b) Gradient direction visualization (2D PCA projection)
        ax = axes[0, 1]
        
        # Stack and project to 2D using PCA
        all_grads = torch.stack(gradient_pairs_qf + gradient_pairs_bn).numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_grads)
        
        n = len(gradient_pairs_qf)
        qf_proj = projected[:n]
        bn_proj = projected[n:]
        
        # Plot arrows from origin
        for i in range(min(10, n)):  # Plot first 10
            ax.arrow(0, 0, qf_proj[i, 0]*0.8, qf_proj[i, 1]*0.8, 
                    head_width=0.05, head_length=0.02, fc='#0077BB', ec='#0077BB', alpha=0.6)
            ax.arrow(0, 0, bn_proj[i, 0]*0.8, bn_proj[i, 1]*0.8, 
                    head_width=0.05, head_length=0.02, fc='#CC3311', ec='#CC3311', alpha=0.6)
        
        # Plot mean directions
        qf_mean = qf_proj.mean(axis=0)
        bn_mean = bn_proj.mean(axis=0)
        ax.arrow(0, 0, qf_mean[0], qf_mean[1], head_width=0.1, head_length=0.05, 
                fc='#0077BB', ec='black', linewidth=2, label='Queue-Free')
        ax.arrow(0, 0, bn_mean[0], bn_mean[1], head_width=0.1, head_length=0.05, 
                fc='#CC3311', ec='black', linewidth=2, label='Bottleneck')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('(b) Gradient Directions (PCA Projection)\nArrows show opposite directions!', fontweight='bold')
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_aspect('equal')
        
        # (c) Gradient cancellation effect
        ax = axes[1, 0]
        
        # Compute norms
        qf_norms = [g.norm().item() for g in gradient_pairs_qf]
        bn_norms = [g.norm().item() for g in gradient_pairs_bn]
        combined_norms = [(gradient_pairs_qf[i] + gradient_pairs_bn[i]).norm().item() 
                         for i in range(len(gradient_pairs_qf))]
        expected_norms = [qf_norms[i] + bn_norms[i] for i in range(len(qf_norms))]
        
        x = np.arange(min(20, len(qf_norms)))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, expected_norms[:20], width, label='Expected (no conflict)', 
                      color='#009988', edgecolor='black')
        bars2 = ax.bar(x + width/2, combined_norms[:20], width, label='Actual (with conflict)', 
                      color='#CC3311', edgecolor='black')
        
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('(c) Gradient Cancellation Effect\n(Actual << Expected due to conflict)', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Add efficiency annotation
        mean_efficiency = np.mean(combined_norms) / np.mean(expected_norms) * 100
        ax.text(0.5, 0.85, f'Training Efficiency: {mean_efficiency:.0f}%\n({100-mean_efficiency:.0f}% wasted!)', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # (d) Within vs Cross Domain Comparison
        ax = axes[1, 1]
        
        within_qf = same_model_data.get('within_qf_sims', [])
        within_bn = same_model_data.get('within_bn_sims', [])
        cross_domain = similarities
        
        # Box plot comparison
        data = [within_qf, within_bn, cross_domain]
        labels = ['Within\nQueue-Free', 'Within\nBottleneck', 'Cross-Domain\n(QF vs BN)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['#0077BB', '#EE7733', '#CC3311']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label='Conflict threshold')
        ax.axhspan(-1, 0, alpha=0.1, color='red')
        
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('(d) Within-Domain vs Cross-Domain\nGradient Similarity', fontweight='bold')
        
        # Add mean annotations
        for i, (d, label) in enumerate(zip(data, ['QF', 'BN', 'Cross'])):
            if len(d) > 0:
                mean_val = np.mean(d)
                ax.text(i+1, mean_val + 0.08, f'{mean_val:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylim(-1.1, 1.1)
        ax.text(0.5, 0.05, 'CONFLICT ZONE', transform=ax.transAxes, fontsize=10, 
               color='#CC3311', fontweight='bold', ha='center', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'same_model_conflict.pdf', dpi=300, facecolor='white')
        plt.savefig(output_dir / 'same_model_conflict.png', dpi=300, facecolor='white')
        plt.close()
        print(f"Same-model conflict plots saved to: {output_dir / 'same_model_conflict.png'}")
        
        # =========================================================================
        # FIGURE 3: Within vs Cross Domain Comparison (Detailed)
        # =========================================================================
        fig, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # (a) Histogram comparison
        ax = axes2[0]
        ax.hist(within_qf, bins=15, alpha=0.6, color='#0077BB', label=f'Within QF (μ={np.mean(within_qf):.2f})', edgecolor='black')
        ax.hist(within_bn, bins=15, alpha=0.6, color='#EE7733', label=f'Within BN (μ={np.mean(within_bn):.2f})', edgecolor='black')
        ax.hist(cross_domain, bins=15, alpha=0.6, color='#CC3311', label=f'Cross-Domain (μ={np.mean(cross_domain):.2f})', edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
        ax.axvspan(-1, 0, alpha=0.1, color='red')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('(a) Distribution Comparison', fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        
        # (b) Conflict rates comparison
        ax = axes2[1]
        
        qf_conflict = sum(1 for s in within_qf if s < 0) / len(within_qf) * 100 if within_qf else 0
        bn_conflict = sum(1 for s in within_bn if s < 0) / len(within_bn) * 100 if within_bn else 0
        cross_conflict = sum(1 for s in cross_domain if s < 0) / len(cross_domain) * 100
        
        bars = ax.bar(['Within\nQueue-Free', 'Within\nBottleneck', 'Cross-Domain\n(QF vs BN)'],
                     [qf_conflict, bn_conflict, cross_conflict],
                     color=['#0077BB', '#EE7733', '#CC3311'], edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Conflict Rate (%)')
        ax.set_title('(b) Conflict Rate Comparison', fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', label='Random baseline')
        
        for bar, val in zip(bars, [qf_conflict, bn_conflict, cross_conflict]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.0f}%', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 115)
        
        # (c) Summary statistics table
        ax = axes2[2]
        ax.axis('off')
        
        table_data = [
            ['Metric', 'Within QF', 'Within BN', 'Cross-Domain'],
            ['Mean Similarity', f'{np.mean(within_qf):.3f}', f'{np.mean(within_bn):.3f}', f'{np.mean(cross_domain):.3f}'],
            ['Std Dev', f'{np.std(within_qf):.3f}', f'{np.std(within_bn):.3f}', f'{np.std(cross_domain):.3f}'],
            ['Conflict Rate', f'{qf_conflict:.0f}%', f'{bn_conflict:.0f}%', f'{cross_conflict:.0f}%'],
            ['Min Similarity', f'{min(within_qf):.3f}', f'{min(within_bn):.3f}', f'{min(cross_domain):.3f}'],
            ['Max Similarity', f'{max(within_qf):.3f}', f'{max(within_bn):.3f}', f'{max(cross_domain):.3f}'],
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.25, 0.2, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Color header row
        for j in range(4):
            table[(0, j)].set_facecolor('#DDDDDD')
            table[(0, j)].set_text_props(fontweight='bold')
        
        # Color data cells
        for i in range(1, 6):
            table[(i, 1)].set_facecolor('#CCE5FF')  # QF blue
            table[(i, 2)].set_facecolor('#FFE5CC')  # BN orange
            table[(i, 3)].set_facecolor('#FFCCCC')  # Cross red
        
        ax.set_title('(c) Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'within_vs_cross_domain.pdf', dpi=300, facecolor='white')
        plt.savefig(output_dir / 'within_vs_cross_domain.png', dpi=300, facecolor='white')
        plt.close()
        print(f"Within vs Cross domain plots saved to: {output_dir / 'within_vs_cross_domain.png'}")
    
    # =========================================================================
    # FIGURE 2: Regime Comparison Heatmap
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) Similarity heatmap
    ax = axes[0]
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(regimes)))
    ax.set_yticks(range(len(regimes)))
    ax.set_xticklabels([r.replace('_', '\n') for r in regimes], fontsize=9)
    ax.set_yticklabels([r.replace('_', '\n') for r in regimes], fontsize=9)
    ax.set_title('(a) Gradient Cosine Similarity', fontweight='bold')
    
    # Add values
    for i in range(len(regimes)):
        for j in range(len(regimes)):
            color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # (b) Conflict rates bar chart
    ax = axes[1]
    pairs = list(conflict_rates.keys())
    rates = [conflict_rates[p] * 100 for p in pairs]
    pair_labels = [f'{p[0][:8]}\nvs\n{p[1][:8]}' for p in pairs]
    
    colors = ['#CC3311' if r > 30 else '#EE7733' if r > 10 else '#009988' for r in rates]
    bars = ax.bar(range(len(pairs)), rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Conflict Rate (%)')
    ax.set_title('(b) Gradient Conflict Rate', fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random baseline')
    ax.legend(loc='upper right')
    
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # (c) Queue-free vs Bottleneck comparison
    ax = axes[2]
    
    # Extract specific comparisons
    qf_vs_others = []
    labels = []
    for regime in regimes:
        if regime != 'queue_free':
            key = ('queue_free', regime) if ('queue_free', regime) in conflict_rates else (regime, 'queue_free')
            if key in conflict_rates:
                qf_vs_others.append(conflict_rates[key] * 100)
                labels.append(regime.replace('_', '\n'))
    
    if qf_vs_others:
        colors = ['#0077BB', '#EE7733', '#CC3311'][:len(qf_vs_others)]
        bars = ax.bar(range(len(qf_vs_others)), qf_vs_others, color=colors, 
                     edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(qf_vs_others)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Conflict Rate with Queue-Free (%)')
        ax.set_title('(c) Queue-Free vs Bottleneck Conflict', fontweight='bold')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1)
        
        for bar, val in zip(bars, qf_vs_others):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'queue_regime_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'queue_regime_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Regime comparison plots saved to: {output_dir / 'queue_regime_conflict.png'}")


def create_scheduling_scenario(num_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a scheduling scenario where queue-free and bottleneck regimes
    prefer DIFFERENT actions for the SAME state.
    
    Scenario: 4 VMs with different speed/efficiency profiles
    - VM0: Fast (speed=2.0), Power-hungry (efficiency=0.5)
    - VM1: Medium (speed=1.5), Medium efficiency (efficiency=0.8)
    - VM2: Slow (speed=1.0), Very efficient (efficiency=1.0)
    - VM3: Very fast (speed=2.5), Very power-hungry (efficiency=0.3)
    
    State includes: VM features + current queue lengths
    
    Returns:
        states: (N, state_dim) state observations
        optimal_qf: (N,) optimal action indices for queue-free
        optimal_bn: (N,) optimal action indices for bottleneck
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_vms = 4
    state_dim = num_vms * 3  # speed, efficiency, queue_length per VM
    
    # VM properties (fixed)
    vm_speed = torch.tensor([2.0, 1.5, 1.0, 2.5])
    vm_efficiency = torch.tensor([0.5, 0.8, 1.0, 0.3])
    
    # Generate random queue states
    # Queue lengths vary to create different optimal actions
    queue_lengths = torch.rand(num_samples, num_vms) * 5  # 0-5 queue length
    
    # Build state vectors
    states = torch.zeros(num_samples, state_dim)
    for i in range(num_vms):
        states[:, i*3] = vm_speed[i]
        states[:, i*3 + 1] = vm_efficiency[i]
        states[:, i*3 + 2] = queue_lengths[:, i]
    
    # Compute optimal actions for each regime
    # Queue-free: choose highest efficiency (VM2)
    # Bottleneck: choose highest speed / (queue + 1) ratio
    
    # Queue-free optimal: maximize efficiency
    optimal_qf = torch.argmax(vm_efficiency.unsqueeze(0).expand(num_samples, -1), dim=1)
    
    # Bottleneck optimal: maximize speed / (queue + 1)
    effective_speed = vm_speed.unsqueeze(0) / (queue_lengths + 1)
    optimal_bn = torch.argmax(effective_speed, dim=1)
    
    return states, optimal_qf, optimal_bn


def collect_policy_gradients(
    model: nn.Module,
    states: torch.Tensor,
    optimal_actions: torch.Tensor,
    num_batches: int = 10,
) -> List[torch.Tensor]:
    """Collect policy gradients using cross-entropy loss on optimal actions."""
    gradients = []
    batch_size = len(states) // num_batches
    
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_states = states[start_idx:end_idx]
        batch_actions = optimal_actions[start_idx:end_idx]
        
        model.zero_grad()
        logits = model.network(batch_states)  # Get logits before softmax
        loss = loss_fn(logits, batch_actions)
        loss.backward()
        
        # Flatten gradients
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        gradients.append(grad.detach().clone())
    
    return gradients


def run_synthetic_experiment(
    num_samples: int = 500,
    num_batches: int = 20,
    seed: int = 42,
):
    """
    Run gradient conflict experiment with synthetic data.
    
    KEY INSIGHT: Gradient conflict happens when the SAME model receives
    updates from BOTH regimes simultaneously. We measure this by:
    1. Taking a batch of states
    2. Computing gradient for QF labels on those states
    3. Computing gradient for BN labels on the SAME states
    4. Checking if gradients conflict (negative inner product)
    """
    print("="*70)
    print("Queue Regime Gradient Conflict Experiment (Synthetic)")
    print("="*70)
    
    # Create scheduling scenario
    print("\nCreating scheduling scenario...")
    states, optimal_qf, optimal_bn = create_scheduling_scenario(num_samples, seed)
    
    # Check how often optimal actions differ
    action_conflict = (optimal_qf != optimal_bn).float().mean()
    print(f"  Action conflict rate: {action_conflict*100:.1f}%")
    print(f"  (This is how often queue-free and bottleneck prefer different VMs)")
    
    # Create SHARED policy network
    torch.manual_seed(seed)
    state_dim = states.shape[1]
    num_actions = 4  # 4 VMs
    shared_policy = SimplePolicyNetwork(input_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    
    print("\n" + "="*70)
    print("MEASURING GRADIENT CONFLICT ON SAME MODEL")
    print("(Simulating joint training on both regimes)")
    print("="*70)
    
    # For each batch: compute gradients from BOTH regimes on SAME model state
    batch_size = num_samples // num_batches
    loss_fn = nn.CrossEntropyLoss()
    
    conflicts = []
    similarities = []
    gradient_pairs_qf = []
    gradient_pairs_bn = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        
        batch_states = states[start:end]
        batch_qf_actions = optimal_qf[start:end]
        batch_bn_actions = optimal_bn[start:end]
        
        # Compute gradient for Queue-Free labels
        shared_policy.zero_grad()
        logits_qf = shared_policy.network(batch_states)
        loss_qf = loss_fn(logits_qf, batch_qf_actions)
        loss_qf.backward()
        grad_qf = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        # Compute gradient for Bottleneck labels (SAME model state!)
        shared_policy.zero_grad()
        logits_bn = shared_policy.network(batch_states)
        loss_bn = loss_fn(logits_bn, batch_bn_actions)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        # Measure conflict
        inner_product = torch.dot(grad_qf, grad_bn)
        cos_sim = inner_product / (grad_qf.norm() * grad_bn.norm() + 1e-8)
        
        conflicts.append(inner_product < 0)
        similarities.append(cos_sim.item())
        gradient_pairs_qf.append(grad_qf)
        gradient_pairs_bn.append(grad_bn)
        
        if batch_idx < 5:  # Print first 5 batches
            print(f"  Batch {batch_idx}: cos_sim={cos_sim.item():.3f}, conflict={inner_product < 0}")
    
    # Summary statistics
    conflict_rate = sum(conflicts) / len(conflicts)
    mean_similarity = np.mean(similarities)
    
    print(f"\n  CROSS-DOMAIN (QF vs BN) Conflict Rate: {conflict_rate*100:.1f}%")
    print(f"  CROSS-DOMAIN Mean Cosine Similarity: {mean_similarity:.3f}")
    
    # =========================================================================
    # WITHIN-DOMAIN GRADIENT SIMILARITY
    # Compare gradients from the SAME regime to show they DON'T conflict
    # =========================================================================
    print("\n" + "="*70)
    print("WITHIN-DOMAIN GRADIENT SIMILARITY")
    print("(Comparing batches from SAME regime)")
    print("="*70)
    
    # Split data into two halves for within-domain comparison
    half = num_samples // 2
    
    # Queue-Free: compare first half vs second half
    within_qf_sims = []
    within_bn_sims = []
    
    for batch_idx in range(num_batches // 2):
        # First half batch
        start1 = batch_idx * batch_size
        end1 = start1 + batch_size
        # Second half batch (offset by half)
        start2 = half + batch_idx * batch_size
        end2 = start2 + batch_size
        
        if end2 > num_samples:
            break
        
        # Within Queue-Free
        shared_policy.zero_grad()
        logits1 = shared_policy.network(states[start1:end1])
        loss1 = loss_fn(logits1, optimal_qf[start1:end1])
        loss1.backward()
        grad_qf1 = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        shared_policy.zero_grad()
        logits2 = shared_policy.network(states[start2:end2])
        loss2 = loss_fn(logits2, optimal_qf[start2:end2])
        loss2.backward()
        grad_qf2 = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        cos_sim_qf = torch.dot(grad_qf1, grad_qf2) / (grad_qf1.norm() * grad_qf2.norm() + 1e-8)
        within_qf_sims.append(cos_sim_qf.item())
        
        # Within Bottleneck
        shared_policy.zero_grad()
        logits1 = shared_policy.network(states[start1:end1])
        loss1 = loss_fn(logits1, optimal_bn[start1:end1])
        loss1.backward()
        grad_bn1 = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        shared_policy.zero_grad()
        logits2 = shared_policy.network(states[start2:end2])
        loss2 = loss_fn(logits2, optimal_bn[start2:end2])
        loss2.backward()
        grad_bn2 = torch.cat([p.grad.flatten() for p in shared_policy.parameters() if p.grad is not None]).clone()
        
        cos_sim_bn = torch.dot(grad_bn1, grad_bn2) / (grad_bn1.norm() * grad_bn2.norm() + 1e-8)
        within_bn_sims.append(cos_sim_bn.item())
    
    within_qf_conflict = sum(1 for s in within_qf_sims if s < 0) / len(within_qf_sims) * 100
    within_bn_conflict = sum(1 for s in within_bn_sims if s < 0) / len(within_bn_sims) * 100
    
    print(f"\n  WITHIN Queue-Free:")
    print(f"    Mean Cosine Similarity: {np.mean(within_qf_sims):.3f}")
    print(f"    Conflict Rate: {within_qf_conflict:.1f}%")
    
    print(f"\n  WITHIN Heavy Bottleneck:")
    print(f"    Mean Cosine Similarity: {np.mean(within_bn_sims):.3f}")
    print(f"    Conflict Rate: {within_bn_conflict:.1f}%")
    
    print(f"\n  COMPARISON:")
    print(f"    Cross-domain (QF vs BN): {mean_similarity:.3f} similarity, {conflict_rate*100:.0f}% conflict")
    print(f"    Within Queue-Free:       {np.mean(within_qf_sims):.3f} similarity, {within_qf_conflict:.0f}% conflict")
    print(f"    Within Bottleneck:       {np.mean(within_bn_sims):.3f} similarity, {within_bn_conflict:.0f}% conflict")
    
    # Store for plotting
    within_domain_data = {
        'within_qf_sims': within_qf_sims,
        'within_bn_sims': within_bn_sims,
        'cross_domain_sims': similarities,
    }
    
    # Now also collect regime-specific gradients for heatmap visualization
    print("\n" + "="*70)
    print("Collecting regime-specific gradients for visualization")
    print("="*70)
    
    # Reset model
    torch.manual_seed(seed)
    shared_policy = SimplePolicyNetwork(input_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    initial_state = {k: v.clone() for k, v in shared_policy.state_dict().items()}
    
    # Collect gradients for queue-free regime
    print("\nCollecting gradients for: queue_free")
    shared_policy.load_state_dict({k: v.clone() for k, v in initial_state.items()})
    gradients_qf = collect_policy_gradients(shared_policy, states, optimal_qf, num_batches)
    print(f"  Collected {len(gradients_qf)} gradient samples")
    
    # Collect gradients for bottleneck regime (heavy)
    print("\nCollecting gradients for: heavy_bottleneck")
    shared_policy.load_state_dict({k: v.clone() for k, v in initial_state.items()})
    gradients_bn = collect_policy_gradients(shared_policy, states, optimal_bn, num_batches)
    print(f"  Collected {len(gradients_bn)} gradient samples")
    
    # Also create intermediate regimes by mixing optimal actions
    torch.manual_seed(seed)
    mix_probs = torch.rand(num_samples)
    
    # Light bottleneck: 70% queue-free, 30% bottleneck
    optimal_light = torch.where(mix_probs < 0.7, optimal_qf, optimal_bn)
    print("\nCollecting gradients for: light_bottleneck")
    shared_policy.load_state_dict({k: v.clone() for k, v in initial_state.items()})
    gradients_light = collect_policy_gradients(shared_policy, states, optimal_light, num_batches)
    
    # Medium bottleneck: 40% queue-free, 60% bottleneck
    optimal_medium = torch.where(mix_probs < 0.4, optimal_qf, optimal_bn)
    print("\nCollecting gradients for: medium_bottleneck")
    shared_policy.load_state_dict({k: v.clone() for k, v in initial_state.items()})
    gradients_medium = collect_policy_gradients(shared_policy, states, optimal_medium, num_batches)
    
    # Build regime gradients dict
    regime_gradients = {
        "queue_free": gradients_qf,
        "light_bottleneck": gradients_light,
        "medium_bottleneck": gradients_medium,
        "heavy_bottleneck": gradients_bn,
    }
    
    # Store same-model conflict data for plotting
    same_model_data = {
        "conflict_rate": conflict_rate,
        "mean_similarity": mean_similarity,
        "similarities": similarities,
        "gradient_pairs_qf": gradient_pairs_qf,
        "gradient_pairs_bn": gradient_pairs_bn,
        "within_qf_sims": within_qf_sims,
        "within_bn_sims": within_bn_sims,
    }
    
    # Compute similarity matrix
    print("\n" + "="*70)
    print("Computing Gradient Similarity Matrix")
    print("="*70)
    
    similarity, regimes = compute_regime_similarity_matrix(regime_gradients)
    
    print("\nCosine Similarity Matrix:")
    print("-" * 60)
    header = "            " + "  ".join([f"{r[:12]:>12}" for r in regimes])
    print(header)
    for i, r1 in enumerate(regimes):
        row = f"{r1[:12]:>12} " + "  ".join([f"{similarity[i,j]:>12.3f}" for j in range(len(regimes))])
        print(row)
    
    # Compute conflict rates
    print("\n" + "="*70)
    print("Computing Conflict Rates")
    print("="*70)
    
    conflict_rates = compute_conflict_rate(regime_gradients)
    
    print("\nPairwise Conflict Rates:")
    for (r1, r2), rate in sorted(conflict_rates.items(), key=lambda x: -x[1]):
        status = "HIGH CONFLICT" if rate > 0.3 else "Low conflict" if rate < 0.1 else "Moderate"
        print(f"  {r1} vs {r2}: {rate*100:.1f}% [{status}]")
    
    # Generate plots
    plot_results(similarity, regimes, conflict_rates, same_model_data=same_model_data)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Check theoretical predictions
    qf_heavy_key = ('queue_free', 'heavy_bottleneck')
    if qf_heavy_key not in conflict_rates:
        qf_heavy_key = ('heavy_bottleneck', 'queue_free')
    
    qf_heavy_conflict = conflict_rates.get(qf_heavy_key, 0)
    qf_heavy_sim = similarity[0, -1] if len(regimes) > 1 else 0
    
    print(f"\nQueue-Free vs Heavy Bottleneck:")
    print(f"  Cosine Similarity: {qf_heavy_sim:.3f}")
    print(f"  Conflict Rate: {qf_heavy_conflict*100:.1f}%")
    
    if qf_heavy_sim < 0:
        print(f"\n  ✓ CONFIRMED: Negative gradient similarity ({qf_heavy_sim:.3f} < 0)")
        print("    Theory prediction validated!")
    else:
        print(f"\n  ✗ NOT CONFIRMED: Positive gradient similarity ({qf_heavy_sim:.3f} >= 0)")
        print("    This may indicate homogeneous power profiles or insufficient regime separation")
    
    if qf_heavy_conflict > 0.3:
        print(f"\n  ✓ CONFIRMED: High conflict rate ({qf_heavy_conflict*100:.1f}% > 30%)")
        print("    Gradient surgery recommended!")
    else:
        print(f"\n  Note: Moderate conflict rate ({qf_heavy_conflict*100:.1f}%)")
    
    return similarity, conflict_rates


def run_scheduler_experiment():
    """Run experiment with actual scheduler environments."""
    if not HAS_SCHEDULER:
        print("Scheduler modules not available. Running synthetic experiment instead.")
        return run_synthetic_experiment()
    
    print("="*70)
    print("Queue Regime Gradient Conflict Experiment (Scheduler)")
    print("="*70)
    
    # TODO: Implement with actual scheduler environments
    # This requires setting up the full training loop
    
    print("\nScheduler experiment not yet implemented.")
    print("Running synthetic experiment instead...\n")
    return run_synthetic_experiment()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Queue Regime Gradient Conflict Experiment")
    parser.add_argument("--mode", choices=["synthetic", "scheduler"], default="synthetic",
                       help="Experiment mode")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples per regime")
    parser.add_argument("--batches", type=int, default=20, help="Number of gradient batches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        run_synthetic_experiment(
            num_samples=args.samples,
            num_batches=args.batches,
            seed=args.seed,
        )
    else:
        run_scheduler_experiment()
