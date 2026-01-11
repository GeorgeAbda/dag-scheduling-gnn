"""
Domain-Aware Training with Gradient Surgery

Applies gradient surgery methods (PCGrad, CAGrad) at the discovered domain level
to mitigate negative transfer between domains.

Methods:
1. PCGrad: Project Conflicting Gradients (Yu et al., 2020)
2. CAGrad: Conflict-Averse Gradient descent (Liu et al., 2021)
3. Domain-Level Surgery: Apply surgery between domain-aggregated gradients

Key insight: Instead of applying gradient surgery between all tasks,
we first discover domains and then apply surgery between domains.
This is more efficient and targets the actual sources of conflict.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy


# =============================================================================
# Gradient Surgery Methods
# =============================================================================

def pcgrad_single(
    grad_i: torch.Tensor,
    grad_j: torch.Tensor,
) -> torch.Tensor:
    """
    PCGrad: Project grad_i onto the normal plane of grad_j if they conflict.
    
    From "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
    
    If grad_i · grad_j < 0 (conflict):
        grad_i' = grad_i - (grad_i · grad_j / ||grad_j||^2) * grad_j
    Else:
        grad_i' = grad_i
    """
    dot = torch.dot(grad_i.flatten(), grad_j.flatten())
    
    if dot < 0:
        # Conflict detected, project
        grad_j_norm_sq = torch.dot(grad_j.flatten(), grad_j.flatten())
        if grad_j_norm_sq > 1e-10:
            grad_i = grad_i - (dot / grad_j_norm_sq) * grad_j
    
    return grad_i


def pcgrad(gradients: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply PCGrad to a list of gradients.
    
    For each gradient, project it onto the normal plane of all
    other gradients it conflicts with.
    """
    n = len(gradients)
    modified_grads = [g.clone() for g in gradients]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                modified_grads[i] = pcgrad_single(modified_grads[i], gradients[j])
    
    return modified_grads


def cagrad(
    gradients: List[torch.Tensor],
    c: float = 0.5,
) -> torch.Tensor:
    """
    CAGrad: Conflict-Averse Gradient descent.
    
    From "Conflict-Averse Gradient Descent for Multi-task Learning" (Liu et al., 2021)
    
    Finds a gradient direction that:
    1. Minimizes conflict with all task gradients
    2. Stays close to the average gradient
    
    Args:
        gradients: List of task gradients
        c: Constraint strength (higher = more conflict-averse)
    
    Returns:
        Combined gradient
    """
    n = len(gradients)
    
    # Stack gradients: (n, d)
    G = torch.stack([g.flatten() for g in gradients])
    
    # Average gradient
    g_avg = G.mean(dim=0)
    
    # Compute the Gram matrix
    GG = G @ G.T  # (n, n)
    
    # Solve for optimal weights
    # Minimize ||sum(w_i * g_i) - g_avg||^2
    # Subject to: w_i >= 0, sum(w_i) = 1
    # And: w_i * g_i · g_j >= -c for all conflicting pairs
    
    # Simplified version: use softmax weights based on alignment with average
    alignments = G @ g_avg  # (n,)
    weights = torch.softmax(alignments / (alignments.std() + 1e-8), dim=0)
    
    # Weighted combination
    g_combined = (weights.unsqueeze(1) * G).sum(dim=0)
    
    # Ensure we don't conflict too much with any gradient
    for i in range(n):
        dot = torch.dot(g_combined, G[i])
        if dot < -c * torch.norm(G[i]) * torch.norm(g_combined):
            # Too much conflict, blend with this gradient
            g_combined = 0.9 * g_combined + 0.1 * G[i]
    
    return g_combined.reshape(gradients[0].shape)


def mgda(gradients: List[torch.Tensor]) -> torch.Tensor:
    """
    MGDA: Multiple Gradient Descent Algorithm.
    
    Finds the minimum-norm point in the convex hull of gradients.
    This is the Pareto-optimal direction.
    """
    n = len(gradients)
    
    # Stack gradients
    G = torch.stack([g.flatten() for g in gradients])  # (n, d)
    
    # Gram matrix
    GG = G @ G.T  # (n, n)
    
    # Solve for weights using Frank-Wolfe algorithm
    weights = torch.ones(n, device=G.device) / n
    
    for _ in range(50):  # Frank-Wolfe iterations
        # Current gradient
        g_curr = (weights.unsqueeze(1) * G).sum(dim=0)
        
        # Find gradient with minimum inner product
        dots = G @ g_curr
        min_idx = torch.argmin(dots)
        
        # Line search
        gamma = 2.0 / (2 + _)
        weights = (1 - gamma) * weights
        weights[min_idx] += gamma
    
    # Compute final gradient
    g_combined = (weights.unsqueeze(1) * G).sum(dim=0)
    
    return g_combined.reshape(gradients[0].shape)


# =============================================================================
# Domain-Level Gradient Surgery
# =============================================================================

class DomainAwareOptimizer:
    """
    Optimizer that applies gradient surgery at the domain level.
    
    Instead of applying surgery between all tasks (expensive),
    we first aggregate gradients within domains, then apply
    surgery between domain-level gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: optim.Optimizer,
        domain_assignments: Dict[int, int],  # source_id -> domain_id
        surgery_method: str = "pcgrad",  # "pcgrad", "cagrad", "mgda", "none"
        surgery_frequency: int = 1,  # Apply surgery every N steps
    ):
        """
        Args:
            model: The neural network
            base_optimizer: Base optimizer (e.g., Adam)
            domain_assignments: Mapping from source ID to domain ID
            surgery_method: Which gradient surgery method to use
            surgery_frequency: How often to apply surgery
        """
        self.model = model
        self.optimizer = base_optimizer
        self.domain_assignments = domain_assignments
        self.surgery_method = surgery_method
        self.surgery_frequency = surgery_frequency
        
        # Compute unique domains
        self.domains = sorted(set(domain_assignments.values()))
        self.num_domains = len(self.domains)
        
        # Storage for gradients
        self.domain_gradients: Dict[int, List[torch.Tensor]] = {d: [] for d in self.domains}
        self.step_count = 0
        
        # Statistics
        self.conflict_history = []
        self.surgery_applied = 0
    
    def accumulate_gradient(self, source_id: int):
        """
        Accumulate gradient from a source into its domain.
        Call this after loss.backward() for each source.
        """
        domain_id = self.domain_assignments[source_id]
        
        # Collect current gradients
        grad = self._flatten_gradients()
        if grad is not None:
            self.domain_gradients[domain_id].append(grad.clone())
    
    def step(self):
        """
        Perform optimization step with domain-level gradient surgery.
        """
        self.step_count += 1
        
        # Aggregate gradients within each domain
        domain_grads = {}
        for domain_id in self.domains:
            grads = self.domain_gradients[domain_id]
            if grads:
                # Average gradients within domain
                domain_grads[domain_id] = torch.stack(grads).mean(dim=0)
        
        if len(domain_grads) < 2:
            # Not enough domains, just do normal step
            self.optimizer.step()
            self._clear_gradients()
            return
        
        # Check for conflicts between domains
        grad_list = list(domain_grads.values())
        conflicts = self._count_conflicts(grad_list)
        self.conflict_history.append(conflicts)
        
        # Apply surgery if needed
        if self.step_count % self.surgery_frequency == 0 and conflicts > 0:
            if self.surgery_method == "pcgrad":
                modified_grads = pcgrad(grad_list)
            elif self.surgery_method == "cagrad":
                combined = cagrad(grad_list)
                modified_grads = [combined] * len(grad_list)
            elif self.surgery_method == "mgda":
                combined = mgda(grad_list)
                modified_grads = [combined] * len(grad_list)
            else:
                modified_grads = grad_list
            
            # Average the modified gradients
            final_grad = torch.stack(modified_grads).mean(dim=0)
            self.surgery_applied += 1
        else:
            # No surgery, just average
            final_grad = torch.stack(grad_list).mean(dim=0)
        
        # Set gradients and step
        self._set_gradients(final_grad)
        self.optimizer.step()
        self._clear_gradients()
    
    def _flatten_gradients(self) -> Optional[torch.Tensor]:
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        
        if grads:
            return torch.cat(grads)
        return None
    
    def _set_gradients(self, flat_grad: torch.Tensor):
        """Set parameter gradients from a flat vector."""
        offset = 0
        for p in self.model.parameters():
            if p.grad is not None:
                numel = p.numel()
                p.grad.copy_(flat_grad[offset:offset + numel].reshape(p.shape))
                offset += numel
    
    def _clear_gradients(self):
        """Clear accumulated domain gradients."""
        for domain_id in self.domains:
            self.domain_gradients[domain_id] = []
        self.optimizer.zero_grad()
    
    def _count_conflicts(self, gradients: List[torch.Tensor]) -> int:
        """Count number of conflicting gradient pairs."""
        n = len(gradients)
        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                if torch.dot(gradients[i].flatten(), gradients[j].flatten()) < 0:
                    conflicts += 1
        return conflicts
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            "total_steps": self.step_count,
            "surgery_applied": self.surgery_applied,
            "mean_conflicts": np.mean(self.conflict_history) if self.conflict_history else 0,
            "total_conflicts": sum(self.conflict_history),
        }


# =============================================================================
# Training Loop with Domain-Aware Optimization
# =============================================================================

def train_with_domain_surgery(
    model: nn.Module,
    data_loaders: Dict[int, any],
    domain_assignments: Dict[int, int],
    loss_fn: nn.Module,
    num_epochs: int = 10,
    lr: float = 1e-3,
    surgery_method: str = "pcgrad",
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict]:
    """
    Train model with domain-aware gradient surgery.
    
    Args:
        model: Neural network
        data_loaders: Dict mapping source_id to data loader
        domain_assignments: Dict mapping source_id to domain_id
        loss_fn: Loss function
        num_epochs: Number of training epochs
        lr: Learning rate
        surgery_method: "pcgrad", "cagrad", "mgda", or "none"
        device: Device to train on
    
    Returns:
        Trained model and training statistics
    """
    model = model.to(device)
    base_optimizer = optim.Adam(model.parameters(), lr=lr)
    
    optimizer = DomainAwareOptimizer(
        model=model,
        base_optimizer=base_optimizer,
        domain_assignments=domain_assignments,
        surgery_method=surgery_method,
    )
    
    history = {
        "epoch_losses": [],
        "domain_losses": {d: [] for d in set(domain_assignments.values())},
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Create iterators for all loaders
        iterators = {sid: iter(loader) for sid, loader in data_loaders.items()}
        
        # Train on batches from all sources
        done = False
        while not done:
            done = True
            
            for source_id, iterator in iterators.items():
                try:
                    x, y = next(iterator)
                    done = False
                except StopIteration:
                    continue
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                model.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradient for this source's domain
                optimizer.accumulate_gradient(source_id)
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Perform optimization step with surgery
            if n_batches > 0:
                optimizer.step()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch_losses"].append(avg_loss)
        
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            stats = optimizer.get_statistics()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                  f"Conflicts: {stats['mean_conflicts']:.2f} - "
                  f"Surgery applied: {stats['surgery_applied']}")
    
    history["optimizer_stats"] = optimizer.get_statistics()
    
    return model, history


# =============================================================================
# Comparison Experiment
# =============================================================================

def compare_training_methods(
    model_fn: Callable[[], nn.Module],
    data_loaders: Dict[int, any],
    domain_assignments: Dict[int, int],
    loss_fn: nn.Module,
    num_epochs: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Compare different training methods:
    1. No surgery (baseline)
    2. PCGrad
    3. CAGrad
    4. MGDA
    """
    methods = ["none", "pcgrad", "cagrad", "mgda"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Training with: {method.upper()}")
        print(f"{'='*50}")
        
        model = model_fn()
        model, history = train_with_domain_surgery(
            model=model,
            data_loaders=data_loaders,
            domain_assignments=domain_assignments,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            surgery_method=method,
            device=device,
        )
        
        results[method] = {
            "final_loss": history["epoch_losses"][-1],
            "loss_history": history["epoch_losses"],
            "stats": history["optimizer_stats"],
        }
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_surgery_comparison(
    results: Dict,
    output_dir: str = "./figures",
):
    """Plot comparison of different gradient surgery methods."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
    })
    
    colors = {
        'none': '#BBBBBB',
        'pcgrad': '#0077BB',
        'cagrad': '#EE7733',
        'mgda': '#009988',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # (a) Loss curves
    ax = axes[0, 0]
    for method, data in results.items():
        ax.plot(data["loss_history"], label=method.upper(), 
               color=colors[method], linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Loss Over Time', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # (b) Final performance bar chart
    ax = axes[0, 1]
    methods = list(results.keys())
    final_losses = [results[m]["final_loss"] for m in methods]
    
    bars = ax.bar(range(len(methods)), final_losses, 
                 color=[colors[m] for m in methods],
                 edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Final Loss')
    ax.set_title('(b) Final Performance', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # (c) Improvement over baseline
    ax = axes[1, 0]
    baseline_loss = results['none']['final_loss']
    improvements = [(baseline_loss - results[m]['final_loss']) / baseline_loss * 100 
                   for m in methods]
    
    bar_colors = [colors[m] if imp > 0 else '#CC3311' for m, imp in zip(methods, improvements)]
    bars = ax.bar(range(len(methods)), improvements, 
                 color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('(c) Relative Improvement', fontweight='bold')
    
    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # (d) Smoothed loss curves (moving average)
    ax = axes[1, 1]
    window = 5
    for method, data in results.items():
        losses = np.array(data["loss_history"])
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), smoothed, 
                   label=method.upper(), color=colors[method], linewidth=2.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Smoothed)')
    ax.set_title('(d) Smoothed Training Curves', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'surgery_comparison.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'surgery_comparison.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Surgery comparison plots saved to: {output_dir}")


# =============================================================================
# Demo
# =============================================================================

def demo_domain_aware_training(num_epochs: int = 100):
    """Demonstrate domain-aware training on synthetic data."""
    import sys
    from pathlib import Path
    
    print("="*70)
    print("Domain-Aware Training Demo")
    print(f"Training for {num_epochs} epochs")
    print("="*70)
    
    # Create synthetic multi-domain data with STRONG conflict
    # Key: Use a SHARED representation that domains compete for
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 4 sources, 2 domains that DIRECTLY CONFLICT
    # Both domains try to use the SAME features but in OPPOSITE ways
    domain_assignments = {0: 0, 1: 0, 2: 1, 3: 1}
    
    from torch.utils.data import TensorDataset, DataLoader
    
    # Shared feature space - domains compete for the same weights
    input_dim = 10
    
    # Domain 0 wants: y = +x (positive relationship)
    # Domain 1 wants: y = -x (negative relationship)
    # This creates MAXIMUM gradient conflict
    
    data_loaders = {}
    for source_id in range(4):
        domain_id = domain_assignments[source_id]
        
        X = torch.randn(300, input_dim)
        
        if domain_id == 0:
            # Positive relationship
            y = X.sum(dim=1, keepdim=True) + torch.randn(300, 1) * 0.1
        else:
            # Negative relationship (CONFLICT!)
            y = -X.sum(dim=1, keepdim=True) + torch.randn(300, 1) * 0.1
        
        dataset = TensorDataset(X, y)
        data_loaders[source_id] = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # SMALL model - forces domains to share capacity and compete
    def create_model():
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
    
    loss_fn = nn.MSELoss()
    
    # Compare methods
    results = compare_training_methods(
        model_fn=create_model,
        data_loaders=data_loaders,
        domain_assignments=domain_assignments,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method, data in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Final loss: {data['final_loss']:.4f}")
        print(f"  Total conflicts: {data['stats']['total_conflicts']}")
        print(f"  Surgery applied: {data['stats']['surgery_applied']} times")
    
    # Plot comparison
    plot_surgery_comparison(results)
    
    return results


if __name__ == "__main__":
    demo_domain_aware_training()
