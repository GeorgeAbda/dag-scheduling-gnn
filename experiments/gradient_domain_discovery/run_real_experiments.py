"""
Run REAL experiments for Gradient Domain Discovery paper.
All figures generated from actual experimental data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
})


class PolicyNetwork(nn.Module):
    """Simple policy network for scheduling."""
    
    def __init__(self, state_dim: int = 12, hidden_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.network(x)


def create_scheduling_environment(num_samples: int = 2000):
    """
    Create realistic scheduling environment with queue regimes.
    
    4 VMs with heterogeneous characteristics:
    - VM0: Fast (2.0), High power (efficiency=0.5)
    - VM1: Medium (1.5), Medium power (efficiency=0.8)  
    - VM2: Slow (1.0), Very efficient (efficiency=1.0)
    - VM3: Very fast (2.5), Very high power (efficiency=0.3)
    """
    num_vms = 4
    state_dim = num_vms * 3  # speed, efficiency, queue_length per VM
    
    # Fixed VM properties
    vm_speed = torch.tensor([2.0, 1.5, 1.0, 2.5])
    vm_efficiency = torch.tensor([0.5, 0.8, 1.0, 0.3])
    
    # Generate varying queue states
    queue_lengths = torch.rand(num_samples, num_vms) * 5
    
    # Build state vectors
    states = torch.zeros(num_samples, state_dim)
    for i in range(num_vms):
        states[:, i*3] = vm_speed[i]
        states[:, i*3 + 1] = vm_efficiency[i]
        states[:, i*3 + 2] = queue_lengths[:, i]
    
    # Queue-Free optimal: maximize efficiency
    optimal_qf = torch.argmax(vm_efficiency.unsqueeze(0).expand(num_samples, -1), dim=1)
    
    # Bottleneck optimal: maximize speed / (queue + 1)
    effective_speed = vm_speed.unsqueeze(0) / (queue_lengths + 1)
    optimal_bn = torch.argmax(effective_speed, dim=1)
    
    return states, optimal_qf, optimal_bn, vm_speed, vm_efficiency


def pcgrad_update(grad_qf: torch.Tensor, grad_bn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply PCGrad to two gradients."""
    # Project QF onto BN
    dot_qf_bn = torch.dot(grad_qf, grad_bn)
    if dot_qf_bn < 0:
        grad_qf_new = grad_qf - (dot_qf_bn / (torch.dot(grad_bn, grad_bn) + 1e-8)) * grad_bn
    else:
        grad_qf_new = grad_qf
    
    # Project BN onto QF
    dot_bn_qf = torch.dot(grad_bn, grad_qf)
    if dot_bn_qf < 0:
        grad_bn_new = grad_bn - (dot_bn_qf / (torch.dot(grad_qf, grad_qf) + 1e-8)) * grad_qf
    else:
        grad_bn_new = grad_bn
    
    return grad_qf_new, grad_bn_new


def run_training_experiment(num_epochs: int = 100, batch_size: int = 64):
    """
    Run real training experiment comparing:
    1. Joint training without surgery
    2. Joint training with PCGrad
    3. Specialist (QF only)
    4. Specialist (BN only)
    """
    print("="*70)
    print("RUNNING REAL TRAINING EXPERIMENTS")
    print("="*70)
    
    # Create environment
    states, optimal_qf, optimal_bn, vm_speed, vm_efficiency = create_scheduling_environment(2000)
    state_dim = states.shape[1]
    num_actions = 4
    
    # Split into train/test
    train_states = states[:1600]
    train_qf = optimal_qf[:1600]
    train_bn = optimal_bn[:1600]
    test_states = states[1600:]
    test_qf = optimal_qf[1600:]
    test_bn = optimal_bn[1600:]
    
    loss_fn = nn.CrossEntropyLoss()
    
    results = {
        'joint_no_surgery': {'loss': [], 'acc_qf': [], 'acc_bn': [], 'conflict_rate': []},
        'joint_pcgrad': {'loss': [], 'acc_qf': [], 'acc_bn': [], 'conflict_rate': []},
        'specialist_qf': {'loss': [], 'acc_qf': [], 'acc_bn': []},
        'specialist_bn': {'loss': [], 'acc_qf': [], 'acc_bn': []},
    }
    
    # =========================================================================
    # 1. Joint Training WITHOUT Surgery
    # =========================================================================
    print("\n[1/4] Training: Joint (no surgery)...")
    torch.manual_seed(42)
    model = PolicyNetwork(state_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        conflicts = 0
        total_batches = 0
        
        # Shuffle
        perm = torch.randperm(len(train_states))
        
        for i in range(0, len(train_states), batch_size):
            idx = perm[i:i+batch_size]
            batch_states = train_states[idx]
            batch_qf = train_qf[idx]
            batch_bn = train_bn[idx]
            
            optimizer.zero_grad()
            
            # Combined loss (average of both regimes)
            logits = model(batch_states)
            loss_qf = loss_fn(logits, batch_qf)
            loss_bn = loss_fn(logits, batch_bn)
            loss = (loss_qf + loss_bn) / 2
            
            # Measure conflict before update
            model.zero_grad()
            loss_qf.backward(retain_graph=True)
            grad_qf = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            
            model.zero_grad()
            loss_bn.backward(retain_graph=True)
            grad_bn = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            
            cos_sim = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
            if cos_sim < 0:
                conflicts += 1
            total_batches += 1
            
            # Actual update with combined gradient
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(test_states)
            pred = logits.argmax(dim=1)
            acc_qf = (pred == test_qf).float().mean().item()
            acc_bn = (pred == test_bn).float().mean().item()
        
        results['joint_no_surgery']['loss'].append(epoch_loss / (len(train_states) // batch_size))
        results['joint_no_surgery']['acc_qf'].append(acc_qf)
        results['joint_no_surgery']['acc_bn'].append(acc_bn)
        results['joint_no_surgery']['conflict_rate'].append(conflicts / total_batches)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={epoch_loss:.3f}, acc_qf={acc_qf:.3f}, acc_bn={acc_bn:.3f}, conflict={conflicts/total_batches:.2f}")
    
    # =========================================================================
    # 2. Joint Training WITH PCGrad
    # =========================================================================
    print("\n[2/4] Training: Joint + PCGrad...")
    torch.manual_seed(42)
    model = PolicyNetwork(state_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        conflicts = 0
        total_batches = 0
        
        perm = torch.randperm(len(train_states))
        
        for i in range(0, len(train_states), batch_size):
            idx = perm[i:i+batch_size]
            batch_states = train_states[idx]
            batch_qf = train_qf[idx]
            batch_bn = train_bn[idx]
            
            # Compute gradients for each regime
            logits = model(batch_states)
            loss_qf = loss_fn(logits, batch_qf)
            loss_bn = loss_fn(logits, batch_bn)
            
            model.zero_grad()
            loss_qf.backward(retain_graph=True)
            grad_qf = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            
            model.zero_grad()
            loss_bn.backward()
            grad_bn = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            
            # Measure conflict
            cos_sim = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
            if cos_sim < 0:
                conflicts += 1
            total_batches += 1
            
            # Apply PCGrad
            grad_qf_pc, grad_bn_pc = pcgrad_update(grad_qf, grad_bn)
            combined_grad = (grad_qf_pc + grad_bn_pc) / 2
            
            # Apply gradient manually
            offset = 0
            for p in model.parameters():
                numel = p.numel()
                p.grad = combined_grad[offset:offset+numel].view_as(p)
                offset += numel
            
            optimizer.step()
            epoch_loss += (loss_qf.item() + loss_bn.item()) / 2
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(test_states)
            pred = logits.argmax(dim=1)
            acc_qf = (pred == test_qf).float().mean().item()
            acc_bn = (pred == test_bn).float().mean().item()
        
        results['joint_pcgrad']['loss'].append(epoch_loss / (len(train_states) // batch_size))
        results['joint_pcgrad']['acc_qf'].append(acc_qf)
        results['joint_pcgrad']['acc_bn'].append(acc_bn)
        results['joint_pcgrad']['conflict_rate'].append(conflicts / total_batches)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={epoch_loss:.3f}, acc_qf={acc_qf:.3f}, acc_bn={acc_bn:.3f}")
    
    # =========================================================================
    # 3. Specialist (QF only)
    # =========================================================================
    print("\n[3/4] Training: Specialist (Queue-Free only)...")
    torch.manual_seed(42)
    model = PolicyNetwork(state_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(len(train_states))
        
        for i in range(0, len(train_states), batch_size):
            idx = perm[i:i+batch_size]
            batch_states = train_states[idx]
            batch_qf = train_qf[idx]
            
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = loss_fn(logits, batch_qf)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(test_states)
            pred = logits.argmax(dim=1)
            acc_qf = (pred == test_qf).float().mean().item()
            acc_bn = (pred == test_bn).float().mean().item()
        
        results['specialist_qf']['loss'].append(epoch_loss / (len(train_states) // batch_size))
        results['specialist_qf']['acc_qf'].append(acc_qf)
        results['specialist_qf']['acc_bn'].append(acc_bn)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={epoch_loss:.3f}, acc_qf={acc_qf:.3f}, acc_bn={acc_bn:.3f}")
    
    # =========================================================================
    # 4. Specialist (BN only)
    # =========================================================================
    print("\n[4/4] Training: Specialist (Bottleneck only)...")
    torch.manual_seed(42)
    model = PolicyNetwork(state_dim=state_dim, hidden_dim=64, action_dim=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(len(train_states))
        
        for i in range(0, len(train_states), batch_size):
            idx = perm[i:i+batch_size]
            batch_states = train_states[idx]
            batch_bn = train_bn[idx]
            
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = loss_fn(logits, batch_bn)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(test_states)
            pred = logits.argmax(dim=1)
            acc_qf = (pred == test_qf).float().mean().item()
            acc_bn = (pred == test_bn).float().mean().item()
        
        results['specialist_bn']['loss'].append(epoch_loss / (len(train_states) // batch_size))
        results['specialist_bn']['acc_qf'].append(acc_qf)
        results['specialist_bn']['acc_bn'].append(acc_bn)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={epoch_loss:.3f}, acc_qf={acc_qf:.3f}, acc_bn={acc_bn:.3f}")
    
    return results


def run_gradient_conflict_experiment(num_batches: int = 100):
    """Run gradient conflict measurement experiment."""
    print("\n" + "="*70)
    print("RUNNING GRADIENT CONFLICT EXPERIMENT")
    print("="*70)
    
    states, optimal_qf, optimal_bn, _, _ = create_scheduling_environment(2000)
    state_dim = states.shape[1]
    batch_size = 20
    
    torch.manual_seed(42)
    model = PolicyNetwork(state_dim=state_dim, hidden_dim=64, action_dim=4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Cross-domain measurements
    cross_domain_sims = []
    within_qf_sims = []
    within_bn_sims = []
    
    print("\nMeasuring gradient similarities...")
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        if end > len(states):
            break
        
        batch_states = states[start:end]
        batch_qf = optimal_qf[start:end]
        batch_bn = optimal_bn[start:end]
        
        # Cross-domain: QF gradient vs BN gradient
        model.zero_grad()
        logits = model(batch_states)
        loss_qf = loss_fn(logits, batch_qf)
        loss_qf.backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        model.zero_grad()
        loss_bn = loss_fn(logits, batch_bn)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        cos_sim = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
        cross_domain_sims.append(cos_sim.item())
    
    # Within-domain: compare different batches with same labels
    half = len(states) // 2
    for batch_idx in range(num_batches // 2):
        start1 = batch_idx * batch_size
        end1 = start1 + batch_size
        start2 = half + batch_idx * batch_size
        end2 = start2 + batch_size
        
        if end2 > len(states):
            break
        
        # Within QF
        model.zero_grad()
        logits1 = model(states[start1:end1])
        loss1 = loss_fn(logits1, optimal_qf[start1:end1])
        loss1.backward()
        grad1 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        model.zero_grad()
        logits2 = model(states[start2:end2])
        loss2 = loss_fn(logits2, optimal_qf[start2:end2])
        loss2.backward()
        grad2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        cos_sim = torch.dot(grad1, grad2) / (grad1.norm() * grad2.norm() + 1e-8)
        within_qf_sims.append(cos_sim.item())
        
        # Within BN
        model.zero_grad()
        logits1 = model(states[start1:end1])
        loss1 = loss_fn(logits1, optimal_bn[start1:end1])
        loss1.backward()
        grad1 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        model.zero_grad()
        logits2 = model(states[start2:end2])
        loss2 = loss_fn(logits2, optimal_bn[start2:end2])
        loss2.backward()
        grad2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        
        cos_sim = torch.dot(grad1, grad2) / (grad1.norm() * grad2.norm() + 1e-8)
        within_bn_sims.append(cos_sim.item())
    
    return {
        'cross_domain': cross_domain_sims,
        'within_qf': within_qf_sims,
        'within_bn': within_bn_sims,
    }


def plot_training_results(results: Dict):
    """Generate training comparison figure from REAL experimental data."""
    print("\nGenerating training figures from real data...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = np.arange(len(results['joint_no_surgery']['loss']))
    
    # (a) Training Loss
    ax = axes[0]
    ax.plot(epochs, results['joint_no_surgery']['loss'], color='#CC3311', linewidth=2, label='Joint (no surgery)')
    ax.plot(epochs, results['joint_pcgrad']['loss'], color='#009988', linewidth=2, label='Joint + PCGrad')
    ax.plot(epochs, results['specialist_qf']['loss'], color='#0077BB', linewidth=2, linestyle='--', label='Specialist (QF)')
    ax.plot(epochs, results['specialist_bn']['loss'], color='#EE7733', linewidth=2, linestyle='--', label='Specialist (BN)')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Loss', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # (b) Conflict Rate During Training
    ax = axes[1]
    ax.fill_between(epochs, 0.5, 1.0, alpha=0.1, color='red')
    ax.plot(epochs, results['joint_no_surgery']['conflict_rate'], color='#CC3311', linewidth=2, label='Without Surgery')
    ax.plot(epochs, results['joint_pcgrad']['conflict_rate'], color='#009988', linewidth=2, label='With PCGrad')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Conflict Rate')
    ax.set_title('(b) Gradient Conflict During Training', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.text(epochs[-1]*0.7, 0.75, 'High Conflict\nZone', fontsize=9, color='#CC3311')
    
    # (c) Final Performance
    ax = axes[2]
    
    methods = ['Joint\n(no surgery)', 'Joint +\nPCGrad', 'Specialist\n(QF)', 'Specialist\n(BN)']
    qf_perf = [
        results['joint_no_surgery']['acc_qf'][-1],
        results['joint_pcgrad']['acc_qf'][-1],
        results['specialist_qf']['acc_qf'][-1],
        results['specialist_bn']['acc_qf'][-1],
    ]
    bn_perf = [
        results['joint_no_surgery']['acc_bn'][-1],
        results['joint_pcgrad']['acc_bn'][-1],
        results['specialist_qf']['acc_bn'][-1],
        results['specialist_bn']['acc_bn'][-1],
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, qf_perf, width, label='Queue-Free Acc', color='#0077BB', edgecolor='black')
    bars2 = ax.bar(x + width/2, bn_perf, width, label='Bottleneck Acc', color='#EE7733', edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Final Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_training_comparison.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'real_training_comparison.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'real_training_comparison.png'}")


def plot_gradient_conflict_results(conflict_data: Dict):
    """Generate gradient conflict figure from REAL experimental data."""
    print("\nGenerating gradient conflict figures from real data...")
    
    cross = conflict_data['cross_domain']
    within_qf = conflict_data['within_qf']
    within_bn = conflict_data['within_bn']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # (a) Per-batch similarity
    ax = axes[0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in cross]
    ax.bar(range(len(cross)), cross, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=np.mean(cross), color='#0077BB', linewidth=2, linestyle='--', 
              label=f'Mean: {np.mean(cross):.3f}')
    ax.fill_between(range(len(cross)), -1.1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Cross-Domain Gradient Similarity\n(QF vs BN)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper right')
    ax.text(len(cross)*0.7, -0.7, 'CONFLICT\nZONE', fontsize=11, color='#CC3311', fontweight='bold')
    
    # (b) Distribution comparison
    ax = axes[1]
    ax.hist(within_qf, bins=15, alpha=0.6, color='#0077BB', label=f'Within QF (μ={np.mean(within_qf):.2f})', edgecolor='black')
    ax.hist(within_bn, bins=15, alpha=0.6, color='#EE7733', label=f'Within BN (μ={np.mean(within_bn):.2f})', edgecolor='black')
    ax.hist(cross, bins=15, alpha=0.6, color='#CC3311', label=f'Cross-Domain (μ={np.mean(cross):.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.axvspan(-1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Similarity Distribution', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # (c) Conflict rate comparison
    ax = axes[2]
    
    qf_conflict = sum(1 for s in within_qf if s < 0) / len(within_qf) * 100
    bn_conflict = sum(1 for s in within_bn if s < 0) / len(within_bn) * 100
    cross_conflict = sum(1 for s in cross if s < 0) / len(cross) * 100
    
    bars = ax.bar(['Within\nQueue-Free', 'Within\nBottleneck', 'Cross-Domain\n(QF vs BN)'],
                 [qf_conflict, bn_conflict, cross_conflict],
                 color=['#0077BB', '#EE7733', '#CC3311'], edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Conflict Rate (%)')
    ax.set_title('(c) Conflict Rate Comparison', fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle='--', label='Random baseline')
    
    for bar, val in zip(bars, [qf_conflict, bn_conflict, cross_conflict]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.0f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 115)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_gradient_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'real_gradient_conflict.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'real_gradient_conflict.png'}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nCross-Domain (QF vs BN):")
    print(f"  Mean Cosine Similarity: {np.mean(cross):.3f}")
    print(f"  Conflict Rate: {cross_conflict:.1f}%")
    print(f"\nWithin Queue-Free:")
    print(f"  Mean Cosine Similarity: {np.mean(within_qf):.3f}")
    print(f"  Conflict Rate: {qf_conflict:.1f}%")
    print(f"\nWithin Bottleneck:")
    print(f"  Mean Cosine Similarity: {np.mean(within_bn):.3f}")
    print(f"  Conflict Rate: {bn_conflict:.1f}%")


def main():
    # Run gradient conflict experiment
    conflict_data = run_gradient_conflict_experiment(num_batches=100)
    plot_gradient_conflict_results(conflict_data)
    
    # Run training experiment
    training_results = run_training_experiment(num_epochs=100, batch_size=64)
    plot_training_results(training_results)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {output_dir}")
    print("  - real_gradient_conflict.png")
    print("  - real_training_comparison.png")


if __name__ == "__main__":
    main()
