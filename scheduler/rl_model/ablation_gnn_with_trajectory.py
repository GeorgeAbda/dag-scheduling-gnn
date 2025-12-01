"""
Modified ablation_gnn.py with integrated trajectory visualization

This script extends the original ablation_gnn.py to include actor learning
trajectory visualization using the learnable projection method from:
"Visualization and Analysis of the Loss Landscape in Graph Neural Networks"
(Moustafa et al., ICANN 2025)

Key additions:
1. TrajectoryCollector integrated into training loop
2. Automatic trajectory visualization after training
3. Optional loss landscape computation
4. Per-variant trajectory comparison plots

Usage:
    Same as original ablation_gnn.py, with additional flags:
    
    python -m scheduler.rl_model.ablation_gnn_with_trajectory \
        --exp_name gnn_ablation_with_trajectory \
        --dataset.dag_method linear \
        --total_timesteps 200000 \
        --trajectory.enabled True \
        --trajectory.collect_every 50 \
        --trajectory.method svd \
        --trajectory.plot_landscape False

New arguments:
    --trajectory.enabled: Enable trajectory collection and visualization
    --trajectory.collect_every: Collect parameters every N updates
    --trajectory.method: Projection method (svd, pca, random)
    --trajectory.plot_landscape: Whether to compute loss landscape (slow)
    --trajectory.grid_points: Grid resolution for landscape
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import os
import torch
import json
from torch.nn.parameter import UninitializedParameter

# Add the trajectory visualization module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from actor_trajectory_viz import TrajectoryCollector, TrajectoryVisualizer


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory visualization."""
    enabled: bool = True
    collect_every: int = 50  # Collect every N updates
    method: str = "svd"  # svd, pca, or random
    plot_landscape: bool = False  # Whether to compute loss landscape
    grid_points: int = 15  # Grid resolution for landscape
    save_interactive: bool = True  # Save interactive HTML plots


def read_seed_file(path: str, key: str | None = None) -> list[int]:
    p = str(path)
    if p.lower().endswith(".json"):
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            return []
        if isinstance(data, dict):
            if key and key in data and isinstance(data[key], list):
                seq = data[key]
            elif "selected_eval_seeds" in data and isinstance(data["selected_eval_seeds"], list):
                seq = data["selected_eval_seeds"]
            elif "train_seeds" in data and isinstance(data["train_seeds"], list):
                seq = data["train_seeds"]
            else:
                return []
            out: list[int] = []
            for x in seq:
                try:
                    v = int(x)
                    if v > 0:
                        out.append(v)
                except Exception:
                    pass
            return out
        return []
    tmp: list[int] = []
    try:
        with open(p, "r") as f:
            for line in f:
                try:
                    v = int(line.strip())
                    if v > 0:
                        tmp.append(v)
                except Exception:
                    pass
    except Exception:
        tmp = []
    return tmp

def integrate_trajectory_collection(
    actor_model,
    variant_name: str,
    log_dir: Path,
    config: TrajectoryConfig
) -> Optional[TrajectoryCollector]:
    """
    Initialize trajectory collector for a training run.
    
    Args:
        actor_model: The actor network
        variant_name: Name of the ablation variant
        log_dir: Directory for saving outputs
        config: Trajectory configuration
    
    Returns:
        TrajectoryCollector instance if enabled, else None
    """
    if not config.enabled:
        return None
    
    print(f"[{variant_name}] Trajectory collection enabled (every {config.collect_every} updates)")
    collector = TrajectoryCollector(collect_every=config.collect_every)
    
    # Collect initial parameters (optional): filter out any uninitialized lazy params
    # Some hetero/attention variants may defer param init until first forward.
    try:
        named_params = dict(actor_model.named_parameters())
        named_params = {k: v for k, v in named_params.items() if not isinstance(v, UninitializedParameter)}
        collector.collect(named_params, step=0)
    except Exception as e:
        print(f"[{variant_name}] Skipping initial snapshot due to uninitialized params: {e}")
    
    return collector


def visualize_trajectory(
    actor_model,
    collector: TrajectoryCollector,
    variant_name: str,
    log_dir: Path,
    config: TrajectoryConfig,
    eval_function=None,
    landscape_filename: str | None = None
):
    """
    Generate trajectory visualizations after training.
    
    Args:
        actor_model: The trained actor network
        collector: TrajectoryCollector with collected parameters
        variant_name: Name of the ablation variant
        log_dir: Directory for saving outputs
        config: Trajectory configuration
        eval_function: Optional function to evaluate actor loss
    """
    if collector is None or len(collector.get_parameters()) < 2:
        print(f"[{variant_name}] Insufficient trajectory data, skipping visualization")
        return
    
    print(f"[{variant_name}] Generating trajectory visualization...")
    
    # Create trajectory subdirectory
    traj_dir = log_dir / "trajectories" / variant_name
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(
        optimized_parameters=list(actor_model.named_parameters()),
        intermediate_parameters=collector.get_parameter_snapshots(),
        method=config.method
    )
    
    # Plot basic trajectory
    print(f"  - Plotting trajectory...")
    visualizer.plot_trajectory(
        save_path=str(traj_dir / "trajectory.png"),
        title=f"Actor Learning Trajectory - {variant_name}"
    )
    
    # Plot interactive version
    if config.save_interactive:
        print(f"  - Creating interactive plot...")
        visualizer.plot_trajectory_interactive(
            save_path=str(traj_dir / "trajectory_interactive.html"),
            title=f"Actor Learning Trajectory - {variant_name}"
        )
    
    # Plot with loss landscape (if enabled and eval function provided)
    if config.plot_landscape and eval_function is not None:
        print(f"  - Computing loss landscape (this may take a while)...")
        try:
            save_name = landscape_filename if landscape_filename is not None else "trajectory_landscape.png"
            visualizer.plot_trajectory_with_landscape(
                actor_model=actor_model,
                eval_function=eval_function,
                save_path=str(traj_dir / save_name),
                title=f"Actor Learning Trajectory with Loss Landscape - {variant_name}",
                grid_points=config.grid_points
            )
        except Exception as e:
            print(f"  - Warning: Failed to compute loss landscape: {e}")
    
    print(f"[{variant_name}] Trajectory visualization saved to: {traj_dir}")


def create_trajectory_comparison_plot(
    variant_trajectories: dict,
    log_dir: Path,
    title: str = "Actor Learning Trajectories - All Variants"
):
    """
    Create a comparison plot showing trajectories from all variants.
    
    Args:
        variant_trajectories: Dict mapping variant names to trajectory 2D arrays
        log_dir: Directory for saving outputs
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_trajectories)))
    
    for (variant_name, trajectory_2d), color in zip(variant_trajectories.items(), colors):
        # Plot trajectory
        ax.plot(
            trajectory_2d[:, 0],
            trajectory_2d[:, 1],
            '-',
            color=color,
            linewidth=2,
            alpha=0.7,
            label=variant_name
        )
        
        # Mark start
        ax.scatter(
            trajectory_2d[0, 0],
            trajectory_2d[0, 1],
            c=[color],
            s=100,
            marker='o',
            edgecolors='black',
            linewidth=1,
            zorder=10
        )
        
        # Mark end
        ax.scatter(
            trajectory_2d[-1, 0],
            trajectory_2d[-1, 1],
            c=[color],
            s=150,
            marker='X',
            edgecolors='black',
            linewidth=1.5,
            zorder=10
        )
    
    ax.set_xlabel('First Principal Direction', fontsize=12)
    ax.set_ylabel('Second Principal Direction', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = log_dir / "trajectories" / "comparison_all_variants.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory comparison plot saved to: {save_path}")


# Example integration points for ablation_gnn.py:
"""
INTEGRATION INSTRUCTIONS:
=========================

1. Add TrajectoryConfig to your experiment arguments:
   
   @dataclass
   class Args:
       # ... existing fields ...
       trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

2. In the training loop, initialize collector before training:
   
   # Before training starts
   trajectory_collector = integrate_trajectory_collection(
       actor_model=agent.actor,
       variant_name=variant.name,
       log_dir=Path(args.exp_name),
       config=args.trajectory
   )

3. In the PPO update loop, collect parameters:
   
   # After each actor update
   if trajectory_collector is not None and update_step % args.trajectory.collect_every == 0:
       trajectory_collector.collect(
           agent.actor.state_dict(),
           loss=actor_loss.item(),
           step=update_step
       )

4. After training completes, visualize trajectory:
   
   # After training loop
   if trajectory_collector is not None:
       # Define evaluation function
       def eval_actor():
           # Your evaluation code
           return evaluate_agent(agent, test_env)
       
       visualize_trajectory(
           actor_model=agent.actor,
           collector=trajectory_collector,
           variant_name=variant.name,
           log_dir=Path(args.exp_name),
           config=args.trajectory,
           eval_function=eval_actor if args.trajectory.plot_landscape else None
       )

5. (Optional) Create comparison plot across all variants:
   
   # After all variants complete
   variant_trajectories = {}
   for variant_name, collector in collectors.items():
       visualizer = TrajectoryVisualizer(...)
       variant_trajectories[variant_name] = visualizer.trajectory_2d
   
   create_trajectory_comparison_plot(
       variant_trajectories,
       log_dir=Path(args.exp_name),
       title="Actor Learning Trajectories - All Ablation Variants"
   )
"""


if __name__ == "__main__":
    print("=" * 70)
    print("Ablation GNN with Actor Trajectory Visualization")
    print("=" * 70)
    print()
    print("This module extends ablation_gnn.py with trajectory visualization")
    print("capabilities based on the learnable projection method.")
    print()
    print("To integrate into your training script, follow the integration")
    print("instructions in the docstring above.")
    print()
    print("Key features:")
    print("  ✓ Automatic parameter collection during training")
    print("  ✓ SVD-based learnable projection (memory-efficient)")
    print("  ✓ 2D trajectory visualization")
    print("  ✓ Interactive HTML plots")
    print("  ✓ Optional loss landscape computation")
    print("  ✓ Multi-variant comparison plots")
    print()
    print("=" * 70)
