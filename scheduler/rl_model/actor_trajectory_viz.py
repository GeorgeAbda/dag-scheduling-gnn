"""
Actor Learning Trajectory Visualization for Deep RL Architectures

This module implements the learnable projection method from the paper:
"Visualization and Analysis of the Loss Landscape in Graph Neural Networks"
(Moustafa et al., ICANN 2025)

It generates learning trajectories similar to Figure 3 in the paper, showing how
the actor network parameters evolve during training in a 2D projection space.

Usage:
    1. During training, collect intermediate actor parameters using TrajectoryCollector
    2. After training, use TrajectoryVisualizer to project and visualize the trajectory
    3. Optionally compute loss landscape around the trajectory

Example:
    # During training setup
    from scheduler.rl_model.actor_trajectory_viz import TrajectoryCollector
    
    collector = TrajectoryCollector()
    
    # In training loop (after each update or every N steps)
    collector.collect(actor.state_dict())
    
    # After training
    from scheduler.rl_model.actor_trajectory_viz import TrajectoryVisualizer
    
    visualizer = TrajectoryVisualizer(
        optimized_parameters=actor.parameters(),
        intermediate_parameters=collector.get_parameters(),
        method="svd"  # or "pca", "random"
    )
    
    # Generate trajectory plot
    visualizer.plot_trajectory(
        save_path="logs/actor_trajectory.png",
        title="Actor Learning Trajectory"
    )
    
    # Optionally compute and plot loss landscape
    visualizer.plot_trajectory_with_landscape(
        actor_model=actor,
        eval_function=lambda: evaluate_actor(actor, test_env),
        save_path="logs/actor_trajectory_landscape.png"
    )
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Callable, Dict, Any
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


class TrajectoryCollector:
    """
    Collects intermediate actor parameters during training for trajectory visualization.
    
    This class stores snapshots of model parameters at different training steps,
    which can later be used to visualize the learning trajectory in parameter space.
    """
    
    def __init__(self, collect_every: int = 1):
        """
        Initialize the trajectory collector.
        
        Args:
            collect_every: Collect parameters every N calls to collect()
        """
        self.collect_every = collect_every
        self._step_counter = 0
        self._parameter_snapshots: List[Dict[str, torch.Tensor]] = []
        self._losses: List[float] = []
        self._step_indices: List[int] = []
    
    def collect(self, state_dict: Dict[str, torch.Tensor], loss: Optional[float] = None, step: Optional[int] = None):
        """
        Collect a snapshot of model parameters.
        
        Args:
            state_dict: Model state dictionary (from model.state_dict())
            loss: Optional loss value at this step
            step: Optional step/iteration number
        """
        if self._step_counter % self.collect_every == 0:
            # Deep copy to avoid reference issues; enforce deterministic ordering by sorted key name
            items = sorted(((k, v) for k, v in state_dict.items()), key=lambda kv: kv[0])
            snapshot = {k: v.detach().cpu().clone() for k, v in items}
            self._parameter_snapshots.append(snapshot)
            
            if loss is not None:
                self._losses.append(loss)
            
            if step is not None:
                self._step_indices.append(step)
            else:
                self._step_indices.append(self._step_counter)
        
        self._step_counter += 1
    
    def get_parameters(self) -> List[List[torch.Tensor]]:
        """
        Get collected parameters as a list of parameter lists.
        
        Returns:
            List of parameter lists, where each inner list contains all parameters
            from one snapshot
        """
        result = []
        for snapshot in self._parameter_snapshots:
            # Ensure deterministic order when reading back as well
            params = [snapshot[k] for k in sorted(snapshot.keys())]
            result.append(params)
        return result

    def get_parameter_snapshots(self) -> List[Dict[str, torch.Tensor]]:
        """Return raw named parameter snapshots (each dict is already sorted by key at creation time)."""
        return list(self._parameter_snapshots)
    
    def get_losses(self) -> List[float]:
        """Get collected loss values."""
        return self._losses
    
    def get_steps(self) -> List[int]:
        """Get step indices for collected parameters."""
        return self._step_indices
    
    def clear(self):
        """Clear all collected data."""
        self._parameter_snapshots.clear()
        self._losses.clear()
        self._step_indices.clear()
        self._step_counter = 0


class TrajectoryVisualizer:
    """
    Visualizes actor learning trajectories using dimensionality reduction.
    
    Implements three projection methods:
    - SVD (learnable projection): Low-rank SVD for efficient 2D projection
    - PCA: Principal Component Analysis on covariance matrix
    - Random: Random normalized directions
    """
    
    def __init__(
        self,
        optimized_parameters: List[torch.Tensor] | List[tuple[str, torch.Tensor]],
        intermediate_parameters: List[List[torch.Tensor]] | List[Dict[str, torch.Tensor]],
        method: str = "svd",
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trajectory visualizer.
        
        Args:
            optimized_parameters: Final optimized parameters (from model.parameters())
            intermediate_parameters: List of parameter snapshots from training
            method: Projection method - "svd", "pca", or "random"
            device: Device for computations (default: cuda if available, else cpu)
        """
        self.method = method
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Normalize inputs: support named and unnamed parameters
        opt_is_named = len(optimized_parameters) > 0 and isinstance(optimized_parameters[0], tuple)
        interm_are_dicts = len(intermediate_parameters) > 0 and isinstance(intermediate_parameters[0], dict)

        self._named_mode = False
        if opt_is_named and interm_are_dicts:
            # Build name->tensor for optimized
            opt_dict = {name: p.detach().cpu() for name, p in optimized_parameters}  # type: ignore[arg-type]
            # Build canonical name order = intersection across optimized and all snapshots, sorted
            common_names = set(opt_dict.keys())
            for snap in intermediate_parameters:  # type: ignore[union-attr]
                common_names &= set(snap.keys())  # type: ignore[arg-type]
            name_list = sorted(common_names)
            # Create aligned lists
            self.optimized_parameters = [opt_dict[n] for n in name_list]
            self.intermediate_parameters = [
                [snap[n] for n in name_list]  # type: ignore[index]
                for snap in intermediate_parameters  # type: ignore[union-attr]
            ]
            # Store metadata for reconstruction
            self._named_mode = True
            self._common_names = name_list
            self._common_shapes = [opt_dict[n].shape for n in name_list]
            self._common_numels = [int(opt_dict[n].numel()) for n in name_list]
            # Offsets into the reduced vector
            offs = []
            c = 0
            for k in self._common_numels:
                offs.append((c, c + k))
                c += k
            self._common_offsets = offs
        else:
            # Fallback to original behavior (unnamed lists); assume shapes/orders match
            if not isinstance(optimized_parameters, list):
                optimized_parameters = list(optimized_parameters)
            self.optimized_parameters = [p.detach().cpu() for p in optimized_parameters]  # type: ignore[list-item]
            self.intermediate_parameters = intermediate_parameters  # type: ignore[assignment]
        
        # Compute projection directions
        self.b1, self.b2 = self._compute_directions()
        
        # Project trajectory
        self.trajectory_2d = self._project_trajectory()
    
    def _compute_directions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute two basis directions for 2D projection.
        
        Returns:
            Tuple of (b1, b2) direction vectors
        """
        if self.method == "svd":
            return self._compute_svd_directions()
        elif self.method == "pca":
            return self._compute_pca_directions()
        elif self.method == "random":
            return self._compute_random_directions()
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'svd', 'pca', or 'random'")
    
    def _compute_svd_directions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute directions using low-rank SVD (learnable projection method).
        
        This is the method used in the paper for Figure 3.
        """
        # Convert optimized parameters to vector
        optimized_vector = parameters_to_vector(self.optimized_parameters)
        
        # Compute differences from optimized parameters
        differences = []
        for params in self.intermediate_parameters:
            param_vector = parameters_to_vector(params)
            diff = param_vector - optimized_vector
            differences.append(diff)
        
        # Stack into matrix [N x D] where N = number of snapshots, D = parameter dimension
        dataset = torch.stack(differences).to(dtype=torch.float32, device=self.device)
        
        # Use low-rank SVD to extract top 2 components
        # This is memory-efficient for large parameter spaces
        U, S, V = torch.svd_lowrank(dataset, q=2)
        
        # V contains the right singular vectors (directions in parameter space)
        b1 = V[:, 0].cpu()
        b2 = V[:, 1].cpu()
        
        return b1, b2
    
    def _compute_pca_directions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute directions using PCA on covariance matrix.
        
        Note: This can be memory-intensive for large models.
        """
        optimized_vector = parameters_to_vector(self.optimized_parameters)
        
        differences = []
        for params in self.intermediate_parameters:
            param_vector = parameters_to_vector(params)
            diff = param_vector - optimized_vector
            differences.append(diff)
        
        # Stack and compute covariance
        dataset = torch.stack(differences).to(dtype=torch.float32, device=self.device)
        
        # Compute covariance matrix
        cov_matrix = torch.cov(dataset.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        indices = torch.argsort(eigenvalues, descending=True)
        
        # Take top 2 eigenvectors
        b1 = eigenvectors[:, indices[0]].cpu()
        b2 = eigenvectors[:, indices[1]].cpu()
        
        return b1, b2
    
    def _compute_random_directions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute random normalized directions.
        """
        # Generate random directions
        b1 = torch.randn_like(parameters_to_vector(self.optimized_parameters))
        b2 = torch.randn_like(parameters_to_vector(self.optimized_parameters))
        
        # Normalize using filter normalization (scale by parameter magnitudes)
        opt_vector = parameters_to_vector(self.optimized_parameters)
        b1 = b1 * (opt_vector.norm() / (b1.norm() + 1e-10))
        b2 = b2 * (opt_vector.norm() / (b2.norm() + 1e-10))
        
        return b1, b2
    
    def _project_trajectory(self) -> np.ndarray:
        """
        Project the training trajectory onto the 2D basis.
        
        Returns:
            Array of shape [N, 2] with 2D coordinates
        """
        optimized_vector = parameters_to_vector(self.optimized_parameters)
        
        trajectory = []
        for params in self.intermediate_parameters:
            param_vector = parameters_to_vector(params)
            diff = param_vector - optimized_vector
            
            # Project onto basis directions
            x = torch.dot(diff, self.b1).item()
            y = torch.dot(diff, self.b2).item()
            trajectory.append([x, y])
        
        return np.array(trajectory)
    
    def plot_trajectory(
        self,
        save_path: Optional[str] = None,
        title: str = "Actor Learning Trajectory",
        figsize: Tuple[int, int] = (10, 8),
        show_start_end: bool = True,
        show_arrows: bool = True
    ) -> Figure:
        """
        Plot the 2D learning trajectory.
        
        Args:
            save_path: Path to save the figure (if None, figure is not saved)
            title: Plot title
            figsize: Figure size (width, height)
            show_start_end: Whether to mark start and end points
            show_arrows: Whether to show direction arrows along trajectory
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectory line
        ax.plot(
            self.trajectory_2d[:, 0],
            self.trajectory_2d[:, 1],
            'b-',
            linewidth=2,
            alpha=0.6,
            label='Training trajectory'
        )
        
        # Plot points
        ax.scatter(
            self.trajectory_2d[:, 0],
            self.trajectory_2d[:, 1],
            c=np.arange(len(self.trajectory_2d)),
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Mark start and end
        if show_start_end:
            ax.scatter(
                self.trajectory_2d[0, 0],
                self.trajectory_2d[0, 1],
                c='green',
                s=200,
                marker='o',
                edgecolors='black',
                linewidth=2,
                label='Start',
                zorder=10
            )
            ax.scatter(
                self.trajectory_2d[-1, 0],
                self.trajectory_2d[-1, 1],
                c='red',
                s=200,
                marker='X',
                edgecolors='black',
                linewidth=2,
                label='End (Θ*)',
                zorder=10
            )
        
        # Add direction arrows
        if show_arrows and len(self.trajectory_2d) > 5:
            # Add arrows at regular intervals
            arrow_indices = np.linspace(0, len(self.trajectory_2d) - 2, min(10, len(self.trajectory_2d) - 1), dtype=int)
            for idx in arrow_indices:
                dx = self.trajectory_2d[idx + 1, 0] - self.trajectory_2d[idx, 0]
                dy = self.trajectory_2d[idx + 1, 1] - self.trajectory_2d[idx, 1]
                ax.arrow(
                    self.trajectory_2d[idx, 0],
                    self.trajectory_2d[idx, 1],
                    dx * 0.3,
                    dy * 0.3,
                    head_width=0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    head_length=0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    fc='blue',
                    ec='blue',
                    alpha=0.5,
                    zorder=5
                )
        
        ax.set_xlabel('First Principal Direction', fontsize=12)
        ax.set_ylabel('Second Principal Direction', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        
        return fig
    
    def plot_trajectory_interactive(
        self,
        save_path: Optional[str] = None,
        title: str = "Actor Learning Trajectory (Interactive)"
    ):
        """
        Create an interactive Plotly visualization of the trajectory.
        
        Args:
            save_path: Path to save HTML file (if None, file is not saved)
            title: Plot title
        """
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=self.trajectory_2d[:, 0],
            y=self.trajectory_2d[:, 1],
            mode='lines+markers',
            name='Trajectory',
            line=dict(color='blue', width=2),
            marker=dict(
                size=8,
                color=np.arange(len(self.trajectory_2d)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Step")
            ),
            text=[f"Step {i}" for i in range(len(self.trajectory_2d))],
            hovertemplate='<b>Step %{text}</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
        
        # Mark start
        fig.add_trace(go.Scatter(
            x=[self.trajectory_2d[0, 0]],
            y=[self.trajectory_2d[0, 1]],
            mode='markers',
            name='Start',
            marker=dict(size=15, color='green', symbol='circle', line=dict(width=2, color='black'))
        ))
        
        # Mark end
        fig.add_trace(go.Scatter(
            x=[self.trajectory_2d[-1, 0]],
            y=[self.trajectory_2d[-1, 1]],
            mode='markers',
            name='End (Θ*)',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='black'))
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='First Principal Direction',
            yaxis_title='Second Principal Direction',
            hovermode='closest',
            width=900,
            height=700
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            print(f"Interactive trajectory plot saved to: {save_path}")
        
        return fig
    
    def compute_loss_landscape(
        self,
        actor_model: nn.Module,
        eval_function: Callable[[], float],
        grid_points: int = 20,
        range_scale: float = 1.5,
        include_trajectory_points: bool = False,
        trajectory_stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute loss landscape around the trajectory.
        
        Args:
            actor_model: The actor model to evaluate
            eval_function: Function that evaluates the model and returns loss
            grid_points: Number of grid points in each dimension
            range_scale: Scale factor for grid range (relative to trajectory extent)
        
        Returns:
            Tuple of (X, Y, Z) meshgrid arrays for plotting
        """
        # Determine grid range based on trajectory
        x_min, x_max = self.trajectory_2d[:, 0].min(), self.trajectory_2d[:, 0].max()
        y_min, y_max = self.trajectory_2d[:, 1].min(), self.trajectory_2d[:, 1].max()
        
        x_range = (x_max - x_min) * range_scale
        y_range = (y_max - y_min) * range_scale
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        x_grid = np.linspace(x_center - x_range / 2, x_center + x_range / 2, grid_points)
        y_grid = np.linspace(y_center - y_range / 2, y_center + y_range / 2, grid_points)

        # Optionally force grid to include the exact projected trajectory points
        if include_trajectory_points and self.trajectory_2d is not None and len(self.trajectory_2d) > 0:
            try:
                stride = max(1, int(trajectory_stride))
            except Exception:
                stride = 1
            traj_x = self.trajectory_2d[::stride, 0]
            traj_y = self.trajectory_2d[::stride, 1]
            # Merge, sort, and unique to avoid duplicates
            x_grid = np.unique(np.concatenate([x_grid, traj_x]))
            y_grid = np.unique(np.concatenate([y_grid, traj_y]))
        
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        # Store original parameters (full)
        original_named = [(n, p.clone()) for n, p in actor_model.named_parameters()]
        # Reduced optimized vector over common names (or full if unnamed mode)
        reduced_optimized_vector = parameters_to_vector(self.optimized_parameters)
        
        # Evaluate loss at each grid point
        # Progress bar reflects actual grid size after optional augmentation
        _gx, _gy = X.shape
        print(f"Computing loss landscape on {_gx}x{_gy} grid...")
        pbar = tqdm(total=_gx * _gy, desc="Loss landscape", leave=True)
        for i in range(_gx):
            for j in range(_gy):
                # Compute reduced parameter vector at this grid point
                reduced_vec = reduced_optimized_vector + X[i, j] * self.b1 + Y[i, j] * self.b2
                
                if self._named_mode:
                    # Reconstruct full parameter tensors aligned to actor_model.named_parameters()
                    full_tensors = []
                    # Cursor over reduced vector
                    for (name, _p) in actor_model.named_parameters():
                        if name in self._common_names:
                            idx = self._common_names.index(name)
                            s, e = self._common_offsets[idx]
                            piece = reduced_vec[s:e].view(self._common_shapes[idx])
                            full_tensors.append(piece)
                        else:
                            # keep original value
                            # find in original_named
                            for (on, op) in original_named:
                                if on == name:
                                    full_tensors.append(op)
                                    break
                    # Set model parameters
                    vector_to_parameters(parameters_to_vector(full_tensors), actor_model.parameters())
                else:
                    # Unnamed mode: shapes/orders assumed to match
                    vector_to_parameters(reduced_vec, actor_model.parameters())
                
                # Evaluate loss
                try:
                    loss = eval_function()
                    Z[i, j] = loss
                except Exception as e:
                    print(f"Error evaluating at ({X[i, j]:.2f}, {Y[i, j]:.2f}): {e}")
                    Z[i, j] = np.nan
                finally:
                    pbar.update(1)
        pbar.close()
        
        # Restore original parameters
        vector_to_parameters(parameters_to_vector([p for (_n, p) in original_named]), actor_model.parameters())
        
        return X, Y, Z
    
    def plot_trajectory_with_landscape(
        self,
        actor_model: nn.Module,
        eval_function: Callable[[], float],
        save_path: Optional[str] = None,
        title: str = "Actor Learning Trajectory with Loss Landscape",
        grid_points: int = 20,
        figsize: Tuple[int, int] = (12, 10),
        include_trajectory_points: bool = True,
        trajectory_stride: int = 1,
    ) -> Figure:
        """
        Plot trajectory overlaid on loss landscape contour.
        
        Args:
            actor_model: The actor model
            eval_function: Function to evaluate loss
            save_path: Path to save figure
            title: Plot title
            grid_points: Number of grid points for landscape
            figsize: Figure size
        
        Returns:
            Matplotlib Figure object
        """
        # Compute loss landscape
        X, Y, Z = self.compute_loss_landscape(
            actor_model,
            eval_function,
            grid_points=grid_points,
            include_trajectory_points=include_trajectory_points,
            trajectory_stride=trajectory_stride,
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(contour, ax=ax, label='Loss')
        
        # Plot trajectory
        ax.plot(
            self.trajectory_2d[:, 0],
            self.trajectory_2d[:, 1],
            'r-',
            linewidth=3,
            alpha=0.8,
            label='Training trajectory'
        )
        
        # Mark start and end
        ax.scatter(
            self.trajectory_2d[0, 0],
            self.trajectory_2d[0, 1],
            c='green',
            s=200,
            marker='o',
            edgecolors='white',
            linewidth=2,
            label='Start',
            zorder=10
        )
        ax.scatter(
            self.trajectory_2d[-1, 0],
            self.trajectory_2d[-1, 1],
            c='red',
            s=200,
            marker='X',
            edgecolors='white',
            linewidth=2,
            label='End (Θ*)',
            zorder=10
        )
        
        ax.set_xlabel('First Principal Direction', fontsize=12)
        ax.set_ylabel('Second Principal Direction', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, color='white')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory with landscape plot saved to: {save_path}")
        
        return fig


def create_example_usage_script(output_path: str = "example_actor_trajectory.py"):
    """
    Create an example script showing how to use the trajectory visualization.
    
    Args:
        output_path: Where to save the example script
    """
    example_code = '''"""
Example: Using Actor Trajectory Visualization

This script demonstrates how to integrate trajectory visualization
into your Deep RL training loop.
"""

import torch
from scheduler.rl_model.actor_trajectory_viz import TrajectoryCollector, TrajectoryVisualizer

# Initialize trajectory collector
trajectory_collector = TrajectoryCollector(collect_every=10)  # Collect every 10 updates

# Your training loop
for iteration in range(num_iterations):
    # ... your training code ...
    
    # Collect actor parameters (after each PPO update)
    if iteration % 10 == 0:
        trajectory_collector.collect(
            state_dict=actor.state_dict(),
            loss=actor_loss.item(),
            step=iteration
        )
    
    # ... continue training ...

# After training, visualize trajectory
print("Generating trajectory visualization...")

visualizer = TrajectoryVisualizer(
    optimized_parameters=list(actor.parameters()),
    intermediate_parameters=trajectory_collector.get_parameters(),
    method="svd"  # Use learnable projection (SVD)
)

# Plot simple trajectory
visualizer.plot_trajectory(
    save_path="logs/actor_trajectory.png",
    title="Actor Learning Trajectory"
)

# Plot interactive version
visualizer.plot_trajectory_interactive(
    save_path="logs/actor_trajectory_interactive.html"
)

# Optional: Plot with loss landscape (computationally expensive)
def evaluate_actor():
    """Evaluate actor on test environment."""
    # Your evaluation code here
    # Return a scalar loss value
    return test_loss

visualizer.plot_trajectory_with_landscape(
    actor_model=actor,
    eval_function=evaluate_actor,
    save_path="logs/actor_trajectory_landscape.png",
    grid_points=15  # Lower for faster computation
)

print("Trajectory visualization complete!")
'''
    
    with open(output_path, 'w') as f:
        f.write(example_code)
    
    print(f"Example usage script created at: {output_path}")


if __name__ == "__main__":
    print("Actor Trajectory Visualization Module")
    print("=" * 60)
    print("This module provides tools for visualizing actor learning trajectories")
    print("in Deep RL using the learnable projection method from Moustafa et al. (2025)")
    print()
    print("Key classes:")
    print("  - TrajectoryCollector: Collect parameters during training")
    print("  - TrajectoryVisualizer: Project and visualize trajectories")
    print()
    print("For usage examples, see the docstring or run:")
    print("  create_example_usage_script()")
