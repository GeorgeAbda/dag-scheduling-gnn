import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import tyro

# Ensure project root on path (so that 'import cogito.***' works)
# This script lives at: <project_root>/scheduler/viz_results/decision_boundaries/...
# We therefore need to add <project_root> (three levels up from this file's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cogito.dataset_generator.core.gen_dataset import generate_dataset
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.core.utils.helpers import active_energy_consumption_per_mi


@dataclass
class Args:
    model_path: str = "scheduler/rl_model/logs/1756790375_makespan/model_71680.pt"
    """Path to a trained GinAgent .pt file (state_dict)."""

    out_dir: str = "decision_boundaries"
    """Directory to write plots into."""

    # Linear single-workflow dataset knobs
    seed: int = 123
    vm_count: int = 3
    max_memory_gb: int = 10
    min_cpu_speed: int = 500
    max_cpu_speed: int = 5000
    task_count: Tuple[int, int] = (16, 16)  # (min,max) for linear chain length

    device: str = "cpu"

    # Visualization
    method: str = "pca"  # pca | tsne (pca is implemented; tsne falls back to pca)
    grid_res: int = 300   # mesh grid resolution for 1-NN coloring


def load_agent(model_path: str, device: torch.device) -> GinAgent:
    agent = GinAgent(device)
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD on covariance proxy
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:2].T  # (D,2)
    X2 = Xc @ comp   # (N,2)
    return X2.astype(np.float64), mu.squeeze().astype(np.float64), comp.astype(np.float64)


def nearest_label_grid(X2: np.ndarray, y: np.ndarray, grid_res: int = 300, pad: float = 0.05):
    # Bounding box with padding
    xmin, ymin = X2.min(axis=0)
    xmax, ymax = X2.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy

    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    # 1-NN assignment
    # Efficient pairwise min via broadcasting can be memory heavy; do in chunks
    labels = np.empty(grid.shape[0], dtype=y.dtype)
    chunk = max(1, 20000 // max(1, X2.shape[1]))
    for i in range(0, grid.shape[0], chunk):
        G = grid[i:i+chunk]  # (g,2)
        # distances to all samples
        d2 = ((G[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
        nn = np.argmin(d2, axis=1)
        labels[i:i+chunk] = y[nn]
    Z = labels.reshape(grid_res, grid_res)
    return XX, YY, Z


def local_energy_heuristic(prev_obs) -> int:
    """Return VM id chosen by an energy-optimal local heuristic for the single ready task."""
    # Identify single ready task (linear chain)
    ready_ids = [i for i, t in enumerate(prev_obs.task_observations) if t.is_ready and t.assigned_vm_id is None]
    if not ready_ids:
        return 0
    t_id = ready_ids[0]
    # Candidate VMs
    candidates = [vid for tid, vid in prev_obs.compatibilities if tid == t_id]
    # Task req
    req_cores = prev_obs.task_observations[t_id].req_cpu_cores

    best_vm, best_cost = 0, float('inf')
    for vm_id in candidates:
        vm = prev_obs.vm_observations[vm_id]
        total_cores = max(1, int(vm.cpu_cores))
        used_cores = int(round((vm.used_cpu_fraction_cores) * total_cores))
        cpu_fraction = min(1.0, max(0.0, (used_cores + req_cores) / total_cores))
        rate = active_energy_consumption_per_mi(vm, cpu_fraction)
        # Local proxy: energy per MI; absolute task MI cancels for comparison
        if rate < best_cost:
            best_cost = rate
            best_vm = vm_id
    return best_vm


def run_linear_episode_collect(args: Args, agent: GinAgent):
    fam_args = DatasetArgs(
        seed=args.seed,
        host_count=2,
        vm_count=args.vm_count,
        max_memory_gb=args.max_memory_gb,
        min_cpu_speed=args.min_cpu_speed,
        max_cpu_speed=args.max_cpu_speed,
        workflow_count=1,
        dag_method="linear",
        gnp_min_n=args.task_count[0],
        gnp_max_n=args.task_count[1],
        task_length_dist="normal",
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
        arrival_rate=3.0,
    )
    dataset = generate_dataset(
        seed=fam_args.seed,
        host_count=fam_args.host_count,
        vm_count=fam_args.vm_count,
        max_memory_gb=fam_args.max_memory_gb,
        min_cpu_speed_mips=fam_args.min_cpu_speed,
        max_cpu_speed_mips=fam_args.max_cpu_speed,
        workflow_count=fam_args.workflow_count,
        dag_method=fam_args.dag_method,
        gnp_min_n=fam_args.gnp_min_n,
        gnp_max_n=fam_args.gnp_max_n,
        task_length_dist=fam_args.task_length_dist,
        min_task_length=fam_args.min_task_length,
        max_task_length=fam_args.max_task_length,
        task_arrival=fam_args.task_arrival,
        arrival_rate=fam_args.arrival_rate,
        vm_rng_seed=0,
    )

    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=args.seed)

    X, y_agent, y_heur, steps = [], [], [], 0
    num_vms = len(env.prev_obs.vm_observations)

    while True:
        steps += 1
        # Record state vector and labels
        X.append(obs.copy())
        vm_heur = local_energy_heuristic(env.prev_obs)

        obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1))
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
        vm_action = int(action.item())
        vm_agent = vm_action % num_vms

        y_agent.append(vm_agent)
        y_heur.append(vm_heur)

        # Step environment
        obs, _, terminated, truncated, _ = env.step(vm_action)
        if terminated or truncated:
            break

    env.close()
    return np.vstack(X), np.array(y_agent, dtype=int), np.array(y_heur, dtype=int)


def plot_boundaries(X: np.ndarray, y_agent: np.ndarray, y_heur: np.ndarray, out_png: str, grid_res: int = 300):
    # Dimensionality reduction
    X2, mu, comp = pca_2d(X)
    # 1-NN regions from agent labels
    XX, YY, Z = nearest_label_grid(X2, y_agent, grid_res=grid_res)

    K = max(1, int(y_agent.max()) + 1)
    # Blue/Red themed palette: alternate blue/red shades for clarity in papers
    blue_shades = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
    red_shades = ["#67000d", "#cb181d", "#fb6a4a", "#fcbba1"]
    palette = []
    i = 0
    while len(palette) < K:
        palette.append(blue_shades[i % len(blue_shades)])
        if len(palette) < K:
            palette.append(red_shades[i % len(red_shades)])
        i += 1
    cmap = ListedColormap(palette[:K])

    plt.figure(figsize=(7.5, 6.0))
    plt.contourf(XX, YY, Z, levels=K, cmap=cmap, alpha=0.35, antialiased=True)

    # Scatter states colored by agent action
    for k in range(K):
        mask = (y_agent == k)
        if mask.any():
            plt.scatter(X2[mask, 0], X2[mask, 1], s=40, color=palette[k], edgecolor="white", linewidth=0.5, label=f"agent: VM {k}")

    # Highlight disagreements with heuristic (black edge)
    disagree = (y_agent != y_heur)
    if disagree.any():
        plt.scatter(X2[disagree, 0], X2[disagree, 1], s=80, facecolors='none', edgecolors='black', linewidths=1.2, label="agent ≠ heuristic")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", frameon=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    # Print a quick summary
    disagree_rate = float(np.mean(disagree)) if len(disagree) > 0 else 0.0
    print(f"Saved plot: {out_png} | steps={len(y_agent)} | disagreements={disagree.sum()} ({disagree_rate*100:.1f}%)")


def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    agent = load_agent(args.model_path, device)

    X, y_agent, y_heur = run_linear_episode_collect(args, agent)
    out_png = os.path.join(args.out_dir, "decision_boundaries_linear.png")
    plot_boundaries(X, y_agent, y_heur, out_png, grid_res=args.grid_res)


if __name__ == "__main__":
    main(tyro.cli(Args))
