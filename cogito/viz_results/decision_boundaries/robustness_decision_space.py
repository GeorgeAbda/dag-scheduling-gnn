import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

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
    model_path: str
    """Path to a trained GinAgent .pt file (state_dict)."""

    out_dir: str = "decision_boundaries"
    """Directory to write plots into."""

    device: str = "cpu"

    # Sample generation
    samples: int = 200
    base_seed: int = 1000
    vm_count: int = 3
    max_memory_gb: int = 10
    min_cpu_speed: int = 500
    max_cpu_speed: int = 5000
    task_len_low: int = 1_000
    task_len_high: int = 100_000

    # Dimensionality reduction / grid
    grid_res: int = 300


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
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:2].T
    X2 = Xc @ comp
    return X2.astype(np.float64), mu.squeeze().astype(np.float64), comp.astype(np.float64)


def nearest_label_grid(X2: np.ndarray, y: np.ndarray, grid_res: int = 300, pad: float = 0.05):
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

    labels = np.empty(grid.shape[0], dtype=y.dtype)
    chunk = max(1, 20000 // max(1, X2.shape[1]))
    for i in range(0, grid.shape[0], chunk):
        G = grid[i:i+chunk]
        d2 = ((G[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
        nn = np.argmin(d2, axis=1)
        labels[i:i+chunk] = y[nn]
    Z = labels.reshape(grid_res, grid_res)
    return XX, YY, Z


def first_decision_heuristic_vm(prev_obs) -> int:
    ready_ids = [i for i, t in enumerate(prev_obs.task_observations) if t.is_ready and t.assigned_vm_id is None]
    if not ready_ids:
        return 0
    t_id = ready_ids[0]
    candidates = [vid for tid, vid in prev_obs.compatibilities if tid == t_id]
    req_cores = prev_obs.task_observations[t_id].req_cpu_cores

    best_vm, best_cost = 0, float('inf')
    for vm_id in candidates:
        vm = prev_obs.vm_observations[vm_id]
        total_cores = max(1, int(vm.cpu_cores))
        used_cores = int(round((vm.used_cpu_fraction_cores) * total_cores))
        cpu_fraction = min(1.0, max(0.0, (used_cores + req_cores) / total_cores))
        rate = active_energy_consumption_per_mi(vm, cpu_fraction)
        if rate < best_cost:
            best_cost = rate
            best_vm = vm_id
    return best_vm


def sample_first_decision_states(args: Args, fixed_vm_seed: int = 0):
    # Fix VM configs across samples by fixing vm_rng_seed
    X_list: List[np.ndarray] = []
    y_agent_list: List[int] = []
    y_heur_list: List[int] = []

    # Build a baseline dataset to get the reference heuristic decision
    base_args = DatasetArgs(
        seed=args.base_seed,
        host_count=2,
        vm_count=args.vm_count,
        max_memory_gb=args.max_memory_gb,
        min_cpu_speed=args.min_cpu_speed,
        max_cpu_speed=args.max_cpu_speed,
        workflow_count=1,
        dag_method="linear",
        gnp_min_n=8, gnp_max_n=8,
        task_length_dist="normal",
        min_task_length=args.task_len_low,
        max_task_length=args.task_len_high,
        task_arrival="static",
        arrival_rate=3.0,
    )
    base_ds = generate_dataset(
        seed=base_args.seed,
        host_count=base_args.host_count,
        vm_count=base_args.vm_count,
        max_memory_gb=base_args.max_memory_gb,
        min_cpu_speed_mips=base_args.min_cpu_speed,
        max_cpu_speed_mips=base_args.max_cpu_speed,
        workflow_count=base_args.workflow_count,
        dag_method=base_args.dag_method,
        gnp_min_n=base_args.gnp_min_n,
        gnp_max_n=base_args.gnp_max_n,
        task_length_dist=base_args.task_length_dist,
        min_task_length=base_args.min_task_length,
        max_task_length=base_args.max_task_length,
        task_arrival=base_args.task_arrival,
        arrival_rate=base_args.arrival_rate,
        vm_rng_seed=fixed_vm_seed,
    )
    env0 = CloudSchedulingGymEnvironment(dataset=base_ds, collect_timelines=False)
    env0 = GinAgentWrapper(env0)
    obs0, _ = env0.reset(seed=args.base_seed)
    ref_vm = first_decision_heuristic_vm(env0.prev_obs)
    env0.close()

    # Now sample varied inputs where heuristic choice remains ref_vm
    rng = np.random.RandomState(args.base_seed * 7919 + 13)

    def gen_one(seed: int):
        # Vary task lengths and memory bounds; keep VMs fixed
        gmin = rng.randint(6, 20)
        gmax = gmin
        ds_args = DatasetArgs(
            seed=seed,
            host_count=2,
            vm_count=args.vm_count,
            max_memory_gb=args.max_memory_gb,
            min_cpu_speed=args.min_cpu_speed,
            max_cpu_speed=args.max_cpu_speed,
            workflow_count=1,
            dag_method="linear",
            gnp_min_n=gmin, gnp_max_n=gmax,
            task_length_dist="normal",
            min_task_length=args.task_len_low,
            max_task_length=args.task_len_high,
            task_arrival="static",
            arrival_rate=3.0,
        )
        ds = generate_dataset(
            seed=ds_args.seed,
            host_count=ds_args.host_count,
            vm_count=ds_args.vm_count,
            max_memory_gb=ds_args.max_memory_gb,
            min_cpu_speed_mips=ds_args.min_cpu_speed,
            max_cpu_speed_mips=ds_args.max_cpu_speed,
            workflow_count=ds_args.workflow_count,
            dag_method=ds_args.dag_method,
            gnp_min_n=ds_args.gnp_min_n,
            gnp_max_n=ds_args.gnp_max_n,
            task_length_dist=ds_args.task_length_dist,
            min_task_length=ds_args.min_task_length,
            max_task_length=ds_args.max_task_length,
            task_arrival=ds_args.task_arrival,
            arrival_rate=ds_args.arrival_rate,
            vm_rng_seed=fixed_vm_seed,
        )
        env = CloudSchedulingGymEnvironment(dataset=ds, collect_timelines=False)
        env = GinAgentWrapper(env)
        obs, _ = env.reset(seed=seed)
        hv = first_decision_heuristic_vm(env.prev_obs)
        num_vms = len(env.prev_obs.vm_observations)
        obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1))
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
        vm_action = int(action.item()) % num_vms
        env.close()
        return obs, hv, vm_action

    # Load agent
    device = torch.device(args.device)
    agent = load_agent(args.model_path, device)

    kept = 0
    for i in range(args.samples):
        seed_i = args.base_seed + i + 1
        obs, hv, avm = gen_one(seed_i)
        if hv == ref_vm:
            X_list.append(obs.copy())
            y_agent_list.append(avm)
            y_heur_list.append(hv)
            kept += 1
    if kept == 0:
        print("No samples where heuristic decision stayed constant; try different seeds or constraints.")
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    return np.vstack(X_list), np.array(y_agent_list, dtype=int), np.array(y_heur_list, dtype=int)


def plot_partition(X: np.ndarray, y_agent: np.ndarray, y_heur: np.ndarray, out_png: str, grid_res: int = 300):
    X2, _, _ = pca_2d(X)
    XX, YY, Z = nearest_label_grid(X2, y_agent, grid_res=grid_res)

    K = max(1, int(y_agent.max()) + 1)
    # Blue/Red themed palette suitable for papers
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

    for k in range(K):
        m = (y_agent == k)
        if m.any():
            plt.scatter(X2[m, 0], X2[m, 1], s=40, color=palette[k], edgecolor="white", linewidth=0.5, label=f"agent: VM {k}")

    # The heuristic is constant in this constructed space; annotate its VM id
    if len(y_heur) > 0:
        ref = int(y_heur[0])
        plt.text(0.02, 0.98, f"heuristic VM = {ref}", transform=plt.gca().transAxes,
                 ha="left", va="top", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.9))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", frameon=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    uniq = np.unique(y_agent)
    print(f"Saved plot: {out_png} | samples={len(y_agent)} | agent regions={len(uniq)} -> {uniq}")


def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)
    X, y_agent, y_heur = sample_first_decision_states(args)
    if X.shape[0] == 0:
        return
    out_png = os.path.join(args.out_dir, "robustness_decision_space.png")
    plot_partition(X, y_agent, y_heur, out_png, grid_res=args.grid_res)


if __name__ == "__main__":
    main(tyro.cli(Args))
