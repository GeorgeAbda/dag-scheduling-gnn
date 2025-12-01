from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Dict, Optional

import numpy as np
import torch
import tyro

# Ensure project root on path (same pattern as train.py)
this_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(this_dir, ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.dataset_generator.core.models import Dataset
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.robustness.reference_builder import build_reference_set, RefOptions
from scheduler.rl_model.robustness.metrics import (
    ideal_nadir,
    normalize_points,
    hypervolume_2d,
    pareto_non_dominated,
    dominance_rate,
    pareto_regret,
    cvar,
)
from scheduler.rl_model.train import Args as TrainArgs


Point = Tuple[float, float]


@dataclass
class GridArgs:
    model_path: str
    out_dir: str = "logs/robustness"

    # Dataset base
    seed: int = 123
    host_count: int = 4
    vm_count: int = 10
    max_memory_gb: int = 10
    min_cpu_speed: int = 500
    max_cpu_speed: int = 5000
    workflow_count: int = 10
    dag_method: str = "linear"  # {"gnp","linear","pegasus"}
    gnp_min_n: int = 12
    gnp_max_n: int = 24
    task_length_dist: str = "normal"
    min_task_length: int = 500
    max_task_length: int = 100_000
    task_arrival: str = "static"
    arrival_rate: float = 0.0

    # Robustness knobs: scale task requirements
    alpha_cpu_values: Sequence[float] = tuple(np.linspace(0.2, 0.5, 2))
    alpha_mem_values: Sequence[float] = tuple(np.linspace(0.2, 0.3, 2))
    seeds_per_cell: int = 5

    # Reference options
    recompute_ref_per_severity: bool = True
    hv_reference_point: Tuple[float, float] = (1.0, 1.0)
    regret_thetas: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11))

    device: str = "cpu"
    collect_timelines: bool = False
    use_train_defaults: bool = False


def load_agent(model_path: str, device: torch.device) -> GinAgent:
    agent = GinAgent(device)
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def apply_scaling(dataset: Dataset, alpha_cpu: float, alpha_mem: float) -> Dataset:
    # Deep copy not strictly necessary if we don't reuse base
    for wf in dataset.workflows:
        for t in wf.tasks:
            # Keep dummy tasks unscaled by convention (ids 0 and last). Safe guard by length==0? Here we honor ids.
            if t.id in (0, wf.tasks[-1].id):
                continue
            t.req_cpu_cores = max(0.5, int(round(t.req_cpu_cores * alpha_cpu)))
            t.req_memory_mb = max(0.5, int(round(t.req_memory_mb * alpha_mem)))

            # print(f"Task {t.id}: {t.req_cpu_cores} cores, {t.req_memory_mb} MB")
    return dataset


def run_agent_once(agent: GinAgent, dataset: Dataset, seed: int) -> Point:
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=seed)
    final_info = None
    while True:
        obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1)).to(agent.device)
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
        vm_action = int(action.item())
        obs, _, terminated, truncated, info = env.step(vm_action)
        if terminated or truncated:
            final_info = info
            break
    makespan = env.prev_obs.makespan()
    total_energy = float(final_info.get("total_energy", env.prev_obs.energy_consumption())) if isinstance(final_info, dict) else env.prev_obs.energy_consumption()
    env.close()
    return float(makespan), float(total_energy)


def evaluate_grid(args: GridArgs):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    agent = load_agent(args.model_path, device)

    # Pre-generate a base dataset to ensure same VM pool across grid (VM RNG held fixed by vm_rng_seed=0)
    if args.use_train_defaults:
        # Mirror dataset defaults used by train.py
        train_defaults = TrainArgs()
        base_ds_args = train_defaults.dataset
        # Override only seed so grid remains reproducible per cell
        base_ds_args.seed = args.seed
    else:
        base_ds_args = DatasetArgs(
            seed=args.seed,
            host_count=args.host_count,
            vm_count=args.vm_count,
            max_memory_gb=args.max_memory_gb,
            min_cpu_speed=args.min_cpu_speed,
            max_cpu_speed=args.max_cpu_speed,
            workflow_count=args.workflow_count,
            dag_method=args.dag_method,
            gnp_min_n=args.gnp_min_n,
            gnp_max_n=args.gnp_max_n,
            task_length_dist=args.task_length_dist,
            min_task_length=args.min_task_length,
            max_task_length=args.max_task_length,
            task_arrival=args.task_arrival,
            arrival_rate=args.arrival_rate,
        )

    # Prepare CSV outputs
    csv_path = os.path.join(args.out_dir, "grid_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(
            "alpha_cpu,alpha_mem,seed,agent_m,agent_e,ref_hv,agent_hv,hv_ratio,dom_rate,regret_mean,regret_max,regret_q90\n"
        )

    # Storage for contour aggregation
    grid_stats: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for a_cpu in args.alpha_cpu_values:
        for a_mem in args.alpha_mem_values:
            key = (float(a_cpu), float(a_mem))
            grid_stats[key] = {
                "hv_ratio": [],
                "dom_rate": [],
                "regret_mean": [],
                "regret_max": [],
                "regret_q90": [],
                "energy_regret": [],  # normalized energy regret (agent energy - best ref energy)
            }
            for s in range(args.seeds_per_cell):
                # Generate dataset for this seed
                ds = generate_dataset(
                    seed=base_ds_args.seed + s,
                    host_count=base_ds_args.host_count,
                    vm_count=base_ds_args.vm_count,
                    max_memory_gb=base_ds_args.max_memory_gb,
                    min_cpu_speed_mips=base_ds_args.min_cpu_speed,
                    max_cpu_speed_mips=base_ds_args.max_cpu_speed,
                    workflow_count=base_ds_args.workflow_count,
                    dag_method=base_ds_args.dag_method,
                    gnp_min_n=base_ds_args.gnp_min_n,
                    gnp_max_n=base_ds_args.gnp_max_n,
                    task_length_dist=base_ds_args.task_length_dist,
                    min_task_length=base_ds_args.min_task_length,
                    max_task_length=base_ds_args.max_task_length,
                    task_arrival=base_ds_args.task_arrival,
                    arrival_rate=base_ds_args.arrival_rate,
                    vm_rng_seed=0,
                )
                ds = apply_scaling(ds, a_cpu, a_mem)
                print(f'building reference set for alpha_cpu={a_cpu}, alpha_mem={a_mem}')
                # Build reference at this severity or reuse base
                if args.recompute_ref_per_severity:
                    ref_pts = build_reference_set(ds, RefOptions(cp_sat_timeout_s=60, alpha_cpu=a_cpu, alpha_mem=a_mem))
                else:
                    # Build once on unscaled dataset and reuse ideal/nadir; still compute regret to that front
                    ref_pts = build_reference_set(ds, RefOptions(cp_sat_timeout_s=60, alpha_cpu=a_cpu, alpha_mem=a_mem))
                

                print(f'normalizing points for alpha_cpu={a_cpu}, alpha_mem={a_mem}')
                # Ideal-nadir for normalization from ref
                ideal, nadir = ideal_nadir(ref_pts)
                ref_norm = normalize_points(ref_pts, ideal, nadir)


                print(f'evaluating agent for alpha_cpu={a_cpu}, alpha_mem={a_mem}')
                # Evaluate agent
                a_m, a_e = run_agent_once(agent, ds, seed=base_ds_args.seed + s)
                agent_norm = normalize_points([(a_m, a_e)], ideal, nadir)[0]

                # Hypervolume
                ref_hv = hypervolume_2d(ref_norm, reference=args.hv_reference_point)
                agent_hv = hypervolume_2d([agent_norm], reference=args.hv_reference_point)
                hv_ratio = (agent_hv / ref_hv) if ref_hv > 0 else 0.0

                # Dominance rate of agent point vs ref set (0 or 1 for a single point)
                dom_rate = dominance_rate([agent_norm], ref_norm)

                # Regret metrics across theta grid
                reg = pareto_regret(agent_norm, ref_norm, args.regret_thetas)

                # Energy-only robustness regret (normalized energy):
                # distance from the best (lowest) normalized energy on the reference front
                if ref_norm:
                    best_ref_energy = min(y for (_x, y) in ref_norm)
                    energy_regret = max(0.0, agent_norm[1] - best_ref_energy)
                else:
                    energy_regret = float("nan")

                # Log row
                with open(csv_path, "a") as f:
                    f.write(
                        f"{a_cpu},{a_mem},{base_ds_args.seed + s},{a_m:.6f},{a_e:.6f},{ref_hv:.6f},{agent_hv:.6f},{hv_ratio:.6f},{dom_rate:.6f},{reg['mean']:.6f},{reg['max']:.6f},{reg['q90']:.6f}\n"
                    )

                # Aggregate
                grid_stats[key]["hv_ratio"].append(hv_ratio)
                grid_stats[key]["dom_rate"].append(dom_rate)
                grid_stats[key]["regret_mean"].append(reg["mean"])
                grid_stats[key]["regret_max"].append(reg["max"])
                grid_stats[key]["regret_q90"].append(reg["q90"])
                grid_stats[key]["energy_regret"].append(energy_regret)

                # No per-cell Pareto front plotting or raw dumps

    # Save aggregated summaries and plots
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    agg_rows = []
    for (a_cpu, a_mem), d in grid_stats.items():
        agg_rows.append({
            "alpha_cpu": a_cpu,
            "alpha_mem": a_mem,
            "hv_ratio_mean": float(np.mean(d["hv_ratio"])) if d["hv_ratio"] else np.nan,
            "hv_ratio_cvar90": cvar(d["hv_ratio"], alpha=0.9) if d["hv_ratio"] else np.nan,
            "dom_rate_mean": float(np.mean(d["dom_rate"])) if d["dom_rate"] else np.nan,
            "regret_mean": float(np.mean(d["regret_mean"])) if d["regret_mean"] else np.nan,
            "regret_cvar90": cvar(d["regret_mean"], alpha=0.9) if d["regret_mean"] else np.nan,
            "energy_regret_mean": float(np.nanmean(d["energy_regret"])) if d["energy_regret"] else np.nan,
        })
    agg = pd.DataFrame(agg_rows)
    agg_path = os.path.join(args.out_dir, "grid_metrics_aggregated.csv")
    agg.to_csv(agg_path, index=False)

    # Pivot to matrices for contour plots
    def pivot_metric(name: str) -> np.ndarray:
        mat = agg.pivot(index="alpha_cpu", columns="alpha_mem", values=name).sort_index(axis=0).sort_index(axis=1).values
        return mat

    X = sorted(set(agg["alpha_mem"]))
    Y = sorted(set(agg["alpha_cpu"]))
    Xv, Yv = np.meshgrid(X, Y)

    # Plot HV ratio mean
    try:
        plt.figure(figsize=(6.4, 4.8))
        Z = pivot_metric("hv_ratio_mean")
        cs = plt.contourf(Xv, Yv, Z, levels=np.linspace(np.nanmin(Z), np.nanmax(Z), 11), cmap="viridis")
        plt.colorbar(cs, label="HV ratio (mean)")
        plt.xlabel("alpha_mem")
        plt.ylabel("alpha_cpu")
        plt.title("Robustness: HV ratio vs scaling")
        out_png = os.path.join(args.out_dir, "contour_hv_ratio_mean.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting hv_ratio_mean failed: {e}")

    # Plot regret mean
    try:
        plt.figure(figsize=(6.4, 4.8))
        Z = pivot_metric("regret_mean")
        cs = plt.contourf(Xv, Yv, Z, levels=11, cmap="magma")
        plt.colorbar(cs, label="Pareto regret (mean)")
        plt.xlabel("alpha_mem")
        plt.ylabel("alpha_cpu")
        plt.title("Robustness: Pareto regret vs scaling")
        out_png = os.path.join(args.out_dir, "contour_regret_mean.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting regret_mean failed: {e}")

    # Plot energy-only robustness regret (normalized)
    try:
        plt.figure(figsize=(6.4, 4.8))
        Z = pivot_metric("energy_regret_mean")
        # If Z is constant or invalid, this may raise; rely on try/except like above
        cs = plt.contourf(Xv, Yv, Z, levels=11, cmap="plasma")
        plt.colorbar(cs, label="Energy regret (mean, normalized)")
        plt.xlabel("alpha_mem")
        plt.ylabel("alpha_cpu")
        plt.title("Robustness: Energy-only regret vs scaling")
        out_png = os.path.join(args.out_dir, "contour_energy_regret_mean.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting energy_regret_mean failed: {e}")

    # No combined Pareto front plot

    print(f"Saved CSV: {csv_path}")
    print(f"Saved aggregated CSV: {agg_path}")


if __name__ == "__main__":
    evaluate_grid(tyro.cli(GridArgs))
