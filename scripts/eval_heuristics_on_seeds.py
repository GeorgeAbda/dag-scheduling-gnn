#!/usr/bin/env python3
"""
Evaluate makespan and energy heuristics on initial observations.

Given a seed config JSON and host specs, this script:
1. Generates 100 jobs (workflows) from the config seeds
2. For each job, runs two heuristics on the initial observation:
   - Makespan heuristic: greedy earliest-completion-time scheduling
   - Energy heuristic: greedy minimum-energy-rate scheduling
3. Computes final makespan and active energy for each heuristic solution
4. Saves results to CSV and generates a LaTeX table with mean values
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

import numpy as np
import tyro
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cogito.dataset_generator.core.gen_vm as gen_vm
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable


@dataclass
class Args:
    """Arguments for heuristic evaluation."""
    config_path: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    host_specs_path: str = "data/host_specs.json"
    out_csv: str = "logs/heuristic_eval_results.csv"
    out_tex: str = "logs/heuristic_eval_table.tex"
    dataset_req_divisor: int | None = None


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor for the dataset."""
    from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
    
    host_count = int(dataset_cfg.get("host_count", 10))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40))
    
    if n_tasks <= 0:
        n_tasks = 1
    
    rng = np.random.RandomState(int(seed))
    hosts = generate_hosts(n=host_count, rng=rng)
    vms = generate_vms(
        n=vm_count,
        max_memory_gb=max_memory_gb,
        min_cpu_speed_mips=min_cpu_speed,
        max_cpu_speed_mips=max_cpu_speed,
        rng=rng,
    )
    allocate_vms(vms, hosts, rng)
    
    if not vms:
        return 1
    
    mem_caps = [int(getattr(vm, "memory_mb", 0)) for vm in vms]
    core_caps = [int(max(1, getattr(vm, "cpu_cores", 1))) for vm in vms]
    
    min_mem = max(1, min(mem_caps))
    max_mem = max(mem_caps)
    min_cores = max(1, min(core_caps))
    max_cores = max(core_caps)
    
    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)
    
    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)
    
    return int(max(req_div_mem, req_div_core))


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int, override_req_divisor: int | None = None) -> DatasetArgs:
    """Build DatasetArgs from config dictionary."""
    if override_req_divisor is not None:
        req_div = override_req_divisor
    else:
        req_div = _compute_optimal_req_divisor(dataset_cfg, seed)
    
    return DatasetArgs(
        seed=int(seed),
        host_count=int(dataset_cfg.get("host_count", 10)),
        vm_count=int(dataset_cfg.get("vm_count", 10)),
        max_memory_gb=int(dataset_cfg.get("max_memory_gb", 10)),
        min_cpu_speed=int(dataset_cfg.get("min_cpu_speed", 500)),
        max_cpu_speed=int(dataset_cfg.get("max_cpu_speed", 5000)),
        workflow_count=int(dataset_cfg.get("workflow_count", 1)),
        dag_method=str(dataset_cfg.get("dag_method", "gnp")),
        gnp_min_n=int(dataset_cfg.get("gnp_min_n", 10)),
        gnp_max_n=int(dataset_cfg.get("gnp_max_n", 40)),
        task_length_dist=str(dataset_cfg.get("task_length_dist", "normal")),
        min_task_length=int(dataset_cfg.get("min_task_length", 500)),
        max_task_length=int(dataset_cfg.get("max_task_length", 100_000)),
        task_arrival=str(dataset_cfg.get("task_arrival", "static")),
        arrival_rate=float(dataset_cfg.get("arrival_rate", 3.0)),
        style=str(dataset_cfg.get("style", "generic")),
        gnp_p=dataset_cfg.get("gnp_p", None),
        req_divisor=int(req_div),
    )


def makespan_heuristic_schedule(env: GinAgentWrapper) -> Dict[str, float]:
    """
    Greedy makespan heuristic: schedule ready tasks to minimize completion time.
    Returns dict with 'makespan' and 'active_energy'.
    """
    final_info = None
    step_count = 0
    max_steps = 10000  # Safety limit
    
    def choose_action() -> int:
        """Pick (task, VM) with earliest completion time."""
        vm_count = len(env.prev_obs.vm_observations)
        compat_set = set(env.prev_obs.compatibilities)
        
        best_end = float('inf')
        best_idx = 0
        found_any = False
        
        # Use the earliest VM release time as the current scheduling horizon
        current_time = min(
            [getattr(vm, "next_release_time", 0.0) for vm in env.prev_obs.vm_observations]
        ) if env.prev_obs.vm_observations else 0.0
        
        for t_id, t in enumerate(env.prev_obs.task_observations):
            # Skip dummy tasks and already scheduled tasks
            if t_id in (0, len(env.prev_obs.task_observations) - 1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue
            
            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                # Check compatibility
                if (t_id, vm_id) not in compat_set:
                    continue
                
                found_any = True
                # Capacity-aware earliest feasible start time
                req_mem = int(t.req_memory_mb)
                req_cores = int(t.req_cpu_cores)
                avail_now = (vm.available_memory_mb >= req_mem) and (max(1, vm.available_cpu_cores) >= req_cores)
                if avail_now:
                    start_t = current_time
                else:
                    start_t = max(
                        getattr(vm, "next_release_time", 0.0),
                        getattr(vm, "next_core_release_time", 0.0),
                    )
                # Estimate completion time on this VM
                task_runtime = t.length / max(1e-9, vm.cpu_speed_mips)
                new_comp = start_t + task_runtime
                
                idx = int(t_id * vm_count + vm_id)
                if new_comp < best_end:
                    best_end = new_comp
                    best_idx = idx
        
        if not found_any:
            # Fallback: return first valid action
            return 0
        
        return best_idx
    
    # Run the heuristic
    while step_count < max_steps:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        step_count += 1
        if term or trunc:
            final_info = info
            break

    makespan = env.prev_obs.makespan()
    # Report energy using the observation-based estimate, consistent with
    # the agent's training objective.
    energy = float(env.prev_obs.energy_consumption())

    return {
        "makespan": float(makespan),
        # Kept the key name for backwards-compatibility; it now holds
        # obs.energy_consumption(), not total_energy_active.
        "active_energy": float(energy),
    }


def energy_heuristic_schedule(env: GinAgentWrapper) -> Dict[str, float]:
    """
    Greedy energy heuristic: schedule ready tasks to minimize energy consumption.
    Returns dict with 'makespan' and 'active_energy'.
    """
    final_info = None
    step_count = 0
    max_steps = 10000  # Safety limit
    
    def choose_action() -> int:
        """Pick (task, VM) with minimum energy rate."""
        vm_count = len(env.prev_obs.vm_observations)
        compat_set = set(env.prev_obs.compatibilities)
        
        best_rate = float('inf')
        best_idx = 0
        found_any = False
        
        for t_id, t in enumerate(env.prev_obs.task_observations):
            # Skip dummy tasks and already scheduled tasks
            if t_id in (0, len(env.prev_obs.task_observations) - 1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue
            
            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                # Check compatibility
                if (t_id, vm_id) not in compat_set:
                    continue
                
                found_any = True
                # Compute energy rate
                rate = active_energy_consumption_per_mi(vm)
                
                idx = int(t_id * vm_count + vm_id)
                if rate < best_rate:
                    best_rate = rate
                    best_idx = idx
        
        if not found_any:
            # Fallback: return first valid action
            return 0
        
        return best_idx
    
    # Run the heuristic
    while step_count < max_steps:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        step_count += 1
        if term or trunc:
            final_info = info
            break

    makespan = env.prev_obs.makespan()
    # Use the same observation-based energy metric as in the makespan heuristic.
    energy = float(env.prev_obs.energy_consumption())

    return {
        "makespan": float(makespan),
        "active_energy": float(energy),
    }


def evaluate_heuristics_on_seed(
    seed: int,
    dataset_cfg: dict,
    override_req_divisor: int | None = None,
) -> Dict[str, Any]:
    """
    Evaluate both heuristics on a single seed.
    Returns dict with results for both heuristics.
    """
    ds_args = _dataset_args_from_cfg(dataset_cfg, seed=seed, override_req_divisor=override_req_divisor)
    
    # Evaluate makespan heuristic
    base_env_mk = CloudSchedulingGymEnvironment(
        dataset_args=ds_args,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
        fixed_env_seed=True,
    )
    env_mk = GinAgentWrapper(base_env_mk)
    env_mk.reset(seed=seed)
    mk_results = makespan_heuristic_schedule(env_mk)
    env_mk.close()
    
    # Evaluate energy heuristic
    base_env_en = CloudSchedulingGymEnvironment(
        dataset_args=ds_args,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
        fixed_env_seed=True,
    )
    env_en = GinAgentWrapper(base_env_en)
    env_en.reset(seed=seed)
    en_results = energy_heuristic_schedule(env_en)
    env_en.close()
    
    return {
        "seed": int(seed),
        "mk_heuristic_makespan": mk_results["makespan"],
        "mk_heuristic_active_energy": mk_results["active_energy"],
        "en_heuristic_makespan": en_results["makespan"],
        "en_heuristic_active_energy": en_results["active_energy"],
    }


def main(args: Args) -> None:
    """Main evaluation function."""
    # Override host specs path if provided
    if args.host_specs_path:
        gen_vm.HOST_SPECS_PATH = Path(args.host_specs_path)
        print(f"Using host specs from: {gen_vm.HOST_SPECS_PATH}")
    
    # Load config
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = json.loads(config_path.read_text())
    train_cfg = cfg.get("train", {})
    seeds = [int(s) for s in train_cfg.get("seeds", [])]
    dataset_cfg = dict(train_cfg.get("dataset", {}))
    
    print(f"Loaded config: {config_path}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Dataset style: {dataset_cfg.get('style', 'unknown')}")
    
    # Evaluate heuristics on all seeds
    results: List[Dict[str, Any]] = []
    
    for seed in tqdm(seeds, desc="Evaluating heuristics"):
        try:
            result = evaluate_heuristics_on_seed(
                seed=seed,
                dataset_cfg=dataset_cfg,
                override_req_divisor=args.dataset_req_divisor,
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating seed {seed}: {e}")
            continue
    
    if not results:
        print("No results collected. Exiting.")
        return
    
    # Save results to CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    import csv
    fieldnames = [
        "seed",
        "mk_heuristic_makespan",
        "mk_heuristic_active_energy",
        "en_heuristic_makespan",
        "en_heuristic_active_energy",
    ]
    
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"Saved results to: {out_csv}")
    
    # Compute mean values
    mk_makespan_mean = np.mean([r["mk_heuristic_makespan"] for r in results])
    mk_energy_mean = np.mean([r["mk_heuristic_active_energy"] for r in results])
    en_makespan_mean = np.mean([r["en_heuristic_makespan"] for r in results])
    en_energy_mean = np.mean([r["en_heuristic_active_energy"] for r in results])
    
    # Generate LaTeX table
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    
    latex_content = r"""\begin{table}[h]
\centering
\caption{Heuristic Performance on Initial Observations}
\label{tab:heuristic_eval}
\begin{tabular}{lcc}
\hline
\textbf{Heuristic} & \textbf{Makespan} & \textbf{Active Energy} \\
\hline
Makespan Heuristic & """ + f"{mk_makespan_mean:.2f}" + r""" & """ + f"{mk_energy_mean:.2f}" + r""" \\
Energy Heuristic & """ + f"{en_makespan_mean:.2f}" + r""" & """ + f"{en_energy_mean:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    
    with out_tex.open("w") as f:
        f.write(latex_content)
    
    print(f"Saved LaTeX table to: {out_tex}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Heuristic':<25} {'Makespan':<15} {'Active Energy':<15}")
    print("-"*60)
    print(f"{'Makespan Heuristic':<25} {mk_makespan_mean:<15.2f} {mk_energy_mean:<15.2f}")
    print(f"{'Energy Heuristic':<25} {en_makespan_mean:<15.2f} {en_energy_mean:<15.2f}")
    print("="*60)


if __name__ == "__main__":
    main(tyro.cli(Args))
