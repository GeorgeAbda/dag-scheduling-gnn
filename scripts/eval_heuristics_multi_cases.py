#!/usr/bin/env python3
"""
Run the makespan and energy heuristics for both long_cp and wide configs
across multiple host-spec cases in a single script run.

Cases:
  - AL: data/host_specs.json
  - HS: data/host_specs_homospeed.json
  - HP: data/host_specs_homoPower.json
  - NA: logs/NAL_case/host_specs.json

Configs:
  - longcp: data/rl_configs/train_long_cp_p08_seeds.json
  - wide:   data/rl_configs/train_wide_p005_seeds.json

For each (config, case) pair we:
  - run the makespan and energy heuristics over all train seeds
  - measure makespan and energy via obs.energy_consumption()
  - write a per-seed CSV and a small LaTeX table
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

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
from cogito.gnn_deeprl_model.core.utils.helpers import active_energy_consumption_per_mi


@dataclass
class Args:
    """Arguments for multi-case heuristic evaluation."""
    # Optional override for req_divisor; if None, use the auto rule from configs
    dataset_req_divisor: int | None = None
    # Output directory root
    out_dir: str = "logs/heuristic_multi_cases"


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor for the dataset.

    Copied from eval_heuristics_on_seeds.py to keep behavior consistent.
    """
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
    """Greedy makespan heuristic: earliest completion time.

    Returns dict with 'makespan' and 'active_energy' (obs.energy_consumption).
    """
    step_count = 0
    max_steps = 10000

    def choose_action() -> int:
        vm_count = len(env.prev_obs.vm_observations)
        compat_set = set(env.prev_obs.compatibilities)

        best_end = float("inf")
        best_idx = 0
        found_any = False

        current_time = min(
            [getattr(vm, "next_release_time", 0.0) for vm in env.prev_obs.vm_observations]
        ) if env.prev_obs.vm_observations else 0.0

        for t_id, t in enumerate(env.prev_obs.task_observations):
            if t_id in (0, len(env.prev_obs.task_observations) - 1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue

            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                if (t_id, vm_id) not in compat_set:
                    continue
                found_any = True
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
                task_runtime = t.length / max(1e-9, vm.cpu_speed_mips)
                new_comp = start_t + task_runtime
                idx = int(t_id * vm_count + vm_id)
                if new_comp < best_end:
                    best_end = new_comp
                    best_idx = idx

        if not found_any:
            return 0
        return best_idx

    while step_count < max_steps:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        step_count += 1
        if term or trunc:
            break

    makespan = float(env.prev_obs.makespan())
    energy = float(env.prev_obs.energy_consumption())
    return {"makespan": makespan, "active_energy": energy}


def energy_heuristic_schedule(env: GinAgentWrapper) -> Dict[str, float]:
    """Greedy energy heuristic: minimum active energy rate.

    Returns dict with 'makespan' and 'active_energy' (obs.energy_consumption).
    """
    step_count = 0
    max_steps = 10000

    def choose_action() -> int:
        vm_count = len(env.prev_obs.vm_observations)
        compat_set = set(env.prev_obs.compatibilities)

        best_rate = float("inf")
        best_idx = 0
        found_any = False

        for t_id, t in enumerate(env.prev_obs.task_observations):
            if t_id in (0, len(env.prev_obs.task_observations) - 1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue

            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                if (t_id, vm_id) not in compat_set:
                    continue
                found_any = True
                rate = active_energy_consumption_per_mi(vm)
                idx = int(t_id * vm_count + vm_id)
                if rate < best_rate:
                    best_rate = rate
                    best_idx = idx

        if not found_any:
            return 0
        return best_idx

    while step_count < max_steps:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        step_count += 1
        if term or trunc:
            break

    makespan = float(env.prev_obs.makespan())
    energy = float(env.prev_obs.energy_consumption())
    return {"makespan": makespan, "active_energy": energy}


def evaluate_heuristics_for_config_and_hosts(
    config_path: Path,
    host_specs_path: Path,
    config_label: str,
    host_case_label: str,
    dataset_req_divisor: int | None,
    out_root: Path,
) -> None:
    """Run both heuristics for one (config, host_specs) pair and write outputs."""
    # Set host specs used by dataset generator
    gen_vm.HOST_SPECS_PATH = host_specs_path
    print(f"Using host specs from: {gen_vm.HOST_SPECS_PATH} for case {host_case_label}")

    cfg = json.loads(config_path.read_text())
    train_cfg = cfg.get("train", {})
    seeds = [int(s) for s in train_cfg.get("seeds", [])]
    dataset_cfg = dict(train_cfg.get("dataset", {}))

    print(f"Config {config_label}, case {host_case_label}: {config_path} with {len(seeds)} seeds")

    results: List[Dict[str, Any]] = []

    for seed in tqdm(seeds, desc=f"{config_label}_{host_case_label}"):
        ds_args = _dataset_args_from_cfg(dataset_cfg, seed=seed, override_req_divisor=dataset_req_divisor)

        # Makespan heuristic
        base_env_mk = CloudSchedulingGymEnvironment(
            dataset_args=ds_args,
            collect_timelines=False,
            compute_metrics=True,
            profile=False,
            fixed_env_seed=True,
        )
        env_mk = GinAgentWrapper(base_env_mk)
        env_mk.reset(seed=seed)
        mk_res = makespan_heuristic_schedule(env_mk)
        env_mk.close()

        # Energy heuristic
        base_env_en = CloudSchedulingGymEnvironment(
            dataset_args=ds_args,
            collect_timelines=False,
            compute_metrics=True,
            profile=False,
            fixed_env_seed=True,
        )
        env_en = GinAgentWrapper(base_env_en)
        env_en.reset(seed=seed)
        en_res = energy_heuristic_schedule(env_en)
        env_en.close()

        results.append(
            {
                "seed": int(seed),
                "mk_heuristic_makespan": mk_res["makespan"],
                "mk_heuristic_active_energy": mk_res["active_energy"],
                "en_heuristic_makespan": en_res["makespan"],
                "en_heuristic_active_energy": en_res["active_energy"],
            }
        )

    if not results:
        print(f"No results for {config_label} {host_case_label}")
        return

    # Outputs
    out_root.mkdir(parents=True, exist_ok=True)
    base_name = f"heuristic_eval_{config_label}_{host_case_label}"
    out_csv = out_root / f"{base_name}.csv"
    out_tex = out_root / f"{base_name}.tex"

    import csv

    fieldnames = [
        "seed",
        "mk_heuristic_makespan",
        "mk_heuristic_active_energy",
        "en_heuristic_makespan",
        "en_heuristic_active_energy",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Saved per-seed results to: {out_csv}")

    mk_mk = float(np.mean([r["mk_heuristic_makespan"] for r in results]))
    mk_en = float(np.mean([r["mk_heuristic_active_energy"] for r in results]))
    en_mk = float(np.mean([r["en_heuristic_makespan"] for r in results]))
    en_en = float(np.mean([r["en_heuristic_active_energy"] for r in results]))

    latex_content = r"""\begin{table}[h]
\centering
\caption{Heuristic Performance (%s, %s)}
\label{tab:heuristic_%s_%s}
\begin{tabular}{lcc}
\hline
\textbf{Heuristic} & \textbf{Makespan} & \textbf{Active Energy} \\
\hline
Makespan Heuristic & """ % (config_label, host_case_label, config_label, host_case_label) + f"{mk_mk:.2f}" + r""" & """ + f"{mk_en:.2f}" + r""" \\
Energy Heuristic & """ + f"{en_mk:.2f}" + r""" & """ + f"{en_en:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""

    with out_tex.open("w") as f:
        f.write(latex_content)
    print(f"Saved LaTeX table to: {out_tex}")

    print("\nSUMMARY", config_label, host_case_label)
    print(f"  Makespan Heuristic: makespan={mk_mk:.4f}, energy={mk_en:.4f}")
    print(f"  Energy Heuristic:   makespan={en_mk:.4f}, energy={en_en:.4f}")


def main(a: Args) -> None:
    out_root = Path(a.out_dir)

    # Define configs and cases
    configs = [
        ("longcp", Path("data/rl_configs/train_long_cp_p08_seeds.json")),
        ("wide",   Path("data/rl_configs/train_wide_p005_seeds.json")),
    ]
    cases = [
        ("AL", Path("data/host_specs.json")),
        ("HS", Path("data/host_specs_homospeed.json")),
        ("HP", Path("data/host_specs_homoPower.json")),
        ("NA", Path("logs/NAL_case/host_specs.json")),
    ]

    for cfg_label, cfg_path in configs:
        if not cfg_path.exists():
            print(f"WARNING: config not found: {cfg_path}")
            continue
        for case_label, host_path in cases:
            if not host_path.exists():
                print(f"WARNING: host specs for case {case_label} not found: {host_path}")
                continue
            evaluate_heuristics_for_config_and_hosts(
                config_path=cfg_path,
                host_specs_path=host_path,
                config_label=cfg_label,
                host_case_label=case_label,
                dataset_req_divisor=a.dataset_req_divisor,
                out_root=out_root,
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
