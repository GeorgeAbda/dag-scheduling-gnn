#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any

import os
import sys

import numpy as np
import torch
import tyro
from tqdm import tqdm

# Ensure project root (one level up from scripts/) is on sys.path so that 'scheduler' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant, _pick_device
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor so that, on the smallest VM, repeating
    the per-task demand (CPU cores and memory) for all tasks in a job does not
    exceed that VM's capacity.

    We approximate the number of tasks per workflow using gnp_max_n from the
    config and use the host/vm generation code so that capacities match the
    scheduler's dataset generator.
    """\

    # Basic dataset parameters
    host_count = int(dataset_cfg.get("host_count", 4))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))

    # Upper bound on tasks per workflow ("job"): gnp_max_n
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40))
    if n_tasks <= 0:
        n_tasks = 1

    # Recreate hosts/VMs using the same logic as the dataset generator so that
    # capacities are realistic for this config. We then allocate VMs to hosts so
    # VM capacities mirror host capacities, matching fixed-dataset behavior.
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

    # Safe per-task demand on the smallest VM: if every task had this demand and
    # all tasks were hypothetically placed on that VM, we would not exceed its
    # capacity.
    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)

    # Translate the safe per-task demand into a divisor relative to the maximum
    # VM capacity, matching the way generate_dataset sets max_req_*.
    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)
    
    print(f"req_div_mem={req_div_mem}, req_div_core={req_div_core}, max_mem={max_mem}, max_cores={max_cores}, n_tasks={n_tasks}")
    return int(max(req_div_mem, req_div_core))


@dataclass
class Args:
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"

    longcp_ckpt: str = ""
    wide_ckpt: str = ""

    device: str = "cpu"

    out_csv: str = "logs/hetero_eval_over_seeds.csv"


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int) -> DatasetArgs:
    req_div = _compute_optimal_req_divisor(dataset_cfg, seed)

    return DatasetArgs(
        seed=int(seed),
        host_count=int(dataset_cfg.get("host_count", 4)),
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


def _build_fixed_dataset(dataset_cfg: dict, seed: int) -> Any:
    """Build a deterministic Dataset object for a given config+seed using the same code path as the env.
    We call CloudSchedulingGymEnvironment.gen_dataset so all style/req_divisor/gnp_p logic is identical.
    """
    ds_args = _dataset_args_from_cfg(dataset_cfg, seed)
    return CloudSchedulingGymEnvironment.gen_dataset(seed, ds_args)


def _print_dataset_summary(ds: Any, domain: str, seed: int) -> None:
    try:
        vms = getattr(ds, "vms", [])
        ws = getattr(ds, "workflows", [])
        speeds = sorted({int(getattr(v, "cpu_speed_mips", 0)) for v in vms})
        cores = sorted({int(getattr(v, "cpu_cores", 0)) for v in vms})
        mems = sorted({int(getattr(v, "memory_mb", 0)) for v in vms})
        print(f"[eval][{domain}][seed={seed}] VMs: count={len(vms)} cpu_speed_mips_unique={speeds} cores_unique={cores} mem_mb_unique={mems}")

        all_tasks = []
        edges_total = 0
        for wf in ws or []:
            ts = getattr(wf, "tasks", [])
            all_tasks.extend(ts)
            for t in ts:
                edges_total += len(getattr(t, "child_ids", []) or [])
        if all_tasks:
            mem_arr = np.array([int(getattr(t, "req_memory_mb", 0)) for t in all_tasks], dtype=np.int64)
            core_arr = np.array([int(getattr(t, "req_cpu_cores", 0)) for t in all_tasks], dtype=np.int64)
            print(
                f"[eval][{domain}][seed={seed}] Jobs: workflows={len(ws)} tasks_total={len(all_tasks)} edges_total={int(edges_total)} "
                f"req_mem_mb[min,mean,max]=[{mem_arr.min()},{float(mem_arr.mean()):.1f},{mem_arr.max()}] "
                f"req_cpu_cores[min,mean,max]=[{core_arr.min()},{float(core_arr.mean()):.1f},{core_arr.max()}]"
            )
        else:
            print(f"[eval][{domain}][seed={seed}] Jobs: workflows=0 tasks_total=0 edges_total=0")
    except Exception as e:
        try:
            print(f"[eval][{domain}][seed={seed}] Dataset summary unavailable: {e}")
        except Exception:
            pass


def _load_hetero_agent(ckpt: Path, device: torch.device) -> AblationGinAgent:
    var = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    agent = AblationGinAgent(device, var, embedding_dim=16)
    state = torch.load(str(ckpt), map_location=device)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def _eval_agent_on_seeds(
    agent: AblationGinAgent,
    agent_train_domain: str,
    eval_domain: str,
    seeds: List[int],
    dataset_cfg: dict,
    device: torch.device,
    fixed_datasets: Dict[int, Any] | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    total_makespan = 0.0
    total_energy = 0.0
    total_energy_active = 0.0
    total_energy_idle = 0.0

    for s in tqdm(seeds, desc=f"{agent_train_domain}->{eval_domain}"):
        s_int = int(s)
        if fixed_datasets is not None and s_int in fixed_datasets:
            base_env = CloudSchedulingGymEnvironment(
                dataset=fixed_datasets[s_int],
                collect_timelines=False,
                compute_metrics=True,
                profile=False,
                fixed_env_seed=True,
            )
        else:
            ds_args = _dataset_args_from_cfg(dataset_cfg, seed=s_int)
            base_env = CloudSchedulingGymEnvironment(
                dataset_args=ds_args,
                collect_timelines=False,
                compute_metrics=True,
                profile=False,
                fixed_env_seed=True,
            )
        env = GinAgentWrapper(base_env)

        obs_np, _ = env.reset(seed=None)
        done = False
        final_info: dict | None = None

        while not done:
            obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            obs_np, _, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                final_info = info
                done = True

        assert env.prev_obs is not None
        mk = float(env.prev_obs.makespan())
        en_obs = float(env.prev_obs.energy_consumption())

        en_total = 0.0
        en_active = 0.0
        en_idle = 0.0
        if isinstance(final_info, dict):
            en_total = float(final_info.get("total_energy", 0.0))
            en_active = float(final_info.get("total_energy_active", 0.0))
            en_idle = float(final_info.get("total_energy_idle", 0.0))

        total_makespan += mk
        total_energy += en_total if en_total > 0.0 else en_obs
        total_energy_active += en_active
        total_energy_idle += en_idle

        rows.append(
            {
                "agent_train_domain": agent_train_domain,
                "eval_domain": eval_domain,
                "seed": int(s),
                "makespan": mk,
                "energy_total": en_total if en_total > 0.0 else en_obs,
                "energy_active": en_active,
                "energy_idle": en_idle,
            }
        )

        env.close()

    n = max(1, len(seeds))
    summary = {
        "agent_train_domain": agent_train_domain,
        "eval_domain": eval_domain,
        "seeds": float(len(seeds)),
        "mean_makespan": total_makespan / n,
        "mean_energy_total": total_energy / n,
        "mean_energy_active": (total_energy_active / n) if total_energy_active > 0.0 else 0.0,
        "mean_energy_idle": (total_energy_idle / n) if total_energy_idle > 0.0 else 0.0,
    }
    return rows, summary


def main(a: Args) -> None:
    # Resolve device (cpu/mps/cuda/auto) using the same helper as ablation_gnn.
    device = _pick_device(a.device)

    cfg_long = json.loads(Path(a.longcp_config).read_text())
    cfg_wide = json.loads(Path(a.wide_config).read_text())

    tr_long = cfg_long.get("train", {})
    tr_wide = cfg_wide.get("train", {})

    seeds_long: List[int] = [int(s) for s in tr_long.get("seeds", [])]
    seeds_wide: List[int] = [int(s) for s in tr_wide.get("seeds", [])]

    ds_long = dict(tr_long.get("dataset", {}))
    ds_wide = dict(tr_wide.get("dataset", {}))

    long_ckpt = Path(a.longcp_ckpt)
    wide_ckpt = Path(a.wide_ckpt)
    if not long_ckpt.exists():
        raise SystemExit(f"Long_cp checkpoint not found: {long_ckpt}")
    if not wide_ckpt.exists():
        raise SystemExit(f"Wide checkpoint not found: {wide_ckpt}")

    agent_long = _load_hetero_agent(long_ckpt, device)
    agent_wide = _load_hetero_agent(wide_ckpt, device)

    # Pre-generate fixed datasets per domain and seed so both agents see identical jobs
    fixed_long: Dict[int, Any] = {int(s): _build_fixed_dataset(ds_long, int(s)) for s in seeds_long}
    fixed_wide: Dict[int, Any] = {int(s): _build_fixed_dataset(ds_wide, int(s)) for s in seeds_wide}

    # Print summary once at the start for a representative seed per domain (first seed)
    if seeds_long:
        s0 = int(seeds_long[0])
        if s0 in fixed_long:
            _print_dataset_summary(fixed_long[s0], "long_cp", s0)
    if seeds_wide:
        s0 = int(seeds_wide[0])
        if s0 in fixed_wide:
            _print_dataset_summary(fixed_wide[s0], "wide", s0)

    all_rows: List[Dict[str, float]] = []
    summaries: List[Dict[str, float]] = []

    # long_cp agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(agent_long, "long_cp", "long_cp", seeds_long, ds_long, device, fixed_datasets=fixed_long)
    all_rows.extend(rows)
    summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(agent_long, "long_cp", "wide", seeds_wide, ds_wide, device, fixed_datasets=fixed_wide)
    all_rows.extend(rows)
    summaries.append(summ)

    # wide agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(agent_wide, "wide", "long_cp", seeds_long, ds_long, device, fixed_datasets=fixed_long)
    all_rows.extend(rows)
    summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(agent_wide, "wide", "wide", seeds_wide, ds_wide, device, fixed_datasets=fixed_wide)
    all_rows.extend(rows)
    summaries.append(summ)

    out_path = Path(a.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write per-seed rows
    import csv as _csv

    fieldnames = [
        "agent_train_domain",
        "eval_domain",
        "seed",
        "makespan",
        "energy_total",
        "energy_active",
        "energy_idle",
    ]

    with out_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Also write a companion summary CSV
    summary_path = out_path.with_suffix(".summary.csv")
    summary_fields = [
        "agent_train_domain",
        "eval_domain",
        "seeds",
        "mean_makespan",
        "mean_energy_total",
        "mean_energy_active",
        "mean_energy_idle",
    ]
    with summary_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print(f"Wrote per-seed metrics to: {out_path}")
    print(f"Wrote summary metrics to: {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
