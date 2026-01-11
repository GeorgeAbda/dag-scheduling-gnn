#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import tyro

# Ensure project root (one level up) is importable
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class Args:
    # RL config JSONs
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    # Optional override: representativeness-selected eval seeds
    wide_eval_seeds_path: str | None = "runs/datasets/wide/representativeness/selected_eval_seeds.json"
    longcp_eval_seeds_path: str | None = "runs/datasets/longcp/representativeness/selected_eval_seeds.json"

    # Which domains/policies to simulate
    domain: str = "both"  # one of {"wide","long_cp","both"}
    strategy: str = "both"  # one of {"fast","energy","both"}

    # Episodes
    repeats_per_seed: int = 1

    # Optional: global req_divisor override for dataset generation
    dataset_req_divisor: int | None = None


def _load_eval_cfg(path_str: str) -> Tuple[dict, List[int]]:
    p = Path(path_str)
    cfg = json.loads(p.read_text())
    eval_cfg = cfg.get("eval", {})
    ds_cfg = dict(eval_cfg.get("dataset", {}))
    seeds = [int(s) for s in eval_cfg.get("seeds", [])]
    return ds_cfg, seeds


def _load_selected_seeds(path_str: str | None) -> List[int] | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "selected_eval_seeds" in data:
            return [int(s) for s in data["selected_eval_seeds"]]
        if isinstance(data, list):
            return [int(s) for s in data]
    except Exception:
        return None
    return None


def _dataset_args_from_cfg(ds: dict, seed: int, req_div: int | None) -> DatasetArgs:
    return DatasetArgs(
        seed=int(seed),
        host_count=int(ds.get("host_count", 10)),
        vm_count=int(ds.get("vm_count", 10)),
        max_memory_gb=int(ds.get("max_memory_gb", 128)),
        min_cpu_speed=int(ds.get("min_cpu_speed", 500)),
        max_cpu_speed=int(ds.get("max_cpu_speed", 5000)),
        workflow_count=int(ds.get("workflow_count", 1)),
        dag_method=str(ds.get("dag_method", "gnp")),
        gnp_min_n=int(ds.get("gnp_min_n", 12)),
        gnp_max_n=int(ds.get("gnp_max_n", 30)),
        task_length_dist=str(ds.get("task_length_dist", "normal")),
        min_task_length=int(ds.get("min_task_length", 500)),
        max_task_length=int(ds.get("max_task_length", 100_000)),
        task_arrival=str(ds.get("task_arrival", "static")),
        arrival_rate=float(ds.get("arrival_rate", 3.0)),
        style=str(ds.get("style", "generic")),
        gnp_p=ds.get("gnp_p", None),
        req_divisor=int(req_div) if req_div is not None else None,
    )


def _choose_action(prev_obs, strategy: str) -> int:
    # Build ready, not-scheduled task list (skip dummy 0 and T-1)
    T = len(prev_obs.task_observations)
    V = len(prev_obs.vm_observations)
    ready_tasks = [
        i for i, t in enumerate(prev_obs.task_observations)
        if t.is_ready and (t.assigned_vm_id is None) and (i not in (0, T-1))
    ]
    if not ready_tasks:
        # Fallback: pick any valid pair from compatibilities
        if prev_obs.compatibilities:
            t_id, v_id = prev_obs.compatibilities[0]
            return int(t_id * V + v_id)
        return 0

    compat = set(prev_obs.compatibilities)
    # Priority: longer tasks first (approx CP pressure)
    ready_tasks.sort(key=lambda i: float(getattr(prev_obs.task_observations[i], "length", 0.0)), reverse=True)

    for t_id in ready_tasks:
        # Gather compatible VMs
        vms = [v for v in range(V) if (t_id, v) in compat]
        if not vms:
            continue
        # Strategy: pick VM by speed or by power
        if strategy == "fast":
            v_best = max(vms, key=lambda v: float(getattr(prev_obs.vm_observations[v], "cpu_speed_mips", 0.0)))
        elif strategy == "energy":
            v_best = min(vms, key=lambda v: float(getattr(prev_obs.vm_observations[v], "host_power_peak_watt", 0.0)))
        else:
            # Default to fast
            v_best = max(vms, key=lambda v: float(getattr(prev_obs.vm_observations[v], "cpu_speed_mips", 0.0)))
        return int(t_id * V + v_best)

    # If no valid pair for ready tasks, fallback to first compat edge
    if prev_obs.compatibilities:
        t_id, v_id = prev_obs.compatibilities[0]
        return int(t_id * V + v_id)
    return 0


def _run_one_episode(ds_args: DatasetArgs, seed: int, strategy: str) -> Tuple[float, float, float]:
    """Return (total_return, episodic_makespan_return, episodic_active_energy_return)."""
    env = CloudSchedulingGymEnvironment(dataset_args=ds_args, collect_timelines=False, compute_metrics=False, profile=False, fixed_env_seed=True)
    env = GinAgentWrapper(env)

    # Ensure step rewards include both components
    os.environ.setdefault("GIN_STEP_MAKESPAN", "1")
    os.environ.setdefault("GIN_ENERGY_WEIGHT", "1.0")
    os.environ.setdefault("GIN_MAKESPAN_WEIGHT", "1.0")

    obs_np, _ = env.reset(seed=int(seed))
    done = False
    total_ret = 0.0
    ep_energy_ret = 0.0
    ep_ms_ret = 0.0
    final_info = None

    while not done:
        prev_obs = env.prev_obs
        act = _choose_action(prev_obs, strategy)
        obs_np, r, terminated, truncated, info = env.step(int(act))
        total_ret += float(r)
        if terminated or truncated:
            final_info = info
            done = True

    if isinstance(final_info, dict):
        ep_energy_ret = float(final_info.get("active_energy_return", 0.0))
        ep_ms_ret = float(final_info.get("makespan_return", 0.0))
    env.close()
    return float(total_ret), float(ep_ms_ret), float(ep_energy_ret)


def _simulate_domain(ds_cfg: dict, seeds: Sequence[int], strategy: str, req_div: int | None) -> Dict[str, float]:
    ms_rets: List[float] = []
    en_rets: List[float] = []

    for s in seeds:
        ds_args = _dataset_args_from_cfg(ds_cfg, int(s), req_div)
        # Use fixed dataset per seed to ensure determinism across repeats
        # Run repeats per seed if requested (typically 1)
        total_ms = 0.0
        total_en = 0.0
        for _ in range(1):
            _tot, ms_ret, en_ret = _run_one_episode(ds_args, int(s), strategy)
            total_ms += ms_ret
            total_en += en_ret
        ms_rets.append(total_ms)
        en_rets.append(total_en)

    # Compare dominance by magnitude
    ms_mean = float(np.mean(np.abs(ms_rets))) if ms_rets else 0.0
    en_mean = float(np.mean(np.abs(en_rets))) if en_rets else 0.0
    return {"mean_abs_makespan_return": ms_mean, "mean_abs_active_energy_return": en_mean}


def main(a: Args) -> None:
    # Load configs and seeds
    ds_wide, seeds_wide_cfg = _load_eval_cfg(a.wide_config)
    ds_long, seeds_long_cfg = _load_eval_cfg(a.longcp_config)

    seeds_wide_sel = _load_selected_seeds(a.wide_eval_seeds_path)
    seeds_long_sel = _load_selected_seeds(a.longcp_eval_seeds_path)

    seeds_wide = seeds_wide_sel if seeds_wide_sel else seeds_wide_cfg
    seeds_long = seeds_long_sel if seeds_long_sel else seeds_long_cfg

    strategies = [a.strategy] if a.strategy in ("fast", "energy") else ["fast", "energy"]
    domains = [a.domain] if a.domain in ("wide", "long_cp") else ["wide", "long_cp"]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for strat in strategies:
        results[strat] = {}
        if "wide" in domains:
            r = _simulate_domain(ds_wide, seeds_wide, strat, a.dataset_req_divisor)
            results[strat]["wide"] = r
        if "long_cp" in domains:
            r = _simulate_domain(ds_long, seeds_long, strat, a.dataset_req_divisor)
            results[strat]["long_cp"] = r

    # Print concise summary
    def _dom(ms: float, en: float) -> str:
        if ms > en:
            return "makespan"
        if en > ms:
            return "active_energy"
        return "tie"

    for strat, per_dom in results.items():
        print(f"\nStrategy={strat}")
        for dom, vals in per_dom.items():
            ms = vals.get("mean_abs_makespan_return", 0.0)
            en = vals.get("mean_abs_active_energy_return", 0.0)
            print(f"  Domain={dom:7s} | mean|MsRet|={ms:.6g} | mean|EnRet|={en:.6g} | Dominant={_dom(ms, en)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
