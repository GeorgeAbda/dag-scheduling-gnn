#!/usr/bin/env python3
"""
Evaluate hetero agents across all four controlled cases in a single run:
  - AL:  logs/AL_case/
  - NAL: logs/NAL_case/
  - HP:  logs/HP_controlled/
  - HS:  logs/HS_controlled/

Each case has its own:
  - long_cp checkpoint
  - wide checkpoint
  - host_specs.json

For each case, we evaluate:
  - long_cp agent on long_cp seeds
  - long_cp agent on wide seeds
  - wide agent on long_cp seeds
  - wide agent on wide seeds

Outputs per case:
  - {case}_hetero_eval.csv (per-seed metrics)
  - {case}_hetero_eval.summary.csv (mean metrics)
  - {case}_hetero_eval.meta.json (metadata)
"""

from __future__ import annotations

import json
import os
import sys
import platform
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Sequence

import numpy as np
import torch
import tyro
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cogito.dataset_generator.core.gen_vm as gen_vm
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from cogito.gnn_deeprl_model.ablation_gnn import AblationGinAgent, AblationVariant, _pick_device
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class Args:
    """Arguments for multi-case hetero agent evaluation."""
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"
    device: str = "cpu"
    dataset_req_divisor: int | None = None
    eval_repeats_per_seed: int = 5
    out_dir: str = "logs/hetero_eval_all_cases"


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor for the dataset."""
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


def _load_hetero_agent(ckpt: Path, device: torch.device) -> AblationGinAgent:
    """Load hetero agent with auto-detected architecture from checkpoint."""
    var = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    state = torch.load(str(ckpt), map_location=device)

    hidden_dim = 64
    embedding_dim = 32

    if "actor.network.task_encoder.0.weight" in state:
        hidden_dim = state["actor.network.task_encoder.0.weight"].shape[0]

    if "actor.network.task_encoder.6.weight" in state:
        embedding_dim = state["actor.network.task_encoder.6.weight"].shape[0]

    print(f"[load_ckpt] {ckpt.name}: detected hidden_dim={hidden_dim}, embedding_dim={embedding_dim}")

    agent = AblationGinAgent(device, var, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def _eval_agent_on_seeds(
    agent: AblationGinAgent,
    agent_train_domain: str,
    eval_domain: str,
    seeds: Sequence[int],
    dataset_cfg: dict,
    device: torch.device,
    override_req_divisor: int | None = None,
    repeats_per_seed: int = 1,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    total_makespan = 0.0
    total_energy = 0.0
    total_energy_active = 0.0
    total_energy_idle = 0.0
    total_entropy = 0.0

    for s in tqdm(seeds, desc=f"{agent_train_domain}->{eval_domain}"):
        s_int = int(s)

        seed_mk = 0.0
        seed_en_total = 0.0
        seed_en_active = 0.0
        seed_en_idle = 0.0
        seed_entropy = 0.0

        n_rep = max(1, int(repeats_per_seed))

        for r in range(n_rep):
            ds_args = _dataset_args_from_cfg(dataset_cfg, seed=s_int, override_req_divisor=override_req_divisor)
            base_env = CloudSchedulingGymEnvironment(
                dataset_args=ds_args,
                collect_timelines=False,
                compute_metrics=True,
                profile=False,
                fixed_env_seed=True,
            )

            env = GinAgentWrapper(base_env)
            env_seed = s_int
            obs_np, _ = env.reset(seed=env_seed)
            done = False
            final_info: dict | None = None

            ep_entropy_sum = 0.0
            ep_steps = 0

            while not done:
                obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action, _, entropy, _ = agent.get_action_and_value(obs_t, deterministic=False)
                try:
                    ep_entropy_sum += float(entropy.mean().item())
                except Exception:
                    pass
                ep_steps += 1

                obs_np, _, terminated, truncated, info = env.step(int(action.item()))
                if terminated or truncated:
                    final_info = info
                    done = True

            assert env.prev_obs is not None
            mk = float(env.prev_obs.makespan())
            en_obs = float(env.prev_obs.energy_consumption())

            en_total = en_obs
            en_active = en_obs
            en_idle = 0.0

            seed_mk += mk
            seed_en_total += en_total
            seed_en_active += en_active
            seed_en_idle += en_idle

            if ep_steps > 0:
                seed_entropy += ep_entropy_sum / float(ep_steps)

            env.close()

        mk_mean = seed_mk / float(n_rep)
        en_total_mean = seed_en_total / float(n_rep)
        en_active_mean = seed_en_active / float(n_rep)
        en_idle_mean = seed_en_idle / float(n_rep)
        entropy_mean = seed_entropy / float(n_rep) if n_rep > 0 else 0.0

        total_makespan += mk_mean
        total_energy += en_total_mean
        total_energy_active += en_active_mean
        total_energy_idle += en_idle_mean
        total_entropy += entropy_mean

        rows.append(
            {
                "agent_train_domain": agent_train_domain,
                "eval_domain": eval_domain,
                "seed": int(s),
                "makespan": mk_mean,
                "energy_total": en_total_mean,
                "energy_active": en_active_mean,
                "energy_idle": en_idle_mean,
                "entropy": entropy_mean,
            }
        )

    n = max(1, len(seeds))
    summary = {
        "agent_train_domain": agent_train_domain,
        "eval_domain": eval_domain,
        "seeds": float(len(seeds)),
        "mean_makespan": total_makespan / n,
        "mean_energy_total": total_energy / n,
        "mean_energy_active": (total_energy_active / n) if total_energy_active > 0.0 else 0.0,
        "mean_energy_idle": (total_energy_idle / n) if total_energy_idle > 0.0 else 0.0,
        "mean_entropy": (total_entropy / n) if total_entropy > 0.0 else 0.0,
    }
    return rows, summary


def eval_one_case(
    case_label: str,
    longcp_ckpt: Path,
    wide_ckpt: Path,
    host_specs: Path,
    mixed_ckpt: Path | None,
    longcp_config: Path,
    wide_config: Path,
    device: torch.device,
    dataset_req_divisor: int | None,
    eval_repeats: int,
    out_root: Path,
) -> None:
    """Evaluate one case (AL/NAL/HP/HS) with its checkpoints and host specs."""
    print(f"\n{'='*60}")
    print(f"EVALUATING CASE: {case_label}")
    print(f"{'='*60}")

    # Set host specs
    gen_vm.HOST_SPECS_PATH = host_specs
    print(f"Using host specs: {gen_vm.HOST_SPECS_PATH}")

    # Load configs
    cfg_long = json.loads(longcp_config.read_text())
    cfg_wide = json.loads(wide_config.read_text())

    # Support both old format (train.seeds) and new format (all_seeds)
    if "train" in cfg_long:
        tr_long = cfg_long.get("train", {})
        seeds_long: List[int] = [int(s) for s in tr_long.get("seeds", [])]
        ds_long = dict(tr_long.get("dataset", {}))
    else:
        seeds_long: List[int] = [int(s) for s in cfg_long.get("all_seeds", [])]
        ds_long = dict(cfg_long.get("dataset", {}))

    if "train" in cfg_wide:
        tr_wide = cfg_wide.get("train", {})
        seeds_wide: List[int] = [int(s) for s in tr_wide.get("seeds", [])]
        ds_wide = dict(tr_wide.get("dataset", {}))
    else:
        seeds_wide: List[int] = [int(s) for s in cfg_wide.get("all_seeds", [])]
        ds_wide = dict(cfg_wide.get("dataset", {}))

    # Load agents
    agent_long = _load_hetero_agent(longcp_ckpt, device)
    agent_wide = _load_hetero_agent(wide_ckpt, device)

    agent_mixed: AblationGinAgent | None = None
    if mixed_ckpt is not None:
        agent_mixed = _load_hetero_agent(mixed_ckpt, device)

    all_rows: List[Dict[str, float]] = []
    summaries: List[Dict[str, float]] = []

    # long_cp agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(
        agent_long,
        "long_cp",
        "long_cp",
        seeds_long,
        ds_long,
        device,
        override_req_divisor=dataset_req_divisor,
        repeats_per_seed=eval_repeats,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    # Optional: mixed agent on long_cp and wide seeds (if provided)
    if agent_mixed is not None:
        rows, summ = _eval_agent_on_seeds(
            agent_mixed,
            "mixed",
            "long_cp",
            seeds_long,
            ds_long,
            device,
            override_req_divisor=dataset_req_divisor,
            repeats_per_seed=eval_repeats,
        )
        all_rows.extend(rows)
        summaries.append(summ)

        rows, summ = _eval_agent_on_seeds(
            agent_mixed,
            "mixed",
            "wide",
            seeds_wide,
            ds_wide,
            device,
            override_req_divisor=dataset_req_divisor,
            repeats_per_seed=eval_repeats,
        )
        all_rows.extend(rows)
        summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(
        agent_long,
        "long_cp",
        "wide",
        seeds_wide,
        ds_wide,
        device,
        override_req_divisor=dataset_req_divisor,
        repeats_per_seed=eval_repeats,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    # wide agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(
        agent_wide,
        "wide",
        "long_cp",
        seeds_long,
        ds_long,
        device,
        override_req_divisor=dataset_req_divisor,
        repeats_per_seed=eval_repeats,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(
        agent_wide,
        "wide",
        "wide",
        seeds_wide,
        ds_wide,
        device,
        override_req_divisor=dataset_req_divisor,
        repeats_per_seed=eval_repeats,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    # Write outputs
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{case_label}_hetero_eval.csv"
    summary_csv = out_root / f"{case_label}_hetero_eval.summary.csv"
    meta_json = out_root / f"{case_label}_hetero_eval.meta.json"

    import csv as _csv

    fieldnames = [
        "agent_train_domain",
        "eval_domain",
        "seed",
        "makespan",
        "energy_total",
        "energy_active",
        "energy_idle",
        "entropy",
    ]

    with out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary_fields = [
        "agent_train_domain",
        "eval_domain",
        "seeds",
        "mean_makespan",
        "mean_energy_total",
        "mean_energy_active",
        "mean_energy_idle",
        "mean_entropy",
    ]
    with summary_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    meta: Dict[str, Any] = {
        "case": case_label,
        "longcp_ckpt": str(longcp_ckpt),
        "wide_ckpt": str(wide_ckpt),
        "host_specs": str(host_specs),
        "mixed_ckpt": str(mixed_ckpt) if mixed_ckpt is not None else None,
        "longcp_config": str(longcp_config),
        "wide_config": str(wide_config),
        "device": str(device),
        "eval_repeats_per_seed": eval_repeats,
        "dataset_req_divisor": dataset_req_divisor,
    }
    with meta_json.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote per-seed metrics to: {out_csv}")
    print(f"Wrote summary metrics to: {summary_csv}")
    print(f"Wrote metadata to: {meta_json}")


def main(a: Args) -> None:
    device = _pick_device(a.device)

    # Global determinism
    eval_seed = 12345
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

    longcp_config = Path(a.longcp_config)
    wide_config = Path(a.wide_config)
    out_root = Path(a.out_dir)

    # Define all four cases (specialists only, no mixed agents)
    cases = [
        (
            "AL",
            Path("logs/AL_case/long_cp_specialist_traj_aligned/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/AL_case/wide_specialist_traj_aligned/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/AL_case/host_specs_AL.json"),
            None,
        ),
        (
            "NAL",
            Path("logs/NAL_case/long_cp_specialist_traj/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/NAL_case/wide_specialist_traj/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/NAL_case/host_specs.json"),
            None,
        ),
        (
            "HP",
            Path("logs/HP_controlled/long_cp_specialist_homopower_controlled/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/HP_controlled/hetero_wide_homopower_controlled/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/HP_controlled/host_specs_homoPower.json"),
            None,
        ),
        (
            "HS",
            Path("logs/HS_controlled/long_cp_specialist_traj_homospeed_controlled/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/HS_controlled/hetero_wide_homospeed_controlled/ablation/per_variant/hetero/hetero_best_return.pt"),
            Path("logs/HS_controlled/host_specs_homospeed.json"),
            None,
        ),
    ]

    for case_label, longcp_ckpt, wide_ckpt, host_specs, mixed_ckpt in cases:
        if not longcp_ckpt.exists():
            print(f"WARNING: {case_label} long_cp checkpoint not found: {longcp_ckpt}")
            continue
        if not wide_ckpt.exists():
            print(f"WARNING: {case_label} wide checkpoint not found: {wide_ckpt}")
            continue
        if not host_specs.exists():
            print(f"WARNING: {case_label} host specs not found: {host_specs}")
            continue

        if mixed_ckpt is not None and (not mixed_ckpt.exists()):
            print(f"WARNING: {case_label} mixed checkpoint not found: {mixed_ckpt}")

        eval_one_case(
            case_label=case_label,
            longcp_ckpt=longcp_ckpt,
            wide_ckpt=wide_ckpt,
            host_specs=host_specs,
            mixed_ckpt=mixed_ckpt,
            longcp_config=longcp_config,
            wide_config=wide_config,
            device=device,
            dataset_req_divisor=a.dataset_req_divisor,
            eval_repeats=a.eval_repeats_per_seed,
            out_root=out_root,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
