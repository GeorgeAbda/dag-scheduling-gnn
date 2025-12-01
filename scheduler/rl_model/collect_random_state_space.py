import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tyro

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class Args:
    # Paths for configs and selected eval seeds
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"
    wide_selected_eval: str = "runs/datasets/wide/representativeness/selected_eval_seeds.json"
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    longcp_selected_eval: str = "runs/datasets/longcp/representativeness/selected_eval_seeds.json"

    # Random agent settings
    num_random_agents: int = 5

    # Output
    output_dir: str = "runs/state_space_random"


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int) -> DatasetArgs:
    """Build DatasetArgs from the JSON config + a specific seed.

    We mirror the keys used in the training configs.
    """
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
    )


def _collect_for_domain(domain: str, cfg_path: Path, sel_path: Path, args: Args) -> None:
    cfg = json.loads(cfg_path.read_text())
    train_cfg = cfg.get("train", {})
    ds_cfg = train_cfg.get("dataset", {})

    sel = json.loads(sel_path.read_text())
    # File produced by representative_eval.py
    seeds: List[int] = [int(s) for s in sel.get("selected_eval_seeds", [])]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{domain}_random_states.npz"

    all_states: List[np.ndarray] = []
    meta_domain: List[str] = []
    meta_seed: List[int] = []
    meta_agent: List[int] = []
    meta_step: List[int] = []

    print(f"[collect] Domain={domain} | seeds={seeds}")

    for seed_idx, seed in enumerate(seeds):
        ds_args = _dataset_args_from_cfg(ds_cfg, seed=int(seed))

        for agent_id in range(args.num_random_agents):
            # Create fresh env per (seed, agent) so each episode is independent
            base_env = CloudSchedulingGymEnvironment(
                dataset_args=ds_args,
                collect_timelines=False,
                compute_metrics=True,
                profile=False,
                fixed_env_seed=True,
            )
            env = GinAgentWrapper(base_env)

            # Random policy RNG distinct per agent and seed
            rng = np.random.RandomState(seed * 1000 + agent_id)

            obs, info = env.reset(seed=None)
            obs = np.asarray(obs, dtype=np.float32)

            step = 0
            done = False
            while not done:
                # Collect current state (flat GIN observation)
                all_states.append(obs.copy())
                meta_domain.append(domain)
                meta_seed.append(int(seed))
                meta_agent.append(int(agent_id))
                meta_step.append(int(step))

                # Sample a random VALID (task, vm) pair from the underlying EnvObservation
                prev_obs = env.prev_obs  # EnvObservation
                vm_count = len(prev_obs.vm_observations)

                valid_pairs = []
                compat_set = set(prev_obs.compatibilities)
                for t_id, t in enumerate(prev_obs.task_observations):
                    if not t.is_ready or (t.assigned_vm_id is not None):
                        continue
                    for v_id in range(vm_count):
                        if (t_id, v_id) in compat_set:
                            valid_pairs.append((t_id, v_id))

                if not valid_pairs:
                    # Fallback: use wrapper's own remapping via a uniform random flat index
                    flat_action = int(rng.randint(0, env.action_space.n))
                else:
                    t_id, v_id = valid_pairs[rng.randint(0, len(valid_pairs))]
                    flat_action = t_id * vm_count + v_id

                next_obs, reward, terminated, truncated, info = env.step(flat_action)
                obs = np.asarray(next_obs, dtype=np.float32)
                done = bool(terminated or truncated)
                step += 1

            env.close()
            print(f"[collect] Domain={domain} seed={seed} agent={agent_id} steps={step}")

    if not all_states:
        print(f"[collect] No states collected for domain {domain}; skipping write.")
        return

    X = np.stack(all_states, axis=0)
    meta = {
        "domain": np.array(meta_domain, dtype=object),
        "seed": np.array(meta_seed, dtype=np.int64),
        "agent": np.array(meta_agent, dtype=np.int64),
        "step": np.array(meta_step, dtype=np.int64),
    }

    np.savez_compressed(out_file, X=X, **meta)
    print(f"[collect] Saved {X.shape[0]} states (dim={X.shape[1]}) to {out_file}")


def main(args: Args) -> None:
    wide_cfg = Path(args.wide_config)
    wide_sel = Path(args.wide_selected_eval)
    long_cfg = Path(args.longcp_config)
    long_sel = Path(args.longcp_selected_eval)

    _collect_for_domain("wide", wide_cfg, wide_sel, args)
    _collect_for_domain("longcp", long_cfg, long_sel, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
