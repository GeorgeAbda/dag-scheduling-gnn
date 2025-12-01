#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env as make_test_env,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def _pick_ckpt_from_dir(dir_path: Path, prefix: str = "hetero_mean_pf_it", fallback_prefix: str = "hetero_worst_pf_it") -> Optional[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    cands = sorted(dir_path.glob(f"{prefix}*.pt"))
    if not cands:
        cands = sorted(dir_path.glob(f"{fallback_prefix}*.pt"))
    if not cands:
        cands = sorted(dir_path.glob("*.pt"))
    return cands[-1] if cands else None


def build_args_style(style: str, device: str, hosts: int, vms: int, workflows: int,
                     min_tasks: int, max_tasks: int, episodes: int, seed: int) -> AblArgs:
    a = AblArgs()
    a.device = device
    a.test_iterations = max(1, int(episodes))
    ds = DatasetArgs(
        host_count=hosts,
        vm_count=vms,
        workflow_count=workflows,
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
    )
    setattr(ds, "style", str(style))
    a.dataset = ds
    return a


def load_hetero_agent(ckpt: Path, device: str) -> AblationGinAgent:
    var = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    dev = torch.device(device)
    agent = AblationGinAgent(dev, var, embedding_dim=16)
    state = torch.load(str(ckpt), map_location=dev)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def collect_invalid_ready_stats(agent: AblationGinAgent, args: AblArgs, seed_base: int,
                                topks: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, float]:
    topk_counts = {f"top{k}_invalid": 0 for k in topks}
    total_states = 0
    sum_valid = 0.0
    sum_invalid = 0.0
    cnt_valid = 0
    cnt_invalid = 0

    for s in range(args.test_iterations):
        env = make_test_env(args)
        try:
            obs_np, _ = env.reset(seed=seed_base + s)
            while True:
                obs_t = torch.tensor(np.asarray(obs_np, dtype=np.float32))
                obs = agent.mapper.unmap(obs_t)
                with torch.no_grad():
                    node_h, edge_h, g_h = agent.actor.network(obs)
                    E = int(obs.compatibilities.shape[1])
                    edge_h = edge_h[:E]
                    scorer_in = edge_h
                    if getattr(agent.actor, 'use_actor_global_embedding', False):
                        rep_g = g_h.expand(edge_h.shape[0], agent.actor.embedding_dim)
                        scorer_in = torch.cat([scorer_in, rep_g], dim=1)
                    logits = agent.actor.edge_scorer(scorer_in).flatten()
                t_idx = obs.compatibilities[0][:E].to(torch.long)
                ready = (obs.task_state_ready[t_idx] == 1)
                not_sched = (obs.task_state_scheduled[t_idx] == 0)
                valid = (ready & not_sched)
                invalid = ~valid

                # Aggregate mean logits by validity
                if torch.any(valid):
                    sum_valid += float(logits[valid].mean().item())
                    cnt_valid += 1
                if torch.any(invalid):
                    sum_invalid += float(logits[invalid].mean().item())
                    cnt_invalid += 1

                # Top-k invalid rates ignoring readiness mask (over compat edges only)
                for k in topks:
                    k_eff = min(k, int(logits.numel()))
                    vals, idx = torch.topk(logits, k_eff)
                    invalid_in_topk = int(invalid[idx].sum().item())
                    if k_eff > 0 and invalid_in_topk > 0:
                        topk_counts[f"top{k}_invalid"] += 1
                total_states += 1

                obs_np, _, terminated, truncated, info = env.step(int(0))  # dummy; wrapper remaps
                if terminated or truncated:
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass

    out: Dict[str, float] = {}
    for k in topks:
        out[f"top{k}_invalid_rate"] = (topk_counts[f"top{k}_invalid"] / max(1, total_states))
    out["mean_logit_valid"] = (sum_valid / max(1, cnt_valid))
    out["mean_logit_invalid"] = (sum_invalid / max(1, cnt_invalid))
    out["states"] = float(total_states)
    return out


def main():
    p = argparse.ArgumentParser(description="Cross-style invalid action score diagnostics (pre-mask)")
    p.add_argument("--longcp-ckpt", type=str, default="",
                   help="Path to hetero checkpoint trained on long_cp style; if empty, auto-pick from logs/hetero_longcp_selected/ablation/per_variant/hetero")
    p.add_argument("--wide-ckpt", type=str, default="",
                   help="Path to hetero checkpoint trained on wide style; if empty, auto-pick from logs/hetero_wide/ablation/per_variant/hetero")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hosts", type=int, default=4)
    p.add_argument("--vms", type=int, default=10)
    p.add_argument("--workflows", type=int, default=10)
    p.add_argument("--min-tasks", type=int, default=12)
    p.add_argument("--max-tasks", type=int, default=24)
    p.add_argument("--seed-base", type=int, default=1_000_000_001)
    p.add_argument("--out-csv", type=Path, default=Path("logs/invalid_score_eval.csv"))

    args = p.parse_args()

    if not args.longcp_ckpt:
        dir_l = Path("logs/hetero_longcp_selected/ablation/per_variant/hetero")
        ckpt_l = _pick_ckpt_from_dir(dir_l)
    else:
        ckpt_l = Path(args.longcp_ckpt)
    if not args.wide_ckpt:
        dir_w = Path("logs/hetero_wide/ablation/per_variant/hetero")
        ckpt_w = _pick_ckpt_from_dir(dir_w)
    else:
        ckpt_w = Path(args.wide_ckpt)

    if ckpt_l is None or not ckpt_l.exists():
        raise SystemExit(f"Could not resolve long_cp checkpoint. Provided: {args.longcp_ckpt}")
    if ckpt_w is None or not ckpt_w.exists():
        raise SystemExit(f"Could not resolve wide checkpoint. Provided: {args.wide_ckpt}")

    agent_long = load_hetero_agent(ckpt_l, args.device)
    agent_wide = load_hetero_agent(ckpt_w, args.device)

    eval_args_wide = build_args_style("wide", args.device, args.hosts, args.vms, args.workflows,
                                      args.min_tasks, args.max_tasks, args.episodes, args.seed_base)
    eval_args_long = build_args_style("long_cp", args.device, args.hosts, args.vms, args.workflows,
                                      args.min_tasks, args.max_tasks, args.episodes, args.seed_base)

    stats_long_on_wide = collect_invalid_ready_stats(agent_long, eval_args_wide, args.seed_base)
    stats_wide_on_long = collect_invalid_ready_stats(agent_wide, eval_args_long, args.seed_base)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with args.out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "agent_train_style","eval_style","states",
            "top1_invalid_rate","top3_invalid_rate","top5_invalid_rate",
            "mean_logit_valid","mean_logit_invalid",
        ])
        w.writeheader()
        w.writerow({
            "agent_train_style": "long_cp",
            "eval_style": "wide",
            **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in stats_long_on_wide.items()},
        })
        w.writerow({
            "agent_train_style": "wide",
            "eval_style": "long_cp",
            **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in stats_wide_on_long.items()},
        })

    print(f"Wrote results to: {args.out_csv}")


if __name__ == "__main__":
    main()
