#!/usr/bin/env python3
"""
Analyze correlation between step reward shaping and (idle + active) energy.

Scope:
- No Lagrangian, no hard constraint override.
- Equal weights for makespan and energy deltas per step (default 1.0 each).
- Runs a fixed trained agent for multiple episodes on a specified dataset config.
- Records per-episode: sum_step_reward, total_energy_active, total_energy_idle, total_energy, makespan.
- Computes Pearson/Spearman correlations and saves a CSV + optional scatter plot.

Usage example:
  python -m scripts.analyze_reward_energy_correlation \
    --checkpoint logs/hetero_both_lag/baseline_model.pt \
    --episodes 50 \
    --out-csv csv/reward_energy_corr.csv \
    --out-fig figures/reward_vs_energy_scatter.png \
    --device cpu \
    --dag-method gnp \
    --gnp-min-n 12 --gnp-max-n 24 \
    --workflow-count 10 --host-count 4 --vm-count 10

Notes:
- This script instantiates the CloudSchedulingGymEnvironment and wraps it with GinAgentWrapper
  to obtain the step reward shaping used by the GIN agent pipeline. It disables Lagrangian
  and constraint logic and sets equal weights for makespan and energy deltas.
- The agent architecture is the AblationGinAgent with the 'hetero' variant by default.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant  # reuse model class


def build_env(args) -> GinAgentWrapper:
    ds_args = DatasetArgs(
        dag_method=args.dag_method,
        gnp_min_n=args.gnp_min_n,
        gnp_max_n=args.gnp_max_n,
        workflow_count=args.workflow_count,
        host_count=args.host_count,
        vm_count=args.vm_count,
        seed=None,
    )
    base_env = CloudSchedulingGymEnvironment(dataset=None, dataset_args=ds_args,
                                             collect_timelines=False, compute_metrics=True,
                                             profile=False)
    # Configure wrapper for NO Lagrangian, NO constraint, equal weights
    os.environ["GIN_CONSTRAINED"] = "0"
    os.environ["GIN_LAGRANGIAN"] = "0"
    os.environ["GIN_ENERGY_WEIGHT"] = str(args.energy_weight)
    os.environ["GIN_MAKESPAN_WEIGHT"] = str(args.makespan_weight)
    wrapped = GinAgentWrapper(base_env, constrained_mode=False)
    return wrapped


def load_agent(checkpoint: Path, device: torch.device) -> AblationGinAgent:
    # Default to hetero variant used in training; can be extended via CLI if needed
    variant = AblationVariant(name="hetero", graph_type="hetero", gin_num_layers=2, hetero_base="sage")
    agent = AblationGinAgent(device=device, variant=variant)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        agent.load_state_dict(state["state_dict"], strict=False)
    else:
        agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def run_episodes(env: GinAgentWrapper, agent: AblationGinAgent, episodes: int, device: torch.device) -> pd.DataFrame:
    rows: List[dict] = []
    for ep in range(episodes):
        obs, info = env.reset()
        # obs is a flattened np.ndarray (GinAgentWrapper.map_observation)
        sum_reward = 0.0
        done = False
        term = False
        trunc = False
        steps = 0
        while not (term or trunc):
            x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, ent, val = agent.get_action_and_value(x)
            a = int(action.item())
            obs, r, term, trunc, info = env.step(a)
            sum_reward += float(r)
            steps += 1
            if steps > 100000:  # safety
                break
        total_active = float(info.get("total_energy_active", np.nan))
        total_idle = float(info.get("total_energy_idle", np.nan))
        total_energy = total_active + total_idle if (np.isfinite(total_active) and np.isfinite(total_idle)) else np.nan
        makespan = float(info.get("makespan", np.nan))
        rows.append({
            "episode": ep,
            "sum_step_reward": sum_reward,
            "total_energy_active": total_active,
            "total_energy_idle": total_idle,
            "total_energy": total_energy,
            "makespan": makespan,
        })
    df = pd.DataFrame(rows)
    return df


def compute_and_save(df: pd.DataFrame, out_csv: Path, out_fig: Path | None) -> None:
    df.to_csv(out_csv, index=False)
    # Correlations
    sub = df.dropna()
    pearson = sub[["sum_step_reward", "total_energy"]].corr(method="pearson").iloc[0, 1]
    spearman = sub[["sum_step_reward", "total_energy"]].corr(method="spearman").iloc[0, 1]
    print(f"[corr] Pearson(sum_step_reward, total_energy) = {pearson:.6g}")
    print(f"[corr] Spearman(sum_step_reward, total_energy) = {spearman:.6g}")
    # Optional scatter
    if out_fig is not None:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,4))
            plt.scatter(sub["sum_step_reward"], sub["total_energy"], alpha=0.6, s=16)
            plt.xlabel("Sum of step rewards (no Lagrangian, equal weights)")
            plt.ylabel("Total energy (active + idle)")
            plt.title("Reward vs Total Energy")
            plt.tight_layout()
            out_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_fig, dpi=180)
            plt.close()
            print(f"[fig] Saved scatter to {out_fig}")
        except Exception as e:
            print(f"[fig] Failed to save scatter: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze correlation between step reward and total energy")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-csv", type=Path, default=Path("csv/reward_energy_corr.csv"))
    p.add_argument("--out-fig", type=Path, default=Path("figures/reward_vs_energy_scatter.png"))
    # Dataset config
    p.add_argument("--dag-method", type=str, choices=["linear", "gnp"], default="gnp")
    p.add_argument("--gnp-min-n", type=int, default=12)
    p.add_argument("--gnp-max-n", type=int, default=24)
    p.add_argument("--workflow-count", type=int, default=10)
    p.add_argument("--host-count", type=int, default=4)
    p.add_argument("--vm-count", type=int, default=10)
    # Reward weights (equal by default)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--makespan-weight", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    env = build_env(args)
    agent = load_agent(args.checkpoint, device)
    df = run_episodes(env, agent, args.episodes, device)
    out_csv: Path = args.out_csv
    out_fig: Path | None = args.out_fig
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    compute_and_save(df, out_csv, out_fig)
    print(f"[done] Wrote CSV to {out_csv}")


if __name__ == "__main__":
    main()
