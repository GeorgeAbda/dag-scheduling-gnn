#!/usr/bin/env python3
"""
Analyze correlation between step reward shaping and (active energy + makespan).

Scope:
- No Lagrangian, no hard constraint override.
- Equal weights for makespan and energy deltas per step (default 1.0 each), configurable.
- Runs a fixed trained agent for multiple episodes on a specified dataset config.
- Records per-episode: sum_step_reward, active_energy, makespan, combined_metric.
- Computes Pearson/Spearman correlations and saves a CSV + optional scatter plot.

Usage example:
  python -m scripts.analyze_reward_vs_active_plus_makespan \
    --checkpoint logs/hetero_both_lag/baseline_model.pt \
    --episodes 50 \
    --out-csv csv/reward_vs_active_plus_makespan.csv \
    --out-fig figures/reward_vs_active_plus_makespan.png \
    --device cpu \
    --dag-method gnp \
    --gnp-min-n 12 --gnp-max-n 24 \
    --workflow-count 10 --host-count 4 --vm-count 10 \
    --combine-method none

Notes:
- (active energy + makespan) mixes units; we provide --combine-method to control combination:
    none   : raw sum (active_energy + makespan)
    zscore : z-score each across episodes, then sum
    minmax : min-max scale each to [0,1] across episodes, then sum
- This script disables Lagrangian and constraint logic and sets per-step weights via env.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant


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
    # Configure wrapper for NO Lagrangian, NO constraint, specified weights
    os.environ["GIN_CONSTRAINED"] = "0"
    os.environ["GIN_LAGRANGIAN"] = "0"
    os.environ["GIN_ENERGY_WEIGHT"] = str(args.energy_weight)
    os.environ["GIN_MAKESPAN_WEIGHT"] = str(args.makespan_weight)
    wrapped = GinAgentWrapper(base_env, constrained_mode=False)
    return wrapped


def load_agent(checkpoint: Path, device: torch.device) -> AblationGinAgent:
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
        sum_reward = 0.0
        term = trunc = False
        steps = 0
        while not (term or trunc):
            x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, ent, val = agent.get_action_and_value(x)
            a = int(action.item())
            obs, r, term, trunc, info = env.step(a)
            sum_reward += float(r)
            steps += 1
            if steps > 100000:
                break
        active = float(info.get("total_energy_active", np.nan))
        makespan = float(info.get("makespan", np.nan))
        rows.append({
            "episode": ep,
            "sum_step_reward": sum_reward,
            "active_energy": active,
            "makespan": makespan,
        })
    return pd.DataFrame(rows)


def _combine_metric(df: pd.DataFrame, method: str) -> pd.Series:
    a = df["active_energy"].to_numpy()
    m = df["makespan"].to_numpy()
    if method == "zscore":
        a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
        m = (m - np.nanmean(m)) / (np.nanstd(m) + 1e-9)
    elif method == "minmax":
        a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-9)
        m = (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m) + 1e-9)
    # method == "none": raw sum
    return pd.Series(a + m)


def compute_and_save(df: pd.DataFrame, out_csv: Path, out_fig: Path | None, method: str) -> None:
    df = df.copy()
    df["active_plus_makespan"] = _combine_metric(df, method)
    df.to_csv(out_csv, index=False)
    sub = df.dropna()
    pearson = sub[["sum_step_reward", "active_plus_makespan"]].corr(method="pearson").iloc[0, 1]
    spearman = sub[["sum_step_reward", "active_plus_makespan"]].corr(method="spearman").iloc[0, 1]
    print(f"[corr] Pearson(sum_step_reward, active_plus_makespan[{method}]) = {pearson:.6g}")
    print(f"[corr] Spearman(sum_step_reward, active_plus_makespan[{method}]) = {spearman:.6g}")
    if out_fig is not None:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,4))
            plt.scatter(sub["sum_step_reward"], sub["active_plus_makespan"], alpha=0.6, s=16)
            plt.xlabel("Sum of step rewards (no Lagrangian, weights)")
            plt.ylabel(f"Active energy + makespan ({method})")
            plt.title("Reward vs Active+Makespan")
            plt.tight_layout()
            out_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_fig, dpi=180)
            plt.close()
            print(f"[fig] Saved scatter to {out_fig}")
        except Exception as e:
            print(f"[fig] Failed to save scatter: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze correlation between step reward and active_energy+makespan")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-csv", type=Path, default=Path("csv/reward_vs_active_plus_makespan.csv"))
    p.add_argument("--out-fig", type=Path, default=Path("figures/reward_vs_active_plus_makespan.png"))
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
    # Combination method
    p.add_argument("--combine-method", type=str, choices=["none", "zscore", "minmax"], default="none")
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
    compute_and_save(df, out_csv, out_fig, args.combine_method)
    print(f"[done] Wrote CSV to {out_csv}")


if __name__ == "__main__":
    main()
