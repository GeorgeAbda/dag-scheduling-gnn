"""
Quick specialist training script (wide or longcp) using scheduler package.
- Minimal CPU run for sanity check
- Saves final checkpoint and basic stats

Usage:
  python -u train_specialist_quick.py <style> <host_specs_file> <total_timesteps>

Examples:
  python -u train_specialist_quick.py wide data/host_specs.json 2000
  python -u train_specialist_quick.py longcp data/host_specs.json 2000
"""
from __future__ import annotations
import os
import sys
from typing import Tuple
from pathlib import Path
import random
import json

import numpy as np
import torch
from tqdm import tqdm

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def make_dataset_args(style: str) -> DatasetArgs:
    style = style.lower()
    if style in {"longcp", "long_cp", "long"}:
        # Dense graphs
        edge_prob = random.uniform(0.70, 0.85)
        min_tasks = 12
        max_tasks = 24
        min_length = 50_000
        max_length = 150_000
        style_key = "long_cp"
    elif style in {"wide", "wide_cp", "sparse"}:
        # Sparse graphs
        edge_prob = random.uniform(0.03, 0.06)
        min_tasks = 24
        max_tasks = 30
        min_length = 100
        max_length = 50_000
        style_key = "wide"
    else:
        raise ValueError("style must be one of: wide, longcp")

    return DatasetArgs(
        host_count=10,
        vm_count=10,
        workflow_count=1,
        style=style_key,
        gnp_p=edge_prob,
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        min_task_length=min_length,
        max_task_length=max_length,
    )


def create_env(style: str, host_specs_file: str):
    os.environ["HOST_SPECS_PATH"] = host_specs_file
    ds = make_dataset_args(style)
    env = GinAgentWrapper(
        CloudSchedulingGymEnvironment(dataset_args=ds, collect_timelines=False, compute_metrics=True)
    )
    return env


def train_specialist(
    style: str,
    host_specs_file: str,
    output_dir: str,
    total_timesteps: int = 2000,
    num_envs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    update_epochs: int = 4,
):
    device = torch.device("cpu")

    variant = AblationVariant(
        name="hetero",
        graph_type="hetero",
        gin_num_layers=2,
        use_batchnorm=True,
        use_task_dependencies=True,
        use_actor_global_embedding=True,
        mlp_only=False,
        gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)

    # warmup forward
    dummy_env = create_env(style, host_specs_file)
    obs, _ = dummy_env.reset(seed=42)
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_t)
    dummy_env.close()

    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    envs = [create_env(style, host_specs_file) for _ in range(num_envs)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    num_updates = max(1, total_timesteps // max(1, batch_size))

    # Buffers
    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    obs_list = [env.reset(seed=42 + i)[0] for i, env in enumerate(envs)]

    episode_returns = []
    episode_lengths = []

    pbar = tqdm(range(num_updates), desc=f"Training({style})", ncols=100)
    for update in pbar:
        # Collect rollout
        steps_per_update = max(1, batch_size // max(1, num_envs))
        for step in range(steps_per_update):
            global_step += num_envs

            obs_tensors = [
                torch.from_numpy(np.asarray(o, dtype=np.float32).reshape(1, -1))
                for o in obs_list
            ]
            with torch.no_grad():
                actions = []
                logprobs = []
                values = []
                for ot in obs_tensors:
                    a, lp, _, v = agent.get_action_and_value(ot)
                    actions.append(a)
                    logprobs.append(lp)
                    values.append(v)

            next_obs_list = []
            for i, (env, obs, action) in enumerate(zip(envs, obs_list, actions)):
                obs_buffer.append(obs)
                actions_buffer.append(action.cpu())
                logprobs_buffer.append(logprobs[i].cpu())
                values_buffer.append(values[i].cpu())

                next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated

                rewards_buffer.append(reward)
                dones_buffer.append(done)

                if done:
                    if "makespan" in info:
                        episode_returns.append(-info["makespan"])
                        episode_lengths.append(step)
                    next_obs, _ = env.reset()
                next_obs_list.append(next_obs)
            obs_list = next_obs_list

        # Bootstrap value
        with torch.no_grad():
            next_values = [
                agent.get_action_and_value(
                    torch.from_numpy(np.asarray(o, dtype=np.float32).reshape(1, -1))
                )[3]
                for o in obs_list
            ]

        # GAE
        advantages = []
        returns = []
        gae = 0.0
        for t in reversed(range(len(rewards_buffer))):
            if t == len(rewards_buffer) - 1:
                next_value = next_values[t % max(1, num_envs)]
                next_done = 0
            else:
                next_value = values_buffer[t + 1]
                next_done = dones_buffer[t]
            delta = rewards_buffer[t] + gamma * next_value * (1 - next_done) - values_buffer[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_buffer[t])

        obs_batch = torch.from_numpy(np.array([np.asarray(o, dtype=np.float32) for o in obs_buffer]))
        actions_batch = torch.stack(actions_buffer)
        logprobs_batch = torch.stack(logprobs_buffer)
        advantages_batch = torch.tensor(advantages, dtype=torch.float32)
        returns_batch = torch.tensor(returns, dtype=torch.float32)
        values_batch = torch.stack(values_buffer)

        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        for _ in range(update_epochs):
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                obs_batch, actions_batch.squeeze()
            )
            logratio = newlogprob - logprobs_batch
            ratio = logratio.exp()

            pg_loss1 = -advantages_batch * ratio
            pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue - returns_batch) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        obs_buffer.clear(); actions_buffer.clear(); logprobs_buffer.clear(); rewards_buffer.clear(); dones_buffer.clear(); values_buffer.clear()

        if episode_returns:
            avg_return = float(np.mean(episode_returns[-100:]))
            pbar.set_postfix({"step": f"{global_step:,}", "return": f"{avg_return:.2f}", "episodes": len(episode_returns)})

    # Save
    final_path = Path(output_dir) / f"{style}_specialist_final.pt"
    torch.save(agent.state_dict(), final_path)

    stats = {
        "style": style,
        "total_timesteps": global_step,
        "num_episodes": len(episode_returns),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }
    with open(Path(output_dir) / f"{style}_training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n==============================================")
    print(f"TRAINING COMPLETE ({style})")
    print("==============================================")
    print(f"Saved: {final_path}")
    return final_path, stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -u train_specialist_quick.py <style: wide|longcp> [host_specs_file] [total_timesteps]")
        sys.exit(1)
    style = sys.argv[1]
    host_specs = sys.argv[2] if len(sys.argv) > 2 else "data/host_specs.json"
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    out_dir = f"logs/{style}_specialist_quick"
    train_specialist(style, host_specs, out_dir, total_timesteps=total, num_envs=1)
