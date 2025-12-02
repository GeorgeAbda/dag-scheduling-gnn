import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import tyro
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Ensure project root on path (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.config.settings import MIN_TESTING_DS_SEED
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant


@dataclass
class Args:
    # Models
    model_a_dir: str | None = None
    """Subdirectory under logs/ that contains a model for Agent A (e.g., 1720000000_expA)"""
    model_b_dir: str | None = None
    """Subdirectory under logs/ that contains a model for Agent B (e.g., 1720001000_expB)"""
    model_a_filename: str = "model.pt"
    """Filename when using logs/<dir>/<filename> for Agent A"""
    model_b_filename: str = "model.pt"
    """Filename when using logs/<dir>/<filename> for Agent B"""
    model_a_path: str | None = None
    """Explicit full path to Agent A checkpoint (.pt). Overrides dir/filename if provided."""
    model_b_path: str | None = None
    """Explicit full path to Agent B checkpoint (.pt). Overrides dir/filename if provided."""

    agent_type: str = "gin"
    """Agent type: 'gin' for standard GinAgent, 'hetero' for AblationGinAgent hetero variant."""

    # Episode and sampling
    seed: int = 1
    """Base seed; test environment is seeded as MIN_TESTING_DS_SEED + 0"""
    max_decision_steps: int | None = None
    """If set, limit to the first K decision steps collected."""
    driver: str = "A"
    """Which agent drives environment actions: 'A', 'B', or 'random'"""

    # Plot
    out_path: str | None = None
    """Output image path. If None, saved to logs/<model_a_dir>_vs_<model_b_dir>/score_correlation.pdf"""
    dpi: int = 300
    """Figure DPI."""

    # Dataset
    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=1,
            gnp_min_n=20,
            gnp_max_n=20,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="gnp",
        )
    )


def load_agent(
    model_dir: str | None,
    model_filename: str,
    model_path_explicit: str | None,
    device: torch.device,
    agent_type: str,
):
    agent_type = (agent_type or "gin").lower()
    if agent_type == "hetero":
        variant = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
        agent: GinAgent | AblationGinAgent = AblationGinAgent(device, variant, embedding_dim=16)
        strict = False
    else:
        agent = GinAgent(device)
        strict = True
    if model_path_explicit:
        model_path = Path(model_path_explicit)
    else:
        if model_dir is None:
            raise ValueError("Either model_path_explicit or model_dir must be provided")
        model_path = Path(PROJECT_ROOT) / "logs" / model_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    # weights_only=True for safety with new torch (both plain and ablation agents use state_dicts)
    state_dict = torch.load(str(model_path), weights_only=True)
    agent.load_state_dict(state_dict, strict=strict)
    agent.eval()
    return agent


def build_env(dataset: DatasetArgs) -> GinAgentWrapper:
    env = CloudSchedulingGymEnvironment(dataset_args=dataset, collect_timelines=False)
    return GinAgentWrapper(env)


def valid_action_mask(decoded_obs) -> torch.Tensor:
    # Recreate the flat mask used in GinAgent.get_action_and_value()
    num_vms = decoded_obs.vm_completion_time.shape[0]
    ready_mask_tasks: torch.Tensor = decoded_obs.task_state_ready.bool()
    not_scheduled_mask_tasks: torch.Tensor = (decoded_obs.task_state_scheduled == 0)
    valid_task_mask = ready_mask_tasks & not_scheduled_mask_tasks
    flat_mask = valid_task_mask.repeat_interleave(num_vms)
    return flat_mask


def get_flat_scores(agent: GinAgent, obs_np: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
    # Map to tensor, unmap to structured obs for this agent
    x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(agent.device)
    decoded = agent.mapper.unmap(x[0])
    scores: torch.Tensor = agent.actor(decoded)  # [T, V]
    flat = scores.flatten()
    mask = valid_action_mask(decoded)
    # Filter by valid actions only
    flat_valid = flat[mask]
    return flat_valid.detach().cpu().numpy(), mask.detach().cpu()


def collect_scores(args: Args, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    agent_a = load_agent(args.model_a_dir, args.model_a_filename, args.model_a_path, device, args.agent_type)
    agent_b = load_agent(args.model_b_dir, args.model_b_filename, args.model_b_path, device, args.agent_type)

    env = build_env(args.dataset)
    obs_np, _ = env.reset(seed=MIN_TESTING_DS_SEED)

    scores_a_all: List[np.ndarray] = []
    scores_b_all: List[np.ndarray] = []

    decision_steps = 0
    while True:
        # Both agents evaluate the SAME observation
        a_scores, a_mask = get_flat_scores(agent_a, obs_np)
        b_scores, b_mask = get_flat_scores(agent_b, obs_np)

        # Masks should be identical given same observation mapping; use intersection for safety
        if a_mask.shape != b_mask.shape:
            # Align by length (should not happen)
            mlen = min(a_mask.numel(), b_mask.numel())
            a_valid = a_mask[:mlen] & b_mask[:mlen]
            a_scores = a_scores[:int(a_valid.sum().item())]
            b_scores = b_scores[:int(a_valid.sum().item())]
        else:
            # Already filtered to valid actions, so just ensure shapes match
            mlen = min(a_scores.shape[0], b_scores.shape[0])
            a_scores = a_scores[:mlen]
            b_scores = b_scores[:mlen]

        if mlen > 0:
            scores_a_all.append(a_scores)
            scores_b_all.append(b_scores)
            decision_steps += 1

        if args.max_decision_steps is not None and decision_steps >= args.max_decision_steps:
            break

        # Choose action to progress the environment
        if args.driver.upper() == "A":
            with torch.no_grad():
                a, _, _, _ = agent_a.get_action_and_value(torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device))
                action = int(a.item())
        elif args.driver.upper() == "B":
            with torch.no_grad():
                b, _, _, _ = agent_b.get_action_and_value(torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device))
                action = int(b.item())
        else:
            # Random valid action via Agent A's probabilities
            with torch.no_grad():
                a, _, _, _ = agent_a.get_action_and_value(torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device))
                action = int(a.item())

        obs_np, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()

    if len(scores_a_all) == 0:
        return np.array([]), np.array([])

    A = np.concatenate(scores_a_all, axis=0)
    B = np.concatenate(scores_b_all, axis=0)
    return A, B


def plot_correlation(A: np.ndarray, B: np.ndarray, out_path: str, dpi: int = 300,
                     label_a: str = "Agent A", label_b: str = "Agent B") -> None:
    # Style closer to the example: plain background, readable default fonts
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    sns.set_theme(context="notebook", style="white", font_scale=1.0)

    # Compute Pearson r
    if len(A) == 0 or len(B) == 0:
        raise ValueError("No scores collected to plot.")
    r, p = pearsonr(A, B)

    # Colors to match the provided style
    reds_cmap = "Reds"
    uni_color = "#fba47e"

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8.0, 6.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 4.0], width_ratios=[4.0, 1.2], hspace=0.05, wspace=0.05)

    # Top univariate KDE (Agent A)
    ax_top = fig.add_subplot(gs[0, 0])
    sns.kdeplot(A, color=uni_color, fill=True, alpha=0.5, warn_singular=False, ax=ax_top)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Density")
    ax_top.tick_params(axis='x', labelbottom=False)

    # Joint KDE
    ax_joint = fig.add_subplot(gs[1, 0])
    # Filled 2D KDE and overlay scatter points
    sns.kdeplot(x=A, y=B, cmap=reds_cmap, fill=True, thresh=0.05, warn_singular=False, ax=ax_joint, zorder=1)
    # Add a tiny jitter so overlapping points become visible (esp. for highly correlated models)
    rng = np.random.default_rng(12345)
    jitter_ax = max(1e-8, 0.005 * (np.std(A) if np.std(A) > 0 else 1.0))
    jitter_by = max(1e-8, 0.005 * (np.std(B) if np.std(B) > 0 else 1.0))
    A_sc = A + rng.normal(0.0, jitter_ax, size=A.shape)
    B_sc = B + rng.normal(0.0, jitter_by, size=B.shape)
    ax_joint.scatter(A_sc, B_sc, c="darkred", alpha=0.6, s=14, edgecolors="black", linewidths=0.2, zorder=3)
    ax_joint.set_xlabel("Fitness from agent A")
    ax_joint.set_ylabel("Fitness from agent B ")
    # Annotation at top-left inside the joint
    ax_joint.text(0.05, 0.95, f"pearsonr = {r:.2f}; p = {p:.6f}", transform=ax_joint.transAxes,
                  ha="left", va="top", fontsize=12)

    # Right univariate KDE (Agent B) horizontal
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    sns.kdeplot(y=B, color=uni_color, fill=True, alpha=0.5, warn_singular=False, ax=ax_right)
    ax_right.set_xlabel("Density")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis='y', labelleft=False)

    # Align limits
    ax_top.set_xlim(ax_joint.get_xlim())
    ax_right.set_ylim(ax_joint.get_ylim())

    # Clean top-right corner (empty)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis('off')

    sns.despine(fig=fig)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A, B = collect_scores(args, device)
    # Debug info: how many points will be plotted
    try:
        n_points = int(min(len(A), len(B)))
        print(f"Collected score pairs: {n_points} (A len={len(A)}, B len={len(B)})")
    except Exception:
        pass

    # Prepare output path
    if args.out_path is None:
        a_tag = args.model_a_dir if args.model_a_dir else Path(args.model_a_path).stem if args.model_a_path else "A"
        b_tag = args.model_b_dir if args.model_b_dir else Path(args.model_b_path).stem if args.model_b_path else "B"
        out_dir = Path(PROJECT_ROOT) / "logs" / f"{a_tag}_vs_{b_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "score_correlation.pdf")
    else:
        out_path = args.out_path
        Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    a_label = args.model_a_dir or (Path(args.model_a_path).name if args.model_a_path else "Agent A")
    b_label = args.model_b_dir or (Path(args.model_b_path).name if args.model_b_path else "Agent B")
    plot_correlation(A, B, out_path, dpi=args.dpi, label_a=a_label, label_b=b_label)
    print(f"Saved correlation figure to: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
