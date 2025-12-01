import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt

# Ensure project root on path (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scheduler.config.settings import MIN_TESTING_DS_SEED, MAX_OBS_SIZE
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class Args:
    # Model
    model_path: str

    # Sampling
    max_decision_steps: int = 100
    seed: int = 1

    # Plot/output
    out_dir: str = str(Path(PROJECT_ROOT) / "logs" / "shap_importance")
    dpi: int = 300

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


def load_agent(model_path: str, device: torch.device) -> GinAgent:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent = GinAgent(device)
    state_dict = torch.load(str(model_path), weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def build_env(dataset: DatasetArgs) -> GinAgentWrapper:
    env = CloudSchedulingGymEnvironment(dataset_args=dataset, collect_timelines=False)
    return GinAgentWrapper(env)


def collect_observations(env: GinAgentWrapper, agent: GinAgent, max_steps: int) -> np.ndarray:
    obs_np, _ = env.reset(seed=MIN_TESTING_DS_SEED)
    collected = []
    steps = 0
    while True:
        collected.append(obs_np.copy())
        steps += 1
        if steps >= max_steps:
            break
        # Use the agent to select a valid action for this observation
        with torch.no_grad():
            x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(agent.device)
            action_idx, _, _, _ = agent.get_action_and_value(x)
            action_int = int(action_idx.squeeze(0).item())
        obs_np, _, terminated, truncated, _ = env.step(action_int)
        if terminated or truncated:
            break
    return np.stack(collected, axis=0)


def make_scalar_model(agent: GinAgent, reference_obs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function f(X) -> y where X is [n, MAX_OBS_SIZE] and y is [n],
    computing the maximum action probability over valid actions.
    This scalar targets "how confident is the policy in its top action" per state.
    """
    device = agent.device
    def calc_payload_len(hdr: np.ndarray) -> int:
        T = int(hdr[0])
        V = int(hdr[1])
        D = int(hdr[2])
        C = int(hdr[3])
        return 4 + 6*T + 12*V + 2*D + 2*C

    def valid_action_mask(decoded_obs) -> torch.Tensor:
        num_vms = decoded_obs.vm_completion_time.shape[0]
        ready_mask_tasks: torch.Tensor = decoded_obs.task_state_ready.bool()
        not_scheduled_mask_tasks: torch.Tensor = (decoded_obs.task_state_scheduled == 0)
        valid_task_mask = ready_mask_tasks & not_scheduled_mask_tasks
        flat_mask = valid_task_mask.repeat_interleave(num_vms)
        return flat_mask

    def f(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            outs = []
            for i in range(X_t.shape[0]):
                # enforce valid structured encoding for unmap using the sample's own header
                try:
                    xi = X_t[i].clone()
                    header_i = xi[:4].cpu().numpy().astype(np.float64)
                    used_len = calc_payload_len(header_i)
                    if used_len < xi.shape[0]:
                        xi[used_len:] = 0.0
                    decoded = agent.mapper.unmap(xi)
                    scores: torch.Tensor = agent.actor(decoded).flatten()
                    mask = valid_action_mask(decoded)
                    valid_scores = scores[mask]
                    if valid_scores.numel() == 0:
                        outs.append(0.0)
                    else:
                        probs = torch.softmax(valid_scores, dim=0)
                        outs.append(float(probs.max().item()))
                except Exception:
                    outs.append(0.0)
            return np.array(outs, dtype=np.float32)

    return f


def plot_importance(feature_values: np.ndarray, feature_importance: np.ndarray, out_path: str, dpi: int = 300):
    # Simple bar plot of mean |SHAP| per feature
    mean_abs = np.mean(np.abs(feature_importance), axis=0)
    idx = np.argsort(mean_abs)[::-1]
    topk = min(30, len(idx))
    idx = idx[:topk]

    plt.figure(figsize=(8, 6))
    plt.barh(range(topk), mean_abs[idx][::-1], color="#4C72B0", alpha=0.8)
    plt.yticks(range(topk), [f"f{j}" for j in idx[::-1]])
    plt.xlabel("Mean |SHAP| (top action confidence)")
    plt.ylabel("Feature index")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def compute_grouped_importance(feature_importance: np.ndarray, header: np.ndarray):
    """Aggregate mean |importance| over semantic feature families using the header (T,V,D,C).
    Returns (labels, values) where labels are human-readable names and values are mean |importance|.
    """
    T = int(header[0]); V = int(header[1]); D = int(header[2]); C = int(header[3])
    mean_abs = np.mean(np.abs(feature_importance), axis=0)

    labels: List[str] = []
    values: List[float] = []

    idx = 4
    # Task blocks (6 × T)
    groups = [
        ("Task: scheduled (0/1) [no]", T, False),
        ("Task: ready (0/1) [no]", T, False),
        ("Task: length [yes]", T, True),
        ("Task: completion_time [yes]", T, True),
        ("Task: memory_req_mb [yes]", T, True),
        ("Task: cpu_req_cores [yes]", T, True),
    ]
    for name, length, _pert in groups:
        sl = slice(idx, idx + length)
        labels.append(name)
        values.append(float(mean_abs[sl].mean()) if length > 0 else 0.0)
        idx += length

    # VM blocks (12 × V) in mapper order
    vm_groups = [
        ("VM: speed [yes]", V, True),
        ("VM: energy_rate [yes]", V, True),
        ("VM: completion_time [yes]", V, True),
        ("VM: memory_mb [yes]", V, True),
        ("VM: available_memory_mb [yes]", V, True),
        ("VM: used_memory_fraction [yes]", V, True),
        ("VM: active_tasks_count [yes]", V, True),
        ("VM: next_release_time [yes]", V, True),
        ("VM: cpu_cores (int) [no]", V, False),
        ("VM: available_cpu_cores (int) [no]", V, False),
        ("VM: used_cpu_fraction_cores [yes]", V, True),
        ("VM: next_core_release_time [yes]", V, True),
    ]
    for name, length, _pert in vm_groups:
        sl = slice(idx, idx + length)
        labels.append(name)
        values.append(float(mean_abs[sl].mean()) if length > 0 else 0.0)
        idx += length

    # Skip edges (2D + 2C)
    idx += 2 * D
    idx += 2 * C

    return labels, np.array(values, dtype=np.float32)


def plot_grouped_importance(labels: List[str], values: np.ndarray, out_path: str, dpi: int = 300, topk: int = 20):
    order = np.argsort(values)[::-1]
    order = order[: min(topk, len(order))]
    labels_top = [labels[i] for i in order]
    vals_top = values[order]

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(order)), vals_top[::-1], color="#4C72B0", alpha=0.85)
    plt.yticks(range(len(order)), [labels_top[i] for i in range(len(order)-1, -1, -1)])
    plt.xlabel("Mean |importance| (top action confidence)")
    plt.ylabel("Feature family")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_continuous_index_mask(header: np.ndarray) -> np.ndarray:
    """Return a boolean mask over [MAX_OBS_SIZE] marking continuous fields we can safely perturb.
    We exclude header, binary flags, integer indices (deps/compat), and integer counts.
    Layout per GinAgentMapper.map():
      header(4), then per-task blocks (6 arrays), per-vm blocks (12 arrays), then edges (ints).
    We choose continuous arrays:
      task_length, task_completion_time, task_memory_req_mb, task_cpu_req_cores,
      vm_speed, vm_energy_rate, vm_completion_time, vm_memory_mb,
      vm_available_memory_mb, vm_used_memory_fraction, vm_active_tasks_count,
      vm_next_release_time, vm_used_cpu_fraction_cores, vm_next_core_release_time.
    We skip: task_state_scheduled (int), task_state_ready (int), vm_cpu_cores (int), vm_available_cpu_cores (int),
    and the dependency/compatibility integer arrays at the end.
    """
    T = int(header[0]); V = int(header[1]); D = int(header[2]); C = int(header[3])
    m = np.zeros(MAX_OBS_SIZE, dtype=bool)

    def mark(start: int, length: int):
        m[start:start+length] = True

    idx = 4
    # task_state_scheduled (int)
    idx += T
    # task_state_ready (int)
    idx += T
    # task_length (float)
    mark(idx, T); idx += T
    # task_completion_time (float)
    mark(idx, T); idx += T
    # task_memory_req_mb (float)
    mark(idx, T); idx += T
    # task_cpu_req_cores (float)
    mark(idx, T); idx += T

    # VM blocks (each length V)
    # vm_speed
    mark(idx, V); idx += V
    # vm_energy_rate
    mark(idx, V); idx += V
    # vm_completion_time
    mark(idx, V); idx += V
    # vm_memory_mb
    mark(idx, V); idx += V
    # vm_available_memory_mb
    mark(idx, V); idx += V
    # vm_used_memory_fraction
    mark(idx, V); idx += V
    # vm_active_tasks_count (treated as continuous importance)
    mark(idx, V); idx += V
    # vm_next_release_time
    mark(idx, V); idx += V
    # vm_cpu_cores (int) -> skip
    idx += V
    # vm_available_cpu_cores (int) -> skip
    idx += V
    # vm_used_cpu_fraction_cores
    mark(idx, V); idx += V
    # vm_next_core_release_time
    mark(idx, V); idx += V

    # task_dependencies (2*D ints)
    idx += 2*D
    # compatibilities (2*C ints)
    idx += 2*C

    # zero out any padding indices beyond used length
    used_len = 4 + 6*T + 12*V + 2*D + 2*C
    if used_len < MAX_OBS_SIZE:
        m[used_len:] = False
    # Never include header
    m[:4] = False
    return m


def perturbation_importance(f: Callable[[np.ndarray], np.ndarray], eval_X: np.ndarray, header: np.ndarray,
                            per_feature_samples: int = 2, rel_eps: float = 0.01) -> np.ndarray:
    """Compute per-feature local sensitivity by small relative perturbations on continuous fields only.
    Returns an array [n_eval, d] of signed deltas; we will aggregate mean |delta|.
    """
    mask = build_continuous_index_mask(header)
    n, d = eval_X.shape
    base_y = f(eval_X)
    deltas = np.zeros((n, d), dtype=np.float32)

    rng = np.random.default_rng(123)
    # For each feature index allowed, do a couple of perturbations and average effect
    idxs = np.where(mask)[0]
    for j in idxs:
        acc = np.zeros(n, dtype=np.float32)
        for _ in range(per_feature_samples):
            Xp = eval_X.copy()
            # scale step size per-feature
            scale = np.maximum(1e-6, np.abs(eval_X[:, j]))
            step = rel_eps * scale * rng.choice([-1.0, 1.0])
            Xp[:, j] = eval_X[:, j] + step
            yp = f(Xp)
            acc += np.abs(yp - base_y)
        deltas[:, j] = acc / per_feature_samples
    return deltas


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(args.model_path, device)

    env = build_env(args.dataset)
    X = collect_observations(env, agent, args.max_decision_steps)
    env.close()

    # Background and evaluation sets for importance (keep small for speed)
    bg_size = min(60, len(X))
    eval_size = min(30, len(X))
    raw_background = X[:bg_size]
    raw_eval_X = X[:eval_size]

    # Prefilter decodable observations
    def is_valid(obs_np: np.ndarray) -> bool:
        try:
            x = torch.from_numpy(obs_np.astype(np.float32))
            hdr = x[:4].numpy().astype(np.float64)
            used_len = 4 + 6*int(hdr[0]) + 12*int(hdr[1]) + 2*int(hdr[2]) + 2*int(hdr[3])
            if used_len < len(x):
                x[used_len:] = 0.0
            _ = agent.mapper.unmap(x)
            return True
        except Exception:
            return False

    background = np.array([o for o in raw_background if is_valid(o)])
    eval_X = np.array([o for o in raw_eval_X if is_valid(o)])
    if len(background) == 0 or len(eval_X) == 0:
        raise SystemExit("No valid observations to analyze. Try increasing --max-decision-steps.")

    f = make_scalar_model(agent, reference_obs=background[0])
    # Compute perturbation-based importance
    importance = perturbation_importance(f, eval_X, header=background[0][:4], per_feature_samples=2, rel_eps=0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bar = str(out_dir / (Path(args.model_path).stem + "_shap_bar.png"))
    plot_importance(eval_X, importance, out_bar, dpi=args.dpi)
    print(f"Saved SHAP bar plot to: {out_bar}")

    # Grouped plot
    labels, vals = compute_grouped_importance(importance, header=background[0][:4])
    out_grouped = str(out_dir / (Path(args.model_path).stem + "_shap_grouped.png"))
    plot_grouped_importance(labels, vals, out_grouped, dpi=args.dpi, topk=20)
    print(f"Saved grouped importance plot to: {out_grouped}")

    # Beeswarm not applicable for this custom importance; skip


if __name__ == "__main__":
    main(tyro.cli(Args))
