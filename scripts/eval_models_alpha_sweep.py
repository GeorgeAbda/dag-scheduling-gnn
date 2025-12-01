#!/usr/bin/env python3
"""
Evaluate and compare multiple models in-domain (same DAG type and counts),
while varying task characteristics by an alpha factor away from the training distribution.

For each alpha value, we:
- Build a DatasetArgs derived from the training setup (e.g., gnp p=0.3, counts).
- Apply an alpha variation to task characteristics (task length range primarily).
  By default:
    min_task_length = base_min_task_length * (1 - alpha)
    max_task_length = base_max_task_length * (1 + alpha)
  (Clipped to sensible bounds and ensuring min <= max)
- Run short evaluations for each model and collect:
    total_energy (host active+idle is derived from env totals: total_energy_active + total_energy_idle)
    total_energy_active
    makespan
- Produce per-alpha CSVs and a combined plot comparing the models across metrics.

Usage example:
  python scripts/eval_models_alpha_sweep.py \
    --models \
      GNN:/Users/anashattay/Documents/GitHub/PERFECT/logs/1758445626_gnn_ablation_gnp_p03_baseline/ablation/per_variant/baseline_best_model.pt \
      NoGlobal:/Users/anashattay/Documents/GitHub/PERFECT/logs/1758441462_gnn_ablation_gnp_p03_noglobal/ablation/per_variant/no_global_actor_best_model.pt \
      MLP:/Users/anashattay/Documents/GitHub/PERFECT/logs/1758457165_gnn_ablation_gnp_p03_mlp/ablation/per_variant/mlp_only_best_model.pt \
    --variants baseline noglobal mlp_only \
    --dag gnp --gnp_p 0.3 \
    --hosts 4 --vms 10 --workflows 10 \
    --base_min_len 500 --base_max_len 100000 \
    --alpha_list 0.0 0.25 0.5 0.75 1.0 \
    --episodes 5 \
    --device cpu \
    --out_dir logs/alpha_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.utils.helpers import is_suitable


def pick_variant(name: str) -> AblationVariant:
    """Create variant configuration by name."""
    n = name.strip().lower()    
    if n in ("noglobal", "no_global_actor", "no_global"):
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    elif n == "homo":
        return AblationVariant(name="homogeneous")
    elif n in ("mlp_only", "mlp"):
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)
    elif n == "hetero":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    elif n == "hetero_noglobal":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True, use_actor_global_embedding=False)
    else:
        return AblationVariant(name="baseline")



def load_agent(ckpt: str, device: torch.device, variant_name: str) -> AblationGinAgent:
    var = pick_variant(variant_name)
    state = torch.load(ckpt, map_location=device)
    # For hetero variant, default to embedding_dim=16 as requested
    emb_dim = 8
    agent = AblationGinAgent(device, var, embedding_dim=emb_dim)
    try:
        agent.load_state_dict(state, strict=False)
    except RuntimeError as e:
        # Heuristic: infer whether use_actor_global_embedding should be False/True from edge_scorer weight shape
        try:
            w = state.get('actor.edge_scorer.0.weight', None)
            if isinstance(w, torch.Tensor):
                in_dim_ckpt = w.shape[1]
                # embedding_dim default is 8 in AblationGinAgent; in_dim formula: (2 + (1 if use_actor_global_embedding else 0)) * embedding_dim
                # Try to infer embedding dim from checkpoint if needed (supports 8 or 16)
                emb_guess = 8 if (in_dim_ckpt % 8 == 0) else (16 if (in_dim_ckpt % 16 == 0) else None)
                if emb_guess is not None:
                    k = in_dim_ckpt // emb_guess
                    use_global = (k >= 3)
                    if use_global != var.use_actor_global_embedding:
                        var2 = pick_variant(variant_name)
                        # Preserve hetero embedding dim preference
                        emb_dim2 = 16 if var2.graph_type == "hetero" else emb_guess
                        var2.use_actor_global_embedding = use_global
                        agent = AblationGinAgent(device, var2, embedding_dim=emb_dim2)
                        agent.load_state_dict(state, strict=False)
                    else:
                        raise e
                else:
                    raise e
            else:
                raise e
        except Exception:
            raise e
    agent.eval()
    return agent


def build_args(dag: str, p: float | None, hosts: int, vms: int, workflows: int,
               min_len: int, max_len: int, min_tasks: int, max_tasks: int, device: str) -> AblArgs:
    a = AblArgs()
    a.device = device
    if dag == 'gnp':
        a.dataset = DatasetArgs(
            host_count=hosts,
            vm_count=vms,
            workflow_count=workflows,
            dag_method="gnp",
            gnp_min_n=min_tasks,
            gnp_max_n=max_tasks,
            gnp_p=p,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=min_len,
            max_task_length=max_len,
            task_arrival="static",
        )
    else:
        a.dataset = DatasetArgs(
            host_count=hosts,
            vm_count=vms,
            workflow_count=workflows,
            dag_method="linear",
            gnp_min_n=min_tasks,
            gnp_max_n=max_tasks,
            gnp_p=None,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=min_len,
            max_task_length=max_len,
            task_arrival="static",
        )
    return a


def apply_alpha_on_lengths(base_min: int, base_max: int, alpha: float) -> Tuple[int, int]:
    # Expand the range symmetrically: decrease min, increase max
    new_min = int(round(max(1, base_min * (1 - alpha))))
    new_max = int(round(max(new_min + 1, base_max * (1 + alpha))))
    return new_min, new_max


def apply_alpha_prime_on_tasks(base_min_tasks: int, base_max_tasks: int, alpha_prime: float) -> Tuple[int, int]:
    """Scale the min/max number of tasks per workflow relative to training.
    For alpha' >= 0, we set:
      min_tasks = round(base_min_tasks * (1 + alpha'))
      max_tasks = round(base_max_tasks * (1 + alpha'))
    Ensure min_tasks >= 2 and max_tasks >= min_tasks+1.
    """
    new_min = int(round(max(2, base_min_tasks * (1 + alpha_prime))))
    new_max = int(round(max(new_min + 1, base_max_tasks * (1 + alpha_prime))))
    return new_min, new_max


def _apply_scaling_and_rebuild_compat(env, alpha: float, 
                                      feasibility_strategy: str = 'clamp_to_vm',
                                      feasible_vm_policy: str = 'max_capacity') -> None:
    """Scale task CPU core requirements and memory by alpha and rebuild compatibilities.
    - CPU cores: req_cpu_cores <- ceil(req_cpu_cores * (1 + alpha))
    - Memory:    req_memory_mb <- round(req_memory_mb * (1 + mem_alpha * alpha))
    Both clamped to VM capacities. Updates env.state.static_state.compatibilities and env._compat_bool.
    """
    try:
        tasks = env.state.static_state.tasks
        vms = env.state.static_state.vms
        T = len(tasks)
        V = len(vms)
        max_vm_cores = max(int(v.cpu_cores) for v in vms)
        max_vm_mem = max(int(v.memory_mb) for v in vms)
        # Scale requirements
        for t in tasks:
            # CPU cores
            new_cores = int(np.ceil(max(1, t.req_cpu_cores) * (1.0 + float(alpha))))
            new_cores = int(np.clip(new_cores, 1, max_vm_cores))
            t.req_cpu_cores = new_cores
            # Memory
            factor_m = 1 + float(alpha)

            new_mem = int(np.round(max(1, t.req_memory_mb) * factor_m))
            new_mem = int(np.clip(new_mem, 1024, max_vm_mem))
            t.req_memory_mb = new_mem
        # Rebuild compatibilities
        compatibilities = []
        compat_bool = np.zeros((T, V), dtype=bool)
        for ti in range(T):
            task = tasks[ti]
            pairs = []
            for vi in range(V):
                if is_suitable(vms[vi], task):
                    pairs.append((ti, vi))
            # Safety knob: ensure at least one feasible VM if requested
            if not pairs and feasibility_strategy == 'clamp_to_vm':
                # choose a VM according to policy
                if feasible_vm_policy == 'max_capacity':
                    vm_idx = max(range(V), key=lambda i: (int(vms[i].memory_mb) * max(1, int(vms[i].cpu_cores))))
                else:  # best_fit: minimize relative overflow
                    def overflow(i: int) -> float:
                        mem_over = max(0, int(task.req_memory_mb) - int(vms[i].memory_mb)) / max(1, int(vms[i].memory_mb))
                        core_over = max(0, int(task.req_cpu_cores) - int(vms[i].cpu_cores)) / max(1, int(vms[i].cpu_cores))
                        return mem_over + core_over
                    vm_idx = min(range(V), key=overflow)
                # Clamp task requirements to this VM capacities
                task.req_memory_mb = min(int(task.req_memory_mb), int(vms[vm_idx].memory_mb))
                task.req_cpu_cores = min(int(task.req_cpu_cores), int(vms[vm_idx].cpu_cores))
                # Re-check suitability and add at least this pair
                if is_suitable(vms[vm_idx], task):
                    pairs.append((ti, vm_idx))
            compatibilities.extend(pairs)
            for _t, _v in pairs:
                compat_bool[_t, _v] = True

        env.state.static_state.compatibilities = compatibilities
        env._compat_bool = compat_bool
    except Exception:
        # Be robust; if anything fails, keep original compatibilities
        pass


def plot_comparison_vs_alpha_prime(alpha_primes: list[float], per_model_metric_series: dict[str, dict[str, list[float]]], out_dir: Path, title: str = '') -> None:
    """Plot one figure per metric: mean vs alpha' (task count scaling), with one curve per model.
    per_model_metric_series format: { model_label: { metric_key: [values aligned with alpha_primes] } }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ('total_energy_host_mean', 'Total Energy (Host = Active+Idle)'),
        ('total_energy_active_mean', 'Active Energy'),
        ('total_energy_idle_mean', 'Idle Energy'),
        ('makespan_mean', 'Makespan'),
    ]
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for met_key, met_label in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        for i, (model_label, series) in enumerate(per_model_metric_series.items()):
            ys = series.get(met_key, [])
            if len(ys) != len(alpha_primes):
                # pad or skip if mismatch
                continue
            ax.plot(alpha_primes, ys, marker='o', color=colors[i % len(colors)], label=model_label)
        ax.set_xlabel("alpha' (task count scaling)")
        ax.set_ylabel(met_label)
        ax.set_title(f"{met_label} vs alpha' — {title}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fname = {
            'makespan_mean': 'makespan_mean_vs_alpha_prime.png',
            'total_energy_active_mean': 'total_energy_active_mean_vs_alpha_prime.png',
            'total_energy_idle_mean': 'total_energy_idle_mean_vs_alpha_prime.png',
            'total_energy_host_mean': 'total_energy_host_mean_vs_alpha_prime.png',
        }.get(met_key, f'{met_key}_vs_alpha_prime.png')
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)


def run_policy_metrics(agent: AblationGinAgent, args: AblArgs, episodes: int, alpha: float,
                       feasibility_strategy: str, feasible_vm_policy: str) -> Tuple[List[dict], dict]:
    device = agent.device
    per_ep: List[dict] = []
    for ep in range(episodes):
        env = _make_test_env(args)
        # Apply scaling knobs and rebuild compatibilities
        _apply_scaling_and_rebuild_compat(env, alpha=alpha, 
                                          feasibility_strategy=feasibility_strategy,
                                          feasible_vm_policy=feasible_vm_policy)
        obs, _ = env.reset()
        final_makespan = None
        totals = {"total_energy": np.nan, "total_energy_active": np.nan, "total_energy_idle": np.nan, "makespan": np.nan}
        with torch.no_grad():
            while True:
                x = torch.tensor(obs, dtype=torch.float32, device=device)
                action, _, _, _ = agent.get_action_and_value(x.unsqueeze(0))
                nxt, reward, term, trunc, info = env.step(action.cpu().numpy())
                if bool(term) or bool(trunc):
                    # env returns makespan by -reward
                    # final_makespan = float(-reward)

                    if isinstance(info, dict):
                        for k in list(totals.keys()):
                            if k in info and isinstance(info[k], (int, float)):
                                totals[k] = float(info[k])
                    break
                obs = nxt
        env.close()
        per_ep.append({
            'episode': ep,
            'makespan': float(totals['makespan']),
            'total_energy': float(totals['total_energy']),
            'total_energy_active': float(totals['total_energy_active']),
            'total_energy_idle': float(totals['total_energy_idle']),
            'total_energy_host': float(totals['total_energy_active'] + totals['total_energy_idle']) if not (np.isnan(totals['total_energy_active']) or np.isnan(totals['total_energy_idle'])) else np.nan,
        })
    # summary
    def meanstd(arr):
        a = np.array(arr, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))
    ms_mean, ms_std = meanstd([r['makespan'] for r in per_ep])
    tea_mean, tea_std = meanstd([r['total_energy_active'] for r in per_ep])
    tei_mean, tei_std = meanstd([r['total_energy_idle'] for r in per_ep])
    teh_mean, teh_std = meanstd([r['total_energy_host'] for r in per_ep])
    return per_ep, {
        'makespan_mean': ms_mean, 'makespan_std': ms_std,
        'total_energy_active_mean': tea_mean, 'total_energy_active_std': tea_std,
        'total_energy_idle_mean': tei_mean, 'total_energy_idle_std': tei_std,
        'total_energy_host_mean': teh_mean, 'total_energy_host_std': teh_std,
    }


def plot_comparison(alphas: list[float], summaries: dict[str, dict[float, dict]], out_dir: Path, title: str = '') -> None:
    # summaries: {model_label: {alpha_value: summary_dict}}
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ('total_energy_host_mean', 'Total Energy (Host = Active+Idle)'),
        ('total_energy_active_mean', 'Active Energy'),
        ('total_energy_idle_mean', 'Idle Energy'), 
        ('makespan_mean', 'Makespan'),
    ]
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for met_key, met_label in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        for i, (model_label, per_alpha) in enumerate(summaries.items()):
            ys = [per_alpha[a][met_key] for a in alphas]
            ax.plot(alphas, ys, marker='o', color=colors[i % len(colors)], label=model_label)
        ax.set_xlabel('alpha (task distribution variation)')
        ax.set_ylabel(met_label)
        ax.set_title(f'{met_label} vs alpha — {title}')
        ax.legend(frameon=False)
        fig.tight_layout()
        # Save with metric-specific filename
        fname = {
            'makespan_mean': 'makespan_mean_vs_alpha.png',
            'total_energy_active_mean': 'total_energy_active_mean_vs_alpha.png',
            'total_energy_idle_mean': 'total_energy_idle_mean_vs_alpha.png',
            'total_energy_host_mean': 'total_energy_host_mean_vs_alpha.png',
        }.get(met_key, f'{met_key}_vs_alpha.png')
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Scaling summary plot (normalized to alpha=0 for comparability)
    try:
        if alpha_stats['alpha']:
            import csv
            # Write CSV of raw means
            with (out_dir / 'scaling_summary.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['alpha','mean_req_cpu_cores','mean_req_memory_mb','mean_task_length'])
                for i in range(len(alpha_stats['alpha'])):
                    w.writerow([
                        alpha_stats['alpha'][i],
                        alpha_stats['mean_req_cpu_cores'][i],
                        alpha_stats['mean_req_memory_mb'][i],
                        alpha_stats['mean_task_length'][i],
                    ])

            # Normalize by first entry (alpha closest to 0)
            base_cpu = max(alpha_stats['mean_req_cpu_cores'][0], 1e-8)
            base_mem = max(alpha_stats['mean_req_memory_mb'][0], 1e-8)
            base_len = max(alpha_stats['mean_task_length'][0], 1e-8)
            norm_cpu = [v / base_cpu for v in alpha_stats['mean_req_cpu_cores']]
            norm_mem = [v / base_mem for v in alpha_stats['mean_req_memory_mb']]
            norm_len = [v / base_len for v in alpha_stats['mean_task_length']]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(alpha_stats['alpha'], norm_cpu, marker='o', label='req_cpu_cores (norm)')
            ax.plot(alpha_stats['alpha'], norm_mem, marker='s', label='req_memory_mb (norm)')
            ax.plot(alpha_stats['alpha'], norm_len, marker='^', label='task_length (norm)')
            ax.set_xlabel('alpha')
            ax.set_ylabel('normalized mean (relative to alpha=0)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / 'scaling_summary.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            # Cross-alpha boxplots for distributions
            try:
                # Helper to build labels
                labels = [f"{a:.2f}" for a in alpha_distros['alpha']]
                # CPU cores boxplot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot([arr for arr in alpha_distros['req_cpu_cores']], tick_labels=labels, showfliers=False)
                ax.set_title('req_cpu_cores distribution by alpha')
                ax.set_xlabel('alpha')
                ax.set_ylabel('req_cpu_cores')
                fig.tight_layout()
                fig.savefig(out_dir / 'box_req_cpu_cores_by_alpha.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                # Memory boxplot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot([arr for arr in alpha_distros['req_memory_mb']], tick_labels=labels, showfliers=False)
                ax.set_title('req_memory_mb distribution by alpha')
                ax.set_xlabel('alpha')
                ax.set_ylabel('req_memory_mb')
                fig.tight_layout()
                fig.savefig(out_dir / 'box_req_memory_mb_by_alpha.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                # Length boxplot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot([arr for arr in alpha_distros['task_length']], tick_labels=labels, showfliers=False)
                ax.set_title('task_length (MI) distribution by alpha')
                ax.set_xlabel('alpha')
                ax.set_ylabel('task_length (MI)')
                fig.tight_layout()
                fig.savefig(out_dir / 'box_task_length_by_alpha.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Compare multiple models under in-domain DAG while varying task characteristics by alpha.')
    ap.add_argument('--models', nargs='+', required=True, help='List like Label:/path/to/ckpt.pt ...')
    ap.add_argument('--variants', nargs='+', required=True, help='Variant names aligned with models (baseline|noglobal|mlp_only|hetero)')
    ap.add_argument('--dag', type=str, default='gnp', choices=['gnp','linear'])
    ap.add_argument('--gnp_p', type=float, default=0.3)
    ap.add_argument('--hosts', type=int, default=4)
    ap.add_argument('--vms', type=int, default=10)
    ap.add_argument('--workflows', type=int, default=10)
    ap.add_argument('--base_min_len', type=int, default=500)
    ap.add_argument('--base_max_len', type=int, default=100_000)
    ap.add_argument('--base_min_tasks', type=int, default=12, help='Training min tasks per workflow')
    ap.add_argument('--base_max_tasks', type=int, default=24, help='Training max tasks per workflow')
    ap.add_argument('--alpha_list', nargs='+', type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--alpha_prime_list', nargs='+', type=float, default=[0.0], help='Scales number of tasks per workflow relative to training (min/max)')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--feasibility_strategy', type=str, default='clamp_to_vm', choices=['none','clamp_to_vm'],
                    help='If a task has no compatible VMs after scaling, either do nothing (none) or clamp requirements to a VM capacity (clamp_to_vm).')
    ap.add_argument('--feasible_vm_policy', type=str, default='max_capacity', choices=['max_capacity','best_fit'],
                    help='VM selection policy used by clamp_to_vm safety knob.')
    ap.add_argument('--out_dir', type=str, default='logs/alpha_sweep')
    args = ap.parse_args()

    device = torch.device(args.device)

    # Parse models
    labels: List[str] = []
    ckpts: List[str] = []
    if len(args.models) != len(args.variants):
        raise SystemExit('Error: --models and --variants must have the same length')
    for spec in args.models:
        if ':' not in spec:
            raise SystemExit('Error: --models entries must be Label:/full/path/to/ckpt.pt')
        label, path = spec.split(':', 1)
        labels.append(label)
        ckpts.append(path)

    # Load agents
    agents: List[AblationGinAgent] = []
    for ckpt, var in zip(ckpts, args.variants):
        agents.append(load_agent(ckpt, device, var))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep alpha-prime (task count scaling) and, inside, alpha (length/cpu/mem scaling)
    alpha_prime_values = [float(a) for a in args.alpha_prime_list]
    alpha_values = [float(a) for a in args.alpha_list]
    summaries_by_model: dict[str, dict[float, dict]] = {lab: {} for lab in labels}
    # For scaling summary across alphas (reinitialized per alpha-prime)

    # Collect summaries per alpha-prime
    all_prime_summaries: dict[float, dict[str, dict[float, dict]] ] = {}
    for alpha_prime in alpha_prime_values:
        # Prepare per-alpha-prime directory
        prime_dir = out_dir / f'tasks_{alpha_prime:.2f}'
        prime_dir.mkdir(parents=True, exist_ok=True)
        # Reinitialize scaling stats for plots within this alpha'
        global alpha_stats, alpha_distros
        alpha_stats = {
            'alpha': [],
            'mean_req_cpu_cores': [],
            'mean_req_memory_mb': [],
            'mean_task_length': [],
        }
        alpha_distros = {
            'alpha': [],
            'req_cpu_cores': [],
            'req_memory_mb': [],
            'task_length': [],
        }
        summaries_by_model: dict[str, dict[float, dict]] = {lab: {} for lab in labels}
        # Scale number of tasks per workflow according to alpha'
        min_tasks, max_tasks = apply_alpha_prime_on_tasks(args.base_min_tasks, args.base_max_tasks, alpha_prime)
        for alpha in alpha_values:
            min_len, max_len = apply_alpha_on_lengths(args.base_min_len, args.base_max_len, alpha)
            cfg_dir = prime_dir / f'alpha_{alpha:.2f}'
            cfg_dir.mkdir(parents=True, exist_ok=True)
            # Build args (vary task lengths and number of tasks per workflow)
            eval_args = build_args(args.dag, args.gnp_p if args.dag=='gnp' else None,
                                   args.hosts, args.vms, args.workflows,
                                   min_len, max_len, min_tasks, max_tasks, args.device)
            # Gather scaling stats by instantiating env once, applying scaling, and reading state
            try:
                tmp_env = _make_test_env(eval_args)
                tmp_env.reset()
                _apply_scaling_and_rebuild_compat(tmp_env, alpha=alpha,
                                                  feasibility_strategy=args.feasibility_strategy,
                                                  feasible_vm_policy=args.feasible_vm_policy)
                tasks = tmp_env.unwrapped.state.static_state.tasks
                if tasks:
                    alpha_stats['alpha'].append(alpha)
                    req_cores = np.array([int(t.req_cpu_cores) for t in tasks], dtype=np.int32)
                    req_mem = np.array([int(t.req_memory_mb) for t in tasks], dtype=np.int32)
                    lengths = np.array([float(t.length) for t in tasks], dtype=np.float64)
                    alpha_stats['mean_req_cpu_cores'].append(float(req_cores.mean()))
                    alpha_stats['mean_req_memory_mb'].append(float(req_mem.mean()))
                    alpha_stats['mean_task_length'].append(float(lengths.mean()))
                    # Save per-alpha histograms
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(req_cores, bins=20, color='#1f77b4', alpha=0.8)
                        ax.set_title(f'Req CPU cores — alpha={alpha:.2f}')
                        ax.set_xlabel('req_cpu_cores')
                        ax.set_ylabel('count')
                        fig.tight_layout()
                        fig.savefig(cfg_dir / 'hist_req_cpu_cores.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(req_mem, bins=20, color='#2ca02c', alpha=0.8)
                        ax.set_title(f'Req memory (MB) — alpha={alpha:.2f}')
                        ax.set_xlabel('req_memory_mb')
                        ax.set_ylabel('count')
                        fig.tight_layout()
                        fig.savefig(cfg_dir / 'hist_req_memory_mb.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(lengths, bins=20, color='#d62728', alpha=0.8)
                        ax.set_title(f'Task length (MI) — alpha={alpha:.2f}')
                        ax.set_xlabel('task_length (MI)')
                        ax.set_ylabel('count')
                        fig.tight_layout()
                        fig.savefig(cfg_dir / 'hist_task_length.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    except Exception:
                        pass
                    # Store for cross-alpha boxplots
                    alpha_distros['alpha'].append(alpha)
                    alpha_distros['req_cpu_cores'].append(req_cores)
                    alpha_distros['req_memory_mb'].append(req_mem)
                    alpha_distros['task_length'].append(lengths)
                tmp_env.close()
            except Exception:
                pass
            # Evaluate each model
            for label, agent in zip(labels, agents):
                per_ep, summary = run_policy_metrics(agent, eval_args, args.episodes, alpha=alpha,
                                                     feasibility_strategy=args.feasibility_strategy,
                                                     feasible_vm_policy=args.feasible_vm_policy)
                summaries_by_model[label][alpha] = summary
                # Write per-episode CSV
                import csv
                with (cfg_dir / f'metrics_{label}.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['episode','makespan','total_energy','total_energy_active','total_energy_idle','total_energy_host'])
                    w.writeheader()
                    for r in per_ep:
                        w.writerow(r)
                # Write summary CSV per config
                with (cfg_dir / f'metrics_{label}_summary.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(summary.keys()))
                    w.writeheader(); w.writerow(summary)

        # Plots across models for each metric vs alpha (within this alpha')
        plot_comparison(alpha_values, summaries_by_model, prime_dir, title=f"{args.dag} tasks'(min={min_tasks},max={max_tasks}), counts(H={args.hosts},V={args.vms},W={args.workflows})")

        # Also write a combined summary table across models and alphas for this alpha'
        import csv
        with (prime_dir / 'combined_summary.csv').open('w', newline='') as f:
            fieldnames = ['model','alpha','makespan_mean','makespan_std','total_energy_active_mean','total_energy_active_std','total_energy_idle_mean','total_energy_idle_std','total_energy_host_mean','total_energy_host_std']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for model, d in summaries_by_model.items():
                for alpha in alpha_values:
                    s = d[alpha]
                    w.writerow({'model': model, 'alpha': alpha, **s})

        # Store for cross alpha-prime aggregation
        all_prime_summaries[alpha_prime] = summaries_by_model

    # Build per-model metric series vs alpha-prime
    metrics_keys = ['makespan_mean', 'total_energy_active_mean', 'total_energy_idle_mean', 'total_energy_host_mean']
    per_model_metric_series: dict[str, dict[str, list[float]]] = { lab: {k: [] for k in metrics_keys} for lab in labels }
    for ap in alpha_prime_values:
        summaries_for_ap = all_prime_summaries.get(ap, {})
        for lab in labels:
            per_alpha = summaries_for_ap.get(lab, {})  # dict alpha -> summary
            # Prefer alpha=0.0 slice; else average across alphas for that model
            chosen = None
            if 0.0 in per_alpha:
                chosen = per_alpha[0.0]
            elif len(per_alpha) > 0:
                # average across available alphas for each metric
                for k in metrics_keys:
                    vals = [float(per_alpha[a][k]) for a in per_alpha.keys() if k in per_alpha[a]]
                    v = float(np.mean(vals)) if len(vals)>0 else float('nan')
                    per_model_metric_series[lab][k].append(v)
                continue
            else:
                # no data for this alpha-prime/model
                for k in metrics_keys:
                    per_model_metric_series[lab][k].append(float('nan'))
                continue
            # append chosen slice values
            for k in metrics_keys:
                per_model_metric_series[lab][k].append(float(chosen.get(k, float('nan'))))

    # Plot vs alpha-prime at out_dir root
    plot_comparison_vs_alpha_prime(alpha_prime_values, per_model_metric_series, out_dir, title=f"{args.dag} counts(H={args.hosts},V={args.vms},W={args.workflows})")

    print(f"[alpha, alpha'] Wrote results to: {out_dir}")


if __name__ == '__main__':
    main()
