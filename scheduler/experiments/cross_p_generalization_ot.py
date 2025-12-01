#!/usr/bin/env python3
"""
Cross-p generalization and OT mapping experiments.

This script orchestrates two studies over G(n,p) datasets:

1) Cross-p performance matrix
   - Train or load heterogeneous-architecture agents trained at specified p values
     (we will train for p in {0.7, 0.5, 0.1} if checkpoints are missing; use provided
      checkpoints for p in {0.8, 0.3}).
   - Evaluate every trained model on all eval p values in {0.8, 0.7, 0.5, 0.3, 0.1}.
   - Save a CSV matrix with performance metrics and a summary of relative drops
     versus the in-domain (train p == eval p) baseline.

2) OT mapping of edge embeddings (per target p_t)
   - For pairs (p_s -> p_t), collect edge-embedding distributions for:
       A) target model trained on p_t and evaluated on p_t (reference, X)
       B) source model trained on p_s and evaluated on p_t (eval, Y)
   - Compute OT distances (Sinkhorn, GW if available; SWD fallback) and the
     barycentric coupling Pi (if GW available).
   - Map the source model's scorer inputs for sampled states using the barycentric map
     Y -> X_bary(Y) and feed those into the source edge_scorer to approximate a
     'mapped policy'. Compare its logits/top-k choices with the target model's logits
     on the same states to assess how well mapping recovers target behavior.

Outputs:
  - Cross-p matrix CSV: cross_p_results.csv
  - Drop summary CSV: cross_p_relative_drop.csv
  - OT summary CSV: ot_mapping_summary.csv
  - Optional: plots can be added later if needed

Example usage:
  python -m scheduler.experiments.cross_p_generalization_ot \
    --ps-train "0.7,0.5,0.1" \
    --ps-all "0.8,0.7,0.5,0.3,0.1" \
    --hetero-p08 /path/to/hetero_best_model_p0.8.pt \
    --hetero-p03 /path/to/hetero_best_model_p0.3.pt \
    --train-missing \
    --total-timesteps 150000 \
    --hosts 4 --vms 10 --workflows 10 --min-tasks 12 --max-tasks 24 \
    --test-iters 4 --episodes 3 --device cpu \
    --out-dir scheduler/viz_results/decision_boundaries/cross_p

Notes:
  - Training is only performed for missing heterogeneous checkpoints at ps-train.
  - Checkpoints are automatically saved under out-dir/runs/<timestamp>/ablation/per_variant/.
  - We default to variant 'hetero' from ablation_gnn.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import shutil

# Optional dependencies for OT
try:
    import ot  # type: ignore
except Exception:
    ot = None
try:
    from sklearn.metrics.pairwise import pairwise_distances  # type: ignore
except Exception:
    pairwise_distances = None  # type: ignore

# Project-relative imports
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.rl_model.ablation_gnn import (
    Args as AblArgs,
    AblationVariant,
    AblationGinAgent,
    _test_agent,
    main as ablation_main,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.viz_results.compare_embeddings_panel import (
    build_args_gnp,
    load_agent as load_agent_panel,
    extract_embeddings_and_labels,
)


# ----------------------------
# Utilities: normalization and OT
# ----------------------------

def normalize(X: np.ndarray, kind: str = "l2") -> np.ndarray:
    if X.size == 0:
        return X
    if kind == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / n
    if kind == "z":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / sd
    return X


def sinkhorn_distance(X: np.ndarray, Y: np.ndarray, reg: float = 1e-2, num_iter: int = 3000) -> float:
    if ot is None:
        return float('nan')
    if X.size == 0 or Y.size == 0:
        return float('nan')
    a = np.ones(X.shape[0], dtype=np.float64) / max(1, X.shape[0])
    b = np.ones(Y.shape[0], dtype=np.float64) / max(1, Y.shape[0])
    C = ot.utils.dist(X, Y, metric='euclidean')
    C /= (C.max() + 1e-12)
    try:
        W = ot.sinkhorn(a, b, C, reg, numItermax=num_iter)
        cost = float((W * C).sum())
    except Exception:
        try:
            cost = float(ot.emd2(a, b, C))
        except Exception:
            cost = float('nan')
    return cost


def gw_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'cosine', epsilon: float = 5e-3):
    if ot is None or pairwise_distances is None:
        return float('nan'), None
    if X.size == 0 or Y.size == 0:
        return float('nan'), None
    try:
        n, m = X.shape[0], Y.shape[0]
        a = np.ones(n, dtype=np.float64) / max(1, n)
        b = np.ones(m, dtype=np.float64) / max(1, m)
        DX = pairwise_distances(X, X, metric=metric)
        DY = pairwise_distances(Y, Y, metric=metric)
        gw, log = ot.gromov.gromov_wasserstein2(DX, DY, a, b, loss_fun='square_loss', epsilon=epsilon, log=True)
        Pi = log.get('T', None)
        return float(gw), (Pi if isinstance(Pi, np.ndarray) else None)
    except Exception:
        return float('nan'), None


def barycentric_project_Y_to_X(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Return X_bary_from_Y for each Y point under coupling Pi (shape (m,d))."""
    if Pi is None or Pi.size == 0:
        return np.zeros_like(Y)
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    XbY = (X.T @ (Pi / col)).T
    return XbY


# ----------------------------
# Training / loading
# ----------------------------

def build_training_args(p: float, device: str, total_timesteps: int, hosts: int, vms: int, workflows: int,
                        min_tasks: int, max_tasks: int, train_test_every: int, eval_episodes: int, seed: int, output_dir: Path) -> AblArgs:
    a = AblArgs()
    a.device = device
    a.seed = seed
    a.output_dir = str(output_dir)
    a.exp_name = f"hetero_p{p:.1f}"
    a.total_timesteps = int(total_timesteps)
    a.test_every_iters = max(1, int(train_test_every))
    a.test_iterations = max(1, int(eval_episodes))
    a.train_only_variant = "hetero"
    a.skip_variant_summary_eval = True
    a.skip_additional_eval = True
    a.dataset = DatasetArgs(
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
        dag_method="gnp",
    )
    # Set gnp p
    a.dataset.gnp_p = float(p)
    return a


def train_hetero_if_missing(p: float, device: str, total_timesteps: int, hosts: int, vms: int, workflows: int,
                            min_tasks: int, max_tasks: int, train_test_every: int, eval_episodes: int, seed: int, out_dir: Path) -> Path | None:
    """Train a heterogeneous agent for p if no checkpoint is found under out_dir.
    Returns path to the best checkpoint if exists after training, else None.
    """
    # Heuristic checkpoint path after training (see ablation_gnn._train_one_variant)
    # Best model is saved as per_variant/hetero_best_model.pt
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Train to a new run dir
    args = build_training_args(p, device, total_timesteps, hosts, vms, workflows, min_tasks, max_tasks,
                               train_test_every, eval_episodes, seed, runs_dir)
    try:
        ablation_main(args)  # This will write checkpoints under runs/<ts>/ablation/per_variant/
    except SystemExit:
        # tyro or argparse inside ablation_main may raise SystemExit; ignore here
        pass
    except Exception as e:
        print(f"[train] Error during training for p={p}: {e}")

    # Find most recent hetero_best_model.pt under runs/*/ablation/per_variant/
    candidates: List[Path] = list(runs_dir.glob("*/ablation/per_variant/hetero_best_model.pt"))
    if not candidates:
        # Fallback to final model name
        candidates = list(runs_dir.glob("*/ablation/per_variant/hetero_model.pt"))
    if not candidates:
        print(f"[train] No checkpoint found after training for p={p}")
        return None
    # Pick the latest by mtime
    best = max(candidates, key=lambda pth: pth.stat().st_mtime)
    # Export to a stable checkpoint path for downstream usage
    ckpt_out_dir = out_dir / "checkpoints"
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_out_dir / f"hetero_p{p:.1f}_best_model.pt"
    try:
        shutil.copy2(best, dst)
        print(f"[train] Exported best checkpoint for p={p} to: {dst}")
    except Exception as e:
        print(f"[train] Warning: could not export best checkpoint to {dst}: {e}")
        # Fallback to returning discovered best directly
        return best
    return dst


# ----------------------------
# Loading and evaluation helpers
# ----------------------------
def _resolve_best_in_dir(root: Path) -> Path | None:
    """Search recursively under 'root' for the latest hetero_best_model.pt (fallback to hetero_model.pt)."""
    if not root.exists() or not root.is_dir():
        return None
    cands = list(root.rglob("hetero_best_model.pt"))
    if not cands:
        cands = list(root.rglob("hetero_model.pt"))
    if not cands:
        return None
    return max(cands, key=lambda pth: pth.stat().st_mtime)


def resolve_best_checkpoint(path_str: str) -> Path | None:
    """Given a file or directory path, resolve the best checkpoint to use.
    - If it's a file, return it (prefer *_best_model.pt by user selection).
    - If it's a directory, search for latest hetero_best_model.pt (fallback hetero_model.pt).
    """
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        return _resolve_best_in_dir(p)
    return None

def build_eval_args(p: float, hosts: int, vms: int, workflows: int, min_tasks: int, max_tasks: int, device: str,
                    eval_episodes: int, eval_seed_base: int = 1_000_000_001) -> AblArgs:
    a = build_args_gnp(p, hosts, vms, workflows, min_tasks, max_tasks, device)
    a.test_iterations = max(1, int(eval_episodes))
    setattr(a, 'eval_seed_base', int(eval_seed_base))
    return a


def eval_agent(agent: AblationGinAgent, eval_args: AblArgs) -> Tuple[float, float, float, dict]:
    return _test_agent(agent, eval_args)


def load_hetero_agent(ckpt_path: Path, device: str) -> AblationGinAgent:
    return load_agent_panel(str(ckpt_path), device=device, variant_name='hetero')  # type: ignore[arg-type]


# ----------------------------
# Embedding and scorer I/O for OT experiment
# ----------------------------

def collect_edge_embeddings(agent: AblationGinAgent, p: float, hosts: int, vms: int, workflows: int,
                            min_tasks: int, max_tasks: int, device: str, episodes: int,
                            seed_base: int) -> np.ndarray:
    args = build_args_gnp(p, hosts, vms, workflows, min_tasks, max_tasks, device)
    X, _ = extract_embeddings_and_labels(agent, args, episodes=episodes, seed_base=seed_base)
    return X.astype(np.float32)


def collect_scorer_inputs_and_logits(agent: AblationGinAgent, p: float, hosts: int, vms: int, workflows: int,
                                     min_tasks: int, max_tasks: int, device: str, episodes: int,
                                     seed_base: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect scorer input tensors and corresponding edge_scorer logits across 'episodes' states.
    Returns two parallel lists of numpy arrays per episode: (X_in_list, logits_list), where
    X_in has shape (E, D_in) and logits has shape (E,).
    """
    X_in_list: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    for ep in range(episodes):
        args = build_args_gnp(p, hosts, vms, workflows, min_tasks, max_tasks, device)
        env = None
        try:
            from scheduler.rl_model.ablation_gnn import _make_test_env as _make_env
            env = _make_env(args)
            obs_np, _ = env.reset(seed=seed_base + ep)
            obs_t = torch.tensor(np.asarray(obs_np, dtype=np.float32))
            obs = agent.mapper.unmap(obs_t)
            with torch.no_grad():
                _, edge_embeddings, graph_embedding = agent.actor.network(obs)
                E = int(obs.compatibilities.shape[1])
                edge_embeddings = edge_embeddings[:E]
                scorer_in = edge_embeddings
                if getattr(agent.variant, 'use_actor_global_embedding', False):
                    rep_graph = graph_embedding.expand(edge_embeddings.shape[0], agent.actor.embedding_dim)
                    scorer_in = torch.cat([scorer_in, rep_graph], dim=1)
                logits = agent.actor.edge_scorer(scorer_in).flatten()
            X_in_list.append(scorer_in.cpu().numpy().astype(np.float32))
            logits_list.append(logits.cpu().numpy().astype(np.float32))
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
    return X_in_list, logits_list


def evaluate_with_mapped_embeddings(agent: AblationGinAgent, eval_args: AblArgs, W: np.ndarray | None,
                                    split_assume_half: bool = True) -> Tuple[float, float, float, dict]:
    """Evaluate a source agent on eval_args using mapped edge embeddings via linear map W.
    We re-implement a lightweight evaluation loop similar to _test_agent, but replace the
    edge_scorer input with mapped embeddings before computing logits.
    """
    from scheduler.rl_model.ablation_gnn import _make_test_env as _make_env
    # Ensure deterministic evaluation: eval mode and fixed RNG per episode
    try:
        agent.eval()
    except Exception:
        pass
    total_makespan = 0.0
    total_energy_obs = 0.0
    total_energy_full = 0.0
    total_active = 0.0
    total_idle = 0.0
    seed_base = getattr(eval_args, 'eval_seed_base', 1_000_000_001)

    for s in range(eval_args.test_iterations):
        try:
            torch.manual_seed(int(seed_base + s))
            np.random.seed(int(seed_base + s) & 0x7fffffff)
        except Exception:
            pass
        env = _make_env(eval_args)
        next_obs, _ = env.reset(seed=seed_base + s)
        final_info: dict | None = None
        while True:
            # Build obs struct
            obs_t = torch.from_numpy(np.asarray(next_obs, dtype=np.float32))
            obs = agent.mapper.unmap(obs_t)
            with torch.no_grad():
                # Backbone embeddings
                _, edge_embeddings, graph_embedding = agent.actor.network(obs)
                E = int(obs.compatibilities.shape[1])
                edge_embeddings = edge_embeddings[:E]
                scorer_in = edge_embeddings
                if getattr(agent.variant, 'use_actor_global_embedding', False):
                    rep_graph = graph_embedding.expand(edge_embeddings.shape[0], agent.actor.embedding_dim)
                    scorer_in = torch.cat([scorer_in, rep_graph], dim=1)

                # Apply mapping W on the edge-embedding slice only (leave graph part unchanged)
                if W is not None:
                    dT = int(scorer_in.shape[1])
                    if split_assume_half and (dT % 2 == 0):
                        d = dT // 2
                        edge_part = scorer_in[:, :d].cpu().numpy()
                        rest = scorer_in[:, d:]
                    else:
                        d = dT
                        edge_part = scorer_in[:, :d].cpu().numpy()
                        rest = None
                    if edge_part.shape[1] + 1 == W.shape[0]:
                        edge_aug = np.concatenate([edge_part, np.ones((edge_part.shape[0], 1), dtype=edge_part.dtype)], axis=1)
                        mapped_edge = edge_aug @ W
                        mapped_edge_t = torch.tensor(mapped_edge, dtype=torch.float32)
                        scorer_in = torch.cat([mapped_edge_t.to(scorer_in.device), rest], dim=1) if rest is not None else mapped_edge_t.to(scorer_in.device)

                # Compute logits and construct action_scores matrix
                logits = agent.actor.edge_scorer(scorer_in).flatten()
                num_tasks = int(obs.task_state_ready.shape[0])
                num_vms = int(obs.vm_completion_time.shape[0])
                action_scores = torch.ones((num_tasks, num_vms), dtype=torch.float32).to(logits.device) * -1e8
                # Fill compat edges with logits order
                t_idx = obs.compatibilities[0][:E].to(torch.long)
                v_idx = obs.compatibilities[1][:E].to(torch.long)
                action_scores[t_idx, v_idx] = logits
                # Apply readiness and scheduled masks
                action_scores[obs.task_state_ready == 0, :] = -1e8
                action_scores[obs.task_state_scheduled == 1, :] = -1e8
                # Greedy action (deterministic)
                flat_idx = torch.argmax(action_scores.view(-1)).item()
                action = int(flat_idx)

            next_obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                final_info = info
                break

        assert env.prev_obs is not None
        total_makespan += float(env.prev_obs.makespan())
        total_energy_obs += float(env.prev_obs.energy_consumption())
        if isinstance(final_info, dict):
            total_energy_full += float(final_info.get("total_energy", 0.0))
            total_active += float(final_info.get("total_energy_active", 0.0))
            total_idle += float(final_info.get("total_energy_idle", 0.0))
        env.close()

    n = max(1, eval_args.test_iterations)
    return (
        total_makespan / n,
        total_energy_obs / n,
        total_energy_full / n,
        {
            "avg_active_energy": (total_active / n) if total_active > 0 else None,
            "avg_idle_energy": (total_idle / n) if total_idle > 0 else None,
        },
    )


# ----------------------------
# Main orchestration
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Cross-p eval and OT mapping for heterogeneous GNN agent')
    # Training/eval p sets
    parser.add_argument('--ps-train', type=str, default='0.7,0.5,0.1', help='p values to train if missing (hetero)')
    parser.add_argument('--ps-all', type=str, default='0.8,0.7,0.5,0.3,0.1', help='p values to evaluate on')
    # Existing hetero checkpoints for 0.8 and 0.3
    parser.add_argument('--hetero-p08', type=str, default='', help='Path to heterogeneous checkpoint trained at p=0.8')
    parser.add_argument('--hetero-p03', type=str, default='', help='Path to heterogeneous checkpoint trained at p=0.3')
    # Train control
    parser.add_argument('--train-missing', action='store_true', help='Train hetero models for ps-train if checkpoints are missing')
    parser.add_argument('--total-timesteps', type=int, default=150000)
    parser.add_argument('--train-test-every', type=int, default=20, help='Test every N iterations during training')
    parser.add_argument('--eval-episodes', type=int, default=3, help='Episodes per evaluation (train-time tests and cross-p evals)')
    parser.add_argument('--seed', type=int, default=1)
    # Dataset params
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--min-tasks', type=int, default=12)
    parser.add_argument('--max-tasks', type=int, default=24)
    # Eval sampling
    parser.add_argument('--episodes', type=int, default=3, help='Episodes for embedding collection (OT)')
    parser.add_argument('--eval-seed-base', type=int, default=1_000_000_001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/cross_p'))

    # OT options
    parser.add_argument('--normalize', type=str, default='l2', choices=['none','l2','z'])
    parser.add_argument('--sinkhorn-reg', type=float, default=1e-2)
    parser.add_argument('--sinkhorn-iter', type=int, default=3000)
    parser.add_argument('--gw-epsilon', type=float, default=5e-3)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse p-lists
    try:
        ps_train = [float(x) for x in args.ps_train.split(',') if x.strip()]
        ps_all = [float(x) for x in args.ps_all.split(',') if x.strip()]
    except Exception:
        raise SystemExit("Failed to parse --ps-train or --ps-all")

    device = args.device

    # Map p -> checkpoint path for all train p values (include provided 0.8/0.3)
    ckpts: Dict[float, Path] = {}
    if args.hetero_p08:
        pth = resolve_best_checkpoint(args.hetero_p08)
        if pth is not None and pth.exists():
            ckpts[0.8] = pth
            print(f"[ckpt] Using p=0.8 checkpoint: {pth}")
        else:
            print(f"[warn] Could not resolve checkpoint from --hetero-p08: {args.hetero_p08}")
    if args.hetero_p03:
        pth = resolve_best_checkpoint(args.hetero_p03)
        if pth is not None and pth.exists():
            ckpts[0.3] = pth
            print(f"[ckpt] Using p=0.3 checkpoint: {pth}")
        else:
            print(f"[warn] Could not resolve checkpoint from --hetero-p03: {args.hetero_p03}")

    # Train or locate checkpoints for requested ps_train
    for p in ps_train:
        if p in ckpts and ckpts[p].exists():
            continue
        if not args.train_missing:
            print(f"[info] No checkpoint provided for p={p}, and --train-missing is not set; this p will be skipped.")
            continue
        print(f"[train] Training heterogeneous model for p={p} ...")
        ckpt = train_hetero_if_missing(p, device, args.total_timesteps, args.hosts, args.vms, args.workflows,
                                       args.min_tasks, args.max_tasks, args.train_test_every, args.eval_episodes, args.seed, args.out_dir)
        if ckpt is not None:
            ckpts[p] = ckpt
            print(f"[train] Trained checkpoint for p={p}: {ckpt}")

    # Load all agents we have checkpoints for
    agents: Dict[float, AblationGinAgent] = {}
    for p, path in ckpts.items():
        try:
            agents[p] = load_hetero_agent(path, device)
            print(f"[load] Loaded hetero agent p_train={p} from {path}")
        except Exception as e:
            print(f"[load] Failed to load agent for p={p} from {path}: {e}")

    if not agents:
        print("[abort] No agents available. Provide checkpoints and/or enable --train-missing.")
        return

    # ----------------------------
    # 1) Cross-p evaluation matrix
    # ----------------------------
    import csv as _csv
    cross_csv = args.out_dir / 'cross_p_results.csv'
    with cross_csv.open('w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['p_train','p_eval','makespan','total_energy','active_energy','idle_energy','active_plus_idle'])
        w.writeheader()
        # First, compute in-domain baselines for later drop calculations
        in_domain: Dict[float, Dict[str, float]] = {}
        for p_train, agent in agents.items():
            eval_args = build_eval_args(p_train, args.hosts, args.vms, args.workflows, args.min_tasks, args.max_tasks, device, args.eval_episodes, args.eval_seed_base)
            mk, eobs, etot, mm = evaluate_with_mapped_embeddings(agent, eval_args, W=None)
            total_e = etot if etot > 0 else eobs
            ae = mm.get('avg_active_energy') if mm else None
            ie = mm.get('avg_idle_energy') if mm else None
            api = (ae or 0.0) + (ie or 0.0)
            in_domain[p_train] = {
                'makespan': float(mk),
                'total_energy': float(total_e),
                'active_energy': float(ae or 0.0),
                'idle_energy': float(ie or 0.0),
                'active_plus_idle': float(api),
            }
            w.writerow({'p_train': p_train, 'p_eval': p_train, 'makespan': mk, 'total_energy': total_e, 'active_energy': ae or 0.0, 'idle_energy': ie or 0.0, 'active_plus_idle': api})
            print(f"[eval] In-domain p={p_train}: mk={mk:.4g} totalE={total_e:.4g} api={api:.4g}")

        # Cross-evals
        for p_train, agent in agents.items():
            for p_eval in ps_all:
                if p_eval == p_train:
                    continue
                eval_args = build_eval_args(p_eval, args.hosts, args.vms, args.workflows, args.min_tasks, args.max_tasks, device, args.eval_episodes, args.eval_seed_base)
                mk, eobs, etot, mm = evaluate_with_mapped_embeddings(agent, eval_args, W=None)
                total_e = etot if etot > 0 else eobs
                ae = mm.get('avg_active_energy') if mm else 0.0
                ie = mm.get('avg_idle_energy') if mm else 0.0
                api = float(ae) + float(ie)
                w.writerow({'p_train': p_train, 'p_eval': p_eval, 'makespan': mk, 'total_energy': total_e, 'active_energy': ae, 'idle_energy': ie, 'active_plus_idle': api})
                print(f"[eval] p_train={p_train} -> p_eval={p_eval}: mk={mk:.4g} totalE={total_e:.4g} api={api:.4g}")

    # Relative drops vs in-domain
    try:
        import pandas as pd
        df = pd.read_csv(cross_csv)
        rows = []
        for p_train, base in in_domain.items():
            sub = df[df['p_train'] == p_train]
            for _, r in sub.iterrows():
                p_eval = float(r['p_eval'])
                def rel_drop(metric: str):
                    b = base[metric]
                    v = float(r[metric])
                    if b == 0:
                        return np.nan
                    return (v - b) / b
                rows.append({
                    'p_train': p_train,
                    'p_eval': p_eval,
                    'rel_drop_makespan': rel_drop('makespan'),
                    'rel_drop_total_energy': rel_drop('total_energy'),
                    'rel_drop_active_plus_idle': rel_drop('active_plus_idle'),
                })
        drop_csv = args.out_dir / 'cross_p_relative_drop.csv'
        pd.DataFrame(rows).to_csv(drop_csv, index=False)
        print(f"[eval] Wrote: {cross_csv} and {drop_csv}")
    except Exception as e:
        print(f"[eval] Could not compute relative drops: {e}")

    # ----------------------------
    # 2) OT mapping experiments
    # ----------------------------
    ot_csv = args.out_dir / 'ot_mapping_summary.csv'
    with ot_csv.open('w', newline='') as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=[
            'p_source','p_target','sinkhorn','gw','disp_mean','disp_median','n_ref','n_eval',
            'logit_corr_mean','top1_agree','top5_agree',
            'tgt_mk','tgt_api','src_mk_unmapped','src_api_unmapped','src_mk_mapped','src_api_mapped',
            'delta_mk_unmapped','delta_mk_mapped','delta_api_unmapped','delta_api_mapped'
        ])
        w.writeheader()

        for p_t in ps_all:
            # Need target model at p_t for reference; skip if not present
            if p_t not in agents:
                print(f"[ot] Skipping p_t={p_t}: no target model trained at this p")
                continue
            target_agent = agents[p_t]
            eval_args_t = build_eval_args(p_t, args.hosts, args.vms, args.workflows, args.min_tasks, args.max_tasks, device, args.eval_episodes, args.eval_seed_base)
            # Target in-domain performance (once per p_t)
            tgt_mk, tgt_eobs, tgt_etot, tgt_mm = evaluate_with_mapped_embeddings(target_agent, eval_args_t, W=None)
            tgt_ae = tgt_mm.get('avg_active_energy') if tgt_mm else 0.0
            tgt_ie = tgt_mm.get('avg_idle_energy') if tgt_mm else 0.0
            tgt_api = float(tgt_ae) + float(tgt_ie)
            # Reference embeddings X at p_t from target model
            X_ref = collect_edge_embeddings(target_agent, p_t, args.hosts, args.vms, args.workflows,
                                            args.min_tasks, args.max_tasks, device, args.episodes, args.eval_seed_base)
            if args.normalize != 'none':
                X_ref = normalize(X_ref, args.normalize)

            # Also collect scorer I/O for state-level comparison later
            T_Xin_list, T_logits_list = collect_scorer_inputs_and_logits(target_agent, p_t, args.hosts, args.vms, args.workflows,
                                                                         args.min_tasks, args.max_tasks, device, args.episodes, args.eval_seed_base)

            for p_s, src_agent in agents.items():
                # Evaluate source model on p_t and collect embeddings Y
                X_eval = collect_edge_embeddings(src_agent, p_t, args.hosts, args.vms, args.workflows,
                                                 args.min_tasks, args.max_tasks, device, args.episodes, args.eval_seed_base)
                if args.normalize != 'none':
                    X_eval = normalize(X_eval, args.normalize)

                # Distances and coupling
                sink = sinkhorn_distance(X_ref, X_eval, reg=args.sinkhorn_reg, num_iter=args.sinkhorn_iter)
                gw, Pi = gw_distance(X_ref, X_eval, metric='cosine', epsilon=args.gw_epsilon)
                disp_mean = float('nan'); disp_median = float('nan')
                if Pi is not None and Pi.size > 0:
                    XbY = barycentric_project_Y_to_X(X_ref, X_eval, Pi)
                    d = np.linalg.norm(X_eval - XbY, axis=1)
                    if d.size:
                        disp_mean = float(np.mean(d)); disp_median = float(np.median(d))

                # Now, attempt behavior recovery via mapping on the sampled states
                # Strategy: for each of the sampled episodes at p_t, we compute source scorer inputs,
                # map only the edge-embedding slice using a linear Procrustes from (X_eval -> X_ref)
                # as a simple distribution alignment (since Pi is global and not per-state). This is a
                # practical approximation to transport without changing the network.
                logit_corrs: List[float] = []
                top1_agree = 0; top5_agree = 0; total_states = 0

                # Learn a linear map via least-squares from X_eval to X_ref using Pi barycentric projections if available
                W = None
                try:
                    if Pi is not None and Pi.size > 0:
                        XbY = barycentric_project_Y_to_X(X_ref, X_eval, Pi)  # target for each Y
                        # Solve min_W || Y @ W - XbY ||_F via least squares
                        # Add bias by augmenting Y with ones
                        Y_aug = np.concatenate([X_eval, np.ones((X_eval.shape[0], 1), dtype=X_eval.dtype)], axis=1)
                        W_ls, *_ = np.linalg.lstsq(Y_aug, XbY, rcond=None)
                        W = W_ls  # shape (d+1, d)
                except Exception:
                    W = None

                # Collect source scorer inputs/logits per state
                S_Xin_list, S_logits_list = collect_scorer_inputs_and_logits(src_agent, p_t, args.hosts, args.vms, args.workflows,
                                                                             args.min_tasks, args.max_tasks, device, args.episodes, args.eval_seed_base)
                # For each paired state index, compare target logits vs source logits with mapped inputs
                for i in range(min(len(T_Xin_list), len(S_Xin_list))):
                    T_in = T_Xin_list[i]
                    S_in = S_Xin_list[i]
                    # They may have different input dims if global graph embedding is concatenated. Both hetero variants
                    # use actor global embedding by default in ablation unless disabled; we will align on edge-embedding dims.
                    # Assume the concatenation pattern: [edge_emb (d) || optional graph_emb (d)]. We will map only the first d dims.
                    dT = T_in.shape[1]
                    dS = S_in.shape[1]
                    # Heuristic: if dims equal, proceed; else, try to split halves
                    if dT != dS:
                        d = min(dT, dS) // 2  # assume edge d then graph d
                        if d > 0:
                            T_edge = T_in[:, :d]
                            S_edge = S_in[:, :d]
                            S_rest = S_in[:, d:]
                        else:
                            # Fallback: match min dims
                            d = min(dT, dS)
                            T_edge = T_in[:, :d]
                            S_edge = S_in[:, :d]
                            S_rest = None
                    else:
                        # Split in half if even
                        if dT % 2 == 0:
                            d = dT // 2
                            T_edge = T_in[:, :d]
                            S_edge = S_in[:, :d]
                            S_rest = S_in[:, d:]
                        else:
                            d = dT
                            T_edge = T_in[:, :d]
                            S_edge = S_in[:, :d]
                            S_rest = None

                    # Apply linear map W (if available) to S_edge
                    if W is not None and S_edge.shape[1] + 1 == W.shape[0]:
                        S_edge_aug = np.concatenate([S_edge, np.ones((S_edge.shape[0], 1), dtype=S_edge.dtype)], axis=1)
                        S_edge_mapped = S_edge_aug @ W  # (E,d)
                    else:
                        S_edge_mapped = S_edge  # no mapping available

                    # Rebuild scorer input for source scorer
                    if S_rest is not None and S_rest.shape[1] > 0:
                        S_in_mapped = np.concatenate([S_edge_mapped, S_rest], axis=1)
                    else:
                        S_in_mapped = S_edge_mapped

                    # Compute logits through source edge_scorer on mapped inputs
                    with torch.no_grad():
                        logits_src_mapped = src_agent.actor.edge_scorer(torch.tensor(S_in_mapped, dtype=torch.float32)).flatten().cpu().numpy()
                    logits_tgt = T_logits_list[i]

                    # Align lengths (edge counts may differ slightly across agents if masks differ); match min length
                    n = min(logits_src_mapped.shape[0], logits_tgt.shape[0])
                    if n == 0:
                        continue
                    a = logits_src_mapped[:n]
                    b = logits_tgt[:n]
                    # Pearson correlation
                    try:
                        corr = float(np.corrcoef(a, b)[0, 1])
                        if np.isfinite(corr):
                            logit_corrs.append(corr)
                    except Exception:
                        pass
                    # Top-k agreement metrics
                    top1_src = int(np.argmax(a))
                    top1_tgt = int(np.argmax(b))
                    if top1_src == top1_tgt:
                        top1_agree += 1
                    k = min(5, n)
                    topk_src = set(list(np.argsort(a)[-k:]))
                    topk_tgt = set(list(np.argsort(b)[-k:]))
                    if len(topk_src.intersection(topk_tgt)) > 0:
                        top5_agree += 1
                    total_states += 1

                # Evaluate source agent performance on p_t (unmapped and mapped)
                src_mk_unmapped, src_eobs_u, src_etot_u, src_mm_u = evaluate_with_mapped_embeddings(src_agent, eval_args_t, W=None)
                src_ae_u = src_mm_u.get('avg_active_energy') if src_mm_u else 0.0
                src_ie_u = src_mm_u.get('avg_idle_energy') if src_mm_u else 0.0
                src_api_u = float(src_ae_u) + float(src_ie_u)

                src_mk_mapped = float('nan'); src_api_mapped = float('nan')
                try:
                    mk_m, eobs_m, etot_m, mm_m = evaluate_with_mapped_embeddings(src_agent, eval_args_t, W)
                    src_mk_mapped = float(mk_m)
                    ae_m = mm_m.get('avg_active_energy') if mm_m else 0.0
                    ie_m = mm_m.get('avg_idle_energy') if mm_m else 0.0
                    src_api_mapped = float(ae_m) + float(ie_m)
                except Exception as e:
                    print(f"[ot] Warning: mapped-eval failed for p_s={p_s} -> p_t={p_t}: {e}")

                rec = {
                    'p_source': p_s,
                    'p_target': p_t,
                    'sinkhorn': sink,
                    'gw': gw,
                    'disp_mean': disp_mean,
                    'disp_median': disp_median,
                    'n_ref': X_ref.shape[0],
                    'n_eval': X_eval.shape[0],
                    'logit_corr_mean': (float(np.mean(logit_corrs)) if logit_corrs else float('nan')),
                    'top1_agree': (top1_agree / max(1, total_states)),
                    'top5_agree': (top5_agree / max(1, total_states)),
                    'tgt_mk': float(tgt_mk),
                    'tgt_api': float(tgt_api),
                    'src_mk_unmapped': float(src_mk_unmapped),
                    'src_api_unmapped': float(src_api_u),
                    'src_mk_mapped': float(src_mk_mapped),
                    'src_api_mapped': float(src_api_mapped),
                    'delta_mk_unmapped': float(src_mk_unmapped - tgt_mk),
                    'delta_mk_mapped': float(src_mk_mapped - tgt_mk) if np.isfinite(src_mk_mapped) else float('nan'),
                    'delta_api_unmapped': float(src_api_u - tgt_api),
                    'delta_api_mapped': float(src_api_mapped - tgt_api) if np.isfinite(src_api_mapped) else float('nan'),
                }
                w.writerow(rec)
                print(f"[ot] p_s={p_s} -> p_t={p_t} | sink={sink:.5g} gw={gw:.5g} disp_mean={disp_mean:.5g} | corr={rec['logit_corr_mean']:.3f} | top1={rec['top1_agree']:.2%} top5={rec['top5_agree']:.2%} | mk_uΔ={rec['delta_mk_unmapped']:.4g} mk_mΔ={rec['delta_mk_mapped']:.4g} api_uΔ={rec['delta_api_unmapped']:.4g} api_mΔ={rec['delta_api_mapped']:.4g}")

    print(f"[done] Cross-p and OT experiments complete. Results in {args.out_dir}")


if __name__ == '__main__':
    main()
