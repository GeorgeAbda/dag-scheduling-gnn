import argparse
import math
import os
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# Ensure project root is importable when running as a script
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# Internal imports from the repo
from scheduler.rl_model.ablation_gnn_traj_main import Args as TrainArgs
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


# -----------------------------
# Pareto utilities
# -----------------------------

def is_dominated(p: Tuple[float, float], q: Tuple[float, float]) -> bool:
    """Return True if q dominates p (minimization)."""
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])


def non_dominated(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = list(points)
    keep = [True] * len(pts)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(len(pts)):
            if i == j or not keep[j]:
                continue
            if is_dominated(pts[i], pts[j]):
                keep[i] = False
                break
    return [pts[i] for i, k in enumerate(keep) if k]


def hv2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """2D hypervolume (minimization) w.r.t. upper-right reference ref."""
    if not points:
        return 0.0
    fr = non_dominated(points)
    # clip to reference box and sort
    fr = [(min(x, ref[0]), min(y, ref[1])) for (x, y) in fr]
    fr = [(x, y) for (x, y) in fr if x <= ref[0] and y <= ref[1]]
    if not fr:
        return 0.0
    fr.sort(key=lambda t: (t[0], t[1]))
    # enforce strictly decreasing y
    chain = []
    best_y = math.inf
    for x, y in fr:
        if y < best_y:
            chain.append((x, y))
            best_y = y
    # accumulate rectangles
    hv = 0.0
    y_prev = ref[1]
    for x, y in chain:
        dx = max(ref[0] - x, 0.0)
        dy = max(y_prev - y, 0.0)
        hv += dx * dy
        y_prev = y
    return hv


def coverage(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> float:
    """Fraction of B dominated by the Pareto front of A (minimization)."""
    if not B:
        return 0.0
    A_front = non_dominated(A)
    cnt = 0
    for p in B:
        if any(is_dominated(p, q) for q in A_front):
            cnt += 1
    return cnt / len(B)


def igd(candidate: List[Tuple[float, float]], reference_front: List[Tuple[float, float]]) -> float:
    """Inverted Generational Distance (IGD): average distance from ref points to nearest candidate point."""
    if not reference_front:
        return float('inf')
    if not candidate:
        return float('inf')
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    dists = []
    for r in ref:
        diffs = cand - r[None, :]
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))


def igd_plus(candidate: List[Tuple[float, float]], reference_front: List[Tuple[float, float]]) -> float:
    """IGD+ (minimization): for each ref point z, distance uses only under-performance components.
    d_+(s,z) = || max(s - z, 0) ||_2, IGD+ = (1/|Z|) sum_z min_s d_+(s,z).
    """
    if not reference_front:
        return float('inf')
    if not candidate:
        return float('inf')
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    vals = []
    for r in ref:
        diffs = cand - r[None, :]
        diffs = np.maximum(diffs, 0.0)  # only penalize under-performance (s worse than z)
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        vals.append(float(np.min(ds)))
    return float(np.mean(vals))


def schott_spacing(points: List[Tuple[float, float]], norm_min: Tuple[float, float], norm_max: Tuple[float, float]) -> float:
    """Schott spacing on normalized points (lower is more uniform).
    d_i = min_{j != i} sum_k |f_k(i) - f_k(j)|; spacing = sqrt( (1/(n-1)) sum (d_i - d_bar)^2 ).
    """
    if not points or len(points) <= 1:
        return 0.0
    pts = np.asarray(points, dtype=float)
    mins = np.asarray(norm_min, dtype=float)
    maxs = np.asarray(norm_max, dtype=float)
    denom = np.maximum(maxs - mins, 1e-12)
    pn = (pts - mins) / denom
    n = pn.shape[0]
    # compute pairwise L1 distances
    dists = np.sum(np.abs(pn[:, None, :] - pn[None, :, :]), axis=2)
    # for each i, take min over j != i
    d_i = []
    for i in range(n):
        row = np.delete(dists[i], i)
        d_i.append(float(np.min(row)))
    d_i = np.asarray(d_i, dtype=float)
    d_bar = float(np.mean(d_i))
    return float(np.sqrt(np.sum((d_i - d_bar) ** 2) / max(n - 1, 1)))


def sample_dirichlet_weights(n: int) -> np.ndarray:
    """Sample n weight vectors on the 2D simplex for R2 indicator."""
    w = np.random.dirichlet(alpha=[1.0, 1.0], size=int(max(1, n)))
    return w.astype(float)


def r2_indicator(points: List[Tuple[float, float]], weights: np.ndarray, norm_min: Tuple[float, float], norm_max: Tuple[float, float]) -> float:
    """Compute R2 indicator (lower is better) over normalized points.
    Normalize each objective to [0,1] using (norm_min, norm_max) from ALL points.
    """
    if not points:
        return float('inf')
    pts = np.asarray(points, dtype=float)
    mins = np.asarray(norm_min, dtype=float)
    maxs = np.asarray(norm_max, dtype=float)
    denom = np.maximum(maxs - mins, 1e-12)
    pts_n = (pts - mins) / denom
    # For each weight vector w, compute min_x w·x
    vals = []
    for w in weights:
        proj = pts_n @ w.reshape(2, 1)
        vals.append(float(np.min(proj)))
    return float(np.mean(vals))


# -----------------------------
# Arch and model loading
# -----------------------------

KNOWN_VARIANTS = {
    "baseline": AblationVariant(name="baseline"),
    "mlp_only": AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False),
    "hetero": AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True),
}


def infer_variant_from_filename(path: Path) -> AblationVariant:
    """Infer variant by prefix of filename up to first underscore."""
    name = path.name
    base = name.split("_")[0].strip().lower()
    if base in KNOWN_VARIANTS:
        return KNOWN_VARIANTS[base]
    # fallback
    return AblationVariant(name=base)


def load_agent(ckpt_path: Path, device: torch.device, eval_args: TrainArgs) -> Tuple[AblationGinAgent, AblationVariant]:
    variant = infer_variant_from_filename(ckpt_path)
    agent = AblationGinAgent(device, variant)
    # Initialize any lazy params (e.g., hetero SAGE with (-1,-1)) via one dummy forward
    try:
        _env_init = CloudSchedulingGymEnvironment(
            dataset_args=eval_args.dataset,
            collect_timelines=False,
            compute_metrics=False,
            profile=False,
        )
        _env_init = GinAgentWrapper(_env_init, constrained_mode=False)
        _obs0, _ = _env_init.reset(seed=int(123))
        with torch.no_grad():
            x0 = torch.as_tensor(_obs0, dtype=torch.float32, device=agent.device).unsqueeze(0)
            agent.get_action_and_value(x0)
    except Exception:
        pass
    state = torch.load(str(ckpt_path), map_location=device)
    # Support pareto checkpoint format saved as {"state_dict": ..., "metrics": ...}
    if isinstance(state, dict) and 'state_dict' in state:
        try:
            state = state['state_dict']
        except Exception:
            pass
    # Load only intersecting parameters with matching shapes to avoid size-mismatch errors
    try:
        curr = agent.state_dict()
        compat = {}
        for k, v in state.items():
            if k in curr:
                try:
                    if curr[k].shape == v.shape:
                        compat[k] = v
                except Exception:
                    # Skip keys whose current param is uninitialized or shape-unavailable
                    continue
        agent.load_state_dict(compat, strict=False)
    except Exception:
        # Fallback: try best-effort non-strict load
        try:
            agent.load_state_dict(state, strict=False)
        except Exception:
            pass
    agent.eval()
    return agent, variant


# -----------------------------
# Episode evaluation
# -----------------------------

def build_eval_args(base: TrainArgs, args_cli) -> TrainArgs:
    new = dc_replace(base)
    ds = dc_replace(base.dataset)
    # Apply CLI overrides commonly used in training eval
    if args_cli.dag_method:
        ds = dc_replace(ds, dag_method=args_cli.dag_method)
    if args_cli.gnp_min_n is not None:
        ds = dc_replace(ds, gnp_min_n=int(args_cli.gnp_min_n))
    if args_cli.gnp_max_n is not None:
        ds = dc_replace(ds, gnp_max_n=int(args_cli.gnp_max_n))
    # Optional p override (GNP probability)
    if getattr(args_cli, 'gnp_p', None) is not None:
        try:
            ds = dc_replace(ds, gnp_p=float(args_cli.gnp_p))
        except Exception:
            pass
    if args_cli.host_count is not None:
        ds = dc_replace(ds, host_count=int(args_cli.host_count))
    if args_cli.vm_count is not None:
        ds = dc_replace(ds, vm_count=int(args_cli.vm_count))
    if args_cli.workflow_count is not None:
        ds = dc_replace(ds, workflow_count=int(args_cli.workflow_count))
    # Optional size scaling for evaluation (applied after explicit min/max overrides)
    if getattr(args_cli, 'size_scale', None) is not None:
        try:
            sc = float(args_cli.size_scale)
            if sc > 0:
                new_min = int(round(max(1, ds.gnp_min_n * sc)))
                new_max = int(round(max(new_min, ds.gnp_max_n * sc)))
                ds = dc_replace(ds, gnp_min_n=new_min, gnp_max_n=new_max)
        except Exception:
            pass
    new.dataset = ds
    new.device = str(args_cli.device)
    new.test_iterations = int(args_cli.episodes)
    return new


def eval_checkpoint_points(ckpt_path: Path, eval_args: TrainArgs, seed_base: int) -> List[Tuple[float, float]]:
    device = torch.device(eval_args.device if eval_args.device else "cpu")
    agent, _ = load_agent(ckpt_path, device, eval_args)

    # Build env like in eval script
    env = CloudSchedulingGymEnvironment(
        dataset_args=eval_args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    env = GinAgentWrapper(env, constrained_mode=False)

    points: List[Tuple[float, float]] = []
    for epi in tqdm(range(eval_args.test_iterations), desc=f"episodes_{ckpt_path.stem}"):
        obs, _ = env.reset(seed=seed_base + epi)
        # Baselines
        baseline_mk = float(env.prev_obs.makespan()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
        baseline_active = float(getattr(env.prev_obs, 'energy_consumption', lambda: float('nan'))()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')

        done = False
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a, _lp, _ent, _val = agent.get_action_and_value(x)
                action = int(a[0].item())
            obs, reward, done, trunc, info = env.step(action)
            done = bool(done or trunc)
        # Final metrics
        mk = float(info.get('makespan', env.prev_obs.makespan()))
        ae = float(info.get('total_energy_active', getattr(env.prev_obs, 'energy_consumption', lambda: mk)()))
        # Ratios for comparability across configs
        if np.isfinite(baseline_mk) and baseline_mk > 0 and np.isfinite(baseline_active) and baseline_active > 0:
            x = mk / baseline_mk
            y = ae / baseline_active
        else:
            x, y = mk, ae
        points.append((x, y))
    return points


def _safe_env_stats(env: GinAgentWrapper) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        vm_obs = getattr(env.prev_obs, 'vm_observations', None)
        if vm_obs:
            speeds = []
            mems = []
            for v in vm_obs:
                for attr in ('cpu_speed_mips', 'cpu_speed', 'speed', 'mips'):
                    if hasattr(v, attr):
                        try:
                            speeds.append(float(getattr(v, attr)))
                            break
                        except Exception:
                            pass
                for attr in ('memory_mb', 'mem_mb', 'memory'):
                    if hasattr(v, attr):
                        try:
                            mems.append(float(getattr(v, attr)))
                            break
                        except Exception:
                            pass
            if speeds:
                stats.update({
                    'vm_speed_min': float(np.min(speeds)),
                    'vm_speed_max': float(np.max(speeds)),
                    'vm_speed_mean': float(np.mean(speeds)),
                })
            if mems:
                stats.update({
                    'vm_mem_min': float(np.min(mems)),
                    'vm_mem_max': float(np.max(mems)),
                    'vm_mem_mean': float(np.mean(mems)),
                })
            stats['vm_count_obs'] = int(len(vm_obs))
    except Exception:
        pass
    try:
        task_obs = getattr(env.prev_obs, 'task_observations', None)
        if task_obs:
            lens = []
            for t in task_obs:
                if hasattr(t, 'length'):
                    try:
                        lens.append(float(getattr(t, 'length')))
                    except Exception:
                        pass
            if lens:
                stats.update({
                    'task_len_min': float(np.min(lens)),
                    'task_len_max': float(np.max(lens)),
                    'task_len_mean': float(np.mean(lens)),
                })
            stats['task_count_obs'] = int(len(task_obs))
    except Exception:
        pass
    # DAG structure statistics (best-effort)
    try:
        deps = getattr(env.prev_obs, 'task_dependencies', None)
        n_tasks = int(stats.get('task_count_obs') or 0)
        if deps is not None and hasattr(deps, 'numel') and int(deps.numel()) > 0 and n_tasks > 0:
            # deps expected shape [2, E] with edges u->v between task indices
            try:
                src = deps[0].detach().cpu().numpy().astype(int).tolist()
                dst = deps[1].detach().cpu().numpy().astype(int).tolist()
            except Exception:
                src = []
                dst = []
            E = int(len(src))
            stats['dag_edges'] = float(E)
            denom = float(max(1, n_tasks * (n_tasks - 1)))
            stats['dag_density'] = float(E) / denom
            # degrees
            indeg = [0] * n_tasks
            outdeg = [0] * n_tasks
            for u, v in zip(src, dst):
                if 0 <= u < n_tasks:
                    outdeg[u] += 1
                if 0 <= v < n_tasks:
                    indeg[v] += 1
            stats['avg_in_degree'] = float(np.mean(indeg)) if n_tasks > 0 else 0.0
            stats['avg_out_degree'] = float(np.mean(outdeg)) if n_tasks > 0 else 0.0
            # critical path length and level widths via topological DP
            try:
                # build adjacency and indegree
                adj: List[List[int]] = [[] for _ in range(n_tasks)]
                indeg_arr = [0] * n_tasks
                for u, v in zip(src, dst):
                    if 0 <= u < n_tasks and 0 <= v < n_tasks:
                        adj[u].append(v)
                        indeg_arr[v] += 1
                # Kahn's algorithm to obtain levels
                from collections import deque
                q = deque([i for i in range(n_tasks) if indeg_arr[i] == 0])
                level = [0] * n_tasks
                order = []
                while q:
                    u = q.popleft()
                    order.append(u)
                    for v in adj[u]:
                        # longest path to v via u
                        if level[v] < level[u] + 1:
                            level[v] = level[u] + 1
                        indeg_arr[v] -= 1
                        if indeg_arr[v] == 0:
                            q.append(v)
                # If cycle exists, order may be incomplete; guard
                if order:
                    cpl = max(level) + 1  # nodes count along longest chain
                    stats['critical_path_len'] = float(cpl)
                    # layer widths
                    from collections import Counter
                    cnt = Counter(level)
                    widths = [cnt.get(l, 0) for l in range(max(level) + 1)]
                    if widths:
                        stats['avg_layer_width'] = float(np.mean(widths))
                        stats['max_layer_width'] = float(np.max(widths))
                else:
                    stats['critical_path_len'] = float('nan')
            except Exception:
                pass
        else:
            # No deps observed; set safe defaults
            stats.setdefault('dag_edges', 0.0)
            if n_tasks:
                stats.setdefault('dag_density', 0.0)
                stats.setdefault('avg_in_degree', 0.0)
                stats.setdefault('avg_out_degree', 0.0)
    except Exception:
        pass
    return stats


def eval_checkpoint_points_with_epi(ckpt_path: Path, eval_args: TrainArgs, seed_base: int) -> List[Tuple[int, float, float, float, float, Dict[str, Any]]]:
    """Evaluate a checkpoint and return per-episode points including episode index and raw metrics.
    Returns a list of (epi, mk_ratio, ae_ratio, mk_raw, ae_raw, env_stats).
    """
    device = torch.device(eval_args.device if eval_args.device else "cpu")
    agent, _ = load_agent(ckpt_path, device, eval_args)

    env = CloudSchedulingGymEnvironment(
        dataset_args=eval_args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    env = GinAgentWrapper(env, constrained_mode=False)

    out: List[Tuple[int, float, float, float, float, Dict[str, Any]]] = []
    for epi in tqdm(range(eval_args.test_iterations), desc=f"episodes_{ckpt_path.stem}"):
        obs, _ = env.reset(seed=seed_base + epi)
        baseline_mk = float(env.prev_obs.makespan()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
        baseline_active = float(getattr(env.prev_obs, 'energy_consumption', lambda: float('nan'))()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
        done = False
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a, _lp, _ent, _val = agent.get_action_and_value(x)
                action = int(a[0].item())
            obs, reward, done, trunc, info = env.step(action)
            done = bool(done or trunc)
        mk = float(info.get('makespan', env.prev_obs.makespan()))
        ae = float(info.get('total_energy_active', getattr(env.prev_obs, 'energy_consumption', lambda: mk)()))
        if np.isfinite(baseline_mk) and baseline_mk > 0 and np.isfinite(baseline_active) and baseline_active > 0:
            x = mk / baseline_mk
            y = ae / baseline_active
        else:
            x, y = mk, ae
        stats = _safe_env_stats(env)
        out.append((epi, x, y, mk, ae, stats))
    return out


# -----------------------------
# Step 1: Parse PF files and recompute ND set; optional deletion
# -----------------------------

def parse_pf_files(per_variant_dir: Path) -> List[Tuple[float, float, Path]]:
    """Parse traditional *_pf_iter*_mk*_ae*.pt files in this directory (non-recursive)."""
    files = sorted(per_variant_dir.glob("*_pf_iter*_mk*_ae*.pt"))
    out: List[Tuple[float, float, Path]] = []
    for fp in files:
        name = fp.name
        mk_idx = name.find("_mk")
        ae_idx = name.find("_ae")
        dot = name.rfind(".pt")
        if mk_idx >= 0 and ae_idx > mk_idx and dot > ae_idx:
            mk_tag = name[mk_idx + 3:ae_idx]
            ae_tag = name[ae_idx + 3:dot]
            try:
                mk_v = float(mk_tag.replace('p', '.'))
                ae_v = float(ae_tag.replace('p', '.'))
                out.append((mk_v, ae_v, fp))
            except Exception:
                continue
    return out


def parse_pareto_ckpts_recursive(root_dir: Path) -> List[Tuple[float, float, Path, str]]:
    """Recursively find *_pareto_*.pt files and extract (mk, ae, path, arch).
    Expects checkpoint saved as {"state_dict": ..., "metrics": {"makespan": float, "active_energy": float, ...}}
    Arch is inferred from the filename prefix before "_pareto_" or from the immediate parent directory name.
    """
    out: List[Tuple[float, float, Path, str]] = []
    for fp in sorted(root_dir.rglob("*_pareto_*.pt")):
        arch_name = None
        try:
            stem = fp.stem
            if "_pareto_" in stem:
                arch_name = stem.split("_pareto_", 1)[0]
        except Exception:
            arch_name = None
        if not arch_name:
            arch_name = fp.parent.name
        mk_v = float('nan')
        ae_v = float('nan')
        try:
            obj = torch.load(str(fp), map_location='cpu')
            if isinstance(obj, dict):
                m = obj.get('metrics', {})
                mk_v = float(m.get('makespan', float('nan')))
                ae_v = float(m.get('active_energy', float('nan')))
        except Exception:
            pass
        out.append((mk_v, ae_v, fp, str(arch_name)))
    return out


def filter_pf_nondominated(pfs: List[Tuple[float, float, Path]], delete_dominated: bool = False) -> List[Tuple[float, float, Path]]:
    if not pfs:
        return []
    pts = [(mk, ae) for (mk, ae, _p) in pfs]
    nd = non_dominated(pts)
    nd_set = set(nd)
    kept: List[Tuple[float, float, Path]] = []
    for mk, ae, p in pfs:
        if (mk, ae) in nd_set:
            kept.append((mk, ae, p))
        elif delete_dominated:
            try:
                p.unlink()
            except Exception:
                pass
    return kept


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare architectures via PF checkpoints and compute HV/IGD/Coverage vs reference front.")
    ap.add_argument("--dirs", nargs="+", required=True, help="Per-variant directories containing *_pf_*.pt files")
    ap.add_argument("--delete-dominated", action="store_true", help="Delete dominated PF files on disk (per directory)")
    ap.add_argument("--episodes", type=int, default=8, help="Episodes per checkpoint evaluation")
    ap.add_argument("--seed-base", type=int, default=12345, help="Base seed for evaluation episodes")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="logs/pareto_compare")
    ap.add_argument("--ref-front-csv", type=str, default=None, help="Path to NSGA-II reference_front.csv (mk_ratio,ae_ratio). If set, use this as the reference front.")
    # Dataset overrides to match training eval configs
    ap.add_argument("--dag-method", type=str, default=None)
    ap.add_argument("--gnp-min-n", type=int, default=None)
    ap.add_argument("--gnp-max-n", type=int, default=None)
    ap.add_argument("--host-count", type=int, default=None)
    ap.add_argument("--vm-count", type=int, default=None)
    ap.add_argument("--workflow-count", type=int, default=None)
    ap.add_argument("--front-scope", type=str, choices=["arch_all", "per_ckpt", "per_seed"], default="per_seed",
                    help="arch_all: one front per architecture across all checkpoints and episodes (default). per_ckpt: emit a front per checkpoint over its episodes. per_seed: emit a front per episode (seed) across all checkpoints of the same architecture.")
    ap.add_argument("--plot-per-seed", action="store_true", default=True, help="If set, generate a Pareto plot per episode seed overlaying the three architectures.")
    ap.add_argument("--gnp-p", type=float, default=None, help="Override GNP edge probability at evaluation time")
    ap.add_argument("--size-scale", type=float, default=None, help="Scale gnp_min_n/gnp_max_n by this factor at evaluation time")
    ap.add_argument("--r2-weights", type=int, default=64, help="Number of weight vectors to estimate R2 indicator (lower is better)")
    ap.add_argument("--plot-ref-front", action="store_true", help="Overlay NSGA reference front on per-seed plots (requires --ref-front-csv)")
    ap.add_argument("--ref-pop-csv", type=str, default=None, help="Optional NSGA history CSV (generation,idx,mk_ratio,ae_ratio) to scatter the population on per-seed plots")
    ap.add_argument("--pub-style", action="store_true", help="Apply publication-quality styling to plots (fonts, colors, grids)")
    ap.add_argument("--style-ga", action="store_true", help="Use a GA-style aesthetic: light gray background, white grid, small dots, colored fronts (NSGA front red)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: publication style
    if args.style_ga:
        mpl.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.facecolor": "#f2f2f2",
            "figure.facecolor": "white",
            "grid.color": "white",
            "grid.linestyle": "-",
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "axes.grid": True,
            "axes.grid.which": "both",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        })

    # Step 1: discover checkpoints per directory
    # Supports two formats:
    #  (A) Traditional PF files: *_pf_iter*_mk*_ae*.pt in the provided dir (non-recursive)
    #  (B) Pareto archive files: *_pareto_*.pt possibly under nested per_variant/<arch>/ subdirs (recursive)
    arch_ckpts: Dict[str, List[Path]] = {}
    for d in tqdm(args.dirs, desc="scan_dirs"):
        p = Path(d)
        if not p.exists():
            print(f"[warn] directory not found: {p}")
            continue
        # Try PF files first (non-recursive)
        pf_list = parse_pf_files(p)
        if pf_list:
            arch = p.parent.parent.name.split("_")[-1] if p.parent.parent.name else p.name
            kept = filter_pf_nondominated(pf_list, delete_dominated=args.delete_dominated)
            arch_ckpts.setdefault(arch, []).extend([fp for (_mk, _ae, fp) in kept])
            print(f"[step1] {arch}: kept {len(kept)} / {len(pf_list)} PF checkpoints")
            continue
        # Else look for pareto_* checkpoints recursively
        pareto_found = parse_pareto_ckpts_recursive(p)
        if pareto_found:
            # Group by inferred arch name
            from collections import defaultdict as _dd
            grp: Dict[str, List[Path]] = _dd(list)
            cnt_by_arch: Dict[str, int] = {}
            for mk, ae, fp, arch_name in pareto_found:
                grp[arch_name].append(fp)
                cnt_by_arch[arch_name] = cnt_by_arch.get(arch_name, 0) + 1
            for arch_name, fps in grp.items():
                arch_ckpts.setdefault(arch_name, []).extend(sorted(fps))
            pretty = ", ".join([f"{a}:{cnt_by_arch[a]}" for a in sorted(cnt_by_arch.keys())])
            print(f"[step1] pareto: discovered {sum(cnt_by_arch.values())} ckpts [{pretty}] under {p}")
        else:
            print(f"[step1] no PF or pareto checkpoints found under {p}")

    # Build eval args
    base = TrainArgs()
    eval_args = build_eval_args(base, args)

    # Step 2: Evaluate checkpoints and collect points
    # If --ref-front-csv is provided, load it as the reference; else use union of all kept checkpoints across dirs evaluated
    all_points: List[Tuple[float, float]] = []
    per_arch_points: Dict[str, List[Tuple[float, float]]] = {}
    per_arch_ckpt_points: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = {}
    per_arch_seed_points: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    for arch, ckpts in arch_ckpts.items():
        pts_arch: List[Tuple[float, float]] = []
        if args.front_scope in ("per_ckpt", "per_seed") or args.plot_per_seed:
            per_arch_ckpt_points[arch] = {}
            per_arch_seed_points[arch] = {}
            for fp in tqdm(ckpts, desc=f"eval_ckpts_{arch}"):
                epi_pts = eval_checkpoint_points_with_epi(fp, eval_args, seed_base=args.seed_base)
                per_arch_ckpt_points[arch][fp.stem] = epi_pts
                for rec in epi_pts:
                    epi = int(rec[0])
                    mk_raw, ae_raw = float(rec[3]), float(rec[4])
                    pts_arch.append((mk_raw, ae_raw))
                    per_arch_seed_points[arch].setdefault(epi, []).append((mk_raw, ae_raw))
        else:
            for fp in tqdm(ckpts, desc=f"eval_ckpts_{arch}"):
                pts = eval_checkpoint_points(fp, eval_args, seed_base=args.seed_base)
                pts_arch.extend(pts)
        per_arch_points[arch] = pts_arch
        all_points.extend(pts_arch)
        print(f"[eval] {arch}: {len(pts_arch)} points")

    ref_front: List[Tuple[float, float]]
    ref_source = "union"
    if args.ref_front_csv:
        ref_path = Path(args.ref_front_csv)
        if ref_path.exists():
            import csv
            pts: List[Tuple[float, float]] = []
            with ref_path.open("r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                # Try to detect header names or fallback to two columns
                for row in r:
                    try:
                        if header:
                            lower = [h.strip().lower() for h in header]
                            if "mk_ratio" in lower and "ae_ratio" in lower:
                                mk_idx = lower.index("mk_ratio")
                                ae_idx = lower.index("ae_ratio")
                                x = float(row[mk_idx])
                                y = float(row[ae_idx])
                            elif "makespan" in lower and "active_energy" in lower:
                                mk_idx = lower.index("makespan")
                                ae_idx = lower.index("active_energy")
                                x = float(row[mk_idx])
                                y = float(row[ae_idx])
                            else:
                                x = float(row[0])
                                y = float(row[1])
                        else:
                            x = float(row[0])
                            y = float(row[1])
                        pts.append((x, y))
                    except Exception:
                        continue
            ref_front = non_dominated(pts)
            ref_source = f"csv:{ref_path}"
            print(f"[ref] Loaded reference front from {ref_path} with {len(ref_front)} points (after ND)")
        else:
            print(f"[ref] Warning: ref CSV not found at {ref_path}, falling back to union")
            ref_front = non_dominated(all_points)
    else:
        ref_front = non_dominated(all_points)

    # Optional: load NSGA population history for scatter overlay
    ref_population: List[Tuple[float, float]] = []
    if args.ref_pop_csv:
        pop_path = Path(args.ref_pop_csv)
        if pop_path.exists():
            import csv as _csv
            with pop_path.open("r", newline="") as f:
                rd = _csv.reader(f)
                header = next(rd, None)
                mk_idx, ae_idx = None, None
                lower = [h.strip().lower() for h in header] if header else []
                if header:
                    if "mk_ratio" in lower and "ae_ratio" in lower:
                        mk_idx = lower.index("mk_ratio")
                        ae_idx = lower.index("ae_ratio")
                    elif "makespan" in lower and "active_energy" in lower:
                        mk_idx = lower.index("makespan")
                        ae_idx = lower.index("active_energy")
                for row in rd:
                    try:
                        if mk_idx is not None and ae_idx is not None:
                            x = float(row[mk_idx])
                            y = float(row[ae_idx])
                        else:
                            # Fallback assume columns: generation, idx, mk, ae
                            x = float(row[2])
                            y = float(row[3])
                        ref_population.append((x, y))
                    except Exception:
                        continue
            print(f"[ref] Loaded NSGA population from {pop_path} with {len(ref_population)} points")
        else:
            print(f"[ref] Warning: ref population CSV not found at {pop_path}")

    # Step 3: generate Pareto fronts per architecture from their evaluated points
    per_arch_fronts: Dict[str, List[Tuple[float, float]]] = {
        arch: non_dominated(pts) for arch, pts in per_arch_points.items()
    }

    # Optional emission: per-checkpoint fronts (over episodes) and per-seed fronts (across checkpoints)
    import csv as _csv
    if args.front_scope == "per_ckpt":
        base_dir = Path(args.out_dir) / "fronts_per_ckpt"
        for arch, ckpt_map in per_arch_ckpt_points.items():
            for ckpt_stem, epi_pts in ckpt_map.items():
                fr = non_dominated([(float(rec[3]), float(rec[4])) for rec in epi_pts])
                outp = base_dir / arch / f"{ckpt_stem}.csv"
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["makespan", "active_energy"]) 
                    for x, y in sorted(fr):
                        w.writerow([x, y])
        print(f"[fronts] Wrote per-ckpt fronts to {base_dir}")
    elif args.front_scope == "per_seed":
        base_dir = Path(args.out_dir) / "fronts_per_seed"
        for arch, seed_map in per_arch_seed_points.items():
            for epi, pts in seed_map.items():
                fr = non_dominated(pts)
                abs_seed = int(args.seed_base) + int(epi)
                outp = base_dir / arch / f"seed_{abs_seed}.csv"
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["makespan", "active_energy"]) 
                    for x, y in sorted(fr):
                        w.writerow([x, y])
        print(f"[fronts] Wrote per-seed fronts to {base_dir}")

    # Optional plotting: per-seed Pareto plots overlaying architectures
    if args.plot_per_seed:
        plots_dir = Path(args.out_dir) / "plots_per_seed"
        plots_dir.mkdir(parents=True, exist_ok=True)
    # Collect union of episode indices observed across architectures (for reports and optional plots)
    seeds_all: List[int] = []
    for arch, seed_map in per_arch_seed_points.items():
        for epi in seed_map.keys():
            if epi not in seeds_all:
                seeds_all.append(int(epi))
    seeds_all = sorted(seeds_all)

    if args.plot_per_seed:
        # Refined palette and markers (keep simple, harmonious)
        colors = {"hetero": "#1f77b4", "baseline": "#ff7f0e", "mlp_only": "#2ca02c"}
        markers = {"hetero": "o", "baseline": "s", "mlp_only": "^"}
        display_names = {"hetero": "DeepRL", "baseline": "baseline", "mlp_only": "MLP"}
        for epi in seeds_all:
            plt.figure(figsize=(6.0, 4.2))
            plotted_any = False
            ax = plt.gca()
            if args.style_ga:
                # Subtle panel style: hide top/right spines
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                ax.grid(True, which="major", linewidth=1.2, color="white")
                ax.grid(True, which="minor", linewidth=0.6, color="white")
            for arch in per_arch_points.keys():
                pts = per_arch_seed_points.get(arch, {}).get(int(epi), [])
                if not pts:
                    continue
                xs = [x for (x, _y) in pts]
                ys = [y for (_x, y) in pts]
                c = colors.get(arch, None)
                m = markers.get(arch, "o")
                dot_size = 10 if args.style_ga else 18
                dot_alpha = 0.35 if args.style_ga else 0.25
                name = display_names.get(arch, arch)
                plt.scatter(xs, ys, s=dot_size, alpha=dot_alpha, label=f"{name} (pts)", color=c, edgecolors="none")
                fr = non_dominated(pts)
                if fr:
                    frs = sorted(fr)
                    lw = 2.0 if args.style_ga else 1.6
                    ms = 4.0 if args.style_ga else 4.5
                    plt.plot([x for x, _ in frs], [y for _, y in frs],
                             marker=m, markersize=ms, linewidth=lw,
                             label=f"{name} front", color=c)
                plotted_any = True
            # Overlay NSGA reference elements (global, not per-seed)
            if (args.plot_ref_front or True) and ref_front:
                rfs = sorted(ref_front)
                nsga_color = "#d62728" if args.style_ga else "#000000"
                plt.plot([x for x, _ in rfs], [y for _, y in rfs], color=nsga_color,
                         marker="o" if args.style_ga else "x", markersize=3 if args.style_ga else 5,
                         linestyle="-" if args.style_ga else "--", linewidth=2.2 if args.style_ga else 1.6,
                         label="NSGA front")
            if ref_population:
                px = [x for x, _ in ref_population]
                py = [y for _, y in ref_population]
                # GA (NSGA) population uses same hue as the NSGA front but lighter
                pop_color = nsga_color if (args.plot_ref_front and ref_front) else ("#d62728" if args.style_ga else "#7f7f7f")
                pop_alpha = 0.18 if args.style_ga else 0.06
                plt.scatter(px, py, s=6, alpha=pop_alpha, color=pop_color, label="NSGA pop", edgecolors="none")
            if plotted_any:
                abs_seed = int(args.seed_base) + int(epi)
                plt.xlabel("makespan (lower better)")
                plt.ylabel("active energy (lower better)")
                if not args.style_ga:
                    plt.title(f"Pareto per seed {abs_seed}")
                leg = plt.legend(loc="best", frameon=False, ncol=1)
                for line in leg.get_lines():
                    line.set_linewidth(1.4)
                plt.tight_layout()
                # Save PNG and PDF for publication quality
                plt.savefig(plots_dir / f"seed_{abs_seed}.png", dpi=300)
                plt.savefig(plots_dir / f"seed_{abs_seed}.pdf")
                plt.close()
        print(f"[plots] Wrote per-seed plots to {plots_dir}")

    # Per-seed CSV reports: include env stats and per-architecture episode results (+ dominance per seed)
    if seeds_all:
        reports_dir = Path(args.out_dir) / "seed_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        for epi in seeds_all:
            abs_seed = int(args.seed_base) + int(epi)
            outp = reports_dir / f"seed_{abs_seed}.csv"
            with outp.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["kind","seed","key","value","arch","ckpt","makespan","active_energy","coverage_over","coverage_value","dominance"])
                # Env stats: take from first available arch/ckpt for this epi
                env_stats_written = False
                for arch, ckpt_map in per_arch_ckpt_points.items():
                    if env_stats_written:
                        break
                    for ckpt_stem, epi_pts in ckpt_map.items():
                        for rec in epi_pts:
                            if int(rec[0]) == int(epi):
                                stats = rec[5] if len(rec) >= 6 else {}
                                for k, v in (stats or {}).items():
                                    w.writerow(["env", abs_seed, k, v, None, None, None, None, None, None, None])
                                env_stats_written = True
                                break
                        if env_stats_written:
                            break
                # Points per architecture for this seed (list all checkpoints)
                per_arch_pts_this_seed: Dict[str, List[Tuple[float, float]]] = {}
                for arch, ckpt_map in per_arch_ckpt_points.items():
                    for ckpt_stem, epi_pts in ckpt_map.items():
                        for rec in epi_pts:
                            if int(rec[0]) == int(epi):
                                mk_raw, ae_raw = float(rec[3]), float(rec[4])
                                w.writerow(["point", abs_seed, None, None, arch, ckpt_stem, mk_raw, ae_raw, None, None, None])
                                per_arch_pts_this_seed.setdefault(arch, []).append((mk_raw, ae_raw))
                # Dominance/coverage summary between architectures for this seed
                archs = sorted(per_arch_pts_this_seed.keys())
                if len(archs) >= 2:
                    # Pairwise coverage
                    cov_matrix: Dict[Tuple[str,str], float] = {}
                    for a in archs:
                        for b in archs:
                            if a == b:
                                continue
                            cov_val = coverage(per_arch_pts_this_seed.get(a, []), per_arch_pts_this_seed.get(b, []))
                            cov_matrix[(a,b)] = cov_val
                            w.writerow(["coverage", abs_seed, None, None, a, None, None, None, b, cov_val, None])
                    # Simple dominance decision by net coverage
                    net: Dict[str, float] = {a: 0.0 for a in archs}
                    for a in archs:
                        for b in archs:
                            if a == b:
                                continue
                            net[a] += cov_matrix.get((a,b), 0.0) - cov_matrix.get((b,a), 0.0)
                    best = max(net.items(), key=lambda t: t[1]) if net else (None, 0.0)
                    winner = best[0] if best and best[1] > 0 else "tie_or_unclear"
                    w.writerow(["dominance_summary", abs_seed, None, None, winner, None, None, None, None, None, "net_coverage>0 implies winner"]) 
            print(f"[seed] Wrote report: {outp}")

    # Step 4: per-seed indicators on raw metrics (HV, IGD, IGD+, spacing, coverage)
    if seeds_all:
        indicators_dir = Path(args.out_dir) / "seed_indicators"
        indicators_dir.mkdir(parents=True, exist_ok=True)
        for epi in seeds_all:
            abs_seed = int(args.seed_base) + int(epi)
            # Gather per-seed points per arch and union
            per_arch_pts_this_seed: Dict[str, List[Tuple[float, float]]] = {}
            for arch, ckpt_map in per_arch_ckpt_points.items():
                for ckpt_stem, epi_pts in ckpt_map.items():
                    for rec in epi_pts:
                        if int(rec[0]) == int(epi):
                            mk_raw, ae_raw = float(rec[3]), float(rec[4])
                            per_arch_pts_this_seed.setdefault(arch, []).append((mk_raw, ae_raw))
            all_pts_this_seed: List[Tuple[float, float]] = []
            for pts in per_arch_pts_this_seed.values():
                all_pts_this_seed.extend(pts)
            # Reference point for HV for this seed
            if all_pts_this_seed:
                rx = 1.1 * max(x for (x, _y) in all_pts_this_seed)
                ry = 1.1 * max(y for (_x, y) in all_pts_this_seed)
                ref_pt_seed = (rx, ry)
                norm_min_seed = (min(x for (x, _y) in all_pts_this_seed), min(y for (_x, y) in all_pts_this_seed))
                norm_max_seed = (max(x for (x, _y) in all_pts_this_seed), max(y for (_x, y) in all_pts_this_seed))
            else:
                ref_pt_seed = (1.0, 1.0)
                norm_min_seed = (0.0, 0.0)
                norm_max_seed = (1.0, 1.0)
            import csv as _csv
            with (indicators_dir / f"seed_{abs_seed}.csv").open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["arch", "hv", "igd", "igd_plus", "spacing", "coverage_all_pts"])
                for arch in sorted(per_arch_pts_this_seed.keys()):
                    pts = per_arch_pts_this_seed[arch]
                    fr = non_dominated(pts)
                    hv_arch = hv2d(fr, ref_pt_seed) if fr else 0.0
                    igd_val = igd(fr, ref_front) if fr and ref_front else float('inf')
                    igd_p = igd_plus(fr, ref_front) if fr and ref_front else float('inf')
                    spacing_val = schott_spacing(fr, norm_min_seed, norm_max_seed) if fr else 0.0
                    cov_val = coverage(fr, all_pts_this_seed) if fr and all_pts_this_seed else 0.0
                    w.writerow([arch, hv_arch, igd_val, igd_p, spacing_val, cov_val])
            print(f"[indicators] Wrote per-seed indicators: {indicators_dir / f'seed_{abs_seed}.csv'}")

    # Visualizations: combined fronts (match per-seed styling)
    plt.figure(figsize=(6.0, 4.2))
    ax = plt.gca()
    if args.style_ga:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, which="major", linewidth=1.2, color="white")
        ax.grid(True, which="minor", linewidth=0.6, color="white")

    colors_map = {"hetero": "#1f77b4", "baseline": "#ff7f0e", "mlp_only": "#2ca02c"}
    markers_map = {"hetero": "o", "baseline": "s", "mlp_only": "^"}
    display_names = {"hetero": "DeepRL", "baseline": "baseline", "mlp_only": "MLP"}

    # plot all points lightly per arch
    for arch, pts in per_arch_points.items():
        if not pts:
            continue
        xs = [x for (x, _y) in pts]
        ys = [y for (_x, y) in pts]
        c = colors_map.get(arch, None)
        name = display_names.get(arch, arch)
        plt.scatter(xs, ys, s=10, alpha=0.25, color=c, label=f"{name} (pts)", edgecolors="none")

    # plot each architecture front
    for arch, fr in per_arch_fronts.items():
        if not fr:
            continue
        frs = sorted(fr)
        c = colors_map.get(arch, None)
        m = markers_map.get(arch, "o")
        lw = 2.0 if args.style_ga else 1.6
        ms = 4.0 if args.style_ga else 4.5
        name = display_names.get(arch, arch)
        plt.plot([x for x, _ in frs], [y for _, y in frs], marker=m, markersize=ms, linewidth=lw, color=c, label=f"{name} front")

    # NSGA overlays
    nsga_color = "#d62728" if args.style_ga else "#000000"
    if ref_population:
        px = [x for x, _ in ref_population]
        py = [y for _, y in ref_population]
        pop_alpha = 0.18 if args.style_ga else 0.06
        plt.scatter(px, py, s=6, alpha=pop_alpha, color=nsga_color, label="NSGA pop", edgecolors="none")
    if ref_front:
        rfs = sorted(ref_front)
        plt.plot([x for x, _ in rfs], [y for _, y in rfs], color=nsga_color,
                 marker="o" if args.style_ga else "x", markersize=3 if args.style_ga else 5,
                 linestyle="-" if args.style_ga else "--", linewidth=2.2 if args.style_ga else 1.6,
                 label="NSGA front")

    plt.xlabel("makespan (lower better)")
    plt.ylabel("active energy (lower better)")
    if not args.style_ga:
        plt.title("Pareto fronts vs Reference")
    leg = plt.legend(loc="best", frameon=False, ncol=1)
    for line in leg.get_lines():
        line.set_linewidth(1.4)
    plt.tight_layout()
    plt.savefig(out_dir / "fronts.png", dpi=300)
    plt.savefig(out_dir / "fronts.pdf")
    plt.close()

    # Write a small meta file noting the reference source
    with (out_dir / "_ref_source.txt").open("w") as f:
        f.write(str(ref_source) + "\n")

    print(f"[done] Results in: {out_dir}")


if __name__ == "__main__":
    main()
