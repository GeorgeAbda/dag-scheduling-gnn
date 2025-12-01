import argparse
import math
import random
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Make project root importable when running as a script (so 'scheduler' resolves)
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# Internal imports
from scheduler.rl_model.ablation_gnn import Args as TrainArgs, AblationVariant, AblationGinAgent
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from tqdm import tqdm, trange


# ===========================
# Multi-objective helpers
# ===========================

def is_dominated(p: Tuple[float, float], q: Tuple[float, float]) -> bool:
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])


def non_dominated(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    keep = [True] * len(points)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i == j or not keep[j]:
                continue
            if is_dominated(points[i], points[j]):
                keep[i] = False
                break
    return [points[i] for i, k in enumerate(keep) if k]


def hv2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    if not points:
        return 0.0
    fr = non_dominated(points)
    if not fr:
        return 0.0
    fr = [(min(x, ref[0]), min(y, ref[1])) for (x, y) in fr if x <= ref[0] and y <= ref[1]]
    if not fr:
        return 0.0
    fr.sort(key=lambda t: (t[0], t[1]))
    chain = []
    best_y = math.inf
    for x, y in fr:
        if y < best_y:
            chain.append((x, y))
            best_y = y
    hv = 0.0
    y_prev = ref[1]
    for x, y in chain:
        dx = max(ref[0] - x, 0.0)
        dy = max(y_prev - y, 0.0)
        hv += dx * dy
        y_prev = y
    return hv


def crowding_distance(front_idx: List[int], objs: List[Tuple[float, float]]) -> Dict[int, float]:
    if not front_idx:
        return {}
    n = len(front_idx)
    dist = {i: 0.0 for i in front_idx}
    # For each objective
    for dim in [0, 1]:
        front_idx_sorted = sorted(front_idx, key=lambda i: objs[i][dim])
        dist[front_idx_sorted[0]] = float('inf')
        dist[front_idx_sorted[-1]] = float('inf')
        min_v = objs[front_idx_sorted[0]][dim]
        max_v = objs[front_idx_sorted[-1]][dim]
        denom = max(max_v - min_v, 1e-12)
        for k in range(1, n - 1):
            i_prev = front_idx_sorted[k - 1]
            i_next = front_idx_sorted[k + 1]
            dist[front_idx_sorted[k]] += (objs[i_next][dim] - objs[i_prev][dim]) / denom
    return dist


def fast_non_dominated_sort(objs: List[Tuple[float, float]]) -> List[List[int]]:
    S = [set() for _ in objs]
    n_dom = [0] * len(objs)
    fronts: List[List[int]] = [[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if p == q:
                continue
            if is_dominated(objs[q], objs[p]):
                S[p].add(q)
            elif is_dominated(objs[p], objs[q]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


# ===========================
# Evaluation
# ===========================

def build_eval_args(base: TrainArgs, nsga_args) -> TrainArgs:
    new = dc_replace(base)
    ds = dc_replace(base.dataset)
    if nsga_args.dag_method:
        ds = dc_replace(ds, dag_method=nsga_args.dag_method)
    if nsga_args.gnp_min_n is not None:
        ds = dc_replace(ds, gnp_min_n=int(nsga_args.gnp_min_n))
    if nsga_args.gnp_max_n is not None:
        ds = dc_replace(ds, gnp_max_n=int(nsga_args.gnp_max_n))
    if nsga_args.host_count is not None:
        ds = dc_replace(ds, host_count=int(nsga_args.host_count))
    if nsga_args.vm_count is not None:
        ds = dc_replace(ds, vm_count=int(nsga_args.vm_count))
    if nsga_args.workflow_count is not None:
        ds = dc_replace(ds, workflow_count=int(nsga_args.workflow_count))
    new.dataset = ds
    new.device = str(nsga_args.device)
    new.test_iterations = int(nsga_args.episodes)
    return new


def evaluate_agent_mean_ratios(agent: AblationGinAgent, eval_args: TrainArgs, seed_base: int) -> Tuple[float, float, List[int], List[float], List[float]]:
    device = torch.device(eval_args.device if eval_args.device else "cpu")
    env = CloudSchedulingGymEnvironment(
        dataset_args=eval_args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    env = GinAgentWrapper(env, constrained_mode=False)

    xs: List[float] = []
    ys: List[float] = []
    seeds: List[int] = []

    def _select_action(_agent: AblationGinAgent, mapped_obs: np.ndarray) -> int:
        x = torch.as_tensor(mapped_obs, dtype=torch.float32, device=_agent.device).unsqueeze(0)
        a, _lp, _ent, _val = _agent.get_action_and_value(x)
        return int(a[0].item())
    for epi in range(eval_args.test_iterations):
        cur_seed = int(seed_base + epi)
        obs, _ = env.reset(seed=cur_seed)
        # baselines
        baseline_mk = float(env.prev_obs.makespan()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
        baseline_active = float(getattr(env.prev_obs, 'energy_consumption', lambda: float('nan'))()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
        done = False
        while not done:
            with torch.no_grad():
                action = _select_action(agent, obs)
            obs, reward, done, trunc, info = env.step(action)
            done = bool(done or trunc)
        mk = float(info.get('makespan', env.prev_obs.makespan()))
        ae = float(info.get('total_energy_active', getattr(env.prev_obs, 'energy_consumption', lambda: mk)()))
        if np.isfinite(baseline_mk) and baseline_mk > 0 and np.isfinite(baseline_active) and baseline_active > 0:
            xs.append(mk / baseline_mk)
            ys.append(ae / baseline_active)
        else:
            xs.append(mk)
            ys.append(ae)
        seeds.append(cur_seed)
    return float(np.mean(xs)), float(np.mean(ys)), seeds, xs, ys


def evaluate_agent_seed_ratio(agent: AblationGinAgent, eval_args: TrainArgs, seed: int) -> Tuple[float, float]:
    device = torch.device(eval_args.device if eval_args.device else "cpu")
    env = CloudSchedulingGymEnvironment(
        dataset_args=eval_args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    env = GinAgentWrapper(env, constrained_mode=False)

    def _select_action(_agent: AblationGinAgent, mapped_obs: np.ndarray) -> int:
        x = torch.as_tensor(mapped_obs, dtype=torch.float32, device=_agent.device).unsqueeze(0)
        a, _lp, _ent, _val = _agent.get_action_and_value(x)
        return int(a[0].item())

    obs, _ = env.reset(seed=int(seed))
    baseline_mk = float(env.prev_obs.makespan()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
    baseline_active = float(getattr(env.prev_obs, 'energy_consumption', lambda: float('nan'))()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
    done = False
    while not done:
        with torch.no_grad():
            action = _select_action(agent, obs)
        obs, reward, done, trunc, info = env.step(action)
        done = bool(done or trunc)
    mk = float(info.get('makespan', env.prev_obs.makespan()))
    ae = float(info.get('total_energy_active', getattr(env.prev_obs, 'energy_consumption', lambda: mk)()))
    if np.isfinite(baseline_mk) and baseline_mk > 0 and np.isfinite(baseline_active) and baseline_active > 0:
        return float(mk / baseline_mk), float(ae / baseline_active)
    return float(mk), float(ae)


# ===========================
# NSGA-II over policy parameters
# ===========================

KNOWN_VARIANTS = {
    "baseline": AblationVariant(name="baseline"),
    "mlp_only": AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False),
    "hetero": AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True),
}


def init_agent(variant_name: str, device: torch.device) -> AblationGinAgent:
    variant = KNOWN_VARIANTS.get(variant_name, AblationVariant(name=variant_name))
    agent = AblationGinAgent(device, variant)
    agent.eval()
    return agent


def clone_agent(agent: AblationGinAgent) -> AblationGinAgent:
    device = next(agent.parameters()).device
    variant = agent.variant
    new_agent = AblationGinAgent(device, variant)
    new_agent.load_state_dict(agent.state_dict(), strict=False)
    new_agent.eval()
    return new_agent


def params_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in model.parameters()])


def assign_from_vector(model: nn.Module, vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(vec[idx:idx+num].view_as(p))
        idx += num


def blend_crossover(p1: torch.Tensor, p2: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Simulated blend crossover
    w = torch.rand_like(p1) * (1 + 2*alpha) - alpha
    c1 = w * p1 + (1 - w) * p2
    c2 = w * p2 + (1 - w) * p1
    return c1, c2


def gaussian_mutation(vec: torch.Tensor, sigma: float, ratio: float) -> torch.Tensor:
    mask = (torch.rand_like(vec) < ratio).float()
    noise = torch.randn_like(vec) * sigma
    return vec + mask * noise


# ===========================
# Main
# ===========================

def main():
    ap = argparse.ArgumentParser(description="NSGA-II reference policy search (mk_ratio, ae_ratio)")
    ap.add_argument("--variant", type=str, default="hetero", help="Architecture to evolve (hetero|baseline|mlp_only|...")
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=4242)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="logs/nsga_reference")
    ap.add_argument("--sigma", type=float, default=0.02, help="Gaussian mutation std")
    ap.add_argument("--mut-ratio", type=float, default=0.1, help="Fraction of params to mutate")
    ap.add_argument("--cx-alpha", type=float, default=0.1, help="Blend crossover alpha")
    ap.add_argument("--cx-prob", type=float, default=0.9)
    # Dataset overrides matching training eval
    ap.add_argument("--dag-method", type=str, default=None)
    ap.add_argument("--gnp-min-n", type=int, default=None)
    ap.add_argument("--gnp-max-n", type=int, default=None)
    ap.add_argument("--host-count", type=int, default=None)
    ap.add_argument("--vm-count", type=int, default=None)
    ap.add_argument("--workflow-count", type=int, default=None)
    ap.add_argument("--gnp-p", type=float, default=None)
    ap.add_argument("--opt-per-seed", action="store_true", help="Run independent NSGA for each episode seed and output per-seed fronts under out-dir/seed_<seed>/")
    # Episode seed scheme
    ap.add_argument("--resample-episodes", action="store_true", help="If set, use different episode seeds per generation (seed_base + 1000*g + epi). By default, use fixed seeds across all generations (seed_base + epi) to match ablation evaluation.")
    args = ap.parse_args()

    random.seed(args.seed_base)
    np.random.seed(args.seed_base)
    torch.manual_seed(args.seed_base)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build eval args
    base = TrainArgs()
    eval_args = build_eval_args(base, args)
    device = torch.device(args.device)

    # Initialize population (randomly initialized weights)
    population: List[AblationGinAgent] = [init_agent(args.variant, device) for _ in range(args.population)]

    # If requested, optimize independently per seed and exit
    if args.opt_per_seed:
        seeds = [int(args.seed_base + i) for i in range(int(eval_args.test_iterations))]
        for seed in seeds:
            seed_dir = out_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            # initialize fresh population for this seed
            population = [init_agent(args.variant, device) for _ in range(args.population)]
            # Eval gen 0
            import csv
            objs: List[Tuple[float, float]] = []
            with (seed_dir / "history.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["generation", "idx", "mk_ratio", "ae_ratio"])  # single-seed fitness
                for i, ind in enumerate(tqdm(population, desc=f"Eval gen 0 seed {seed}", leave=False)):
                    x, y = evaluate_agent_seed_ratio(ind, eval_args, seed)
                    objs.append((x, y))
                    w.writerow([0, i, x, y])
            # Evolution per seed
            for gen in trange(1, args.generations + 1, desc=f"NSGA-II generations seed {seed}"):
                fronts = fast_non_dominated_sort(objs)
                distances: Dict[int, float] = {}
                for f in fronts:
                    distances.update(crowding_distance(f, objs))
                next_pop: List[AblationGinAgent] = []
                next_objs: List[Tuple[float, float]] = []
                for f in fronts:
                    if len(next_pop) + len(f) <= args.population:
                        for idx in f:
                            next_pop.append(population[idx])
                            next_objs.append(objs[idx])
                    else:
                        f_sorted = sorted(f, key=lambda i: distances.get(i, 0.0), reverse=True)
                        slots = args.population - len(next_pop)
                        for idx in f_sorted[:slots]:
                            next_pop.append(population[idx])
                            next_objs.append(objs[idx])
                        break
                # Variation
                offspring: List[AblationGinAgent] = []
                while len(offspring) + len(next_pop) < args.population * 2:
                    def tour_pick() -> AblationGinAgent:
                        i, j = np.random.randint(0, len(next_pop)), np.random.randint(0, len(next_pop))
                        if distances.get(i, 0.0) > distances.get(j, 0.0):
                            return next_pop[i]
                        if distances.get(i, 0.0) < distances.get(j, 0.0):
                            return next_pop[j]
                        return next_pop[i] if (next_objs[i][0] + next_objs[i][1]) <= (next_objs[j][0] + next_objs[j][1]) else next_pop[j]
                    p1 = tour_pick()
                    p2 = tour_pick()
                    v1 = params_vector(p1)
                    v2 = params_vector(p2)
                    if np.random.rand() < args.cx_prob:
                        c1v, c2v = blend_crossover(v1, v2, args.cx_alpha)
                    else:
                        c1v, c2v = v1.clone(), v2.clone()
                    c1v = gaussian_mutation(c1v, sigma=args.sigma, ratio=args.mut_ratio)
                    c2v = gaussian_mutation(c2v, sigma=args.sigma, ratio=args.mut_ratio)
                    c1 = clone_agent(p1)
                    c2 = clone_agent(p2)
                    assign_from_vector(c1, c1v)
                    assign_from_vector(c2, c2v)
                    offspring.extend([c1, c2])
                # Evaluate offspring at this fixed seed
                off_objs: List[Tuple[float, float]] = []
                for j, ind in enumerate(tqdm(offspring, desc=f"Eval offspring g{gen} seed {seed}", leave=False)):
                    x, y = evaluate_agent_seed_ratio(ind, eval_args, seed)
                    off_objs.append((x, y))
                population = next_pop + offspring
                objs = next_objs + off_objs
                # Log and save front
                with (seed_dir / "history.csv").open("a", newline="") as f:
                    w = csv.writer(f)
                    for i, (x, y) in enumerate(objs):
                        w.writerow([gen, i, x, y])
                ref_front = non_dominated(objs)
                with (seed_dir / "reference_front.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["mk_ratio", "ae_ratio"])
                    for x, y in sorted(ref_front):
                        w.writerow([x, y])
            print(f"[nsga][per-seed] Done seed={seed}. Outputs in: {seed_dir}")
        print(f"[nsga][per-seed] All seeds done. Root: {out_dir}")
        return

    # Evaluate population
    objs: List[Tuple[float, float]] = []
    # Per-seed accumulation across the whole run
    per_seed_objs: Dict[int, List[Tuple[float, float]]] = {}
    # Prepare per-seed history CSV
    import csv
    per_seed_hist_path = out_dir / "history_per_seed.csv"
    with per_seed_hist_path.open("w", newline="") as fhs:
        whs = csv.writer(fhs)
        whs.writerow(["generation", "idx", "seed", "mk_ratio", "ae_ratio"])  # per-episode values

    for idx, ind in enumerate(tqdm(population, desc="Eval gen 0", leave=False)):
        x, y, seeds_list, xs_list, ys_list = evaluate_agent_mean_ratios(ind, eval_args, args.seed_base)
        objs.append((x, y))
        # write per-seed rows
        with per_seed_hist_path.open("a", newline="") as fhs:
            whs = csv.writer(fhs)
            for s, vx, vy in zip(seeds_list, xs_list, ys_list):
                whs.writerow([0, idx, s, vx, vy])
                per_seed_objs.setdefault(int(s), []).append((float(vx), float(vy)))

    # Archives for logging
    with (out_dir / "history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "idx", "mk_ratio", "ae_ratio"])  # mean ratios per individual per generation
        for i, (x, y) in enumerate(objs):
            w.writerow([0, i, x, y])

    # Evolution loop
    for gen in trange(1, args.generations + 1, desc="NSGA-II generations"):
        # Fast non-dominated sorting
        fronts = fast_non_dominated_sort(objs)
        # Crowding distances
        distances: Dict[int, float] = {}
        for f in fronts:
            distances.update(crowding_distance(f, objs))

        # Selection (NSGA-II): fill next gen by fronts and crowding
        next_pop: List[AblationGinAgent] = []
        next_objs: List[Tuple[float, float]] = []
        for f in fronts:
            if len(next_pop) + len(f) <= args.population:
                for idx in f:
                    next_pop.append(population[idx])
                    next_objs.append(objs[idx])
            else:
                # sort f by crowding desc
                f_sorted = sorted(f, key=lambda i: distances.get(i, 0.0), reverse=True)
                slots = args.population - len(next_pop)
                for idx in f_sorted[:slots]:
                    next_pop.append(population[idx])
                    next_objs.append(objs[idx])
                break

        # Variation: crossover + mutation to create offspring
        offspring: List[AblationGinAgent] = []
        while len(offspring) + len(next_pop) < args.population * 2:  # create enough, we'll evaluate and select via NSGA-II
            # parent selection via binary tournaments on crowding within the first fronts
            def tour_pick() -> AblationGinAgent:
                i, j = np.random.randint(0, len(next_pop)), np.random.randint(0, len(next_pop))
                # Prefer lower mk+ae as a quick heuristic when distances equal
                if distances.get(i, 0.0) > distances.get(j, 0.0):
                    return next_pop[i]
                if distances.get(i, 0.0) < distances.get(j, 0.0):
                    return next_pop[j]
                return next_pop[i] if (next_objs[i][0] + next_objs[i][1]) <= (next_objs[j][0] + next_objs[j][1]) else next_pop[j]

            p1 = tour_pick()
            p2 = tour_pick()
            v1 = params_vector(p1)
            v2 = params_vector(p2)
            if np.random.rand() < args.cx_prob:
                c1v, c2v = blend_crossover(v1, v2, args.cx_alpha)
            else:
                c1v, c2v = v1.clone(), v2.clone()
            c1v = gaussian_mutation(c1v, sigma=args.sigma, ratio=args.mut_ratio)
            c2v = gaussian_mutation(c2v, sigma=args.sigma, ratio=args.mut_ratio)

            c1 = clone_agent(p1)
            c2 = clone_agent(p2)
            assign_from_vector(c1, c1v)
            assign_from_vector(c2, c2v)
            offspring.extend([c1, c2])

        # Evaluate offspring
        off_objs: List[Tuple[float, float]] = []
        for j, ind in enumerate(tqdm(offspring, desc=f"Eval offspring g{gen}", leave=False)):
            seed_for_gen = args.seed_base + (gen * 1000 if args.resample_episodes else 0)
            x, y, seeds_list, xs_list, ys_list = evaluate_agent_mean_ratios(ind, eval_args, seed_for_gen)
            off_objs.append((x, y))
            # append to per-seed history and accumulation
            with per_seed_hist_path.open("a", newline="") as fhs:
                whs = csv.writer(fhs)
                for s, vx, vy in zip(seeds_list, xs_list, ys_list):
                    whs.writerow([gen, j, s, vx, vy])
                    per_seed_objs.setdefault(int(s), []).append((float(vx), float(vy)))

        # Combine and select next generation
        population = next_pop + offspring
        objs = next_objs + off_objs

        # Log
        with (out_dir / "history.csv").open("a", newline="") as f:
            w = csv.writer(f)
            for i, (x, y) in enumerate(objs):
                w.writerow([gen, i, x, y])

        # Save current reference front (mean-based)
        ref_front = non_dominated(objs)
        with (out_dir / "reference_front.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mk_ratio", "ae_ratio"])
            for x, y in sorted(ref_front):
                w.writerow([x, y])
        # Also write per-seed reference fronts over all individuals evaluated so far
        per_seed_dir = out_dir / "reference_fronts_per_seed"
        per_seed_dir.mkdir(parents=True, exist_ok=True)
        for seed_key, pts in per_seed_objs.items():
            fr_seed = non_dominated(pts)
            with (per_seed_dir / f"seed_{int(seed_key)}.csv").open("w", newline="") as fs:
                ws = csv.writer(fs)
                ws.writerow(["mk_ratio", "ae_ratio"]) 
                for x, y in sorted(fr_seed):
                    ws.writerow([x, y])
        # Plot
        plt.figure(figsize=(7, 5))
        xs = [x for x, _ in objs]
        ys = [y for _, y in objs]
        plt.scatter(xs, ys, s=8, alpha=0.25, label="population")
        rfs = sorted(ref_front)
        plt.plot([x for x, _ in rfs], [y for _, y in rfs], color="black", marker="x", linestyle="--", label="NSGA-II front")
        plt.xlabel("mk_ratio (lower better)")
        plt.ylabel("ae_ratio (lower better)")
        plt.title(f"NSGA-II generation {gen}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"front_gen{gen:03d}.png", dpi=180)
        plt.close()

    print(f"[nsga] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
