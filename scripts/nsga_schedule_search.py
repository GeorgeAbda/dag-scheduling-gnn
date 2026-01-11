#!/usr/bin/env python3
"""
NSGA-II over real schedules (task ordering + VM assignment), evaluated in the
CloudSchedulingGymEnvironment.

Genome encoding
- Each genome consists of two parts for T tasks and V VMs:
  1) priority keys: real vector of length T. Tasks are scheduled by increasing
     key, but always respecting precedence; at each step we pick, from the READY
     set, the unscheduled task with the smallest key.
  2) vm choices: integer vector of length T in [0, V-1]. When scheduling a task,
     we try the assigned VM; if incompatible we greedily map to the closest
     compatible VM by round-robin search. This yields a concrete, feasible
     schedule for the environment.

Objectives
- Minimize (mk_ratio, ae_ratio) measured with the environment and its baseline
  metrics. The ratios are computed exactly like in other analysis scripts.

Notes
- This script focuses on single-seed optimization (use --seed). You can run a
  loop externally to optimize different seeds independently.
- For speed/robustness, variation operators act separately on the two genome
  parts (Gaussian for priority keys, uniform mutation for VM choices; and simple
  one-point crossover for both parts).

"""
from __future__ import annotations

import argparse
import random
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# tqdm progress bar with safe fallback
try:
    from tqdm import trange
except Exception:
    def trange(*args, **kwargs):
        if len(args) == 1:
            return range(int(args[0]))
        if len(args) == 2:
            return range(int(args[0]), int(args[1]))
        if len(args) == 3:
            return range(int(args[0]), int(args[1]), int(args[2]))
        return range(0)

# Project imports
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.ablation_gnn import Args as TrainArgs

# ===========================
# NSGA helpers (domination, sorting, crowding)
# ===========================

def is_dominated(p: Tuple[float, float], q: Tuple[float, float]) -> bool:
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])


def non_dominated(points: List[Tuple[float, float]]) -> List[int]:
    n = len(points)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if is_dominated(points[i], points[j]):
                keep[i] = False
                break
    return [i for i, k in enumerate(keep) if k]


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


def crowding_distance(front_idx: List[int], objs: List[Tuple[float, float]]) -> Dict[int, float]:
    if not front_idx:
        return {}
    dist = {i: 0.0 for i in front_idx}
    for dim in [0, 1]:
        front_idx_sorted = sorted(front_idx, key=lambda i: objs[i][dim])
        dist[front_idx_sorted[0]] = float('inf')
        dist[front_idx_sorted[-1]] = float('inf')
        min_v = objs[front_idx_sorted[0]][dim]
        max_v = objs[front_idx_sorted[-1]][dim]
        denom = max(max_v - min_v, 1e-12)
        for k in range(1, len(front_idx_sorted) - 1):
            i_prev = front_idx_sorted[k - 1]
            i_next = front_idx_sorted[k + 1]
            dist[front_idx_sorted[k]] += (objs[i_next][dim] - objs[i_prev][dim]) / denom
    return dist

# ===========================
# Genome and decoding
# ===========================

class ScheduleGenome:
    def __init__(self, task_count: int, vm_count: int):
        self.task_count = int(task_count)
        self.vm_count = int(vm_count)
        # priority keys ~ N(0,1)
        self.keys = np.random.randn(self.task_count).astype(np.float32)
        # vm choices uniform [0, V)
        self.vm = np.random.randint(0, self.vm_count, size=(self.task_count,), dtype=np.int32)

    def clone(self) -> 'ScheduleGenome':
        g = ScheduleGenome(self.task_count, self.vm_count)
        g.keys = self.keys.copy()
        g.vm = self.vm.copy()
        return g


def decode_and_evaluate(genome: ScheduleGenome, env: GinAgentWrapper, use_ratios: bool) -> Tuple[float, float]:
    """Schedule by keys over READY tasks and assign VMs with compatibility aware mapping,
    using env.prev_obs (structured). Returns (mk_ratio, ae_ratio).
    """
    obs, _ = env.reset(seed=None)  # env should have already been seeded by caller
    # baselines
    baseline_mk = float(env.prev_obs.makespan()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')
    baseline_active = float(getattr(env.prev_obs, 'energy_consumption', lambda: float('nan'))()) if hasattr(env, 'prev_obs') and env.prev_obs is not None else float('nan')

    def allowed_vms_for_task(o, task: int, vm_count: int) -> np.ndarray:
        """Return boolean mask length V of VMs compatible for a task. If compat not provided, allow all."""
        try:
            comp = getattr(o, 'compatibilities', None)
            if comp is None or getattr(comp, 'numel', lambda: 0)() == 0:
                return np.ones(vm_count, dtype=bool)
            t = comp[0].detach().cpu().numpy().astype(int)
            v = comp[1].detach().cpu().numpy().astype(int)
            m = np.zeros(vm_count, dtype=bool)
            m[np.array([vv for (tt, vv) in zip(t, v) if int(tt) == int(task)], dtype=int) % vm_count] = True
            return m
        except Exception:
            return np.ones(vm_count, dtype=bool)

    done = False
    steps = 0
    info = {}
    while not done and steps < genome.task_count * 10:
        o = env.prev_obs
        # READY tasks from observation
        try:
            ready_vec = getattr(o, 'task_state_ready')
            ready_np = ready_vec.detach().cpu().numpy().astype(bool)
        except Exception:
            # Fallback: consider all tasks potentially schedulable
            ready_np = np.ones(genome.task_count, dtype=bool)

        # pick smallest-key task that has at least one compatible VM
        vm_count = genome.vm_count
        candidate_t = None
        for t in np.where(ready_np)[0]:
            vmask = allowed_vms_for_task(o, int(t), vm_count)
            if vmask.any():
                if candidate_t is None or float(genome.keys[int(t)]) < float(genome.keys[int(candidate_t)]):
                    candidate_t = int(t)
        if candidate_t is None:
            # No valid action; attempt to progress time by taking a harmless action: choose any flat index 0
            action = 0
            obs, reward, done, trunc, info = env.step(action)
            done = bool(done or trunc)
            steps += 1
            continue

        # choose VM: prefer genome choice, otherwise first compatible
        pref_v = int(genome.vm[candidate_t] % vm_count)
        vmask = allowed_vms_for_task(o, candidate_t, vm_count)
        if not vmask[pref_v]:
            compat_vs = np.where(vmask)[0]
            if compat_vs.size > 0:
                pref_v = int(compat_vs[0])
            else:
                # no compatible VMs found; fallback to 0
                pref_v = 0

        # Map to flat action t*V + v
        action = int(candidate_t * vm_count + pref_v)
        obs, reward, done, trunc, info = env.step(action)
        done = bool(done or trunc)
        steps += 1

    mk = float(info.get('makespan', env.prev_obs.makespan()))
    ae = float(info.get('total_energy_active', getattr(env.prev_obs, 'energy_consumption', lambda: mk)()))

    if use_ratios and np.isfinite(baseline_mk) and baseline_mk > 0 and np.isfinite(baseline_active) and baseline_active > 0:
        return float(mk / baseline_mk), float(ae / baseline_active)
    return float(mk), float(ae)

# ===========================
# Variation operators
# ===========================

def one_point_crossover(a: ScheduleGenome, b: ScheduleGenome, key_cx_prob: float = 1.0, vm_cx_prob: float = 1.0) -> Tuple[ScheduleGenome, ScheduleGenome]:
    assert a.task_count == b.task_count and a.vm_count == b.vm_count
    T = a.task_count
    c1 = a.clone(); c2 = b.clone()
    if random.random() < key_cx_prob:
        cut = random.randint(1, T - 1)
        c1.keys[:cut], c2.keys[:cut] = b.keys[:cut].copy(), a.keys[:cut].copy()
    if random.random() < vm_cx_prob:
        cut = random.randint(1, T - 1)
        c1.vm[:cut], c2.vm[:cut] = b.vm[:cut].copy(), a.vm[:cut].copy()
    return c1, c2


def mutate(genome: ScheduleGenome, key_sigma: float, key_ratio: float, vm_flip_prob: float):
    # Gaussian mutation on a fraction of keys
    T = genome.task_count
    mask = (np.random.rand(T) < key_ratio)
    genome.keys[mask] += np.random.randn(mask.sum()).astype(np.float32) * float(key_sigma)
    # Randomly flip VM choice
    flip_mask = (np.random.rand(T) < vm_flip_prob)
    if flip_mask.any():
        genome.vm[flip_mask] = np.random.randint(0, genome.vm_count, size=int(flip_mask.sum()), dtype=np.int32)

# ===========================
# Main
# ===========================

def build_eval_args(base: TrainArgs, args) -> TrainArgs:
    new = dc_replace(base)
    ds = dc_replace(base.dataset)
    if args.dag_method:
        ds = dc_replace(ds, dag_method=args.dag_method)
    if args.gnp_min_n is not None:
        ds = dc_replace(ds, gnp_min_n=int(args.gnp_min_n))
    if args.gnp_max_n is not None:
        ds = dc_replace(ds, gnp_max_n=int(args.gnp_max_n))
    if args.host_count is not None:
        ds = dc_replace(ds, host_count=int(args.host_count))
    if args.vm_count is not None:
        ds = dc_replace(ds, vm_count=int(args.vm_count))
    if args.workflow_count is not None:
        ds = dc_replace(ds, workflow_count=int(args.workflow_count))
    if args.gnp_p is not None:
        ds = dc_replace(ds, gnp_p=float(args.gnp_p))
    new.dataset = ds
    new.device = str(args.device)
    new.test_iterations = 1  # single seed
    return new


def evaluate_population(pop: List[ScheduleGenome], env: GinAgentWrapper, use_ratios: bool) -> List[Tuple[float, float]]:
    objs: List[Tuple[float, float]] = []
    for g in pop:
        objs.append(decode_and_evaluate(g, env, use_ratios))
    return objs


def main():
    ap = argparse.ArgumentParser(description="NSGA-II over real schedules (task order + VM assignment)")
    # GA
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--key-sigma", type=float, default=0.25)
    ap.add_argument("--key-ratio", type=float, default=0.15)
    ap.add_argument("--vm-flip-prob", type=float, default=0.05)
    ap.add_argument("--cx-prob", type=float, default=0.9)
    # Dataset / eval
    ap.add_argument("--dag-method", type=str, default="linear")
    ap.add_argument("--gnp-p", type=float, default=None)
    ap.add_argument("--gnp-min-n", type=int, default=20)
    ap.add_argument("--gnp-max-n", type=int, default=20)
    ap.add_argument("--host-count", type=int, default=4)
    ap.add_argument("--vm-count", type=int, default=10)
    ap.add_argument("--workflow-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345, help="Fixed dataset seed for this run")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default="logs/nsga_schedule_reference")
    ap.add_argument("--use-ratios", action="store_true", default=False, help="If true, output mk_ratio and ae_ratio; otherwise output raw makespan and active energy")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build environment (wrapped)
    base = TrainArgs()
    eval_args = build_eval_args(base, args)
    env = CloudSchedulingGymEnvironment(
        dataset_args=eval_args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    env = GinAgentWrapper(env, constrained_mode=False)
    # seed the environment once for determinism
    _obs, _ = env.reset(seed=int(args.seed))

    # Derive sizes from the first observation if possible
    try:
        T = int(env.prev_obs.task_state_ready.shape[0])
        V = int(env.prev_obs.vm_completion_time.shape[0])
    except Exception:
        T = int(args.gnp_max_n)
        V = int(args.vm_count)

    # Initialize population
    population: List[ScheduleGenome] = [ScheduleGenome(T, V) for _ in range(int(args.population))]

    # Evaluate gen0
    objs = evaluate_population(population, env, bool(args.use_ratios))
    # Log history
    import csv
    with (out_dir / "history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        if args.use_ratios:
            w.writerow(["generation", "idx", "mk_ratio", "ae_ratio"])  # single-seed fitness
        else:
            w.writerow(["generation", "idx", "makespan", "active_energy"])  # single-seed fitness
        for i, (x, y) in enumerate(objs):
            w.writerow([0, i, x, y])

    for gen in trange(1, int(args.generations) + 1, desc="NSGA generations"):
        fronts = fast_non_dominated_sort(objs)
        distances: Dict[int, float] = {}
        for f in fronts:
            distances.update(crowding_distance(f, objs))
        # selection into mating pool
        next_pop: List[ScheduleGenome] = []
        next_objs: List[Tuple[float, float]] = []
        for f in fronts:
            if len(next_pop) + len(f) <= int(args.population):
                for idx in f:
                    next_pop.append(population[idx])
                    next_objs.append(objs[idx])
            else:
                f_sorted = sorted(f, key=lambda i: distances.get(i, 0.0), reverse=True)
                slots = int(args.population) - len(next_pop)
                for idx in f_sorted[:slots]:
                    next_pop.append(population[idx])
                    next_objs.append(objs[idx])
                break
        # variation
        offspring: List[ScheduleGenome] = []
        while len(offspring) + len(next_pop) < int(args.population) * 2:
            # tournament pick
            def tour_pick() -> ScheduleGenome:
                i, j = np.random.randint(0, len(next_pop)), np.random.randint(0, len(next_pop))
                if distances.get(i, 0.0) > distances.get(j, 0.0):
                    return next_pop[i]
                if distances.get(i, 0.0) < distances.get(j, 0.0):
                    return next_pop[j]
                # tie-breaker on simple sum
                return next_pop[i] if (next_objs[i][0] + next_objs[i][1]) <= (next_objs[j][0] + next_objs[j][1]) else next_pop[j]

            p1 = tour_pick().clone()
            p2 = tour_pick().clone()
            if np.random.rand() < float(args.cx_prob):
                c1, c2 = one_point_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            mutate(c1, args.key_sigma, args.key_ratio, args.vm_flip_prob)
            mutate(c2, args.key_sigma, args.key_ratio, args.vm_flip_prob)
            offspring.extend([c1, c2])

        # evaluate offspring (environment uses fixed seed each episode start)
        off_objs = evaluate_population(offspring, env, bool(args.use_ratios))

        # combine
        population = next_pop + offspring
        objs = next_objs + off_objs

        # log and save current reference front
        with (out_dir / "history.csv").open("a", newline="") as f:
            w = csv.writer(f)
            for i, (x, y) in enumerate(objs):
                w.writerow([gen, i, x, y])
        nd_idx = non_dominated(objs)
        with (out_dir / "reference_front.csv").open("w", newline="") as f:
            w = csv.writer(f)
            if args.use_ratios:
                w.writerow(["mk_ratio", "ae_ratio"]) 
            else:
                w.writerow(["makespan", "active_energy"]) 
            for i in sorted(nd_idx, key=lambda k: (objs[k][0], objs[k][1])):
                w.writerow([objs[i][0], objs[i][1]])

    print(f"[nsga-schedule] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
