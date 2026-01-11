import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.core.env.action import EnvAction


LONG_CP_CFG = "data/rl_configs/train_long_cp_p08_seeds.json"
WIDE_CFG = "data/rl_configs/train_wide_p005_seeds.json"
OUT_DIR = "logs/dags"
OUT_PNG = os.path.join(OUT_DIR, "state_visitation_pi_ei.png")


def _load_eval(path: str) -> Tuple[List[int], dict]:
    with open(path, "r") as f:
        cfg = json.load(f)
    seeds: List[int] = list(cfg.get("eval", {}).get("seeds", []))
    ds: dict = dict(cfg.get("eval", {}).get("dataset", {}))
    return seeds, ds


def _mk_dataset(seed: int, ds: dict):
    return generate_dataset(
        seed=seed,
        host_count=int(ds.get("host_count", 10)),
        vm_count=int(ds.get("vm_count", 10)),
        max_memory_gb=int(ds.get("max_memory_gb", 128)),
        min_cpu_speed_mips=int(ds.get("min_cpu_speed", 500)),
        max_cpu_speed_mips=int(ds.get("max_cpu_speed", 5000)),
        workflow_count=int(ds.get("workflow_count", 1)),
        dag_method=str(ds.get("dag_method", "gnp")),
        gnp_min_n=int(ds.get("gnp_min_n", 12)),
        gnp_max_n=int(ds.get("gnp_max_n", 24)),
        task_length_dist=str(ds.get("task_length_dist", "normal")),
        min_task_length=int(ds.get("min_task_length", 500)),
        max_task_length=int(ds.get("max_task_length", 100000)),
        task_arrival=str(ds.get("task_arrival", "static")),
        arrival_rate=float(ds.get("arrival_rate", 3)),
        vm_rng_seed=0,
        gnp_p=float(ds.get("gnp_p")) if ds.get("gnp_p") is not None else None,
    )


def _topo_layers(children: Dict[int, List[int]]) -> List[List[int]]:
    indeg: Dict[int, int] = {u: 0 for u in children}
    for u, vs in children.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    frontier: List[int] = [u for u in children if indeg.get(u, 0) == 0]
    layers: List[List[int]] = []
    seen: set[int] = set()
    while frontier:
        cur = list(frontier)
        layers.append(cur)
        frontier = []
        for u in cur:
            if u in seen:
                continue
            seen.add(u)
            for v in children.get(u, []):
                indeg[v] = indeg.get(v, 0) - 1
                if indeg[v] == 0:
                    frontier.append(v)
    if not layers:
        return [list(children.keys())]
    return layers


def _max_width_for_workflow(dataset) -> int:
    # Single workflow per dataset
    wf = dataset.workflows[0]
    ch: Dict[int, List[int]] = {int(t.id): list(t.child_ids) for t in wf.tasks}
    layers = _topo_layers(ch)
    return max((len(L) for L in layers), default=1)


def _ready_tasks(env: CloudSchedulingGymEnvironment) -> List[int]:
    # Exclude dummy start (0) and dummy end (T-1)
    mask = (env._is_ready_mask & (~env._is_assigned_mask)).copy()
    mask[0] = False
    mask[len(mask) - 1] = False
    return np.nonzero(mask)[0].tolist()


def _energy_intensity_index(env: CloudSchedulingGymEnvironment) -> float:
    # Aggregate normalized active power: sum_v ( (peak - idle) * u_v ) / sum_v (peak - idle )
    total_span = 0.0
    active = 0.0
    tnow = float(env.state.current_time)
    for vm_id, vm_state in enumerate(env.state.vm_states):
        vstat = env.state.static_state.vms[vm_id]
        span = float(vstat.host_power_peak_watt - vstat.host_power_idle_watt)
        total_span += max(0.0, span)
        total_cores = max(1, int(vstat.cpu_cores))
        used_cores = 0
        for (t_id, t_comp, _mem, cores) in (vm_state.active_tasks or []):
            t_state = env.state.task_states[t_id]
            t_start = t_state.start_time
            if t_start is None:
                continue
            if (t_start <= tnow + 1e-9) and (tnow < t_comp - 1e-9):
                used_cores += int(cores)
        u = min(1.0, max(0.0, used_cores / float(total_cores)))
        active += max(0.0, span) * u
    if total_span <= 0:
        return 0.0
    return float(active / total_span)


def _earliest_feasible_on_vm(env: CloudSchedulingGymEnvironment, vm_id: int, t_id: int, t0: float) -> float:
    vm_total_mem_i = int(env._vm_total_mem[vm_id])
    vm_total_cores_i = int(env._vm_total_cores[vm_id])
    req_mem_i = int(env._task_req_mem[t_id])
    req_cores_i = int(env._task_req_cores[t_id])
    existing = env.state.vm_states[vm_id].active_tasks or []
    events_i: List[Tuple[float, int, int]] = []
    used_mem_i = 0
    used_cores_i = 0
    for (et_id, et_comp, mem, cores) in existing:
        t_s = env.state.task_states[et_id]
        s = t_s.start_time if t_s.start_time is not None else env.state.current_time
        events_i.append((s, +int(mem), +int(cores)))
        events_i.append((float(et_comp), -int(mem), -int(cores)))
    events_i.sort(key=lambda x: x[0])
    for (ev_t, dm, dc) in events_i:
        if ev_t <= t0 + 1e-9:
            used_mem_i += dm
            used_cores_i += dc
    candidate = float(t0)
    times_i = [candidate] + [t for (t, _dm, _dc) in events_i if t > candidate + 1e-9]
    for t in times_i:
        if t > candidate + 1e-9:
            for (ev_t, dm, dc) in events_i:
                if candidate - 1e-9 < ev_t <= t + 1e-9:
                    used_mem_i += dm
                    used_cores_i += dc
        candidate = float(t)
        if used_mem_i + req_mem_i <= vm_total_mem_i + 1e-9 and used_cores_i + req_cores_i <= vm_total_cores_i + 1e-9:
            return candidate
    return times_i[-1] if times_i else candidate


def _simulate_collect_points(dataset) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    env = CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=False, dataset_episode_mode="single")
    obs, _ = env.reset()
    # Precompute DAG width
    width = max(1, _max_width_for_workflow(dataset))

    pis: List[float] = []
    eis: List[float] = []
    traj: List[Tuple[float, float]] = []

    # Loop until dummy end is assigned
    T = len(env.state.task_states)
    final_id = T - 1
    step_guard = 0
    while env.state.task_states[final_id].assigned_vm_id is None and step_guard < 10000:
        step_guard += 1
        # Record state metrics BEFORE scheduling the next task
        ready = _ready_tasks(env)
        pi = float(len(ready)) / float(width)
        ei = _energy_intensity_index(env)
        pis.append(pi)
        eis.append(ei)
        traj.append((pi, ei))

        if not ready:
            # Shouldn't happen often; break to avoid infinite loops
            break
        # Greedy ECT: for each ready task, choose VM with earliest completion (start feasible + proc time)
        best = None
        best_end = float("inf")
        for t_id in ready:
            # Parent ready time
            prt = float(env.state.current_time)
            for p_id in env._parents[t_id]:
                p_comp = env.state.task_states[p_id].completion_time
                if p_comp is not None:
                    prt = max(prt, float(p_comp))
            # Iterate compatible VMs
            for vm_id in np.nonzero(env._compat_bool[t_id])[0].tolist():
                est = _earliest_feasible_on_vm(env, vm_id, t_id, prt)
                proc = env.state.static_state.tasks[t_id].length / max(1e-9, env.state.static_state.vms[vm_id].cpu_speed_mips)
                end_t = est + proc
                if end_t < best_end - 1e-9:
                    best_end = end_t
                    best = (t_id, vm_id)
        if best is None:
            # Fallback: pick arbitrary compatible pair
            t_id = ready[0]
            vm_id = int(np.nonzero(env._compat_bool[t_id])[0][0])
            best = (t_id, vm_id)
        action = EnvAction(task_id=int(best[0]), vm_id=int(best[1]))
        obs, rew, term, trunc, info = env.step(action)

    return pis, eis, traj


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    long_seeds, long_ds = _load_eval(LONG_CP_CFG)
    wide_seeds, wide_ds = _load_eval(WIDE_CFG)

    # Use 10 seeds each (matches your previous plots); increase if you want smoother densities
    long_seeds = long_seeds[:10]
    wide_seeds = wide_seeds[:10]

    data = {"style": [], "pi": [], "ei": []}
    traj_long: List[Tuple[float, float]] = []
    traj_wide: List[Tuple[float, float]] = []

    for s in long_seeds:
        ds = _mk_dataset(int(s), long_ds)
        pis, eis, traj = _simulate_collect_points(ds)
        data["style"] += ["long_cp"] * len(pis)
        data["pi"] += pis
        data["ei"] += eis
        if not traj_long:
            traj_long = traj

    for s in wide_seeds:
        ds = _mk_dataset(int(s), wide_ds)
        pis, eis, traj = _simulate_collect_points(ds)
        data["style"] += ["wide"] * len(pis)
        data["pi"] += pis
        data["ei"] += eis
        if not traj_wide:
            traj_wide = traj

    # Plot densities and example trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200, sharex=True, sharey=True)

    # Panel for long_cp
    ax = axes[0]
    sns.kdeplot(x=np.array([p for (p, s) in zip(data["pi"], data["style"]) if s == "long_cp"]),
                y=np.array([e for (e, s) in zip(data["ei"], data["style"]) if s == "long_cp"]),
                fill=True, levels=15, cmap="Greens", thresh=0.05, ax=ax)
    ax.plot([p for (p, _e) in traj_long], [e for (_p, e) in traj_long], color="#E67E22", linewidth=1.5, alpha=0.9)
    ax.set_title("LongCP — state visitation")
    ax.set_xlabel("Parallelism Index (ready / width)")
    ax.set_ylabel("Energy Intensity Index (active power rate)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Panel for wide
    ax = axes[1]
    sns.kdeplot(x=np.array([p for (p, s) in zip(data["pi"], data["style"]) if s == "wide"]),
                y=np.array([e for (e, s) in zip(data["ei"], data["style"]) if s == "wide"]),
                fill=True, levels=15, cmap="Greens", thresh=0.05, ax=ax)
    ax.plot([p for (p, _e) in traj_wide], [e for (_p, e) in traj_wide], color="#E67E22", linewidth=1.5, alpha=0.9)
    ax.set_title("Wide — state visitation")
    ax.set_xlabel("Parallelism Index (ready / width)")
    ax.set_ylabel("Energy Intensity Index (active power rate)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
