from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict

import numpy as np

# Ensure project root on path (similar to train.py/eval_robustness.py)
this_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(this_dir, ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scheduler.dataset_generator.core.models import Dataset, Workflow, Task, VmAssignment
from scheduler.dataset_generator.solvers.cp_sat_solver import solve_cp_sat, solve_cp_sat_energy
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.robustness.metrics import pareto_non_dominated


Point = Tuple[float, float]  # (makespan, total_energy)


@dataclass
class RefOptions:
    cp_sat_timeout_s: Optional[int] = 20
    include_energy_greedy: bool = False
    include_energy_assign_opt: bool = False
    include_theta_greedy: bool = False
    theta_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11))
    # Optional diagnostics (populated by caller for logging only)
    alpha_cpu: Optional[float] = None
    alpha_mem: Optional[float] = None


def compute_objectives_via_env(dataset: Dataset, assignments: Optional[List[VmAssignment]] = None) -> Point:
    """
    Compute (makespan, total_energy) by executing the dataset in the environment.
    If assignments is provided, we attempt to follow that order by always picking the next ready
    task for each VM according to the assignment order.
    """
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=0)

    # Build per-VM schedule order if assignments provided
    per_vm_order: Dict[int, List[int]] = {}
    per_vm_idx: Dict[int, int] = {}
    if assignments is not None:
        for a in sorted(assignments, key=lambda x: (x.vm_id, x.start_time)):
            per_vm_order.setdefault(a.vm_id, []).append(a.task_id)
        per_vm_idx = {vm_id: 0 for vm_id in per_vm_order.keys()}

    # Helper to choose next action
    def choose_action() -> int:
        # Decode observation to know ready tasks and VM counts
        vm_count = len(env.prev_obs.vm_observations)
        compat_set = set(env.prev_obs.compatibilities)
        # Strategy:
        # - If we have per-VM order, try to schedule the first ready task that matches the next planned task on that VM.
        # - If the CP-SAT VM is incompatible under env constraints, remap to the first compatible VM for that task.
        if assignments is not None:
            for vm_id, order in per_vm_order.items():
                idx = per_vm_idx.get(vm_id, 0)
                if idx >= len(order):
                    continue
                target_tid = order[idx]
                # Check readiness
                tstate = env.prev_obs.task_observations[target_tid]
                if not (tstate.is_ready and tstate.assigned_vm_id is None):
                    continue
                # If CP-SAT-chosen VM is compatible, use it; otherwise map to first compatible VM
                if (target_tid, vm_id) in compat_set:
                    return int(target_tid * vm_count + vm_id)
                # Remap to first compatible VM for this task
                for v2 in range(vm_count):
                    if (target_tid, v2) in compat_set:
                        return int(target_tid * vm_count + v2)
                # If no compatible VM at all, skip to fallback below
        # Fallback: pick any ready, not-assigned task on first compatible VM
        from scheduler.rl_model.core.utils.helpers import is_suitable
        for t_id, t in enumerate(env.prev_obs.task_observations):
            if t_id in (0, len(env.prev_obs.task_observations)-1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue
            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                if (t_id, vm_id) in compat_set and is_suitable(vm, t):
                    return int(t_id * vm_count + vm_id)
        # If nothing found, pick 0
        return 0

    final_info = None
    while True:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        # If we successfully scheduled a per-VM next task on vm_id, advance pointer
        if assignments is not None and isinstance(info, dict):
            # We can detect last scheduled mapping via env.prev_obs state
            pass  # not strictly needed; order is greedy each step
        if term or trunc:
            final_info = info
            break

    makespan = env.prev_obs.makespan()
    total_energy = float(final_info.get("total_energy", env.prev_obs.energy_consumption())) if isinstance(final_info, dict) else env.prev_obs.energy_consumption()
    env.close()
    return float(makespan), float(total_energy)


def energy_greedy_schedule(dataset: Dataset) -> Point:
    """Constructively schedule ready tasks to the compatible VM with minimal active energy rate (proxy)."""
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=0)

    from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable

    def choose_action() -> int:
        vm_count = len(env.prev_obs.vm_observations)
        best = None
        best_idx = 0
        # Get compatibility list from observation
        compat_set = set(env.prev_obs.compatibilities)
        for t_id, t in enumerate(env.prev_obs.task_observations):
            if t_id in (0, len(env.prev_obs.task_observations)-1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue
            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                # Check compatibility using the observation's compatibility list
                if (t_id, vm_id) not in compat_set:
                    continue
                rate = active_energy_consumption_per_mi(vm)
                idx = int(t_id * vm_count + vm_id)
                if best is None or rate < best:
                    best = rate
                    best_idx = idx
        return best_idx

    final_info = None
    while True:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            final_info = info
            break
    makespan = env.prev_obs.makespan()
    total_energy = float(final_info.get("total_energy", env.prev_obs.energy_consumption())) if isinstance(final_info, dict) else env.prev_obs.energy_consumption()
    env.close()
    return float(makespan), float(total_energy)


def energy_assignment_min_rate(dataset: Dataset) -> Point:
    """
    Build a per-task VM assignment by minimizing energy-per-MI per task (proxy for energy-optimal).
    Then simulate in the env to obtain actual (makespan, energy).
    """
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=0)

    from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi

    # Precompute energy rate per VM
    vm_rates = [active_energy_consumption_per_mi(vm_obs) for vm_obs in env.prev_obs.vm_observations]
    compat_set = set(env.prev_obs.compatibilities)

    # Assign each non-dummy task to the compatible VM with minimum rate
    per_vm_order: Dict[int, List[int]] = {}
    for t_id, t in enumerate(env.prev_obs.task_observations):
        if t_id in (0, len(env.prev_obs.task_observations) - 1):
            continue
        # Candidates
        candidates = [vm_id for vm_id in range(len(vm_rates)) if (t_id, vm_id) in compat_set]
        if not candidates:
            # Fallback: skip; env will handle infeasible via its relaxations
            continue
        best_vm = min(candidates, key=lambda vid: vm_rates[vid])
        per_vm_order.setdefault(best_vm, []).append(t_id)

    # Convert to VmAssignment list with increasing order indices as pseudo times
    assigns: List[VmAssignment] = []
    for vm_id, tids in per_vm_order.items():
        for order_idx, tid in enumerate(tids):
            assigns.append(VmAssignment(workflow_id=0, task_id=tid, vm_id=vm_id, start_time=order_idx, end_time=order_idx + 1))

    env.close()
    return compute_objectives_via_env(dataset, assignments=assigns)


def theta_greedy_schedule(dataset: Dataset, theta: float) -> Point:
    """
    Weighted greedy: at each decision pick (task, vm) minimizing
    theta * delta_makespan + (1-theta) * delta_energy_proxy.
    We approximate deltas via VM completion time increase and energy rate.
    """
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=0)

    from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable

    def choose_action() -> int:
        vm_count = len(env.prev_obs.vm_observations)
        best = None
        best_idx = 0
        cur_mk = env.prev_obs.makespan()
        # Get compatibility list from observation
        compat_set = set(env.prev_obs.compatibilities)
        for t_id, t in enumerate(env.prev_obs.task_observations):
            if t_id in (0, len(env.prev_obs.task_observations)-1):
                continue
            if t.assigned_vm_id is not None or not t.is_ready:
                continue
            for vm_id, vm in enumerate(env.prev_obs.vm_observations):
                # Check compatibility using the observation's compatibility list
                if (t_id, vm_id) not in compat_set:
                    continue
                # proxy deltas
                vm_comp = vm.completion_time
                task_runtime = t.length / max(1e-9, vm.cpu_speed_mips)
                new_vm_comp = max(vm_comp, env.prev_obs.current_time) + task_runtime
                delta_mk = max(0.0, new_vm_comp - cur_mk)
                delta_energy = active_energy_consumption_per_mi(vm) * t.length
                score = theta * delta_mk + (1.0 - theta) * delta_energy
                idx = int(t_id * vm_count + vm_id)
                if best is None or score < best:
                    best = score
                    best_idx = idx
        return best_idx

    final_info = None
    while True:
        a = choose_action()
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            final_info = info
            break
    makespan = env.prev_obs.makespan()
    total_energy = float(final_info.get("total_energy", env.prev_obs.energy_consumption())) if isinstance(final_info, dict) else env.prev_obs.energy_consumption()
    env.close()
    return float(makespan), float(total_energy)


def build_reference_set(dataset: Dataset, options: RefOptions = RefOptions()) -> List[Point]:
    """Build a multi-objective reference set by union of diverse solvers/heuristics."""
    ref: List[Point] = []
    # 1) CP-SAT makespan-optimal schedule
    print(f'building reference set ')
    try:
        print(f'building makespan-optimal reference set for alpha_cpu={options.alpha_cpu}, alpha_mem={options.alpha_mem}')
        is_opt, assigns = solve_cp_sat(dataset.workflows, dataset.vms, timeout=options.cp_sat_timeout_s)
        if not assigns:
            print('  CP-SAT (makespan) returned no assignments')
        else:
            mk, en = compute_objectives_via_env(dataset, assignments=assigns)
            ref.append((mk, en))
    except Exception as e:
        print(f'failed to build makespan-optimal reference set: {type(e).__name__}: {e}')
        pass
    # 1b) CP-SAT energy-optimal schedule
    try:
        print(f'building energy-optimal reference set ')
        is_opt_e, assigns_e = solve_cp_sat_energy(dataset.workflows, dataset.vms, dataset.hosts, timeout=options.cp_sat_timeout_s)
        if not assigns_e:
            print('  CP-SAT (energy) returned no assignments')
        else:
            mk_e, en_e = compute_objectives_via_env(dataset, assignments=assigns_e)
            ref.append((mk_e, en_e))
    except Exception as e:
        print(f'failed to build energy-optimal reference set: {type(e).__name__}: {e}')
        pass
    return pareto_non_dominated(ref)
