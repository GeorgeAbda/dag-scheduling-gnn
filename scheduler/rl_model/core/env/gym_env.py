import copy
import time
import numpy as np
import logging
import random
from typing import Any, Callable
from bisect import bisect_left, insort

import gymnasium as gym

from scheduler.config.settings import MAX_TRAINING_DS_SEED
from scheduler.dataset_generator.core.gen_dataset import (
    generate_dataset,
    generate_dataset_long_cp_queue_free,
    generate_dataset_wide_queue_free,
)
from scheduler.dataset_generator.core.models import Dataset
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation

from scheduler.rl_model.core.env.state import EnvState, TaskState, VmState, StaticState
from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable
from scheduler.rl_model.core.utils.task_mapper import TaskMapper


global_reset_counter = 0


class CloudSchedulingGymEnvironment(gym.Env):
    dataset_generator: Callable[[int | None], Dataset]
    state: EnvState | None = None

    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset: Dataset | None = None, dataset_args: DatasetArgs | None = None, collect_timelines: bool = False, compute_metrics: bool = False, profile: bool = False, feasibility_mode: str = "optimized", fixed_env_seed: bool = False, dataset_episode_mode: str = "all"):
        super().__init__()
        if dataset is not None:
            assert dataset_args is None, "When dataset is passed, dataset_arg must be None"
            self.dataset_generator = lambda _: dataset
        if dataset_args is not None:
            assert dataset is None, "When dataset_arg is passed, dataset must be None"
            self.dataset_generator = lambda seed: self.gen_dataset(seed, dataset_args)
        # Deterministic seeding support: use DatasetArgs.seed as a base when reset() is called without an explicit seed
        self._base_seed: int | None = None
        self._episode_counter: int = 0
        try:
            # DatasetArgs may provide a seed; store it to derive deterministic seeds per episode
            if dataset_args is not None and getattr(dataset_args, "seed", None) is not None:
                self._base_seed = int(getattr(dataset_args, "seed"))
        except Exception:
            # If DatasetArgs has no seed, remain None and fall back to 0
            self._base_seed = None
        # Whether to keep the dataset seed fixed across episodes for this env
        self._fixed_env_seed: bool = bool(fixed_env_seed)
        # Whether to build per-VM timelines/capacities in the episode-end info
        self.collect_timelines: bool = collect_timelines
        # Whether to compute testing-only metrics (refined bottlenecks, per-VM energy breakdown, CP breakdown, assignability series)
        self.compute_metrics: bool = compute_metrics
        # Profiling toggle and timers (accumulated seconds)
        self.profile: bool = profile
        self._timers: dict[str, float] = {"ready": 0.0, "feasible": 0.0, "end_energy": 0.0, "end_cp": 0.0}
        # Debug reconciliation to ensure active_events equivalence with legacy rebuild
        self._reconcile_events: bool = False
        # Feasibility method: "optimized" (default) uses active_events; "legacy" rebuilds events each decision
        assert feasibility_mode in ("optimized", "legacy"), "feasibility_mode must be 'optimized' or 'legacy'"
        self.feasibility_mode = feasibility_mode
        # Cached static arrays (populated on reset)
        self._task_req_mem = None
        self._task_req_cores = None
        self._vm_total_mem = None
        self._vm_total_cores = None
        self._vm_speed_mips = None
        # Dataset selection mode when a Dataset with multiple workflows is provided
        # Modes: "all" -> use all workflows together; "single" -> one workflow per episode (rotating)
        self._dataset_episode_mode: str = str(dataset_episode_mode or "all")
        # One-time debug print toggle for dataset_json single-episode mode
        self._dbg_workflow_printed: bool = False
        # Cached dynamic masks (populated on reset, updated on step)
        self._is_ready_mask = None      # shape (T,)
        self._is_assigned_mask = None   # shape (T,)

    def reset_timers(self) -> None:
        self._timers = {"ready": 0.0, "feasible": 0.0, "end_energy": 0.0, "end_cp": 0.0}

    def get_timers(self) -> dict[str, float]:
        return dict(self._timers)

    # Internal helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _rebuild_vm_events_from_active_tasks(self, vm_id: int) -> list[tuple[float, int, int]]:
        events: list[tuple[float, int, int]] = []
        vm_state = self.state.vm_states[vm_id]
        for (t_id, t_comp, mem, cores) in (vm_state.active_tasks or []):
            t_s = self.state.task_states[t_id]
            s = t_s.start_time if t_s.start_time is not None else self.state.current_time
            events.append((s, +mem, +cores))
            events.append((t_comp, -mem, -cores))
        events.sort(key=lambda x: x[0])
        return events

    def _events_equal(self, a: list[tuple[float, int, int]], b: list[tuple[float, int, int]], atol: float = 1e-9) -> bool:
        if len(a) != len(b):
            return False
        for (t1, dm1, dc1), (t2, dm2, dc2) in zip(a, b):
            if abs(t1 - t2) > atol or dm1 != dm2 or dc1 != dc2:
                return False
        return True

    def _reconcile_vm_events(self, vm_id: int) -> None:
        rebuilt = self._rebuild_vm_events_from_active_tasks(vm_id)
        current = self.state.vm_states[vm_id].active_events or []
        if not self._events_equal(rebuilt, current):
            self.state.vm_states[vm_id].active_events = rebuilt

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[EnvObservation, dict[str, Any]]:
        global global_reset_counter

        super().reset(seed=seed, options=options)
        global_reset_counter += 1

        # If no explicit seed was provided, derive a deterministic seed.
        # If fixed_env_seed=True, reuse base seed every episode (no increment).
        if seed is None:
            base = int(self._base_seed) if self._base_seed is not None else 0
            if self._fixed_env_seed:
                seed = base
            else:
                seed = base + self._episode_counter
                self._episode_counter += 1

        # Generate/select dataset according to (possibly derived) seed
        dataset = self.dataset_generator(seed)
        try:
            # If using a fixed Dataset with multiple workflows, optionally select a single workflow per episode
            if hasattr(dataset, "workflows") and isinstance(dataset.workflows, list) and len(dataset.workflows) > 1:
                if str(getattr(self, "_dataset_episode_mode", "all")).lower() == "single":
                    n_wf = len(dataset.workflows)
                    # Base on episode counter (rotates when seed is None on auto-resets)
                    base_idx = 0 if self._fixed_env_seed else (self._episode_counter % n_wf)
                    # Add seed-based offset if a seed was provided (to diversify initial vector env workers)
                    seed_offset = int(seed) % n_wf if (seed is not None) else 0
                    idx = (base_idx + seed_offset) % n_wf
                    from scheduler.dataset_generator.core.models import Dataset as _DsCls
                    dataset = _DsCls(workflows=[dataset.workflows[int(idx)]], vms=dataset.vms, hosts=dataset.hosts)
        except Exception:
            pass

        # Optional one-time debug of selected workflow characteristics (first episode)
        try:
            if (not self._dbg_workflow_printed) and str(getattr(self, "_dataset_episode_mode", "all")).lower() == "single":
                wf0 = dataset.workflows[0] if hasattr(dataset, "workflows") and len(dataset.workflows) > 0 else None
                if wf0 is not None and hasattr(wf0, "tasks"):
                    n_tasks = len(wf0.tasks)
                    n_edges = 0
                    min_mem = 1 << 30
                    max_mem = 0
                    min_cores = 1 << 30
                    max_cores = 0
                    for t in wf0.tasks:
                        n_edges += len(getattr(t, "child_ids", []) or [])
                        try:
                            m = int(getattr(t, "req_memory_mb", 0)); c = int(getattr(t, "req_cpu_cores", 0))
                            min_mem = min(min_mem, m); max_mem = max(max_mem, m)
                            min_cores = min(min_cores, c); max_cores = max(max_cores, c)
                        except Exception:
                            pass
                    print(f"[dataset][episode=1] workflow_id={getattr(wf0, 'id', 0)} tasks={n_tasks} edges={n_edges} req_mem_mb=[{min_mem},{max_mem}] req_cores=[{min_cores},{max_cores}]")
                    self._dbg_workflow_printed = True
        except Exception:
            pass

        # Map the tasks and VMs from the dataset to the required format
        host_map = {host.id: host for host in dataset.hosts}
        vms = [VmDto.from_vm(vm, host_map[vm.host_id]) for vm in dataset.vms]
        tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
        task_mapper = TaskMapper(tasks)
        mapped_tasks = task_mapper.map_tasks()

        # Sanity check - we should be able to use index and id interchangeably
        for i, task in enumerate(mapped_tasks):
            assert task.id == i, f"Sanity Check Failed: Task ID mismatch, {task.id} != {i}"
        for i, vm in enumerate(vms):
            assert vm.id == i, f"Sanity Check Failed: VM ID mismatch, {vm.id} != {i}"

        # Initial states of tasks and VMs
        task_states = [TaskState() for _ in mapped_tasks]
        vm_states = [VmState(assigned_task_id=None, completion_time=0, used_memory_mb=0, used_cpu_cores=0, active_tasks=[]) for _ in vms]
        # Initialize per-VM incremental structures
        for vs in vm_states:
            vs.active_events = []
            vs.used_mem_snapshot = None
            vs.used_cores_snapshot = None
            vs.snapshot_time = None

        # Dummy task is scheduled initially
        task_states[0].assigned_vm_id = 0
        for task_id in mapped_tasks[0].child_ids:
            task_states[task_id].is_ready = True
        # Create compatibility edges (task, vm) and boolean matrix cache
        T = len(mapped_tasks)
        V = len(vms)
        compatibilities: list[tuple[int, int]] = []
        compat_bool = np.zeros((T, V), dtype=bool)
        for task_id in range(T):
            task_compatible_vms = []
            for vm_id in range(V):
                if is_suitable(vms[vm_id], mapped_tasks[task_id]):
                    task_compatible_vms.append((task_id, vm_id))
            if not task_compatible_vms:
                eligible = [(task_id, vm_id) for vm_id in range(V) if vms[vm_id].memory_mb >= mapped_tasks[task_id].req_memory_mb]
                print(f'  Warning: Task {task_id} has no compatible VMs, relaxing CPU requirement!!!!!!!!!')
                if not eligible:
                    eligible = [(task_id, 0)]
                task_compatible_vms = eligible
                print(f'  Warning: Task {task_id} has no compatible VMs, adding dummies')
            compatibilities.extend(task_compatible_vms)
            for _t, _v in task_compatible_vms:
                compat_bool[_t, _v] = True
        # Create dependencies from parent->child relations
        dependencies = set(
            (task_id, child_id)
            for task_id in range(len(mapped_tasks))
            for child_id in mapped_tasks[task_id].child_ids
        )
        # Cache parents, children, and indegrees
        parents: list[list[int]] = [[] for _ in range(T)]
        children: list[list[int]] = [[] for _ in range(T)]
        indegree = np.zeros(T, dtype=np.int32)
        for u, v in dependencies:
            children[u].append(v)
            parents[v].append(u)
            indegree[v] += 1

        # Map to the state
        self.state = EnvState(
            static_state=StaticState(
                task_mapper=task_mapper,
                tasks=mapped_tasks,
                vms=vms,
                compatibilities=compatibilities,
            ),
            task_states=task_states,
            vm_states=vm_states,
            task_dependencies=dependencies,
            current_time=0.0,
        )
        # Attach caches to the instance (not to state to avoid serialization bloat)
        self._compat_bool = compat_bool
        self._parents = parents
        self._children = children
        self._indegree = indegree
        # Cache static per-task and per-VM arrays for fast numeric access
        self._task_req_mem = np.asarray([t.req_memory_mb for t in mapped_tasks], dtype=np.int32)
        self._task_req_cores = np.asarray([t.req_cpu_cores for t in mapped_tasks], dtype=np.int32)
        self._vm_total_mem = np.asarray([v.memory_mb for v in vms], dtype=np.int32)
        self._vm_total_cores = np.asarray([v.cpu_cores for v in vms], dtype=np.int32)
        self._vm_speed_mips = np.asarray([v.cpu_speed_mips for v in vms], dtype=np.float64)
        # Initialize dynamic masks
        Tn = T
        self._is_assigned_mask = np.zeros(Tn, dtype=bool)
        self._is_ready_mask = np.zeros(Tn, dtype=bool)
        # Ready mask from TaskState list
        for tid, ts in enumerate(task_states):
            self._is_ready_mask[tid] = bool(ts.is_ready)
        # Mark dummy start as assigned and not-ready, end as not-ready initially
        self._is_assigned_mask[0] = True
        self._is_ready_mask[0] = False

        # Initialize bottleneck metrics for this episode
        self._metrics = {
            "decision_steps": 0,                 # number of scheduling decisions
            "bottleneck_steps": 0,              # steps where at least one ready task could not start immediately
            "sum_ready_tasks": 0,               # total ready tasks observed at decision times
            "sum_bottleneck_ready_tasks": 0,    # total ready tasks that were blocked at decision times
            "cumulative_wait_time": 0.0,        # sum over actions of (start_time - available_time) when > 0
            # Refined metrics based on parent-ready time and earliest feasible start across compatible VMs
            "refined_bottleneck_steps": 0,
            "sum_ready_tasks_refined": 0,
            "sum_blocked_ready_tasks_refined": 0,
        }

        # Initialize optional assignability time series for plotting (first episode use)
        if getattr(self, "collect_timelines", False):
            self._assign_series = {
                "t": [],                 # decision times
                "ready": [],             # number of ready tasks
                "schedulable": [],       # number of ready tasks that can start now on at least one compatible VM
            }

        return EnvObservation(self.state, is_wait=False), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: EnvAction) -> tuple[EnvObservation, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "State must be initialized"

        # Waiting logic removed: the agent must choose a VM; start time will be the earliest time when enough memory is available on that VM

        # Check validity of the action
        penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.state.task_states)
        if not (0 <= action.task_id < len(self.state.task_states)):
            print(f'Invalid action: task_id {action.task_id} out of range')
            return EnvObservation(self.state, is_wait=False), penalty, True, False, {"error": f"{action}: Invalid task (out of range)"}
        if self.state.task_states[action.task_id].assigned_vm_id is not None:
            print(f'Invalid action: task {action.task_id} already assigned to VM {self.state.task_states[action.task_id].assigned_vm_id}')
            return EnvObservation(self.state, is_wait=False), penalty, True, False, {"error": f"{action}: Already scheduled task"}
        if not self.state.task_states[action.task_id].is_ready:
            print(f'Invalid action: task {action.task_id} not ready')
            return EnvObservation(self.state, is_wait=False), penalty, True, False, {"error": f"{action}: Not ready task"}
        if not self._compat_bool[action.task_id, action.vm_id]:

            # Detailed diagnostics for incompatibility
            try:
                t_static = self.state.static_state.tasks[action.task_id]
                v_static = self.state.static_state.vms[action.vm_id]
                print(
                    f"Incompatible action: {action} not in {self.state.static_state.compatibilities}\n"
                    f"  Task requirements -> cores={t_static.req_cpu_cores}, mem_mb={t_static.req_memory_mb}\n"
                    f"  VM capacity      -> cores={v_static.cpu_cores}, mem_mb={v_static.memory_mb}"

                )
                raise Exception ('Incompatible action: {action} not in {self.state.static_state.compatibilities')

            except Exception:
                # Fallback to original message if any attribute is missing
                print(f'Incompatible action: {action} not in {self.state.static_state.compatibilities}')
                raise Exception ('Incompatible action: {action} not in {self.state.static_state.compatibilities')

            return EnvObservation(self.state, is_wait=False), penalty, True, False, {"error": f"{action}: Task/VM are not compatible"}
        # Use cached relations
        child_task_ids = self._children[action.task_id]
        parent_task_ids = self._parents[action.task_id]
        processing_time = (
            self.state.static_state.tasks[action.task_id].length
            / self.state.static_state.vms[action.vm_id].cpu_speed_mips
        )

        # Avoid deep copies: mutate in place
        new_task_states = self.state.task_states
        new_vm_states = self.state.vm_states

        # Update scheduled states
        new_task_states[action.task_id].assigned_vm_id = action.vm_id
        new_vm_states[action.vm_id].assigned_task_id = action.task_id
        # Update ready states using new state
        new_task_states[action.task_id].is_ready = False
        for child_id in child_task_ids:
            # decrease indegree and mark ready when all parents assigned
            self._indegree[child_id] = max(0, int(self._indegree[child_id]) - 1)
            new_task_states[child_id].is_ready = (self._indegree[child_id] == 0)

        # Update completion times
        # Earliest parent-ready time
        parent_ready_time = self.state.current_time
        for parent_id in parent_task_ids:
            parent_ready_time = max(parent_ready_time, self.state.task_states[parent_id].completion_time)
        # Overlap diagnostic for the chosen VM
        cur_active = [(t, c) for (t, c, _m, _cpu) in (new_vm_states[action.vm_id].active_tasks or [])]
        logging.debug(
            f"[SCHEDULE] Scheduling task {action.task_id} on VM {action.vm_id} at current_time={self.state.current_time:.4f}; "
            f"parents={parent_task_ids}, VM_active={cur_active}"
        )
        # Memory-aware earliest start time on the chosen VM
        req_mem = self.state.static_state.tasks[action.task_id].req_memory_mb
        req_cores = self.state.static_state.tasks[action.task_id].req_cpu_cores
        vm_total_mem = self.state.static_state.vms[action.vm_id].memory_mb
        vm_total_cores = self.state.static_state.vms[action.vm_id].cpu_cores

        # Compute earliest feasible start time depending on feasibility_mode
        vm_state = new_vm_states[action.vm_id]
        if self.feasibility_mode == "legacy":
            # Rebuild events from active_tasks and rescan (legacy behavior)
            events_i: list[tuple[float, int, int]] = []
            used_mem = 0
            used_cores = 0
            for (et_id, et_comp, mem_i, cores_i) in (vm_state.active_tasks or []):
                t_s = self.state.task_states[et_id]
                s_i = t_s.start_time if t_s.start_time is not None else self.state.current_time
                events_i.append((s_i, +mem_i, +cores_i))
                events_i.append((et_comp, -mem_i, -cores_i))
            events_i.sort(key=lambda x: x[0])
            for (ev_t, dm, dc) in events_i:
                if ev_t <= parent_ready_time + 1e-9:
                    used_mem += dm
                    used_cores += dc
            candidate_time = parent_ready_time
            start_time = candidate_time
            times_i = [candidate_time] + [t for (t, _dm, _dc) in events_i if t > candidate_time + 1e-9]
            for t in times_i:
                if t > candidate_time + 1e-9:
                    for (ev_t, dm, dc) in events_i:
                        if candidate_time - 1e-9 < ev_t <= t + 1e-9:
                            used_mem += dm
                            used_cores += dc
                candidate_time = t
                if (used_mem + req_mem <= vm_total_mem + 1e-9) and (used_cores + req_cores <= vm_total_cores + 1e-9):
                    start_time = t
                    break
        else:
            # Optimized pointer-based scan over active_events with legacy-equivalent boundary semantics
            if self._reconcile_events:
                self._reconcile_vm_events(action.vm_id)
            events = vm_state.active_events or []
            used_mem = 0
            used_cores = 0
            idx = 0
            n_ev = len(events)
            # Apply all events <= parent_ready_time
            while idx < n_ev and events[idx][0] <= parent_ready_time + 1e-9:
                _t, dm0, dc0 = events[idx]
                used_mem += dm0
                used_cores += dc0
                idx += 1
            candidate_time = parent_ready_time
            start_time = candidate_time
            # Build future check times (no dedup to mirror legacy list-comp semantics)
            check_times = [candidate_time] + [events[j][0] for j in range(idx, n_ev)]
            for t in check_times:
                if t > candidate_time + 1e-9:
                    # Apply all deltas between (candidate_time, t] inclusively with tol
                    while idx < n_ev and events[idx][0] <= t + 1e-9:
                        _t, dm, dc = events[idx]
                        used_mem += dm
                        used_cores += dc
                        idx += 1
                candidate_time = t
                if (used_mem + req_mem <= vm_total_mem + 1e-9) and (used_cores + req_cores <= vm_total_cores + 1e-9):
                    start_time = t
                    break

        # Bottleneck metrics at this decision (simple)
        # Count how many currently-ready tasks cannot start at current_time due to capacity limits on all VMs
        # def vm_used_at_time(vm_id: int, tnow: float) -> tuple[int, int]:
        #     used_m, used_c = 0, 0
        #     for (t_id, t_comp, mem, cores) in (self.state.vm_states[vm_id].active_tasks or []):
        #         t_state = self.state.task_states[t_id]
        #         t_start = t_state.start_time if t_state.start_time is not None else self.state.current_time
        #         if t_start <= tnow + 1e-9 and tnow < t_comp - 1e-9:
        #             used_m += mem
        #             used_c += cores
        #     return used_m, used_c

        t0_ready = time.perf_counter()
        # Vectorized ready task computation via masks, excluding dummy start/end
        if self._is_ready_mask is None or self._is_assigned_mask is None:
            # Fallback (should not happen)
            ready_tasks = [t_id for t_id, t_s in enumerate(self.state.task_states)
                           if t_s.is_ready and t_s.assigned_vm_id is None and t_id not in (0, len(self.state.task_states)-1)]
        else:
            mask = self._is_ready_mask & (~self._is_assigned_mask)
            # Exclude dummy tasks 0 and T-1
            mask[0] = False
            mask[len(mask)-1] = False
            ready_tasks = np.nonzero(mask)[0].tolist()
        if self.profile:
            self._timers["ready"] += (time.perf_counter() - t0_ready)
        blocked_count = 0
        # for t_id in ready_tasks:
        #     t_req_mem = self.state.static_state.tasks[t_id].req_memory_mb
        #     t_req_cores = self.state.static_state.tasks[t_id].req_cpu_cores
        #     can_start_now = False
        #     for vm_id_iter, vm_static_iter in enumerate(self.state.static_state.vms):
        #         # Skip incompatible pairs for realism
        #         if (t_id, vm_id_iter) not in self.state.static_state.compatibilities:
        #             continue
        #         used_m, used_c = vm_used_at_time(vm_id_iter, self.state.current_time)
        #         if used_m + t_req_mem <= vm_static_iter.memory_mb + 1e-9 and \
        #            used_c + t_req_cores <= vm_static_iter.cpu_cores + 1e-9:
        #             can_start_now = True
        #             break
        #     if not can_start_now:
        #         blocked_count += 1
        #
        # self._metrics["decision_steps"] += 1
        # self._metrics["sum_ready_tasks"] += len(ready_tasks)
        # self._metrics["sum_bottleneck_ready_tasks"] += blocked_count
        # if blocked_count > 0:
        #     self._metrics["bottleneck_steps"] += 1

        # Wait time for the chosen action if it cannot start immediately
        available_time = max(self.state.current_time, parent_ready_time)
        if start_time > available_time + 1e-9:
            self._metrics["cumulative_wait_time"] += (start_time - available_time)

        # Refined bottleneck metrics at this decision (only if metrics enabled)
        if self.compute_metrics:
            t0_feas = time.perf_counter()
            def earliest_feasible_on_vm(vm_id: int, t_id: int, t0: float) -> float:
                # Match legacy refined feasibility by rebuilding from active_tasks only
                vm_total_mem_i = int(self._vm_total_mem[vm_id])
                vm_total_cores_i = int(self._vm_total_cores[vm_id])
                req_mem_i = int(self._task_req_mem[t_id])
                req_cores_i = int(self._task_req_cores[t_id])
                existing = self.state.vm_states[vm_id].active_tasks or []
                events_i: list[tuple[float, int, int]] = []
                used_mem_i = 0
                used_cores_i = 0
                # Build events only from currently active tasks as in legacy path
                for (et_id, et_comp, mem, cores) in existing:
                    t_s = self.state.task_states[et_id]
                    s = t_s.start_time if t_s.start_time is not None else self.state.current_time
                    events_i.append((s, +mem, +cores))
                    events_i.append((et_comp, -mem, -cores))
                events_i.sort(key=lambda x: x[0])
                for (ev_t, dm, dc) in events_i:
                    if ev_t <= t0 + 1e-9:
                        used_mem_i += dm
                        used_cores_i += dc
                candidate = t0
                times_i = [candidate] + [t for (t, _dm, _dc) in events_i if t > candidate + 1e-9]
                for t in times_i:
                    if t > candidate + 1e-9:
                        for (ev_t, dm, dc) in events_i:
                            if candidate - 1e-9 < ev_t <= t + 1e-9:
                                used_mem_i += dm
                                used_cores_i += dc
                    candidate = t
                    if used_mem_i + req_mem_i <= vm_total_mem_i + 1e-9 and used_cores_i + req_cores_i <= vm_total_cores_i + 1e-9:
                        return t
                return times_i[-1] if times_i else candidate

            blocked_refined = 0
            for t_id in ready_tasks:
                prt = self.state.current_time
                for p_id in self._parents[t_id]:
                    prt = max(prt, self.state.task_states[p_id].completion_time)
                earliest_any = float("inf")
                # iterate only compatible VMs
                for vm_id_iter in np.nonzero(self._compat_bool[t_id])[0].tolist():
                    est = earliest_feasible_on_vm(vm_id_iter, t_id, prt)
                    earliest_any = min(earliest_any, est)
                if earliest_any > prt + 1e-9:
                    blocked_refined += 1
            self._metrics["sum_ready_tasks_refined"] += len(ready_tasks)
            self._metrics["sum_blocked_ready_tasks_refined"] += blocked_refined
            if blocked_refined > 0:
                self._metrics["refined_bottleneck_steps"] += 1
            if self.profile:
                self._timers["feasible"] += (time.perf_counter() - t0_feas)

        # Optional: record assignability now (how many of the ready tasks can start immediately)
        if self.compute_metrics and hasattr(self, "_assign_series"):
            sched_now = 0
            t_now = self.state.current_time
            for t_id in ready_tasks:
                # Check if any compatible VM can start this task now
                can_start = False
                for vm_id_iter in np.nonzero(self._compat_bool[t_id])[0].tolist():
                    est_now = earliest_feasible_on_vm(vm_id_iter, t_id, t_now)
                    if est_now <= t_now + 1e-9:
                        can_start = True
                        break
                if can_start:
                    sched_now += 1
            self._assign_series["t"].append(float(t_now))
            self._assign_series["ready"].append(int(len(ready_tasks)))
            self._assign_series["schedulable"].append(int(sched_now))

        # Set times for the new task
        new_task_states[action.task_id].start_time = start_time
        new_task_states[action.task_id].completion_time = start_time + processing_time
        # Update masks reflecting assignment and readiness changes
        if self._is_assigned_mask is not None:
            self._is_assigned_mask[action.task_id] = True
        if self._is_ready_mask is not None:
            self._is_ready_mask[action.task_id] = False
            for child_id in child_task_ids:
                self._is_ready_mask[child_id] = (self._indegree[child_id] == 0)

        # Update VM memory usage only if task starts now; always append to active tasks timeline
        before_used_mem = new_vm_states[action.vm_id].used_memory_mb
        before_used_cores = new_vm_states[action.vm_id].used_cpu_cores
        if start_time <= self.state.current_time + 1e-9:
            new_vm_states[action.vm_id].used_memory_mb += req_mem
            new_vm_states[action.vm_id].used_cpu_cores += req_cores
        if new_vm_states[action.vm_id].active_tasks is None:
            new_vm_states[action.vm_id].active_tasks = []
        # Keep active_tasks sorted by completion_time
        comp_time_new = new_task_states[action.task_id].completion_time
        at_list = new_vm_states[action.vm_id].active_tasks
        # Find insertion index by completion time
        insert_pos = bisect_left([t for (_tid, t, _m, _c) in at_list], comp_time_new)
        at_list.insert(insert_pos, (action.task_id, comp_time_new, req_mem, req_cores))
        new_vm_states[action.vm_id].completion_time = max([t for (_tid, t, _m, _c) in new_vm_states[action.vm_id].active_tasks], default=start_time)
        logging.debug(
            f"[MEM-ALLOC] VM {action.vm_id}: scheduled task {action.task_id} start={start_time:.4f} "
            f"end={new_task_states[action.task_id].completion_time:.4f}, total mem={self.state.static_state.vms[action.vm_id].memory_mb}, cores={vm_total_cores};  "
            f"mem +{req_mem} MB, cores +{req_cores}; used mem {before_used_mem} -> {new_vm_states[action.vm_id].used_memory_mb} MB; "
            f"used cores {before_used_cores} -> {new_vm_states[action.vm_id].used_cpu_cores}"
        )
        logging.debug(
            f"[VM-STATE] VM {action.vm_id}: completion_time={new_vm_states[action.vm_id].completion_time:.4f}, "
            f"active_tasks={[(t, c) for (t, c, _m, _cores) in (new_vm_states[action.vm_id].active_tasks or [])]}"
        )

        # Maintain per-VM event list with the newly scheduled task
        # Insert (start_time, +mem/+cores) and (end_time, -mem/-cores) keeping the list sorted
        insort(vm_state.active_events, (start_time, +req_mem, +req_cores))
        insort(vm_state.active_events, (new_task_states[action.task_id].completion_time, -req_mem, -req_cores))

        # Update energy consumption with piecewise integration across future events during execution
        vm = self.state.static_state.vms[action.vm_id]
        vm_speed = float(self._vm_speed_mips[action.vm_id]) if self._vm_speed_mips is not None else vm.cpu_speed_mips
        total_cores = max(vm_total_cores, 1)
        start_t = start_time
        end_t = new_task_states[action.task_id].completion_time

        # Build other-tasks events on this VM using active_tasks only: (time, delta_cores)
        other_events: list[tuple[float, int]] = []
        for (t_id, t_comp, _mem_i, cores_i) in (vm_state.active_tasks or []):
            if t_id == action.task_id:
                continue
            t_state = self.state.task_states[t_id]
            t_start_i = t_state.start_time
            if t_start_i is None:
                continue
            if t_start_i < end_t and t_comp > start_t:
                other_events.append((t_start_i, +cores_i))
                other_events.append((t_comp, -cores_i))
        other_events.sort(key=lambda x: x[0])

        # Initialize used_cores at start_t by applying all deltas up to start_t
        used_cores_at_start = used_cores  # from feasibility sweep before scheduling
        i = 0
        n = len(other_events)
        while i < n and other_events[i][0] <= start_t:
            _, dc = other_events[i]
            used_cores_at_start += dc
            i += 1

        # Create segment boundaries within [start_t, end_t]
        times = [start_t]
        times += [t for (t, _dc) in other_events if start_t < t < end_t]
        times.append(end_t)
        # Deduplicate and sort
        times = sorted(set(times))

        energy = 0.0
        used_cores_seg = used_cores_at_start
        # Walk segments
        for idx in range(len(times) - 1):
            t0, t1 = times[idx], times[idx + 1]
            dt = t1 - t0
            cpu_fraction_seg = min(1.0, (used_cores_seg + req_cores) / total_cores)
            rate = active_energy_consumption_per_mi(vm, cpu_fraction_seg)

            seg_mi = vm_speed * dt
            # print(f'[ENERGY] VM {action.vm_id} segment {t0:.4f}-{t1:.4f} dt={dt:.4f}, used_cores={used_cores_seg}, cpu_fraction={cpu_fraction_seg:.4f}, rate={rate:.4f} W/MI, seg_mi={seg_mi:.4f}')
            energy += rate * seg_mi
            # Apply all events at t1
            while i < n and other_events[i][0] == t1:
                _t, dc = other_events[i]
                used_cores_seg += dc
                i += 1

        new_task_states[action.task_id].energy_consumption = energy

        # New dependencies (a new edge between the old task in the VM and this task)
        # New dependencies (a new edge between the old task in the VM and this task)
        new_task_dependencies = self.state.task_dependencies.copy()
        vm_prev_task_id = self.state.vm_states[action.vm_id].assigned_task_id or 0
        new_task_dependencies.add((vm_prev_task_id, action.task_id))
        # Update caches for parents/children/indegree due to new edge
        self._children[vm_prev_task_id].append(action.task_id)
        self._parents[action.task_id].append(vm_prev_task_id)
        self._indegree[action.task_id] += 1  # this edge is not for readiness but ordering for end dummy
        # Mask update is not needed here for readiness; assignment mask already updated

        # Check if dummy end active, then all tasks have been scheduled
        # If so, assign it to 0 vm (it doesn't matter since length is 0)
        if new_task_states[-1].is_ready:
            logging.debug("[END-READY] (SCHED) Dummy end task ready; assigning to VM 0")
            new_task_states[-1].is_ready = False
            new_task_states[-1].assigned_vm_id = 0
            new_task_states[-1].start_time = max(vm.completion_time for vm in new_vm_states)
            new_task_states[-1].completion_time = new_task_states[-1].start_time
            new_vm_states[0].assigned_task_id = len(new_task_states) - 1
            logging.debug(
                f"[END-ASSIGN] (SCHED) Dummy end task assigned: t={new_task_states[-1].completion_time:.4f}"
            )

        # Change the state
        self.state = EnvState(
            static_state=self.state.static_state,
            task_states=new_task_states,
            vm_states=new_vm_states,
            task_dependencies=new_task_dependencies,
        )

        # If the final task is not submitted yet, we can give the immediate rewards (if any)
        if self.state.task_states[-1].assigned_vm_id is None:
            # print(f"[SCHEDULED] task {action.task_id} on VM {action.vm_id} , ")
            return EnvObservation(self.state, is_wait=False), 0, False, False, {}

        # Finalization triggered via scheduling branch: log and build sorted assignments
        # Robust end-of-episode energy aggregation (avoid undefined variables)
        if self.profile:
            t0_end_energy = time.perf_counter()
        # Aggregate per-task energies to per-VM totals
        num_vms = len(self.state.vm_states)
        per_vm_energy: list[float] = [0.0 for _ in range(num_vms)]
        for tid, tstate in enumerate(self.state.task_states):
            if tid == 0 or tid == len(self.state.task_states) - 1:
                continue
            vm_id_assigned = tstate.assigned_vm_id
            if vm_id_assigned is None:
                continue
            e = tstate.energy_consumption if tstate.energy_consumption is not None else 0.0
            per_vm_energy[vm_id_assigned] += float(e)
        # Detailed breakdowns by integrating power over entire episode time (testing/metrics mode)
        per_vm_energy_active: list[float] = []
        per_vm_energy_idle: list[float] = []
        try:
            # Compute episode horizon as max completion across all tasks
            makespan_end = 0.0
            for t in self.state.task_states:
                if t.completion_time is not None:
                    makespan_end = max(makespan_end, float(t.completion_time))
            # Build per-VM active/idle energy by scanning core usage timelines
            for vm_id in range(num_vms):
                vm_static = self.state.static_state.vms[vm_id]
                idle_w = float(vm_static.host_power_idle_watt)
                peak_w = float(vm_static.host_power_peak_watt)
                total_cores = max(1, int(vm_static.cpu_cores))

                # Collect start/end events for tasks on this VM
                events: list[tuple[float, int]] = []  # (time, delta_cores)
                for (t_id, t_comp, _mem, cores) in (self.state.vm_states[vm_id].active_tasks or []):
                    t_state = self.state.task_states[t_id]
                    t_start = t_state.start_time
                    if t_start is None or t_comp is None:
                        continue
                    events.append((float(t_start), +int(cores)))
                    events.append((float(t_comp), -int(cores)))
                # Add boundary times to cover idle periods
                boundary_times = [0.0, float(makespan_end)]
                boundary_times += [t for (t, _dc) in events]
                boundary_times = sorted(set(boundary_times))
                # Prepare cumulative walk
                events.sort(key=lambda x: x[0])
                idx = 0
                n = len(events)
                used_cores = 0
                energy_active = 0.0
                energy_idle = 0.0
                for j in range(len(boundary_times) - 1):
                    t0, t1 = boundary_times[j], boundary_times[j + 1]
                    # apply all events at t0
                    while idx < n and events[idx][0] <= t0:
                        _t, dc = events[idx]
                        used_cores += int(dc)
                        idx += 1
                    dt = max(0.0, t1 - t0)
                    if dt <= 0.0:
                        continue
                    cpu_fraction = min(1.0, max(0.0, used_cores / total_cores))
                    # Decompose instantaneous power into idle and active components
                    p_idle = idle_w
                    p_active = (peak_w - idle_w) * cpu_fraction
                    energy_idle += p_idle * dt
                    energy_active += p_active * dt
                per_vm_energy_active.append(energy_active)
                per_vm_energy_idle.append(energy_idle)
        except Exception:
            # If anything goes wrong, leave lists empty; downstream code handles empties
            per_vm_energy_active = []
            per_vm_energy_idle = []
        if self.profile:
            self._timers["end_energy"] += (time.perf_counter() - t0_end_energy)
        # Create an assignment array that is sorted by completion time
        sorted_assignments: list[tuple[float, VmAssignmentDto]] = []
        for task_id, task in enumerate(self.state.task_states):
            assert task.assigned_vm_id is not None, "Final task assigned but there are tasks without VM assigned"
            if task_id == 0 or task_id == len(self.state.task_states) - 1:
                continue
            u_workflow_id, u_vm_id = self.state.static_state.task_mapper.unmap_id(task_id)
            u_assignment = VmAssignmentDto(task.assigned_vm_id, u_workflow_id, u_vm_id)
            sorted_assignments.append((task.completion_time, u_assignment))
        sorted_assignments.sort(key=lambda x: x[0])

        # Store the assignment in info and submit it
        # Build per-VM timelines and capacities for downstream analysis/plots (optional)
        vm_total_mem_list = []
        vm_total_cores_list = []
        vm_timelines: list[list[dict[str, float | int]]] = []
        if self.collect_timelines:
            try:
                vm_total_mem_list = [vm.memory_mb for vm in self.state.static_state.vms]
                vm_total_cores_list = [vm.cpu_cores for vm in self.state.static_state.vms]
                vm_timelines = [[] for _ in range(len(self.state.vm_states))]
                for task_id, task in enumerate(self.state.task_states):
                    # Skip dummy start/end tasks
                    if task_id == 0 or task_id == len(self.state.task_states) - 1:
                        continue
                    vm_id_assigned = task.assigned_vm_id
                    if vm_id_assigned is None:
                        continue
                    t_start = task.start_time
                    t_end = task.completion_time
                    if t_start is None or t_end is None:
                        continue
                    # Static requirements
                    req_mem = self.state.static_state.tasks[task_id].req_memory_mb
                    req_cores = self.state.static_state.tasks[task_id].req_cpu_cores
                    vm_timelines[vm_id_assigned].append({
                        "t_start": float(t_start),
                        "t_end": float(t_end),
                        "mem": int(req_mem),
                        "cores": int(req_cores),
                    })
            except Exception:
                # Be robust: if anything goes wrong, still return other info
                vm_total_mem_list = []
                vm_total_cores_list = []
                vm_timelines = []

        info = {
            "assignments": [assignment[1] for assignment in sorted_assignments],
            "per_vm_energy": per_vm_energy,
            "per_vm_energy_active": per_vm_energy_active,
            "per_vm_energy_idle": per_vm_energy_idle,
            "total_energy": float(sum(per_vm_energy)) if per_vm_energy else 0.0,
            "total_energy_active": float(sum(per_vm_energy_active)) if per_vm_energy_active else 0.0,
            "total_energy_idle": float(sum(per_vm_energy_idle)) if per_vm_energy_idle else 0.0,
            # Bottleneck metrics
            "bottleneck_steps": int(self._metrics.get("bottleneck_steps", 0)),
            "decision_steps": int(self._metrics.get("decision_steps", 0)),
            # "sum_ready_tasks": int(self._metrics.get("sum_ready_tasks", 0)),
            # "sum_bottleneck_ready_tasks": int(self._metrics.get("sum_bottleneck_ready_tasks", 0)),
            "cumulative_wait_time": float(self._metrics.get("cumulative_wait_time", 0.0)),
            # Refined bottleneck metrics
            "refined_bottleneck_steps": int(self._metrics.get("refined_bottleneck_steps", 0)),
            "sum_ready_tasks_refined": int(self._metrics.get("sum_ready_tasks_refined", 0)),
            "sum_blocked_ready_tasks_refined": int(self._metrics.get("sum_blocked_ready_tasks_refined", 0)),
            # Timelines and capacities for plotting utilization heatmaps
            "vm_total_mem": vm_total_mem_list,
            "vm_total_cores": vm_total_cores_list,
            "vm_timelines": vm_timelines,
    
        }

        # Attach assignability time series if available
        if self.compute_metrics and hasattr(self, "_assign_series"):
            info["timeline_t"] = list(self._assign_series.get("t", []))
            info["timeline_ready"] = list(self._assign_series.get("ready", []))
            info["timeline_schedulable"] = list(self._assign_series.get("schedulable", []))

        # Critical Path (CP) vs off-CP wait-time breakdown computed at episode end
        # Compute parent-ready-time-based wait for each task and split by CP membership
        try:
            # Identify a CP by backtracking from the task with max completion time
            ct = [t.completion_time for t in self.state.task_states]
            st = [t.start_time for t in self.state.task_states]
            t0_cp = time.perf_counter()
            if self.compute_metrics and ct and st:
                end_task = max(range(len(ct)), key=lambda i: ct[i])
                cp_set: set[int] = set()
                cur = end_task
                visited = set()
                while cur not in visited:
                    visited.add(cur)
                    cp_set.add(cur)
                    # Use cached parents list instead of scanning all dependencies
                    preds = [u for u in (self._parents[cur] if hasattr(self, "_parents") else []) if abs(ct[u] - st[cur]) <= 1e-9]
                    if not preds:
                        break
                    cur = preds[0]
                wait_cp = 0.0
                wait_off = 0.0
                for tid, tstate in enumerate(self.state.task_states):
                    if tid in (0, len(self.state.task_states) - 1):
                        continue
                    # Parent-ready time from cached parents
                    prt = 0.0
                    for u in (self._parents[tid] if hasattr(self, "_parents") else []):
                        prt = max(prt, self.state.task_states[u].completion_time)
                    s = tstate.start_time if tstate.start_time is not None else 0.0
                    wait = max(0.0, s - prt)
                    if tid in cp_set:
                        wait_cp += wait
                    else:
                        wait_off += wait
                info["wait_time_cp"] = float(wait_cp)
                info["wait_time_offcp"] = float(wait_off)
                if self.profile:
                    self._timers["end_cp"] += (time.perf_counter() - t0_cp)
        except Exception:
            # be robust; skip CP breakdown if anything goes wrong
            pass
        # Define reward as negative makespan at episode end (final dummy task completion)
        final_ct = self.state.task_states[-1].completion_time
        if final_ct is None:
            final_ct = max(vm.completion_time for vm in self.state.vm_states)
        makespan = float(final_ct)
        reward = -makespan
        return EnvObservation(self.state, is_wait=False), reward, False, True, info

    @staticmethod
    def gen_dataset(seed: int | None, dataset_args: DatasetArgs):
        sd = seed if seed is not None else random.randint(1, MAX_TRAINING_DS_SEED)
        style = str(getattr(dataset_args, "style", "generic")).lower()
        gnp_p_val = getattr(dataset_args, "gnp_p", None)
        req_div = getattr(dataset_args, "req_divisor", None)
        if style == "long_cp":
            # Use queue-free generator for long_cp; pin p if provided
            p_range = (float(gnp_p_val), float(gnp_p_val)) if gnp_p_val is not None else (0.70, 0.95)
            return generate_dataset_long_cp_queue_free(
                seed=sd,
                host_count=dataset_args.host_count,
                vm_count=dataset_args.vm_count,
                max_memory_gb=dataset_args.max_memory_gb,
                min_cpu_speed_mips=dataset_args.min_cpu_speed,
                max_cpu_speed_mips=dataset_args.max_cpu_speed,
                workflow_count=dataset_args.workflow_count,
                gnp_min_n=dataset_args.gnp_min_n,
                gnp_max_n=dataset_args.gnp_max_n,
                task_length_dist=dataset_args.task_length_dist,
                min_task_length=dataset_args.min_task_length,
                max_task_length=dataset_args.max_task_length,
                task_arrival=dataset_args.task_arrival,
                arrival_rate=dataset_args.arrival_rate,
                vm_rng_seed=0,
                p_range=p_range,
                alpha_range=(0.8, 0.95),
                req_divisor=req_div,
            )
        if style == "wide":
            # Use queue-free generator for wide; pin p if provided
            p_range = (float(gnp_p_val), float(gnp_p_val)) if gnp_p_val is not None else (0.02, 0.20)
            return generate_dataset_wide_queue_free(
                seed=sd,
                host_count=dataset_args.host_count,
                vm_count=dataset_args.vm_count,
                max_memory_gb=dataset_args.max_memory_gb,
                min_cpu_speed_mips=dataset_args.min_cpu_speed,
                max_cpu_speed_mips=dataset_args.max_cpu_speed,
                workflow_count=dataset_args.workflow_count,
                gnp_min_n=dataset_args.gnp_min_n,
                gnp_max_n=dataset_args.gnp_max_n,
                task_length_dist=dataset_args.task_length_dist,
                min_task_length=dataset_args.min_task_length,
                max_task_length=dataset_args.max_task_length,
                task_arrival=dataset_args.task_arrival,
                arrival_rate=dataset_args.arrival_rate,
                vm_rng_seed=0,
                p_range=p_range,
                alpha_range=(0.8, 0.95),
                req_divisor=req_div,
            )
        # Fallback: legacy non-queue-free generator
        return generate_dataset(
            seed=sd,
            host_count=dataset_args.host_count,
            vm_count=dataset_args.vm_count,
            max_memory_gb=dataset_args.max_memory_gb,
            min_cpu_speed_mips=dataset_args.min_cpu_speed,
            max_cpu_speed_mips=dataset_args.max_cpu_speed,
            workflow_count=dataset_args.workflow_count,
            dag_method=dataset_args.dag_method,
            gnp_min_n=dataset_args.gnp_min_n,
            gnp_max_n=dataset_args.gnp_max_n,
            task_length_dist=dataset_args.task_length_dist,
            min_task_length=dataset_args.min_task_length,
            max_task_length=dataset_args.max_task_length,
            task_arrival=dataset_args.task_arrival,
            arrival_rate=dataset_args.arrival_rate,
            gnp_p=gnp_p_val,
        )
