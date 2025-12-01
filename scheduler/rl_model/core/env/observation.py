import copy
from dataclasses import dataclass

import numpy as np

from scheduler.rl_model.core.env.state import EnvState


@dataclass
class EnvObservation:
    task_observations: list["TaskObservation"]
    vm_observations: list["VmObservation"]
    task_dependencies: list[tuple[int, int]]
    compatibilities: list[tuple[int, int]]

    _makespan: float | None = None
    _energy_consumption: float | None = None
    _task_completion_time: np.ndarray | None = None
    is_wait: bool = False

    def __init__(self, state: EnvState, is_wait: bool = False):
        self.task_observations = [
            TaskObservation(
                is_ready=state.task_states[task_id].is_ready,
                assigned_vm_id=state.task_states[task_id].assigned_vm_id,
                start_time=state.task_states[task_id].start_time,
                completion_time=state.task_states[task_id].completion_time,
                energy_consumption=state.task_states[task_id].energy_consumption,
                length=state.static_state.tasks[task_id].length,
                req_memory_mb=state.static_state.tasks[task_id].req_memory_mb,
                req_cpu_cores=state.static_state.tasks[task_id].req_cpu_cores,
            )
            for task_id in range(len(state.task_states))
        ]
        self.vm_observations = [
            VmObservation(
                assigned_task_id=state.vm_states[vm_id].assigned_task_id,
                completion_time=state.vm_states[vm_id].completion_time,
                cpu_speed_mips=state.static_state.vms[vm_id].cpu_speed_mips,
                memory_mb=state.static_state.vms[vm_id].memory_mb,
                available_memory_mb=state.static_state.vms[vm_id].memory_mb - state.vm_states[vm_id].used_memory_mb,
                active_tasks_count=len(state.vm_states[vm_id].active_tasks or []),
                next_release_time=min(
                    [t for (_tid, t, _m, _c) in (state.vm_states[vm_id].active_tasks or []) if t > state.current_time],
                    default=state.current_time,
                ),
                cpu_cores=state.static_state.vms[vm_id].cpu_cores,
                available_cpu_cores=state.static_state.vms[vm_id].cpu_cores - state.vm_states[vm_id].used_cpu_cores,
                used_cpu_fraction_cores=(
                    (state.vm_states[vm_id].used_cpu_cores / max(state.static_state.vms[vm_id].cpu_cores, 1))
                ),
                next_core_release_time=min(
                    [t for (_tid, t, _m, _c) in (state.vm_states[vm_id].active_tasks or []) if t > state.current_time],
                    default=state.current_time,
                ),
                host_power_idle_watt=state.static_state.vms[vm_id].host_power_idle_watt,
                host_power_peak_watt=state.static_state.vms[vm_id].host_power_peak_watt,
                host_cpu_speed_mips=state.static_state.vms[vm_id].host_cpu_speed_mips,
            )
            for vm_id in range(len(state.vm_states))
        ]
        self.task_dependencies = copy.deepcopy(list(state.task_dependencies))
        # Static compatibilities by capacity: VM total memory and cores must be >= task requirements
        static_capacity_compat = []
        for (t_id, v_id) in state.static_state.compatibilities:
            if (
                state.static_state.vms[v_id].memory_mb >= state.static_state.tasks[t_id].req_memory_mb
                and state.static_state.vms[v_id].cpu_cores >= state.static_state.tasks[t_id].req_cpu_cores
            ):
                static_capacity_compat.append((t_id, v_id))
        
        self.compatibilities = static_capacity_compat

        self.is_wait = is_wait

    def makespan(self):
        if self._makespan is not None:
            return self._makespan

        # Realistic concurrency-aware estimate:
        # Build a greedy feasible schedule from the current observation by simulating per-VM
        # (mem, cores) event timelines. Already-scheduled tasks seed the event lists; unscheduled
        # tasks are placed respecting dependencies, memory, and core capacities.

        T = len(self.task_observations)
        V = len(self.vm_observations)

        # Parents/children adjacency
        parents = [[] for _ in range(T)]
        children = [[] for _ in range(T)]
        for (u, v) in self.task_dependencies:
            if 0 <= u < T and 0 <= v < T:
                parents[v].append(u)
                children[u].append(v)

        # Build per-VM event timelines from already-scheduled tasks
        vm_events: list[list[tuple[float, int, int]]] = [[] for _ in range(V)]
        for t_id, t_obs in enumerate(self.task_observations):
            if t_obs.assigned_vm_id is not None:
                vm_id = int(t_obs.assigned_vm_id)
                s_t = float(t_obs.start_time)
                e_t = float(t_obs.completion_time)
                dm = int(t_obs.req_memory_mb)
                dc = int(t_obs.req_cpu_cores)
                vm_events[vm_id].append((s_t, +dm, +dc))
                vm_events[vm_id].append((e_t, -dm, -dc))
        for v in range(V):
            vm_events[v].sort(key=lambda x: x[0])

        vm_total_mem = [int(vm.memory_mb) for vm in self.vm_observations]
        vm_total_cores = [int(max(vm.cpu_cores, 1)) for vm in self.vm_observations]
        vm_speed = [float(vm.cpu_speed_mips) for vm in self.vm_observations]

        # Completion times per task (scheduled tasks known; others to be filled)
        t_done = [0.0] * T
        scheduled = set()
        for t_id, t_obs in enumerate(self.task_observations):
            if t_obs.assigned_vm_id is not None:
                t_done[t_id] = float(t_obs.completion_time)
                scheduled.add(t_id)

        # Helper to compute earliest feasible start time on a VM timeline
        def earliest_feasible_on_vm(v_id: int, ready_time: float, req_mem: int, req_cores: int) -> float:
            events = vm_events[v_id]
            used_mem = 0
            used_cores = 0
            idx = 0
            n = len(events)
            # apply all events <= ready_time
            while idx < n and events[idx][0] <= ready_time + 1e-9:
                _t, dm, dc = events[idx]
                used_mem += dm
                used_cores += dc
                idx += 1
            candidate = ready_time
            # candidate times: ready_time and all future event times
            check_times = [candidate] + [events[j][0] for j in range(idx, n)]
            for t in check_times:
                if t > candidate + 1e-9:
                    # apply all deltas up to and including t
                    while idx < n and events[idx][0] <= t + 1e-9:
                        _tt, dm, dc = events[idx]
                        used_mem += dm
                        used_cores += dc
                        idx += 1
                candidate = t
                if (used_mem + req_mem <= vm_total_mem[v_id] + 1e-9) and (used_cores + req_cores <= vm_total_cores[v_id] + 1e-9):
                    return t
            # after the last event, resources are free
            return check_times[-1] if check_times else candidate

        # Compatibility set for quick membership checks
        compat = set(self.compatibilities)

        # Greedy global earliest-completion-time list scheduling for unscheduled tasks
        remaining = [i for i in range(T) if i not in scheduled]

        while remaining:
            # Collect tasks whose parents are done
            ready_tasks = [i for i in remaining if all((p in scheduled) for p in parents[i])]
            if not ready_tasks:
                # Cycle or missing parents due to dummies; fall back to all remaining
                ready_tasks = list(remaining)

            best_tuple = None  # (end_time, task_id, vm_id, start_time)
            for i in ready_tasks:
                # Parent-ready time
                prt = 0.0
                for p in parents[i]:
                    prt = max(prt, t_done[p])
                # Candidate VMs by compatibility (fallback: all VMs)
                cand_vms = [v for v in range(V) if (i, v) in compat] or list(range(V))
                req_mem = int(self.task_observations[i].req_memory_mb)
                req_cores = int(self.task_observations[i].req_cpu_cores)
                length = float(self.task_observations[i].length)
                for v in cand_vms:
                    start_t = earliest_feasible_on_vm(v, prt, req_mem, req_cores)
                    exec_time = length / max(vm_speed[v], 1e-9)
                    end_t = start_t + exec_time
                    if (best_tuple is None) or (end_t < best_tuple[0] - 1e-12):
                        best_tuple = (end_t, i, v, start_t)

            # Schedule the best (earliest finish) task
            if best_tuple is None:
                # Should not happen; break to avoid infinite loop
                break
            end_t, i_best, v_best, start_t = best_tuple
            # Insert events for the planned task and update completion times
            ev = vm_events[v_best]
            ev.append((start_t, +int(self.task_observations[i_best].req_memory_mb), +int(self.task_observations[i_best].req_cpu_cores)))
            ev.append((end_t, -int(self.task_observations[i_best].req_memory_mb), -int(self.task_observations[i_best].req_cpu_cores)))
            ev.sort(key=lambda x: x[0])
            t_done[i_best] = end_t
            scheduled.add(i_best)
            remaining.remove(i_best)

        self._task_completion_time = np.asarray(t_done, dtype=float)
        self._makespan = float(max(t_done) if t_done else 0.0)
        return self._makespan

    def energy_consumption(self):
        """Compute the total energy consumption of the schedule up to this observation.
        
        The energy model assumes each VM v has:
        - P_idle_v: Power draw (W) when idle
        - P_peak_v: Power draw (W) at 100% utilization
        - U_v(t) ∈ [0,1]: CPU utilization at time t
        
        Instantaneous power for VM v at time t:
            P_v(t) = P_idle_v + (P_peak_v - P_idle_v) * U_v(t)
        
        Total energy integrates power over the schedule makespan T_mk = max_i c_i:
            E_total = ∑_v ∫₀^T_mk P_v(t) dt
                   = E_idle + E_active
        where:
            E_idle = (∑_v P_idle_v) * T_mk
            E_active = ∑_v (P_peak_v - P_peak_v) * ∫₀^T_mk U_v(t) dt
        
        The integral is computed piecewise-constant between task start/completion events.
        For unscheduled tasks, estimates minimum possible energy.
        """
        if self._energy_consumption is not None:
            return self._energy_consumption

        from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi

        # Calculates the energy consumption of an observation or and estimate of it if the env is still running
        # Uses minimum possible energy for each unscheduled task
        task_energy_consumption = np.ones(len(self.task_observations)) * 1e8
        for task_id in range(len(self.task_observations)):
            # Check if already scheduled task
            if self.task_observations[task_id].assigned_vm_id is not None:
                task_energy_consumption[task_id] = self.task_observations[task_id].energy_consumption
                continue

            # Ensure task completion times are available (parents' completion times needed)
            if self._task_completion_time is None:
                self.makespan()

            # Parent readiness time
            parent_ids = [pid for pid, cid in self.task_dependencies if cid == task_id]
            parent_ready_time = max(self._task_completion_time[parent_ids], default=0)

            # Candidate VMs for this task
            compatible_vm_ids = [vid for tid, vid in self.compatibilities if tid == task_id]

            # Task requirements and length
            req_mem = self.task_observations[task_id].req_memory_mb
            req_cores = self.task_observations[task_id].req_cpu_cores
            task_len_mi = self.task_observations[task_id].length

            for vm_id in compatible_vm_ids:
                vm_obs = self.vm_observations[vm_id]
                total_mem = vm_obs.memory_mb
                total_cores = max(vm_obs.cpu_cores, 1)

                # Build events from scheduled tasks on this VM: (time, delta_mem, delta_cores)
                events: list[tuple[float, int, int]] = []
                for t_idx, t_obs in enumerate(self.task_observations):
                    if t_obs.assigned_vm_id == vm_id:
                        # Start and end events
                        events.append((t_obs.start_time, +t_obs.req_memory_mb, +t_obs.req_cpu_cores))
                        events.append((t_obs.completion_time, -t_obs.req_memory_mb, -t_obs.req_cpu_cores))

                # Sort events by time
                events.sort(key=lambda x: x[0])

                # Initialize usage at parent_ready_time by applying all deltas up to that time
                used_mem = 0
                used_cores = 0
                i = 0
                n = len(events)
                while i < n and events[i][0] <= parent_ready_time:
                    _, dm, dc = events[i]
                    used_mem += dm
                    used_cores += dc
                    i += 1

                # If feasible immediately at parent_ready_time
                t_cursor = parent_ready_time
                feasible = (used_mem + req_mem <= total_mem) and (used_cores + req_cores <= total_cores)
                while not feasible and i < n:
                    # Advance to next event time, apply its delta, and recheck
                    t_cursor = max(t_cursor, events[i][0])
                    # Apply all events at this same timestamp
                    same_time = t_cursor
                    while i < n and events[i][0] == same_time:
                        _, dm, dc = events[i]
                        used_mem += dm
                        used_cores += dc
                        i += 1
                    feasible = (used_mem + req_mem <= total_mem) and (used_cores + req_cores <= total_cores)

                # If still not feasible after all events, it becomes feasible after last event (no load)
                if not feasible:
                    t_cursor = events[-1][0] if events else parent_ready_time
                    used_mem = 0
                    used_cores = 0
                    feasible = True

                # Piecewise energy integration from earliest feasible start until completion
                vm_speed = vm_obs.cpu_speed_mips
                start_t = t_cursor
                exec_time = task_len_mi / max(vm_speed, 1e-8)
                end_t = start_t + exec_time

                # Gather event boundaries within (start_t, end_t)
                boundary_times = [start_t, end_t]
                for (ev_t, _dm, _dc) in events:
                    if start_t < ev_t < end_t:
                        boundary_times.append(ev_t)
                boundary_times = sorted(set(boundary_times))

                # Pre-aggregate deltas per time for quick updates
                deltas_at_time: dict[float, tuple[int, int]] = {}
                for (ev_t, dm, dc) in events:
                    if start_t < ev_t <= end_t:
                        prev_dm, prev_dc = deltas_at_time.get(ev_t, (0, 0))
                        deltas_at_time[ev_t] = (prev_dm + dm, prev_dc + dc)

                energy_acc = 0.0
                used_cores_seg = used_cores
                for j in range(len(boundary_times) - 1):
                    t0, t1 = boundary_times[j], boundary_times[j + 1]
                    dt = t1 - t0
                    cpu_fraction_seg = min(1.0, (used_cores_seg + req_cores) / total_cores)

                    rate = active_energy_consumption_per_mi(vm_obs, cpu_fraction_seg)
                    # print(f'For task {task_id} on VM {vm_id} segment {t0}-{t1}: used_cores={used_cores_seg}, cpu_fraction={cpu_fraction_seg}, rate={rate}')
                    seg_mi = vm_speed * dt
                    # print(f'  Segment length in MI: {seg_mi}')
                    # print(f'  Energy for segment: {rate} * {seg_mi} = {rate * seg_mi}')
                    energy_acc += rate * seg_mi
                    # Apply all deltas at t1
                    if t1 in deltas_at_time:
                        _dm, dc = deltas_at_time[t1]
                        used_cores_seg += dc

                new_energy = energy_acc
                task_energy_consumption[task_id] = min(new_energy, task_energy_consumption[task_id].item())
        # print(f'task_energy_consumption: {task_energy_consumption}')

        self._energy_consumption = float(task_energy_consumption.sum())
        return self._energy_consumption

    def task_completion_time(self):
        if self._task_completion_time is None:
            self.makespan()
        return self._task_completion_time


@dataclass
class TaskObservation:
    is_ready: bool
    assigned_vm_id: int | None
    start_time: float
    completion_time: float
    energy_consumption: float
    length: float
    req_memory_mb: int
    req_cpu_cores: int


@dataclass
class VmObservation:
    assigned_task_id: int | None
    completion_time: float
    cpu_speed_mips: float
    memory_mb: int
    available_memory_mb: int
    active_tasks_count: int
    next_release_time: float
    cpu_cores: int
    available_cpu_cores: int
    used_cpu_fraction_cores: float
    next_core_release_time: float

    host_power_idle_watt: float
    host_power_peak_watt: float
    host_cpu_speed_mips: float
