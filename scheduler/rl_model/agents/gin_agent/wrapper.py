from typing import SupportsFloat, Any
import os
import numpy as np
import gymnasium as gym

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi
from scheduler.rl_model.agents.gin_agent.shared_debug import get_last_action_probs


class GinAgentWrapper(gym.Wrapper):
    observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Discrete(MAX_OBS_SIZE)

    prev_obs: EnvObservation
    initial_obs: EnvObservation

    def __init__(self, env: gym.Env[np.ndarray, int], *,
                 constrained_mode: bool = False,
                 constraint_ratio: float = 1.60,
                 constraint_penalty_scale: float = 1000.0,
                 constraint_step_penalty_scale: float = 10.0,
                 use_lagrangian: bool | None = None,
                 lag_alpha: float = 1e-3,
                 lag_init: float = 0.0):
        super().__init__(env)
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)
        # Terminal energy normalization state
        self.energy_ref: float | None = None  # running reference for total_energy (EMA)
        self.energy_ref_alpha: float = 0.9    # higher -> slower update
        self.terminal_energy_weight: float = 1.0
        # Weights for multi-objective reward (can be overridden by env vars)
        self.energy_weight: float = float(os.environ.get("GIN_ENERGY_WEIGHT", 1.0))
        self.makespan_weight: float = float(os.environ.get("GIN_MAKESPAN_WEIGHT", 1.0))  # set 0.0 for energy-only shaping
        # Toggle: compute makespan per-step (1) or only at terminal (0) to save time
        self.calc_step_makespan: bool = os.environ.get("GIN_STEP_MAKESPAN", "1") == "1"
        # Debug flag (enable by setting env var GIN_DEBUG=1)
        self.debug: bool = os.environ.get("GIN_DEBUG", "0") == "1"
        # Disable constrained/Lagrangian modes: force off regardless of args/env
        self.constrained_mode: bool = False
        self.constraint_ratio: float = constraint_ratio  # e.g., 1.10 -> 110% of baseline makespan
        self.constraint_penalty_scale: float = constraint_penalty_scale
        self.constraint_step_penalty_scale: float = constraint_step_penalty_scale
        # Baseline cache (set on reset when constrained_mode)
        self._baseline_makespan: float | None = None
        self._baseline_active_energy: float | None = None
        # Lagrangian CMDP option disabled permanently
        self.use_lagrangian: bool = False
        self.lag_alpha: float = float(lag_alpha)
        self.lag_lambda: float = float(lag_init)
        # When enabled, restrict scoring state to valid actions only by filtering compatibilities
        # to edges where the task is READY and not yet scheduled.
        self.valid_only_scoring: bool = os.environ.get("VALID_ONLY_SCORING", "0") == "1"

    def set_lag_alpha(self, value: float) -> None:
        """No-op: Lagrangian is disabled."""
        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        assert isinstance(obs, EnvObservation)
        mapped_obs = self.map_observation(obs)

        self.prev_obs = obs
        self.initial_obs = obs
        # Initialize episodic return accumulators
        self._episode_energy_return: float = 0.0
        self._episode_makespan_return: float = 0.0
        # Compute baseline for constraint mode
        if self.constrained_mode:
            try:
                bm, be = self._compute_fastest_baseline(obs)
                self._baseline_makespan = bm
                self._baseline_active_energy = be
                if self.debug:
                    print(f"[WRAPPER][constraint] baseline_makespan={bm:.6g}, baseline_active_energy={be:.6g}")
            except Exception as e:
                if self.debug:
                    print(f"[WRAPPER][constraint] baseline computation failed: {e}")
                self._baseline_makespan = None
                self._baseline_active_energy = None
        return mapped_obs, info

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        # Map to (task_id, vm_id)
        vm_count = len(self.prev_obs.vm_observations)
        task_id = int(action // vm_count)
        vm_id = int(action % vm_count)
        # Validate against READY ∧ NOT-SCHEDULED ∧ COMPATIBLE
        ready = False
        not_scheduled = False
        if 0 <= task_id < len(self.prev_obs.task_observations):
            t = self.prev_obs.task_observations[task_id]
            ready = bool(t.is_ready)
            not_scheduled = (t.assigned_vm_id is None)
        compat = (task_id, vm_id) in set(self.prev_obs.compatibilities)
        if not (ready and not_scheduled and compat):
            if self.debug:
                total_compat = len(self.prev_obs.compatibilities)
                ready_cnt = sum(int(t.is_ready and (t.assigned_vm_id is None)) for t in self.prev_obs.task_observations)
                # Build the set/list of compatible VMs for this task
                compat_list = [(t_id, v_id) for (t_id, v_id) in self.prev_obs.compatibilities if t_id == task_id]
                compat_vm_ids = [v for (_t, v) in compat_list]
                has_any_compat = len(compat_vm_ids) > 0
                # Detailed diagnostics for the incompatible pair (debug only)
                try:
                    t_obs = self.prev_obs.task_observations[task_id]
                    v_obs = self.prev_obs.vm_observations[vm_id]
                    print(
                        f"[WRAPPER] INVALID proposed (t={task_id},v={vm_id}) ready={ready} not_scheduled={not_scheduled} compat={compat} | ready_cnt={ready_cnt} compat_pairs={total_compat}\n"
                        f"  Task requirements -> cores={getattr(t_obs, 'req_cpu_cores', 'N/A')}, mem_mb={getattr(t_obs, 'req_memory_mb', 'N/A')}\n"
                        f"  VM capacity      -> cores={getattr(v_obs, 'cpu_cores', 'N/A')}, mem_mb={getattr(v_obs, 'memory_mb', 'N/A')}\n"
                        f"  Task has compatible VMs? {has_any_compat} ; count={len(compat_vm_ids)} ; sample={compat_vm_ids[:10]}"
                    )
                    # Print action probability of proposed (t, v) and top-5 overall
                    probs = get_last_action_probs()
                    if probs is not None and isinstance(probs, np.ndarray):
                        T = len(self.prev_obs.task_observations)
                        V = len(self.prev_obs.vm_observations)
                        flat_idx = task_id * V + vm_id
                        p = float(probs[flat_idx]) if 0 <= flat_idx < probs.size else float('nan')
                        print(f"  Prob(proposed t={task_id}, v={vm_id}) = {p:.6g}")
                        # Top-5
                        try:
                            topk = min(5, probs.size)
                            top_idx = np.argsort(probs)[-topk:][::-1]
                            lines = []
                            for i in top_idx:
                                t_i = int(i // V)
                                v_i = int(i % V)
                                lines.append(f"    idx={int(i)} -> (t={t_i}, v={v_i}) p={float(probs[i]):.6g}")
                            print("  Top-5 actions by prob:\n" + "\n".join(lines))
                        except Exception:
                            pass
                except Exception:
                    print(
                        f"[WRAPPER] INVALID proposed (t={task_id},v={vm_id}) ready={ready} not_scheduled={not_scheduled} compat={compat} | ready_cnt={ready_cnt} compat_pairs={total_compat}"
                    )
            # Remap: pick first valid pair
            remapped = False
            compat_set = set(self.prev_obs.compatibilities)
            for t_id, tobs in enumerate(self.prev_obs.task_observations):
                if not tobs.is_ready or (tobs.assigned_vm_id is not None):
                    continue
                for v_id in range(vm_count):
                    if (t_id, v_id) in compat_set:
                        task_id, vm_id = t_id, v_id
                        remapped = True
                        break
                if remapped:
                    break
            if self.debug:
                if remapped:
                    print(f"[GIN_DEBUG][wrapper] REMAP -> (t={task_id},v={vm_id})")
                else:
                    print(f"[GIN_DEBUG][wrapper] NO VALID REMAP; using original (t={task_id},v={vm_id})")
        mapped_action = self.map_action(task_id * vm_count + vm_id)
        obs, _, terminated, truncated, info = super().step(mapped_action)
        assert isinstance(obs, EnvObservation)
        # If this is a wait observation, propagate zero reward and update prev_obs to avoid large next-step deltas
        if info.get('skip_step', False) or getattr(obs, 'is_wait', False):
            # print(f'In wrapper step: Skipping step (wait observation), returning mapped observation with zero reward')
            mapped_obs = self.map_observation(obs)
            self.prev_obs = obs
            return mapped_obs, 0.0, terminated, truncated, info
        # print(f'In wrapper step: action={mapped_action}, terminated={terminated}, truncated={truncated}')

        mapped_obs = self.map_observation(obs)

        eps = 1e-8
        energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / max(obs.energy_consumption(), eps)
        if self.calc_step_makespan:
            ms_curr = obs.makespan()
            ms_prev = self.prev_obs.makespan()
            makespan_reward = -(ms_curr - ms_prev) / max(ms_curr, eps)
        else:
            makespan_reward = 0.0
        reward = self.energy_weight * energy_reward + self.makespan_weight * makespan_reward
        # Accumulate component returns for this episode
        try:
            self._episode_energy_return += float(energy_reward)
            self._episode_makespan_return += float(makespan_reward)
        except Exception:
            pass
        if self.debug:
            try:
                print(
                    f"[reward/step] E={energy_reward:.4g}*{self.energy_weight:.3g}={self.energy_weight*energy_reward:.4g}; "
                    f"M={makespan_reward:.4g}*{self.makespan_weight:.3g}={self.makespan_weight*makespan_reward:.4g}; "
                    f"total(before step-pen)={reward:.4g}"
                )
            except Exception:
                pass
        try:
            info["dbg_energy_reward"] = float(energy_reward)
            info["dbg_makespan_reward"] = float(makespan_reward)
            info["dbg_step_reward_total"] = float(reward)
        except Exception:
            pass
        # Constraint shaping disabled
        if False and self.calc_step_makespan and self.constrained_mode and self._baseline_makespan is not None:
            try:
                constraint = self._baseline_makespan * self.constraint_ratio
                current_ms = float(self.prev_obs.makespan())  # use prev_obs to compare consistency with deltas
                over = max(0.0, current_ms - constraint)
                step_pen = 0.0
                if over > 0.0:
                    step_pen = self.constraint_step_penalty_scale * over
                    reward -= step_pen
                if self.debug:
                    try:
                        print(
                            f"[reward/step-constraint] current_ms={current_ms:.4g} constraint={constraint:.4g} "
                            f"over={over:.4g} step_pen={step_pen:.4g} total(after)={reward:.4g}"
                        )
                    except Exception:
                        pass
                try:
                    info["dbg_step_constraint_over"] = float(over)
                    info["dbg_step_constraint_penalty"] = float(step_pen)
                except Exception:
                    pass
            except Exception:
                pass
        if (terminated or truncated) and isinstance(info, dict) and ("total_energy" in info):
            total_energy = float(info["total_energy"])
            if self.energy_ref is None:
                self.energy_ref = max(total_energy, 1e-6)
            else:
                self.energy_ref = (
                    self.energy_ref_alpha * self.energy_ref + (1.0 - self.energy_ref_alpha) * max(total_energy, 1e-6)
                )
            terminal_term = - total_energy / max(self.energy_ref, 1e-6)
            reward += self.terminal_energy_weight * terminal_term
            if self.debug:
                try:
                    print(
                        f"[reward/terminal] term_energy={terminal_term:.4g}*{self.terminal_energy_weight:.3g}="
                        f"{self.terminal_energy_weight*terminal_term:.4g} total={reward:.4g}"
                    )
                except Exception:
                    pass
            try:
                info["dbg_terminal_energy_term"] = float(terminal_term)
            except Exception:
                pass
            # Compute final makespan once at terminal (realistic, concurrency-aware)
            final_ms = obs.makespan()
            info["makespan"] = final_ms
            # Expose episodic component returns for downstream logging
            try:
                info["active_energy_return"] = float(self._episode_energy_return)
                info["makespan_return"] = float(self._episode_makespan_return)
            except Exception:
                pass
            # Constrained/Lagrangian terminal processing disabled
            if False and self.constrained_mode and self._baseline_makespan is not None:
                try:
                    actual_active = float(info.get("total_energy_active", np.nan))
                    actual_makespan = float(info.get("makespan", final_ms))
                    constraint = self._baseline_makespan * self.constraint_ratio
                    violation = max(0.0, actual_makespan - constraint)
                    info["constraint_ratio"] = self.constraint_ratio
                    info["constraint_threshold"] = constraint
                    info["baseline_makespan"] = self._baseline_makespan
                    # Lagrangian terminal penalty
                    if self.use_lagrangian:
                        lag_pen = - self.lag_lambda * violation
                        reward = reward + lag_pen
                        # Dual update (projected gradient ascent on lambda ≥ 0)
                        lag_pre = float(self.lag_lambda)
                        self.lag_lambda = float(max(0.0, self.lag_lambda + self.lag_alpha * violation))
                        info["lag_lambda"] = self.lag_lambda
                        info["lag_alpha"] = self.lag_alpha
                        info["lag_penalty"] = lag_pen
                        if self.debug:
                            try:
                                print(
                                    f"[reward/lagrangian] violation={violation:.4g} λ_pre={lag_pre:.4g} α={self.lag_alpha:.3g} "
                                    f"lag_pen={lag_pen:.4g} λ_post={self.lag_lambda:.4g} total={reward:.4g}"
                                )
                            except Exception:
                                pass
                        try:
                            info["dbg_violation"] = float(violation)
                            info["dbg_lambda_pre"] = float(lag_pre)
                            info["dbg_lambda_post"] = float(self.lag_lambda)
                        except Exception:
                            pass
                    # Fallback: hard override using active-energy improvement when satisfied; heavy penalty otherwise
                    elif self._baseline_active_energy is not None and not np.isnan(actual_active):
                        if violation == 0.0:
                            reward = float(self._baseline_active_energy - actual_active)
                        else:
                            reward = - self.constraint_penalty_scale * violation
                        info["baseline_active_energy"] = self._baseline_active_energy
                        info["constrained_reward"] = reward
                        if self.debug:
                            try:
                                print(
                                    f"[reward/constrained-fallback] violation={violation:.4g} "
                                    f"baseline_active={self._baseline_active_energy:.4g} actual_active={actual_active:.4g} "
                                    f"total={reward:.4g}"
                                )
                            except Exception:
                                pass
                except Exception as e:
                    if self.debug:
                        print(f"[WRAPPER][constraint] terminal constraint processing failed: {e}")
        self.prev_obs = obs
        return mapped_obs, reward, terminated, truncated, info

    def _compute_fastest_baseline(self, obs: EnvObservation) -> tuple[float, float]:
        """Greedy baseline with multi-core and memory concurrency.
        - Topologically traverse tasks; for each ready task, pick the VM that minimizes earliest feasible finish
          given (parent-ready time, per-VM event timeline of (mem, cores)).
        - Feasibility: allow up to vm.cpu_cores concurrent cores and vm.memory_mb memory.
        - Active energy: accumulate energy_rate_per_mi(vm) * task.length for chosen VM (approximate).
        - Skip dummy start/end tasks.
        Optionally fuse with the observation estimate via GIN_BASELINE_USE_OBS=1 using
            baseline_makespan = max(greedy_makespan, obs.makespan()).
        """
        # Basic structures
        T = len(obs.task_observations)
        V = len(obs.vm_observations)
        parents = [[] for _ in range(T)]
        children = [[] for _ in range(T)]
        for (u, v) in obs.task_dependencies:
            if 0 <= u < T and 0 <= v < T:
                parents[v].append(u)
                children[u].append(v)
        indeg = [len(parents[i]) for i in range(T)]

        dummy_start = 0
        dummy_end = T - 1
        compat = set(obs.compatibilities)

        # Per-VM resource events: sorted list of (time, +mem/+cores) and (time, -mem/-cores)
        vm_events: list[list[tuple[float, int, int]]] = [[] for _ in range(V)]
        vm_total_mem = [int(vm.memory_mb) for vm in obs.vm_observations]
        vm_total_cores = [int(max(vm.cpu_cores, 1)) for vm in obs.vm_observations]
        vm_speed = [float(vm.cpu_speed_mips) for vm in obs.vm_observations]
        vm_energy_rate = [float(active_energy_consumption_per_mi(vm)) for vm in obs.vm_observations]

        # Completion time per task
        t_done = [0.0] * T
        ready = [i for i in range(T) if indeg[i] == 0]
        processed: set[int] = set()
        baseline_active_energy = 0.0

        def earliest_feasible_on_vm(v_id: int, parent_ready: float, req_mem: int, req_cores: int) -> float:
            events = vm_events[v_id]
            used_mem = 0
            used_cores = 0
            idx = 0
            n = len(events)
            # apply all events <= parent_ready
            while idx < n and events[idx][0] <= parent_ready + 1e-9:
                _t, dm, dc = events[idx]
                used_mem += dm
                used_cores += dc
                idx += 1
            candidate = parent_ready
            # Generate check times: parent_ready and all future event times
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
            # If no feasible slot within events, it is feasible after last event
            return check_times[-1] if check_times else candidate

        while ready:
            i = ready.pop(0)
            if i in processed:
                continue
            processed.add(i)
            # Skip dummy start/end tasks from energy accumulation
            if i in (dummy_start, dummy_end):
                for v in children[i]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        ready.append(v)
                continue

            # Earliest parent completion
            parent_ct = 0.0
            for u in parents[i]:
                parent_ct = max(parent_ct, t_done[u])

            # Candidate VMs (respect compatibility if provided)
            cand_vms = [v for v in range(V) if (i, v) in compat] or list(range(V))

            # Task params
            t_obs = obs.task_observations[i]
            req_mem = int(t_obs.req_memory_mb)
            req_cores = int(t_obs.req_cpu_cores)
            length = float(t_obs.length)

            best_vm = None
            best_end = float('inf')
            for v in cand_vms:
                start_t = earliest_feasible_on_vm(v, parent_ct, req_mem, req_cores)
                exec_time = length / max(vm_speed[v], 1e-9)
                end_t = start_t + exec_time
                if end_t < best_end:
                    best_end = end_t
                    best_vm = v

            if best_vm is None:
                # Fallback: schedule at parent_ct on VM 0 with zero duration
                best_vm = 0
                best_end = parent_ct

            # Insert events for chosen VM
            start_chosen = best_end - (length / max(vm_speed[best_vm], 1e-9))
            # Keep events sorted by time
            from bisect import insort
            insort(vm_events[best_vm], (start_chosen, +req_mem, +req_cores))
            insort(vm_events[best_vm], (best_end, -req_mem, -req_cores))

            # Record completion and energy (approximate)
            t_done[i] = best_end
            baseline_active_energy += vm_energy_rate[best_vm] * length

            # Release children
            for v in children[i]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)

        greedy_ms = max(t_done) if t_done else 0.0

        # Optionally fuse with observation estimate (safer not to be over-optimistic)
        if os.environ.get("GIN_BASELINE_USE_OBS", "0") == "1":
            try:
                obs_ms = float(obs.makespan())
                greedy_ms = max(greedy_ms, obs_ms)
            except Exception:
                pass

        return float(greedy_ms), float(baseline_active_energy)

    def map_action(self, action: int) -> EnvAction:
        vm_count = len(self.prev_obs.vm_observations)
        return EnvAction(task_id=int(action // vm_count), vm_id=int(action % vm_count))

    def map_observation(self, observation: EnvObservation) -> np.ndarray:
        # Task observations
        task_state_scheduled = np.array([task.assigned_vm_id is not None for task in observation.task_observations])
        task_state_ready = np.array([task.is_ready for task in observation.task_observations])
        task_length = np.array([task.length for task in observation.task_observations])
        task_memory_req_mb = np.array([task.req_memory_mb for task in observation.task_observations])
        task_cpu_req_cores = np.array([task.req_cpu_cores for task in observation.task_observations])

        # VM observations
        vm_speed = np.array([vm.cpu_speed_mips for vm in observation.vm_observations])
        vm_energy_rate = np.array([active_energy_consumption_per_mi(vm) for vm in observation.vm_observations])
        vm_completion_time = np.array([vm.completion_time for vm in observation.vm_observations])
        vm_memory_mb = np.array([vm.memory_mb for vm in observation.vm_observations])
        vm_available_memory_mb = np.array([vm.available_memory_mb for vm in observation.vm_observations])
        # New VM features
        vm_active_tasks_count = np.array([vm.active_tasks_count for vm in observation.vm_observations])
        # Guard against division by zero if any VM has 0 memory (shouldn't happen but safe)
        vm_used_memory_fraction = 1.0 - (vm_available_memory_mb / np.maximum(vm_memory_mb, 1e-8))
        vm_next_release_time = np.array([vm.next_release_time for vm in observation.vm_observations])
        vm_cpu_cores = np.array([vm.cpu_cores for vm in observation.vm_observations])
        vm_available_cpu_cores = np.array([vm.available_cpu_cores for vm in observation.vm_observations])
        vm_used_cpu_fraction_cores = np.array([vm.used_cpu_fraction_cores for vm in observation.vm_observations])
        vm_next_core_release_time = np.array([vm.next_core_release_time for vm in observation.vm_observations])

        # Task-Task observations
        task_dependencies = np.array(observation.task_dependencies).T

        # Task-VM observations
        compat_array = np.array(observation.compatibilities, dtype=int)
        if self.valid_only_scoring and compat_array.size > 0:
            # Keep only edges where task is READY and not yet scheduled
            task_ids = compat_array[:, 0]
            ready_mask = task_state_ready[task_ids] == 1
            not_scheduled_mask = task_state_scheduled[task_ids] == 0
            keep = np.logical_and(ready_mask, not_scheduled_mask)
            compat_array = compat_array[keep]
        compatibilities = compat_array.T if compat_array.size > 0 else np.empty((2, 0), dtype=int)

        # Task completion times
        task_completion_time = observation.task_completion_time()
        assert task_completion_time is not None

        return self.mapper.map(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,
            task_completion_time=task_completion_time,
            task_memory_req_mb=task_memory_req_mb,
            task_cpu_req_cores=task_cpu_req_cores,
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            vm_memory_mb=vm_memory_mb,
            vm_available_memory_mb=vm_available_memory_mb,
            vm_used_memory_fraction=vm_used_memory_fraction,
            vm_active_tasks_count=vm_active_tasks_count,
            vm_next_release_time=vm_next_release_time,
            vm_cpu_cores=vm_cpu_cores,
            vm_available_cpu_cores=vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=vm_used_cpu_fraction_cores,
            vm_next_core_release_time=vm_next_core_release_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
