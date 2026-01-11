# Modifications to Problem Formulation

## Summary of Changes from Sensors Paper

This document summarizes how we adapted the MDP formulation from the Sensors paper (sensors-25-01428-v2.pdf) to support concurrent task execution on multi-core VMs.

### 1. **Concurrent Task Execution Model**

**Original (Sensors paper):** One task per VM at a time (sequential execution)

**Our extension:** Multiple tasks can execute concurrently on a VM as long as aggregate resource demands (CPU cores + memory) respect VM capacities.

**Implementation:**
- Residual capacity tracking: `residual_capacity_m(t) = (C_cpu_m - Σ cpu_j, C_mem_m - Σ mem_j)` for all active tasks
- Event-based timeline simulation with task start/completion events
- Earliest feasible start time accounts for concurrent resource availability

### 2. **CPU Fractionation in Energy Model**

**Original (Sensors paper):** Likely assumes full CPU utilization or binary on/off states

**Our extension:** Fractional CPU utilization based on aggregate core demands:
```
U_m(t) = min(1, (1/C_cpu_m) * Σ cpu_i for i ∈ A_m(t))
```

**Impact:**
- More accurate power modeling: P_m(t) = P_idle + (P_peak - P_idle) * U_m(t)
- Energy integrates piecewise-constant power over concurrent task intervals
- Reflects realistic multi-core VM behavior

### 3. **Heuristic-Based Regret Rewards**

**Original (Sensors paper):** Likely uses raw cost deltas or simpler shaping

**Our extension:** Normalized regret reductions relative to integrated heuristics:
```
ΔR^mk_k = -(T̂(s_{k+1}) - T̂(s_k)) / max(T̂(s_{k+1}), ε)
ΔR^en_k = -(Ê(s_{k+1}) - Ê(s_k)) / max(Ê(s_{k+1}), ε)
```

Where:
- `T̂(s) = obs.makespan()`: Greedy earliest-completion-time with concurrent execution
- `Ê(s) = obs.energy_consumption()`: Integrated power over concurrent task timelines

**Advantages:**
- Credit assignment: Rewards actions that reduce estimated cost-to-go
- Scale-invariant: Normalization handles varying workflow sizes
- Multi-objective: Weighted sum of makespan and energy regrets

### 4. **Dynamic Capacity Tracking**

**Original (Sensors paper):** Static or simpler capacity checks

**Our extension:** Continuous event-based timeline simulation:
- Per-VM event lists: (time, +/- mem, +/- cores) for task starts/completions
- Sorted event processing to find earliest feasible slots
- Supports arbitrary task overlaps respecting capacity constraints

### 5. **State Space Extensions**

**Added to state representation:**
- VM residual capacities over time: `{residual_capacity_m(t) for t ∈ [τ_k, ∞)}`
- Active task allocations per VM: `A_m(t)` for all VMs
- Per-task resource demands: `(cpu_i, mem_i)` for capacity checks

## Key Implementation Files

- **Wrapper:** `/scheduler/rl_model/agents/gin_agent/wrapper.py`
  - Lines 175-183: Regret reward computation
  - Lines 317-453: Greedy baseline with concurrent execution

- **Observation:** `/scheduler/rl_model/core/env/observation.py`
  - Lines 76-199: `makespan()` with event-based timeline simulation
  - Lines 201-344: `energy_consumption()` with piecewise power integration

- **Helpers:** `/scheduler/rl_model/core/utils/helpers.py`
  - Lines 16-41: `active_energy_consumption_per_mi()` with fractional CPU

## Citation Note

When citing the Sensors paper, emphasize:
1. We adopt their MDP framework (states, actions, transitions)
2. We extend it for concurrent multi-core execution
3. We modify makespan/energy heuristics to account for overlapping tasks
4. We introduce regret-based reward shaping tied to these heuristics

Suggested citation text:
```latex
We adapt the MDP formulation from \cite{sensors-paper} for workflow 
scheduling, extending it to support concurrent task execution on 
multi-core VMs with CPU fractionation and memory-aware resource 
allocation. Our key modifications include...
```
