# Concurrency-Aware Greedy Heuristics Summary

## Two Core Heuristics

### 1. Makespan Heuristic T̂(s) = obs.makespan()

**Purpose:** Estimate earliest completion time from current state

**Algorithm:** Greedy earliest-completion-time (ECT) scheduling
- Build VM event timelines from scheduled tasks: (time, Δmem, Δcores)
- For each unscheduled task (in topological order):
  - Find earliest feasible start on each compatible VM
  - Select VM with earliest completion
  - Update VM timeline with new events
- Return max completion time

**Key feature:** Multiple tasks can overlap on a VM if aggregate resources fit

### 2. Energy Heuristic Ê(s) = obs.energy_consumption()

**Purpose:** Estimate minimum active energy from current state

**Algorithm:** Piecewise power integration with fractional CPU
- For each unscheduled task:
  - Find earliest feasible start on each compatible VM
  - Partition execution interval by VM event times
  - For each segment: compute U(t), power P(t), energy E_seg
  - Select VM with minimum energy
- Sum energies across all tasks

**Key feature:** Power = P_idle + (P_peak - P_idle) × U(t), where U(t) reflects concurrent core usage

## Regret-Based Rewards

```
ΔR^mk_k = -(T̂(s_{k+1}) - T̂(s_k)) / max(T̂(s_{k+1}), ε)
ΔR^en_k = -(Ê(s_{k+1}) - Ê(s_k)) / max(Ê(s_{k+1}), ε)
r_k = w_T × ΔR^mk_k + w_E × ΔR^en_k
```

Positive reward when action reduces estimated cost-to-go.

## Implementation

- **Makespan:** `observation.py` lines 76-199
- **Energy:** `observation.py` lines 201-344
- **Rewards:** `wrapper.py` lines 175-183

## Example Impact

VM: 4 cores, Task j (2 cores, running), Task i (1 core, ready)

**Sequential:** i waits → Makespan=20s, Energy=2125J
**Concurrent:** i starts now → Makespan=10s, Energy=1625J (23% savings)

Reward signal guides agent to exploit concurrency!
