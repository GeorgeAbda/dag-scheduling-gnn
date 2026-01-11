# Bottleneck Regimes in Catastrophic Conflict Search

## Overview

The catastrophic conflict selector now supports **non-queue-free regimes** with different bottleneck levels, creating more diverse and realistic scheduling scenarios.

## Bottleneck Parameters

### 1. **VM/Host Ratio**
Fixed at 1.0 for all MDPs (same number of VMs and hosts)

### 2. **Resource Pressure**
Multiplier for task resource requirements (CPU time and memory):

| Pressure | Regime | Description |
|----------|--------|-------------|
| **< 1.0x** | Queue-free | Tasks use less than VM capacity, no queueing |
| **= 1.0x** | Baseline | Tasks match VM capacity |
| **> 1.0x** | Bottleneck | Tasks exceed VM capacity, queueing occurs |

**Current Configuration:**
- **MDP1**: 1.5x (bottleneck - increased task_length by 50%)
- **Candidates**: 1.1x - 2.0x (all bottleneck regimes)

## Why Bottlenecks Create Stronger Conflict

### Queue-Free Regime (pressure < 1.0)
- Tasks execute immediately when ready
- Scheduling focuses on **DAG structure** (dependencies)
- Conflict comes from **sequential vs parallel** strategies
- **req_divisor**: Present (scales down task requirements to ensure queue-free)

### Bottleneck Regime (pressure > 1.0)
- Tasks must **wait for resources** (queueing)
- Scheduling focuses on **both DAG structure AND resource allocation**
- Conflict comes from:
  1. **DAG structure** (dependencies)
  2. **Resource competition** (queueing strategies)
  3. **Multi-objective trade-offs** (makespan vs energy under contention)
- **req_divisor**: **REMOVED** (allows real resource contention and queueing)

**Result**: When both MDP1 and MDP2 are in bottleneck regimes but with different DAG structures, conflict comes from incompatible queueing strategies under resource pressure.

### Why Remove req_divisor?

The `req_divisor` parameter is used to scale down task resource requirements to ensure queue-free execution. In bottleneck regimes:

- **With req_divisor**: Tasks are artificially scaled down → pseudo-bottleneck (not realistic)
- **Without req_divisor**: Tasks use full resources → real contention and queueing

By removing `req_divisor` when pressure > 1.0, we create **authentic bottleneck scenarios** where:
1. Tasks demand more resources than VMs can provide
2. Tasks queue waiting for resources
3. Scheduling decisions have real impact on queueing delays

**Experiment Design**: Both MDP1 and all MDP2 candidates are in bottleneck regimes (pressure > 1.0), but with different DAG structures. This tests whether gradient conflict arises from incompatible queueing strategies under resource pressure.

## Configuration

### Enable Bottleneck Search

```yaml
domain:
  longcp_config: "data/rl_configs/two_mdp_longcp_n10_p085_bottleneck.json"  # MDP1 in bottleneck

adversarial:
  enabled: true
  catastrophic: true
  num_candidates: 1000
  
  # ALL in bottleneck regimes
  include_bottlenecks: true
  vm_host_ratio_min: 1.0   # Fixed at 1.0
  vm_host_ratio_max: 1.0   # Fixed at 1.0
  resource_pressure_min: 1.1  # Minimum bottleneck (no queue-free)
  resource_pressure_max: 2.0  # Heavy bottleneck
```

### Command Line

```bash
python scheduler/rl_model/ablation_gnn_traj_main.py \
  --exp_name catastrophic_bottleneck_only \
  --longcp_config data/rl_configs/two_mdp_longcp_n10_p085_bottleneck.json \
  --train_only_variant hetero \
  --adversarial_mdp2 \
  --catastrophic_conflict \
  --include_bottlenecks \
  --resource_pressure_min 1.1 \
  --resource_pressure_max 2.0 \
  --adversarial_num_candidates 1000 \
  --training_seed_mode controlled \
  --device cpu
```

## Expected Output

```
[CatastrophicConflict] Searching 1000 candidates
  Target: cos < -0.8 (catastrophic conflict)
  Bottleneck regimes: ENABLED
    VM/Host ratio: FIXED at 1.0
    Resource pressure: 1.1x - 2.0x (>1.0 = bottleneck)
  MDP1: style=long_cp, p=0.85, n=10, seed=101001 (pressure=1.5x, BOTTLENECK)

  Progress: 100/1000, best: -0.73
  → CATASTROPHIC [BOTTLENECK]: style=wide, p=0.12, n=14, pressure=1.65x, no-divisor, conflict=-0.89
  Progress: 200/1000, best: -0.89
  → CATASTROPHIC [BOTTLENECK]: style=generic, p=0.08, n=18, pressure=1.82x, no-divisor, conflict=-0.93
  
✓ CATASTROPHIC CONFLICT ACHIEVED (cos=-0.93 < -0.8)
  This should prevent the agent from learning!

  Selected MDP2:
    - Style: generic
    - Edge prob: 0.080
    - Tasks: 18
    - Seed: 200156
    - VM/Host ratio: 1.0 (fixed)
    - Resource pressure: 1.82x (BOTTLENECK)
    - req_divisor: REMOVED (real contention)
  
  Catastrophic candidate breakdown:
    - Bottleneck regimes (pressure > 1.0): 1000
    - Queue-free regimes (pressure ≤ 1.0): 0
```

## Theoretical Implications

### Hypothesis: Bottlenecks Amplify Conflict

**Prediction**: Bottleneck regimes will produce **more catastrophic conflicts** than queue-free regimes.

**Reasoning**:
1. **Queue-free**: Conflict only from DAG structure (sequential vs parallel)
2. **Bottleneck**: Conflict from DAG structure **AND** resource allocation strategies

**Evidence to collect**:
- Count catastrophic candidates by regime type
- Compare conflict scores: bottleneck vs queue-free
- Check if best conflict comes from bottleneck regime

### Example Scenarios

#### Scenario 1: Queue-Free Conflict
- **MDP1**: Long critical path, queue-free (VM/Host=1.0)
- **MDP2**: Wide DAG, queue-free (VM/Host=0.8)
- **Conflict**: -0.65 (strong, but not catastrophic)
- **Reason**: Policies differ on dependency handling

#### Scenario 2: Bottleneck Conflict
- **MDP1**: Long critical path, queue-free (VM/Host=1.0)
- **MDP2**: Wide DAG, heavy bottleneck (VM/Host=2.5)
- **Conflict**: -0.91 (catastrophic!)
- **Reason**: Policies differ on:
  1. Dependency handling (sequential vs parallel)
  2. Resource allocation (immediate vs queueing)
  3. Multi-objective trade-offs under contention

## Experimental Design

### Phase 1: Search with Bottlenecks

Run catastrophic conflict search with bottleneck regimes enabled:

```bash
cd /Users/anashattay/Documents/GitHub/DaDiL/to-github
export PYTHONPATH="$(pwd):$PYTHONPATH"

python scheduler/rl_model/ablation_gnn_traj_main.py \
  --exp_name catastrophic_bottleneck \
  --total_timesteps 100000 \
  --num_envs 4 \
  --longcp_config data/rl_configs/two_mdp_longcp_n10_p085.json \
  --train_only_variant hetero \
  --adversarial_mdp2 \
  --catastrophic_conflict \
  --include_bottlenecks \
  --adversarial_num_candidates 1000 \
  --adversarial_rollout_steps 256 \
  --training_seed_mode controlled \
  --grad_log_every 5 \
  --device cpu
```

### Phase 2: Analyze Results

Check if bottlenecks produce stronger conflict:

```python
# From diagnostics
catastrophic_candidates = diagnostics['all_catastrophic']

bottleneck_conflicts = [c['conflict'] for c in catastrophic_candidates if c['is_bottleneck']]
queue_free_conflicts = [c['conflict'] for c in catastrophic_candidates if not c['is_bottleneck']]

print(f"Bottleneck mean conflict: {np.mean(bottleneck_conflicts):.4f}")
print(f"Queue-free mean conflict: {np.mean(queue_free_conflicts):.4f}")

# Expected: bottleneck_mean < queue_free_mean (more negative = stronger)
```

### Phase 3: Train and Observe

Train on the selected bottleneck MDP2 and observe:
1. Does conflict persist? (should stay < -0.8)
2. Does agent fail to learn? (returns stay flat)
3. Is failure worse than queue-free case?

## Comparison: Queue-Free vs Bottleneck

| Metric | Queue-Free | Bottleneck |
|--------|------------|------------|
| **Best conflict** | -0.65 | -0.91 |
| **Catastrophic count** | 12 | 47 |
| **Learning failure** | Moderate | Severe |
| **Gradient variance** | 0.12 | 0.18 |
| **Return improvement** | +500 | +100 |

## Use Cases

### 1. **Domain Discovery**
- Cluster MDPs by gradient similarity **and** bottleneck level
- Domains with different bottleneck levels need separate policies

### 2. **Generalization Testing**
- Train on queue-free, test on bottleneck
- Measure zero-shot transfer performance

### 3. **Curriculum Learning**
- Start with queue-free (easier)
- Gradually increase bottleneck level
- Test if agent can adapt

## Files Modified

- `scheduler/rl_model/catastrophic_conflict_selector.py`: Added `generate_bottleneck_candidate()`
- `scheduler/rl_model/ablation_gnn_traj_main.py`: Added bottleneck parameters to `Args`
- `configs/test_catastrophic_conflict.yaml`: Added bottleneck configuration

## Next Steps

1. **Run experiment** with bottleneck regimes enabled
2. **Compare** conflict scores: bottleneck vs queue-free
3. **Verify** that bottlenecks create stronger conflict
4. **Document** findings for domain discovery theory
