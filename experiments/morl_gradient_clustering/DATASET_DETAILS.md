# MORL Datasets and Environments: Detailed Information

This document provides comprehensive details about the publicly available multi-objective reinforcement learning datasets and environments that can be used with the gradient-based domain clustering method.

## 1. MO-Gymnasium (Farama Foundation)

### Overview
MO-Gymnasium is the standard library for multi-objective RL environments, providing a consistent API similar to Gymnasium but with vectorized rewards.

### Installation
```bash
pip install mo-gymnasium
```

### Available Environments

#### Discrete Action Spaces

##### Deep Sea Treasure
- **Environment IDs**: 
  - `deep-sea-treasure-v0`
  - `deep-sea-treasure-concave-v0`
- **Objectives**: 
  - Treasure value (maximize)
  - Time penalty (minimize)
- **State Space**: Discrete grid (11x11)
- **Action Space**: Discrete (4 actions: up, down, left, right)
- **Reward Dimension**: 2
- **Characteristics**: Classic MORL benchmark with known Pareto front
- **Use Case**: Testing clustering on discrete domains with clear trade-offs

##### Minecart
- **Environment IDs**:
  - `minecart-v0`
  - `minecart-deterministic-v0`
- **Objectives**:
  - Ore collection (maximize)
  - Fuel consumption (minimize)
- **State Space**: Discrete
- **Action Space**: Discrete (5 actions)
- **Reward Dimension**: 2
- **Characteristics**: Stochastic transitions (except deterministic variant)
- **Use Case**: Comparing stochastic vs. deterministic domain clustering

##### Resource Gathering
- **Environment ID**: `resource-gathering-v0`
- **Objectives**:
  - Gold collection (maximize)
  - Gem collection (maximize)
- **State Space**: Discrete grid
- **Action Space**: Discrete (4 directions)
- **Reward Dimension**: 2
- **Characteristics**: Competitive objectives (resources in different locations)
- **Use Case**: Clustering domains with competing objectives

##### Four Room
- **Environment ID**: `four-room-v0`
- **Objectives**:
  - Reach goal 1 (maximize)
  - Reach goal 2 (maximize)
- **State Space**: Discrete grid with walls
- **Action Space**: Discrete (4 directions)
- **Reward Dimension**: 2
- **Characteristics**: Navigation with multiple goals
- **Use Case**: Spatial reasoning in multi-objective settings

#### Continuous Action Spaces

##### MO-MuJoCo Environments

All MuJoCo environments have continuous state and action spaces.

**HalfCheetah** (`mo-halfcheetah-v4`)
- **Objectives**:
  - Forward velocity (maximize)
  - Energy efficiency (minimize control cost)
- **State Dimension**: 17
- **Action Dimension**: 6
- **Episode Length**: 1000 steps
- **Characteristics**: Speed vs. efficiency trade-off
- **Variants**:
  - Standard: Velocity + energy
  - With cost objective: Additional safety constraint

**Hopper** (`mo-hopper-v4`)
- **Objectives**:
  - Forward progress (maximize)
  - Healthy reward (maximize stability)
- **State Dimension**: 11
- **Action Dimension**: 3
- **Episode Length**: 1000 steps
- **Characteristics**: Progress vs. stability trade-off

**Walker2d** (`mo-walker2d-v4`)
- **Objectives**:
  - Forward velocity (maximize)
  - Energy efficiency (minimize)
- **State Dimension**: 17
- **Action Dimension**: 6
- **Episode Length**: 1000 steps
- **Characteristics**: Bipedal locomotion with multiple objectives

**Ant** (`mo-ant-v4`)
- **Objectives**:
  - Forward velocity (maximize)
  - Energy efficiency (minimize)
- **State Dimension**: 27
- **Action Dimension**: 8
- **Episode Length**: 1000 steps
- **Characteristics**: Quadrupedal locomotion, complex dynamics

**Swimmer** (`mo-swimmer-v4`)
- **Objectives**:
  - Forward velocity (maximize)
  - Energy efficiency (minimize)
- **State Dimension**: 8
- **Action Dimension**: 2
- **Episode Length**: 1000 steps
- **Characteristics**: Simpler dynamics, good for quick tests

**Humanoid** (`mo-humanoid-v4`)
- **Objectives**:
  - Forward velocity (maximize)
  - Energy efficiency (minimize)
- **State Dimension**: 376
- **Action Dimension**: 17
- **Episode Length**: 1000 steps
- **Characteristics**: Most complex, high-dimensional

### Usage Example
```python
import mo_gymnasium as mo_gym

env = mo_gym.make('mo-halfcheetah-v4')
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward vector: {reward}")  # e.g., [velocity, -energy]
```

### References
- **Paper**: Alegre et al., "MO-Gym: A Library of Multi-Objective Reinforcement Learning Environments", BNAIC 2022
- **GitHub**: https://github.com/Farama-Foundation/MO-Gymnasium
- **Documentation**: https://mo-gymnasium.farama.org/

---

## 2. D4MORL Dataset (Offline MORL)

### Overview
D4MORL is a benchmark dataset for offline multi-objective reinforcement learning, containing pre-collected trajectories from MuJoCo environments.

### Dataset Structure

#### Environments
- **HalfCheetah**: Speed vs. energy efficiency
- **Hopper**: Progress vs. stability
- **Walker2d**: Velocity vs. energy

#### Data Collection Policies
1. **Expert**: High-performing policy
2. **Medium**: Moderately trained policy
3. **Random**: Random action policy
4. **Medium-Expert**: Mix of medium and expert
5. **Medium-Replay**: Replay buffer from medium training

#### Dataset Compositions
- **Uniform**: Equal mix of different policies
- **Expert-Uniform**: Mix of expert and uniform
- **Medium-Uniform**: Mix of medium and uniform

### Dataset Characteristics

Each dataset contains:
- **Observations**: State vectors
- **Actions**: Continuous actions
- **Rewards**: Multi-objective reward vectors
- **Terminals**: Episode termination flags
- **Timeouts**: Episode timeout flags

Typical dataset sizes:
- Small: ~10K transitions
- Medium: ~50K transitions
- Large: ~100K transitions

### Download Instructions

```bash
# Install gdown
pip install gdown

# Download D4MORL datasets
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data/d4morl
```

### Dataset Files

Each dataset is stored as a pickle file containing:
```python
{
    'observations': np.ndarray,  # Shape: (N, obs_dim)
    'actions': np.ndarray,       # Shape: (N, action_dim)
    'rewards': np.ndarray,       # Shape: (N, reward_dim)
    'terminals': np.ndarray,     # Shape: (N,)
    'timeouts': np.ndarray,      # Shape: (N,)
}
```

### Usage for Gradient Clustering

Note: The current implementation focuses on online gradient computation. To use D4MORL datasets, you would need to:

1. Load pre-collected trajectories
2. Train a policy on the offline data
3. Compute gradients using the trained policy
4. Apply clustering

Example adaptation:
```python
import pickle
import numpy as np

# Load D4MORL dataset
with open('data/d4morl/MO-HalfCheetah-v2_expert_uniform.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Extract multi-objective rewards
rewards = dataset['rewards']  # Shape: (N, 2)
r1 = rewards[:, 0]  # Objective 1
r2 = rewards[:, 1]  # Objective 2

# Use for offline policy training...
```

### References
- **Paper**: Zhu et al., "Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning", NeurIPS 2022 Workshop
- **OpenReview**: https://openreview.net/forum?id=viY2lIr_SGx
- **GitHub (PEDA)**: https://github.com/baitingzbt/PEDA
- **GitHub (MODULI)**: https://github.com/pickxiguapi/MODULI

---

## 3. MORL-Baselines Benchmark

### Overview
MORL-Baselines provides a comprehensive collection of MORL algorithms and standardized evaluation protocols.

### Installation
```bash
pip install morl-baselines
```

### Benchmark Environments

MORL-Baselines uses MO-Gymnasium environments but provides:
- Standardized evaluation protocols
- Hyperparameter configurations
- Baseline results for comparison

### Key Features
- **Multiple algorithms**: GPI-LS, MORL/D, Envelope Q-Learning, PGMORL, etc.
- **Automatic logging**: Weights & Biases integration
- **Reproducible results**: Fixed seeds and configurations
- **Utility functions**: Pareto pruning, experience buffers

### Usage for Clustering

MORL-Baselines can provide:
1. Pre-trained policies for different preferences
2. Evaluation metrics for cluster validation
3. Baseline performance for comparison

Example:
```python
from morl_baselines.multi_policy.gpi_pd import GPILS
import mo_gymnasium as mo_gym

env = mo_gym.make('deep-sea-treasure-v0')
agent = GPILS(env, ...)

# Train agent
agent.train(...)

# Use trained policies for gradient computation
policy = agent.get_policy(preference=[0.5, 0.5])
```

### References
- **Paper**: Felten et al., "A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning", NeurIPS 2023
- **GitHub**: https://github.com/LucasAlegre/morl-baselines
- **Documentation**: Included in repository

---

## 4. Synthetic Domains (Built-in)

### Overview
The implementation includes synthetic multi-objective environments for rapid prototyping and testing.

### Characteristics

**Configurable Parameters**:
- Observation dimension (default: 4)
- Action dimension (default: 2)
- Reward weights (controllable trade-offs)
- Episode length (default: 100)

**Reward Structure**:
- Objective 1: Minimize state magnitude (||s||²)
- Objective 2: Minimize action magnitude (||a||²)

**Dynamics**:
```python
s_{t+1} = s_t + 0.1 * a_t + noise
```

### Creating Synthetic Domains

```python
from mo_gymnasium_loader import create_synthetic_mo_domains

# Create 9 domains with 3 distinct reward structures
domains = create_synthetic_mo_domains(
    n_domains=9,
    obs_dim=4,
    action_dim=2,
    seed=42
)

# Domains are automatically grouped:
# - Domains 0-2: Favor objective 1 (weights ≈ [0.8, 0.2])
# - Domains 3-5: Balanced (weights ≈ [0.5, 0.5])
# - Domains 6-8: Favor objective 2 (weights ≈ [0.2, 0.8])
```

### Advantages
- **Fast execution**: No complex physics simulation
- **Known ground truth**: Pre-defined cluster structure
- **Controllable difficulty**: Adjust dimensions and dynamics
- **No dependencies**: Works without external libraries

### Use Cases
- Testing clustering algorithm
- Validating implementation
- Quick prototyping
- Educational demonstrations

---

## 5. Comparison Table

| Dataset/Environment | Type | Objectives | State Space | Action Space | Availability |
|-------------------|------|-----------|------------|-------------|--------------|
| **MO-Gymnasium** |
| Deep Sea Treasure | Online | 2 | Discrete | Discrete | ✓ Free |
| Minecart | Online | 2 | Discrete | Discrete | ✓ Free |
| MO-HalfCheetah | Online | 2 | Continuous (17) | Continuous (6) | ✓ Free |
| MO-Hopper | Online | 2 | Continuous (11) | Continuous (3) | ✓ Free |
| MO-Walker2d | Online | 2 | Continuous (17) | Continuous (6) | ✓ Free |
| **D4MORL** |
| HalfCheetah-Expert | Offline | 2 | Continuous (17) | Continuous (6) | ✓ Free |
| Hopper-Medium | Offline | 2 | Continuous (11) | Continuous (3) | ✓ Free |
| Walker2d-Random | Offline | 2 | Continuous (17) | Continuous (6) | ✓ Free |
| **Synthetic** |
| Custom Domains | Online | 2 | Configurable | Configurable | ✓ Built-in |

---

## 6. Recommendations for Clustering Experiments

### Quick Testing (< 5 minutes)
- **Synthetic domains**: 6-9 domains, 10-20 gradient samples
- **Purpose**: Verify implementation, test parameters

### Standard Benchmarking (10-30 minutes)
- **MO-Gymnasium discrete**: Deep Sea Treasure, Minecart (4-6 environments)
- **Gradient samples**: 30-50
- **Purpose**: Reproducible results, algorithm comparison

### Comprehensive Evaluation (1-2 hours)
- **MO-MuJoCo**: HalfCheetah, Hopper, Walker2d variants (6-9 environments)
- **Gradient samples**: 50-100
- **Purpose**: Publication-quality results

### Large-Scale Studies (several hours)
- **All MO-Gymnasium + MO-MuJoCo**: 10-15 environments
- **Multiple α values**: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- **Gradient samples**: 100+
- **Purpose**: Comprehensive domain analysis

---

## 7. Data Availability Statement

All datasets and environments mentioned are publicly available:

- **MO-Gymnasium**: MIT License, freely available via pip
- **D4MORL**: Available via Google Drive (see PEDA/MODULI repositories)
- **MORL-Baselines**: MIT License, freely available via pip
- **Synthetic Domains**: Included in this implementation

No proprietary or restricted datasets are required to use this implementation.

---

## 8. Citation Information

If you use these datasets in your research, please cite the appropriate papers:

### MO-Gymnasium
```bibtex
@inproceedings{alegre2022mo,
  title={MO-Gym: A Library of Multi-Objective Reinforcement Learning Environments},
  author={Alegre, Lucas N and Felten, Florian and Now{\'e}, Ann and Bazzan, Ana LC and Roijers, Diederik M},
  booktitle={BNAIC/BeNeLearn},
  year={2022}
}
```

### MORL-Baselines
```bibtex
@article{felten2023toolkit,
  title={A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning},
  author={Felten, Florian and Alegre, Lucas N and Now{\'e}, Ann and Bazzan, Ana LC and El Asri, Layla and Radulescu, Roxana and Roijers, Diederik M},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

### D4MORL
```bibtex
@article{zhu2022pareto,
  title={Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning},
  author={Zhu, Baiting and Dang, Meihua and Grover, Aditya},
  journal={NeurIPS 2022 Workshop on Offline RL},
  year={2022}
}
```
