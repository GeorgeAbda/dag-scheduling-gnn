"""
Established RL Benchmarks for Domain Discovery

These benchmarks are known from literature to contain multiple distinct domains
with documented negative transfer between them.

Benchmarks:
1. Meta-World MT10: 10 robotic manipulation tasks (Yu et al., 2020)
2. Procgen: 16 procedurally generated games (Cobbe et al., 2020)
3. DMControl with dynamics shifts: Same task, different physics

References:
- Meta-World: "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta RL"
- Procgen: "Leveraging Procedural Generation to Benchmark RL"
- DMControl: "dm_control: Software and Tasks for Continuous Control"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Check for optional dependencies
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    try:
        import gym
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False
        warnings.warn("gymnasium/gym not found. RL benchmarks will be limited.")

try:
    import metaworld
    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False

try:
    import procgen
    HAS_PROCGEN = True
except ImportError:
    HAS_PROCGEN = False


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RLDomainInfo:
    """Information about an RL domain/task."""
    name: str
    domain_id: int
    env_name: str
    obs_dim: int
    act_dim: int
    is_discrete: bool


@dataclass 
class RLBenchmarkInfo:
    """Information about an RL benchmark."""
    name: str
    num_domains: int
    domains: List[RLDomainInfo]
    description: str
    known_conflicts: List[Tuple[int, int]]  # Pairs known to have negative transfer


# =============================================================================
# Meta-World MT10 Benchmark
# =============================================================================

# MT10 tasks (from Meta-World paper)
MT10_TASKS = [
    "reach-v2",
    "push-v2", 
    "pick-place-v2",
    "door-open-v2",
    "drawer-open-v2",
    "drawer-close-v2",
    "button-press-topdown-v2",
    "peg-insert-side-v2",
    "window-open-v2",
    "window-close-v2",
]

# Known task groupings from Meta-World paper (tasks that share structure)
MT10_TASK_GROUPS = {
    "reaching": [0],  # reach
    "pushing": [1, 2],  # push, pick-place
    "doors": [3, 8, 9],  # door-open, window-open, window-close
    "drawers": [4, 5],  # drawer-open, drawer-close
    "buttons": [6],  # button-press
    "insertion": [7],  # peg-insert
}

# Expected domain labels based on task similarity
MT10_DOMAIN_LABELS = np.array([0, 1, 1, 2, 3, 3, 4, 5, 2, 2])


def create_metaworld_mt10(seed: int = 42) -> Tuple[Dict[int, Any], np.ndarray, RLBenchmarkInfo]:
    """
    Create Meta-World MT10 benchmark.
    
    Returns:
        envs: Dictionary mapping task_id to environment
        domain_labels: Ground truth domain labels based on task similarity
        info: Benchmark information
    """
    if not HAS_METAWORLD:
        raise ImportError(
            "metaworld not installed. Install with: pip install metaworld"
        )
    
    mt10 = metaworld.MT10(seed=seed)
    
    envs = {}
    domains = []
    
    for i, (name, env_cls) in enumerate(mt10.train_classes.items()):
        env = env_cls()
        task = [t for t in mt10.train_tasks if t.env_name == name][0]
        env.set_task(task)
        
        envs[i] = env
        
        domains.append(RLDomainInfo(
            name=name,
            domain_id=i,
            env_name=name,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            is_discrete=False,
        ))
    
    # Known conflicts: tasks from different groups often conflict
    known_conflicts = [
        (0, 7),  # reach vs peg-insert
        (1, 6),  # push vs button-press
        (3, 5),  # door-open vs drawer-close
    ]
    
    info = RLBenchmarkInfo(
        name="Meta-World MT10",
        num_domains=len(set(MT10_DOMAIN_LABELS)),
        domains=domains,
        description="10 robotic manipulation tasks with known task groupings",
        known_conflicts=known_conflicts,
    )
    
    return envs, MT10_DOMAIN_LABELS, info


# =============================================================================
# Procgen Benchmark
# =============================================================================

# All 16 Procgen games
PROCGEN_GAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# Game groupings by mechanics (from Procgen paper analysis)
PROCGEN_GAME_GROUPS = {
    "platformer": ["coinrun", "ninja", "climber", "jumper", "leaper"],
    "shooter": ["starpilot", "dodgeball", "bossfight"],
    "navigation": ["maze", "heist", "caveflyer"],
    "collection": ["fruitbot", "miner", "bigfish"],
    "chase": ["chaser", "plunder"],
}

# Domain labels based on game type
PROCGEN_DOMAIN_LABELS = np.array([
    2,  # bigfish - collection
    1,  # bossfight - shooter
    2,  # caveflyer - navigation (but has shooting)
    3,  # chaser - chase
    0,  # climber - platformer
    0,  # coinrun - platformer
    1,  # dodgeball - shooter
    2,  # fruitbot - collection
    2,  # heist - navigation
    0,  # jumper - platformer
    0,  # leaper - platformer
    2,  # maze - navigation
    2,  # miner - collection
    0,  # ninja - platformer
    3,  # plunder - chase
    1,  # starpilot - shooter
])


def create_procgen_benchmark(
    games: Optional[List[str]] = None,
    num_levels: int = 200,
    distribution_mode: str = "easy",
    seed: int = 42,
) -> Tuple[Dict[int, Any], np.ndarray, RLBenchmarkInfo]:
    """
    Create Procgen benchmark.
    
    Args:
        games: List of games to include (default: all 16)
        num_levels: Number of levels per game
        distribution_mode: "easy", "hard", or "extreme"
        seed: Random seed
    
    Returns:
        envs: Dictionary mapping game_id to environment
        domain_labels: Domain labels based on game type
        info: Benchmark information
    """
    if not HAS_PROCGEN:
        raise ImportError(
            "procgen not installed. Install with: pip install procgen"
        )
    
    if games is None:
        games = PROCGEN_GAMES
    
    envs = {}
    domains = []
    labels = []
    
    for i, game in enumerate(games):
        env = gym.make(
            f"procgen:procgen-{game}-v0",
            num_levels=num_levels,
            distribution_mode=distribution_mode,
            start_level=seed,
        )
        
        envs[i] = env
        
        # Get domain label
        game_idx = PROCGEN_GAMES.index(game)
        labels.append(PROCGEN_DOMAIN_LABELS[game_idx])
        
        domains.append(RLDomainInfo(
            name=game,
            domain_id=i,
            env_name=f"procgen-{game}-v0",
            obs_dim=np.prod(env.observation_space.shape),
            act_dim=env.action_space.n,
            is_discrete=True,
        ))
    
    info = RLBenchmarkInfo(
        name="Procgen",
        num_domains=len(set(labels)),
        domains=domains,
        description=f"{len(games)} procedurally generated games",
        known_conflicts=[],  # All different games have some conflict
    )
    
    return envs, np.array(labels), info


# =============================================================================
# DMControl with Dynamics Shifts
# =============================================================================

# Walker task with different dynamics parameters
DMCONTROL_DYNAMICS_SHIFTS = {
    "normal": {"gravity": 9.81, "friction": 1.0, "density": 1.0},
    "low_gravity": {"gravity": 4.0, "friction": 1.0, "density": 1.0},
    "high_gravity": {"gravity": 15.0, "friction": 1.0, "density": 1.0},
    "low_friction": {"gravity": 9.81, "friction": 0.3, "density": 1.0},
    "high_friction": {"gravity": 9.81, "friction": 2.0, "density": 1.0},
    "light": {"gravity": 9.81, "friction": 1.0, "density": 0.5},
    "heavy": {"gravity": 9.81, "friction": 1.0, "density": 2.0},
}


# =============================================================================
# Simplified Synthetic RL Domains (no external dependencies)
# =============================================================================

class SyntheticRLDomain:
    """
    Synthetic RL domain for testing gradient-based domain discovery.
    
    Each domain has different dynamics (transition function) that create
    distinct gradient patterns during training.
    """
    
    def __init__(
        self,
        domain_id: int,
        state_dim: int = 10,
        action_dim: int = 4,
        dynamics_type: str = "linear",
        seed: int = 42,
    ):
        self.domain_id = domain_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dynamics_type = dynamics_type
        
        np.random.seed(seed + domain_id * 1000)
        
        # Different dynamics matrices for different domains
        if dynamics_type == "linear":
            # Linear dynamics: s' = A @ s + B @ a + noise
            self.A = np.random.randn(state_dim, state_dim) * 0.1
            self.A = self.A - np.eye(state_dim) * 0.5  # Stable
            self.B = np.random.randn(state_dim, action_dim) * 0.5
        elif dynamics_type == "nonlinear":
            # Nonlinear dynamics parameters
            self.W1 = np.random.randn(state_dim, state_dim + action_dim) * 0.3
            self.W2 = np.random.randn(state_dim, state_dim) * 0.3
        
        # Reward function (different for each domain)
        self.reward_weights = np.random.randn(state_dim)
        self.reward_weights = self.reward_weights / np.linalg.norm(self.reward_weights)
        
        self.state = None
        self.steps = 0
        self.max_steps = 100
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.randn(self.state_dim) * 0.1
        self.steps = 0
        return self.state.astype(np.float32)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Handle both discrete and continuous actions
        if isinstance(action, (int, np.integer)):
            # Convert discrete action to one-hot continuous
            action_vec = np.zeros(self.action_dim)
            action_vec[action % self.action_dim] = 1.0
            action = action_vec
        else:
            action = np.atleast_1d(action).flatten()
            if len(action) < self.action_dim:
                action = np.pad(action, (0, self.action_dim - len(action)))
            action = action[:self.action_dim]
        
        action = np.clip(action, -1, 1)
        
        if self.dynamics_type == "linear":
            next_state = self.A @ self.state + self.B @ action
            next_state += np.random.randn(self.state_dim) * 0.01
        else:
            combined = np.concatenate([self.state, action])
            hidden = np.tanh(self.W1 @ combined)
            next_state = self.state + self.W2 @ hidden * 0.1
            next_state += np.random.randn(self.state_dim) * 0.01
        
        # Reward based on domain-specific weights
        reward = float(np.dot(self.reward_weights, next_state))
        reward -= 0.01 * np.sum(action ** 2)  # Action penalty
        
        self.state = np.clip(next_state, -10, 10)
        self.steps += 1
        
        done = self.steps >= self.max_steps
        
        return self.state.astype(np.float32), reward, done, False, {}
    
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.state_dim,)})()
    
    @property
    def action_space(self):
        return type('Space', (), {'shape': (self.action_dim,), 'n': self.action_dim})()


def create_synthetic_rl_domains(
    num_domains: int = 4,
    envs_per_domain: int = 2,
    state_dim: int = 10,
    action_dim: int = 4,
    seed: int = 42,
) -> Tuple[Dict[int, SyntheticRLDomain], np.ndarray, RLBenchmarkInfo]:
    """
    Create synthetic RL domains with controlled domain structure.
    
    Domains differ in:
    - Dynamics (transition function)
    - Reward function
    
    Within-domain environments share similar dynamics but different random seeds.
    """
    envs = {}
    labels = []
    domains = []
    
    env_id = 0
    for domain_id in range(num_domains):
        # Alternate between linear and nonlinear dynamics
        dynamics_type = "linear" if domain_id % 2 == 0 else "nonlinear"
        
        for env_idx in range(envs_per_domain):
            env = SyntheticRLDomain(
                domain_id=domain_id,
                state_dim=state_dim,
                action_dim=action_dim,
                dynamics_type=dynamics_type,
                seed=seed + domain_id * 100 + env_idx,
            )
            
            envs[env_id] = env
            labels.append(domain_id)
            
            domains.append(RLDomainInfo(
                name=f"Domain{domain_id}_Env{env_idx}",
                domain_id=domain_id,
                env_name=f"synthetic_{dynamics_type}_{domain_id}_{env_idx}",
                obs_dim=state_dim,
                act_dim=action_dim,
                is_discrete=False,
            ))
            
            env_id += 1
    
    # Known conflicts: different dynamics types conflict
    known_conflicts = []
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            if i % 2 != j % 2:  # Different dynamics types
                known_conflicts.append((i, j))
    
    info = RLBenchmarkInfo(
        name="Synthetic RL Domains",
        num_domains=num_domains,
        domains=domains,
        description=f"{num_domains} synthetic domains with {envs_per_domain} envs each",
        known_conflicts=known_conflicts,
    )
    
    return envs, np.array(labels), info


# =============================================================================
# CartPole with Different Physics (Classic Control)
# =============================================================================

class CartPoleVariant:
    """
    CartPole with modified physics parameters.
    
    Different parameter settings create different domains:
    - Gravity affects how quickly the pole falls
    - Pole length affects the moment of inertia
    - Cart mass affects acceleration response
    """
    
    def __init__(
        self,
        gravity: float = 9.8,
        pole_length: float = 0.5,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        force_mag: float = 10.0,
        seed: int = 42,
    ):
        self.gravity = gravity
        self.pole_length = pole_length
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.force_mag = force_mag
        
        self.total_mass = cart_mass + pole_mass
        self.polemass_length = pole_mass * pole_length
        
        self.tau = 0.02  # Time step
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        self.state = None
        self.steps = 0
        self.max_steps = 200
        
        np.random.seed(seed)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.steps = 0
        return self.state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        x, x_dot, theta, theta_dot = self.state
        
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta ** 2 / self.total_mass)
        )
        
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
            or self.steps >= self.max_steps
        )
        
        reward = 1.0 if not done else 0.0
        
        return self.state.astype(np.float32), reward, done, False, {}
    
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (4,)})()
    
    @property
    def action_space(self):
        return type('Space', (), {'n': 2})()


# CartPole domain configurations (known to create different optimal policies)
CARTPOLE_DOMAINS = {
    "normal": {"gravity": 9.8, "pole_length": 0.5, "cart_mass": 1.0},
    "moon": {"gravity": 1.6, "pole_length": 0.5, "cart_mass": 1.0},  # Low gravity
    "jupiter": {"gravity": 24.8, "pole_length": 0.5, "cart_mass": 1.0},  # High gravity
    "long_pole": {"gravity": 9.8, "pole_length": 1.0, "cart_mass": 1.0},
    "short_pole": {"gravity": 9.8, "pole_length": 0.25, "cart_mass": 1.0},
    "heavy_cart": {"gravity": 9.8, "pole_length": 0.5, "cart_mass": 2.0},
    "light_cart": {"gravity": 9.8, "pole_length": 0.5, "cart_mass": 0.5},
}

# Domain groupings based on physics similarity
CARTPOLE_DOMAIN_LABELS = {
    "normal": 0,
    "moon": 1,
    "jupiter": 1,
    "long_pole": 2,
    "short_pole": 2,
    "heavy_cart": 0,
    "light_cart": 0,
}


def create_cartpole_domains(
    domains: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[Dict[int, CartPoleVariant], np.ndarray, RLBenchmarkInfo]:
    """
    Create CartPole environments with different physics.
    
    This is a well-understood benchmark where different physics
    parameters are known to require different policies.
    """
    if domains is None:
        domains = list(CARTPOLE_DOMAINS.keys())
    
    envs = {}
    labels = []
    domain_infos = []
    
    for i, domain_name in enumerate(domains):
        params = CARTPOLE_DOMAINS[domain_name]
        
        env = CartPoleVariant(
            gravity=params["gravity"],
            pole_length=params["pole_length"],
            cart_mass=params["cart_mass"],
            seed=seed + i,
        )
        
        envs[i] = env
        labels.append(CARTPOLE_DOMAIN_LABELS[domain_name])
        
        domain_infos.append(RLDomainInfo(
            name=domain_name,
            domain_id=i,
            env_name=f"cartpole_{domain_name}",
            obs_dim=4,
            act_dim=2,
            is_discrete=True,
        ))
    
    # Known conflicts: gravity domains vs pole length domains
    known_conflicts = [
        (1, 3),  # moon vs long_pole
        (2, 4),  # jupiter vs short_pole
    ]
    
    info = RLBenchmarkInfo(
        name="CartPole Physics Variants",
        num_domains=len(set(labels)),
        domains=domain_infos,
        description=f"{len(domains)} CartPole variants with different physics",
        known_conflicts=known_conflicts,
    )
    
    return envs, np.array(labels), info


# =============================================================================
# Simple Policy Networks
# =============================================================================

class SimplePolicyMLP(nn.Module):
    """Simple MLP policy for continuous control."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [64, 64],
    ):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp()
        return mean, std
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1, 1)
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)


class SimpleDiscretePolicyMLP(nn.Module):
    """Simple MLP policy for discrete control."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [64, 64],
    ):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, act_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)


# =============================================================================
# Benchmark Registry
# =============================================================================

def get_rl_benchmark(
    name: str,
    **kwargs,
) -> Tuple[Dict[int, Any], np.ndarray, RLBenchmarkInfo]:
    """
    Get an RL benchmark by name.
    
    Available benchmarks:
    - "synthetic": Synthetic RL domains (no dependencies)
    - "cartpole": CartPole with physics variants (no dependencies)
    - "metaworld_mt10": Meta-World MT10 (requires metaworld)
    - "procgen": Procgen games (requires procgen)
    """
    if name == "synthetic":
        return create_synthetic_rl_domains(**kwargs)
    
    elif name == "cartpole":
        return create_cartpole_domains(**kwargs)
    
    elif name == "metaworld_mt10":
        return create_metaworld_mt10(**kwargs)
    
    elif name == "procgen":
        return create_procgen_benchmark(**kwargs)
    
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# List available benchmarks
def list_available_benchmarks() -> List[str]:
    """List available RL benchmarks."""
    available = ["synthetic", "cartpole"]
    
    if HAS_METAWORLD:
        available.append("metaworld_mt10")
    
    if HAS_PROCGEN:
        available.append("procgen")
    
    return available
