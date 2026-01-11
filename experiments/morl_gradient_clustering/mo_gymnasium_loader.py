"""
MO-Gymnasium Environment Loader and Utilities

This module provides utilities for loading and working with MO-Gymnasium environments
for gradient-based domain clustering experiments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    import mo_gymnasium as mo_gym
    MO_GYM_AVAILABLE = True
except ImportError:
    MO_GYM_AVAILABLE = False
    warnings.warn("MO-Gymnasium not installed. Install with: pip install mo-gymnasium")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn("Gymnasium not installed. Install with: pip install gymnasium")


class MOEnvironmentWrapper:
    """Wrapper for MO-Gymnasium environments to provide consistent interface"""
    
    def __init__(self, env_name: str, **kwargs):
        if not MO_GYM_AVAILABLE:
            raise ImportError("MO-Gymnasium is required. Install with: pip install mo-gymnasium")
        
        self.env_name = env_name
        self.env = mo_gym.make(env_name, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()
    
    @property
    def reward_dim(self):
        """Get the dimension of the reward vector"""
        return self.env.reward_space.shape[0] if hasattr(self.env, 'reward_space') else 2


def load_mo_gymnasium_environments(
    env_configs: List[Dict[str, Any]]
) -> Dict[str, MOEnvironmentWrapper]:
    """
    Load multiple MO-Gymnasium environments
    
    Args:
        env_configs: List of environment configurations, each containing:
            - name: Environment name
            - env_id: MO-Gymnasium environment ID
            - kwargs: Additional environment arguments
    
    Returns:
        Dictionary mapping environment names to wrapped environments
    """
    environments = {}
    
    for config in env_configs:
        name = config['name']
        env_id = config['env_id']
        kwargs = config.get('kwargs', {})
        
        try:
            env = MOEnvironmentWrapper(env_id, **kwargs)
            environments[name] = env
            print(f"Loaded environment: {name} ({env_id})")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    return environments


def get_default_mo_environments() -> List[Dict[str, Any]]:
    """
    Get default MO-Gymnasium environment configurations
    
    Returns:
        List of environment configurations suitable for clustering experiments
    """
    configs = [
        # Deep Sea Treasure variants
        {
            'name': 'deep_sea_treasure_v0',
            'env_id': 'deep-sea-treasure-v0',
            'kwargs': {}
        },
        {
            'name': 'deep_sea_treasure_concave_v0',
            'env_id': 'deep-sea-treasure-concave-v0',
            'kwargs': {}
        },
        
        # Minecart variants
        {
            'name': 'minecart_v0',
            'env_id': 'minecart-v0',
            'kwargs': {}
        },
        {
            'name': 'minecart_deterministic_v0',
            'env_id': 'minecart-deterministic-v0',
            'kwargs': {}
        },
        
        # Resource gathering
        {
            'name': 'resource_gathering_v0',
            'env_id': 'resource-gathering-v0',
            'kwargs': {}
        },
        
        # Four room
        {
            'name': 'four_room_v0',
            'env_id': 'four-room-v0',
            'kwargs': {}
        },
    ]
    
    return configs


def get_mo_mujoco_environments() -> List[Dict[str, Any]]:
    """
    Get MO-MuJoCo environment configurations
    
    Returns:
        List of MO-MuJoCo environment configurations
    """
    configs = [
        # HalfCheetah variants
        {
            'name': 'mo_halfcheetah_v4',
            'env_id': 'mo-halfcheetah-v4',
            'kwargs': {}
        },
        {
            'name': 'mo_halfcheetah_v4_high_cost',
            'env_id': 'mo-halfcheetah-v4',
            'kwargs': {'cost_objective': True}
        },
        
        # Hopper variants
        {
            'name': 'mo_hopper_v4',
            'env_id': 'mo-hopper-v4',
            'kwargs': {}
        },
        {
            'name': 'mo_hopper_v4_high_cost',
            'env_id': 'mo-hopper-v4',
            'kwargs': {'cost_objective': True}
        },
        
        # Walker2d variants
        {
            'name': 'mo_walker2d_v4',
            'env_id': 'mo-walker2d-v4',
            'kwargs': {}
        },
        
        # Ant variants
        {
            'name': 'mo_ant_v4',
            'env_id': 'mo-ant-v4',
            'kwargs': {}
        },
    ]
    
    return configs


def create_domain_variants(
    base_env_id: str,
    n_variants: int = 3,
    variant_type: str = 'reward_scaling'
) -> List[Dict[str, Any]]:
    """
    Create domain variants from a base environment
    
    Args:
        base_env_id: Base MO-Gymnasium environment ID
        n_variants: Number of variants to create
        variant_type: Type of variation ('reward_scaling', 'dynamics', 'cost')
    
    Returns:
        List of environment configurations with variants
    """
    configs = []
    
    for i in range(n_variants):
        if variant_type == 'reward_scaling':
            # Scale different reward components
            scale_r1 = 1.0 + i * 0.5
            scale_r2 = 1.0 - i * 0.3
            
            config = {
                'name': f'{base_env_id}_variant_{i}',
                'env_id': base_env_id,
                'kwargs': {
                    'reward_scaling': [scale_r1, scale_r2]
                }
            }
        elif variant_type == 'cost':
            # Vary cost objectives
            config = {
                'name': f'{base_env_id}_cost_{i}',
                'env_id': base_env_id,
                'kwargs': {
                    'cost_objective': i % 2 == 0
                }
            }
        else:
            config = {
                'name': f'{base_env_id}_variant_{i}',
                'env_id': base_env_id,
                'kwargs': {}
            }
        
        configs.append(config)
    
    return configs


class SyntheticMOEnvironment:
    """
    Synthetic multi-objective environment for testing
    
    This creates a simple environment with controllable reward trade-offs
    """
    
    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        reward_weights: np.ndarray = None,
        episode_length: int = 100,
        seed: Optional[int] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        
        if reward_weights is None:
            reward_weights = np.array([0.5, 0.5])
        self.reward_weights = reward_weights / reward_weights.sum()
        
        self.current_step = 0
        self.state = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.random.randn(self.obs_dim)
        self.current_step = 0
        return self.state, {}
    
    def step(self, action):
        # Simple dynamics
        # Pad or truncate action to match obs_dim
        action_effect = np.zeros(self.obs_dim)
        min_dim = min(len(action), self.obs_dim)
        action_effect[:min_dim] = action[:min_dim]
        
        self.state = self.state + 0.1 * action_effect + 0.01 * np.random.randn(self.obs_dim)
        self.current_step += 1
        
        # Multi-objective reward
        r1 = -np.sum(self.state ** 2)  # Objective 1: minimize state magnitude
        r2 = -np.sum(action ** 2)       # Objective 2: minimize action magnitude
        
        reward = np.array([r1, r2])
        
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        return self.state, reward, terminated, truncated, {}
    
    def close(self):
        pass


def create_synthetic_mo_domains(
    n_domains: int = 6,
    obs_dim: int = 4,
    action_dim: int = 2,
    seed: int = 42
) -> Dict[str, SyntheticMOEnvironment]:
    """
    Create synthetic MO domains with different reward structures
    
    Args:
        n_domains: Number of domains to create
        obs_dim: Observation dimension
        action_dim: Action dimension
        seed: Random seed
    
    Returns:
        Dictionary of domain name -> environment
    """
    np.random.seed(seed)
    domains = {}
    
    for i in range(n_domains):
        # Create different reward weight configurations
        # Group domains by similar reward structures
        if i < n_domains // 3:
            # Group 1: Favor objective 1
            weights = np.array([0.8, 0.2]) + 0.1 * np.random.randn(2)
        elif i < 2 * n_domains // 3:
            # Group 2: Balanced
            weights = np.array([0.5, 0.5]) + 0.1 * np.random.randn(2)
        else:
            # Group 3: Favor objective 2
            weights = np.array([0.2, 0.8]) + 0.1 * np.random.randn(2)
        
        weights = np.abs(weights)
        
        env = SyntheticMOEnvironment(
            obs_dim=obs_dim,
            action_dim=action_dim,
            reward_weights=weights,
            seed=seed + i
        )
        
        domains[f'domain_{i}'] = env
    
    return domains


def get_domain_info(domains: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Extract information about domains
    
    Args:
        domains: Dictionary of domain environments
    
    Returns:
        Dictionary with domain information
    """
    info = {}
    
    for name, env in domains.items():
        domain_info = {
            'name': name,
        }
        
        if isinstance(env, SyntheticMOEnvironment):
            domain_info['obs_dim'] = env.obs_dim
            domain_info['action_dim'] = env.action_dim
            domain_info['reward_weights'] = env.reward_weights
        elif isinstance(env, MOEnvironmentWrapper):
            domain_info['obs_dim'] = env.observation_space.shape[0]
            domain_info['action_dim'] = env.action_space.shape[0]
            domain_info['reward_dim'] = env.reward_dim
        
        info[name] = domain_info
    
    return info
