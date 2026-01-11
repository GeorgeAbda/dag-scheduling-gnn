"""
Gradient-Based Domain Clustering for Multi-Objective Reinforcement Learning

This package implements unsupervised domain clustering using policy gradients
and spectral clustering on MORL environments.
"""

from .gradient_domain_clustering import (
    GradientDomainClustering,
    ClusteringConfig,
    SimplePolicy
)

from .mo_gymnasium_loader import (
    MOEnvironmentWrapper,
    SyntheticMOEnvironment,
    create_synthetic_mo_domains,
    load_mo_gymnasium_environments,
    get_default_mo_environments,
    get_mo_mujoco_environments,
    get_domain_info
)

__version__ = '1.0.0'
__author__ = 'Research Team'

__all__ = [
    'GradientDomainClustering',
    'ClusteringConfig',
    'SimplePolicy',
    'MOEnvironmentWrapper',
    'SyntheticMOEnvironment',
    'create_synthetic_mo_domains',
    'load_mo_gymnasium_environments',
    'get_default_mo_environments',
    'get_mo_mujoco_environments',
    'get_domain_info',
]
