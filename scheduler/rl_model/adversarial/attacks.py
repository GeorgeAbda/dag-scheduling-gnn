"""
Adversarial attacks for cloud scheduling RL agent.

This module implements various adversarial attacks on both graph structure and node features
to evaluate the robustness of the RL-based cloud scheduling agent.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import sys
import os
from collections import defaultdict

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    # Graph structure attacks
    edge_removal_ratio: float = 0.1  # Fraction of edges to remove
    edge_addition_ratio: float = 0.1  # Fraction of edges to add
    node_removal_ratio: float = 0.05  # Fraction of nodes to remove
    
    # Node feature attacks
    task_length_noise_std: float = 0.1  # Standard deviation for task length noise
    vm_speed_noise_std: float = 0.1  # Standard deviation for VM speed noise
    memory_noise_std: float = 0.05  # Standard deviation for memory noise
    cpu_noise_std: float = 0.05  # Standard deviation for CPU noise
    
    # Attack strength
    max_perturbation: float = 0.2  # Maximum perturbation magnitude
    epsilon: float = 0.1  # Attack budget
    
    # Red-team agent
    red_team_lr: float = 1e-3
    red_team_epochs: int = 100
    red_team_batch_size: int = 32


class BaseAttack(ABC):
    """Base class for adversarial attacks."""
    
    def __init__(self, config: AttackConfig):
        self.config = config
    
    @abstractmethod
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        """Apply the attack to the observation."""
        pass
    
    @abstractmethod
    def get_attack_name(self) -> str:
        """Get the name of the attack."""
        pass


class GraphStructureAttack(BaseAttack):
    """Base class for graph structure attacks."""
    
    def _remove_edges(self, obs: GinAgentObsTensor, removal_ratio: float) -> GinAgentObsTensor:
        """Remove a fraction of edges from the graph while keeping at least one edge per task."""
        if obs.compatibilities.shape[1] == 0:
            return obs
        
        total_edges = obs.compatibilities.shape[1]
        num_edges_to_remove = int(total_edges * removal_ratio)
        if num_edges_to_remove <= 0:
            return obs
        
        # Compute degrees per task
        task_ids = obs.compatibilities[0]
        vm_ids = obs.compatibilities[1]  # noqa: F841
        degrees = torch.bincount(task_ids, minlength=obs.task_state_ready.shape[0])
        
        # Candidate edges to remove: those whose task has degree > 1 (to keep at least one)
        mask_removable = degrees[task_ids] > 1
        removable_indices = torch.nonzero(mask_removable, as_tuple=False).flatten()
        if removable_indices.numel() == 0:
            return obs
        
        # Sample edges to remove from removable set
        perm = torch.randperm(removable_indices.numel())
        pick = removable_indices[perm[:min(num_edges_to_remove, removable_indices.numel())]]
        
        # Apply removal
        keep_mask = torch.ones(total_edges, dtype=torch.bool)
        keep_mask[pick] = False
        remaining_edges = obs.compatibilities[:, keep_mask]
        
        # Ensure still at least one per task (safety net, though enforced above)
        new_degrees = torch.bincount(remaining_edges[0], minlength=obs.task_state_ready.shape[0])
        if torch.any(new_degrees == 0):
            # Re-add one edge for tasks with zero degree by picking from the removed set
            removed_edges = obs.compatibilities[:, ~keep_mask]
            if removed_edges.numel() > 0:
                for t_idx in torch.nonzero(new_degrees == 0, as_tuple=False).flatten().tolist():
                    candidates = torch.nonzero(removed_edges[0] == t_idx, as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        # Take the first candidate back
                        e = candidates[0].item()
                        remaining_edges = torch.cat([remaining_edges, removed_edges[:, e:e+1]], dim=1)
        
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=obs.task_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=obs.task_memory_req_mb,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=obs.vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=remaining_edges,
            task_dependencies=obs.task_dependencies
        )
        return new_obs

    def get_attack_name(self) -> str:
        return f"NodeRemoval_{self.config.node_removal_ratio}"
    
    def _add_edges(self, obs: GinAgentObsTensor, addition_ratio: float) -> GinAgentObsTensor:
        """Add random edges to the graph."""
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]
        
        if num_tasks == 0 or num_vms == 0:
            return obs
        
        num_edges_to_add = int(obs.compatibilities.shape[1] * addition_ratio)
        if num_edges_to_add == 0:
            return obs
        
        # Generate random task-VM pairs
        new_edges = torch.stack([
            torch.randint(0, num_tasks, (num_edges_to_add,)),
            torch.randint(0, num_vms, (num_edges_to_add,))
        ])
        
        # Combine with existing edges
        all_edges = torch.cat([obs.compatibilities, new_edges], dim=1)
        
        # Remove duplicates
        unique_edges = torch.unique(all_edges, dim=1)
        
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=obs.task_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=obs.task_memory_req_mb,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=obs.vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=unique_edges,
            task_dependencies=obs.task_dependencies
        )
        return new_obs


class EdgeRemovalAttack(GraphStructureAttack):
    """Attack that removes edges from the task-VM compatibility graph."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        return self._remove_edges(obs, self.config.edge_removal_ratio)
    
    def get_attack_name(self) -> str:
        return f"EdgeRemoval_{self.config.edge_removal_ratio}"


class EdgeAdditionAttack(GraphStructureAttack):
    """Attack that adds random edges to the task-VM compatibility graph."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        return self._add_edges(obs, self.config.edge_addition_ratio)
    
    def get_attack_name(self) -> str:
        return f"EdgeAddition_{self.config.edge_addition_ratio}"


class NodeFeatureAttack(BaseAttack):
    """Base class for node feature attacks."""
    
    def _add_noise(self, tensor: torch.Tensor, noise_std: float, 
                   min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
        """Add Gaussian noise to a tensor with optional clipping."""
        noise = torch.randn_like(tensor) * noise_std
        perturbed = tensor + noise
        
        if min_val is not None:
            perturbed = torch.clamp(perturbed, min=min_val)
        if max_val is not None:
            perturbed = torch.clamp(perturbed, max=max_val)
        
        return perturbed


class NodeRemovalAttack(GraphStructureAttack):
    """Attack that removes a fraction of task nodes and reindexes the graph consistently."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        num_tasks = obs.task_state_scheduled.shape[0]
        if num_tasks <= 1 or self.config.node_removal_ratio <= 0.0:
            return obs
        
        num_remove = int(num_tasks * self.config.node_removal_ratio)
        if num_remove <= 0 or num_remove >= num_tasks:
            return obs
        
        # Avoid removing all ready tasks: prefer removing from scheduled or non-ready when possible
        all_indices = torch.arange(num_tasks)
        ready_mask = (obs.task_state_ready > 0.5)
        candidates = all_indices[~ready_mask]
        if candidates.numel() < num_remove:
            # If insufficient, allow taking from ready set too but keep at least one ready task
            extra_needed = num_remove - candidates.numel()
            ready_candidates = all_indices[ready_mask]
            if ready_candidates.numel() > extra_needed:
                extra_pick = ready_candidates[torch.randperm(ready_candidates.numel())[:extra_needed]]
                remove_idx = torch.cat([candidates, extra_pick])
            else:
                remove_idx = all_indices[torch.randperm(num_tasks)[:num_remove]]
        else:
            remove_idx = candidates[torch.randperm(candidates.numel())[:num_remove]]
        keep_mask = torch.ones(num_tasks, dtype=torch.bool)
        keep_mask[remove_idx] = False
        
        # Build new index mapping for tasks
        new_index = -torch.ones(num_tasks, dtype=torch.long)
        new_index[keep_mask] = torch.arange(keep_mask.sum())
        
        # Reindex task features
        def sel(x: torch.Tensor) -> torch.Tensor:
            return x[keep_mask]
        new_task_state_scheduled = sel(obs.task_state_scheduled)
        new_task_state_ready = sel(obs.task_state_ready)
        new_task_length = sel(obs.task_length)
        new_task_completion_time = sel(obs.task_completion_time)
        new_task_memory_req_mb = sel(obs.task_memory_req_mb)
        new_task_cpu_req_cores = sel(obs.task_cpu_req_cores)
        
        # Reindex compatibilities: drop edges involving removed tasks, then remap task indices
        comp = obs.compatibilities
        keep_edges = keep_mask[comp[0]]
        comp_kept = comp[:, keep_edges]
        if comp_kept.numel() > 0:
            new_task_ids = new_index[comp_kept[0]]
            reindexed_comp = torch.stack([new_task_ids, comp_kept[1]], dim=0)
        else:
            reindexed_comp = torch.empty((2, 0), dtype=torch.long)
        
        # Ensure at least one edge per remaining task if possible
        if reindexed_comp.numel() > 0:
            deg = torch.bincount(reindexed_comp[0], minlength=int(keep_mask.sum().item()))
            for t in torch.nonzero(deg == 0, as_tuple=False).flatten().tolist():
                # Try to connect to a random VM
                num_vms = obs.vm_completion_time.shape[0]
                if num_vms > 0:
                    vm = torch.randint(0, num_vms, (1,))
                    new_edge = torch.stack([torch.tensor([t], dtype=torch.long), vm.long()])
                    reindexed_comp = torch.cat([reindexed_comp, new_edge], dim=1)
        
        # Reindex dependencies similarly
        deps = obs.task_dependencies
        if deps.numel() > 0:
            keep_dep = keep_mask[deps[0]] & keep_mask[deps[1]]
            deps_kept = deps[:, keep_dep]
            if deps_kept.numel() > 0:
                new_src = new_index[deps_kept[0]]
                new_dst = new_index[deps_kept[1]]
                reindexed_deps = torch.stack([new_src, new_dst], dim=0)
            else:
                reindexed_deps = torch.empty((2, 0), dtype=torch.long)
        else:
            reindexed_deps = torch.empty((2, 0), dtype=torch.long)
        
        # Assemble new observation
        new_obs = GinAgentObsTensor(
            task_state_scheduled=new_task_state_scheduled,
            task_state_ready=new_task_state_ready,
            task_length=new_task_length,
            task_completion_time=new_task_completion_time,
            task_memory_req_mb=new_task_memory_req_mb,
            task_cpu_req_cores=new_task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=obs.vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=reindexed_comp,
            task_dependencies=reindexed_deps,
        )
        return new_obs


class TaskLengthAttack(NodeFeatureAttack):
    """Attack that perturbs task length features."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        # Add noise to task length
        perturbed_length = self._add_noise(
            obs.task_length, 
            self.config.task_length_noise_std,
            min_val=0.1  # Ensure positive task length
        )
        
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=perturbed_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=obs.task_memory_req_mb,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=obs.vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=obs.compatibilities,
            task_dependencies=obs.task_dependencies
        )
        return new_obs
    
    def get_attack_name(self) -> str:
        return f"TaskLength_{self.config.task_length_noise_std}"


class VMSpeedAttack(NodeFeatureAttack):
    """Attack that perturbs VM speed features."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        # Add noise to VM speed
        perturbed_speed = self._add_noise(
            obs.vm_speed,
            self.config.vm_speed_noise_std,
            min_val=0.1  # Ensure positive speed
        )
        
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=obs.task_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=obs.task_memory_req_mb,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=perturbed_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=obs.compatibilities,
            task_dependencies=obs.task_dependencies
        )
        return new_obs
    
    def get_attack_name(self) -> str:
        return f"VMSpeed_{self.config.vm_speed_noise_std}"


class MemoryAttack(NodeFeatureAttack):
    """Attack that perturbs memory-related features."""
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        # Add noise to memory features
        perturbed_task_memory = self._add_noise(
            obs.task_memory_req_mb,
            self.config.memory_noise_std,
            min_val=1.0  # Ensure positive memory
        )
        
        perturbed_vm_memory = self._add_noise(
            obs.vm_memory_mb,
            self.config.memory_noise_std,
            min_val=1.0  # Ensure positive memory
        )
        
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=obs.task_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=perturbed_task_memory,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=obs.vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=perturbed_vm_memory,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=obs.compatibilities,
            task_dependencies=obs.task_dependencies
        )
        return new_obs
    
    def get_attack_name(self) -> str:
        return f"Memory_{self.config.memory_noise_std}"


class RedTeamRLAgent(nn.Module):
    """
    Red-team RL agent that learns to generate adversarial perturbations.
    
    Based on the AttackGNN approach, this agent learns to generate perturbations
    that maximize the target agent's loss while staying within the attack budget.
    """
    
    def __init__(self, config: AttackConfig, target_agent: nn.Module, device: torch.device):
        super().__init__()
        self.config = config
        self.target_agent = target_agent
        self.device = device
        
        # Perturbation generator network
        self.perturbation_net = nn.Sequential(
            nn.Linear(16, 64),  # Input: concatenated task and VM features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # Output: perturbation for each feature
            nn.Tanh()  # Bounded perturbations
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.red_team_lr)
    
    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        """Generate perturbations for the observation."""
        # Extract features
        task_features = torch.stack([
            obs.task_state_scheduled,
            obs.task_state_ready,
            obs.task_length,
            obs.task_completion_time,
            obs.task_memory_req_mb / 1000.0,
            obs.task_cpu_req_cores / 10.0,  # Normalize
        ], dim=-1)
        
        vm_features = torch.stack([
            obs.vm_completion_time,
            1 / (obs.vm_speed + 1e-8),
            obs.vm_energy_rate,
            obs.vm_memory_mb / 1000.0,
            obs.vm_available_memory_mb / 1000.0,
            obs.vm_used_memory_fraction,
            obs.vm_active_tasks_count / 10.0,  # Normalize
            obs.vm_cpu_cores / 10.0,  # Normalize
            obs.vm_available_cpu_cores / 10.0,  # Normalize
            obs.vm_used_cpu_fraction_cores,
        ], dim=-1)
        
        # Pad to same size
        max_len = max(task_features.shape[0], vm_features.shape[0])
        if task_features.shape[0] < max_len:
            task_features = torch.cat([
                task_features,
                torch.zeros(max_len - task_features.shape[0], task_features.shape[1], device=self.device)
            ])
        if vm_features.shape[0] < max_len:
            vm_features = torch.cat([
                vm_features,
                torch.zeros(max_len - vm_features.shape[0], vm_features.shape[1], device=self.device)
            ])
        
        # Concatenate features
        combined_features = torch.cat([task_features, vm_features], dim=-1)
        
        # Generate perturbations
        perturbations = self.perturbation_net(combined_features)
        
        return perturbations
    
    def compute_attack_loss(self, obs: GinAgentObsTensor, perturbations: torch.Tensor) -> torch.Tensor:
        """Compute the attack loss to maximize target agent's loss."""
        # Apply perturbations to create adversarial observation
        adv_obs = self._apply_perturbations(obs, perturbations)
        
        # Convert to tensor format for the agent (keep graph for gradients)
        obs_tensor = self._obs_to_tensor(adv_obs)
        target_scores = self.target_agent(obs_tensor)
        target_probs = torch.softmax(target_scores, dim=-1)
        
        # Attack loss: maximize target uncertainty (maximize entropy)
        entropy = -torch.sum(target_probs * torch.log(target_probs + 1e-8), dim=-1)
        attack_loss = -entropy.mean()
        
        # Small L2 regularization on perturbations to respect budget and stabilize training
        reg = (perturbations ** 2).mean()
        attack_loss = attack_loss + 1e-3 * reg
        
        return attack_loss
    
    def _obs_to_tensor(self, obs: GinAgentObsTensor) -> torch.Tensor:
        """Convert GinAgentObsTensor to tensor format for the agent."""
        # Extract features similar to the mapper
        task_features = torch.stack([
            obs.task_state_scheduled,
            obs.task_state_ready,
            obs.task_length,
            obs.task_completion_time,
            obs.task_memory_req_mb / 1000.0,
            obs.task_cpu_req_cores / 10.0,
        ], dim=-1)
        
        vm_features = torch.stack([
            obs.vm_completion_time,
            1 / (obs.vm_speed + 1e-8),
            obs.vm_energy_rate,
            obs.vm_memory_mb / 1000.0,
            obs.vm_available_memory_mb / 1000.0,
            obs.vm_used_memory_fraction,
            obs.vm_active_tasks_count / 10.0,
            obs.vm_cpu_cores / 10.0,
            obs.vm_available_cpu_cores / 10.0,
            obs.vm_used_cpu_fraction_cores,
        ], dim=-1)
        
        # Pad to same size
        max_len = max(task_features.shape[0], vm_features.shape[0])
        if task_features.shape[0] < max_len:
            task_features = torch.cat([
                task_features,
                torch.zeros(max_len - task_features.shape[0], task_features.shape[1], device=self.device)
            ])
        if vm_features.shape[0] < max_len:
            vm_features = torch.cat([
                vm_features,
                torch.zeros(max_len - vm_features.shape[0], vm_features.shape[1], device=self.device)
            ])
        
        # Concatenate features
        combined_features = torch.cat([task_features, vm_features], dim=-1)
        return combined_features.reshape(1, -1)
    
    def _apply_perturbations(self, obs: GinAgentObsTensor, perturbations: torch.Tensor) -> GinAgentObsTensor:
        """Apply perturbations to the observation."""
        # Scale perturbations by epsilon
        scaled_perturbations = perturbations * self.config.epsilon
        
        # Apply to task features
        task_perturbations = scaled_perturbations[:obs.task_length.shape[0], :6]
        vm_perturbations = scaled_perturbations[:obs.vm_speed.shape[0], 6:]
        
        # Apply perturbations
        new_task_length = obs.task_length + task_perturbations[:, 2] * obs.task_length
        new_vm_speed = obs.vm_speed + vm_perturbations[:, 1] * obs.vm_speed
        
        # Create new observation
        new_obs = GinAgentObsTensor(
            task_state_scheduled=obs.task_state_scheduled,
            task_state_ready=obs.task_state_ready,
            task_length=new_task_length,
            task_completion_time=obs.task_completion_time,
            task_memory_req_mb=obs.task_memory_req_mb,
            task_cpu_req_cores=obs.task_cpu_req_cores,
            vm_completion_time=obs.vm_completion_time,
            vm_speed=new_vm_speed,
            vm_energy_rate=obs.vm_energy_rate,
            vm_memory_mb=obs.vm_memory_mb,
            vm_available_memory_mb=obs.vm_available_memory_mb,
            vm_used_memory_fraction=obs.vm_used_memory_fraction,
            vm_active_tasks_count=obs.vm_active_tasks_count,
            vm_next_release_time=obs.vm_next_release_time,
            vm_cpu_cores=obs.vm_cpu_cores,
            vm_available_cpu_cores=obs.vm_available_cpu_cores,
            vm_used_cpu_fraction_cores=obs.vm_used_cpu_fraction_cores,
            vm_next_core_release_time=obs.vm_next_core_release_time,
            compatibilities=obs.compatibilities,
            task_dependencies=obs.task_dependencies
        )
        return new_obs
    
    def train_step(self, obs: GinAgentObsTensor) -> float:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        perturbations = self.forward(obs)
        loss = self.compute_attack_loss(obs, perturbations)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class RedTeamAttack(BaseAttack):
    """Attack using the red-team RL agent."""
    
    def __init__(self, config: AttackConfig, target_agent: nn.Module, device: torch.device):
        super().__init__(config)
        self.red_team_agent = RedTeamRLAgent(config, target_agent, device)
        self.device = device
    
    def attack(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> GinAgentObsTensor:
        # Train the red-team agent briefly on the current observation to craft a strong perturbation
        self.red_team_agent.train()
        steps = max(1, min(10, self.config.red_team_epochs))
        for _ in range(steps):
            try:
                self.red_team_agent.train_step(obs)
            except Exception:
                # If any gradient issue occurs, fall back to a single forward without training
                break
        
        # Generate final perturbations and apply
        with torch.no_grad():
            perturbations = self.red_team_agent.forward(obs)
            adv_obs = self.red_team_agent._apply_perturbations(obs, perturbations)
        return adv_obs
    
    def get_attack_name(self) -> str:
        return f"RedTeam_{self.config.red_team_epochs}"


class AttackEnsemble:
    """Ensemble of different attacks for comprehensive evaluation."""
    
    def __init__(self, config: AttackConfig, target_agent: nn.Module, device: torch.device):
        self.attacks = [
            EdgeRemovalAttack(config),
            EdgeAdditionAttack(config),
            NodeRemovalAttack(config),
            TaskLengthAttack(config),
            VMSpeedAttack(config),
            MemoryAttack(config),
            RedTeamAttack(config, target_agent, device),
        ]
    
    def evaluate_all_attacks(self, obs: GinAgentObsTensor, target_agent: nn.Module) -> Dict[str, GinAgentObsTensor]:
        """Evaluate all attacks on the given observation."""
        results = {}
        for attack in self.attacks:
            try:
                adv_obs = attack.attack(obs, target_agent)
                results[attack.get_attack_name()] = adv_obs
            except Exception as e:
                print(f"Attack {attack.get_attack_name()} failed: {e}")
                results[attack.get_attack_name()] = obs  # Return original if attack fails
        
        return results

